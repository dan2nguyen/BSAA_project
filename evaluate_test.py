"""
Evaluates a trained HR model on the test database with predict.py, i.e.
the full deployment pipeline: raw EDF -> filtering (as the filtered
database) -> segment metrics -> feasibility -> model -> HR or NONE.

Test segments = split "test" of the parquet database with a valid
reference HR and not flat (hr_valid & ~is_flat, the same population
load_parquet_files gives training). The reference is the Pan-Tompkins ECG
HR (hr_bpm). Errors are computed like training.py (MAE, RMSE, acc +-5 bpm,
balanced MAE over 10 bpm bins).

Reported:
  - prediction: feasible segments only (HR given), coverage = share of the
    test segments that get an HR,
  - no filter: the model on every test segment,
  - rejected: the model on the segments that got NONE,
  - full borders: the training selection (ECG + PPG borders, from the
    database) for comparison with the CV numbers,
  - CV validation (OOF) of the same fold, if the OOF file exists,
  - per dataset, per subject, per HR bin, and a pipeline check (recomputed
    metrics vs the database, feasibility agreement).

Outputs in --output-dir: report.md, summary.json, segments.parquet,
figures/*.png.

Usage:
    python evaluate_test.py
    python evaluate_test.py --model-path data/training_out_cnn/dilated_cnn/fold_2/best_model.pt
    python evaluate_test.py --feasibility full
    python evaluate_test.py --limit-files 2 --output-dir data/test_evaluation/smoke
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from predict import (
    DEFAULT_MODEL,
    FEASIBILITY_MODES,
    HRPredictor,
    bounds_for,
    prepare_edf,
)
from prototyping_dan.feasible_bpm import (
    FEASIBILITY_BOUNDS,
    PPG_BOUNDS,
    within_bounds,
)
from prototyping_dan.load_database import extract_patient_id
from prototyping_dan.report_borders import (
    AXIS,
    BLUE,
    GRID,
    INK2,
    MUTED,
    ORANGE,
    md_table,
)
from prototyping_dan.training import MODEL_REGISTRY, per_bin_errors

KEY = ["base_folder", "edf_filename", "start"]
MANIFEST = ROOT / "databasing" / "references" / "split_manifest.csv"


# -----------------------------
# 1. Test segments
# -----------------------------


def load_test_segments(parquet_dir: Path, all_segments: bool) -> pd.DataFrame:
    """
    Test-split rows of the parquet database with the reference HR, the
    database metrics and the database feasibility (PPG / full borders).
    """
    metric_cols = list(FEASIBILITY_BOUNDS)
    cols = ["base_folder", "edf_filename", "start", "end", "split",
            "hr_bpm", "hr_valid", "is_flat", "ecg_is_flat", "ppg_is_flat",
            *metric_cols]
    parts = []
    for f in sorted(parquet_dir.glob("*/*.parquet")):
        if pq.read_table(f, columns=["split"])["split"][0].as_py() != "test":
            continue
        df = pd.read_parquet(f, columns=cols)
        df["evaluable"] = df["hr_valid"] & ~df["is_flat"]
        parts.append(df if all_segments else df[df["evaluable"]])
    if not parts:
        raise FileNotFoundError(f"No test-split parquet files in {parquet_dir}")
    df = pd.concat(parts, ignore_index=True)

    subject_of = {}
    if MANIFEST.exists():
        m = pd.read_csv(MANIFEST)
        subject_of = dict(zip(m["file"], m["subject"].astype(str)))
    df["subject"] = [
        f"{b}/{subject_of.get(f'{b}/{e}', extract_patient_id(e))}"
        for b, e in zip(df["base_folder"], df["edf_filename"])
    ]
    df["db_feasible_ppg"] = within_bounds(df, PPG_BOUNDS) & ~df["ppg_is_flat"]
    df["db_feasible_full"] = (within_bounds(df, FEASIBILITY_BOUNDS)
                              & ~df["is_flat"])
    return df.rename(columns={"hr_bpm": "hr_ref"})


def pick_files(test: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """One row per EDF file (n segments, length); first `limit` per dataset."""
    files = (test.groupby(["base_folder", "edf_filename"])
             .agg(n=("start", "size"), length=("end", "max"))
             .reset_index())
    if limit:
        files = files.groupby("base_folder").head(limit)
    return files


# -----------------------------
# 2. Prediction (predict.py)
# -----------------------------


def _prepare(task):
    """Worker: preprocessing, metrics and model input of one file."""
    key, path, starts, mode, input_filtered = task
    try:
        seg, x = prepare_edf(path, mode=mode, input_filtered=input_filtered,
                             starts=starts, model_input="all")
        return key, seg, x, None
    except Exception as e:  # noqa: BLE001
        return key, None, None, f"{type(e).__name__}: {e}"


def run_predictions(test, files, predictor, args) -> tuple[pd.DataFrame, dict]:
    """
    Runs predict.py on every test file. The CPU part runs in worker
    processes; the model runs in this process on each finished file.
    Very long recordings run one at a time (RAM).
    """
    starts = test.groupby(["base_folder", "edf_filename"])["start"].apply(
        lambda s: s.to_numpy())
    tasks = {"small": [], "big": []}
    for r in files.itertuples():
        key = (r.base_folder, r.edf_filename)
        path = Path(args.edf_dir) / r.base_folder / r.edf_filename
        size = "big" if r.length > args.big_file_samples else "small"
        tasks[size].append((key, str(path), starts[key], args.feasibility,
                            args.input_filtered))

    results, errors = [], {}
    bar = tqdm(total=len(files), desc="Test files", unit="file")
    for size, workers in (("small", args.workers), ("big", 1)):
        if not tasks[size]:
            continue
        if workers <= 1:
            it = map(_prepare, tasks[size])
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            it = pool.map(_prepare, tasks[size])
        for key, seg, x, err in it:
            bar.update()
            if err:
                errors["/".join(key)] = err
                tqdm.write(f"Error {'/'.join(key)}: {err}")
                continue
            seg = predictor.finish(seg, x)
            seg["base_folder"], seg["edf_filename"] = key
            results.append(seg)
        if pool:
            pool.shutdown()
    bar.close()
    if not results:
        raise RuntimeError(f"No file could be processed: {errors}")
    pred = pd.concat(results, ignore_index=True)
    return pred, errors


# -----------------------------
# 3. Metrics
# -----------------------------


def score(ref, pred) -> dict:
    """Error metrics as in training.py (NaN rows ignored)."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    ok = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[ok], pred[ok]
    if len(ref) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "acc5": np.nan,
                "bias": np.nan, "bmae": np.nan, "r": np.nan}
    err = pred - ref
    _, bmae = per_bin_errors(ref, pred)
    return {
        "n": int(len(ref)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "acc5": float((np.abs(err) <= 5).mean()),
        "bias": float(err.mean()),
        "bmae": bmae,
        "r": float(np.corrcoef(ref, pred)[0, 1]) if len(ref) > 2 else np.nan,
    }


def summary_rows(ev: pd.DataFrame, mode: str) -> pd.DataFrame:
    n = len(ev)
    other = "full" if mode == "ppg" else "ppg"
    rows = [
        (f"prediction (feasibility={mode})", ev["feasible"], "hr_pred"),
        ("no filter (model on all segments)", slice(None), "hr_model"),
        ("rejected (NONE) segments, model output", ~ev["feasible"],
         "hr_model"),
        (f"{other} borders (from database)"
         + (", = training selection" if other == "full" else ""),
         ev[f"db_feasible_{other}"], "hr_model"),
    ]
    out = []
    for name, mask, col in rows:
        sub = ev.loc[mask]
        out.append({"selection": name, "coverage": len(sub) / max(n, 1),
                    **score(sub["hr_ref"], sub[col])})
    return pd.DataFrame(out)


def group_table(ev: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for name, g in ev.groupby(by):
        s = score(g["hr_ref"], g["hr_pred"])
        rows.append({by: name, "segments": len(g),
                     "coverage": g["feasible"].mean(),
                     "mae": s["mae"], "rmse": s["rmse"], "acc5": s["acc5"],
                     "bias": s["bias"],
                     "mae_no_filter": score(g["hr_ref"], g["hr_model"])["mae"],
                     "hr_ref_mean": g["hr_ref"].mean()})
    return pd.DataFrame(rows)


def cv_reference(model_path: Path) -> dict | None:
    """MAE etc. of the out-of-fold predictions of the model's own fold."""
    try:
        fold = int(model_path.parent.name.split("_")[-1])
    except ValueError:
        return None
    oof = model_path.parent.parent / (
        f"oof_predictions_{model_path.parent.parent.name}.csv")
    if not oof.exists():
        return None
    df = pd.read_csv(oof)
    df = df[df["fold"] == fold]
    if df.empty:
        return None
    return {"selection": f"CV validation fold {fold} (OOF, train split)",
            "coverage": np.nan,
            **score(df["target_hr"], df["predicted_hr"])}


def pipeline_check(ev: pd.DataFrame, mode: str) -> dict:
    """Recomputed metrics vs the database values (should be identical)."""
    diffs = {}
    for col in bounds_for(mode):
        a, b = ev[f"{col}_pred"].to_numpy(float), ev[col].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        diffs[col] = float(np.abs(a[ok] - b[ok]).max()) if ok.any() else np.nan
    agree = float((ev["feasible"] == ev[f"db_feasible_{mode}"]).mean())
    return {"feasibility_agreement": agree,
            "max_abs_metric_diff": max(diffs.values()),
            "metric_diffs": diffs}


# -----------------------------
# 4. Figures and report
# -----------------------------


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)


def plot_scatter(ev: pd.DataFrame, path: Path, title: str):
    sub = ev[ev["feasible"]]
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    style(ax)
    lo = float(np.nanmin([ev["hr_ref"].min(), sub["hr_pred"].min(), 30]))
    hi = float(np.nanmax([ev["hr_ref"].max(), sub["hr_pred"].max(), 200]))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    for d in (-5, 5):
        ax.plot([lo, hi], [lo + d, hi + d], color=MUTED, lw=0.8, ls="--")
    alpha = float(np.clip(300 / max(len(sub), 1), 0.03, 0.6))
    ax.scatter(sub["hr_ref"], sub["hr_pred"], s=4, color=BLUE, alpha=alpha,
               linewidths=0, rasterized=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference HR (ECG) [bpm]")
    ax.set_ylabel("Predicted HR (PPG) [bpm]")
    ax.set_title(title, fontsize=10, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bins(ev: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    style(ax)
    for col, mask, color, label in (
        ("hr_pred", ev["feasible"], BLUE, "prediction (feasible)"),
        ("hr_model", slice(None), ORANGE, "no filter"),
    ):
        sub = ev.loc[mask]
        if sub.empty:
            continue
        bins, _ = per_bin_errors(sub["hr_ref"].to_numpy(),
                                 sub[col].to_numpy())
        bins = bins[bins["n"] >= 20]
        ax.plot(bins.index + 5, bins["mae"], color=color, lw=2, marker="o",
                ms=5, label=label)
    ax.set_xlabel("Reference HR bin centre [bpm] (bins with >= 20 segments)")
    ax.set_ylabel("MAE [bpm]")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out: Path, summary: pd.DataFrame, datasets, subjects, bins,
                 reasons, check, meta) -> None:
    f3 = lambda df: df.round(3)  # noqa: E731
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    main = summary.iloc[0]
    lines = [
        "# Test evaluation",
        "",
        f"- model: `{meta['model_path']}` ({meta['model_name']})",
        f"- feasibility: `{meta['feasibility']}`"
        + (" (PPG borders only, as in real use)" if meta["feasibility"]
           == "ppg" else " (ECG + PPG borders, as in training)"),
        f"- input: `{meta['edf_dir']}`"
        + (" (already filtered)" if meta["input_filtered"] else
           " (raw, filtered by predict.py)"),
        f"- test segments (hr_valid & not flat): {meta['n_segments']} in"
        f" {meta['n_files']} files, {meta['n_subjects']} subjects;"
        f" {meta['seconds']:.0f} s",
        "",
        f"**Result: HR for {pct(main['coverage'])} of the test segments,"
        f" MAE {main['mae']:.2f} bpm, RMSE {main['rmse']:.2f} bpm,"
        f" {pct(main['acc5'])} within +-5 bpm.**",
        "",
        "## Summary",
        "",
        "coverage = share of the test segments in the row; mae/rmse/bias in"
        " bpm; acc5 = share within +-5 bpm; bmae = mean MAE over 10 bpm"
        " bins.",
        "",
        md_table(f3(summary)),
        "",
        "The CV row uses other subjects (train split) and the full borders;"
        " compare it with the full-borders row.",
        "",
        "![scatter](figures/scatter.png)",
        "",
        "## Per dataset",
        "",
        md_table(f3(datasets)),
        "",
        "## Per subject",
        "",
        md_table(f3(subjects)),
        "",
        "## Per HR bin (prediction)",
        "",
        md_table(f3(bins.reset_index()).astype({"bin": str, "n": str})
                 if len(bins) else bins),
        "",
        "![bins](figures/mae_per_bin.png)",
        "",
        "## Why segments got NONE",
        "",
        "A segment can fail several checks.",
        "",
        md_table(reasons),
        "",
        "## Pipeline check",
        "",
        "Metrics recomputed by predict.py vs the database values"
        f" (max abs difference): {check['max_abs_metric_diff']:.3g};"
        " feasibility agreement with the database borders:"
        f" {pct(check['feasibility_agreement'])}.",
    ]
    if meta["errors"]:
        lines += ["", "## Files that failed", ""]
        lines += [f"- `{k}`: {v}" for k, v in meta["errors"].items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# 5. Main
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model on the test database with"
        " predict.py")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--model", default="auto",
                   choices=["auto", *MODEL_REGISTRY])
    p.add_argument("--feasibility", default="ppg", choices=FEASIBILITY_MODES)
    p.add_argument("--parquet-dir",
                   default=str(ROOT / "data" / "hr_parts" / "seg10s_step5s"))
    p.add_argument("--edf-dir", default=None,
                   help="raw EDF database (default"
                   " data/harmonized_ecg_ppg_databases, or the filtered one"
                   " with --input-filtered)")
    p.add_argument("--input-filtered", action="store_true",
                   help="read data/harmonized_ecg_ppg_database_filtered"
                   " directly (skips the filtering; same signals)")
    p.add_argument("--output-dir", default=None,
                   help="default data/test_evaluation/<model>_<fold>_<mode>")
    p.add_argument("--all-segments", action="store_true",
                   help="also run the flat / no-reference segments"
                   " (slow; only adds their NONE rate)")
    p.add_argument("--limit-files", type=int, default=None,
                   help="only the first N files per dataset (smoke test)")
    p.add_argument("--workers", type=int, default=min(2, os.cpu_count()),
                   help="processes for filtering + metrics (RAM: ~1-2 GB"
                   " each)")
    p.add_argument("--big-file-samples", type=float, default=2e7,
                   help="recordings longer than this run one at a time")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.edf_dir is None:
        args.edf_dir = str(ROOT / "data" / (
            "harmonized_ecg_ppg_database_filtered" if args.input_filtered
            else "harmonized_ecg_ppg_databases"))
    if args.output_dir is None:
        mp = Path(args.model_path)
        args.output_dir = str(ROOT / "data" / "test_evaluation" / (
            f"{mp.parent.parent.name}_{mp.parent.name}_{args.feasibility}"))
    return args


def main():
    args = parse_args()
    t0 = time.perf_counter()
    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    predictor = HRPredictor(args.model_path, args.model, args.device,
                            args.feasibility, args.batch_size)
    print(f"Model: {predictor.model_name} ({args.model_path}) on"
          f" {predictor.device}, feasibility: {args.feasibility}")

    test = load_test_segments(Path(args.parquet_dir), args.all_segments)
    files = pick_files(test, args.limit_files)
    test = test.merge(files[["base_folder", "edf_filename"]])
    print(f"Test: {len(test)} segments ({int(test['evaluable'].sum())} with"
          f" reference HR), {len(files)} files from {args.edf_dir}")

    pred, errors = run_predictions(test, files, predictor, args)
    pred = pred.rename(columns={"hr_bpm": "hr_pred"})
    keep = [*KEY, "feasible", "reason", "hr_model", "hr_pred",
            *bounds_for(args.feasibility)]
    df = test.merge(pred[[c for c in keep if c in pred]], on=KEY,
                    how="inner", suffixes=("", "_pred"))
    ev = df[df["evaluable"]].copy()
    if ev.empty:
        raise RuntimeError("No evaluable test segment was predicted")

    summary = summary_rows(ev, args.feasibility)
    cv = cv_reference(Path(args.model_path))
    if cv:
        summary = pd.concat([summary, pd.DataFrame([cv])], ignore_index=True)
    datasets = group_table(ev, "base_folder")
    subjects = group_table(ev, "subject")
    bins, _ = per_bin_errors(ev.loc[ev["feasible"], "hr_ref"].to_numpy(),
                             ev.loc[ev["feasible"], "hr_pred"].to_numpy()) \
        if ev["feasible"].any() else (pd.DataFrame(), np.nan)
    reasons = (ev.loc[~ev["feasible"], "reason"].str.split(";").explode()
               .value_counts().rename_axis("check").reset_index(name="segments"))
    reasons["share_of_test"] = (reasons["segments"] / len(ev)).round(3)
    check = pipeline_check(ev, args.feasibility)

    title = (f"{predictor.model_name}, {Path(args.model_path).parent.name},"
             f" feasible test segments (n={int(ev['feasible'].sum())})")
    if ev["feasible"].any():
        plot_scatter(ev, out / "figures" / "scatter.png", title)
    plot_bins(ev, out / "figures" / "mae_per_bin.png")

    meta = {
        "model_path": str(args.model_path),
        "model_name": predictor.model_name,
        "feasibility": args.feasibility,
        "edf_dir": args.edf_dir,
        "input_filtered": args.input_filtered,
        "n_segments": len(ev),
        "n_files": int(files.shape[0] - len(errors)),
        "n_subjects": int(ev["subject"].nunique()),
        "seconds": time.perf_counter() - t0,
        "errors": errors,
    }
    write_report(out, summary, datasets, subjects, bins, reasons, check, meta)
    if args.all_segments:
        meta["none_rate_all_segments"] = float(1 - df["feasible"].mean())
    (out / "summary.json").write_text(json.dumps({
        **meta, "summary": summary.to_dict("records"),
        "datasets": datasets.to_dict("records"),
        "subjects": subjects.to_dict("records"),
        "pipeline_check": check,
    }, indent=2, default=float), encoding="utf-8")
    seg_cols = ["base_folder", "edf_filename", "subject", "start", "end",
                "evaluable", "hr_ref", "hr_model", "hr_pred", "feasible",
                "reason", "db_feasible_ppg", "db_feasible_full"]
    df[seg_cols].to_parquet(out / "segments.parquet", index=False)

    print("\n" + md_table(summary.round(3)))
    print(f"\nPipeline check: feasibility agreement"
          f" {check['feasibility_agreement']:.2%}, max metric diff"
          f" {check['max_abs_metric_diff']:.3g}")
    print(f"Report: {out / 'report.md'} ({meta['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
