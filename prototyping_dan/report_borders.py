"""
Report of a feasibility-border sweep (sweep_borders.py).

Reads <results-dir>/results/*.json, preds/*.npy and val_index.parquet and
writes <report-dir>/report.md, figures/ and CSVs:
    sweep_results.csv, aurc_per_variable.csv, per_subject_coverage.csv

The metric / control-matching / selection functions are also used by
sweep_borders.py (grid planning, checkpoint pruning), so both apply the same
rules.

Usage (local or Kaggle):
    python prototyping_dan/report_borders.py --results-dir data/sweep_borders/fold0
    python prototyping_dan/report_borders.py --results-dir /kaggle/working/sweep_results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from prototyping_dan.feasible_bpm import FEASIBILITY_BOUNDS, within_bounds

# Reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 2,
    "axes.spines.top": False, "axes.spines.right": False,
})


# -----------------------------
# Metrics (shared with sweep_borders.py)
# -----------------------------


def _mae_rmse(err: np.ndarray) -> tuple[float, float]:
    if len(err) == 0:
        return float("nan"), float("nan")
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def coverage_stats(mask: np.ndarray, groups: np.ndarray, prefix: str
                   ) -> dict:
    """Min / 10th percentile / median coverage per group, and #groups < 50%."""
    cov = pd.Series(mask.astype(float)).groupby(groups).mean()
    return {
        f"{prefix}_cov_min": float(cov.min()),
        f"{prefix}_cov_p10": float(cov.quantile(0.10)),
        f"{prefix}_cov_median": float(cov.median()),
        f"{prefix}_below_50": int((cov < 0.5).sum()),
        f"n_{prefix}s": int(len(cov)),
    }


def compute_metrics(val_df: pd.DataFrame, preds: np.ndarray,
                    mask: np.ndarray) -> dict:
    """
    Error and coverage metrics of predictions on all validation segments,
    split by the feasibility mask.
    """
    mask = np.asarray(mask, dtype=bool)
    err = preds.astype(np.float64) - val_df["hr_bpm"].to_numpy()
    m = {"coverage_val": float(mask.mean()), "n_val_feasible": int(mask.sum()),
         "n_val": int(len(mask))}
    m["mae_feasible"], m["rmse_feasible"] = _mae_rmse(err[mask])
    m["mae_all"], m["rmse_all"] = _mae_rmse(err)
    m["mae_rejected"], m["rmse_rejected"] = _mae_rmse(err[~mask])
    m["acc5_feasible"] = (float((np.abs(err[mask]) <= 5).mean())
                          if mask.any() else float("nan"))
    m["bias_feasible"] = float(err[mask].mean()) if mask.any() else float("nan")
    m.update(coverage_stats(mask, val_df["subject"].to_numpy(), "subject"))
    m.update(coverage_stats(mask, val_df["recording"].to_numpy(), "recording"))
    return m


def borders_mask(df: pd.DataFrame, borders: dict) -> np.ndarray:
    """Feasibility mask of borders {col: [lo, hi]} (uses feasible_bpm.py)."""
    return within_bounds(
        df, {c: (b[0], b[1]) for c, b in borders.items()}
    ).to_numpy()


def ppg_only(borders: dict) -> dict:
    return {c: b for c, b in borders.items() if c.startswith("ppg_")}


# -----------------------------
# Loading and control matching
# -----------------------------


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results_dir / "results").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        row = {k: v for k, v in r.items()
               if k not in ("metrics", "history", "borders", "metrics_ppg")}
        row.update(r["metrics"])
        row.update({f"ppg_{k}": v for k, v in r.get("metrics_ppg", {}).items()
                    if k in ("coverage_val", "mae_feasible", "mae_rejected")})
        row["borders"] = r["borders"]
        row["history"] = r["history"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_preds(results_dir: Path, h: str) -> np.ndarray:
    return np.load(results_dir / "preds" / f"{h}.npy")


def add_control_metrics(df: pd.DataFrame, val_df: pd.DataFrame,
                        results_dir: Path, bucket: float) -> pd.DataFrame:
    """
    For every border config, the random-subset control with the nearest
    training coverage (same seed, within `bucket`) is evaluated on the
    config's own feasible validation segments, so the only difference
    between the two models is which training segments were used.
    """
    df = df.copy()
    for col in ("control_hash", "control_coverage_train",
                "control_mae_feasible", "control_rmse_feasible",
                "control_mae_all", "gain_vs_control"):
        df[col] = np.nan if col != "control_hash" else None
    controls = df[df["kind"] == "random"]
    none_rows = df[df["kind"] == "none"]
    pred_cache = {}
    for i, row in df.iterrows():
        if row["kind"] == "random":
            continue
        cands = controls[controls["seed"] == row["seed"]]
        # The unfiltered model is its own control at coverage 1
        cands = pd.concat([cands, none_rows[none_rows["seed"] == row["seed"]]])
        if row["kind"] == "none" or cands.empty:
            continue
        d = (cands["coverage_train"] - row["coverage_train"]).abs()
        j = d.idxmin()
        if d[j] > bucket:
            continue
        h = cands.at[j, "hash"]
        if h not in pred_cache:
            pred_cache[h] = load_preds(results_dir, h)
        mask = borders_mask(val_df, row["borders"])
        m = compute_metrics(val_df, pred_cache[h], mask)
        df.at[i, "control_hash"] = h
        df.at[i, "control_coverage_train"] = cands.at[j, "coverage_train"]
        df.at[i, "control_mae_feasible"] = m["mae_feasible"]
        df.at[i, "control_rmse_feasible"] = m["rmse_feasible"]
        df.at[i, "control_mae_all"] = m["mae_all"]
        df.at[i, "gain_vs_control"] = m["mae_feasible"] - row["mae_feasible"]
    df["beats_control"] = df["gain_vs_control"] > 0
    return df


# -----------------------------
# Selection and AURC
# -----------------------------


def rank_configs(df: pd.DataFrame, max_mae: float, min_coverage: float | None,
                 min_subject_coverage: float) -> tuple[pd.DataFrame, str]:
    """
    Ranks the border configs (main seed only) by the selection rule.

    Default: highest coverage_val with mae_feasible <= max_mae and
    subject_cov_min >= min_subject_coverage. With min_coverage: lowest
    mae_feasible with coverage_val >= min_coverage. A config must also beat
    its random-subset control. If nothing qualifies, the subject constraint
    and then the MAE constraint are dropped (noted in the returned text).
    """
    cand = df[df["kind"].isin(["current", "oat", "grid"])].copy()
    if "is_main_seed" in cand:
        cand = cand[cand["is_main_seed"]]
    if cand.empty:
        return cand, "no border configs finished"
    beats = cand["beats_control"].fillna(False)
    subj = cand["subject_cov_min"] >= min_subject_coverage
    if min_coverage is not None:
        perf = cand["coverage_val"] >= min_coverage
        sort_by, asc = ["mae_feasible", "coverage_val"], [True, False]
        rule = f"lowest mae_feasible with coverage_val >= {min_coverage}"
    else:
        perf = cand["mae_feasible"] <= max_mae
        sort_by, asc = ["coverage_val", "mae_feasible"], [False, True]
        rule = f"highest coverage_val with mae_feasible <= {max_mae} bpm"
    rule += (f", min subject coverage >= {min_subject_coverage},"
             " beating the random-subset control")
    note = ""
    for level, ok in (
        ("all constraints", perf & subj & beats),
        ("subject-coverage constraint dropped", perf & beats),
        ("subject-coverage and MAE/coverage constraints dropped", beats),
    ):
        if ok.any():
            if level != "all constraints":
                note = (f"No config met every constraint ({rule}); fallback:"
                        f" {level}.")
                if level.endswith("dropped") and "MAE" in level:
                    sort_by, asc = ["mae_feasible"], [True]
            cand["eligible"] = ok
            ranked = pd.concat([
                cand[ok].sort_values(sort_by, ascending=asc),
                cand[~ok].sort_values("mae_feasible"),
            ])
            return ranked, note or f"Rule: {rule}."
    cand["eligible"] = False
    return (cand.sort_values("mae_feasible"),
            f"No config beat its random-subset control ({rule}).")


def variable_curves(df: pd.DataFrame, manifest: dict | None = None
                    ) -> dict[str, pd.DataFrame]:
    """One-at-a-time points per variable, the current config included."""
    main = df[df["is_main_seed"]] if "is_main_seed" in df else df
    current = main[main["kind"] == "current"]
    curves = {}
    for var in sorted(main.loc[main["kind"] == "oat", "var"].dropna().unique()):
        cur = current.copy()
        if manifest and len(cur):
            cur["level_actual"] = manifest["current_levels"][var]
            lo, hi = manifest["current_borders"][var]
            f = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
            cur["border_text"] = f"{f(lo)}/{f(hi)}"
        pts = pd.concat([main[(main["kind"] == "oat") & (main["var"] == var)],
                         cur])
        curves[var] = pts.sort_values("coverage_val")
    return curves


def _aurc(x: np.ndarray, y: np.ndarray) -> float:
    """Mean risk over the covered range (trapezoid / range width)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or x.max() == x.min():
        return float("nan")
    o = np.argsort(x)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y[o], x[o]) / (x.max() - x.min()))


def aurc_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    AURC of every variable's one-at-a-time risk-coverage curve
    (coverage_val vs mae_feasible), the AURC of the matched random controls
    on the same points, and the effect of removing the border.
    """
    rows = []
    for var, pts in variable_curves(df).items():
        x = pts["coverage_val"].to_numpy(float)
        cur = pts[pts["kind"] == "current"]
        removed = pts[(pts["kind"] == "oat") & (pts["level"] == 0)]
        r = {
            "variable": var,
            "n_points": len(pts),
            "coverage_min": float(np.nanmin(x)),
            "coverage_max": float(np.nanmax(x)),
            "aurc": _aurc(x, pts["mae_feasible"].to_numpy(float)),
            "aurc_control": _aurc(x, pts["control_mae_feasible"].to_numpy(float)),
            "slope_mae_per_coverage": float("nan"),
            "delta_mae_if_removed": float("nan"),
            "delta_cov_if_removed": float("nan"),
        }
        r["aurc_gain"] = r["aurc_control"] - r["aurc"]
        ok = np.isfinite(x) & np.isfinite(pts["mae_feasible"].to_numpy(float))
        if ok.sum() >= 2 and np.ptp(x[ok]) > 0:
            r["slope_mae_per_coverage"] = float(np.polyfit(
                x[ok], pts["mae_feasible"].to_numpy(float)[ok], 1)[0])
        if len(cur) and len(removed):
            r["delta_mae_if_removed"] = float(
                removed["mae_feasible"].iloc[0] - cur["mae_feasible"].iloc[0])
            r["delta_cov_if_removed"] = float(
                removed["coverage_val"].iloc[0] - cur["coverage_val"].iloc[0])
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        # Useful = filtering on it lowers the error more than random removal
        out = out.sort_values("slope_mae_per_coverage", ascending=False)
        out["useful"] = (out["slope_mae_per_coverage"] > 0) & (
            out["delta_mae_if_removed"].fillna(1) > 0)
    return out


def oracle_curve(val_df: pd.DataFrame, preds: np.ndarray,
                 coverages: np.ndarray) -> np.ndarray:
    """MAE when rejecting the segments with the largest true error."""
    err = np.sort(np.abs(preds - val_df["hr_bpm"].to_numpy()))
    csum = np.cumsum(err)
    n = np.maximum((coverages * len(err)).astype(int), 1)
    return csum[n - 1] / n


# -----------------------------
# Report
# -----------------------------


def fmt_borders(borders: dict) -> str:
    """FEASIBILITY_BOUNDS-style code block."""
    lines = ["FEASIBILITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {"]
    for col, (lo, hi) in borders.items():
        f = lambda v: "None" if v is None else f"{v:.4g}"  # noqa: E731
        lines.append(f'    "{col}": ({f(lo)}, {f(hi)}),')
    lines.append("}")
    return "\n".join(lines)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(none)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def plot_risk_coverage(df, curves, val_df, results_dir, fig_dir, aurc):
    controls = df[(df["kind"] == "random") & df["is_main_seed"]]
    controls = controls.sort_values("coverage_train")
    none = df[(df["kind"] == "none") & df["is_main_seed"]]
    xs = np.linspace(0.05, 1.0, 96)
    oracle = (oracle_curve(val_df, load_preds(results_dir, none["hash"].iloc[0]),
                           xs) if len(none) else None)

    def references(ax):
        if len(controls):
            ax.plot(controls["coverage_val"], controls["mae_feasible"], "--",
                    color=MUTED, lw=1.5, label="random subset (control)")
        if oracle is not None:
            ax.plot(xs, oracle, ":", color=INK2, lw=1.5,
                    label="oracle (reject by true error)")
        if len(none):
            ax.scatter(none["coverage_val"], none["mae_feasible"], s=40,
                       color=INK, zorder=5, label="no filtering")

    # Overview: the 5 most useful variables
    top = list(aurc["variable"].head(5)) if not aurc.empty else []
    fig, ax = plt.subplots(figsize=(8, 5))
    references(ax)
    for k, var in enumerate(top):
        p = curves[var]
        ax.plot(p["coverage_val"], p["mae_feasible"], "-o", ms=4,
                color=SERIES[k], label=var)
    cur = df[(df["kind"] == "current") & df["is_main_seed"]]
    if len(cur):
        ax.scatter(cur["coverage_val"], cur["mae_feasible"], marker="*", s=160,
                   color=ORANGE, edgecolor=SURFACE, zorder=6,
                   label="current borders")
    ax.set_xlabel("coverage of validation segments (feasible share)")
    ax.set_ylabel("MAE on feasible segments (bpm)")
    ax.set_title("Risk-coverage, top 5 border variables (by slope)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_coverage_overview.png", dpi=130)
    plt.close(fig)

    # Small multiples: one panel per variable
    n = len(curves)
    if n == 0:
        return
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (var, p) in zip(axes.flat, curves.items()):
        references(ax)
        ax.plot(p["coverage_val"], p["mae_feasible"], "-o", ms=4, color=BLUE,
                label="this border varied")
        c = p[p["kind"] == "current"]
        ax.scatter(c["coverage_val"], c["mae_feasible"], marker="*", s=120,
                   color=ORANGE, zorder=6, label="current")
        ax.set_title(var)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper left", fontsize=7)
    fig.supxlabel("coverage of validation segments")
    fig.supylabel("MAE on feasible segments (bpm)")
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_coverage_per_variable.png", dpi=110)
    plt.close(fig)


def plot_variable_detail(curves, fig_dir) -> list[str]:
    """MAE (feasible/all/rejected) and coverage vs border level per variable."""
    names = []
    for var, p in curves.items():
        p = p.copy()
        cur_level = p.loc[p["kind"] == "current", "level_actual"]
        p["x"] = p["level_actual"]
        p = p.sort_values("x")
        labels = [f"{x:.0%}\n{b}" for x, b in zip(p["x"], p["border_text"])]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
        for col, color, lab in (("mae_feasible", BLUE, "feasible"),
                                ("mae_all", MUTED, "all"),
                                ("mae_rejected", ORANGE, "rejected")):
            a1.plot(p["x"], p[col], "-o", ms=4, color=color, label=lab)
        if "control_mae_feasible" in p:
            a1.plot(p["x"], p["control_mae_feasible"], "--", color=INK2, lw=1.2,
                    label="feasible, random control")
        a2.plot(p["x"], p["coverage_train"], "-o", ms=4, color=BLUE,
                label="train")
        a2.plot(p["x"], p["coverage_val"], "-o", ms=4, color=AQUA, label="val")
        for a in (a1, a2):
            if len(cur_level):
                a.axvline(cur_level.iloc[0], color=ORANGE, lw=1, ls=":")
            a.set_xticks(p["x"])
            a.set_xticklabels(labels, fontsize=6)
            a.set_xlabel("share rejected by this border alone (border lo/hi)")
            a.legend()
        a1.set_ylabel("MAE (bpm)")
        a2.set_ylabel("coverage")
        fig.suptitle(f"{var} (dotted line: current border)", fontsize=10)
        fig.tight_layout()
        name = f"border_{var}.png"
        fig.savefig(fig_dir / name, dpi=110)
        plt.close(fig)
        names.append(name)
    return names


def plot_learning_curves(rows: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, (_, r) in enumerate(rows.iterrows()):
        h = r["history"]
        c = SERIES[k % len(SERIES)]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "--", color=c, lw=1.2)
        axes[0].plot(ep, h["val_loss"], "-", color=c, label=r["name"])
        axes[1].plot(ep, h["val_mae"], "-", color=c, label=r["name"])
    axes[0].set_title("loss (solid: val, dashed: train)")
    axes[1].set_title("val MAE on the early-stopping set (bpm)")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=110)
    plt.close(fig)


def build_report(results_dir: Path, report_dir: Path | None = None,
                 max_mae: float = 5.0, min_coverage: float | None = None,
                 min_subject_coverage: float = 0.5,
                 bucket: float | None = None) -> Path:
    results_dir = Path(results_dir)
    report_dir = Path(report_dir or results_dir / "report")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((results_dir / "sweep_manifest.json").read_text())
    bucket = bucket or manifest.get("rand_bucket", 0.025)
    val_df = pd.read_parquet(results_dir / "val_index.parquet")

    df = load_results(results_dir)
    if df.empty:
        raise FileNotFoundError(f"No results in {results_dir / 'results'}")
    df["is_main_seed"] = df["seed"] == manifest["seed"]
    df = add_control_metrics(df, val_df, results_dir, bucket)
    ranked, rule_note = rank_configs(df, max_mae, min_coverage,
                                     min_subject_coverage)
    aurc = aurc_table(df)
    curves = variable_curves(df, manifest)

    # --- CSVs ---
    flat = df.drop(columns=["history"]).copy()
    flat["borders"] = flat["borders"].apply(json.dumps)
    flat.to_csv(report_dir / "sweep_results.csv", index=False)
    aurc.to_csv(report_dir / "aurc_per_variable.csv", index=False)

    main = df[df["is_main_seed"]]
    get = lambda kind: main[main["kind"] == kind]  # noqa: E731
    best = ranked.iloc[0] if len(ranked) else None

    # Per-subject coverage of the recommended / current / no filtering
    subj_rows = []
    for label, row in (("recommended", best),
                       ("current", get("current").iloc[0] if len(get("current")) else None)):
        if row is None:
            continue
        mask = borders_mask(val_df, row["borders"])
        preds = load_preds(results_dir, row["hash"])
        err = np.abs(preds - val_df["hr_bpm"].to_numpy())
        g = pd.DataFrame({"subject": val_df["subject"], "mask": mask,
                          "err": err})
        t = g.groupby("subject").agg(
            n=("mask", "size"), coverage=("mask", "mean"),
            mae_all=("err", "mean"))
        t["mae_feasible"] = g[g["mask"]].groupby("subject")["err"].mean()
        t["config"] = label
        subj_rows.append(t.reset_index())
    subj = pd.concat(subj_rows) if subj_rows else pd.DataFrame()
    subj.to_csv(report_dir / "per_subject_coverage.csv", index=False)

    # --- Figures ---
    plot_risk_coverage(df, curves, val_df, results_dir, fig_dir, aurc)
    detail_figs = plot_variable_detail(curves, fig_dir)
    if not subj.empty:
        rec = subj[subj["config"] == "recommended"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(rec["coverage"], bins=np.linspace(0, 1, 21), color=BLUE,
                edgecolor=SURFACE, linewidth=2)
        ax.axvline(min_subject_coverage, color=ORANGE, ls=":", lw=1.5)
        ax.set_xlabel("coverage per subject (validation)")
        ax.set_ylabel("subjects")
        ax.set_title("Per-subject coverage, recommended borders")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "per_subject_coverage.png", dpi=110)
        plt.close(fig)
    top3 = ranked.head(3)
    lc_rows = pd.concat([get("none"), get("current"), top3]).drop_duplicates(
        "hash")
    plot_learning_curves(lc_rows, fig_dir)

    # --- Markdown ---
    def summary_row(label, r):
        if r is None:
            return None
        return {
            "config": label, "name": r["name"],
            "coverage_train": r["coverage_train"],
            "coverage_val": r["coverage_val"],
            "mae_feasible": r["mae_feasible"], "rmse_feasible": r["rmse_feasible"],
            "mae_all": r["mae_all"], "mae_rejected": r["mae_rejected"],
            "worst_subject_cov": r["subject_cov_min"],
            "subjects_below_50%": r["subject_below_50"],
            "control_mae_feasible": r.get("control_mae_feasible", np.nan),
            "gain_vs_control": r.get("gain_vs_control", np.nan),
            "ppg_only_coverage_val": r.get("ppg_coverage_val", np.nan),
            "ppg_only_mae_feasible": r.get("ppg_mae_feasible", np.nan),
        }

    srows = [summary_row("recommended", best),
             summary_row("current", get("current").iloc[0] if len(get("current")) else None),
             summary_row("no filtering", get("none").iloc[0] if len(get("none")) else None)]
    summary = pd.DataFrame([r for r in srows if r is not None])

    gpu = df["gpu"].dropna().unique()
    lines = [
        "# Feasibility-border sweep",
        "",
        f"- Fold {manifest['fold']} of {manifest['n_splits']}"
        f" (validation: {len(val_df)} segments, {val_df['subject'].nunique()}"
        f" subjects); seed {manifest['seed']}; max epochs {manifest['epochs']}",
        f"- Finished configs: {len(df)} ({(df['kind'] == 'random').sum()}"
        " random-subset controls)",
        f"- {rule_note}",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "`gain_vs_control` = MAE of the random-subset control (same training"
        " size) on the same feasible validation segments minus the config's"
        " MAE; positive means the borders select better training data than"
        " chance. `ppg_only_*` applies only the PPG borders of the config to"
        " the validation segments (what is available at prediction time).",
        "",
        "## Recommended `metrics_borders`",
        "",
    ]
    if best is not None:
        lines += ["```python", fmt_borders(best["borders"]), "```", ""]
    lines += [
        "## Top configs",
        "",
        md_table(ranked.head(10)[[
            "name", "kind", "eligible", "coverage_train", "coverage_val",
            "mae_feasible", "mae_all", "mae_rejected", "subject_cov_min",
            "control_mae_feasible", "gain_vs_control"]]),
        "",
        "## Risk-coverage",
        "",
        "![overview](figures/risk_coverage_overview.png)",
        "",
        "![per variable](figures/risk_coverage_per_variable.png)",
        "",
        "## Ranking of border variables",
        "",
        "`slope_mae_per_coverage` > 0: rejecting more on this variable lowers"
        " the feasible MAE. `aurc` = mean feasible MAE over the covered range;"
        " `aurc_gain` = control AURC minus AURC on the same points."
        " `delta_*_if_removed` = change when the border is removed entirely"
        " (others at current). Variables marked not useful can be dropped.",
        "",
        md_table(aurc),
        "",
        "## Per variable",
        "",
    ]
    lines += [f"![{n}](figures/{n})" for n in detail_figs]
    lines += ["", "## Per-subject coverage (validation)", ""]
    if not subj.empty:
        lines += ["![per subject](figures/per_subject_coverage.png)", "",
                  md_table(subj.sort_values(["config", "coverage"]))]
    lines += [
        "", "## Training", "",
        "![learning curves](figures/learning_curves.png)", "",
        f"- GPU: {', '.join(map(str, gpu)) or 'n/a'}",
        f"- Total training time: {df['train_seconds'].sum() / 3600:.2f} h"
        f" over {len(df)} configs",
        "",
        md_table(lc_rows[["name", "n_train", "epochs_run", "best_epoch",
                          "best_val_loss", "seconds_per_epoch",
                          "train_seconds"]]),
        "",
    ]
    multi = df[df["kind"] != "random"].groupby("borders_key").filter(
        lambda g: g["seed"].nunique() > 1)
    if not multi.empty:
        seeds = multi.groupby("borders_key").agg(
            name=("name", "first"), n_seeds=("seed", "nunique"),
            coverage_val=("coverage_val", "mean"),
            mae_feasible_mean=("mae_feasible", "mean"),
            mae_feasible_std=("mae_feasible", "std"),
            mae_all_mean=("mae_all", "mean"), mae_all_std=("mae_all", "std"),
        ).reset_index(drop=True)
        lines += ["## Seed repeats (mean ± std)", "", md_table(seeds), ""]
    lines += ["All raw numbers: `sweep_results.csv`, `aurc_per_variable.csv`,"
              " `per_subject_coverage.csv`."]
    out = report_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", required=True)
    p.add_argument("--report-dir", default=None,
                   help="default: <results-dir>/report")
    p.add_argument("--max-mae", type=float, default=5.0)
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--min-subject-coverage", type=float, default=0.5)
    a = p.parse_args()
    build_report(Path(a.results_dir), a.report_dir, a.max_mae, a.min_coverage,
                 a.min_subject_coverage)


if __name__ == "__main__":
    main()
