"""
Runs the HR model on one EDF file or a folder of EDF files.

It reads the raw PPG of each recording and calls the entry point
estimate_heartrate(ppg, fs) (estimate_heartrate.py), so it tests exactly the
required interface. For every recording it saves the HR at 2 Hz, the HR per
10 s segment (NONE if the PPG is not usable) and, if the file has an ECG
channel, a comparison with the ECG heart rate.

    uv run python test_interface.py <file.edf | folder> --model-path best_model.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from math import gcd  # noqa: E402

import scipy.signal  # noqa: E402

from databasing.pan_tompkins.heart_rate import estimate_hr  # noqa: E402
from estimate_heartrate import (  # noqa: E402
    OUTPUT_FS,
    segment_heartrate,
    to_output_rate,
)
from exploratory_analysis.filters import apply_standard_filters  # noqa: E402
from predict import (  # noqa: E402
    DEFAULT_MODEL,
    MODEL_FS,
    SEGMENT_S,
    STEP_S,
    recording_hr,
)
from utils import read_edf  # noqa: E402


def load_raw(path: Path):
    """Raw ECG (or None) and PPG of a recording, and the sampling rate."""
    ecg, ppg, meta = read_edf.read_edf_signals(str(path))
    read_edf._RAW_CACHE.pop(str(path), None)
    if ppg is None:
        raise ValueError("no PPG channel")
    return ecg, ppg, float(meta["sampling_rate"])


def filtered_ecg(ecg: np.ndarray, fs: float) -> np.ndarray:
    """ECG filtered like the database, at the model's sampling rate."""
    ecg, _ = apply_standard_filters(ecg, None, fs=fs)
    if round(fs) != MODEL_FS:
        g = gcd(int(round(fs)), MODEL_FS)
        ecg = scipy.signal.resample_poly(ecg, MODEL_FS // g,
                                         int(round(fs)) // g)
    return ecg


def reference_hr(ecg: np.ndarray, fs: float) -> pd.DataFrame:
    ref, _ = estimate_hr(ecg, fs, SEGMENT_S, STEP_S)
    ref = ref[["start", "hr_bpm", "valid"]].rename(
        columns={"hr_bpm": "ref_hr_bpm"})
    ref.loc[~ref["valid"], "ref_hr_bpm"] = np.nan
    return ref.drop(columns="valid")


def error_stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {k: np.nan for k in ("MAE", "RMSE", "bias", "LoA_low",
                                    "LoA_high", "within_5bpm_%")}
    sd = err.std(ddof=1) if len(err) > 1 else 0.0
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias": err.mean(),
        "LoA_low": err.mean() - 1.96 * sd,
        "LoA_high": err.mean() + 1.96 * sd,
        "within_5bpm_%": (np.abs(err) <= 5).mean() * 100,
    }


def process(path: Path, model_path: Path, out_dir: Path,
            use_reference: bool, max_gap_s: float):
    ecg_raw, ppg, fs = load_raw(path)
    duration = len(ppg) / fs
    seg = segment_heartrate(ppg, fs, model_path)
    hr_2hz = to_output_rate(seg, duration, max_gap_s=max_gap_s)
    ecg = (filtered_ecg(ecg_raw, fs)
           if use_reference and ecg_raw is not None else None)

    table = pd.DataFrame({
        "start_s": seg["start_t"].round(2),
        "end_s": seg["end_t"].round(2),
        "hr_bpm": seg["hr_bpm"].round(2),
        "reason": np.where(seg["feasible"], "", seg["reason"]),
    })
    if ecg is not None:
        ref = reference_hr(ecg, MODEL_FS)
        table["ref_hr_bpm"] = (
            seg[["start"]].merge(ref, on="start", how="left")["ref_hr_bpm"]
            .round(2).to_numpy())

    name = path.stem
    table.to_csv(out_dir / f"{name}_segments.csv", index=False)
    pd.DataFrame({"time_s": np.arange(len(hr_2hz)) / OUTPUT_FS,
                  "hr_bpm": np.round(hr_2hz, 2)}).to_csv(
        out_dir / f"{name}_hr_2hz.csv", index=False)

    row = {
        "file": str(path),
        "duration_s": round(duration, 1),
        "segments": len(table),
        "coverage_%": table["hr_bpm"].notna().mean() * 100 if len(table)
        else 0.0,
        "recording_hr_bpm": recording_hr(seg),
    }
    errors = np.array([])
    if ecg is not None:
        both = table.dropna(subset=["hr_bpm", "ref_hr_bpm"])
        errors = (both["hr_bpm"] - both["ref_hr_bpm"]).to_numpy()
        row["reference_segments"] = int(table["ref_hr_bpm"].notna().sum())
        row["compared_segments"] = len(both)
        row["reference_hr_bpm"] = table["ref_hr_bpm"].median()
        row.update(error_stats(errors))
        means = ((both["hr_bpm"] + both["ref_hr_bpm"]) / 2).to_numpy()
    else:
        means = np.array([])
    return row, errors, means


def bland_altman(errors: np.ndarray, means: np.ndarray, path: Path):
    s = error_stats(errors)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(means, errors, s=4, alpha=0.3, color="#2a78d6",
               linewidths=0)
    for y, label, style in [
        (s["bias"], f"bias {s['bias']:+.2f}", "-"),
        (s["LoA_high"], f"+1.96 SD {s['LoA_high']:+.1f}", "--"),
        (s["LoA_low"], f"-1.96 SD {s['LoA_low']:+.1f}", "--"),
    ]:
        ax.axhline(y, color="#eb6834", linewidth=1.5, linestyle=style)
        ax.annotate(label, (1, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("Mean of predicted and reference HR (bpm)")
    ax.set_ylabel("Predicted - reference (bpm)")
    ax.set_title(f"Bland-Altman, {len(errors):,} segments", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_edfs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.rglob("*")
                      if p.suffix.lower() == ".edf")
    return [target]


def parse_args():
    p = argparse.ArgumentParser(description="HR from PPG for EDF recordings")
    p.add_argument("input", type=Path, help="EDF file or folder")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "test_interface")
    p.add_argument("--max-gap-s", type=float, default=15.0,
                   help="max gap (s) to interpolate in the 2 Hz output")
    p.add_argument("--no-reference", action="store_true",
                   help="skip the ECG comparison")
    return p.parse_args()


def main():
    a = parse_args()
    files = find_edfs(a.input)
    if not files:
        sys.exit(f"No EDF files found in {a.input}")
    if not Path(a.model_path).exists():
        sys.exit(f"Model file not found: {a.model_path}")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    rows, all_err, all_mean = [], [], []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name} ...", end=" ", flush=True)
        try:
            row, err, mean = process(path, a.model_path, a.output_dir,
                                     not a.no_reference, a.max_gap_s)
        except Exception as e:
            print(f"FAILED ({e})")
            rows.append({"file": str(path), "error": str(e)})
            continue
        rows.append(row)
        all_err.append(err)
        all_mean.append(mean)
        hr = row["recording_hr_bpm"]
        msg = (f"HR {hr:.1f} bpm" if hr is not None else "HR NONE")
        msg += f", coverage {row['coverage_%']:.0f} %"
        if not np.isnan(row.get("MAE", np.nan)):
            msg += f", MAE vs ECG {row['MAE']:.2f} bpm"
        print(msg)

    summary = pd.DataFrame(rows)
    summary.round(3).to_csv(a.output_dir / "summary.csv", index=False)

    err = np.concatenate(all_err) if all_err else np.array([])
    mean = np.concatenate(all_mean) if all_mean else np.array([])
    print(f"\nRecordings: {len(files)}, results in {a.output_dir}")
    if len(err):
        s = error_stats(err)
        print(f"All recordings vs ECG ({len(err):,} segments): "
              f"MAE {s['MAE']:.2f} bpm, RMSE {s['RMSE']:.2f} bpm, "
              f"bias {s['bias']:+.2f} bpm, 95 % LoA "
              f"{s['LoA_low']:+.1f} to {s['LoA_high']:+.1f} bpm, "
              f"within +-5 bpm {s['within_5bpm_%']:.1f} %")
        bland_altman(err, mean, a.output_dir / "bland_altman.png")


if __name__ == "__main__":
    main()
