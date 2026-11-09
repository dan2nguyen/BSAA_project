"""
Predicts HR from PPG with a trained model, but only on segments whose PPG
is usable according to the quality borders in feasible_bpm.py.

For every segment (default 10 s, step 5 s) the PPG metrics are computed
exactly like in databasing/pan_tompkins/build_hr_database.py:
  - is_flat from compute_metrics on the raw segment,
  - the numeric metrics from compute_metrics on the z-scored segment
    (std_piecewise_min = min of the 5 piecewise STDs).
A segment is usable if it is not flat and all PPG metrics lie inside
PPG_BOUNDS. Only usable segments are fed to the model; the others get
hr_pred = NaN and the failed metrics in 'reason'.

Usage:
    python prototyping_dan/predict_hr.py --input <file.edf | folder>
        --model-path data/training_outputs/fold1/best_model_10s_fold1.pt
        [--output predictions.csv]
"""

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from databasing.evaluation_signal_segments import compute_metrics
from databasing.pan_tompkins.build_hr_database import normalize_segment
from prototyping_dan.feasible_bpm import PPG_BOUNDS, failed_bounds
from prototyping_dan.predict_training_segments import (
    MODEL_REGISTRY,
    load_model,
    predict_batch,
)
from utils.read_edf import read_edf_signals

# Sampling rate of the training database (model input)
MODEL_FS = 100


def ppg_quality(segment: np.ndarray, fs: float) -> dict:
    """
    PPG metrics of one segment, named like the database columns
    ("ppg_<metric>"), plus "ppg_is_flat".
    """
    raw = compute_metrics(segment, fs, "ppg")
    norm = compute_metrics(normalize_segment(segment), fs, "ppg")
    norm["std_piecewise_min"] = min(norm["std_piecewise"])
    metrics = {
        col: float(norm[col.removeprefix("ppg_")]) for col in PPG_BOUNDS
    }
    metrics["ppg_is_flat"] = bool(raw["is_flat"])
    return metrics


def check_segments(
    ppg: np.ndarray, fs: float, segment_s: float, step_s: float
) -> pd.DataFrame:
    """
    Slices the PPG into segments and checks their usability.

    Returns:
        One row per segment: start, end (samples), start_t, end_t (s),
        the PPG metrics, 'usable' and 'reason' (why it is not usable).
    """
    win, step = int(round(segment_s * fs)), int(round(step_s * fs))
    rows = []
    for start in range(0, len(ppg) - win + 1, step):
        segment = ppg[start : start + win].astype(np.float64)
        row = {"start": start, "end": start + win,
               "start_t": start / fs, "end_t": (start + win) / fs}
        if not np.all(np.isfinite(segment)):
            row.update(usable=False, reason="nan_or_inf_in_signal")
            rows.append(row)
            continue
        metrics = ppg_quality(segment, fs)
        failed = failed_bounds(metrics, PPG_BOUNDS)
        if metrics["ppg_is_flat"]:
            failed.insert(0, "ppg_is_flat")
        row.update(metrics, usable=not failed, reason=";".join(failed))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_ppg(
    ppg: np.ndarray,
    fs: float,
    model: torch.nn.Module,
    device: torch.device,
    segment_s: float = 10.0,
    step_s: float = 5.0,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Checks every segment of a PPG recording and predicts HR on the usable
    ones (hr_pred = NaN otherwise). The PPG is resampled to MODEL_FS first.
    """
    if fs != MODEL_FS:
        g = gcd(int(round(fs)), MODEL_FS)
        ppg = scipy.signal.resample_poly(
            ppg, MODEL_FS // g, int(round(fs)) // g
        )
        fs = MODEL_FS

    result = check_segments(ppg, fs, segment_s, step_s)
    result["hr_pred"] = np.nan
    if result.empty:
        return result

    usable = result.index[result["usable"]].to_numpy()
    if len(usable):
        x = np.stack([
            ppg[s:e] for s, e in result.loc[usable, ["start", "end"]]
            .to_numpy()
        ]).astype(np.float32)
        preds = np.concatenate([
            predict_batch(model, x[i : i + batch_size], device)
            for i in range(0, len(x), batch_size)
        ])
        result.loc[usable, "hr_pred"] = preds
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict HR from PPG on segments that pass the PPG"
        " quality borders"
    )
    parser.add_argument(
        "--input", required=True,
        help="EDF file, or folder searched recursively for .edf files",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", *MODEL_REGISTRY.keys()],
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV with one row per segment (default: print summary only)",
    )
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, model_name = load_model(args.model_path, args.model, device)
    print(f"Model: {model_name} ({args.model_path})")

    input_path = Path(args.input)
    edf_files = (
        sorted(input_path.rglob("*.edf")) if input_path.is_dir()
        else [input_path]
    )

    results = []
    for edf_path in edf_files:
        _, ppg, metadata = read_edf_signals(str(edf_path))
        if ppg is None:
            print(f"{edf_path}: no PPG channel, skipped")
            continue
        res = predict_ppg(
            ppg, metadata["sampling_rate"], model, device,
            args.segment_s, args.step_s, args.batch_size,
        )
        res.insert(0, "file", str(edf_path))
        n_ok = int(res["usable"].sum()) if len(res) else 0
        hr = res["hr_pred"].median()
        print(f"{edf_path.name}: {n_ok}/{len(res)} usable segments,"
              f" median HR {hr:.1f} bpm" if n_ok else
              f"{edf_path.name}: 0/{len(res)} usable segments")
        results.append(res)

    if not results:
        print("No PPG recordings found.")
        return
    all_res = pd.concat(results, ignore_index=True)
    print(f"\nTotal: {int(all_res['usable'].sum())}/{len(all_res)} usable"
          f" segments in {len(results)} files")
    reasons = (all_res.loc[~all_res["usable"], "reason"]
               .str.split(";").explode().value_counts())
    if len(reasons):
        print("Failed metrics (segments can fail several):")
        for name, n in reasons.items():
            print(f"  {name}: {n}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        all_res.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
