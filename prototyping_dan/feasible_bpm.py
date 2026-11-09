"""
Adds the column 'feasible_bpm' to the per-file parquet database.

feasible_bpm = hr_bpm if the segment lies inside every border in
FEASIBILITY_BOUNDS, otherwise NaN. The borders are the "Borders combined"
rules of data/fold_analysis/combined_report/report.md (also in
metrics_borders_combined.csv), restricted to the ECG/PPG signal metrics.

All borders are strict (lower < value < upper), as in the report. A NaN
metric value fails the check.

The bounds and the check are also used at prediction time
(predict_hr.py), where only the PPG metrics are available.

Usage:
    python prototyping_dan/feasible_bpm.py [--parquet_dir data/hr_parts/seg10s_step5s]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# metric column -> (lower, upper); None = no border on that side
FEASIBILITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "ecg_mean_absolute_derivative": (0.04376, 0.3518),
    "ecg_kurtosis": (11.63, 31.18),
    "ecg_InBand_power_ratio": (0.3653, 0.663),
    "ecg_dynamic_range": (6.788, 10.15),
    "ppg_std_piecewise_min": (0.1462, 0.9931),
    "ppg_mean_absolute_derivative": (0.04394, 0.1988),
    "ppg_dynamic_range": (3.161, 8.376),
    "ppg_zero_dev_dur": (None, 0.075),
    "ppg_kurtosis": (1.453, 10.02),
    "ppg_Skewness": (-0.9212, 1.581),
    "ppg_InBand_power_ratio": (0.6898, 1.386),
    "ecg_std_piecewise_min": (0.8201, 0.9973),
}
PPG_BOUNDS = {
    k: v for k, v in FEASIBILITY_BOUNDS.items() if k.startswith("ppg_")
}


def within_bounds(
    df: pd.DataFrame, bounds: dict = FEASIBILITY_BOUNDS
) -> pd.Series:
    """Boolean mask of the rows inside all borders (NaN -> False)."""
    mask = pd.Series(True, index=df.index)
    for col, (lower, upper) in bounds.items():
        values = df[col]
        if lower is not None:
            mask &= values > lower
        if upper is not None:
            mask &= values < upper
    return mask


def failed_bounds(metrics: dict, bounds: dict = PPG_BOUNDS) -> list[str]:
    """Names of the metrics of one segment that are outside their borders."""
    failed = []
    for col, (lower, upper) in bounds.items():
        value = metrics[col]
        if (
            not np.isfinite(value)
            or (lower is not None and not value > lower)
            or (upper is not None and not value < upper)
        ):
            failed.append(col)
    return failed


def add_feasible_bpm(path: Path) -> tuple[int, int, int]:
    """
    Adds/overwrites 'feasible_bpm' in one parquet file (in place).

    Returns:
        (feasible segments, segments with hr_bpm, all segments)
    """
    df = pd.read_parquet(path)
    df["feasible_bpm"] = df["hr_bpm"].where(within_bounds(df), np.nan)
    df.to_parquet(path, index=False)
    return (
        int(df["feasible_bpm"].notna().sum()),
        int(df["hr_bpm"].notna().sum()),
        len(df),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add the feasible_bpm column to the parquet database"
    )
    parser.add_argument(
        "--parquet_dir",
        default=str(root_dir / "data" / "hr_parts" / "seg10s_step5s"),
    )
    args = parser.parse_args()

    files = sorted(Path(args.parquet_dir).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.parquet_dir}")

    n_feasible = n_valid = n_total = 0
    for path in tqdm(files, desc="Adding feasible_bpm", unit="file"):
        feasible, valid, total = add_feasible_bpm(path)
        n_feasible += feasible
        n_valid += valid
        n_total += total
    print(
        f"{len(files)} files, {n_total} segments, {n_valid} with hr_bpm,"
        f" {n_feasible} feasible ({n_feasible / max(n_valid, 1):.1%} of"
        " segments with hr_bpm)"
    )


if __name__ == "__main__":
    main()
