"""
TRAIN-ONLY evaluation of the Elgendi PPG detector against the ECG
reference HR, with breakdowns for tuning the detector.

Uses the detector of databasing/elgendi (unchanged) and the same segments
and validity rules as the HR database. Per-file PPG results are stored in
data/results/elgendi_train/<parameter tag>/parts/, so a run can be
interrupted and resumed, and runs with different detector parameters do
not mix. Test files are never opened.

Run from the repository root:
    uv run python databasing/elgendi_analysis/evaluate_elgendi.py
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from databasing.elgendi.detection import ElgendiConfig, detect_ppg_peaks
from databasing.pan_tompkins.heart_rate import (
    HR_RANGE_BPM,
    compute_hr_segments,
)
from utils.read_edf import read_edf_signals

DEFAULT_DB = root_dir / "data" / "hr_database_seg10s_step5s.parquet"
DEFAULT_EDF = root_dir / "data" / "harmonized_ecg_ppg_database_filtered"
DEFAULT_MANIFEST = (
    root_dir / "databasing" / "references" / "split_manifest.csv"
)
DEFAULT_OUTPUT = root_dir / "data" / "results" / "elgendi_train"

QUALITY_FLAGS = [
    f"{signal}_{name}"
    for signal in ("ecg", "ppg")
    for name in (
        "mean_absolute_derivative_spike",
        "LF_ratio_motion",
        "InBand_power_ratio_out",
        "kurtosis_out",
        "Skewness_out",
    )
]
DB_COLUMNS = [
    "base_folder",
    "edf_filename",
    "section_id",
    "start",
    "end",
    "segment_s",
    "hr_bpm",
    "hr_valid",
    "rr_cv",
    "ecg_is_flat",
    "ppg_is_flat",
    *QUALITY_FLAGS,
]
HR_BINS = [0, 60, 80, 100, 120, 140, 1000]
RATIO_BINS = [0, 0.6, 0.9, 1.1, 1.6, 100]
RATIO_LABELS = [
    "<0.6 (missed beats)",
    "0.6-0.9",
    "0.9-1.1 (ok)",
    "1.1-1.6",
    ">1.6 (double detections)",
]


# ---------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------


def config_tag(config: ElgendiConfig) -> str:
    """Folder name that identifies the detector parameters."""
    return (
        f"pw{config.peakwindow_s:g}_bw{config.beatwindow_s:g}"
        f"_bo{config.beatoffset:g}_md{config.mindelay_s:g}"
    )


def read_train_files(manifest_path: Path) -> set[str]:
    """Returns the "<subset>/<file>" names of the TRAIN files."""
    with open(manifest_path, newline="") as f:
        return {
            row["file"]
            for row in csv.DictReader(f)
            if row["set"] == "train"
        }


def ppg_segments_for_file(
    edf_path: Path,
    starts: np.ndarray,
    segment_s: float,
    step_s: float,
    config: ElgendiConfig,
) -> pd.DataFrame:
    """
    Detects PPG peaks in one recording and computes the PPG HR for every
    segment, with the same rules as the ECG reference.

    Args:
        edf_path (Path): Filtered EDF file.
        starts (np.ndarray): Segment starts (samples) from the database,
            used to check that the segments match.
        segment_s (float): Segment length in seconds.
        step_s (float): Step between segments in seconds.
        config (ElgendiConfig): Detector parameters.

    Returns:
        pd.DataFrame: section_id, ppg_n_beats, ppg_hr_bpm (valid only),
            ppg_hr_raw (60 / mean RR, also for invalid segments),
            ppg_rr_cv and ppg_reason ("" when valid).
    """
    _, ppg, metadata = read_edf_signals(str(edf_path))
    if ppg is None:
        raise ValueError(f"No PPG in {edf_path}")
    fs = float(metadata["sampling_rate"])
    peaks = detect_ppg_peaks(ppg, fs, config)
    hr = compute_hr_segments(peaks, len(ppg), fs, segment_s, step_s)
    if len(hr) != len(starts) or not np.array_equal(
        hr["start"], starts
    ):
        raise ValueError(
            f"Segments do not match the database: {edf_path}"
        )
    return pd.DataFrame(
        {
            "section_id": np.arange(len(hr)),
            "ppg_n_beats": hr["n_beats"].to_numpy(),
            "ppg_hr_bpm": hr["hr_bpm"].to_numpy(),
            "ppg_hr_raw": 60.0 / hr["rr_mean_s"].to_numpy(),
            "ppg_rr_cv": hr["rr_cv"].to_numpy(),
            "ppg_reason": hr["invalid_reason"].to_numpy(),
        }
    )


def process_and_store(
    edf_path: Path,
    part_path: Path,
    starts: np.ndarray,
    segment_s: float,
    step_s: float,
    config: ElgendiConfig,
) -> int:
    """Worker task: computes one file and stores it atomically."""
    result = ppg_segments_for_file(
        edf_path, starts, segment_s, step_s, config
    )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = part_path.with_name(part_path.name + ".tmp")
    result.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, part_path)
    return len(result)


def run_files(tasks: list[dict], workers: int):
    """Runs the per-file tasks in a process pool and prints progress."""
    if not tasks:
        return
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_and_store, **task): task["edf_path"]
            for task in tasks
        }
        for i, future in enumerate(as_completed(futures), start=1):
            name = futures[future].name
            try:
                n = future.result()
                print(
                    f"  [{i}/{len(tasks)}] {name}: {n} segments,"
                    f" {time.perf_counter() - t_start:.0f} s",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(tasks)}] {name}: ERROR {e}")


def load_windows(db_path: Path, parts_dir: Path, files: list[str]):
    """
    Joins the database rows with a valid ECG reference (TRAIN files only)
    with the stored PPG results.

    Returns:
        pd.DataFrame: One row per window with a valid ECG HR, with the
            columns evaluated (PPG valid too), error_bpm and ratio.
    """
    db = pd.read_parquet(
        db_path,
        columns=DB_COLUMNS,
        filters=[("split", "==", "train"), ("hr_valid", "==", True)],
    )
    db["file"] = db["base_folder"] + "/" + db["edf_filename"]
    db = db[db["file"].isin(files)]

    parts = []
    for rel in files:
        part_path = parts_dir / rel.replace(".edf", ".parquet")
        if part_path.exists():
            part = pd.read_parquet(part_path)
            part["file"] = rel
            parts.append(part)
    ppg = pd.concat(parts, ignore_index=True)
    windows = db.merge(ppg, on=["file", "section_id"], how="inner")

    windows["ppg_valid"] = (windows["ppg_reason"] == "") & ~windows[
        "ppg_is_flat"
    ]
    windows["failure"] = np.where(
        windows["ppg_is_flat"], "flat_ppg", windows["ppg_reason"]
    )
    windows["error_bpm"] = np.where(
        windows["ppg_valid"],
        windows["ppg_hr_bpm"] - windows["hr_bpm"],
        np.nan,
    )
    windows["ratio"] = windows["ppg_hr_raw"] / windows["hr_bpm"]
    windows["subject"] = windows["edf_filename"].str.split("_").str[0]
    windows["group"] = np.select(
        [
            windows["base_folder"] == "multisite-ppg",
            windows["base_folder"].str.startswith("wrist-ppg-during"),
        ],
        [
            windows["edf_filename"].str.split("_").str[1],
            windows["edf_filename"]
            .str.replace(".edf", "", regex=False)
            .str.split("_", n=1)
            .str[1],
        ],
        default="wrist-finger",
    )
    return windows


# ---------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------


def metrics(df: pd.DataFrame) -> pd.Series:
    """
    Accuracy and coverage of a group of windows with a valid ECG HR.
    Coverage = share of these windows where the PPG HR is valid too.
    """
    ok = df[df["ppg_valid"]]
    err = ok["error_bpm"].to_numpy()
    out = {
        "windows_ecg_valid": len(df),
        "windows_evaluated": len(ok),
        "coverage_%": 100.0 * len(ok) / len(df) if len(df) else np.nan,
        "MAE": np.mean(np.abs(err)) if len(err) else np.nan,
        "median_AE": np.median(np.abs(err)) if len(err) else np.nan,
        "RMSE": np.sqrt(np.mean(err**2)) if len(err) else np.nan,
        "bias": np.mean(err) if len(err) else np.nan,
        "within_5bpm_%": 100.0 * np.mean(np.abs(err) < 5)
        if len(err)
        else np.nan,
    }
    for reason in ["flat_ppg", "too_few_beats", "irregular_rr"]:
        out[f"lost_{reason}_%"] = 100.0 * np.mean(
            df["failure"] == reason
        )
    return pd.Series(out)


def by(windows: pd.DataFrame, columns) -> pd.DataFrame:
    """Metrics per group, rounded."""
    return (
        windows.groupby(columns, observed=True)
        .apply(metrics, include_groups=False)
        .round(2)
    )


def flag_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Metrics for windows with each quality flag True vs False."""
    rows = []
    for flag in QUALITY_FLAGS:
        for value in (True, False):
            part = windows[windows[flag] == value]
            if len(part):
                rows.append(
                    metrics(part).rename(f"{flag}={value}").to_frame().T
                )
    return pd.concat(rows).round(2)


def rr_cv_sweep(windows: pd.DataFrame) -> pd.DataFrame:
    """
    What the PPG would give with another rr_cv threshold (other rules
    unchanged): coverage and error of the raw PPG HR.
    """
    base = windows[
        ~windows["ppg_is_flat"]
        & (windows["ppg_n_beats"] >= 3)
        & windows["ppg_hr_raw"].between(*HR_RANGE_BPM)
    ]
    rows = []
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        ok = base[base["ppg_rr_cv"] <= threshold]
        err = ok["ppg_hr_raw"] - ok["hr_bpm"]
        rows.append(
            {
                "ppg_rr_cv_max": threshold,
                "coverage_%": 100.0 * len(ok) / len(windows),
                "MAE": np.abs(err).mean(),
                "within_5bpm_%": 100.0 * (np.abs(err) < 5).mean(),
            }
        )
    return pd.DataFrame(rows).round(2)


def ratio_table(windows: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of PPG HR / ECG HR (raw PPG HR, all windows with at
    least 3 PPG beats, not flat): shows missed vs doubled beats.
    """
    base = windows[
        ~windows["ppg_is_flat"] & (windows["ppg_n_beats"] >= 3)
    ]
    bins = pd.cut(base["ratio"], RATIO_BINS, labels=RATIO_LABELS)
    table = (
        pd.crosstab(base["base_folder"], bins, normalize="index") * 100
    )
    return table.round(1)


def subject_table(windows: pd.DataFrame) -> pd.DataFrame:
    """Metrics per subject, worst MAE first."""
    table = by(windows, ["base_folder", "subject"])
    return table.sort_values("MAE", ascending=False)


def build_report(windows: pd.DataFrame, skipped: list[str]) -> dict:
    """All breakdown tables, keyed by name."""
    windows = windows.copy()
    windows["ecg_hr_bin"] = pd.cut(windows["hr_bpm"], HR_BINS)
    windows["ecg_rr_cv_bin"] = pd.cut(
        windows["rr_cv"], [0, 0.03, 0.06, 0.10, 0.15]
    )
    overall = metrics(windows).round(2).to_frame("ALL").T
    subject_level = subject_table(windows)
    return {
        "overall": overall,
        "by_subset": by(windows, "base_folder"),
        "by_device_or_activity": by(windows, ["base_folder", "group"]),
        "by_subject": subject_level,
        "by_ecg_hr": by(windows, ["base_folder", "ecg_hr_bin"]),
        "by_ecg_rr_cv": by(windows, "ecg_rr_cv_bin"),
        "by_flag": flag_table(windows),
        "ppg_rr_cv_threshold_sweep": rr_cv_sweep(windows),
        "hr_ratio_distribution_%": ratio_table(windows),
        "skipped_files": pd.DataFrame({"file": skipped}),
    }


def write_report(tables: dict, output_dir: Path, header: str):
    """Saves every table as CSV and all of them in one markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [header, ""]
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv")
        lines += [f"## {name}", "", "```", table.to_string(), "```", ""]
    (output_dir / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--edf-root", type=Path, default=DEFAULT_EDF)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--large-file-gb",
        type=float,
        default=0.1,
        help="Files at least this large run at the end, one at a time.",
    )
    parser.add_argument(
        "--max-file-gb",
        type=float,
        default=1.0,
        help="Files at least this large are skipped (memory).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only the first N TRAIN files (sorted), for a quick test.",
    )
    parser.add_argument("--peakwindow", type=float, default=0.111)
    parser.add_argument("--beatwindow", type=float, default=0.667)
    parser.add_argument("--beatoffset", type=float, default=0.02)
    parser.add_argument("--mindelay", type=float, default=0.3)
    args = parser.parse_args()

    config = ElgendiConfig(
        args.peakwindow, args.beatwindow, args.beatoffset, args.mindelay
    )
    run_dir = args.output / config_tag(config)
    parts_dir = run_dir / "parts"

    # TRAIN files present in both the manifest and the database
    train_files = read_train_files(args.manifest)
    layout = pd.read_parquet(
        args.database,
        columns=[
            "base_folder",
            "edf_filename",
            "start",
            "start_t",
            "segment_s",
        ],
        filters=[("split", "==", "train")],
    )
    layout["file"] = (
        layout["base_folder"] + "/" + layout["edf_filename"]
    )
    layout = layout[layout["file"].isin(train_files)]
    starts_of = {
        f: g["start"].to_numpy() for f, g in layout.groupby("file")
    }
    segment_s = float(layout["segment_s"].iloc[0])
    # Step between segments, from the segment start times of one file
    start_times = layout.groupby("file")["start_t"].apply(np.diff)
    step_s = float(next(d[0] for d in start_times if len(d)))
    files = sorted(starts_of)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    def size_gb(rel):
        return (args.edf_root / rel).stat().st_size / 1e9

    skipped = [f for f in files if size_gb(f) >= args.max_file_gb]
    files = [f for f in files if f not in skipped]
    todo = [
        f
        for f in files
        if not (parts_dir / f.replace(".edf", ".parquet")).exists()
    ]
    print(
        f"{len(files)} TRAIN files, {len(files) - len(todo)} already done,"
        f" {len(skipped)} skipped (>= {args.max_file_gb:g} GB)."
        f" Parameters: {config_tag(config)}"
    )

    def task(rel):
        return {
            "edf_path": args.edf_root / rel,
            "part_path": parts_dir / rel.replace(".edf", ".parquet"),
            "starts": starts_of[rel],
            "segment_s": segment_s,
            "step_s": step_s,
            "config": config,
        }

    large = [f for f in todo if size_gb(f) >= args.large_file_gb]
    normal = [f for f in todo if f not in large]
    try:
        run_files([task(f) for f in normal], args.workers)
        run_files([task(f) for f in large], 1)
    except KeyboardInterrupt:
        sys.exit(
            "\nInterrupted. Run the same command again to continue."
        )

    windows = load_windows(args.database, parts_dir, files)
    tables = build_report(windows, skipped)
    windows.to_parquet(run_dir / "windows.parquet", index=False)
    header = (
        "# Elgendi PPG vs ECG reference, TRAIN ONLY\n\n"
        f"Parameters: `{config_tag(config)}`, files: {len(files)}"
        f" (skipped: {len(skipped)}). Windows overlap by 50 %, so they"
        " are not independent; per-subject numbers are more robust."
    )
    write_report(tables, run_dir, header)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for name in ["overall", "by_subset", "by_device_or_activity"]:
        print(f"\n{name}:\n{tables[name].to_string()}")
    print(f"\nFull report: {run_dir / 'report.md'}")
