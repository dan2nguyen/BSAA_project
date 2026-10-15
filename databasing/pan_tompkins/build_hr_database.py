import argparse
import csv
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from databasing.evaluation_signal_segments import compute_metrics
from databasing.pan_tompkins.detection import detect_qrs
from databasing.pan_tompkins.heart_rate import (
    compute_hr_segments,
    mark_flat_segments,
)
from utils.read_edf import read_edf_signals

DEFAULT_INPUT = (
    root_dir / "data" / "harmonized_ecg_ppg_database_filtered"
)
DEFAULT_PARTS_ROOT = root_dir / "data" / "hr_parts"
DEFAULT_MANIFEST = (
    root_dir / "databasing" / "references" / "split_manifest.csv"
)

# Boolean quality flags of compute_metrics, stored for ECG and PPG
# (same names as the shortened output of evaluation_signal_segments.py)
FLAG_NAMES = [
    "is_flat",
    "mean_absolute_derivative_spike",
    "LF_ratio_motion",
    "InBand_power_ratio_out",
    "kurtosis_out",
    "Skewness_out",
]
FLAG_COLUMNS = [
    f"{signal}_{name}"
    for signal in ("ecg", "ppg")
    for name in FLAG_NAMES
]
COLUMNS = [
    "base_folder",
    "edf_filename",
    "section_id",
    "split",
    "start",
    "end",
    "start_t",
    "end_t",
    "segment_s",
    "sampling_rate",
    "is_flat",
    *FLAG_COLUMNS,
    "hr_bpm",
    "hr_valid",
    "invalid_reason",
    "n_beats",
    "hr_median_bpm",
    "rr_mean_s",
    "rr_median_s",
    "rr_std_s",
    "rr_cv",
]

# Summary categories built from the invalid_reason values
CATEGORIES = [
    "valid",
    "flat_ecg",
    "irregular_rr",
    "too_few_beats",
    "hr_out_of_range",
    "other",
]


def find_edf_files(input_dir: Path) -> list[Path]:
    """
    Lists the EDF files of the database, sorted for reproducibility.

    Args:
        input_dir (Path): Database folder with one subfolder per subset.

    Returns:
        list: Paths to all .edf files in the subfolders.
    """
    return sorted(input_dir.glob("*/*.edf"))


def read_split_manifest(manifest_path: Path) -> dict[str, str]:
    """
    Reads the train/test assignment of the files.

    Args:
        manifest_path (Path): split_manifest.csv with the columns
            database, subject, file, set.

    Returns:
        dict: "<subset>/<file name>" -> "train" or "test".
    """
    with open(manifest_path, newline="") as f:
        return {row["file"]: row["set"] for row in csv.DictReader(f)}


def part_path_for(edf_path: Path, parts_dir: Path) -> Path:
    """Returns the path of the per-file result (part) of an EDF file."""
    return parts_dir / edf_path.parent.name / f"{edf_path.stem}.parquet"


def compute_segment_flags(
    signal: np.ndarray,
    fs: float,
    starts: np.ndarray,
    ends: np.ndarray,
    signal_type: str,
) -> dict[str, np.ndarray]:
    """
    Evaluates the quality flags of `compute_metrics` for every segment.

    Args:
        signal (np.ndarray): 1D signal (whole recording).
        fs (float): Sampling rate in Hz.
        starts (np.ndarray): Segment start indices in samples.
        ends (np.ndarray): Segment end indices in samples.
        signal_type (str): "ecg" or "ppg".

    Returns:
        dict: "<signal_type>_<flag name>" -> boolean array per segment.
    """
    flags = {name: [] for name in FLAG_NAMES}
    with warnings.catch_warnings():
        # Kurtosis/skewness of a flat segment: harmless, it is flagged
        warnings.filterwarnings(
            "ignore", message="Precision loss occurred in moment"
        )
        for start, end in zip(starts, ends):
            metrics = compute_metrics(
                signal[start:end], fs, signal_type
            )
            for name in FLAG_NAMES:
                flags[name].append(bool(metrics[name]))
    return {
        f"{signal_type}_{name}": np.array(values, dtype=bool)
        for name, values in flags.items()
    }


def process_file(
    edf_path: Path, segment_s: float, step_s: float, split: str
) -> pd.DataFrame | None:
    """
    Computes the database rows of one EDF file: quality flags of ECG and
    PPG and the Pan-Tompkins HR for every segment.

    Segments are laid out like in evaluation_signal_segments.py, so
    section_id, start and end match its output for the same window and
    step. HR is NaN when the ECG is flat or the beats fail the checks in
    `compute_hr_segments`; flat PPG only sets `is_flat`.

    Args:
        edf_path (Path): Path to the EDF file.
        segment_s (float): Segment length in seconds.
        step_s (float): Step between segment starts in seconds.
        split (str): "train" or "test" from the split manifest.

    Returns:
        pd.DataFrame: One row per segment with the columns in COLUMNS, or
            None if the file could not be read or lacks ECG or PPG.
    """
    ecg, ppg, metadata = read_edf_signals(str(edf_path))
    if ecg is None or ppg is None:
        return None
    fs = metadata["sampling_rate"]

    peaks = detect_qrs(ecg, fs)
    hr = compute_hr_segments(peaks, len(ecg), fs, segment_s, step_s)
    starts = hr["start"].to_numpy()
    ends = hr["end"].to_numpy()
    flags = {
        **compute_segment_flags(ecg, fs, starts, ends, "ecg"),
        **compute_segment_flags(ppg, fs, starts, ends, "ppg"),
    }
    mark_flat_segments(hr, flags["ecg_is_flat"])

    return pd.DataFrame(
        {
            "base_folder": edf_path.parent.name,
            "edf_filename": edf_path.name,
            "section_id": np.arange(len(hr)),
            "split": split,
            "start": starts,
            "end": ends,
            "start_t": hr["start_s"].to_numpy(),
            "end_t": hr["end_s"].to_numpy(),
            "segment_s": segment_s,
            "sampling_rate": fs,
            "is_flat": flags["ecg_is_flat"] | flags["ppg_is_flat"],
            **flags,
            "hr_bpm": hr["hr_bpm"].to_numpy(),
            "hr_valid": hr["valid"].to_numpy(dtype=bool),
            "invalid_reason": hr["invalid_reason"].to_numpy(),
            "n_beats": hr["n_beats"].to_numpy(),
            "hr_median_bpm": hr["hr_median_bpm"].to_numpy(),
            "rr_mean_s": hr["rr_mean_s"].to_numpy(),
            "rr_median_s": hr["rr_median_s"].to_numpy(),
            "rr_std_s": hr["rr_std_s"].to_numpy(),
            "rr_cv": hr["rr_cv"].to_numpy(),
        },
        columns=COLUMNS,
    )


def write_parquet_atomic(df: pd.DataFrame, path: Path):
    """
    Writes a DataFrame to parquet through a temporary file, so an
    interrupted run never leaves a half-written file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp_path, index=False, compression="snappy")
    os.replace(tmp_path, path)


def process_and_store(
    edf_path: Path,
    part_path: Path,
    segment_s: float,
    step_s: float,
    split: str,
) -> tuple[int, int] | None:
    """
    Worker task: processes one file and stores its part, so the rows never
    travel between processes.

    Returns:
        tuple: (number of segments, number of segments with valid HR), or
            None if the file could not be processed.
    """
    rows = process_file(edf_path, segment_s, step_s, split)
    if rows is None:
        return None
    write_parquet_atomic(rows, part_path)
    return len(rows), int(rows["hr_valid"].sum())


def run_batch(
    files: list[Path],
    workers: int,
    parts_dir: Path,
    segment_s: float,
    step_s: float,
    split_of: dict[str, str],
    progress: dict,
) -> list[Path]:
    """
    Processes a list of files in a process pool and stores their parts.

    Args:
        files (list): EDF files to process.
        workers (int): Number of files processed in parallel.
        parts_dir (Path): Folder for the per-file results.
        segment_s (float): Segment length in seconds.
        step_s (float): Step between segment starts in seconds.
        split_of (dict): "<subset>/<file name>" -> "train" / "test".
        progress (dict): Shared counters: done (files finished so far),
            total (files to process in the run), bytes (size of the
            processed files), t_start (start time of the run).

    Returns:
        list: Files that could not be processed.
    """
    failed = []
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(
                process_and_store,
                edf_path,
                part_path_for(edf_path, parts_dir),
                segment_s,
                step_s,
                split_of.get(
                    f"{edf_path.parent.name}/{edf_path.name}", "unknown"
                ),
            ): edf_path
            for edf_path in files
        }
        for future in as_completed(futures):
            edf_path = futures[future]
            progress["done"] += 1
            prefix = f"[{progress['done']}/{progress['total']}]"
            name = f"{edf_path.parent.name}/{edf_path.name}"
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                result = None
                print(f"{prefix} {name}: ERROR {e}")
            if result is None:
                failed.append(edf_path)
                print(f"{prefix} {name}: FAILED")
                continue
            progress["bytes"] += edf_path.stat().st_size
            elapsed = time.perf_counter() - progress["t_start"]
            print(
                f"{prefix} {name}: {result[0]} segments,"
                f" {result[1]} with valid HR, {elapsed:.0f} s elapsed",
                flush=True,
            )
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return failed


def categorize(reason: str) -> str:
    """Maps an invalid_reason value to a summary category."""
    if reason == "":
        return "valid"
    if reason in CATEGORIES:
        return reason
    return "other"


def _column_array(table, name: str) -> np.ndarray:
    """Returns a parquet table column as a numpy array."""
    return table.column(name).to_numpy(zero_copy_only=False)


def assemble_database(
    part_paths: list[Path], output_path: Path | None
) -> dict:
    """
    Joins the per-file parts into one parquet file and collects the
    summary statistics. Parts are streamed one by one, the whole database
    is never held in memory.

    Args:
        part_paths (list): Existing part files to join.
        output_path (Path, optional): Target parquet file. If None, only
            the statistics are computed.

    Returns:
        dict: Counters per subset of segments, HR results (reasons), True
            values of is_flat and the flag columns (flags), usable
            segments (not flat and valid HR), files and files without any
            usable segment; total number of rows (rows).
    """
    stats = {
        "segments": Counter(),
        "reasons": defaultdict(Counter),
        "flags": defaultdict(Counter),
        "usable": Counter(),
        "files": Counter(),
        "files_without_usable": Counter(),
        "rows": 0,
    }
    writer = None
    tmp_path = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(output_path.name + ".tmp")

    try:
        for part_path in part_paths:
            table = pq.read_table(part_path)
            subset = part_path.parent.name
            stats["files"][subset] += 1
            stats["segments"][subset] += table.num_rows
            stats["rows"] += table.num_rows

            reasons = table.column("invalid_reason").to_pylist()
            stats["reasons"][subset].update(
                categorize(r) for r in reasons
            )
            for column in ["is_flat", *FLAG_COLUMNS]:
                stats["flags"][subset][column] += int(
                    _column_array(table, column).sum()
                )
            usable = int(
                (
                    ~_column_array(table, "is_flat")
                    & _column_array(table, "hr_valid")
                ).sum()
            )
            stats["usable"][subset] += usable
            if usable == 0:
                stats["files_without_usable"][subset] += 1

            # Files shorter than one segment have no rows to store
            if tmp_path is None or table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path, table.schema, compression="snappy"
                )
            writer.write_table(table.cast(writer.schema))
    finally:
        if writer is not None:
            writer.close()

    if writer is not None:
        os.replace(tmp_path, output_path)
    return stats


def _percent(count: int, total: int) -> float:
    """Share in percent, NaN for an empty total."""
    return round(100.0 * count / total, 1) if total else float("nan")


def print_summary(stats: dict):
    """
    Prints per subset: usable segments, HR results, the share of True
    values of every flag and the files without any usable segment.
    """
    subsets = sorted(stats["files"])
    counts = pd.DataFrame(
        {
            "files": [stats["files"][s] for s in subsets],
            "segments": [stats["segments"][s] for s in subsets],
            "is_flat": [stats["flags"][s]["is_flat"] for s in subsets],
            "hr_valid": [stats["reasons"][s]["valid"] for s in subsets],
            "usable": [stats["usable"][s] for s in subsets],
        },
        index=subsets,
    )
    counts.loc["ALL"] = counts.sum()
    overview = counts[["files", "segments"]].copy()
    for column in ["is_flat", "hr_valid", "usable"]:
        overview[f"{column}_%"] = [
            _percent(c, t)
            for c, t in zip(counts[column], counts["segments"])
        ]
    print("\nOverview (usable = not is_flat and valid HR):")
    print(overview.to_string())

    reasons = pd.DataFrame(
        {
            s: [stats["reasons"][s][c] for c in CATEGORIES]
            for s in subsets
        },
        index=CATEGORIES,
    ).T
    print("\nHR result per segment (count):")
    print(reasons.to_string())

    flags = pd.DataFrame(
        {
            s: [
                _percent(stats["flags"][s][c], stats["segments"][s])
                for c in FLAG_COLUMNS
            ]
            for s in subsets
        },
        index=FLAG_COLUMNS,
    )
    print("\nFlags True (% of segments):")
    print(flags.to_string())

    print("\nFiles without any usable segment:")
    for subset in subsets:
        print(
            f"  {subset}: {stats['files_without_usable'][subset]}"
            f" of {stats['files'][subset]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the segment database: quality flags of ECG and PPG and"
            " the Pan-Tompkins HR for every segment of all EDF files."
            " Results are stored per file, so an interrupted run continues"
            " where it stopped."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument(
        "--step-s",
        type=float,
        default=None,
        help="Step between segments, default half of --segment-s"
        " (50%% overlap).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final parquet, default"
        " data/hr_database_seg<segment>s_step<step>s.parquet.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Folder for per-file results, default data/hr_parts/<params>/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Files processed in parallel (each holds one file in memory).",
    )
    parser.add_argument(
        "--large-file-gb",
        type=float,
        default=0.5,
        help="Files at least this large are processed at the end, one at"
        " a time, so that only one of them is in memory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Test run on a random sample of N files (no final file).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute files that already have a stored result.",
    )
    args = parser.parse_args()

    step_s = (
        args.step_s if args.step_s is not None else args.segment_s / 2
    )
    tag = f"seg{args.segment_s:g}s_step{step_s:g}s"
    output_path = (
        args.output or root_dir / "data" / f"hr_database_{tag}.parquet"
    )
    parts_dir = args.parts_dir or DEFAULT_PARTS_ROOT / tag

    split_of = read_split_manifest(args.manifest)
    all_files = find_edf_files(args.input)
    if not all_files:
        sys.exit(f"No EDF files found in {args.input}")

    present = {f"{f.parent.name}/{f.name}" for f in all_files}
    missing = sorted(set(split_of) - present)
    if missing:
        print(
            f"Warning: {len(missing)} files of the split manifest are not"
            f" in {args.input}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    files = all_files
    if args.limit is not None:
        files = sorted(
            random.Random(args.seed).sample(
                all_files, min(args.limit, len(all_files))
            )
        )

    todo = [
        f
        for f in files
        if args.overwrite or not part_path_for(f, parts_dir).exists()
    ]
    large = [
        f for f in todo if f.stat().st_size >= args.large_file_gb * 1e9
    ]
    normal = [f for f in todo if f not in large]
    print(
        f"{len(files)} files selected, {len(files) - len(todo)} already done,"
        f" {len(todo)} to process with {args.workers} workers."
        f" Parts: {parts_dir}"
    )
    if large:
        print(
            f"{len(large)} files of at least {args.large_file_gb:g} GB are"
            " processed at the end, one at a time."
        )

    progress = {
        "done": 0,
        "total": len(todo),
        "bytes": 0,
        "t_start": time.perf_counter(),
    }
    failed = []
    try:
        for batch, workers in ((normal, args.workers), (large, 1)):
            if batch:
                failed += run_batch(
                    batch,
                    workers,
                    parts_dir,
                    args.segment_s,
                    step_s,
                    split_of,
                    progress,
                )
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Finished files are kept in the parts folder,"
            " run the same command again to continue."
        )
        sys.exit(130)

    elapsed = time.perf_counter() - progress["t_start"]
    done_bytes = progress["bytes"]
    print(
        f"\nProcessed {len(todo) - len(failed)} files in {elapsed:.1f} s."
    )
    if done_bytes and args.limit is not None:
        # Processing time scales with the recording length, i.e. file size
        total_bytes = sum(f.stat().st_size for f in all_files)
        estimate = elapsed * total_bytes / done_bytes
        print(
            f"Sample is {100.0 * done_bytes / total_bytes:.1f}% of the"
            f" database by size -> estimated full run: {estimate / 60:.1f} min"
            f" ({len(all_files)} files, {args.workers} workers)."
        )
    if failed:
        print(
            f"{len(failed)} files failed (they are retried on the next run):"
        )
        for f in failed:
            print(f"  {f}")

    part_paths = [
        part_path_for(f, parts_dir)
        for f in files
        if part_path_for(f, parts_dir).exists()
    ]
    if args.limit is not None:
        stats = assemble_database(part_paths, None)
        print(
            "Test run (--limit): the final database file was not written."
        )
    else:
        stats = assemble_database(part_paths, output_path)
        print(f"Saved {stats['rows']} segments to {output_path}")
    print_summary(stats)
