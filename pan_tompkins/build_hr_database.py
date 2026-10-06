import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from pan_tompkins.heart_rate import estimate_hr
from utils.read_edf import read_edf_signals

DEFAULT_INPUT = root_dir / "data" / "harmonized_ecg_ppg_database_filtered"
DEFAULT_PARTS_ROOT = root_dir / "data" / "hr_parts"

# Summary categories built from the invalid_reason values
CATEGORIES = [
    "valid",
    "flat",
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


def part_path_for(edf_path: Path, parts_dir: Path) -> Path:
    """Returns the path of the per-file result (part) of an EDF file."""
    return parts_dir / edf_path.parent.name / f"{edf_path.stem}.parquet"


def process_file(
    edf_path: Path, segment_s: float, step_s: float | None
) -> pd.DataFrame | None:
    """
    Computes the segment-wise HR table of one EDF file.

    Args:
        edf_path (Path): Path to the EDF file.
        segment_s (float): Segment length in seconds.
        step_s (float, optional): Step between segment starts in seconds.

    Returns:
        pd.DataFrame: One row per segment (columns of `estimate_hr` plus
            subset, edf_filename and sampling_rate), or None if the file
            could not be read or has no ECG channel.
    """
    ecg, _, metadata = read_edf_signals(str(edf_path))
    if ecg is None:
        return None
    fs = metadata["sampling_rate"]

    hr, _ = estimate_hr(ecg, fs, segment_s, step_s)
    hr.insert(0, "subset", edf_path.parent.name)
    hr.insert(1, "edf_filename", edf_path.name)
    hr.insert(2, "sampling_rate", fs)
    return hr


def write_parquet_atomic(df: pd.DataFrame, path: Path):
    """
    Writes a DataFrame to parquet through a temporary file, so an
    interrupted run never leaves a half-written file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp_path, index=False, compression="snappy")
    os.replace(tmp_path, path)


def categorize(reason: str) -> str:
    """Maps an invalid_reason value to a summary category."""
    if reason == "":
        return "valid"
    if reason.startswith("flat"):
        return "flat"
    if reason in CATEGORIES:
        return reason
    return "other"


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
        dict: segments (Counter per subset of category counts),
            files (number of files per subset),
            files_without_valid (number of files per subset with zero
            valid segments), rows (total number of segments).
    """
    stats = {
        "segments": defaultdict(Counter),
        "files": Counter(),
        "files_without_valid": Counter(),
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

            reasons = table.column("invalid_reason").to_pylist()
            categories = Counter(categorize(r) for r in reasons)
            stats["segments"][subset].update(categories)
            stats["rows"] += table.num_rows
            if categories["valid"] == 0:
                stats["files_without_valid"][subset] += 1

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


def print_summary(stats: dict):
    """Prints the segment counts per subset and the files without valid HR."""
    table = pd.DataFrame(
        {
            subset: [counts[c] for c in CATEGORIES]
            for subset, counts in sorted(stats["segments"].items())
        },
        index=CATEGORIES,
    ).T
    table.insert(0, "files", [stats["files"][s] for s in table.index])
    table["total"] = table[CATEGORIES].sum(axis=1)
    table["valid_%"] = (
        100.0 * table["valid"] / table["total"].where(table["total"] > 0)
    ).round(1)
    table.loc["ALL"] = table.sum(numeric_only=True)
    if table.loc["ALL", "total"] > 0:
        table.loc["ALL", "valid_%"] = round(
            100.0 * table.loc["ALL", "valid"] / table.loc["ALL", "total"], 1
        )
    int_columns = ["files", *CATEGORIES, "total"]
    table[int_columns] = table[int_columns].astype(int)

    print("\nSegments per subset:")
    print(table.to_string())
    print("\nFiles with zero valid segments:")
    for subset in sorted(stats["files"]):
        print(
            f"  {subset}: {stats['files_without_valid'][subset]}"
            f" of {stats['files'][subset]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the HR ground-truth database (Pan-Tompkins) for all EDF"
            " files. Results are stored per file, so an interrupted run"
            " continues where it stopped."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument(
        "--step-s",
        type=float,
        default=None,
        help="Step between segments, defaults to --segment-s (no overlap).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final parquet, default data/hr_database_seg<segment>s.parquet.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Folder for per-file results, default data/hr_parts/<params>/.",
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

    step_s = args.step_s if args.step_s is not None else args.segment_s
    tag = f"seg{args.segment_s:g}s"
    if step_s != args.segment_s:
        tag += f"_step{step_s:g}s"
    output_path = args.output or root_dir / "data" / f"hr_database_{tag}.parquet"
    parts_dir = args.parts_dir or DEFAULT_PARTS_ROOT / tag

    all_files = find_edf_files(args.input)
    if not all_files:
        sys.exit(f"No EDF files found in {args.input}")
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
    print(
        f"{len(files)} files selected, {len(files) - len(todo)} already done,"
        f" {len(todo)} to process. Parts: {parts_dir}"
    )

    failed = []
    done_bytes = 0
    t_start = time.perf_counter()
    try:
        for i, edf_path in enumerate(todo, start=1):
            t_file = time.perf_counter()
            hr = process_file(edf_path, args.segment_s, step_s)
            if hr is None:
                failed.append(edf_path)
                print(f"[{i}/{len(todo)}] {edf_path.name}: FAILED to read")
                continue
            write_parquet_atomic(hr, part_path_for(edf_path, parts_dir))
            done_bytes += edf_path.stat().st_size
            print(
                f"[{i}/{len(todo)}] {edf_path.parent.name}/{edf_path.name}:"
                f" {len(hr)} segments, {int(hr['valid'].sum())} valid,"
                f" {time.perf_counter() - t_file:.2f} s"
            )
            del hr
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Finished files are kept in the parts folder,"
            " run the same command again to continue."
        )
        sys.exit(130)

    elapsed = time.perf_counter() - t_start
    print(f"\nProcessed {len(todo) - len(failed)} files in {elapsed:.1f} s.")
    if done_bytes and args.limit is not None:
        # Processing time scales with the recording length, i.e. file size
        total_bytes = sum(f.stat().st_size for f in all_files)
        estimate = elapsed * total_bytes / done_bytes
        print(
            f"Sample is {100.0 * done_bytes / total_bytes:.1f}% of the"
            f" database by size -> estimated full run: {estimate / 60:.1f} min"
            f" ({len(all_files)} files)."
        )
    if failed:
        print(f"{len(failed)} files failed (they are retried on the next run):")
        for f in failed:
            print(f"  {f}")

    part_paths = [
        part_path_for(f, parts_dir)
        for f in files
        if part_path_for(f, parts_dir).exists()
    ]
    if args.limit is not None:
        stats = assemble_database(part_paths, None)
        print("Test run (--limit): the final database file was not written.")
    else:
        stats = assemble_database(part_paths, output_path)
        print(f"Saved {stats['rows']} segments to {output_path}")
    print_summary(stats)
