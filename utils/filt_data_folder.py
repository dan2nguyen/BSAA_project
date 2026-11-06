"""
Filters a whole EDF database (apply_standard_filters from
exploratory_analysis/filters.py) and saves it as the filtered database
data/harmonized_ecg_ppg_database_filtered/ (EDF, resumable via
filtered_progress.csv). filter_edf() does one file and is reused by
predict.py, so prediction sees exactly the same signals.

Usage:
    python utils/filt_data_folder.py data/harmonized_ecg_ppg_databases
"""

import csv
import gc
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import mne
import numpy as np

# Suppress verbose MNE logs
mne.set_log_level("ERROR")

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from exploratory_analysis.filters import apply_standard_filters
from utils.read_edf import read_edf_signals


def filter_edf(src_path, dst_path, require_ecg=True) -> bool:
    """
    Filters one EDF file (apply_standard_filters) and saves it as EDF.

    This is the preprocessing of the filtered database. The EDF export
    also quantizes the signals to 16 bit, which the quality metrics see
    (e.g. zero_dev_dur), so predict.py reuses this function as is.

    Args:
        src_path: Input EDF file.
        dst_path: Output EDF file (overwritten).
        require_ecg: If False, a file without ECG is filtered and saved
            with the PPG channel only (used at prediction time).

    Returns:
        False if the file lacks the PPG (or the ECG when required).
    """
    # 1. Read signals
    ecg, ppg, metadata = read_edf_signals(src_path)
    if ppg is None or (ecg is None and require_ecg):
        return False

    fs = metadata.get("sampling_rate", 100.0)

    # 2. Filter (avoid extra copies in memory)
    f_ecg, f_ppg = apply_standard_filters(
        ecg.astype(np.float32, copy=False) if ecg is not None else None,
        ppg.astype(np.float32, copy=False),
        fs=fs,
    )
    del ecg, ppg
    signals = [f_ppg] if f_ecg is None else [f_ecg, f_ppg]
    ch_types = ["misc"] if f_ecg is None else ["ecg", "misc"]
    del f_ecg, f_ppg

    # 3. Preallocate array to avoid doubling RAM via vstack/np.array
    min_len = min(len(x) for x in signals)
    data = np.empty((len(signals), min_len), dtype=np.float32)
    for i in range(len(signals)):
        data[i] = signals[i][:min_len]
    del signals

    ch_names = [
        name
        for name in metadata.get("channel_mapping", {}).values()
        if name is not None
    ]
    if len(ch_names) < len(ch_types):
        ch_names = ["ECG", "PPG"][-len(ch_types):]

    # 4. Save to EDF
    try:
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types=ch_types,
        )
        raw_filt = mne.io.RawArray(data, info, verbose=False)
        raw_filt.export(dst_path, fmt="edf", overwrite=True)
    except Exception as e:
        if "exceeds maximum field length" in str(e):
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=fs,
                ch_types=["misc"] * len(ch_types),
            )
            raw_filt = mne.io.RawArray(data, info, verbose=False)
            raw_filt.export(dst_path, fmt="edf", overwrite=True)
        else:
            raise e  # noqa: TRY201
    finally:
        # Explicit cleanup for each file iteration
        del data
        if "raw_filt" in locals():
            del raw_filt
        gc.collect()
    return True


def main():
    # 1. Determine Source Database
    if len(sys.argv) > 1:
        source_folder = sys.argv[1]
        if not os.path.isdir(source_folder):
            print(f"Error: Path '{source_folder}' is not a directory.")
            sys.exit(1)
    else:
        root_tk = tk.Tk()
        root_tk.withdraw()
        source_folder = filedialog.askdirectory(
            title="Select Source Database Folder"
        )
        if not source_folder:
            print("No source folder selected. Exiting.")
            return

    # 2. Define Output Folder and Tracking
    output_root = "data/harmonized_ecg_ppg_database_filtered"
    progress_csv = os.path.join(output_root, "filtered_progress.csv")
    os.makedirs(output_root, exist_ok=True)

    copied_files = set()
    if os.path.exists(progress_csv):
        with open(progress_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    copied_files.add(row[0])
        print(f"Resuming: {len(copied_files)} files already processed.")

    # 3. Process Files
    count = 0
    for root, _, files in os.walk(source_folder):
        for filename in files:
            if not filename.endswith(".edf"):
                continue

            src_path = os.path.join(root, filename)
            rel_path = os.path.relpath(src_path, source_folder)

            if rel_path in copied_files:
                continue

            dst_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            try:
                if not filter_edf(src_path, dst_path):
                    continue

                # Update progress
                with open(progress_csv, "a", newline="") as f:
                    csv.writer(f).writerow([rel_path])

                count += 1
                print(f"Processed: {rel_path}")

            except Exception as e:  # noqa: BLE001
                print(f"Error {rel_path}: {e}")
                gc.collect()

    print(f"Done. Processed: {count}")


if __name__ == "__main__":
    main()
