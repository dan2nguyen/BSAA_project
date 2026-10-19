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
                # 1. Read signals
                ecg, ppg, metadata = read_edf_signals(src_path)
                if ecg is None or ppg is None:
                    continue

                fs = metadata.get("sampling_rate", 100.0)

                # 2. Filter (avoid extra copies in memory)
                f_ecg, f_ppg = apply_standard_filters(
                    ecg.astype(np.float32, copy=False),
                    ppg.astype(np.float32, copy=False),
                    fs=fs,
                )
                del ecg, ppg

                # 3. Preallocate array to avoid doubling RAM via vstack/np.array
                min_len = min(len(f_ecg), len(f_ppg))
                data = np.empty((2, min_len), dtype=np.float32)
                data[0] = f_ecg[:min_len]
                del f_ecg
                data[1] = f_ppg[:min_len]
                del f_ppg

                ch_names = list(
                    metadata.get("channel_mapping", {}).values()
                )
                if len(ch_names) < 2:
                    ch_names = ["ECG", "PPG"]

                # 4. Save to EDF
                try:
                    info = mne.create_info(
                        ch_names=ch_names,
                        sfreq=fs,
                        ch_types=["ecg", "misc"],
                    )
                    raw_filt = mne.io.RawArray(
                        data, info, verbose=False
                    )
                    raw_filt.export(dst_path, fmt="edf", overwrite=True)
                except Exception as e:
                    if "exceeds maximum field length" in str(e):
                        info = mne.create_info(
                            ch_names=ch_names,
                            sfreq=fs,
                            ch_types=["misc", "misc"],
                        )
                        raw_filt = mne.io.RawArray(
                            data, info, verbose=False
                        )
                        raw_filt.export(
                            dst_path, fmt="edf", overwrite=True
                        )
                    else:
                        raise e  # noqa: TRY201
                finally:
                    # Explicit cleanup for each file iteration
                    del data
                    if "raw_filt" in locals():
                        del raw_filt
                    gc.collect()

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
