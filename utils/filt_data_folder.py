import csv
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import mne
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from exploratory_analysis.filters import apply_standard_filters
from utils.read_edf import read_edf_signals


def main():
    # 1. Determine Source Database
    if len(sys.argv) > 1:
        source_folder = sys.argv[1]
        if not os.path.isdir(source_folder):
            print(
                f"Error: The provided path '{source_folder}' is not a valid directory."
            )
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

    if not os.path.exists(output_root):
        os.makedirs(output_root)

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
            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # --- Filtering Process ---
                # 1. Read the signals
                ecg, ppg, metadata = read_edf_signals(src_path)

                # 2. Apply standard filters
                fs = metadata.get("sampling_rate", 100.0)
                f_ecg, f_ppg = apply_standard_filters(ecg, ppg, fs=fs)

                # 3. Save filtered signals back to a new EDF file
                ch_names = list(
                    metadata.get("channel_mapping", {}).values()
                )
                if len(ch_names) < 2:
                    ch_names = ["ECG", "PPG"]

                data_ecg = f_ecg
                data_ppg = f_ppg  # PPG stays raw/misc

                data = np.vstack([data_ecg, data_ppg])

                # 2. Tell MNE that channel 0 is ECG and channel 1 is misc
                info = mne.create_info(
                    ch_names=ch_names,
                    sfreq=fs,
                    ch_types=["ecg", "misc"],
                )
                try:
                    raw_filt = mne.io.RawArray(data, info)
                    # 3. Export: MNE multiplies 'ecg' by 1e6 back into uV for the EDF header
                    raw_filt.export(dst_path, fmt="edf", overwrite=True)
                except Exception as e:
                    if "exceeds maximum field length" in str(e):
                        print(f"Field length error for {rel_path}, retrying with ch_types=['misc', 'misc']")
                        info = mne.create_info(
                            ch_names=ch_names,
                            sfreq=fs,
                            ch_types=["misc", "misc"],
                        )
                        raw_filt = mne.io.RawArray(data, info)
                        raw_filt.export(dst_path, fmt="edf", overwrite=True)
                    else:
                        raise e

                # Update progress CSV immediately after successful process
                with open(progress_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([rel_path])

                count += 1
                print(f"Processed and Filtered: {rel_path}")
            except Exception as e:  # noqa: BLE001
                print(f"Error processing {rel_path}: {e}")

    print(
        f"Successfully filtered and saved {count} new files to {output_root}"
    )


if __name__ == "__main__":
    main()
