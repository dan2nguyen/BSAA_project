import csv
import itertools
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.signal
import scipy.stats

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.read_edf import read_edf_signals


def compute_metrics(x, fs, signal_type):
    """Computes quality metrics for a 10s signal slice."""
    # 1. Standard Deviation
    std = float(np.std(x))

    # 2. Piecewise STD
    std_piecewise = [float(np.std(sub)) for sub in np.array_split(x, 5)]

    # 3. Zero-Derivative Duration
    diff_mask = np.abs(np.diff(x)) < 1e-6
    consec = [
        len(list(g)) for k, g in itertools.groupby(diff_mask) if k
    ]
    zero_dev_dur = float(max(consec) / fs) if consec else 0.0

    # 4. Dynamic Range
    dynamic_range = float(np.ptp(x))

    # 5. Mean Absolute Derivative
    mean_abs_der = float(np.mean(np.abs(np.diff(x))))

    # 6. Welch Spectral Ratios
    f, Pxx = scipy.signal.welch(x, fs=fs, nperseg=int(fs * 2))
    total_p = scipy.integrate.simpson(y=Pxx, x=f) + 1e-12

    lf_mask = (f >= 0.01) & (f <= 0.5)
    lf_p = (
        scipy.integrate.simpson(y=Pxx[lf_mask], x=f[lf_mask])
        if any(lf_mask)
        else 0.0
    )
    lf_ratio = float(lf_p / total_p)

    if signal_type == "ecg":
        band_mask = (f >= 5.0) & (f <= 20.0)
    else:  # ppg
        band_mask = (f >= 0.5) & (f <= 5.0)

    inband_p = (
        scipy.integrate.simpson(y=Pxx[band_mask], x=f[band_mask])
        if any(band_mask)
        else 0.0
    )
    inband_ratio = float(inband_p / total_p)

    kurt = float(scipy.stats.kurtosis(x, fisher=False))
    skew = float(scipy.stats.skew(x))

    std_flat = len([v for v in std_piecewise if v < 1e-4]) >= 3
    zero_dev_flat = zero_dev_dur > 4.0
    dynamic_range_flat = dynamic_range < 1.0
    mean_abs_der_spike = mean_abs_der > (2.0 * std)
    lf_ratio_motion = lf_ratio > 0.6

    if signal_type == "ecg":
        inband_out = inband_ratio < 0.40
        kurtosis_out = kurt < 5.0 or kurt > 30.0
        skewness_out = abs(skew) < 0.2
    else:  # ppg
        inband_out = inband_ratio < 0.50
        kurtosis_out = kurt < 2.0 or kurt > 10.0
        skewness_out = skew < 0.0

    return {
        "signal_type": signal_type,
        "std": std,
        "std_piecewise": std_piecewise,
        "zero_dev_dur": zero_dev_dur,
        "dynamic_range": dynamic_range,
        "mean_absolute_derivative": mean_abs_der,
        "LF_ratio": lf_ratio,
        "InBand_power_ratio": inband_ratio,
        "kurtosis": kurt,
        "Skewness": skew,
        "std_flat": std_flat,
        "zero_dev_flat": zero_dev_flat,
        "dynamic_range_flat": dynamic_range_flat,
        "mean_absolute_derivative_spike": mean_abs_der_spike,
        "LF_ratio_motion": lf_ratio_motion,
        "InBand_power_ratio_out": inband_out,
        "kurtosis_out": kurtosis_out,
        "Skewness_out": skewness_out,
    }


def main():
    # 1. Base Folder Selection
    # Check if base folder is provided as a terminal argument
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
        if not os.path.isdir(base_folder):
            print(
                f"Error: The provided path '{base_folder}' is not a valid directory."
            )
            sys.exit(1)
    else:
        # Fallback to GUI if no argument is provided
        root_tk = tk.Tk()
        root_tk.withdraw()
        base_folder = filedialog.askdirectory(
            title="Select Base Database Folder"
        )
        if not base_folder:
            print("No folder selected. Exiting.")
            return

    # Define output file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = r"data\evaluation_results.parquet"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress_csv = os.path.join(
        output_dir if output_dir else ".", "processed_files.csv"
    )

    # Load processed files for interruption handling
    processed_files = set()
    if os.path.exists(progress_csv):
        with open(progress_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    processed_files.add(row[0])
        print(f"Loaded {len(processed_files)} already processed files.")

    fs = 100
    win_size = 1000
    step_size = 500

    subdirs = sorted(
        [
            d
            for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d))
        ]
    )

    all_results = []

    for subdir_name in subdirs:
        subdir_path = os.path.join(base_folder, subdir_name)
        print(f"\n--- Processing directory: {subdir_name} ---")

        files = sorted(
            [f for f in os.listdir(subdir_path) if f.endswith(".edf")]
        )

        for file in files:
            # Relative path for tracking in CSV
            rel_path = os.path.join(subdir_name, file)

            if rel_path in processed_files:
                continue

            print(f"Processing file: {file}")
            parts = file.replace(".edf", "").split("_", 1)
            if len(parts) < 2:
                continue

            edf_path = os.path.join(subdir_path, file)
            ecg, ppg, _ = read_edf_signals(edf_path)

            if ecg is None or ppg is None:
                print(f"Skipping {file}: Missing signal data")
                continue

            # Windowing
            n_samples = len(ecg)
            for section_id, start in enumerate(
                range(0, n_samples - win_size + 1, step_size)
            ):
                end = start + win_size
                ecg_slice = ecg[start:end]
                ppg_slice = ppg[start:end]

                ecg_metrics = compute_metrics(ecg_slice, fs, "ecg")
                ppg_metrics = compute_metrics(ppg_slice, fs, "ppg")

                # We flatten metrics into rows. One row per signal type per section.
                # Common metadata:
                common_meta = {
                    "base_folder": subdir_name,
                    "edf_filename": file,
                    "section_id": section_id,
                    "start": start,
                    "end": end,
                }

                # ECG row
                ecg_row = common_meta.copy()
                ecg_row.update({"signal_type": "ecg", **ecg_metrics})
                all_results.append(ecg_row)

                # PPG row
                ppg_row = common_meta.copy()
                ppg_row.update({"signal_type": "ppg", **ppg_metrics})
                all_results.append(ppg_row)

            # Update progress CSV immediately after successful processing
            with open(progress_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rel_path])

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_parquet(output_file, index=False, compression="snappy")
        print(f"Evaluation complete. Results saved to {output_file}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
