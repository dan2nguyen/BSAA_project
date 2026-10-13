"""
This script evaluates the quality of ECG and PPG signal segments from EDF files.
It slides a window across the signals and computes a set of quality metrics for each segment.

Functions:
- compute_metrics(x, fs, signal_type):
    Input: signal slice (x), sampling frequency (fs), signal type ('ecg' or 'ppg').
    Output: Dictionary of quality metrics (std, flatness, spectral ratios, etc.).
- process_single_edf(task_info):
    Input: Tuple containing folder paths, filename, fs, window/step sizes, and shortening flag.
    Output: Tuple of (list of results per segment, relative file path).
- main():
    Input: Command line arguments (base folder, output file).
    Output: Parquet file with evaluated metrics and a progress CSV.

Inputs:
    - Base directory containing subdirectories with .edf files.
    - Window size (samples) and step size (samples) for segmentation.
    - Optional output file path.

Outputs:
    - A Parquet file containing a wide-format table with:
        - Metadata: folder, filename, section ID, start/end samples.
        - Signal Metrics: A set of quality metrics for both ECG and PPG (e.g., flatness, spikes, spectral ratios).
    - A 'processed_files.csv' to track progress and avoid re-processing files. (In case of interruption, the script can resume from where it left off.)
"""

import csv
import itertools
import os
import sys
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

    if signal_type == "ecg":
        dynamic_range_flat = dynamic_range < 1e-5
        inband_out = inband_ratio < 0.40
        kurtosis_out = kurt < 5.0 or kurt > 30.0
        skewness_out = abs(skew) < 0.2
    else:  # ppg
        dynamic_range_flat = dynamic_range < 1
        inband_out = inband_ratio < 0.50
        kurtosis_out = kurt < 2.0 or kurt > 10.0
        skewness_out = skew < 0.0

    std_flat = len([v for v in std_piecewise if v < 1e-6]) >= 3
    zero_dev_flat = zero_dev_dur > 4.0
    mean_abs_der_spike = mean_abs_der > (2.0 * std)
    lf_ratio_motion = lf_ratio > 0.6

    is_flat = sum([std_flat, zero_dev_flat, dynamic_range_flat]) >= 2

    return {
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
        "is_flat": is_flat,
        "mean_absolute_derivative_spike": mean_abs_der_spike,
        "LF_ratio_motion": lf_ratio_motion,
        "InBand_power_ratio_out": inband_out,
        "kurtosis_out": kurtosis_out,
        "Skewness_out": skewness_out,
    }


def process_single_edf(task_info):
    """Worker function to process one EDF file."""
    base_folder, subdir_name, file, fs, win_size, step_size, shorten = (
        task_info
    )

    rel_path = os.path.join(subdir_name, file)
    try:
        edf_path = os.path.join(base_folder, subdir_name, file)
        ecg, ppg, _ = read_edf_signals(edf_path)

        if ecg is None or ppg is None:
            return None, rel_path

        file_results = []
        n_samples = len(ecg)
        for section_id, start in enumerate(
            range(0, n_samples - win_size + 1, step_size)
        ):
            end = start + win_size
            ecg_slice = ecg[start:end]
            ppg_slice = ppg[start:end]

            ecg_metrics = compute_metrics(ecg_slice, fs, "ecg")
            ppg_metrics = compute_metrics(ppg_slice, fs, "ppg")

            # WIDE FORMAT: One row per section containing both ECG and PPG
            row = {
                "base_folder": subdir_name,
                "edf_filename": file,
                "section_id": section_id,
                "start": start,
                "end": end,
            }

            # Prefix metrics to avoid collisions
            metrics_to_keep = [
                "is_flat",
                "mean_absolute_derivative_spike",
                "LF_ratio_motion",
                "InBand_power_ratio_out",
                "kurtosis_out",
                "Skewness_out",
            ]

            for k, v in ecg_metrics.items():
                if not shorten or k in metrics_to_keep:
                    row[f"ecg_{k}"] = v
            for k, v in ppg_metrics.items():
                if not shorten or k in metrics_to_keep:
                    row[f"ppg_{k}"] = v

            file_results.append(row)

        return file_results, rel_path
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None, rel_path


def main():
    # 1. Base Folder Selection
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
        if not os.path.isdir(base_folder):
            print(
                f"Error: The provided path '{base_folder}' is not a valid directory."
            )
            sys.exit(1)
    else:
        root_tk = tk.Tk()
        root_tk.withdraw()
        base_folder = filedialog.askdirectory(
            title="Select Base Database Folder"
        )
        if not base_folder:
            print("No folder selected. Exiting.")
            return

    # Define output file
    fs = 100
    win_size = 2000
    step_size = 1000
    shorten = True

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = rf"databasing\references\evaluation_results_win{win_size}.parquet"

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    proccessed_file_name = rf"processed_files{win_size}.csv"
    progress_csv = os.path.join(
        output_dir if output_dir else ".", proccessed_file_name
    )

    processed_files = set()
    if os.path.exists(progress_csv):
        with open(progress_csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    processed_files.add(row[0])
        print(f"Loaded {len(processed_files)} already processed files.")

    # Collect all tasks
    tasks = []
    subdirs = sorted(
        [
            d
            for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d))
        ]
    )
    for subdir_name in subdirs:
        subdir_path = os.path.join(base_folder, subdir_name)
        files = sorted(
            [f for f in os.listdir(subdir_path) if f.endswith(".edf")]
        )
        for file in files:
            rel_path = os.path.join(subdir_name, file)
            if (
                rel_path not in processed_files
                and len(file.replace(".edf", "").split("_", 1)) >= 2
            ):
                tasks.append(
                    (
                        base_folder,
                        subdir_name,
                        file,
                        fs,
                        win_size,
                        step_size,
                        shorten,
                    )
                )

    print(f"Total files to process: {len(tasks)}")

    # Parquet writer initialization
    writer = None
    count = 0

    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_relpath = {
            executor.submit(process_single_edf, task): os.path.join(
                task[1], task[2]
            )
            for task in tasks
        }

        for future in as_completed(future_to_relpath):
            rel_path = future_to_relpath[future]
            try:
                file_results, returned_rel_path = future.result()

                if file_results:
                    df_temp = pd.DataFrame(file_results)
                    table = pa.Table.from_pandas(df_temp)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_file,
                            table.schema,
                            compression="snappy",
                        )
                    writer.write_table(table)

                    with open(progress_csv, "a", newline="") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([rel_path])

                    count += 1
                    if count % 10 == 0:
                        print(
                            f"Processed {count}/{len(tasks)} files..."
                        )

            except Exception as e:
                print(f"Unexpected error processing {rel_path}: {e}")

    if writer:
        writer.close()
        print(f"Evaluation complete. Results streamed to {output_file}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
