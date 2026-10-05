import itertools
import json
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import scipy.integrate
import scipy.signal
import scipy.stats
from read_edf import read_edf_signals


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


def save_json(data, filename="signal_quality_evaluation.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # 1. Base Folder Selection
    root_tk = tk.Tk()
    root_tk.withdraw()
    base_folder = filedialog.askdirectory(
        title="Select Base Database Folder"
    )
    if not base_folder:
        print("No folder selected. Exiting.")
        return

    json_filename = "signal_quality_evaluation.json"

    # Load existing data or initialize
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            output_data = json.load(f)
        print("Loaded existing progress from JSON.")
    else:
        output_data = {
            "base_folder_name": os.path.basename(base_folder),
            "checkpoint": {
                "last_processed_file": None,
                "last_processed_subdir": None,
            },
            "subfolders": {},
        }
        save_json(output_data, json_filename)
        print("Initialized new evaluation JSON.")

    # Ensure checkpoint exists in the data
    if "checkpoint" not in output_data:
        output_data["checkpoint"] = {
            "last_processed_file": None,
            "last_processed_subdir": None,
        }

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

    checkpoint = output_data["checkpoint"]
    last_file = checkpoint["last_processed_file"]
    last_subdir = checkpoint["last_processed_subdir"]

    # State to track if we should skip files to reach the checkpoint
    should_skip = True
    if last_file is None:
        should_skip = False

    for subdir_name in subdirs[:3]:
        subdir_path = os.path.join(base_folder, subdir_name)
        if subdir_name not in output_data["subfolders"]:
            output_data["subfolders"][subdir_name] = {}

        files = sorted(
            [f for f in os.listdir(subdir_path) if f.endswith(".edf")]
        )

        for file in files:
            # Checkpoint Logic
            if should_skip:
                if subdir_name == last_subdir and file == last_file:
                    should_skip = False
                    print(f"Reached checkpoint: {file}. Resuming...")
                else:
                    continue

            print(f"Processing: {file}")
            parts = file.replace(".edf", "").split("_", 1)
            if len(parts) < 2:
                continue

            patient_num, sig_spec = parts[0], parts[1]
            edf_path = os.path.join(subdir_path, file)
            ecg, ppg, _ = read_edf_signals(edf_path)

            if ecg is None or ppg is None:
                print(f"Skipping {file}: Missing signal data")
                continue

            # Ensure nested structure exists
            if (
                patient_num
                not in output_data["subfolders"][subdir_name]
            ):
                output_data["subfolders"][subdir_name][patient_num] = {}
            if (
                sig_spec
                not in output_data["subfolders"][subdir_name][
                    patient_num
                ]
            ):
                output_data["subfolders"][subdir_name][patient_num][
                    sig_spec
                ] = {}

            # Windowing
            n_samples = len(ecg)
            section_id = 0
            for start in range(0, n_samples - win_size + 1, step_size):
                end = start + win_size
                ecg_slice = ecg[start:end]
                ppg_slice = ppg[start:end]

                ecg_metrics = compute_metrics(ecg_slice, fs, "ecg")
                ppg_metrics = compute_metrics(ppg_slice, fs, "ppg")

                output_data["subfolders"][subdir_name][patient_num][
                    sig_spec
                ][str(section_id)] = {
                    "indexing": [start, end],
                    "ecg": ecg_metrics,
                    "ppg": ppg_metrics,
                }
                section_id += 1  # noqa: SIM113

            # Update Checkpoint and Save Incremental
            output_data["checkpoint"]["last_processed_file"] = file
            output_data["checkpoint"]["last_processed_subdir"] = (
                subdir_name
            )
            save_json(output_data, json_filename)

    print(
        "Evaluation complete. Results saved to signal_quality_evaluation.json"
    )


if __name__ == "__main__":
    main()
