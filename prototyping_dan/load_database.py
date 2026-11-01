import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.read_edf import read_edf_signals


def load_parquet_files(
    folder_path: str,
    columns: list[str] | None = None,
    split_stats=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Parquet files from subfolders and split them into train and test dataframes.
    Now captures the source file path to allow for subject-based CV grouping.

    Args:
        folder_path (str): The path to the folder containing subfolders with Parquet files.
        columns (list[str] | None): List of columns to load.
        split_stats (bool): If True, prints the distribution of train/test samples.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    required_cols = {"is_flat", "hr_valid", "split"}
    if columns is None:
        columns = list(required_cols)
    else:
        columns = list(columns)
        for col in required_cols:
            if col not in columns:
                columns.append(col)

    all_data = []
    test_count = 0
    train_count = 0

    for d in os.listdir(folder_path):
        subfolder = os.path.join(folder_path, d)
        if not os.path.isdir(subfolder):
            continue

        for file in os.listdir(subfolder):
            if file.endswith(".parquet"):
                file_path = os.path.join(subfolder, file)
                dataframe = pd.read_parquet(file_path, columns=columns)

                if dataframe.empty:
                    continue

                condition = dataframe["hr_valid"] & (
                    ~dataframe["is_flat"]
                )
                dataframe_filtered = dataframe[condition].copy()

                # Store the filename to extract patient ID later
                dataframe_filtered["_source_file"] = file

                test_count += (
                    dataframe_filtered["split"] == "test"
                ).sum()
                train_count += (
                    dataframe_filtered["split"] == "train"
                ).sum()

                all_data.append(dataframe_filtered)

    if not all_data:
        raise FileNotFoundError(
            "No valid Parquet data found in the specified folder."
        )

    full_df = pd.concat(all_data, ignore_index=True)

    if split_stats:
        sample_count = test_count + train_count
        if sample_count > 0:
            test_percentage = test_count / sample_count
            train_percentage = train_count / sample_count
            print(f"Total samples: {sample_count}")
            print(f"Test samples: {test_count} ({test_percentage:.2%})")
            print(
                f"Train samples: {train_count} ({train_percentage:.2%})"
            )
        else:
            print("No valid samples found after filtering.")

    train_df = full_df[full_df["split"] == "train"].reset_index(
        drop=True
    )
    test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

    return train_df, test_df


def add_cv_groups(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Adds a 'cv_group' column to the dataframe based on patient IDs extracted from the source filename.
    """

    def extract_patient_id(filename: str) -> str:
        match = re.match(r"^(P\d+|s\d+|\d+)", filename)
        if match:
            return match.group(1)
        return "unknown"

    df = df.copy()
    patient_ids = df["_source_file"].apply(extract_patient_id)
    patient_counts = patient_ids.value_counts()
    sorted_patients = patient_counts.index.tolist()

    fold_totals = [0] * n_folds
    patient_to_fold = {}

    for patient in sorted_patients:
        count = patient_counts[patient]
        min_fold_idx = fold_totals.index(min(fold_totals))
        patient_to_fold[patient] = min_fold_idx
        fold_totals[min_fold_idx] += count

    df["cv_group"] = patient_ids.map(patient_to_fold)
    df = df.drop(columns=["_source_file"])

    return df


def load_signal_segment(
    row: pd.Series, base_dir: str, plot: bool = False
):
    """
    Extracts ECG and PPG signal segments from an EDF file using read_edf_signals.

    Args:
        row (pd.Series): A row from the database containing:
                         'base_folder', 'edf_filename', 'start', 'end'
        base_dir (str): The root directory where base_folders are located.
        plot (bool): Whether to plot the extracted segments.

    Returns:
        tuple[np.ndarray, np.ndarray]: (ecg_segment, ppg_segment)
    """
    edf_path = os.path.join(
        base_dir, row["base_folder"], row["edf_filename"]
    )

    # Use the utility function instead of pyedflib
    ecg_full, ppg_full, metadata = read_edf_signals(edf_path)

    if ecg_full is None or ppg_full is None or metadata is None:
        print(f"Error: Could not load signals from {edf_path}")
        return None, None

    fs = metadata["sampling_rate"]
    start_sample = row["start"]
    end_sample = row["end"]

    ecg_segment = ecg_full[start_sample:end_sample]
    ppg_segment = ppg_full[start_sample:end_sample]

    if plot:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(14, 6),
            sharex=True,  # increased width slightly
        )

        t = np.linspace(
            start_sample / fs, end_sample / fs, len(ecg_segment)
        )

        ax1.plot(t, ecg_segment, color="red")
        ax1.set_ylabel("ECG")
        ax1.set_title(
            f"Signal Segment - {row.get('edf_filename', 'Unknown')}"
        )

        ax2.plot(t, ppg_segment, color="blue")
        ax2.set_ylabel("PPG")
        ax2.set_xlabel("Time (s)")

        info_text = "\n".join([f"{k}: {v}" for k, v in row.items()])
        fig.text(
            0.78,
            0.5,
            info_text,
            va="center",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.5},
        )

        # Leaves the right 25% empty for the text box
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()
    return ecg_segment, ppg_segment


def main():
    try:
        train, _ = load_parquet_files(
            r"data\hr_parts\seg10s_step5s",
            columns=[
                "base_folder",
                "edf_filename",
                "start",
                "end",
                "hr_bpm",
                "split",
            ],
            split_stats=True,
        )

        train = add_cv_groups(train, n_folds=5)

        base_dir = r"data\harmonized_ecg_ppg_database_filtered"
        print("\nLoading and plotting a sample segment...")
        sample_row = train.iloc[150]
        ecg, ppg = load_signal_segment(
            sample_row, base_dir=base_dir, plot=True
        )

        if ecg is not None:
            print(
                f"Successfully loaded ECG ({len(ecg)}) and PPG ({len(ppg)}) samples."
            )

    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")
