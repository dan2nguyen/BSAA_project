import argparse
import os
import random
import tkinter as tk
import warnings
from tkinter import filedialog

import matplotlib.pyplot as plt
import mne


def read_edf_signals(
    file_path: str, ecg_keywords: list = None, ppg_keywords: list = None
):
    """
    Reads an EDF file and extracts ECG and PPG records.

    Args:
        file_path (str): Absolute path to the .edf file.
        ecg_keywords (list, optional): List of strings to identify ECG channels.
        ppg_keywords (list, optional): List of strings to identify PPG channels.

    Returns:
        tuple: (ecg_data, ppg_data, metadata)
            - ecg_data: numpy array or None
            - ppg_data: numpy array or None
            - metadata: Dictionary containing sampling rate, channel names, and other EDF header info.
    """
    # Default keywords for signal detection
    if ecg_keywords is None:
        ecg_keywords = ["ECG"]
    if ppg_keywords is None:
        ppg_keywords = ["PPG"]

    try:
        # Load EDF file. preload=True loads data into memory for easier access.
        # verbose=False suppresses MNE's detailed logging
        raw = mne.io.read_raw_edf(
            file_path, preload=True, verbose=False
        )
    except Exception as e:
        print(f"Error reading EDF file {file_path}: {e}")
        return None, None, None

    ch_names = raw.ch_names

    def find_channel(keywords):
        matches = [
            name
            for name in ch_names
            if any(kw.lower() in name.lower() for kw in keywords)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            warnings.warn(
                f"Multiple channels matched keywords {keywords}: {matches}. Using the first one: {matches[0]}"
            )
        return matches[0]

    ecg_channel = find_channel(ecg_keywords)
    ppg_channel = find_channel(ppg_keywords)

    # Extract data
    ecg_data = None
    if ecg_channel:
        # get_data returns (n_channels, n_times), we take [0] to get a 1D array
        ecg_data = raw.get_data(picks=[ecg_channel])[0]

    ppg_data = None
    if ppg_channel:
        ppg_data = raw.get_data(picks=[ppg_channel])[0]

    # Extract metadata
    metadata = {
        "sampling_rate": raw.info["sfreq"],
        "duration": raw.n_times / raw.info["sfreq"],
        "channel_mapping": {"ecg": ecg_channel, "ppg": ppg_channel},
        "all_channels": ch_names,
        "n_times": raw.n_times,
    }

    return ecg_data, ppg_data, metadata


def plot_edf_signals(ecg_data, ppg_data, metadata, figure_name=""):
    """
    Plots ECG and PPG signals using MNE's plotting capabilities.

    Args:
        ecg_data (numpy array): ECG signal data.
        ppg_data (numpy array): PPG signal data.
        metadata (dict): Metadata containing sampling rate and other info.
    """

    sfreq = metadata["sampling_rate"]
    duration = metadata["duration"]
    time_axis = [i / sfreq for i in range(int(duration * sfreq))]

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Signals from EDF File: {figure_name}", fontsize=16)

    if ecg_data is not None:
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, ecg_data, label="ECG", color="blue")
        plt.title("ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

    if ppg_data is not None:
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, ppg_data, label="PPG", color="green")
        plt.title("PPG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()


def get_random_edf_file(data_folder: str):
    """Scans the database folder and returns a random .edf file path."""
    edf_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))

    if not edf_files:
        return None
    return random.choice(edf_files)


def get_specific_edf_file():
    """Opens a file dialog window to select a .edf file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an EDF file",
        filetypes=[("EDF files", "*.edf")],
        initialdir=os.path.join(
            os.getcwd(), "harmonized_ecg_ppg_databases"
        ),
    )
    root.destroy()
    return file_path if file_path else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ECG/PPG signals from EDF files."
    )
    parser.add_argument(
        "--mode",
        choices=["specific", "random"],
        default="random",
        help="Selection mode: 'specific' for file picker, 'random' for random DB selection (default).",
    )
    args = parser.parse_args()

    selected_file = None
    if args.mode == "specific":
        print("Opening file dialog...")
        selected_file = get_specific_edf_file()
        if selected_file:
            print(f"Selected file: {selected_file}")
            ecg, ppg, meta = read_edf_signals(selected_file)
            if meta:
                plot_edf_signals(
                    ecg, ppg, meta, figure_name=selected_file
                )
            else:
                print("Failed to read signals from the selected file.")
        else:
            print("No EDF file was selected.")
    else:
        data_folder = "harmonized_ecg_ppg_databases"
        print(
            f"Random mode: Plotting all files in {data_folder} in random order."
        )
        print(
            "Close the plot window to see the next file, or press Ctrl+C in terminal to stop."
        )

        # Get all files first to shuffle them
        all_edf_files = []
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".edf"):
                    all_edf_files.append(os.path.join(root, file))

        if not all_edf_files:
            print("No EDF files found in the database.")
        else:
            random.shuffle(all_edf_files)
            try:
                for i, selected_file in enumerate(all_edf_files, 1):
                    print(
                        f"[{i}/{len(all_edf_files)}] Plotting: {selected_file}"
                    )
                    ecg, ppg, meta = read_edf_signals(selected_file)
                    if meta:
                        plot_edf_signals(
                            ecg, ppg, meta, figure_name=selected_file
                        )
                    else:
                        print(
                            f"Failed to read {selected_file}, skipping..."
                        )
            except KeyboardInterrupt:
                print("\nStopped by user.")
