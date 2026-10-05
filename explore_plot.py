import argparse
import os
import random
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from utils.filters import apply_standard_filters
from utils.read_edf import read_edf_signals
from utils.time_frequency import plot_frequency_char


class EDFExplorer:
    def __init__(self, all_files):
        self.all_files = all_files
        self.current_idx = 0
        self.start_time = 0.0
        self.zoom_time = None

        # Setup Figure for Time Domain
        self.fig_time = plt.figure(figsize=(16, 10))
        self.ax_ecg_raw = plt.subplot(2, 2, 1)
        self.ax_ecg_filt = plt.subplot(2, 2, 3)
        self.ax_ppg_raw = plt.subplot(2, 2, 2)
        self.ax_ppg_filt = plt.subplot(2, 2, 4)

        # Spectral Analysis Figure (created but not shown initially)
        self.fig_freq = None
        self.ax_freq = None

        # UI elements layout - now tied to fig_time
        self.fig_time.subplots_adjust(bottom=0.2)

        # Next Button
        ax_button = self.fig_time.add_axes([0.65, 0.05, 0.1, 0.06])
        self.btn_next = Button(ax_button, "Next")
        self.btn_next.on_clicked(self.next_signal)

        # Spectral Button
        ax_spec_button = self.fig_time.add_axes([0.53, 0.05, 0.1, 0.06])
        self.btn_spectral = Button(ax_spec_button, "SPECTRAL")
        self.btn_spectral.on_clicked(self.show_spectral)

        # Zoom-Out Button
        ax_zoomout = self.fig_time.add_axes([0.4, 0.05, 0.1, 0.06])
        self.btn_zoomout = Button(ax_zoomout, "Zoom Out")
        self.btn_zoomout.on_clicked(self.zoom_out)

        # Zoom TextBox
        ax_box = self.fig_time.add_axes([0.25, 0.05, 0.1, 0.06])
        self.text_box = TextBox(ax_box, "Window (s): ", initial="10")
        self.text_box.on_submit(self.update_zoom)

        # Start Time TextBox
        ax_start_box = self.fig_time.add_axes([0.05, 0.05, 0.1, 0.06])
        self.start_box = TextBox(
            ax_start_box, "Start (s): ", initial="0"
        )
        self.start_box.on_submit(self.update_start)

        self.update_plot()

    def update_start(self, text):
        try:
            self.start_time = float(text)
            print(f"Start time set to {self.start_time} seconds.")
            self.update_plot()
        except ValueError:
            print("Invalid start time. Please enter a number.")

    def update_zoom(self, text):
        try:
            self.zoom_time = float(text)
            print(f"Window size set to {self.zoom_time} seconds.")
            self.update_plot()
        except ValueError:
            print("Invalid zoom time. Please enter a number.")

    def update_plot(self):
        # Clear time-domain axes
        for ax in [
            self.ax_ecg_raw,
            self.ax_ecg_filt,
            self.ax_ppg_raw,
            self.ax_ppg_filt,
        ]:
            ax.clear()

        file_path = self.all_files[self.current_idx]
        print(
            f"[{self.current_idx + 1}/{len(self.all_files)}] Loading: {file_path}"
        )

        ecg, ppg, meta = read_edf_signals(file_path)
        if not meta:
            print(f"Failed to read {file_path}, skipping...")
            self.next_signal(None)
            return

        sfreq = meta["sampling_rate"]
        duration = meta["duration"]
        time_axis = np.arange(int(duration * sfreq)) / sfreq

        # Apply filters
        f_ecg, f_ppg = apply_standard_filters(ecg, ppg, fs=sfreq)
        self.last_filtered_signals = (f_ecg, f_ppg, sfreq)

        # Time Domain Plotting
        if ecg is not None:
            self.ax_ecg_raw.plot(time_axis, ecg, color="blue")
            self.ax_ecg_raw.set_title("ECG Raw")
            self.ax_ecg_raw.grid(True)

            if f_ecg is not None:
                self.ax_ecg_filt.plot(time_axis, f_ecg, color="blue")
                self.ax_ecg_filt.set_title("ECG Filtered")
                self.ax_ecg_filt.set_xlabel("Time (s)")
                self.ax_ecg_filt.grid(True)

        if ppg is not None:
            self.ax_ppg_raw.plot(time_axis, ppg, color="green")
            self.ax_ppg_raw.set_title("PPG Raw")
            self.ax_ppg_raw.grid(True)

            if f_ppg is not None:
                self.ax_ppg_filt.plot(time_axis, f_ppg, color="green")
                self.ax_ppg_filt.set_title("PPG Filtered")
                self.ax_ppg_filt.set_xlabel("Time (s)")
                self.ax_ppg_filt.grid(True)

        # Apply Zoom and Start Time
        x_min = self.start_time
        x_max = self.start_time + (
            self.zoom_time if self.zoom_time is not None else duration
        )

        for ax in [
            self.ax_ecg_raw,
            self.ax_ecg_filt,
            self.ax_ppg_raw,
            self.ax_ppg_filt,
        ]:
            ax.set_xlim(x_min, x_max)

        self.fig_time.suptitle(
            f"Signal Comparison: Raw vs Filtered\nFile: {os.path.basename(file_path)}",
            fontsize=16,
        )
        self.fig_time.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Ensure the time figure is rendered
        plt.draw()

        # If spectral figure is already open, update it
        if self.fig_freq is not None:
            self.update_spectral_plot()

    def show_spectral(self, event):
        if self.fig_freq is None:
            self.fig_freq = plt.figure(figsize=(16, 12))
            self.ax_freq = self.fig_freq.subplots(2, 2)

        self.update_spectral_plot()
        plt.show()

    def update_spectral_plot(self):
        if self.fig_freq is None or not hasattr(
            self, "last_filtered_signals"
        ):
            print("No data available for spectral plot.")
            return

        # CRITICAL: Clear the axes before plotting to prevent overlays
        for ax in self.ax_freq.flat:
            ax.clear()

        f_ecg, f_ppg, sfreq = self.last_filtered_signals
        file_path = self.all_files[self.current_idx]

        # Ensure axes are passed correctly.
        # self.ax_freq is already a 2x2 array from subplots(2, 2)
        plot_frequency_char(self.ax_freq, f_ecg, f_ppg, fs=sfreq)
        self.fig_freq.suptitle(
            f"Spectral Characteristics\nFile: {os.path.basename(file_path)}",
            fontsize=16,
        )
        self.fig_freq.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()
        self.fig_freq.canvas.draw()

    def zoom_out(self, event):

        self.zoom_time = None
        self.start_time = 0.0
        self.text_box.set_val("")
        self.start_box.set_val("0")
        print("Zoom out: showing full signal.")
        self.update_plot()

    def next_signal(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.all_files)

        # Close spectral window if it's open
        if self.fig_freq is not None:
            plt.close(self.fig_freq)
            self.fig_freq = None
            self.ax_freq = None

        self.update_plot()


def get_specific_edf_file():
    root = tk.Tk()
    root.withdraw()
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
        "--mode", choices=["specific", "random"], default="random"
    )
    args = parser.parse_args()

    if args.mode == "specific":
        selected_file = get_specific_edf_file()
        if selected_file:
            explorer = EDFExplorer([selected_file])
            plt.show()
        else:
            print("No EDF file was selected.")
    else:
        data_folder = "harmonized_ecg_ppg_databases"
        all_edf_files = []
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".edf"):
                    all_edf_files.append(os.path.join(root, file))

        if not all_edf_files:
            print("No EDF files found in the database.")
        else:
            random.shuffle(all_edf_files)
            explorer = EDFExplorer(all_edf_files)
            plt.show()
