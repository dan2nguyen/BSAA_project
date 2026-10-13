import argparse
import os
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.read_edf import read_edf_signals


def load_evaluation_data(
    eval_root=r"references\evaluation_results.parquet",
):
    """Loads evaluation results from a Parquet file."""
    if not os.path.exists(eval_root):
        print(f"Evaluation file {eval_root} not found.")
        return None

    try:
        print(f"Loading Parquet file: {eval_root}...")
        df = pd.read_parquet(eval_root)
        print(f"Successfully loaded {len(df)} sections.")
        return df
    except Exception as e:
        print(f"Error loading Parquet file {eval_root}: {e}")
        return None


def load_corrupted_data(
    corr_root=r"references\corrupted_sections.parquet",
):
    """Loads corrupted section data from a Parquet file."""
    if not os.path.exists(corr_root):
        print(f"Corrupted sections file {corr_root} not found.")
        return None

    try:
        df = pd.read_parquet(corr_root)
        return df
    except Exception as e:
        print(f"Error loading corrupted sections file {corr_root}: {e}")
        return None


class SignalVisualizer:
    def __init__(
        self,
        base_path,
        sections,
        mode="random",
        specific_config=None,
        corrupted_df=None,
    ):
        self.base_path = base_path
        self.sections = sections
        self.mode = mode
        self.specific_config = specific_config
        self.corrupted_df = corrupted_df

        self.fig, (self.ax_ecg, self.ax_ppg) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        plt.subplots_adjust(right=0.75, hspace=0.3)

        # Text box for metrics
        self.ax_text = self.fig.add_axes([0.77, 0.1, 0.2, 0.8])
        self.ax_text.axis("off")

        # Next Button
        ax_button = plt.axes([0.8, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_button, "Next")
        self.btn_next.on_clicked(self.next_section)

        self.current_section_idx = self._get_initial_section_idx()
        self.update_plot()

    def _get_initial_section_idx(self):
        if self.mode == "random":
            return random.randint(0, len(self.sections) - 1)

        # Specific mode logic
        if self.specific_config:
            target_file = self.specific_config.get("file")
            target_id = str(self.specific_config.get("section_num"))
            target_time = self.specific_config.get("section_time")

            # Filter for file match
            # Since rel_edf_path was constructed as base_folder + edf_filename
            # we check if combined path matches target_file
            mask = (
                self.sections["base_folder"]
                + "\\"
                + self.sections["edf_filename"]
            ).apply(os.path.normpath) == os.path.normpath(target_file)
            matching_indices = self.sections.index[mask].tolist()

            if matching_indices:
                for idx in matching_indices:
                    row = self.sections.loc[idx]
                    if (
                        target_id
                        and str(row["section_id"]) == target_id
                    ):
                        return idx
                    if (
                        target_time is not None
                        and abs(row["start"] / 100.0 - target_time)
                        < 0.5
                    ):
                        return idx

        print("Specific section not found, falling back to random.")
        return random.randint(0, len(self.sections) - 1)

    def next_section(self, event):
        if self.mode == "random":
            self.current_section_idx = random.randint(
                0, len(self.sections) - 1
            )
        else:
            # In specific mode, "Next" can either do nothing or move to next section of SAME file
            # Let's make it move to the next section of the same file for convenience
            current_row = self.sections.loc[self.current_section_idx]
            current_file = os.path.normpath(
                os.path.join(
                    current_row["base_folder"],
                    current_row["edf_filename"],
                )
            )

            # Find all indices with the same file
            mask = (
                self.sections["base_folder"]
                + "\\"
                + self.sections["edf_filename"]
            ).apply(os.path.normpath) == current_file
            matching_indices = self.sections.index[mask].tolist()

            # Find current position in the list of matching indices
            try:
                pos = matching_indices.index(self.current_section_idx)
                next_pos = (pos + 1) % len(matching_indices)
                self.current_section_idx = matching_indices[next_pos]
            except ValueError:
                # Fallback if for some reason current_section_idx is not in matching_indices
                self.current_section_idx = matching_indices[0]
        self.update_plot()

    def update_plot(self):
        # Use .iloc if current_section_idx is an integer position, or squeeze if multiple rows match
        # To avoid KeyError, we ensure we access the row as a Series from the DataFrame
        row = self.sections.iloc[self.current_section_idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # Construct file path from DataFrame row
        # We use .get() or check columns to avoid KeyError if the merge changed something
        subdir = str(row["base_folder"])
        filename = str(row["edf_filename"])
        file_path = os.path.join(self.base_path, subdir, filename)

        try:
            ecg, ppg, _ = read_edf_signals(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

        if ecg is None or ppg is None:
            print(f"Error loading {file_path}: Missing signal data")
            return

        # Ensure we are using values, not series/index
        # Access values safely
        try:
            start_col = (
                "start"
                if "start" in row
                else ("start_x" if "start_x" in row else None)
            )
            end_col = (
                "end"
                if "end" in row
                else ("end_x" if "end_x" in row else None)
            )

            if start_col is None or end_col is None:
                print(
                    f"Error: Missing start/end columns. Available: {self.sections.columns.tolist()}"
                )
                return

            start = int(row[start_col])
            end = int(row[end_col])
        except KeyError as e:
            print(
                f"Error: Column {e} not found in the filtered sections DataFrame."
            )
            print(
                f"Available columns: {self.sections.columns.tolist()}"
            )
            return

        ecg_slice = ecg[start:end]
        ppg_slice = ppg[start:end]

        time = np.arange(len(ecg_slice)) / 100.0  # fs = 100

        # Plotting
        self.ax_ecg.clear()
        self.ax_ecg.plot(time, ecg_slice, color="blue")
        self.ax_ecg.set_title(
            f"ECG - {os.path.join(subdir, filename)} | Sec {row['section_id']}"
        )
        self.ax_ecg.set_ylabel("Amplitude")
        self.ax_ecg.grid(True)

        self.ax_ppg.clear()
        self.ax_ppg.plot(time, ppg_slice, color="red")
        self.ax_ppg.set_title(
            f"PPG - {os.path.join(subdir, filename)} | Sec {row['section_id']}"
        )
        self.ax_ppg.set_xlabel("Time (s)")
        self.ax_ppg.set_ylabel("Amplitude")
        self.ax_ppg.grid(True)

        # Metrics Text
        self.ax_text.clear()
        self.ax_text.axis("off")

        text_lines = []

        # Add Corruption Status (from corrupted_df)
        if self.corrupted_df is not None:
            # Match corruption row to current section
            # Use a merge-like approach or indexing if indices align, but usually we match on keys
            match = self.corrupted_df[
                (self.corrupted_df["base_folder"] == subdir)
                & (self.corrupted_df["edf_filename"] == filename)
                & (self.corrupted_df["section_id"] == row["section_id"])
            ]
            if not match.empty:
                corr_row = match.iloc[0]
                # notFlat: True (Green) / False (Red)
                # no_abnormal_waves: True (Green) / False (Red)
                nf_val = corr_row.get("notFlat", None)
                naw_val = corr_row.get("no_abnormal_waves", None)

                text_lines.append("--- CORRUPTION STATUS ---")
                # We use colored circles or markers in text if possible, or just labels.
                # In Matplotlib text, we can't easily change color per word,
                # so we'll add these as separate text objects or just use labels.
                # For simplicity in the monospace block, we use [OK] / [CORRUPT]
                # but I'll add a separate logic to draw indicators if requested.
                # The prompt asks for "red or green depending on whether the value is true or false".
                # Since this is in the text box (ax_text), I'll add labels and then
                # maybe draw small colored rectangles on the plot.
                nf_label = (
                    "NOT FLAT: OK" if nf_val else "NOT FLAT: CORRUPT"
                )
                naw_label = (
                    "NO ABNORMAL: OK"
                    if naw_val
                    else "NO ABNORMAL: CORRUPT"
                )
                text_lines.append(nf_label)
                text_lines.append(naw_label)
                text_lines.append("")
            else:
                text_lines.append(
                    "Corruption data not found for this section\n"
                )

        for sig_type in ["ecg", "ppg"]:
            # Filter columns that start with the signal type
            metrics = {
                k: v
                for k, v in row.items()
                if k.startswith(f"{sig_type}_")
            }
            text_lines.append(f"--- {sig_type.upper()} ---")
            for k, v in metrics.items():
                if isinstance(v, bool):
                    flag = "[FLAGGED]" if v else "[OK]"
                    text_lines.append(f"{k}: {flag}")
                elif k != "std_piecewise":
                    try:
                        text_lines.append(f"{k}: {v:.4f}")
                    except (TypeError, ValueError):
                        text_lines.append(f"{k}: {v}")
            text_lines.append("")

        full_text = "\n".join(text_lines)
        self.ax_text.text(
            0,
            1,
            full_text,
            verticalalignment="top",
            family="monospace",
            fontsize=9,
        )

        # Draw colored status indicators
        if self.corrupted_df is not None:
            match = self.corrupted_df[
                (self.corrupted_df["base_folder"] == subdir)
                & (self.corrupted_df["edf_filename"] == filename)
                & (self.corrupted_df["section_id"] == row["section_id"])
            ]
            if not match.empty:
                corr_row = match.iloc[0]
                nf_color = (
                    "green" if corr_row.get("notFlat", False) else "red"
                )
                naw_color = (
                    "green"
                    if corr_row.get("no_abnormal_waves", False)
                    else "red"
                )

                # Add indicators to the top right of the figure
                self.fig.text(
                    0.76,
                    0.95,
                    "Flat",
                    color="black",
                    fontsize=10,
                    ha="right",
                )
                self.fig.text(
                    0.77,
                    0.95,
                    " ● ",
                    color=nf_color,
                    fontsize=15,
                    ha="left",
                )
                self.fig.text(
                    0.76,
                    0.92,
                    "Abnormal",
                    color="black",
                    fontsize=10,
                    ha="right",
                )
                self.fig.text(
                    0.77,
                    0.92,
                    " ● ",
                    color=naw_color,
                    fontsize=15,
                    ha="left",
                )

        plt.draw()


def main():
    parser = argparse.ArgumentParser(
        description="Plot signal sections from evaluation results."
    )
    parser.add_argument(
        "--mode",
        choices=["random", "specific"],
        default="random",
        help="Selection mode",
    )
    parser.add_argument(
        "--filter",
        choices=[
            "notFlat_True",
            "notFlat_False",
            "no_abnormal_waves_True",
            "no_abnormal_waves_False",
            "both_True",
            "both_False",
            "either_corrupted",
            "neither",
        ],
        default="neither",
        help="Filter random sections based on corrupted_sections.parquet values",
    )
    parser.add_argument(
        "--file",
        help="Path to JSON file relative to evaluation_results",
    )
    parser.add_argument(
        "--section_num", type=int, help="Section index to plot"
    )
    parser.add_argument(
        "--section_time",
        type=float,
        help="Start time of section to plot",
    )
    args = parser.parse_args()

    # 1. Base folder for EDFs
    root_tk = tk.Tk()
    root_tk.withdraw()
    base_path = filedialog.askdirectory(
        title="Select Base Database Folder (where EDFs are stored)"
    )
    if not base_path:
        print("No folder selected. Exiting.")
        return

    # 2. Load all sections from the new structure
    df_sections = load_evaluation_data(
        r"references\evaluation_results.parquet"
    )
    if df_sections is None or df_sections.empty:
        print(
            "No evaluation sections found in 'references\\evaluation_results.parquet'."
        )
        return

    # 3. Load corruption data for filtering and display
    df_corrupted = load_corrupted_data(
        r"references\corrupted_sections.parquet"
    )

    # 4. Filter sections if in random mode and filter is specified
    if (
        args.mode == "random"
        and args.filter != "neither"
        and df_corrupted is not None
    ):
        print(f"Filtering sections by {args.filter}...")

        # Join evaluation results with corrupted sections on common keys
        # The keys are base_folder, edf_filename, section_id
        merged = df_sections.merge(
            df_corrupted,
            on=["base_folder", "edf_filename", "section_id"],
            how="inner",
        )

        if args.filter == "notFlat_True":
            mask = merged["notFlat"] == True
        elif args.filter == "notFlat_False":
            mask = merged["notFlat"] == False
        elif args.filter == "no_abnormal_waves_True":
            mask = merged["no_abnormal_waves"] == True
        elif args.filter == "no_abnormal_waves_False":
            mask = merged["no_abnormal_waves"] == False
        elif args.filter == "both_True":
            mask = (merged["notFlat"] == True) & (
                merged["no_abnormal_waves"] == True
            )
        elif args.filter == "both_False":
            mask = (merged["notFlat"] == False) & (
                merged["no_abnormal_waves"] == False
            )
        elif args.filter == "either_corrupted":
            mask = (merged["notFlat"] == False) | (
                merged["no_abnormal_waves"] == False
            )
        else:
            mask = pd.Series(True, index=merged.index)

        filtered_sections = merged[mask].copy()

        if filtered_sections.empty:
            print("No sections match the selected corruption filter.")
            return

        df_sections = filtered_sections
        print(f"Filtered down to {len(df_sections)} sections.")

    # 5. Handle Specific Mode Configuration
    specific_config = None
    if args.mode == "specific":
        target_file = args.file
        if not target_file:
            print(
                "No --file provided. Please select the JSON file from the evaluation results folder."
            )
            # We need to let the user select a file from the evaluation_results folder
            target_file = filedialog.askopenfilename(
                initialdir="evaluation_results",
                title="Select Evaluation JSON File",
            )
            if not target_file:
                print("No file selected. Exiting.")
                return
            # Convert absolute path back to relative path for the visualizer
            if os.path.isabs(target_file):
                target_file = os.path.relpath(
                    target_file, "evaluation_results"
                )

        section_num = args.section_num
        if section_num is None:
            try:
                # Use input() but be aware it may not work in some terminal environments
                section_num = int(
                    input("Enter section number to plot: ")
                )
            except ValueError:
                print(
                    "Invalid input. Defaulting to first section of, the same file."
                )
                section_num = 0

        specific_config = {
            "file": target_file,
            "section_num": section_num,
            "section_time": args.section_time,
        }

    SignalVisualizer(
        base_path,
        df_sections,
        mode=args.mode,
        specific_config=specific_config,
        corrupted_df=df_corrupted,
    )
    plt.show()


if __name__ == "__main__":
    main()
