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


def load_all_sections(eval_root=r"data\evaluation_results.parquet"):
    """Loads evaluation results from a Parquet file and flattens them."""
    if not os.path.exists(eval_root):
        print(f"Evaluation file {eval_root} not found.")
        return []

    try:
        df = pd.read_parquet(eval_root)
        # We need to group by section to get both ECG and PPG metrics for the visualizer
        # as the original JSON structure had: section_id -> {ecg: ..., ppg: ...}
        sections = []

        # Grouping by the unique identifier of a section: base_folder, edf_filename, section_id
        grouped = df.groupby(["base_folder", "edf_filename", "section_id"])

        for (base_folder, edf_filename, section_id), group in grouped:
            # Extract metrics for each signal type
            ecg_row = group[group["signal_type"] == "ecg"]
            ppg_row = group[group["signal_type"] == "ppg"]

            if ecg_row.empty or ppg_row.empty:
                continue

            ecg_data = ecg_row.iloc[0].to_dict()
            ppg_data = ppg_row.iloc[0].to_dict()

            # Remove the metadata columns from the metrics dictionaries
            meta_cols = ["base_folder", "edf_filename", "section_id", "start", "end", "signal_type"]
            ecg_metrics = {k: v for k, v in ecg_data.items() if k not in meta_cols}
            ppg_metrics = {k: v for k, v in ppg_data.items() if k not in meta_cols}

            sections.append(
                {
                    "rel_json_path": os.path.join(base_folder, edf_filename),
                    "section_id": str(section_id),
                    "indexing": [ecg_data["start"], ecg_data["end"]],
                    "ecg_metrics": ecg_metrics,
                    "ppg_metrics": ppg_metrics,
                    "edf_filename": edf_filename,
                }
            )
    except Exception as e:
        print(f"Error loading Parquet file {eval_root}: {e}")
        return []

    return sections


class SignalVisualizer:
    def __init__(
        self, base_path, sections, mode="random", specific_config=None
    ):
        self.base_path = base_path
        self.sections = sections
        self.mode = mode
        self.specific_config = specific_config

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

            for i, s in enumerate(self.sections):
                # Check file match (normalize paths)
                if (
                    os.path.normpath(s["rel_json_path"])
                    == os.path.normpath(target_file)
                    and target_id
                    and s["section_id"] == target_id
                ):
                    return i
                if (
                    os.path.normpath(s["rel_json_path"])
                    == os.path.normpath(target_file)
                    and target_time is not None
                    and abs(s["indexing"][0] / 100.0 - target_time)
                    < 0.5
                ):
                    return i

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
            current_sec = self.sections[self.current_section_idx]
            next_idx = self.current_section_idx + 1
            while next_idx < len(self.sections):
                if (
                    self.sections[next_idx]["rel_json_path"]
                    == current_sec["rel_json_path"]
                ):
                    self.current_section_idx = next_idx
                    break
                next_idx += 1
            if next_idx == len(self.sections):
                # Wrap around to first section of the same file
                for i, s in enumerate(self.sections):
                    if (
                        s["rel_json_path"]
                        == current_sec["rel_json_path"]
                    ):
                        self.current_section_idx = i
                        break
        self.update_plot()

    def update_plot(self):
        section = self.sections[self.current_section_idx]

        # Need to find the EDF file. The sections list has edf_filename.
        # We need the subdir from the rel_json_path.
        rel_path = section["rel_json_path"]
        subdir = os.path.dirname(rel_path)
        filename = section["edf_filename"]
        file_path = os.path.join(self.base_path, subdir, filename)

        try:
            ecg, ppg, meta = read_edf_signals(file_path)  # noqa: RUF059
        except Exception as e:  # noqa: BLE001
            print(f"Error loading {file_path}: {e}")
            return

        if ecg is None or ppg is None:
            print(f"Error loading {file_path}: Missing signal data")
            return

        start, end = section["indexing"]
        ecg_slice = ecg[start:end]
        ppg_slice = ppg[start:end]

        time = np.arange(len(ecg_slice)) / 100.0  # fs = 100

        # Plotting
        self.ax_ecg.clear()
        self.ax_ecg.plot(time, ecg_slice, color="blue")
        self.ax_ecg.set_title(
            f"ECG - {rel_path} | Sec {section['section_id']}"
        )
        self.ax_ecg.set_ylabel("Amplitude")
        self.ax_ecg.grid(True)

        self.ax_ppg.clear()
        self.ax_ppg.plot(time, ppg_slice, color="red")
        self.ax_ppg.set_title(
            f"PPG - {rel_path} | Sec {section['section_id']}"
        )
        self.ax_ppg.set_xlabel("Time (s)")
        self.ax_ppg.set_ylabel("Amplitude")
        self.ax_ppg.grid(True)

        # Metrics Text
        self.ax_text.clear()
        self.ax_text.axis("off")

        text_lines = []
        for sig_type in ["ecg", "ppg"]:
            metrics = section[f"{sig_type}_metrics"]
            text_lines.append(f"--- {sig_type.upper()} ---")
            for k, v in metrics.items():
                if isinstance(v, bool):
                    flag = "[FLAGGED]" if v else "[OK]"
                    text_lines.append(f"{k}: {flag}")
                elif k != "std_piecewise":
                    try:
                        text_lines.append(f"{k}: {v:.4f}")
                    except:  # noqa: E722
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
    sections = load_all_sections(r"data\evaluation_results.parquet")
    if not sections:
        print("No evaluation sections found in 'data\\evaluation_results.parquet'.")
        return

    # 3. Handle Specific Mode Configuration
    specific_config = None
    if args.mode == "specific":
        target_file = args.file
        if not target_file:
            print(
                "No --file provided. Please select the JSON file from the evaluation results folder."
            )
            # We need to let the user select a file from the evaluation_results directory
            target_file = filedialog.askopenfilename(
                initialdir="evaluation_results",
                title="Select Evaluation JSON File",
                filetypes=[("JSON files", "*.json")],
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
                section_num = int(
                    input("Enter section number to plot: ")
                )
            except ValueError:
                print(
                    "Invalid input. Defaulting to first section of the file."
                )
                section_num = 0

        specific_config = {
            "file": target_file,
            "section_num": section_num,
            "section_time": args.section_time,
        }

    SignalVisualizer(
        base_path,
        sections,
        mode=args.mode,
        specific_config=specific_config,
    )
    plt.show()


if __name__ == "__main__":
    main()
