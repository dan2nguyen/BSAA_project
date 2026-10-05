import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from utils.read_edf import read_edf_signals

def load_evaluation_json():
    with open("signal_quality_evaluation.json", "r") as f:
        return json.load(f)

def flatten_sections(data):
    """Flattens JSON into a list of addressable sections."""
    sections = []
    base_folder_name = data.get("base_folder_name", "")
    # We need the base folder path to load EDFs.
    # Since the JSON only has the base_folder_name, we might need to find the path.
    # However, the visualizer is expected to find the files.
    # Let's assume for now the user runs this in a context where they can specify the path
    # or we search for it. Actually, let's make the base path a configurable variable.

    subfolders = data.get("subfolders", {})
    for subdir, patients in subfolders.items():
        for patient, specs in patients.items():
            for spec, sections_dict in specs.items():
                for section_id, info in sections_dict.items():
                    sections.append({
                        "subdir": subdir,
                        "patient": patient,
                        "spec": spec,
                        "section_id": section_id,
                        "indexing": info["indexing"],
                        "ecg_metrics": info["ecg"],
                        "ppg_metrics": info["ppg"]
                    })
    return sections

class SignalVisualizer:
    def __init__(self, base_path, sections):
        self.base_path = base_path
        self.sections = sections
        self.current_section_idx = random.randint(0, len(self.sections) - 1)

        self.fig, (self.ax_ecg, self.ax_ppg) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(right=0.75, hspace=0.3)

        # Text box for metrics
        self.ax_text = self.fig.add_axes([0.77, 0.1, 0.2, 0.8])
        self.ax_text.axis('off')

        # Next Button
        ax_button = plt.axes([0.8, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_button, 'Next')
        self.btn_next.on_clicked(self.next_section)

        self.update_plot()

    def next_section(self, event):
        self.current_section_idx = random.randint(0, len(self.sections) - 1)
        self.update_plot()

    def update_plot(self):
        section = self.sections[self.current_section_idx]

        # Construct file path
        # Filename: <patient_number>_<signal_specification>.edf
        filename = f"{section['patient']}_{section['spec']}.edf"
        file_path = os.path.join(self.base_path, section['subdir'], filename)

        ecg, ppg, meta = read_edf_signals(file_path)
        if ecg is None or ppg is None:
            print(f"Error loading {file_path}")
            return

        start, end = section['indexing']
        ecg_slice = ecg[start:end]
        ppg_slice = ppg[start:end]

        time = np.arange(len(ecg_slice)) / 100.0 # fs = 100

        # Plotting
        self.ax_ecg.clear()
        self.ax_ecg.plot(time, ecg_slice, color='blue')
        self.ax_ecg.set_title(f"ECG - Section {section['section_id']} ({section['patient']})")
        self.ax_ecg.set_ylabel("Amplitude")
        self.ax_ecg.grid(True)

        self.ax_ppg.clear()
        self.ax_ppg.plot(time, ppg_slice, color='red')
        self.ax_ppg.set_title(f"PPG - Section {section['section_id']} ({section['patient']})")
        self.ax_ppg.set_xlabel("Time (s)")
        self.ax_ppg.set_ylabel("Amplitude")
        self.ax_ppg.grid(True)

        # Metrics Text
        self.ax_text.clear()
        self.ax_text.axis('off')

        text_lines = []
        for sig_type in ["ecg", "ppg"]:
            metrics = section[f"{sig_type}_metrics"]
            text_lines.append(f"--- {sig_type.upper()} ---")
            for k, v in metrics.items():
                if isinstance(v, bool):
                    flag = "[FLAGGED]" if v else "[OK]"
                    color = "red" if v else "black" # Note: Matplotlib text is usually one color per object
                    # We can't easily change color per line in a single text object,
                    # so we'll just use the [FLAGGED] marker.
                    text_lines.append(f"{k}: {flag}")
                elif k != "std_piecewise":
                    text_lines.append(f"{k}: {v:.4f}")
            text_lines.append("")

        full_text = "\n".join(text_lines)
        self.ax_text.text(0, 1, full_text, verticalalignment='top', family='monospace', fontsize=9)

        plt.draw()

def main():
    # We need the base path to find the EDFs.
    # Since it's not in the JSON (only the name is), we ask the user.
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    base_path = filedialog.askdirectory(title="Select Base Database Folder (where EDFs are stored)")
    if not base_path:
        print("No folder selected. Exiting.")
        return

    try:
        data = load_evaluation_json()
    except FileNotFoundError:
        print("signal_quality_evaluation.json not found in root directory.")
        return

    sections = flatten_sections(data)
    if not sections:
        print("No sections found in evaluation JSON.")
        return

    visualizer = SignalVisualizer(base_path, sections)
    plt.show()

if __name__ == "__main__":
    main()
