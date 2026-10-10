import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup paths for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.read_edf import read_edf_signals
from exploratory_analysis.filters import apply_standard_filters

def plot_edf(file_path):
    print(f"Loading: {file_path}")
    ecg, ppg, metadata = read_edf_signals(file_path)
    
    if metadata is None:
        print("Failed to read EDF file.")
        return

    fs = metadata.get("sampling_rate", 100.0)
    f_ecg, f_ppg = apply_standard_filters(ecg, ppg, fs=fs)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    if ecg is not None:
        plt.plot(ecg, color="blue")
    plt.title(f"Raw ECG\n{Path(file_path).name}")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    if f_ecg is not None:
        plt.plot(f_ecg, color="blue")
    plt.title("Filtered ECG")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    if ppg is not None:
        plt.plot(ppg, color="green")
    plt.title(f"Raw PPG\n{Path(file_path).name}")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    if f_ppg is not None:
        plt.plot(f_ppg, color="green")
    plt.title("Filtered PPG")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_single_edf.py <path_to_edf>")
    else:
        plot_edf(sys.argv[1])
