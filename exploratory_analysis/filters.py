# utils/filters.py
# main function to use is apply_standard_filters(ecg, ppg)

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data), b, a


def notch_filter(data, notch_freq, fs, q=30):
    """Applies a notch filter to remove a specific frequency."""
    w0 = notch_freq / (fs / 2)
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, data), b, a


def apply_standard_filters(
    ecg,
    ppg,
    fs=100.0,
    ecg_low=0.5,
    ecg_high=40.0,
    ppg_low=0.5,
    ppg_high=8.0,
    notch_freq=50.0,
):
    """Applies standard filtering chain to ECG and PPG signals."""
    f_ecg, f_ppg = None, None
    if ecg is not None:
        ecg_n, _, _ = notch_filter(ecg, notch_freq, fs)
        f_ecg, _, _ = butter_bandpass_filter(
            ecg_n, ecg_low, ecg_high, fs
        )
    if ppg is not None:
        f_ppg, _, _ = butter_bandpass_filter(ppg, ppg_low, ppg_high, fs)
    return f_ecg, f_ppg


def plot_filter_characteristics(b, a, fs, title):
    """Plots the frequency response and impulse response of a filter."""
    # Frequency Response
    w, h = signal.freqz(b, a, worN=8000)
    freqs = w * fs / (2 * np.pi)

    # Impulse Response
    impulse = np.zeros(100)
    impulse[0] = 1.0
    impulse_resp = signal.lfilter(b, a, impulse)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(freqs, 20 * np.log10(np.abs(h)))
    plt.title(f"Frequency Response: {title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(impulse_resp)
    plt.title(f"Impulse Response: {title}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FS = 100.0

    # Define filter parameters
    params = {
        "ECG Bandpass": (0.5, 40.0, "band"),
        "PPG Bandpass": (0.5, 8.0, "band"),
        "ECG Notch": (50.0, "notch"),
    }

    for name, p in params.items():
        if p[-1] == "band":
            # Using order 4 as per requirements
            b, a = signal.butter(
                4, [p[0] / (FS / 2), p[1] / (FS / 2)], btype="band"
            )
        else:
            b, a = signal.iirnotch(p[0] / (FS / 2), 30)

        plot_filter_characteristics(b, a, FS, name)
