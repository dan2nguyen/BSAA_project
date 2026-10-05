import numpy as np
from scipy import signal


def compute_psd(sig, fs=100.0, nperseg_sec=3.0, noverlap_ratio=0.5):
    """
    Computes Welch's Power Spectral Density (PSD) using a Hanning window.

    Args:
        sig: Input signal.
        fs: Sampling frequency.
        nperseg_sec: Length of each segment in seconds.
        noverlap_ratio: Overlap ratio (0 to 1).

    Returns:
        frequencies: Array of sample frequencies.
        psd: Power spectral density.
    """
    nperseg = int(nperseg_sec * fs)
    noverlap = int(nperseg * noverlap_ratio)

    freqs, psd = signal.welch(
        sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    return freqs, psd


def compute_spectrogram(
    sig, fs=100.0, window_sec=1.5, noverlap_ratio=0.75
):
    """
    Computes the STFT spectrogram of a signal.

    Args:
        sig: Input signal.
        fs: Sampling frequency.
        window_sec: Window length in seconds.
        noverlap_ratio: Overlap ratio (0 to 1).

    Returns:
        frequencies: Array of sample frequencies.
        time_bins: Array of segment times.
        sxx: Spectrogram (PSD) matrix.
    """
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * noverlap_ratio)

    freqs, time_bins, sxx = signal.spectrogram(
        sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap
    )
    return freqs, time_bins, sxx


def extract_spectral_metrics(
    freqs, psd, f_hr_range=(0.7, 3.5), f_powerline=50.0
):
    """
    Extracts key spectral metrics from PSD.

    Args:
        freqs: Array of frequencies.
        psd: Power spectral density array.
        f_hr_range: Range to search for dominant heart rate peak.
        f_powerline: Frequency of power line interference.

    Returns:
        metrics: Dictionary containing dominant_hr_freq, powerline_ratio, and baseline_drift_ratio.
    """
    # 1. Dominant HR Peak
    hr_mask = (freqs >= f_hr_range[0]) & (freqs <= f_hr_range[1])
    if not np.any(hr_mask):
        dominant_hr = None
    else:
        idx_max = np.argmax(psd[hr_mask])
        dominant_hr = freqs[hr_mask][idx_max]

    # 2. Powerline Spike Ratio
    # Find closest freq to f_powerline
    idx_pl = np.argmin(np.abs(freqs - f_powerline))
    # Local floor: mean PSD in +/- 1Hz window excluding the peak itself
    window_size = 5
    start = max(0, idx_pl - window_size)
    end = min(len(psd), idx_pl + window_size + 1)
    local_vals = np.delete(psd[start:end], idx_pl - start)
    local_floor = np.mean(local_vals) if len(local_vals) > 0 else 1e-12
    powerline_ratio = psd[idx_pl] / local_floor

    # 3. Baseline Drift Ratio
    drift_mask = freqs < 0.5
    power_drift = (
        np.trapezoid(psd[drift_mask], freqs[drift_mask])
        if np.any(drift_mask)
        else 0.0
    )
    power_total = np.trapezoid(psd, freqs)
    drift_ratio = power_drift / power_total if power_total > 0 else 0.0

    return {
        "dominant_hr_freq": dominant_hr,
        "powerline_ratio": powerline_ratio,
        "baseline_drift_ratio": drift_ratio,
    }


def plot_psd_comparison(ax, psd_dict, title="PSD Comparison"):
    """
    Plots PSD in dB for multiple signals.
    psd_dict: {label: (freqs, psd)}
    """
    for label, (freqs, psd) in psd_dict.items():
        psd_db = 10 * np.log10(psd + 1e-12)
        ax.plot(freqs, psd_db, label=label)

    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power [dB/Hz]")
    ax.grid(True)
    ax.legend()

    ax.axvspan(0.7, 3.5, color="gray", alpha=0.2, label="HR Range")
    ax.axvline(50.0, color="red", linestyle="--", label="Powerline")


def plot_spectrogram(
    ax, freqs, time_bins, sxx, title="Spectrogram", max_freq=50.0
):
    """
    Plots a 2D spectrogram heatmap.
    """
    sxx_db = 10 * np.log10(sxx + 1e-12)

    # To prevent "doubling" or aliasing artifacts, we use imshow.
    # pcolormesh can struggle with grid alignments; imshow is typically
    # better for regularly spaced spectrogram data.
    extent = [time_bins[0], time_bins[-1], freqs[0], freqs[-1]]
    img = ax.imshow(
        sxx_db,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(0, max_freq)

    fig = ax.get_figure()
    # Only add a colorbar if one doesn't already exist for this axis
    # to prevent colorbar "creep" and layout doubling.
    if not any(
        cb.ax.get_shared_xaxes() == ax.get_shared_xaxes()
        for cb in fig.axes
        if hasattr(cb, "colorbar")
    ):
        fig.colorbar(img, ax=ax, label="Power [dB]")
    else:
        # Update the existing colorbar if possible, or just rely on the new image
        pass

    # Alternative: Just remove the colorbar logic from here and handle it
    # once per figure, but for now, let's try to prevent duplicate colorbars.
    # Actually, the most robust way is to check for existing colorbars.
    # But simpler: just remove the colorbar from the per-plot function and
    # add it once. Let's refine this.


def plot_frequency_char(axes, ecg, ppg, fs=100.0):
    """
    Updates existing axes with spectral characteristics for both ECG and PPG signals.
    - axes: 2x2 array of Matplotlib axes.
    - Column 1: ECG (PSD top, Spectrogram bottom)
    - Column 2: PPG (PSD top, Spectrogram bottom)
    """
    signals = {
        "ECG": (ecg, axes[:, 0]),
        "PPG": (ppg, axes[:, 1]),
    }

    for label, (sig, axs) in signals.items():
        axs[0].clear()
        axs[1].clear()
        if sig is None:
            axs[0].set_title(f"{label} not provided")
            continue

        # 1. Computation
        freqs, psd = compute_psd(sig, fs)
        f_s, t_s, sxx = compute_spectrogram(sig, fs)
        metrics = extract_spectral_metrics(freqs, psd)

        # 2. Plotting
        plot_psd_comparison(
            axs[0], {label: (freqs, psd)}, title=f"{label} PSD"
        )
        plot_spectrogram(
            axs[1], f_s, t_s, sxx, title=f"{label} Spectrogram"
        )

        # 3. Print Metrics
        print(f"\n--- {label} Spectral Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    # Verification Block
    print("Running spectral analysis verification...")

    # 1. Synthetic Signal Test
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    sig = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.5 * np.sin(2 * np.pi * 50.0 * t)
        + 0.2 * np.random.randn(len(t))
    )

    f, p = compute_psd(sig, fs)
    metrics = extract_spectral_metrics(f, p)
    print("Synthetic Metrics:")
    [print(param) for param in metrics.items()]

    # 2. Example Use Case
    # In real use, you'd load from EDF and filter first.
    # Here we use the synthetic signal for both for demo purposes.
    print("\nPlotting Characteristics Demo...")
    plot_frequency_char(sig, sig, fs=fs)
