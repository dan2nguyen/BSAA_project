import numpy as np
import scipy.signal
from sleepecg import detect_heartbeats


def detect_qrs(
    ecg: np.ndarray,
    fs: float,
    chunk_s: float = 300.0,
    overlap_s: float = 10.0,
    max_flat_fraction: float = 0.99,
) -> np.ndarray:
    """
    Detects QRS peaks in an ECG signal with the Pan-Tompkins algorithm.

    The signal is processed in chunks so that long recordings and large
    amplitude changes inside one file are handled. Each chunk is extended
    by `overlap_s` on both sides, z-scored and passed to
    `sleepecg.detect_heartbeats`. Only peaks from the core of the chunk
    (without the overlap) are kept, so no peak is reported twice.

    The input is expected to be already filtered (filtered database), no
    additional filtering is done here apart from the internal stages of
    the detector.

    Chunks that are dead (almost all consecutive samples identical) are
    skipped: there are no beats to find and the searchback of the detector
    gets extremely slow on such data. With the default threshold a skipped
    chunk has only a few seconds of changing signal, so its segments are
    rejected by the quality check anyway.

    Args:
        ecg (np.ndarray): 1D ECG signal.
        fs (float): Sampling rate in Hz.
        chunk_s (float): Length of the chunk core in seconds.
        overlap_s (float): Extension of the chunk on each side in seconds.
        max_flat_fraction (float): Chunks with a higher share of
            zero-derivative samples are skipped.

    Returns:
        np.ndarray: Sorted sample indices of the detected QRS peaks.
    """
    ecg = np.asarray(ecg, dtype=float)
    n_samples = len(ecg)
    chunk = int(round(chunk_s * fs))
    overlap = int(round(overlap_s * fs))
    if chunk <= 0:
        raise ValueError("chunk_s must be positive")

    peaks = []
    for core_start in range(0, n_samples, chunk):
        core_end = min(core_start + chunk, n_samples)
        ext_start = max(core_start - overlap, 0)
        ext_end = min(core_end + overlap, n_samples)

        x = ecg[ext_start:ext_end]
        std = np.std(x)
        # Constant or broken chunk, nothing to detect
        if not np.isfinite(std) or std == 0:
            continue
        if np.mean(np.diff(x) == 0) > max_flat_fraction:
            continue

        z = (x - np.mean(x)) / std
        chunk_peaks = detect_heartbeats(z, fs) + ext_start
        in_core = (chunk_peaks >= core_start) & (chunk_peaks < core_end)
        peaks.append(chunk_peaks[in_core])

    if not peaks:
        return np.array([], dtype=int)
    return np.concatenate(peaks).astype(int)


def pan_tompkins_stages(
    ecg: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] = (5.0, 15.0),
    window_s: float = 0.15,
) -> dict:
    """
    Computes the intermediate signals of the classic Pan-Tompkins
    preprocessing chain. For illustration only (slides): the actual
    detection in `detect_qrs` is done by sleepecg, which uses a 5-30 Hz
    bandpass and its own implementation of these stages.

    Args:
        ecg (np.ndarray): 1D ECG signal.
        fs (float): Sampling rate in Hz.
        band_hz (tuple): Bandpass cutoff frequencies in Hz.
        window_s (float): Length of the moving integration window in seconds.

    Returns:
        dict: Arrays of the same length as `ecg` under the keys
            input, bandpass, derivative, squared, integrated.
    """
    ecg = np.asarray(ecg, dtype=float)
    sos = scipy.signal.butter(2, band_hz, btype="bandpass", fs=fs, output="sos")
    bandpass = scipy.signal.sosfiltfilt(sos, ecg)

    # Five-point derivative from the original paper, centered (no delay)
    kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) * fs / 8.0
    derivative = np.convolve(bandpass, kernel, mode="same")

    squared = derivative**2

    window = max(int(round(window_s * fs)), 1)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    return {
        "input": ecg,
        "bandpass": bandpass,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
    }


def pan_tompkins_threshold(
    integrated: np.ndarray,
    fs: float,
    refractory_s: float = 0.2,
    learning_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the classic Pan-Tompkins adaptive threshold to the integrated
    signal. For illustration only (slides): `detect_qrs` uses the sleepecg
    implementation, which additionally has a searchback and a second
    threshold on the filtered signal.

    The threshold is NPKI + 0.25 * (SPKI - NPKI), where SPKI and NPKI are
    running estimates of the signal and noise peak levels.

    Args:
        integrated (np.ndarray): Output of the moving window integration.
        fs (float): Sampling rate in Hz.
        refractory_s (float): Minimum distance between peak candidates.
        learning_s (float): Initial part used to initialize SPKI and NPKI.

    Returns:
        tuple: (threshold, peaks)
            - threshold: threshold value for every sample of `integrated`
            - peaks: sample indices of the candidates above the threshold
    """
    integrated = np.asarray(integrated, dtype=float)
    candidates, _ = scipy.signal.find_peaks(
        integrated, distance=max(int(round(refractory_s * fs)), 1)
    )
    learning = integrated[: max(int(round(learning_s * fs)), 1)]
    spki = np.max(learning)
    npki = 0.5 * np.mean(learning)
    current = npki + 0.25 * (spki - npki)

    threshold = np.empty(len(integrated))
    peaks = []
    last = 0
    for candidate in candidates:
        threshold[last:candidate] = current
        value = integrated[candidate]
        if value > current:
            spki = 0.125 * value + 0.875 * spki
            peaks.append(candidate)
        else:
            npki = 0.125 * value + 0.875 * npki
        current = npki + 0.25 * (spki - npki)
        last = candidate
    threshold[last:] = current
    return threshold, np.array(peaks, dtype=int)
