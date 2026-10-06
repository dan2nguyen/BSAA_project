import numpy as np
from sleepecg import detect_heartbeats


def detect_qrs(
    ecg: np.ndarray,
    fs: float,
    chunk_s: float = 300.0,
    overlap_s: float = 10.0,
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

    Args:
        ecg (np.ndarray): 1D ECG signal.
        fs (float): Sampling rate in Hz.
        chunk_s (float): Length of the chunk core in seconds.
        overlap_s (float): Extension of the chunk on each side in seconds.

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

        z = (x - np.mean(x)) / std
        chunk_peaks = detect_heartbeats(z, fs) + ext_start
        in_core = (chunk_peaks >= core_start) & (chunk_peaks < core_end)
        peaks.append(chunk_peaks[in_core])

    if not peaks:
        return np.array([], dtype=int)
    return np.concatenate(peaks).astype(int)
