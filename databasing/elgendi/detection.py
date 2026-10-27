"""Elgendi-style PPG systolic peak detector (Elgendi et al., PLOS ONE, 2013).

Input is ALREADY FILTERED PPG (existing project Butterworth 0.5–8 Hz).
This is a transparent prototype, not a bit-for-bit reproduction of the paper.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class ElgendiConfig:
    peakwindow_s: float = 0.111
    beatwindow_s: float = 0.667
    beatoffset: float = 0.02
    mindelay_s: float = 0.300


def detect_ppg_peaks(ppg: np.ndarray, fs: float, config: ElgendiConfig = ElgendiConfig(),
                     return_stages: bool = False):
    """Find systolic peaks; returns integer sample indices (and optional stages).

    We clip negative values and square; compare a short moving average against
    the longer moving average plus an offset (offset = beatoffset * mean(squared)).
    Each above-threshold block is a candidate; narrow blocks are rejected.
    The highest filtered PPG sample in each block is selected as its systolic peak.
    Only peaks spaced >= mindelay_s are retained (stronger of competing peaks).
    """
    x = np.asarray(ppg, dtype=np.float64)
    if x.ndim != 1 or fs <= 0 or not np.isfinite(fs):
        raise ValueError('ppg must be a 1D signal and fs must be positive and finite')
    if min(config.peakwindow_s, config.beatwindow_s, config.mindelay_s) <= 0 or config.beatoffset < 0:
        raise ValueError('Invalid detector parameters')
    n_short = max(1, int(round(config.peakwindow_s * fs)))
    n_long = max(n_short + 1, int(round(config.beatwindow_s * fs)))
    if len(x) < n_long or not np.isfinite(x).all():
        peaks = np.array([], dtype=int)
        return (peaks, {}) if return_stages else peaks
    enhanced = np.maximum(x, 0.0) ** 2
    if np.max(enhanced) <= 0:
        peaks = np.array([], dtype=int)
        return (peaks, {}) if return_stages else peaks
    short = uniform_filter1d(enhanced, size=n_short, mode='nearest')
    long = uniform_filter1d(enhanced, size=n_long, mode='nearest')
    threshold = long + config.beatoffset * float(np.mean(enhanced))
    active = short > threshold
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    beginnings = np.flatnonzero(changes == 1)
    endings = np.flatnonzero(changes == -1)
    candidates = []
    for start, end in zip(beginnings, endings):
        if end - start < n_short:
            continue
        candidate = int(start + np.argmax(x[start:end]))
        candidates.append(candidate)
    gap = max(1, int(round(fs * config.mindelay_s)))
    chosen = []
    for candidate in candidates:
        if not chosen or candidate - chosen[-1] >= gap:
            chosen.append(candidate)
        elif x[candidate] > x[chosen[-1]]:
            chosen[-1] = candidate
    peaks = np.array(chosen, dtype=int)
    if return_stages:
        return peaks, {'enhanced': enhanced, 'short_ma': short,
                       'long_ma': long, 'threshold': threshold, 'active': active}
    return peaks
