"""
Entry point of the algorithm, as required by the interface specification:

    from estimate_heartrate import estimate_heartrate
    heartrate = estimate_heartrate(ppg, fs)

Input
    ppg: raw PPG signal (1-D array), as read from the EDF recording
    fs:  sampling rate of the PPG in Hz
Output
    heartrate: heart rate in beats per minute, sampled at 2 Hz (two values
    per second), same duration as the input. NaN where no estimate is
    possible (PPG not usable for a longer stretch, or shorter than 10 s).

The PPG goes through the same pipeline as the training data (predict.py):
filtering, 10 s segments every 5 s, PPG quality check, CNN. Each segment
estimate is placed at the segment centre and the 2 Hz output is linearly
interpolated between them. Gaps of up to 15 s left by rejected segments are
bridged; longer gaps stay NaN. At the start and the end of the recording
the nearest estimate is held.

The model file is data/training_out_cnn/dilated_cnn/fold_1/best_model.pt by
default; another one can be passed with model_path= or the environment
variable HR_MODEL_PATH.
"""

import os
import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from predict import DEFAULT_MODEL, SEGMENT_S, STEP_S, HRPredictor  # noqa: E402

OUTPUT_FS = 2.0  # Hz, required by the interface specification
MAX_GAP_S = 15.0  # longest gap between two estimates that is interpolated

_predictors = {}


def _get_predictor(model_path) -> HRPredictor:
    """Loads the model once and reuses it for later calls."""
    path = str(model_path or os.environ.get("HR_MODEL_PATH") or DEFAULT_MODEL)
    if path not in _predictors:
        _predictors[path] = HRPredictor(path)
    return _predictors[path]


def segment_heartrate(ppg, fs, model_path=None) -> pd.DataFrame:
    """
    HR per 10 s segment (step 5 s): start_t, end_t, hr_bpm (NaN = NONE),
    reason for NONE. The signal is written to a temporary EDF file so it
    runs through exactly the same preprocessing as the training database.
    """
    ppg = np.asarray(ppg, dtype=np.float64).ravel()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        edf_path = Path(tmp) / "input.edf"
        info = mne.create_info(["PPG"], sfreq=float(fs), ch_types=["misc"])
        raw = mne.io.RawArray(ppg[np.newaxis, :], info, verbose=False)
        raw.export(str(edf_path), fmt="edf", overwrite=True, verbose=False)
        return _get_predictor(model_path).predict_edf(edf_path)


def to_output_rate(seg: pd.DataFrame, duration_s: float,
                   rate: float = OUTPUT_FS,
                   max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """Segment estimates -> HR at `rate` Hz over the whole duration."""
    n = int(round(duration_s * rate))
    t = np.arange(n) / rate
    out = np.full(n, np.nan)
    if seg.empty:
        return out
    centres = ((seg["start_t"] + seg["end_t"]) / 2).to_numpy()
    hr = seg["hr_bpm"].to_numpy(dtype=float)
    ok = np.isfinite(hr)
    if not ok.any():
        return out
    c, h = centres[ok], hr[ok]

    # Linear interpolation between estimates, not across long gaps
    out = np.interp(t, c, h)
    k = np.clip(np.searchsorted(c, t), 1, len(c) - 1)
    inside = (t > c[0]) & (t < c[-1])
    if len(c) > 1:
        long_gap = (c[k] - c[k - 1]) > max_gap_s
        at_estimate = np.isclose(t[:, None], c[None, :]).any(axis=1)
        out[inside & long_gap & ~at_estimate] = np.nan

    # Edges: hold the first / last estimate if it is close enough
    hold = SEGMENT_S / 2 + STEP_S
    out[(t < c[0]) & (c[0] - t > hold)] = np.nan
    out[(t > c[-1]) & (t - c[-1] > hold)] = np.nan
    return out


def estimate_heartrate(ppg, fs, model_path=None) -> np.ndarray:
    """
    Heart rate in bpm at 2 Hz (two values per second) for a raw PPG signal.

    Args:
        ppg: raw PPG signal (1-D array).
        fs: sampling rate of the PPG in Hz.
        model_path: optional path to another model file.

    Returns:
        1-D array of length round(2 * len(ppg) / fs); NaN = no estimate.
    """
    ppg = np.asarray(ppg, dtype=np.float64).ravel()
    seg = segment_heartrate(ppg, fs, model_path)
    return to_output_rate(seg, len(ppg) / fs)


if __name__ == "__main__":
    # Example: python estimate_heartrate.py recording.edf
    from utils.read_edf import read_edf_signals

    _, ppg, meta = read_edf_signals(sys.argv[1])
    fs = meta["sampling_rate"]
    hr = estimate_heartrate(ppg, fs)
    print(f"{len(hr)} values at 2 Hz for {len(ppg) / fs:.0f} s of PPG,"
          f" median {np.nanmedian(hr):.1f} bpm,"
          f" {np.isfinite(hr).mean():.0%} with an estimate")
