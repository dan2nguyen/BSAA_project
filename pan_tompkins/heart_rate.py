import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from pan_tompkins.detection import detect_qrs
from pan_tompkins.quality import check_segment_quality
from utils.read_edf import read_edf_signals

MIN_BEATS = 3
HR_RANGE_BPM = (30.0, 220.0)
MAX_RR_CV = 0.15


def compute_hr_segments(
    peaks: np.ndarray,
    n_samples: int,
    fs: float,
    segment_s: float = 10.0,
    step_s: float | None = None,
    min_beats: int = MIN_BEATS,
    hr_range_bpm: tuple[float, float] = HR_RANGE_BPM,
    max_rr_cv: float = MAX_RR_CV,
) -> pd.DataFrame:
    """
    Computes heart rate in fixed-length segments from detected QRS peaks.

    Only RR intervals lying completely inside a segment are used. The last
    incomplete segment of the recording is skipped.

    A segment is invalid (HR = NaN) when it has too few beats, HR outside
    `hr_range_bpm` or too irregular RR intervals (rr_cv > `max_rr_cv`,
    typical for artifacts and missed/double beats). The RR statistics and
    rr_cv are kept for every segment with enough beats, so the thresholds
    can be changed later without running the detection again.

    Args:
        peaks (np.ndarray): Sorted sample indices of the QRS peaks.
        n_samples (int): Length of the recording in samples.
        fs (float): Sampling rate in Hz.
        segment_s (float): Segment length in seconds.
        step_s (float, optional): Step between segment starts in seconds.
            Defaults to `segment_s` (no overlap).
        min_beats (int): Minimum number of beats in a segment to compute HR.
        hr_range_bpm (tuple): Accepted (min, max) HR in bpm.
        max_rr_cv (float): Maximum accepted rr_std_s / rr_mean_s.

    Returns:
        pd.DataFrame: One row per segment with columns
            start, end (samples), start_s, end_s, segment_s, n_beats,
            hr_bpm (from mean RR), hr_median_bpm (from median RR),
            rr_mean_s, rr_median_s, rr_std_s, rr_cv (rr_std_s / rr_mean_s),
            valid, invalid_reason.
    """
    if step_s is None:
        step_s = segment_s
    seg_len = int(round(segment_s * fs))
    step = int(round(step_s * fs))
    if seg_len <= 0 or step <= 0:
        raise ValueError("segment_s and step_s must be positive")

    peaks = np.asarray(peaks)
    starts = np.arange(0, n_samples - seg_len + 1, step)
    ends = starts + seg_len
    first = np.searchsorted(peaks, starts, side="left")
    last = np.searchsorted(peaks, ends, side="left")

    rows = []
    for start, end, i, j in zip(starts, ends, first, last):
        row = {
            "start": int(start),
            "end": int(end),
            "start_s": start / fs,
            "end_s": end / fs,
            "segment_s": segment_s,
            "n_beats": int(j - i),
            "hr_bpm": np.nan,
            "hr_median_bpm": np.nan,
            "rr_mean_s": np.nan,
            "rr_median_s": np.nan,
            "rr_std_s": np.nan,
            "rr_cv": np.nan,
            "valid": False,
            "invalid_reason": "",
        }
        if row["n_beats"] < min_beats:
            row["invalid_reason"] = "too_few_beats"
            rows.append(row)
            continue

        rr = np.diff(peaks[i:j]) / fs
        row["rr_mean_s"] = float(np.mean(rr))
        row["rr_median_s"] = float(np.median(rr))
        row["rr_std_s"] = float(np.std(rr))
        row["rr_cv"] = row["rr_std_s"] / row["rr_mean_s"]

        hr = 60.0 / row["rr_mean_s"]
        if not hr_range_bpm[0] <= hr <= hr_range_bpm[1]:
            row["invalid_reason"] = "hr_out_of_range"
        elif row["rr_cv"] > max_rr_cv:
            row["invalid_reason"] = "irregular_rr"
        else:
            row["hr_bpm"] = float(hr)
            row["hr_median_bpm"] = float(60.0 / row["rr_median_s"])
            row["valid"] = True
        rows.append(row)

    return pd.DataFrame(rows)


def estimate_hr(
    ecg: np.ndarray,
    fs: float,
    segment_s: float = 10.0,
    step_s: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Runs the whole pipeline on one ECG signal: QRS detection, segment-wise
    HR and the segment quality check.

    Segments failing the quality check get NaN in all HR/RR columns
    (including rr_cv) and the quality reason in `invalid_reason` (it has
    priority over the beat-based reasons from `compute_hr_segments`).

    Args:
        ecg (np.ndarray): 1D ECG signal (already filtered).
        fs (float): Sampling rate in Hz.
        segment_s (float): Segment length in seconds.
        step_s (float, optional): Step between segment starts in seconds.

    Returns:
        tuple: (hr, peaks)
            - hr: DataFrame as returned by `compute_hr_segments`
            - peaks: sample indices of the detected QRS peaks
    """
    peaks = detect_qrs(ecg, fs)
    hr = compute_hr_segments(peaks, len(ecg), fs, segment_s, step_s)

    hr_columns = [
        "hr_bpm",
        "hr_median_bpm",
        "rr_mean_s",
        "rr_median_s",
        "rr_std_s",
        "rr_cv",
    ]
    for idx, start, end in zip(hr.index, hr["start"], hr["end"]):
        ok, reason = check_segment_quality(ecg[start:end])
        if not ok:
            hr.loc[idx, hr_columns] = np.nan
            hr.loc[idx, "valid"] = False
            hr.loc[idx, "invalid_reason"] = reason
    return hr, peaks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute segment-wise HR for a single EDF file."
    )
    parser.add_argument("edf_path", type=Path)
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=None)
    args = parser.parse_args()

    ecg, _, metadata = read_edf_signals(str(args.edf_path))
    if ecg is None:
        sys.exit(f"No ECG channel found in {args.edf_path}")
    fs = metadata["sampling_rate"]

    hr, peaks = estimate_hr(ecg, fs, args.segment_s, args.step_s)
    print(hr.to_string())
    print(f"\n{len(peaks)} beats, median HR {hr['hr_bpm'].median():.1f} bpm")
