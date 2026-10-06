import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from pan_tompkins.detection import pan_tompkins_stages
from pan_tompkins.heart_rate import estimate_hr
from utils.read_edf import read_edf_signals

DEFAULT_EDF = (
    root_dir
    / "data"
    / "harmonized_ecg_ppg_database_filtered"
    / "wrist-finger-ppg"
    / "25_0.edf"
)
DEFAULT_OUTPUT_DIR = root_dir / "data" / "plots"

SURFACE = "#fcfcfb"
SIGNAL_COLOR = "#2a78d6"
PEAK_COLOR = "#eb6834"
INVALID_COLOR = "#898781"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"


def _style_axis(ax):
    """Applies the common look (recessive grid and axes) to an axis."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.xaxis.label.set_color(MUTED_COLOR)
    ax.yaxis.label.set_color(MUTED_COLOR)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-scores a signal, a constant signal is returned as zeros."""
    std = np.std(x)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def _slice_window(n_samples, fs, start_s, duration_s):
    """Returns (start, end) sample indices of a window clipped to the signal."""
    start = min(max(int(round(start_s * fs)), 0), n_samples)
    end = min(start + int(round(duration_s * fs)), n_samples)
    return start, end


def _shade_invalid_segments(ax, hr: pd.DataFrame):
    """Draws gray bands over invalid segments, neighbours are merged."""
    invalid = hr[~hr["valid"]]
    bands = []
    for seg_start, seg_end in zip(invalid["start_s"], invalid["end_s"]):
        if bands and seg_start <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], seg_end)
        else:
            bands.append([seg_start, seg_end])
    for i, (band_start, band_end) in enumerate(bands):
        ax.axvspan(
            band_start,
            band_end,
            color=INVALID_COLOR,
            alpha=0.2,
            linewidth=0,
            label="Invalid segment (HR = NaN)" if i == 0 else None,
        )


def _add_legend(ax):
    """Places the legend above the plot area so it never covers the data."""
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=9,
        ncols=3,
    )


def plot_ecg_with_peaks(
    ecg: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    title: str = "",
    hr: pd.DataFrame | None = None,
):
    """
    Plots a short ECG window with the detected QRS peaks marked.

    Args:
        ecg (np.ndarray): 1D ECG signal (whole recording).
        peaks (np.ndarray): Sample indices of the QRS peaks.
        fs (float): Sampling rate in Hz.
        start_s (float): Start of the plotted window in seconds.
        duration_s (float): Length of the plotted window in seconds.
        title (str): Figure title.
        hr (pd.DataFrame, optional): Output of `estimate_hr`. If given,
            invalid segments inside the window are shown as gray bands.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    start, end = _slice_window(len(ecg), fs, start_s, duration_s)
    z = _zscore(ecg[start:end])
    t = np.arange(start, end) / fs
    window_peaks = peaks[(peaks >= start) & (peaks < end)]

    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    if hr is not None:
        in_window = (hr["end_s"] > t[0]) & (hr["start_s"] < t[-1])
        _shade_invalid_segments(ax, hr[in_window])
    ax.plot(t, z, color=SIGNAL_COLOR, linewidth=1.3, label="ECG")
    ax.plot(
        window_peaks / fs,
        z[window_peaks - start],
        linestyle="none",
        marker="o",
        markersize=8,
        markerfacecolor=PEAK_COLOR,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"Detected QRS ({len(window_peaks)})",
    )
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (z-score)")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, loc="left")
    _add_legend(ax)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_hr_curve(
    hr: pd.DataFrame,
    start_s: float | None = None,
    end_s: float | None = None,
    title: str = "",
):
    """
    Plots HR over time, one point per segment. Invalid segments (HR = NaN)
    are shown as gray bands and break the line.

    Args:
        hr (pd.DataFrame): Output of `estimate_hr` / `compute_hr_segments`.
        start_s (float, optional): Start of the plotted range in seconds.
        end_s (float, optional): End of the plotted range in seconds.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    if start_s is not None:
        hr = hr[hr["end_s"] > start_s]
    if end_s is not None:
        hr = hr[hr["start_s"] < end_s]
    centers = (hr["start_s"] + hr["end_s"]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    _shade_invalid_segments(ax, hr)
    ax.plot(
        centers,
        hr["hr_bpm"],
        color=SIGNAL_COLOR,
        linewidth=2,
        marker="o",
        markersize=4 if len(hr) <= 150 else 0,
        label="HR (Pan-Tompkins)",
    )
    if len(hr):
        ax.set_xlim(hr["start_s"].iloc[0], hr["end_s"].iloc[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, loc="left")
    _add_legend(ax)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_pan_tompkins_stages(
    stages: dict,
    fs: float,
    start_s: float = 0.0,
    duration_s: float = 5.0,
    title: str = "",
):
    """
    Plots the intermediate signals of the Pan-Tompkins preprocessing chain
    as stacked panels with a common time axis.

    Args:
        stages (dict): Output of `pan_tompkins_stages`.
        fs (float): Sampling rate in Hz.
        start_s (float): Start of the plotted window in seconds.
        duration_s (float): Length of the plotted window in seconds.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    labels = {
        "input": "1. Input ECG",
        "bandpass": "2. Bandpass 5-15 Hz",
        "derivative": "3. Derivative",
        "squared": "4. Squaring",
        "integrated": "5. Moving window integration",
    }
    start, end = _slice_window(len(stages["input"]), fs, start_s, duration_s)
    t = np.arange(start, end) / fs

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(10, 7), sharex=True, facecolor=SURFACE
    )
    for ax, (key, label) in zip(axes, labels.items()):
        ax.plot(t, stages[key][start:end], color=SIGNAL_COLOR, linewidth=1.3)
        ax.set_title(label, color=TEXT_COLOR, fontsize=10, loc="left")
        # Amplitudes are arbitrary (files differ by orders of magnitude)
        ax.set_yticks([])
        _style_axis(ax)
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, color=TEXT_COLOR, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the Pan-Tompkins slide plots for one EDF file."
    )
    parser.add_argument("edf_path", type=Path, nargs="?", default=DEFAULT_EDF)
    parser.add_argument(
        "--start-s", type=float, default=0.0, help="Start of the ECG window."
    )
    parser.add_argument(
        "--duration-s", type=float, default=10.0, help="Length of the ECG window."
    )
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=None)
    parser.add_argument(
        "--hr-start-s", type=float, default=None, help="Start of the HR curve."
    )
    parser.add_argument(
        "--hr-end-s", type=float, default=None, help="End of the HR curve."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    ecg, _, metadata = read_edf_signals(str(args.edf_path))
    if ecg is None:
        sys.exit(f"No ECG channel found in {args.edf_path}")
    fs = metadata["sampling_rate"]
    name = args.edf_path.stem

    hr, peaks = estimate_hr(ecg, fs, args.segment_s, args.step_s)

    # The stages are computed only on the plotted window plus a margin,
    # long recordings are never processed or plotted as a whole here.
    margin_s = 5.0
    stage_start, stage_end = _slice_window(
        len(ecg),
        fs,
        max(args.start_s - margin_s, 0.0),
        args.duration_s + 2 * margin_s,
    )
    stages = pan_tompkins_stages(ecg[stage_start:stage_end], fs)
    stages_offset_s = args.start_s - stage_start / fs

    figures = {
        "ecg_peaks": plot_ecg_with_peaks(
            ecg,
            peaks,
            fs,
            args.start_s,
            args.duration_s,
            title=f"{name}: ECG with detected QRS peaks",
            hr=hr,
        ),
        "hr": plot_hr_curve(
            hr,
            args.hr_start_s,
            args.hr_end_s,
            title=f"{name}: heart rate in {args.segment_s:g} s segments",
        ),
        "pt_stages": plot_pan_tompkins_stages(
            stages,
            fs,
            stages_offset_s,
            min(args.duration_s, 5.0),
            title=f"{name}: Pan-Tompkins stages",
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        out_path = args.output_dir / f"{name}_{key}.png"
        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
        print(f"Saved {out_path}")

    if args.show:
        plt.show()
