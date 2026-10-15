import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from databasing.pan_tompkins.detection import (
    pan_tompkins_stages,
    pan_tompkins_threshold,
)
from databasing.pan_tompkins.heart_rate import estimate_hr
from utils.read_edf import read_edf_signals

DEFAULT_EDF = (
    root_dir
    / "data"
    / "harmonized_ecg_ppg_database_filtered"
    / "wrist-finger-ppg"
    / "01_4.edf"
)
DEFAULT_OUTPUT_DIR = root_dir / "data" / "plots"
PLOT_KINDS = ["ecg_peaks", "hr", "pt_stages"]

# Default ECG window and annotations (start_s, end_s, text) for the files
# used on the slides, keyed by the file name without extension.
SLIDE_PRESETS = {
    "P10_necklace_4": {
        "start_s": 44455.0,
        "duration_s": 45.0,
        "annotations": [
            (44455.0, 44470.0, "dead signal"),
            (44470.0, 44480.0, "transient"),
            (44480.0, 44500.0, "valid ECG"),
        ],
    },
}

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
    start = min(max(round(start_s * fs), 0), n_samples)
    end = min(start + round(duration_s * fs), n_samples)
    return start, end


def _merge_runs(segments: pd.DataFrame) -> list[tuple[float, float]]:
    """Merges touching segments into (start_s, end_s) runs."""
    runs = []
    for seg_start, seg_end in zip(
        segments["start_s"], segments["end_s"]
    ):
        if runs and seg_start <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], seg_end)
        else:
            runs.append([seg_start, seg_end])
    return [(run_start, run_end) for run_start, run_end in runs]


def _shade_invalid_segments(ax, hr: pd.DataFrame, x_scale: float = 1.0):
    """Draws gray bands over invalid segments, neighbours are merged."""
    for i, (band_start, band_end) in enumerate(
        _merge_runs(hr[~hr["valid"]])
    ):
        ax.axvspan(
            band_start / x_scale,
            band_end / x_scale,
            color=INVALID_COLOR,
            alpha=0.2,
            linewidth=0,
            label="Invalid segment (HR = NaN)" if i == 0 else None,
        )


def _format_duration(seconds: float) -> str:
    """Formats a duration as a short approximate text."""
    if seconds >= 3600:
        return f"~{seconds / 3600:.0f} h"
    if seconds >= 60:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds:.0f} s"


def _add_title_and_legend(ax, title: str, ncols: int = 4):
    """Puts the title on top and the legend in a row between it and the plot."""
    ax.set_title(
        title, color=TEXT_COLOR, fontsize=12, loc="left", pad=28
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=9,
        ncols=ncols,
        borderaxespad=0.2,
        handletextpad=0.5,
        columnspacing=1.5,
    )


def plot_ecg_with_peaks(
    ecg: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    title: str = "",
    hr: pd.DataFrame | None = None,
    annotations: list[tuple[float, float, str]] | None = None,
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
            segment borders are drawn, invalid segments are shown as gray
            bands and the peaks outside valid segments are marked as
            rejected.
        annotations (list, optional): (start_s, end_s, text) labels written
            above the given time ranges.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    start, end = _slice_window(len(ecg), fs, start_s, duration_s)
    z = _zscore(ecg[start:end])
    t = np.arange(start, end) / fs
    window_peaks = peaks[(peaks >= start) & (peaks < end)]
    accepted = np.ones(len(window_peaks), dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor=SURFACE)
    if hr is not None:
        in_window = (hr["end_s"] > t[0]) & (hr["start_s"] < t[-1])
        segments = hr[in_window]
        _shade_invalid_segments(ax, segments)

        borders = np.union1d(segments["start_s"], segments["end_s"])
        for i, border in enumerate(borders):
            ax.axvline(
                border,
                color=MUTED_COLOR,
                linewidth=0.8,
                linestyle=(0, (4, 3)),
                label="Segment border" if i == 0 else None,
            )

        accepted = np.zeros(len(window_peaks), dtype=bool)
        valid = segments[segments["valid"]]
        for seg_start, seg_end in zip(valid["start"], valid["end"]):
            accepted |= (window_peaks >= seg_start) & (
                window_peaks < seg_end
            )

    ax.plot(t, z, color=SIGNAL_COLOR, linewidth=1.3, label="ECG")
    rejected_peaks = window_peaks[~accepted]
    if len(rejected_peaks):
        ax.plot(
            rejected_peaks / fs,
            z[rejected_peaks - start],
            linestyle="none",
            marker="x",
            markersize=7,
            markeredgecolor=MUTED_COLOR,
            markeredgewidth=1.6,
            label=f"Rejected peak ({len(rejected_peaks)})",
        )
    accepted_peaks = window_peaks[accepted]
    ax.plot(
        accepted_peaks / fs,
        z[accepted_peaks - start],
        linestyle="none",
        marker="o",
        markersize=8,
        markerfacecolor=PEAK_COLOR,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"Detected QRS ({len(accepted_peaks)})",
    )

    if annotations:
        # Headroom so the labels do not sit on the signal
        y_low, y_high = ax.get_ylim()
        ax.set_ylim(y_low, y_high + 0.12 * (y_high - y_low))
        for label_start, label_end, text in annotations:
            ax.text(
                (max(label_start, t[0]) + min(label_end, t[-1])) / 2.0,
                0.97,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=TEXT_COLOR,
            )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ECG (z-score)")
    _add_title_and_legend(ax, title, ncols=5)
    _style_axis(ax)
    if hr is not None:
        # Segment borders replace the vertical grid lines
        ax.grid(False, axis="x")
    fig.tight_layout()
    return fig


def plot_hr_curve(
    hr: pd.DataFrame,
    start_s: float | None = None,
    end_s: float | None = None,
    title: str = "",
    min_y_range_bpm: float = 30.0,
    hours_from_s: float = 3600.0,
):
    """
    Plots HR over time, one point per segment. Invalid segments (HR = NaN)
    are shown as gray bands and break the line. Long flat (dead) stretches
    are labelled with their duration.

    Args:
        hr (pd.DataFrame): Output of `estimate_hr` / `compute_hr_segments`.
        start_s (float, optional): Start of the plotted range in seconds.
        end_s (float, optional): End of the plotted range in seconds.
        title (str): Figure title.
        min_y_range_bpm (float): Minimum span of the y axis in bpm, so
            small HR fluctuations are not visually exaggerated.
        hours_from_s (float): Plotted ranges at least this long get the
            time axis in hours instead of seconds.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Dead stretches are measured on the whole recording, before cropping
    is_flat = hr["invalid_reason"].str.startswith("flat")
    flat_runs = _merge_runs(hr[is_flat])

    if start_s is not None:
        hr = hr[hr["end_s"] > start_s]
    if end_s is not None:
        hr = hr[hr["start_s"] < end_s]

    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor=SURFACE)
    if hr.empty:
        return fig
    x_min, x_max = hr["start_s"].iloc[0], hr["end_s"].iloc[-1]
    in_hours = (x_max - x_min) >= hours_from_s
    x_scale = 3600.0 if in_hours else 1.0
    centers = (hr["start_s"] + hr["end_s"]) / 2.0 / x_scale

    _shade_invalid_segments(ax, hr, x_scale)
    few_points = len(hr) <= 150
    ax.plot(
        centers,
        hr["hr_bpm"],
        color=SIGNAL_COLOR,
        linewidth=2 if few_points else 1.2,
        marker="o",
        markersize=4 if few_points else 0,
        label="HR (Pan-Tompkins)",
    )
    ax.set_xlim(x_min / x_scale, x_max / x_scale)

    y_low, y_high = ax.get_ylim()
    if y_high - y_low < min_y_range_bpm:
        y_center = (y_low + y_high) / 2.0
        ax.set_ylim(
            y_center - min_y_range_bpm / 2.0,
            y_center + min_y_range_bpm / 2.0,
        )

    # Label the dead stretches that are wide enough to hold the text
    for run_start, run_end in flat_runs:
        visible_start, visible_end = (
            max(run_start, x_min),
            min(run_end, x_max),
        )
        if visible_end - visible_start < 0.25 * (x_max - x_min):
            continue
        ax.text(
            (visible_start + visible_end) / 2.0 / x_scale,
            0.5,
            f"no ECG signal ({_format_duration(run_end - run_start)})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=11,
            color=MUTED_COLOR,
        )

    ax.set_xlabel(
        "Time from recording start (h)" if in_hours else "Time (s)"
    )
    ax.set_ylabel("Heart rate (bpm)")
    _add_title_and_legend(ax, title)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_pan_tompkins_stages(
    stages: dict,
    fs: float,
    start_s: float = 0.0,
    duration_s: float = 5.0,
    title: str = "",
    threshold: np.ndarray | None = None,
    peaks: np.ndarray | None = None,
):
    """
    Plots the intermediate signals of the Pan-Tompkins preprocessing chain
    as stacked panels with a common time axis. If `threshold` is given, a
    last panel shows the integrated signal with the adaptive threshold and
    the detected peaks.

    Args:
        stages (dict): Output of `pan_tompkins_stages`.
        fs (float): Sampling rate in Hz.
        start_s (float): Start of the plotted window in seconds.
        duration_s (float): Length of the plotted window in seconds.
        title (str): Figure title.
        threshold (np.ndarray, optional): Threshold for every sample, as
            returned by `pan_tompkins_threshold`.
        peaks (np.ndarray, optional): Sample indices of the peaks above
            the threshold.

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
    start, end = _slice_window(
        len(stages["input"]), fs, start_s, duration_s
    )
    t = np.arange(start, end) / fs
    n_panels = len(labels) + (threshold is not None)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, 1.4 * n_panels),
        sharex=True,
        facecolor=SURFACE,
    )
    for ax, (key, label) in zip(axes, labels.items()):
        ax.plot(
            t, stages[key][start:end], color=SIGNAL_COLOR, linewidth=1.3
        )
        ax.set_title(label, color=TEXT_COLOR, fontsize=10, loc="left")

    if threshold is not None:
        ax = axes[-1]
        integrated = stages["integrated"]
        ax.plot(
            t,
            integrated[start:end],
            color=SIGNAL_COLOR,
            linewidth=1.3,
            label="Integrated signal",
        )
        ax.plot(
            t,
            threshold[start:end],
            color=MUTED_COLOR,
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            label="Adaptive threshold",
        )
        if peaks is not None:
            window_peaks = peaks[(peaks >= start) & (peaks < end)]
            ax.plot(
                window_peaks / fs,
                integrated[window_peaks],
                linestyle="none",
                marker="o",
                markersize=8,
                markerfacecolor=PEAK_COLOR,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label="Detected peak",
            )
        ax.set_title(
            "6. Adaptive threshold and peak detection",
            color=TEXT_COLOR,
            fontsize=10,
            loc="left",
        )
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=9,
            ncols=3,
            borderaxespad=0.1,
        )

    for ax in axes:
        # Amplitudes are arbitrary (files differ by orders of magnitude)
        ax.set_yticks([])
        _style_axis(ax)
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        title, color=TEXT_COLOR, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the Pan-Tompkins slide plots for one EDF file."
    )
    parser.add_argument(
        "edf_path", type=Path, nargs="?", default=DEFAULT_EDF
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=PLOT_KINDS,
        default=PLOT_KINDS,
        help="Which plots to create.",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Start of the ECG window (default: slide preset or 0).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Length of the ECG window (default: slide preset or 10).",
    )
    parser.add_argument("--segment-s", type=float, default=10.0)
    parser.add_argument("--step-s", type=float, default=None)
    parser.add_argument(
        "--hr-start-s",
        type=float,
        default=None,
        help="Start of the HR curve.",
    )
    parser.add_argument(
        "--hr-end-s",
        type=float,
        default=None,
        help="End of the HR curve.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    ecg, _, metadata = read_edf_signals(str(args.edf_path))
    if ecg is None:
        sys.exit(f"No ECG channel found in {args.edf_path}")
    fs = metadata["sampling_rate"]
    name = args.edf_path.stem

    preset = SLIDE_PRESETS.get(name, {})
    start_s = args.start_s
    duration_s = args.duration_s
    annotations = None
    if start_s is None and duration_s is None:
        # The preset annotations only fit the preset window
        annotations = preset.get("annotations")
    if start_s is None:
        start_s = preset.get("start_s", 0.0)
    if duration_s is None:
        duration_s = preset.get("duration_s", 10.0)

    hr, peaks = estimate_hr(ecg, fs, args.segment_s, args.step_s)

    figures = {}
    if "ecg_peaks" in args.plots:
        figures["ecg_peaks"] = plot_ecg_with_peaks(
            ecg,
            peaks,
            fs,
            start_s,
            duration_s,
            title=f"{name}: ECG with detected QRS peaks",
            hr=hr,
            annotations=annotations,
        )
    if "hr" in args.plots:
        figures["hr"] = plot_hr_curve(
            hr,
            args.hr_start_s,
            args.hr_end_s,
            title=f"{name}: heart rate in {args.segment_s:g} s segments",
        )
    if "pt_stages" in args.plots:
        # The stages are computed only on the plotted window plus a margin,
        # long recordings are never processed or plotted as a whole here.
        margin_s = 5.0
        stage_start, stage_end = _slice_window(
            len(ecg),
            fs,
            max(start_s - margin_s, 0.0),
            duration_s + 2 * margin_s,
        )
        stages = pan_tompkins_stages(ecg[stage_start:stage_end], fs)
        threshold, stage_peaks = pan_tompkins_threshold(
            stages["integrated"], fs
        )
        figures["pt_stages"] = plot_pan_tompkins_stages(
            stages,
            fs,
            start_s - stage_start / fs,
            min(duration_s, 5.0),
            title=f"{name}: Pan-Tompkins stages",
            threshold=threshold,
            peaks=stage_peaks,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        out_path = args.output_dir / f"{name}_{key}.png"
        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
        print(f"Saved {out_path}")

    if args.show:
        plt.show()
