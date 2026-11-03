"""
TRAIN-ONLY tuning of the Elgendi PPG detector.

The TRAIN subjects are split (per subset, fixed seed) into a tuning part
and a validation part. Detector variants are compared on the windows with
a valid ECG reference and non-flat PPG. Per-file results of every stage
are stored in data/results/elgendi_tuning/<stage>/parts/, so a stage can
be interrupted and resumed. Test files are never opened.

Run from the repository root:
    uv run python databasing/elgendi_analysis/tune_elgendi.py --stage structure
"""

import argparse
import csv
import itertools
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
from scipy.ndimage import uniform_filter1d

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from databasing.pan_tompkins.heart_rate import (
    HR_RANGE_BPM,
    MAX_RR_CV,
    MIN_BEATS,
)
from utils.read_edf import read_edf_signals

DEFAULT_DB = root_dir / "data" / "hr_database_seg10s_step5s.parquet"
DEFAULT_EDF = root_dir / "data" / "harmonized_ecg_ppg_database_filtered"
DEFAULT_MANIFEST = (
    root_dir / "databasing" / "references" / "split_manifest.csv"
)
DEFAULT_OUTPUT = root_dir / "data" / "results" / "elgendi_tuning"

QUALITY_FLAGS = [
    f"{signal}_{name}"
    for signal in ("ecg", "ppg")
    for name in (
        "InBand_power_ratio_out",
        "kurtosis_out",
        "Skewness_out",
    )
]
# Reason codes of the per-window result
VALID, TOO_FEW_BEATS, IRREGULAR_RR, HR_OUT_OF_RANGE = 0, 1, 2, 3
REASON_NAMES = {
    VALID: "valid",
    TOO_FEW_BEATS: "too_few_beats",
    IRREGULAR_RR: "irregular_rr",
    HR_OUT_OF_RANGE: "hr_out_of_range",
}
GOOD_ERROR_BPM = 5.0


@dataclass(frozen=True)
class Variant:
    """
    One detector variant. The defaults are the original prototype.

    offset_window_s: None = the offset is beatoffset * mean of the squared
        signal over the whole recording (original); a number = the mean
        over a moving window of this length (local offset).
    lowpass_hz: optional extra zero-phase low-pass (2nd order Butterworth)
        on top of the 0.5-8 Hz filter of the database.
    invert: detect on the inverted PPG.
    rr_max_dev: None = HR from the mean RR with the rr_cv rule of the
        ECG reference (original); a number = robust HR: RR intervals
        deviating more than this share from the window median are
        dropped, HR is 60 / mean of the kept RR.
    rr_max_rejected: robust HR only; the window is irregular when more
        than this share of its RR intervals is dropped.
    """

    peakwindow_s: float = 0.111
    beatwindow_s: float = 0.667
    beatoffset: float = 0.02
    mindelay_s: float = 0.3
    offset_window_s: float | None = None
    lowpass_hz: float | None = None
    invert: bool = False
    rr_max_dev: float | None = None
    rr_max_rejected: float = 0.34

    def name(self) -> str:
        """Short unique name of the variant."""
        parts = [
            f"pw{self.peakwindow_s:g}",
            f"bw{self.beatwindow_s:g}",
            f"bo{self.beatoffset:g}",
            f"md{self.mindelay_s:g}",
            "offG"
            if self.offset_window_s is None
            else f"off{self.offset_window_s:g}s",
            "lpN"
            if self.lowpass_hz is None
            else f"lp{self.lowpass_hz:g}",
            "inv" if self.invert else "pos",
        ]
        if self.rr_max_dev is not None:
            parts.append(
                f"rob{self.rr_max_dev:g}-{self.rr_max_rejected:g}"
            )
        return "_".join(parts)


# ---------------------------------------------------------------------
# Detector and segment HR
# ---------------------------------------------------------------------


def prepare_signal(ppg: np.ndarray, fs: float, variant: Variant):
    """Applies the optional extra low-pass and inversion."""
    x = np.asarray(ppg, dtype=np.float64)
    if variant.lowpass_hz is not None:
        sos = scipy.signal.butter(
            2, variant.lowpass_hz, btype="low", fs=fs, output="sos"
        )
        x = scipy.signal.sosfiltfilt(sos, x)
    if variant.invert:
        x = -x
    return x


def elgendi_peaks(x: np.ndarray, fs: float, variant: Variant):
    """
    Elgendi-style systolic peak detection, same steps as
    databasing/elgendi/detection.py, with an optional local offset.

    Args:
        x (np.ndarray): Prepared PPG (see `prepare_signal`).
        fs (float): Sampling rate in Hz.
        variant (Variant): Detector parameters.

    Returns:
        np.ndarray: Sample indices of the detected peaks.
    """
    n_short = max(1, round(variant.peakwindow_s * fs))
    n_long = max(n_short + 1, round(variant.beatwindow_s * fs))
    if len(x) < n_long or not np.isfinite(x).all():
        return np.array([], dtype=int)
    enhanced = np.maximum(x, 0.0) ** 2
    if np.max(enhanced) <= 0:
        return np.array([], dtype=int)
    short = uniform_filter1d(enhanced, size=n_short, mode="nearest")
    long = uniform_filter1d(enhanced, size=n_long, mode="nearest")
    if variant.offset_window_s is None:
        offset = variant.beatoffset * float(np.mean(enhanced))
    else:
        n_offset = max(1, round(variant.offset_window_s * fs))
        offset = variant.beatoffset * uniform_filter1d(
            enhanced, size=n_offset, mode="nearest"
        )
    active = short > long + offset
    del short, long, enhanced

    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    beginnings = np.flatnonzero(changes == 1)
    endings = np.flatnonzero(changes == -1)
    keep = (endings - beginnings) >= n_short
    if not keep.any():
        return np.array([], dtype=int)

    # Highest sample of every kept block (first one on ties), vectorized
    samples = np.flatnonzero(active)
    block = np.searchsorted(beginnings, samples, side="right") - 1
    in_kept = keep[block]
    samples, block = samples[in_kept], block[in_kept]
    order = np.lexsort((-x[samples], block))
    samples, block = samples[order], block[order]
    first = np.r_[True, block[1:] != block[:-1]]
    candidates = samples[first]

    gap = max(1, round(fs * variant.mindelay_s))
    chosen = []
    for candidate in candidates:
        if not chosen or candidate - chosen[-1] >= gap:
            chosen.append(candidate)
        elif x[candidate] > x[chosen[-1]]:
            chosen[-1] = candidate
    return np.array(chosen, dtype=int)


def window_hr(
    peaks: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    HR of every window from the peaks inside it, with the same rules as
    `compute_hr_segments` (vectorized with cumulative sums).

    Returns:
        tuple: (hr_raw, reason)
            - hr_raw: 60 / mean RR (NaN if fewer than 2 RR intervals)
            - reason: VALID, TOO_FEW_BEATS, HR_OUT_OF_RANGE or IRREGULAR_RR
    """
    first = np.searchsorted(peaks, starts, side="left")
    last = np.searchsorted(peaks, ends, side="left")
    n_beats = last - first
    rr = np.diff(peaks) / fs
    cs = np.r_[0.0, np.cumsum(rr)]
    cs2 = np.r_[0.0, np.cumsum(rr**2)]

    # Indices clipped for windows after the last peak (they have fewer
    # than MIN_BEATS beats, so their values are not used)
    n_rr = np.maximum(n_beats - 1, 1)
    lo = np.minimum(first, len(cs) - 1)
    hi = np.minimum(np.maximum(last - 1, first), len(cs) - 1)
    mean = (cs[hi] - cs[lo]) / n_rr
    var = (cs2[hi] - cs2[lo]) / n_rr - mean**2
    enough = n_beats >= MIN_BEATS
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = np.where(enough, 60.0 / mean, np.nan)
        cv = np.sqrt(np.maximum(var, 0.0)) / mean

    reason = np.full(len(starts), VALID, dtype=np.int8)
    reason[~enough] = TOO_FEW_BEATS
    out_of_range = enough & ~(
        (hr >= HR_RANGE_BPM[0]) & (hr <= HR_RANGE_BPM[1])
    )
    reason[out_of_range] = HR_OUT_OF_RANGE
    irregular = enough & ~out_of_range & (cv > MAX_RR_CV)
    reason[irregular] = IRREGULAR_RR
    return hr.astype(np.float32), reason


def window_hr_robust(
    peaks: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    fs: float,
    max_dev: float,
    max_rejected: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust HR of every window: RR intervals deviating more than `max_dev`
    from the window median are dropped (missed or extra peaks), HR is
    60 / mean of the kept RR. Same outputs as `window_hr`.
    """
    first = np.searchsorted(peaks, starts, side="left")
    last = np.searchsorted(peaks, ends, side="left")
    hr = np.full(len(starts), np.nan, dtype=np.float32)
    reason = np.full(len(starts), TOO_FEW_BEATS, dtype=np.int8)
    for k, (i, j) in enumerate(zip(first, last)):
        if j - i < MIN_BEATS:
            continue
        rr = np.diff(peaks[i:j]) / fs
        kept = rr[np.abs(rr / np.median(rr) - 1.0) <= max_dev]
        if len(kept) < MIN_BEATS - 1:
            reason[k] = IRREGULAR_RR
            continue
        hr[k] = 60.0 / kept.mean()
        if 1.0 - len(kept) / len(rr) > max_rejected:
            reason[k] = IRREGULAR_RR
        elif not HR_RANGE_BPM[0] <= hr[k] <= HR_RANGE_BPM[1]:
            reason[k] = HR_OUT_OF_RANGE
        else:
            reason[k] = VALID
    return hr, reason


# ---------------------------------------------------------------------
# Per-file stage computation
# ---------------------------------------------------------------------


def evaluate_file(
    edf_path: Path,
    part_path: Path,
    starts: np.ndarray,
    ends: np.ndarray,
    variants: list[Variant],
) -> int:
    """
    Worker task: runs all variants on one recording (read once) and
    stores hr_raw and reason of the given windows for every variant.
    """
    _, ppg, metadata = read_edf_signals(str(edf_path))
    fs = float(metadata["sampling_rate"])
    hr = np.empty((len(starts), len(variants)), dtype=np.float32)
    reason = np.empty((len(starts), len(variants)), dtype=np.int8)

    # Variants sharing the filter and polarity reuse the prepared signal
    def key(k):
        return (
            variants[k].lowpass_hz or 0.0,
            variants[k].invert,
        )

    prepared_key, prepared = None, None
    for k in sorted(range(len(variants)), key=key):
        if key(k) != prepared_key:
            prepared_key = key(k)
            prepared = prepare_signal(ppg, fs, variants[k])
        variant = variants[k]
        peaks = elgendi_peaks(prepared, fs, variant)
        if variant.rr_max_dev is None:
            hr[:, k], reason[:, k] = window_hr(peaks, starts, ends, fs)
        else:
            hr[:, k], reason[:, k] = window_hr_robust(
                peaks,
                starts,
                ends,
                fs,
                variant.rr_max_dev,
                variant.rr_max_rejected,
            )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = part_path.with_name(part_path.name + ".tmp.npz")
    np.savez(tmp_path, hr=hr, reason=reason)
    os.replace(tmp_path, part_path)
    return len(starts)


def subject_split(subjects: pd.DataFrame, val_share: float, seed: int):
    """
    Assigns every TRAIN subject to "tune" or "val", per subset.

    Args:
        subjects (pd.DataFrame): Columns base_folder, subject (unique).
        val_share (float): Share of validation subjects per subset.
        seed (int): Random seed.

    Returns:
        dict: (base_folder, subject) -> "tune" / "val".
    """
    role = {}
    for folder, group in subjects.groupby("base_folder"):
        names = sorted(group["subject"])
        random.Random(f"{seed}-{folder}").shuffle(names)
        n_val = max(1, round(len(names) * val_share))
        for i, name in enumerate(names):
            role[(folder, name)] = "val" if i < n_val else "tune"
    return role


def load_windows(
    db_path: Path, manifest_path: Path, edf_root: Path, max_gb
):
    """
    TRAIN windows with a valid ECG HR and non-flat PPG, with subject,
    device/activity group and the quality flags.
    """
    with open(manifest_path, newline="") as f:
        train = {
            r["file"] for r in csv.DictReader(f) if r["set"] == "train"
        }
    columns = [
        "base_folder",
        "edf_filename",
        "section_id",
        "start",
        "end",
        "hr_bpm",
        "rr_cv",
        *QUALITY_FLAGS,
    ]
    windows = pd.read_parquet(
        db_path,
        columns=columns,
        filters=[
            ("split", "==", "train"),
            ("hr_valid", "==", True),
            ("ppg_is_flat", "==", False),
        ],
    )
    windows["file"] = (
        windows["base_folder"] + "/" + windows["edf_filename"]
    )
    windows = windows[windows["file"].isin(train)]
    sizes = {
        f: (edf_root / f).stat().st_size / 1e9
        for f in windows["file"].unique()
    }
    skipped = sorted(f for f, gb in sizes.items() if gb >= max_gb)
    windows = windows[~windows["file"].isin(skipped)]
    windows = windows.sort_values(["file", "section_id"]).reset_index(
        drop=True
    )
    name = windows["edf_filename"].str.replace(".edf", "", regex=False)
    windows["subject"] = name.str.split("_").str[0]
    windows["group"] = np.select(
        [
            windows["base_folder"] == "multisite-ppg",
            windows["base_folder"].str.startswith("wrist-ppg-during"),
        ],
        [name.str.split("_").str[1], name.str.split("_", n=1).str[1]],
        default="wrist-finger",
    )
    return windows, skipped, sizes


def run_stage(
    windows: pd.DataFrame,
    variants: list[Variant],
    stage_dir: Path,
    edf_root: Path,
    sizes: dict,
    workers: int,
    large_gb: float,
):
    """
    Computes all variants for all files (resumable) and returns the
    stacked hr_raw and reason matrices aligned with `windows`.
    """
    parts_dir = stage_dir / "parts"
    groups = {f: g for f, g in windows.groupby("file", sort=True)}

    def part_of(f):
        return parts_dir / f.replace(".edf", ".npz")

    def is_done(f):
        if not part_of(f).exists():
            return False
        with np.load(part_of(f)) as data:
            return data["hr"].shape == (len(groups[f]), len(variants))

    todo = [f for f in groups if not is_done(f)]
    print(
        f"{len(groups)} files, {len(todo)} to compute,"
        f" {len(variants)} variants",
        flush=True,
    )
    large = [f for f in todo if sizes[f] >= large_gb]
    normal = [f for f in todo if f not in large]
    t_start = time.perf_counter()
    for batch, n_workers in ((normal, workers), (large, 1)):
        if not batch:
            continue
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_file,
                    edf_root / f,
                    part_of(f),
                    groups[f]["start"].to_numpy(),
                    groups[f]["end"].to_numpy(),
                    variants,
                ): f
                for f in batch
            }
            for i, future in enumerate(as_completed(futures), start=1):
                future.result()
                if i % 25 == 0 or i == len(batch):
                    print(
                        f"  {i}/{len(batch)} files,"
                        f" {time.perf_counter() - t_start:.0f} s",
                        flush=True,
                    )

    hr, reason = [], []
    for f in groups:
        with np.load(part_of(f)) as data:
            hr.append(data["hr"])
            reason.append(data["reason"])
    return np.vstack(hr), np.vstack(reason)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def window_scores(windows, hr, reason) -> dict:
    """Per-window booleans and errors for one variant (1D inputs)."""
    valid = reason == VALID
    error = np.where(valid, hr - windows["hr_bpm"].to_numpy(), np.nan)
    good = valid & (np.abs(error) < GOOD_ERROR_BPM)
    return {"valid": valid, "error": error, "good": good}


def group_metrics(df: pd.DataFrame) -> pd.Series:
    """Coverage, accuracy and the target score of a group of windows."""
    valid = df["valid"].to_numpy()
    error = df["error"].to_numpy()[valid]
    return pd.Series(
        {
            "windows": len(df),
            "good_%": 100.0 * df["good"].mean(),
            "coverage_%": 100.0 * valid.mean(),
            "MAE": np.abs(error).mean() if len(error) else np.nan,
            "bias": error.mean() if len(error) else np.nan,
            "within_5bpm_of_valid_%": 100.0
            * np.mean(np.abs(error) < GOOD_ERROR_BPM)
            if len(error)
            else np.nan,
        }
    )


def score_variants(windows, hr, reason, variants, subset_mask):
    """
    Target scores of every variant on a subset of windows: subject-macro
    good %, window-level metrics and good % per subset and device.

    Returns:
        pd.DataFrame: One row per variant.
    """
    rows = []
    base = windows[subset_mask]
    for k, variant in enumerate(variants):
        scores = window_scores(
            base, hr[subset_mask, k], reason[subset_mask, k]
        )
        df = base[["base_folder", "subject", "group"]].assign(**scores)
        per_subject = df.groupby(["base_folder", "subject"])[
            "good"
        ].mean()
        row = {
            "variant": variant.name(),
            **asdict(variant),
            "macro_good_%": 100.0 * per_subject.mean(),
            **group_metrics(df).add_prefix("all_").to_dict(),
        }
        for (folder, group), part in df.groupby(
            ["base_folder", "group"]
        ):
            label = (
                group
                if folder == "multisite-ppg"
                else folder.split("-")[0]
            )
            if folder.startswith("wrist-ppg-during"):
                label = f"exercise_{group}"
            row[f"good_%_{label}"] = 100.0 * part["good"].mean()
        for folder, part in df.groupby("base_folder"):
            subject_mean = part.groupby("subject")["good"].mean().mean()
            row[f"macro_good_%_{folder}"] = 100.0 * subject_mean
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------


def structure_variants() -> list[Variant]:
    """Offset mode x extra low-pass x polarity, default parameters."""
    return [
        Variant(
            offset_window_s=offset, lowpass_hz=lowpass, invert=invert
        )
        for offset, lowpass, invert in itertools.product(
            [None, 10.0, 60.0], [None, 5.0, 3.0], [False, True]
        )
    ]


def offset_variants() -> list[Variant]:
    """Length of the local offset window, default parameters."""
    return [
        Variant(offset_window_s=window)
        for window in [3.0, 5.0, 10.0, 20.0, 30.0]
    ]


def estimator_variants(base: Variant) -> list[Variant]:
    """The base variant with the original and robust HR estimators."""
    fields = asdict(base)
    fields.pop("rr_max_dev")
    fields.pop("rr_max_rejected")
    return [Variant(**fields)] + [
        Variant(**fields, rr_max_dev=dev, rr_max_rejected=rejected)
        for dev, rejected in itertools.product(
            [0.1, 0.15, 0.2, 0.3], [0.2, 0.34, 0.5]
        )
    ]


def parameter_variants(base: Variant) -> list[Variant]:
    """Grid of the Elgendi window lengths and offset around `base`."""
    return [
        Variant(
            peakwindow_s=pw,
            beatwindow_s=bw,
            beatoffset=bo,
            mindelay_s=base.mindelay_s,
            offset_window_s=base.offset_window_s,
            lowpass_hz=base.lowpass_hz,
            invert=base.invert,
        )
        for pw, bw, bo in itertools.product(
            [0.07, 0.111, 0.15],
            [0.4, 0.5, 0.667, 0.8],
            [0.0, 0.01, 0.02, 0.05, 0.1],
        )
    ]


def variant_from_name(name: str) -> Variant:
    """Inverse of Variant.name()."""
    tokens = name.split("_")
    return Variant(
        peakwindow_s=float(tokens[0][2:]),
        beatwindow_s=float(tokens[1][2:]),
        beatoffset=float(tokens[2][2:]),
        mindelay_s=float(tokens[3][2:]),
        offset_window_s=None
        if tokens[4] == "offG"
        else float(tokens[4][3:-1]),
        lowpass_hz=None if tokens[5] == "lpN" else float(tokens[5][2:]),
        invert=tokens[6] == "inv",
        **(
            {
                "rr_max_dev": float(tokens[7][3:].split("-")[0]),
                "rr_max_rejected": float(tokens[7].split("-")[1]),
            }
            if len(tokens) > 7
            else {}
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "structure",
            "offset",
            "parameters",
            "estimator",
            "final",
        ],
        help="structure: offset/filter/polarity; offset: length of the"
        " local offset window; parameters: window and"
        " offset grid around --base; final: default vs --variants on"
        " the validation subjects (all windows saved).",
    )
    parser.add_argument(
        "--base",
        default=Variant().name(),
        help="Variant name the parameter grid is built around.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Variant names for the final stage (default is added).",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--edf-root", type=Path, default=DEFAULT_EDF)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--large-file-gb", type=float, default=0.1)
    parser.add_argument("--max-file-gb", type=float, default=1.0)
    args = parser.parse_args()

    windows, skipped, sizes = load_windows(
        args.database, args.manifest, args.edf_root, args.max_file_gb
    )
    role = subject_split(
        windows[["base_folder", "subject"]].drop_duplicates(),
        args.val_share,
        args.seed,
    )
    windows["role"] = [
        role[(f, s)]
        for f, s in zip(windows["base_folder"], windows["subject"])
    ]
    print(
        f"{len(windows)} windows (valid ECG, non-flat PPG),"
        f" skipped files: {skipped}"
    )
    print(
        windows.groupby(["base_folder", "role"])["subject"]
        .nunique()
        .unstack()
        .to_string()
    )

    if args.stage == "structure":
        variants = structure_variants()
        stage_dir = args.output / "structure"
    elif args.stage == "offset":
        variants = offset_variants()
        stage_dir = args.output / "offset"
    elif args.stage == "estimator":
        variants = estimator_variants(variant_from_name(args.base))
        stage_dir = args.output / f"estimator_{args.base}"
    elif args.stage == "parameters":
        variants = parameter_variants(variant_from_name(args.base))
        stage_dir = args.output / f"parameters_{args.base}"
    else:
        names = [Variant().name()] + [
            v for v in args.variants if v != Variant().name()
        ]
        variants = [variant_from_name(n) for n in names]
        stage_dir = args.output / "final"

    hr, reason = run_stage(
        windows,
        variants,
        stage_dir,
        args.edf_root,
        sizes,
        args.workers,
        args.large_file_gb,
    )
    for role_name in ("tune", "val"):
        mask = (windows["role"] == role_name).to_numpy()
        scores = score_variants(windows, hr, reason, variants, mask)
        scores.round(3).to_csv(
            stage_dir / f"scores_{role_name}.csv", index=False
        )
    windows.to_parquet(stage_dir / "windows.parquet", index=False)
    np.savez(
        stage_dir / "results.npz",
        hr=hr,
        reason=reason,
        variants=np.array([v.name() for v in variants]),
    )
    best = pd.read_csv(stage_dir / "scores_tune.csv").sort_values(
        "macro_good_%", ascending=False
    )
    pd.set_option("display.width", 220)
    show = [
        "variant",
        "macro_good_%",
        "all_good_%",
        "all_coverage_%",
        "all_MAE",
    ]
    print("\nBest on TUNE subjects:")
    print(best[show].head(10).to_string(index=False))
