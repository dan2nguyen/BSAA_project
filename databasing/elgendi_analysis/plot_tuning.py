"""
Plots of the Elgendi tuning stages (see tune_elgendi.py).

Run from the repository root after the stages:
    uv run python databasing/elgendi_analysis/plot_tuning.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from databasing.elgendi_analysis.tune_elgendi import (
    GOOD_ERROR_BPM,
    QUALITY_FLAGS,
    REASON_NAMES,
    VALID,
)

DEFAULT_TUNING = root_dir / "data" / "results" / "elgendi_tuning"

SURFACE = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
BASELINE_COLOR = "#898781"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#4a3aa7",
]

GROUP_ORDER = [
    "earring",
    "necklace",
    "ring",
    "watch",
    "wrist-finger",
    "walk",
    "run",
    "low_resistance_bike",
    "high_resistance_bike",
]
HR_BINS = [0, 60, 80, 100, 120, 140, 230]


def _style(ax):
    """Recessive grid and axes."""
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


def _title(fig, text, subtitle="", layout=True):
    """
    Figure title (left) with an optional muted subtitle above the plots;
    with `layout`, the axes are laid out below them.
    """
    height = fig.get_figheight()
    fig.text(
        0.01,
        1 - 0.12 / height,
        text,
        ha="left",
        va="top",
        fontsize=13,
        color=TEXT_COLOR,
    )
    if subtitle:
        fig.text(
            0.01,
            1 - 0.42 / height,
            subtitle,
            ha="left",
            va="top",
            fontsize=9,
            color=MUTED_COLOR,
        )
    if layout:
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def structure_label(row) -> str:
    """Readable name of a structure variant."""
    offset = (
        "global offset"
        if pd.isna(row["offset_window_s"])
        else f"local offset {row['offset_window_s']:g} s"
    )
    lowpass = (
        "no extra LP"
        if pd.isna(row["lowpass_hz"])
        else f"LP {row['lowpass_hz']:g} Hz"
    )
    polarity = "inverted" if row["invert"] else "normal"
    return f"{offset} | {lowpass} | {polarity}"


# ---------------------------------------------------------------------
# Structure stage
# ---------------------------------------------------------------------


def plot_structure(scores: pd.DataFrame, out_dir: Path, role: str):
    """
    Target score of the structure variants: bars per offset mode and
    low-pass, one panel per polarity; and a variant x device heatmap.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE
    )
    offsets = [np.nan, 10.0, 60.0]
    offset_names = ["global offset", "local 10 s", "local 60 s"]
    lowpasses = [np.nan, 5.0, 3.0]
    lowpass_names = ["no extra LP", "LP 5 Hz", "LP 3 Hz"]
    width = 0.26
    for ax, invert in zip(axes, [False, True]):
        part = scores[scores["invert"] == invert]
        for i, (offset, name) in enumerate(zip(offsets, offset_names)):
            values = []
            for lowpass in lowpasses:
                match = part[
                    (
                        part["offset_window_s"].isna()
                        if np.isnan(offset)
                        else part["offset_window_s"] == offset
                    )
                    & (
                        part["lowpass_hz"].isna()
                        if np.isnan(lowpass)
                        else part["lowpass_hz"] == lowpass
                    )
                ]
                values.append(match["macro_good_%"].iloc[0])
            x = np.arange(len(lowpasses)) + (i - 1) * width
            bars = ax.bar(
                x,
                values,
                width * 0.92,
                color=SERIES[i],
                label=name,
                edgecolor=SURFACE,
                linewidth=1,
            )
            ax.bar_label(
                bars, fmt="%.1f", fontsize=7, color=MUTED_COLOR
            )
        ax.set_xticks(np.arange(len(lowpasses)), lowpass_names)
        ax.set_title(
            "inverted PPG" if invert else "normal PPG",
            fontsize=10,
            color=TEXT_COLOR,
            loc="left",
        )
        _style(ax)
    axes[0].set_ylabel("good windows, subject mean (%)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    _title(
        fig,
        "Structure: offset mode, extra low-pass and polarity",
        f"{role.upper()} subjects. Good = valid PPG HR with |error| <"
        f" {GOOD_ERROR_BPM:g} bpm, share of windows with valid ECG and"
        " non-flat PPG.",
    )
    _save(fig, out_dir / f"structure_overview_{role}.png")

    _heatmap_by_group(
        scores,
        out_dir / f"structure_by_device_{role}.png",
        "Structure variants per device / activity",
        f"{role.upper()} subjects, good windows (%)",
        scores.apply(structure_label, axis=1),
    )


def _heatmap_by_group(scores, path, title, subtitle, labels):
    """Variant x device/activity heatmap of good %."""
    columns = [
        c
        for c in (
            [f"good_%_{g}" for g in GROUP_ORDER[:4]]
            + ["good_%_wrist"]
            + [f"good_%_exercise_{g}" for g in GROUP_ORDER[5:]]
        )
        if c in scores.columns
    ]
    names = [
        c.replace("good_%_", "").replace("exercise_", "ex. ")
        for c in columns
    ]
    names = ["wrist-finger" if n == "wrist" else n for n in names]
    data = scores[columns].to_numpy()
    order = np.argsort(-scores["macro_good_%"].to_numpy())
    data, labels = data[order], np.asarray(labels)[order]

    fig, ax = plt.subplots(
        figsize=(11, 0.32 * len(labels) + 1.8), facecolor=SURFACE
    )
    image = ax.imshow(
        data, cmap="Blues", aspect="auto", vmin=0, vmax=100
    )
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 60 else TEXT_COLOR,
            )
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.ax.tick_params(labelsize=8, colors=MUTED_COLOR)
    bar.set_label("good windows (%)", color=MUTED_COLOR, fontsize=8)
    _title(fig, title, subtitle + "; rows sorted by the subject mean")
    _save(fig, path)


# ---------------------------------------------------------------------
# Parameter stage
# ---------------------------------------------------------------------


def plot_parameters(scores: pd.DataFrame, out_dir: Path, role: str):
    """
    Heatmaps beatwindow x beatoffset per peakwindow for the target score,
    coverage and MAE, and per-device curves around the best variant.
    """
    metrics = [
        ("macro_good_%", "good windows, subject mean (%)", "Blues"),
        ("all_coverage_%", "coverage (%)", "Blues"),
        ("all_MAE", "MAE of valid windows (bpm)", "Oranges"),
    ]
    peakwindows = sorted(scores["peakwindow_s"].unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(peakwindows),
        figsize=(4.4 * len(peakwindows), 3.6 * len(metrics)),
        facecolor=SURFACE,
    )
    height = fig.get_figheight()
    fig.subplots_adjust(
        left=0.07,
        right=0.92,
        top=1 - 1.0 / height,
        bottom=0.06,
        hspace=0.75,
        wspace=0.25,
    )
    for r, (metric, label, cmap) in enumerate(metrics):
        vmin, vmax = scores[metric].min(), scores[metric].max()
        for c, pw in enumerate(peakwindows):
            ax = axes[r, c]
            part = scores[scores["peakwindow_s"] == pw].pivot_table(
                index="beatwindow_s",
                columns="beatoffset",
                values=metric,
            )
            image = ax.imshow(
                part.to_numpy(),
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            for (i, j), value in np.ndenumerate(part.to_numpy()):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                    if (value - vmin) / (vmax - vmin + 1e-9) > 0.6
                    else TEXT_COLOR,
                )
            ax.set_xticks(
                range(part.shape[1]), [f"{v:g}" for v in part.columns]
            )
            ax.set_yticks(
                range(part.shape[0]), [f"{v:g}" for v in part.index]
            )
            if r == len(metrics) - 1:
                ax.set_xlabel("beatoffset")
            if c == 0:
                ax.set_ylabel("beatwindow (s)")
            ax.set_title(
                f"{label}\npeakwindow {pw:g} s",
                fontsize=9,
                color=TEXT_COLOR,
                loc="left",
            )
            ax.tick_params(colors=MUTED_COLOR, labelsize=8)
            for side in ax.spines.values():
                side.set_visible(False)
        fig.colorbar(image, ax=list(axes[r]), fraction=0.015, pad=0.01)
    _title(
        fig,
        "Elgendi parameters: beatwindow x beatoffset",
        f"{role.upper()} subjects",
        layout=False,
    )
    _save(fig, out_dir / f"parameters_heatmaps_{role}.png")

    best = scores.sort_values("macro_good_%", ascending=False).iloc[0]
    _parameter_curves(
        scores, best, out_dir / f"parameters_by_device_{role}.png", role
    )


def _parameter_curves(scores, best, path, role):
    """Good % per device when one parameter changes around the best."""
    params = [
        ("beatoffset", "beatoffset"),
        ("beatwindow_s", "beatwindow (s)"),
        ("peakwindow_s", "peakwindow (s)"),
    ]
    groups = [
        ("good_%_earring", "earring"),
        ("good_%_necklace", "necklace"),
        ("good_%_ring", "ring"),
        ("good_%_watch", "watch"),
        ("good_%_wrist", "wrist-finger"),
        ("macro_good_%_wrist-ppg-during-exercise-1.0.0", "exercise"),
    ]
    fig, axes = plt.subplots(
        1, len(params), figsize=(13, 4), sharey=True, facecolor=SURFACE
    )
    for ax, (param, label) in zip(axes, params):
        others = [p for p, _ in params if p != param]
        part = scores
        for other in others:
            part = part[np.isclose(part[other], best[other])]
        part = part.sort_values(param)
        for i, (column, name) in enumerate(groups):
            if column in part:
                ax.plot(
                    part[param],
                    part[column],
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=SERIES[i % len(SERIES)],
                    label=name,
                )
        ax.axvline(
            best[param], color=BASELINE_COLOR, linestyle=(0, (4, 3))
        )
        ax.set_xlabel(label)
        _style(ax)
    axes[0].set_ylabel("good windows (%)")
    axes[-1].legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    _title(
        fig,
        "One parameter at a time, the others at the best values",
        f"{role.upper()} subjects; dashed line = best variant"
        f" ({best['variant']})",
    )
    _save(fig, path)


def plot_tradeoff(
    tuning_dir: Path, out_dir: Path, role: str, marked: dict
):
    """
    Coverage vs reliability (share of valid PPG HR within 5 bpm) of all
    tested variants, with lines of equal good % (= product of both).
    """
    frames = []
    for path in sorted(tuning_dir.glob(f"*/scores_{role}.csv")):
        stage = path.parent.name.split("_pw")[0]
        if stage == "final":
            continue
        frames.append(pd.read_csv(path).assign(stage=stage))
    scores = pd.concat(frames).drop_duplicates("variant")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    coverage = np.linspace(1, 100, 200)
    for good in (20, 30, 40, 50, 60):
        ax.plot(
            coverage,
            100 * good / coverage,
            color=GRID_COLOR,
            linewidth=1,
            zorder=1,
        )
        ax.text(
            100 * good / 99.5,
            99.5,
            f"good {good}%",
            fontsize=7,
            color=MUTED_COLOR,
            ha="center",
            va="bottom",
        )
    for i, (stage, part) in enumerate(
        scores.groupby("stage", sort=False)
    ):
        ax.scatter(
            part["all_coverage_%"],
            part["all_within_5bpm_of_valid_%"],
            s=28,
            color=SERIES[i % len(SERIES)],
            label=stage,
            edgecolors=SURFACE,
            linewidths=0.8,
            zorder=3,
        )
    for name, label in marked.items():
        row = scores[scores["variant"] == name]
        if row.empty:
            continue
        x = row["all_coverage_%"].iloc[0]
        y = row["all_within_5bpm_of_valid_%"].iloc[0]
        ax.scatter(
            x,
            y,
            s=110,
            facecolors="none",
            edgecolors=TEXT_COLOR,
            zorder=4,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_COLOR,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(50, 101)
    ax.set_xlabel("coverage: windows with a valid PPG HR (%)")
    ax.set_ylabel("reliability: valid PPG HR within 5 bpm (%)")
    ax.legend(
        frameon=False, fontsize=8, loc="lower left", title="stage"
    )
    _style(ax)
    _title(
        fig,
        "Coverage vs reliability of all tested variants",
        f"{role.upper()} subjects, all windows pooled; grey lines: equal"
        " share of good windows",
    )
    _save(fig, out_dir / f"tradeoff_{role}.png")


# ---------------------------------------------------------------------
# Final comparison (validation subjects)
# ---------------------------------------------------------------------


def final_frame(stage_dir: Path) -> pd.DataFrame:
    """
    Long table of the final stage: one row per window and variant, with
    validity, error and the window metadata.
    """
    windows = pd.read_parquet(stage_dir / "windows.parquet")
    data = np.load(stage_dir / "results.npz")
    frames = []
    for k, name in enumerate(data["variants"]):
        hr = data["hr"][:, k]
        reason = data["reason"][:, k]
        valid = reason == VALID
        error = np.where(
            valid, hr - windows["hr_bpm"].to_numpy(), np.nan
        )
        frames.append(
            windows.assign(
                variant=str(name),
                ppg_hr=hr,
                reason=[REASON_NAMES[r] for r in reason],
                valid=valid,
                error=error,
                good=valid & (np.abs(error) < GOOD_ERROR_BPM),
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_final(
    stage_dir: Path, out_dir: Path, labels: dict, role="val"
):
    """All comparison plots of the final stage on one subject role."""
    df = final_frame(stage_dir)
    df = df[df["role"] == role]
    df["variant_label"] = df["variant"].map(labels)
    order = list(labels.values())
    colors = dict(zip(order, [BASELINE_COLOR] + SERIES))
    df["group"] = pd.Categorical(df["group"], GROUP_ORDER, ordered=True)

    _final_groups(df, order, colors, out_dir, role)
    _final_hr(df, order, colors, out_dir, role)
    _final_reasons(df, order, out_dir, role)
    _final_flags(df, order, colors, out_dir, role)
    _final_bland_altman(df, order, out_dir, role)
    _final_subjects(df, order, colors, out_dir, role)


def _grouped_bars(ax, table, order, colors, fmt="%.0f"):
    """Bars per category (rows) and variant (columns)."""
    width = 0.8 / len(order)
    x = np.arange(len(table))
    for i, name in enumerate(order):
        bars = ax.bar(
            x + (i - (len(order) - 1) / 2) * width,
            table[name].to_numpy(),
            width * 0.92,
            color=colors[name],
            label=name,
            edgecolor=SURFACE,
            linewidth=1,
        )
        ax.bar_label(bars, fmt=fmt, fontsize=6, color=MUTED_COLOR)
    ax.set_xticks(x, table.index, rotation=25, ha="right")


def _final_groups(df, order, colors, out_dir, role):
    """Good %, coverage and MAE per device / activity."""
    stats = df.groupby(["group", "variant_label"], observed=True).agg(
        good=("good", "mean"),
        coverage=("valid", "mean"),
        mae=("error", lambda e: np.nanmean(np.abs(e))),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), facecolor=SURFACE)
    for ax, (column, label, scale, fmt) in zip(
        axes,
        [
            ("good", "good windows (%)", 100, "%.0f"),
            ("coverage", "coverage (%)", 100, "%.0f"),
            ("mae", "MAE of valid windows (bpm)", 1, "%.1f"),
        ],
    ):
        table = (stats[column] * scale).unstack()[order]
        _grouped_bars(ax, table, order, colors, fmt)
        ax.set_ylabel(label)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, ncols=len(order))
    _title(
        fig,
        "Default vs tuned per device / activity",
        f"{role.upper()} subjects (not used for tuning)",
    )
    _save(fig, out_dir / f"final_by_device_{role}.png")


def _final_hr(df, order, colors, out_dir, role):
    """Coverage and MAE against the ECG heart rate."""
    df = df.assign(hr_bin=pd.cut(df["hr_bpm"], HR_BINS))
    stats = df.groupby(["hr_bin", "variant_label"], observed=True).agg(
        coverage=("valid", "mean"),
        good=("good", "mean"),
        mae=("error", lambda e: np.nanmean(np.abs(e))),
        n=("valid", "size"),
    )
    centers = [f"{b.left:g}-{b.right:g}" for b in stats.index.levels[0]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, (column, label, scale) in zip(
        axes,
        [
            ("good", "good windows (%)", 100),
            ("coverage", "coverage (%)", 100),
            ("mae", "MAE of valid windows (bpm)", 1),
        ],
    ):
        table = (stats[column] * scale).unstack()
        for name in order:
            ax.plot(
                range(len(table)),
                table[name].to_numpy(),
                marker="o",
                markersize=5,
                linewidth=2,
                color=colors[name],
                label=name,
            )
        ax.set_xticks(range(len(table)), centers[: len(table)])
        ax.set_xlabel("ECG heart rate (bpm)")
        ax.set_ylabel(label)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    _title(
        fig,
        "Performance against heart rate",
        f"{role.upper()} subjects, all devices",
    )
    _save(fig, out_dir / f"final_by_hr_{role}.png")


def _final_reasons(df, order, out_dir, role):
    """What happens to the windows: good, valid but wrong, rejected."""
    outcome = np.select(
        [df["good"], df["valid"]],
        ["valid, |error| < 5 bpm", "valid, |error| >= 5 bpm"],
        default=df["reason"],
    )
    df = df.assign(outcome=outcome)
    categories = [
        "valid, |error| < 5 bpm",
        "valid, |error| >= 5 bpm",
        "irregular_rr",
        "too_few_beats",
        "hr_out_of_range",
    ]
    colors = [SERIES[0], SERIES[1], "#c3c2b7", "#898781", "#52514e"]
    table = (
        df.groupby(["group", "variant_label"], observed=True)["outcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
        * 100
    )
    groups = table.index.get_level_values(0).unique()
    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=SURFACE)
    x_labels, x = [], 0
    positions, centers = [], []
    for group in groups:
        group_x = []
        for name in order:
            if (group, name) in table.index:
                positions.append((x, group, name))
                x_labels.append(
                    name.replace("tuned + robust HR", "robust")
                    .replace("(", "")
                    .replace(")", "")
                )
                group_x.append(x)
                x += 1
        centers.append((np.mean(group_x), group))
        x += 0.8
    bottom = np.zeros(len(positions))
    for category, color in zip(categories, colors):
        values = [table.loc[(g, n), category] for _, g, n in positions]
        ax.bar(
            [p[0] for p in positions],
            values,
            0.85,
            bottom=bottom,
            color=color,
            label=category,
            edgecolor=SURFACE,
            linewidth=1,
        )
        bottom += values
    ax.set_xticks(
        [p[0] for p in positions], x_labels, fontsize=7, rotation=90
    )
    for center, group in centers:
        ax.text(
            center,
            -0.52,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=TEXT_COLOR,
        )
    ax.set_ylabel("windows (%)")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncols=5,
        loc="lower left",
        bbox_to_anchor=(0, 1),
    )
    _style(ax)
    _title(
        fig,
        "Outcome of the windows per device: default vs tuned",
        f"{role.upper()} subjects",
        layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.4)
    _save(fig, out_dir / f"final_outcomes_{role}.png")


def _final_flags(df, order, colors, out_dir, role):
    """Good % when each quality flag is True vs False."""
    rows = []
    for flag in QUALITY_FLAGS:
        for value in (True, False):
            part = df[df[flag] == value]
            if part.empty:
                continue
            good = part.groupby("variant_label")["good"].mean() * 100
            signal, name = flag.split("_", 1)
            short = name.split("_")[0].replace("Skewness", "skewness")
            rows.append(
                {
                    "label": f"{signal.upper()} {short}\n= {value}"
                    f"\nn={len(part) // len(order)}",
                    **good.to_dict(),
                }
            )
    table = pd.DataFrame(rows).set_index("label")
    fig, ax = plt.subplots(figsize=(14, 4.8), facecolor=SURFACE)
    _grouped_bars(ax, table, order, colors)
    ax.set_xticklabels(table.index, rotation=0, fontsize=7)
    ax.set_ylabel("good windows (%)")
    ax.legend(
        frameon=False,
        fontsize=9,
        ncols=len(order),
        loc="lower left",
        bbox_to_anchor=(0, 1),
    )
    _style(ax)
    _title(
        fig,
        "Quality flags of Dan's metrics vs detector performance",
        f"{role.upper()} subjects; spike and LF_ratio_motion are never True",
    )
    _save(fig, out_dir / f"final_flags_{role}.png")


def _final_bland_altman(df, order, out_dir, role):
    """Bland-Altman hexbin plots of the valid windows."""
    n_cols = 2 if len(order) > 3 else len(order)
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 4.4 * n_rows),
        sharey=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(order) :]:
        ax.set_visible(False)
    for ax, name in zip(axes, order):
        part = df[(df["variant_label"] == name) & df["valid"]]
        mean = (part["ppg_hr"] + part["hr_bpm"]) / 2
        diff = part["ppg_hr"] - part["hr_bpm"]
        ax.hexbin(
            mean,
            diff,
            gridsize=60,
            bins="log",
            cmap="Blues",
            extent=(30, 200, -100, 60),
            mincnt=1,
        )
        bias, sd = diff.mean(), diff.std()
        for y, style in [
            (bias, "-"),
            (bias + 1.96 * sd, (0, (4, 3))),
            (bias - 1.96 * sd, (0, (4, 3))),
        ]:
            ax.axhline(
                y, color=SERIES[1], linewidth=1.2, linestyle=style
            )
        ax.set_title(
            f"{name}: bias {bias:.2f}, limits {bias - 1.96 * sd:.1f} to"
            f" {bias + 1.96 * sd:.1f} bpm (n={len(part)})",
            fontsize=9,
            color=TEXT_COLOR,
            loc="left",
        )
        ax.set_xlabel("mean of PPG and ECG HR (bpm)")
        _style(ax)
    for ax in axes[::n_cols]:
        ax.set_ylabel("PPG HR - ECG HR (bpm)")
    _title(
        fig,
        "Bland-Altman of the valid windows",
        f"{role.upper()} subjects",
    )
    _save(fig, out_dir / f"final_bland_altman_{role}.png")


def _final_subjects(df, order, colors, out_dir, role):
    """Good % per subject, default vs tuned."""
    table = (
        df.groupby(["base_folder", "subject", "variant_label"])["good"]
        .mean()
        .unstack()[order]
        * 100
    )
    table = table.sort_values(order[0])
    short = {
        "multisite-ppg": "multisite",
        "wrist-finger-ppg": "wrist-finger",
        "wrist-ppg-during-exercise-1.0.0": "exercise",
    }
    labels = [f"{s} ({short.get(f, f)})" for f, s in table.index]
    fig, ax = plt.subplots(
        figsize=(8, 0.32 * len(table) + 1.6), facecolor=SURFACE
    )
    y = np.arange(len(table))
    for name in order:
        ax.scatter(
            table[name],
            y,
            s=40,
            color=colors[name],
            label=name,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=1,
        )
    for i, row in enumerate(table.to_numpy()):
        ax.plot(
            [row.min(), row.max()],
            [i, i],
            color=AXIS_COLOR,
            linewidth=1,
        )
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("good windows (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _title(fig, "Per subject", f"{role.upper()} subjects")
    _save(fig, out_dir / f"final_by_subject_{role}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-dir", type=Path, default=DEFAULT_TUNING
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="name=label pairs for the final variants",
    )
    args = parser.parse_args()
    out_dir = args.tuning_dir / "plots"

    structure = args.tuning_dir / "structure"
    for role in ("tune", "val"):
        path = structure / f"scores_{role}.csv"
        if path.exists():
            plot_structure(pd.read_csv(path), out_dir, role)

    for stage in sorted(args.tuning_dir.glob("parameters_*")):
        for role in ("tune", "val"):
            path = stage / f"scores_{role}.csv"
            if path.exists():
                plot_parameters(
                    pd.read_csv(path), out_dir / stage.name, role
                )

    labels = dict(pair.split("=", 1) for pair in args.labels)
    for role in ("tune", "val"):
        plot_tradeoff(args.tuning_dir, out_dir, role, labels)

    final = args.tuning_dir / "final"
    if (final / "results.npz").exists():
        data = np.load(final / "results.npz")
        labels = dict(pair.split("=", 1) for pair in args.labels)
        names = [str(n) for n in data["variants"]]
        labels = {n: labels.get(n, n) for n in names}
        labels[names[0]] = labels.get(names[0], "default")
        if labels[names[0]] == names[0]:
            labels[names[0]] = "default"
        plot_final(final, out_dir, labels)
