"""
Histogram of 'feasible_bpm' (NaN dropped) over all parquet files of the
database, train and test split together. Bottom panel: same counts on a
log scale, so the rare low/high HR tails are visible.

Usage:
    python prototyping_dan/plot_feasible_hist.py [--parquet_dir ...] [--output ...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent

BAR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_dir",
        default=str(root_dir / "data" / "hr_parts" / "seg10s_step5s"),
    )
    parser.add_argument(
        "--output",
        default=str(root_dir / "data" / "fold_analysis"
                    / "feasible_bpm_histogram.png"),
    )
    args = parser.parse_args()

    files = sorted(Path(args.parquet_dir).glob("*/*.parquet"))
    y = pd.concat(
        pd.read_parquet(p, columns=["feasible_bpm"])["feasible_bpm"]
        for p in files
    ).dropna().to_numpy()

    edges = np.arange(np.floor(y.min()), np.ceil(y.max()) + 1, 1.0)
    counts, _ = np.histogram(y, bins=edges)

    plt.rcParams.update({"font.size": 10, "text.color": TEXT,
                         "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
                         "ytick.color": TEXT_2})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2]})
    for ax, log in zip(axes, (False, True)):
        ax.set_facecolor(SURFACE)
        # 1-bpm bars with a thin surface gap between neighbours
        ax.bar(edges[:-1], counts, width=0.85, align="edge", color=BAR,
               linewidth=0)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("Segments (log scale)")
        else:
            ax.set_ylabel("Segments per 1 bpm")
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(length=0)
    axes[1].set_xlabel("feasible_bpm (bpm)")

    p = np.percentile(y, [1, 50, 99])
    share = lambda m: f"{m.mean():.1%}"  # noqa: E731
    fig.suptitle("Distribution of feasible_bpm", x=0.06, ha="left",
                 fontsize=13, fontweight="bold")
    axes[0].set_title(
        f"{len(y):,} segments from {len(files)} files (train + test), NaN"
        f" dropped\nmedian {p[1]:.1f} bpm · 1–99 % range"
        f" {p[0]:.1f}–{p[2]:.1f} bpm · <60 bpm {share(y < 60)} ·"
        f" >100 bpm {share(y > 100)}",
        loc="left", fontsize=9, color=TEXT_2,
    )
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(f"{len(y)} values, saved to {args.output}")


if __name__ == "__main__":
    main()
