"""
Plots where the CNN makes errors, from the cross-validation (out-of-fold)
predictions on the training split.

Input:  oof_predictions_dilated_cnn.csv (written by prototyping_dan/training.py)
Output: PNG figures + summary tables in --output-dir

Figures:
  1_error_by_site.png       absolute error per sensor site / dataset
  2_error_by_hr.png         MAE per reference-HR range
  3_pred_vs_true.png        predicted vs reference HR (density)
  4_bland_altman.png        Bland-Altman plot (bias, 95 % limits of agreement)
  5_error_by_subject.png    MAE per subject

Usage:
  uv run python prototyping_dan/plot_cnn_validation.py \
      data/training_out_cnn/dilated_cnn/oof_predictions_dilated_cnn.csv \
      --output-dir data/results/cnn_validation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colours (reference data-viz palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})

DATASET_LABEL = {
    "multisite-ppg": "Multisite",
    "wrist-finger-ppg": "Wrist-finger",
    "wrist-ppg-during-exercise-1.0.0": "Exercise",
}


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["error"] = d["predicted_hr"] - d["target_hr"]
    d["abs_error"] = d["error"].abs()
    d["subject"] = d["edf_filename"].str.split("_").str[0]
    parts = d["edf_filename"].str.removesuffix(".edf").str.split("_")
    ds = d["base_folder"].map(DATASET_LABEL).fillna(d["base_folder"])
    # Group = multisite sensor site, else the dataset name
    d["group"] = np.where(
        d["base_folder"] == "multisite-ppg",
        "Multisite: " + parts.str[1].fillna(""),
        ds,
    )
    d["subject_label"] = ds + " " + d["subject"]
    return d


def metrics(g: pd.DataFrame) -> pd.Series:
    e = g["error"]
    return pd.Series({
        "segments": len(g),
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "bias": e.mean(),
        "within_5bpm_%": (e.abs() <= 5).mean() * 100,
    })


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def plot_by_site(d, out):
    order = (d.groupby("group")["abs_error"].mean().sort_values().index)
    data = [d.loc[d["group"] == g, "abs_error"].to_numpy() for g in order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bp = ax.boxplot(data, vert=False, showfliers=False, widths=0.55,
                    patch_artist=True, whis=(5, 95))
    for box in bp["boxes"]:
        box.set(facecolor=BLUE, alpha=0.25, edgecolor=BLUE, linewidth=1.2)
    for k in ("whiskers", "caps"):
        for line in bp[k]:
            line.set(color=INK_2, linewidth=1)
    for line in bp["medians"]:
        line.set(color=BLUE, linewidth=2)
    ax.set_yticks(range(1, len(order) + 1), order)
    xmax = max(np.percentile(x, 95) for x in data) * 1.35
    for i, (g, x) in enumerate(zip(order, data), start=1):
        ax.text(xmax, i, f"MAE {x.mean():.1f} bpm   n={len(x):,}",
                va="center", ha="right", color=INK_2, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Absolute error |predicted − reference| (bpm)")
    ax.set_title("Error per sensor site / dataset")
    ax.grid(axis="y", visible=False)
    save(fig, out, "1_error_by_site.png")


def plot_by_hr(d, out):
    bins = [40, 60, 70, 80, 90, 100, 120, 160]
    labels = [f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])]
    d = d.assign(hr_bin=pd.cut(d["target_hr"], bins, labels=labels))
    t = d.groupby("hr_bin", observed=False).agg(
        MAE=("abs_error", "mean"), n=("abs_error", "size"))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(t))
    ax.bar(x, t["MAE"], color=BLUE, width=0.6)
    for xi, (mae, n) in zip(x, t[["MAE", "n"]].to_numpy()):
        if n:
            ax.text(xi, mae + 0.2, f"{mae:.1f}\nn={int(n):,}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, t.index.astype(str))
    ax.set_xlabel("Reference heart rate (bpm)")
    ax.set_ylabel("MAE (bpm)")
    ax.set_ylim(0, t["MAE"].max() * 1.3)
    ax.set_title("Error per heart-rate range")
    ax.grid(axis="x", visible=False)
    save(fig, out, "2_error_by_hr.png")


def plot_pred_vs_true(d, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo, hi = 40, 160
    hb = ax.hexbin(d["target_hr"], d["predicted_hr"], gridsize=70,
                   extent=(lo, hi, lo, hi), bins="log", cmap="Blues",
                   mincnt=1, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    ax.text(hi - 2, hi - 6, "perfect", color=INK_2, ha="right", fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference HR from ECG (bpm)")
    ax.set_ylabel("Predicted HR from PPG (bpm)")
    ax.set_title("Predicted vs reference heart rate")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Segments (log scale)")
    cb.outline.set_visible(False)
    save(fig, out, "3_pred_vs_true.png")


def plot_bland_altman(d, out):
    mean_hr = (d["predicted_hr"] + d["target_hr"]) / 2
    diff = d["error"]
    bias, sd = diff.mean(), diff.std()
    lo_loa, hi_loa = bias - 1.96 * sd, bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hb = ax.hexbin(mean_hr, diff, gridsize=(80, 45),
                   extent=(40, 160, -45, 45), bins="log", cmap="Blues",
                   mincnt=1, linewidths=0)
    for y, label, style in [(bias, f"bias {bias:+.2f}", "-"),
                            (hi_loa, f"+1.96 SD {hi_loa:+.1f}", "--"),
                            (lo_loa, f"−1.96 SD {lo_loa:+.1f}", "--")]:
        ax.axhline(y, color=ORANGE, linewidth=1.5, linestyle=style)
        ax.text(159, y + 1, label, color=INK, ha="right", va="bottom",
                fontsize=9)
    ax.set_xlim(40, 160)
    ax.set_ylim(-45, 45)
    ax.set_xlabel("Mean of predicted and reference HR (bpm)")
    ax.set_ylabel("Predicted − reference (bpm)")
    ax.set_title("Bland–Altman plot (cross-validation, training split)")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Segments (log scale)")
    cb.outline.set_visible(False)
    save(fig, out, "4_bland_altman.png")
    return bias, lo_loa, hi_loa


def plot_by_subject(d, out):
    t = (d.groupby("subject_label")
         .agg(MAE=("abs_error", "mean"), n=("abs_error", "size"))
         .sort_values("MAE"))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(t) + 1)))
    y = np.arange(len(t))
    ax.barh(y, t["MAE"], color=BLUE, height=0.65)
    overall = d["abs_error"].mean()
    ax.axvline(overall, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.text(overall, len(t) - 0.2, f" overall MAE {overall:.1f}",
            color=INK, va="bottom", fontsize=9)
    ax.set_yticks(y, t.index, fontsize=8)
    ax.set_xlabel("MAE (bpm)")
    ax.set_title("Error per subject")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(t) + 0.3)
    save(fig, out, "5_error_by_subject.png")
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("oof_csv", type=Path)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/results/cnn_validation"))
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.oof_csv)
    plot_by_site(d, out)
    plot_by_hr(d, out)
    plot_pred_vs_true(d, out)
    bias, lo, hi = plot_bland_altman(d, out)
    subjects = plot_by_subject(d, out)

    summary = pd.concat({
        "all": metrics(d).to_frame("all").T,
        "group": d.groupby("group").apply(metrics, include_groups=False),
    })
    summary.round(3).to_csv(out / "summary_by_group.csv")
    subjects.round(3).to_csv(out / "summary_by_subject.csv")
    print(summary.round(2).to_string())
    print(f"\nBland-Altman: bias {bias:+.2f} bpm, LoA {lo:+.2f} to {hi:+.2f} bpm")
    print(f"Figures saved in {out}")


if __name__ == "__main__":
    main()
