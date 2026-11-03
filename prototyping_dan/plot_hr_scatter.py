"""
Plots ground-truth HR (x) vs model-estimated HR (y) from the output
folder of `predict_training_segments.py`.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def load_predictions(pred_dir: str) -> pd.DataFrame:
    """Loads hr_bpm, hr_pred and cv_role from all parquet files."""
    files = sorted(Path(pred_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pred_dir}")
    df = pd.concat(
        [
            pd.read_parquet(f, columns=["hr_bpm", "hr_pred", "cv_role"])
            for f in files
        ],
        ignore_index=True,
    )
    return df.dropna(subset=["hr_bpm", "hr_pred"])


def plot_hr_scatter(
    df: pd.DataFrame, title: str, output_path: Path
) -> None:
    """
    2D density plot of true vs predicted HR with the identity line
    and a ±5 bpm band; one panel per cv_role.
    """
    roles = sorted(df["cv_role"].unique())
    lo = np.floor(min(df["hr_bpm"].min(), df["hr_pred"].min()) / 10) * 10
    hi = np.ceil(max(df["hr_bpm"].max(), df["hr_pred"].max()) / 10) * 10

    fig, axes = plt.subplots(
        1, len(roles), figsize=(7 * len(roles), 6.5), squeeze=False
    )
    for ax, role in zip(axes[0], roles):
        g = df[df["cv_role"] == role]
        err = g["hr_pred"] - g["hr_bpm"]
        mae = err.abs().mean()
        rmse = np.sqrt((err**2).mean())
        acc = (err.abs() <= 5).mean()
        r = np.corrcoef(g["hr_bpm"], g["hr_pred"])[0, 1]

        hb = ax.hexbin(
            g["hr_bpm"],
            g["hr_pred"],
            gridsize=120,
            extent=(lo, hi, lo, hi),
            norm=LogNorm(),
            cmap="viridis",
            mincnt=1,
        )
        ax.plot([lo, hi], [lo, hi], "r-", lw=1.2, label="ideal (y = x)")
        ax.fill_between(
            [lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5],
            color="red", alpha=0.12, label="±5 bpm",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Ground-truth HR from ECG (bpm)")
        ax.set_ylabel("Estimated HR by model (bpm)")
        ax.set_title(
            f"{role}: N={len(g)}, MAE={mae:.2f} bpm, RMSE={rmse:.2f} bpm\n"
            f"±5 bpm: {acc:.1%}, Pearson r={r:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.colorbar(hb, ax=ax, label="Segments per bin (log)")

    fig.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth vs estimated HR."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Output folder of predict_training_segments.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG path, default <pred-dir>_hr_scatter.png",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output = Path(args.output or f"{pred_dir}_hr_scatter.png")

    df = load_predictions(args.pred_dir)
    plot_hr_scatter(df, f"HR estimation: {pred_dir.name}", output)
    print(f"{len(df)} segments plotted -> {output}")


if __name__ == "__main__":
    main()
