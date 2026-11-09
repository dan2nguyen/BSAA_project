"""
Combined analysis of the per-segment HR prediction errors of all CV folds
(outputs of `predict_training_segments.py`, one folder per fold).

Main result: a border for every numerical metric that rejects a fixed share
(default 10 %) of segments, placed where the error is largest. For each
metric the rejected share is split between the low and the high tail so
that the MAE of the kept segments is minimal. Borders are fitted on the
pooled folds and, for stability, on every fold separately.

Outputs (in --report-dir):
    report.md, metrics_overall.csv, metrics_borders.csv,
    metrics_borders_per_fold.csv, metrics_borders_combined.csv,
    metrics_boolean.csv, metrics_numerical_spearman.csv,
    metrics_numerical_bins.csv, metrics_per_subject.csv, figures/*.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from prototyping_dan.analyze_predictions import (
    DEFAULT_EXCLUDE,
    analyze_boolean,
    detect_flags,
    error_metrics,
    grouped_metrics,
    keep_rule,
    md_table,
    write_report,
)
from prototyping_dan.load_database import extract_patient_id

# Not needed for the analysis, dropped on load to save memory
DROP_ON_LOAD = ["model_path", "invalid_reason", "split", "start", "end",
                "start_t", "end_t", "segment_s", "sampling_rate"]
ERROR_COLS = {"abs": "hr_abs_error", "rel": "hr_rel_error"}


# -----------------------------
# 1. Loading
# -----------------------------


def find_fold_dirs(analysis_dir: Path) -> dict[int, Path]:
    """Maps fold number -> its predictions folder (foldN/predictions)."""
    folds = {}
    for d in sorted(analysis_dir.iterdir()):
        m = re.fullmatch(r"fold(\d+)", d.name)
        if m and (d / "predictions").is_dir():
            folds[int(m.group(1))] = d / "predictions"
    if not folds:
        raise FileNotFoundError(f"No foldN/predictions folders in {analysis_dir}")
    return folds


def load_fold(pred_dir: Path, fold: int) -> pd.DataFrame:
    """Loads all parquet files of one fold."""
    files = sorted(pred_dir.rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in DROP_ON_LOAD if c in df.columns])
    df["model_fold"] = fold
    return df


# -----------------------------
# 2. Borders
# -----------------------------


def fit_border(
    x: np.ndarray, err: np.ndarray, reject_share: float
) -> tuple[float, float]:
    """
    Finds the borders that reject at most `reject_share` of the segments
    and minimize the mean error of the kept ones.

    The rejected segments are the k_lo lowest and k_hi highest values of
    the metric (k_lo + k_hi <= reject_share * n). Cuts are only placed
    between distinct values so that the rule is realizable as a threshold
    (for discrete metrics like n_beats the rejected share can be smaller).

    Returns:
        (lower, upper): keep lower < x < upper; NaN = no border on that side.
    """
    order = np.argsort(x, kind="stable")
    xs, es = x[order], err[order]
    n = len(xs)
    k = int(np.floor(reject_share * n))
    cum = np.concatenate([[0.0], np.cumsum(es)])

    # Cut positions between distinct values (plus both ends)
    cuts = np.concatenate([[0], np.flatnonzero(xs[1:] > xs[:-1]) + 1, [n]])
    lo = cuts[cuts <= k]
    # For each lower cut, the smallest upper cut keeping the budget
    hi = cuts[np.searchsorted(cuts, n - (k - lo), side="left")]
    kept_mean = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    best = int(np.argmin(kept_mean))
    # No border if no cut keeps segments with a lower error than all of them
    if kept_mean[best] >= cum[-1] / n:
        return np.nan, np.nan
    b_lo, b_hi = lo[best], hi[best]

    # Borders in the middle between the last rejected and first kept value
    lower = (xs[b_lo - 1] + xs[b_lo]) / 2 if b_lo > 0 else np.nan
    upper = (xs[b_hi - 1] + xs[b_hi]) / 2 if b_hi < n else np.nan
    return float(lower), float(upper)


def keep_mask(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Segments inside the borders (NaN metric values are kept)."""
    keep = np.ones(len(x), dtype=bool)
    if not np.isnan(lower):
        keep &= ~(x <= lower)
    if not np.isnan(upper):
        keep &= ~(x >= upper)
    return keep


def evaluate_border(
    sub: pd.DataFrame, col: str, lower: float, upper: float,
    worst: np.ndarray,
) -> dict:
    """
    Kept / rejected errors for a border and the share of the worst-error
    segments it catches (`worst` = boolean mask of the top-error segments).
    """
    keep = keep_mask(sub[col].to_numpy(), lower, upper)
    kept = error_metrics(sub.loc[keep, "hr_error"], sub.loc[keep, "hr_rel_error"])
    rej = error_metrics(sub.loc[~keep, "hr_error"], sub.loc[~keep, "hr_rel_error"])
    return {
        "keep_rule": keep_rule(col, lower, upper).replace("<=", "<"),
        "lower_border": lower,
        "upper_border": upper,
        "rejected_share": float((~keep).mean()),
        "kept_mae": kept["mae"],
        "kept_mape": kept["mape"],
        "kept_acc_5bpm": kept["acc_5bpm"],
        "rejected_mae": rej["mae"],
        "rejected_mape": rej["mape"],
        # Share of the worst-error segments that the border rejects
        "worst_caught": float((~keep & worst).sum() / max(worst.sum(), 1)),
    }


def worst_mask(err: np.ndarray, share: float) -> np.ndarray:
    """Top `share` of segments by error."""
    return err >= np.quantile(err, 1 - share)


def analyze_borders(
    df: pd.DataFrame, flags: list[str], err_col: str, reject_share: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Borders fitted on the pooled folds and on every fold separately.

    Returns:
        (pooled borders sorted by kept MAE, per-fold borders)
    """
    pooled_rows, fold_rows = [], []
    for col in flags:
        sub = df[[col, "model_fold", "hr_error", "hr_rel_error", err_col]].dropna()
        if sub[col].nunique() < 2:
            continue
        err = sub[err_col].to_numpy()
        lower, upper = fit_border(sub[col].to_numpy(), err, reject_share)
        pooled_rows.append({
            "flag": col,
            **evaluate_border(sub, col, lower, upper,
                              worst_mask(err, reject_share)),
        })

        for fold, g in sub.groupby("model_fold"):
            g_err = g[err_col].to_numpy()
            g_worst = worst_mask(g_err, reject_share)
            f_lo, f_hi = fit_border(g[col].to_numpy(), g_err, reject_share)
            own = evaluate_border(g, col, f_lo, f_hi, g_worst)
            # The pooled border applied to this fold
            pooled_on_fold = evaluate_border(g, col, lower, upper, g_worst)
            fold_rows.append({
                "flag": col,
                "model_fold": fold,
                **own,
                "pooled_border_kept_mae": pooled_on_fold["kept_mae"],
                "pooled_border_rejected_share": pooled_on_fold["rejected_share"],
            })

    pooled = pd.DataFrame(pooled_rows).sort_values("kept_mae")
    return pooled, pd.DataFrame(fold_rows)


def combine_borders(df: pd.DataFrame, borders: pd.DataFrame) -> pd.DataFrame:
    """Cumulatively applies the borders in the given order (best first)."""
    keep = np.ones(len(df), dtype=bool)
    rows = []
    for b in borders.itertuples(index=False):
        keep &= keep_mask(df[b.flag].to_numpy(), b.lower_border, b.upper_border)
        m = error_metrics(df.loc[keep, "hr_error"], df.loc[keep, "hr_rel_error"])
        rows.append({"added_rule": b.keep_rule, "kept_share": keep.mean(),
                     **{f"kept_{k}": v for k, v in m.items() if k != "n"}})
    return pd.DataFrame(rows)


def border_stability(per_fold: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Rule per fold as a flag x fold table, in the order of `order`."""
    tab = per_fold.pivot(index="flag", columns="model_fold", values="keep_rule")
    tab.columns = [f"fold {c}" for c in tab.columns]
    spread = per_fold.groupby("flag").agg(
        kept_mae_std=("kept_mae", "std"),
        pooled_border_kept_mae_max=("pooled_border_kept_mae", "max"),
    )
    return tab.join(spread).loc[order].reset_index()


# -----------------------------
# 3. Numerical bins, correlation, figures
# -----------------------------


def numerical_bins(
    df: pd.DataFrame, flags: list[str], n_bins: int
) -> pd.DataFrame:
    """MAE per pooled quantile bin, pooled ('all') and per fold."""
    rows = []
    for col in flags:
        sub = df[[col, "model_fold", "hr_error", "hr_rel_error"]].dropna()
        bins = pd.qcut(sub[col], q=n_bins, duplicates="drop")
        for fold_key, part in [("all", sub)] + list(sub.groupby("model_fold")):
            for interval, g in part.groupby(bins.loc[part.index], observed=True):
                rows.append({
                    "flag": col, "model_fold": fold_key, "bin": str(interval),
                    "x_median": g[col].median(),
                    **error_metrics(g["hr_error"], g["hr_rel_error"]),
                })
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """Spearman rho of each numerical metric vs absolute error (pooled)."""
    rows = []
    for col in flags:
        sub = df[[col, "hr_abs_error"]].dropna()
        rho, p = spearmanr(sub[col], sub["hr_abs_error"])
        rows.append({"flag": col, "n": len(sub), "spearman_rho": rho,
                     "p_value": p})
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False,
                                          key=np.abs)


def plot_border(
    df: pd.DataFrame,
    col: str,
    bins: pd.DataFrame,
    border: pd.Series,
    err_col: str,
    reject_share: float,
    fig_dir: Path,
) -> None:
    """
    Left: MAE per quantile bin (per fold and pooled) with the border and
    rejected regions. Right: distribution of the metric for the worst-error
    segments vs the rest.
    """
    sub = df[[col, err_col]].dropna()
    x = sub[col].to_numpy()
    x_lo, x_hi = np.nanpercentile(x, [0.5, 99.5])
    if x_lo == x_hi:
        x_lo, x_hi = x.min(), x.max()
    pad = 0.02 * (x_hi - x_lo) or 1.0
    xlim = (x_lo - pad, x_hi + pad)
    lower, upper = border["lower_border"], border["upper_border"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    tab = bins[bins.flag == col]
    for fold, g in tab[tab.model_fold != "all"].groupby("model_fold"):
        ax.plot(g["x_median"], g["mae"], "o-", lw=1, ms=3, alpha=0.6,
                label=f"fold {fold}")
    g = tab[tab.model_fold == "all"]
    ax.plot(g["x_median"], g["mae"], "o-", color="black", lw=2.5,
            label="all folds")
    ax.set_ylabel("MAE per quantile bin (bpm)")
    ax.set_title(f"{col}: MAE per bin\n{border['keep_rule']}  "
                 f"(kept MAE {border['kept_mae']:.2f}, "
                 f"rejected MAE {border['rejected_mae']:.2f} bpm)",
                 fontsize=10)

    ax = axes[1]
    worst = worst_mask(sub[err_col].to_numpy(), reject_share)
    edges = np.linspace(*xlim, 80)
    for mask, label, color in [(~worst, "rest", "gray"),
                               (worst, f"worst {reject_share:.0%} error",
                                "red")]:
        ax.hist(np.clip(x[mask], *xlim), bins=edges, density=True, alpha=0.5,
                color=color, label=f"{label} (n={mask.sum()})")
    ax.set_ylabel("Density")
    ax.set_title(f"{col}: distribution, border catches "
                 f"{border['worst_caught']:.1%} of the worst segments "
                 f"(random: {reject_share:.0%})", fontsize=10)

    for ax in axes:
        for x_b, side in [(lower, "left"), (upper, "right")]:
            if np.isnan(x_b):
                continue
            ax.axvline(x_b, color="green", lw=2)
            span = (xlim[0], x_b) if side == "left" else (x_b, xlim[1])
            ax.axvspan(*span, color="red", alpha=0.08)
        ax.set_xlim(*xlim)
        ax.set_xlabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / f"border_{col}.png", dpi=110)
    plt.close(fig)


def plot_fold_errors(df: pd.DataFrame, fig_dir: Path) -> None:
    """Histogram of signed errors per fold."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = np.nanpercentile(df["hr_error"].abs(), 99)
    for fold, g in df.groupby("model_fold"):
        ax.hist(g["hr_error"].clip(-lim, lim), bins=100, density=True,
                histtype="step", lw=1.5, label=f"fold {fold} (n={len(g)})")
    ax.set_xlabel("Signed error HR_pred - HR_true (bpm)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution per fold (clipped at 99th percentile)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_histogram.png", dpi=110)
    plt.close(fig)


# -----------------------------
# 4. Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined HR error analysis of all CV folds with "
        "fixed-share rejection borders for numerical metrics."
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(root_dir / "data" / "fold_analysis"),
        help="Folder with fold1..foldN/predictions",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Output folder (default: <analysis-dir>/combined_report)",
    )
    parser.add_argument(
        "--reject-share",
        type=float,
        default=0.10,
        help="Share of segments each border rejects (default 0.10)",
    )
    parser.add_argument(
        "--error",
        choices=list(ERROR_COLS),
        default="abs",
        help="Error the borders minimize: 'abs' = |HR_pred - HR_true| in bpm, "
        "'rel' = relative error in %%",
    )
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--exclude-columns", nargs="+", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    report_dir = Path(args.report_dir or analysis_dir / "combined_report")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    err_col = ERROR_COLS[args.error]
    share = args.reject_share

    # --- Load ---
    fold_dirs = find_fold_dirs(analysis_dir)
    parts = []
    for fold, d in fold_dirs.items():
        parts.append(load_fold(d, fold))
        print(f"Fold {fold}: {len(parts[-1])} segments from {d}")
    df = pd.concat(parts, ignore_index=True)
    del parts
    n_total = len(df)
    failed = df["hr_pred"].isna()
    df = df[~failed].reset_index(drop=True)
    df["hr_rel_error"] = 100 * df["hr_abs_error"] / df["hr_bpm"]
    df["subject_id"] = df["base_folder"] + "/" + df["edf_filename"].map(
        extract_patient_id
    )
    print(f"Loaded {n_total} segments, dropped {int(failed.sum())} with NaN hr_pred")

    # --- Overall ---
    overall = [{"group": "all", "value": "all",
                **error_metrics(df["hr_error"], df["hr_rel_error"])}]
    for col in ["model_fold", "cv_role", "base_folder"]:
        for value, g in df.groupby(col, sort=True):
            overall.append({"group": col, "value": value,
                            **error_metrics(g["hr_error"], g["hr_rel_error"])})
    overall = pd.DataFrame(overall)
    fold_folder = grouped_metrics(df, ["base_folder", "model_fold"]).sort_values(
        ["base_folder", "model_fold"]
    )
    plot_fold_errors(df, fig_dir)

    # --- Flags ---
    exclude = DEFAULT_EXCLUDE | set(args.exclude_columns)
    boolean, _, numerical, constant = detect_flags(df, None, exclude)
    print(f"Numerical metrics ({len(numerical)}): {numerical}")

    # --- Borders ---
    borders, per_fold = analyze_borders(df, numerical, err_col, share)
    combined = combine_borders(df, borders)
    stability = border_stability(per_fold, borders["flag"].tolist())
    oracle_keep = ~worst_mask(df[err_col].to_numpy(), share)
    oracle = error_metrics(df.loc[oracle_keep, "hr_error"],
                           df.loc[oracle_keep, "hr_rel_error"])
    base = overall.iloc[0]

    # --- Other tables ---
    bins_tab = numerical_bins(df, borders["flag"].tolist(), args.n_bins)
    corr_tab = spearman_table(df, numerical)
    bool_tab = analyze_boolean(df, boolean)
    per_subject = grouped_metrics(df, ["subject_id"])
    for _, b in borders.iterrows():
        plot_border(df, b["flag"], bins_tab, b, err_col, share, fig_dir)

    # --- Save CSVs ---
    overall.to_csv(report_dir / "metrics_overall.csv", index=False)
    borders.to_csv(report_dir / "metrics_borders.csv", index=False)
    per_fold.to_csv(report_dir / "metrics_borders_per_fold.csv", index=False)
    combined.to_csv(report_dir / "metrics_borders_combined.csv", index=False)
    bool_tab.to_csv(report_dir / "metrics_boolean.csv", index=False)
    corr_tab.to_csv(report_dir / "metrics_numerical_spearman.csv", index=False)
    bins_tab.to_csv(report_dir / "metrics_numerical_bins.csv", index=False)
    per_subject.to_csv(report_dir / "metrics_per_subject.csv", index=False)

    # --- Report ---
    err_name = ("absolute error |HR_pred − HR_true| (bpm)" if args.error == "abs"
                else "relative error |HR_pred − HR_true| / HR_true (%)")
    roles = ", ".join(f"{k}: {v}" for k, v in
                      df["cv_role"].value_counts().items())
    border_cols = ["flag", "keep_rule", "rejected_share", "kept_mae",
                   "kept_mape", "kept_acc_5bpm", "rejected_mae",
                   "rejected_mape", "worst_caught"]
    figs = "\n".join(
        f"### {c}\n\n![{c}](figures/border_{c}.png)\n" for c in borders["flag"]
    )
    sections = [
        ("Data",
         f"- Analysis folder: {analysis_dir}\n"
         f"- Folds: {', '.join(map(str, fold_dirs))}\n"
         f"- Segments loaded: {n_total} (dropped NaN `hr_pred`: "
         f"{int(failed.sum())}), analyzed: {len(df)}\n"
         f"- CV roles: {roles}\n"
         f"- Subjects: {df['subject_id'].nunique()}, signals: "
         f"{df.groupby(['base_folder', 'edf_filename']).ngroups}\n"
         f"- Skipped constant columns: {', '.join(constant) or 'none'}\n\n"
         "Each fold's model is evaluated on the segments of its own fold. "
         "A segment appears in several folds, so the pooled numbers weight "
         "it once per fold.\n"),
        ("Overall (bpm)",
         md_table(overall) + "\n![error histogram](figures/error_histogram.png)\n"
         "\nPer dataset and fold:\n\n" + md_table(fold_folder)),
        (f"Numerical metrics: borders rejecting {share:.0%} of segments",
         f"For every metric, at most {share:.0%} of the segments are rejected, "
         "taken from the low tail, the high tail or both. The split between "
         f"the tails minimizes the mean {err_name} of the kept segments, so "
         "the border removes the part of the metric's range where the error "
         "is largest. Borders are placed between distinct values; for "
         "discrete metrics the rejected share can therefore be lower. "
         "`keep all` = no tail has a higher error than the average. "
         "Fitted on all folds pooled; each metric is evaluated on its own. "
         "Sorted by kept MAE (best first).\n\n"
         f"- `worst_caught`: share of the {share:.0%} highest-error segments "
         f"that the border rejects (random rejection: {share:.0%}, "
         "perfect: 100 %).\n"
         f"- Baseline, all segments: MAE {base['mae']:.3f} bpm, MAPE "
         f"{base['mape']:.3f} %.\n"
         f"- Oracle, rejecting the {share:.0%} worst-error segments directly: "
         f"MAE {oracle['mae']:.3f} bpm, MAPE {oracle['mape']:.3f} %.\n\n"
         + md_table(borders[border_cols])),
        ("Border stability across folds",
         "Borders fitted on every fold separately. `kept_mae_std` = std of "
         "the kept MAE over folds; `pooled_border_kept_mae_max` = worst kept "
         "MAE of a fold when the pooled border is applied to it. Full table: "
         "`metrics_borders_per_fold.csv`.\n\n" + md_table(stability)),
        ("Borders combined",
         "The borders applied cumulatively in the order above (each row "
         "adds one more rule).\n\n" + md_table(combined)),
        ("Numerical metrics: Spearman ρ (metric vs absolute error)",
         md_table(corr_tab)),
        ("Boolean flags (sorted by |MAE difference|, True − False)",
         md_table(bool_tab)),
        ("Border figures", figs),
        ("Subjects (by MAE)",
         "Full table: `metrics_per_subject.csv`.\n\n" + md_table(per_subject, 30)),
    ]
    write_report(report_dir / "report.md", sections)
    print(f"Report written to {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
