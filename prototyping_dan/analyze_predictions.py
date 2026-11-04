"""
Analyzes per-segment HR prediction errors produced by
`predict_training_segments.py` against the quality/annotation flags.

Outputs (in --report-dir):
    report.md, metrics_overall.csv, metrics_boolean.csv,
    metrics_categorical.csv, metrics_numerical_bins.csv,
    metrics_numerical_spearman.csv, metrics_numerical_borders.csv,
    metrics_per_signal.csv,
    metrics_per_subject.csv, figures/*.png
"""

import argparse
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

from prototyping_dan.load_database import extract_patient_id

# Columns never treated as flags (IDs, positions, constants, model outputs)
DEFAULT_EXCLUDE = {
    "base_folder",
    "edf_filename",
    "section_id",
    "split",
    "start",
    "end",
    "start_t",
    "end_t",
    "segment_s",
    "sampling_rate",
    "invalid_reason",
    "hr_pred",
    "hr_error",
    "hr_abs_error",
    "hr_rel_error",
    "cv_role",
    "model_fold",
    "model_path",
    "pred_status",
    "subject_id",
}
MAX_CATEGORICAL_UNIQUE = 5


# -----------------------------
# 1. Loading & metrics
# -----------------------------


def load_predictions(pred_dirs: list[str]) -> pd.DataFrame:
    """Loads all parquet files under the given prediction folders."""
    files = [f for d in pred_dirs for f in sorted(Path(d).rglob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pred_dirs}")
    return pd.concat(
        [pd.read_parquet(f) for f in files], ignore_index=True
    )


def error_metrics(
    err: pd.Series, rel: pd.Series | None = None
) -> dict[str, float]:
    """
    N, MAE, RMSE, bias and share within ±5 bpm of signed errors in bpm;
    with rel (|error| / HR_true in %) also the mean relative error MAPE.
    """
    err = err.dropna()
    if len(err) == 0:
        out = {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
               "acc_5bpm": np.nan}
    else:
        out = {
            "n": int(len(err)),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "bias": float(err.mean()),
            "acc_5bpm": float((err.abs() <= 5).mean()),
        }
    if rel is not None:
        out["mape"] = float(rel.mean()) if len(rel) else np.nan
    return out


def grouped_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Error metrics per group, sorted by MAE (worst first)."""
    rows = [
        {**dict(zip(by, key if isinstance(key, tuple) else (key,))),
         **error_metrics(g["hr_error"])}
        for key, g in df.groupby(by, sort=False)
    ]
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


# -----------------------------
# 2. Flag detection
# -----------------------------


def detect_flags(
    df: pd.DataFrame,
    flag_columns: list[str] | None,
    exclude: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Splits columns into boolean, categorical (low-cardinality int),
    and numerical flags.

    Returns:
        (boolean, categorical, numerical, skipped_constant)
    """
    candidates = flag_columns or [c for c in df.columns if c not in exclude]
    boolean, categorical, numerical, constant = [], [], [], []

    for col in candidates:
        if col not in df.columns:
            print(f"Warning: flag column '{col}' not found, ignored")
            continue
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            constant.append(col)
        elif pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_integer_dtype(s) and n_unique <= MAX_CATEGORICAL_UNIQUE:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numerical.append(col)

    return boolean, categorical, numerical, constant


# -----------------------------
# 3. Analyses
# -----------------------------


def analyze_boolean(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """MAE/RMSE/N for True vs False per boolean flag, plus differences."""
    rows = []
    for col in flags:
        t = error_metrics(df.loc[df[col] == True, "hr_error"])  # noqa: E712
        f = error_metrics(df.loc[df[col] == False, "hr_error"])  # noqa: E712
        rows.append(
            {
                "flag": col,
                "n_true": t["n"],
                "mae_true": t["mae"],
                "rmse_true": t["rmse"],
                "n_false": f["n"],
                "mae_false": f["mae"],
                "rmse_false": f["rmse"],
                "mae_diff": t["mae"] - f["mae"],
                "rmse_diff": t["rmse"] - f["rmse"],
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("mae_diff", ascending=False, key=np.abs)
    return out


def analyze_categorical(df: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """MAE/RMSE/N per value of low-cardinality integer columns."""
    rows = []
    for col in flags:
        for value, g in df.groupby(col, sort=True):
            rows.append({"flag": col, "value": value, **error_metrics(g["hr_error"])})
    return pd.DataFrame(rows)


def keep_rule(col: str, lower: float, upper: float) -> str:
    """Human-readable rule for the kept range, e.g. '0.2 < x <= 3.1'."""
    if np.isnan(lower) and np.isnan(upper):
        return "keep all"
    if np.isnan(lower):
        return f"{col} <= {upper:.4g}"
    if np.isnan(upper):
        return f"{col} > {lower:.4g}"
    return f"{lower:.4g} < {col} <= {upper:.4g}"


def find_border(
    table: pd.DataFrame, sub: pd.DataFrame, col: str, limit: float
) -> dict:
    """
    Suggests the acceptable range of a metric for a relative-error limit.

    The acceptable range is the contiguous run of quantile bins with
    MAPE <= limit that contains the best bin. Its edges are the borders:
    'lower' if bins below it exceed the limit, 'upper' if bins above do.

    Returns:
        Row with the borders and the errors of kept / rejected segments.
    """
    mape = table["mape"].to_numpy()
    best = int(np.nanargmin(mape))
    row = {"flag": col, "best_bin_mape": mape[best]}
    if mape[best] > limit:
        return {**row, "status": f"no bin with MAPE <= {limit:g} %"}

    lo = best
    while lo > 0 and mape[lo - 1] <= limit:
        lo -= 1
    hi = best
    while hi < len(mape) - 1 and mape[hi + 1] <= limit:
        hi += 1

    lower = float(table["bin_left"].iloc[lo]) if lo > 0 else np.nan
    upper = float(table["bin_right"].iloc[hi]) if hi < len(mape) - 1 else np.nan
    x = sub[col].to_numpy()
    keep = np.ones(len(sub), dtype=bool)
    if not np.isnan(lower):
        keep &= x > lower
    if not np.isnan(upper):
        keep &= x <= upper

    kept = error_metrics(sub.loc[keep, "hr_error"], sub.loc[keep, "hr_rel_error"])
    rej = error_metrics(sub.loc[~keep, "hr_error"], sub.loc[~keep, "hr_rel_error"])
    no_border = np.isnan(lower) and np.isnan(upper)
    return {
        **row,
        "status": "no border needed" if no_border else "border suggested",
        "lower_border": lower,
        "upper_border": upper,
        "keep_rule": keep_rule(col, lower, upper),
        "kept_share": keep.mean(),
        "kept_mae": kept["mae"],
        "kept_mape": kept["mape"],
        "rejected_mae": rej["mae"],
        "rejected_mape": rej["mape"],
    }


def analyze_numerical(
    df: pd.DataFrame,
    flags: list[str],
    n_bins: int,
    fig_dir: Path,
    max_scatter: int,
    error_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quantile-binned error table, Spearman correlation, suggested border
    for the relative-error limit and a figure per flag.

    Returns:
        (bins table, spearman table, borders table)
    """
    bin_rows, corr_rows, border_rows = [], [], []
    rng = np.random.default_rng(0)

    for col in flags:
        sub = df[[col, "hr_error", "hr_abs_error", "hr_rel_error"]].dropna()
        if sub[col].nunique() < 2:
            continue

        rho, p = spearmanr(sub[col], sub["hr_abs_error"])
        corr_rows.append(
            {"flag": col, "n": len(sub), "spearman_rho": rho, "p_value": p}
        )

        bins = pd.qcut(sub[col], q=n_bins, duplicates="drop")
        table = []
        for interval, g in sub.groupby(bins, observed=True):
            table.append(
                {
                    "flag": col,
                    "bin": str(interval),
                    "bin_left": interval.left,
                    "bin_right": interval.right,
                    "x_median": g[col].median(),
                    **error_metrics(g["hr_error"], g["hr_rel_error"]),
                }
            )
        bin_rows.extend(table)
        table = pd.DataFrame(table)
        border = find_border(table, sub, col, error_limit)
        border_rows.append(border)
        plot_numerical_flag(
            sub, col, table, rho, border, error_limit, fig_dir,
            max_scatter, rng,
        )

    corr = pd.DataFrame(corr_rows)
    if len(corr):
        corr = corr.sort_values("spearman_rho", ascending=False, key=np.abs)
    borders = pd.DataFrame(border_rows)
    if "kept_mape" in borders:
        # Most useful first: biggest relative-error gap rejected vs kept
        gap = borders["rejected_mape"] - borders["kept_mape"]
        borders = borders.iloc[np.argsort(-gap.fillna(-np.inf).to_numpy(),
                                          kind="stable")]
    return pd.DataFrame(bin_rows), corr, borders


def plot_numerical_flag(
    sub: pd.DataFrame,
    col: str,
    table: pd.DataFrame,
    rho: float,
    border: dict,
    error_limit: float,
    fig_dir: Path,
    max_scatter: int,
    rng: np.random.Generator,
) -> None:
    """
    Scatter of flag vs |error| and signed error with binned curves, and
    binned relative error (MAPE) with the limit and the suggested borders.
    """
    pts = sub
    if len(sub) > max_scatter:
        pts = sub.iloc[rng.choice(len(sub), max_scatter, replace=False)]
    # Long tails (e.g. kurtosis) would squeeze the plots: show 0.5-99.5 %
    x_lo, x_hi = np.nanpercentile(sub[col], [0.5, 99.5])
    if x_lo == x_hi:
        x_lo, x_hi = sub[col].min(), sub[col].max()
    pad = 0.02 * (x_hi - x_lo) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    ax = axes[0]
    ax.scatter(pts[col], pts["hr_abs_error"], s=3, alpha=0.15, color="gray",
               rasterized=True, label=f"segments (n={len(pts)} shown)")
    ax.plot(table["x_median"], table["mae"], "o-", color="red", lw=2,
            label="MAE per quantile bin")
    ax.set_xlabel(col)
    ax.set_ylabel("Absolute error |HR_pred - HR_true| (bpm)")
    ax.set_title(f"{col}: absolute error (Spearman ρ = {rho:.3f})")
    ax.set_ylim(0, np.nanpercentile(sub["hr_abs_error"], 99) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(pts[col], pts["hr_error"], s=3, alpha=0.15, color="gray",
               rasterized=True)
    ax.plot(table["x_median"], table["bias"], "o-", color="blue", lw=2,
            label="mean signed error per bin")
    ax.axhline(0, color="black", lw=0.8)
    lim = np.nanpercentile(sub["hr_error"].abs(), 99) * 1.1
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(col)
    ax.set_ylabel("Signed error HR_pred - HR_true (bpm)")
    ax.set_title(f"{col}: signed error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(table["x_median"], table["mape"], "o-", color="purple", lw=2,
            label="MAPE per quantile bin")
    ax.axhline(error_limit, color="red", ls="--", lw=1.2,
               label=f"limit {error_limit:g} %")
    for key in ["lower_border", "upper_border"]:
        x = border.get(key, np.nan)
        if not np.isnan(x):
            ax.axvline(x, color="green", lw=2,
                       label=f"{key.replace('_', ' ')} = {x:.4g}")
    title = f"{col}: relative error\n{border.get('keep_rule', border['status'])}"
    if "kept_share" in border:
        title += (f"  (keeps {border['kept_share']:.1%}, "
                  f"MAPE {border['kept_mape']:.2f} %)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Mean |HR_pred - HR_true| / HR_true (%)")
    ax.set_ylim(0, max(np.nanmax(table["mape"]), error_limit) * 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlim(x_lo - pad, x_hi + pad)

    plt.tight_layout()
    plt.savefig(fig_dir / f"numerical_{col}.png", dpi=110)
    plt.close(fig)


def plot_error_histogram(df: pd.DataFrame, fig_dir: Path) -> None:
    """Histogram of signed errors per cv_role."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = np.nanpercentile(df["hr_error"].abs(), 99)
    for role, g in df.groupby("cv_role"):
        ax.hist(g["hr_error"].clip(-lim, lim), bins=100, alpha=0.5,
                density=True, label=f"{role} (n={len(g)})")
    ax.set_xlabel("Signed error HR_pred - HR_true (bpm)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution (clipped at 99th percentile)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_histogram.png", dpi=110)
    plt.close(fig)


# -----------------------------
# 4. Report
# -----------------------------


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Renders a DataFrame as a Markdown table (no tabulate dependency)."""
    if df.empty:
        return "_(none)_\n"
    if max_rows is not None:
        df = df.head(max_rows)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3g}" if abs(v) < 1e-2 else f"{v:.3f}"
        return str(v)

    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_report(path: Path, sections: list[tuple[str, str]]) -> None:
    """Writes report.md from (heading, body) sections."""
    text = "# HR prediction error analysis\n\n"
    text += "".join(f"## {h}\n\n{b}\n" for h, b in sections)
    path.write_text(text, encoding="utf-8")


# -----------------------------
# 5. Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-segment HR errors vs quality flags."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        nargs="+",
        help="Output folder(s) of predict_training_segments.py",
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flag-columns", nargs="+", default=None)
    parser.add_argument("--exclude-columns", nargs="+", default=[])
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Quantile bins per numerical metric (curves and borders)",
    )
    parser.add_argument(
        "--error-limit",
        type=float,
        default=5.0,
        help="Relative error limit in %% (|HR_pred - HR_true| / HR_true) "
        "used to suggest a border for each numerical metric",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=50000,
        help="Random subsample size for scatter plots (curves use all data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.pred_dir)
    n_total = len(df)
    failed = df["hr_pred"].isna()
    fail_counts = df.loc[failed, "pred_status"].value_counts()
    df = df[~failed].reset_index(drop=True)
    print(f"Loaded {n_total} segments, dropped {int(failed.sum())} with NaN hr_pred")

    df["hr_rel_error"] = 100 * df["hr_abs_error"] / df["hr_bpm"]
    df["subject_id"] = df["base_folder"] + "/" + df["edf_filename"].map(
        extract_patient_id
    )

    # --- Overall ---
    overall = [{"group": "all", "value": "all",
                **error_metrics(df["hr_error"], df["hr_rel_error"])}]
    for col in ["cv_role", "model_fold"]:
        for value, g in df.groupby(col, sort=True):
            overall.append({"group": col, "value": value,
                            **error_metrics(g["hr_error"], g["hr_rel_error"])})
    if df["cv_role"].nunique() > 1 and df["model_fold"].nunique() > 1:
        for (fold, role), g in df.groupby(["model_fold", "cv_role"]):
            overall.append({"group": "model_fold/cv_role",
                            "value": f"{fold}/{role}",
                            **error_metrics(g["hr_error"])})
    for value, g in df.groupby("base_folder", sort=True):
        overall.append({"group": "base_folder", "value": value,
                        **error_metrics(g["hr_error"])})
    overall = pd.DataFrame(overall)
    plot_error_histogram(df, fig_dir)

    # --- Flags ---
    exclude = DEFAULT_EXCLUDE | set(args.exclude_columns)
    boolean, categorical, numerical, constant = detect_flags(
        df, args.flag_columns, exclude
    )
    print(f"Boolean flags ({len(boolean)}): {boolean}")
    print(f"Categorical int flags ({len(categorical)}): {categorical}")
    print(f"Numerical flags ({len(numerical)}): {numerical}")
    print(f"Skipped constant columns: {constant}")

    bool_tab = analyze_boolean(df, boolean)
    cat_tab = analyze_categorical(df, categorical)
    bins_tab, corr_tab, border_tab = analyze_numerical(
        df, numerical, args.n_bins, fig_dir, args.max_scatter_points,
        args.error_limit,
    )

    # --- Per signal / subject ---
    per_signal = grouped_metrics(df, ["base_folder", "edf_filename"])
    roles = df.groupby(["base_folder", "edf_filename"])["cv_role"].agg(
        lambda s: ",".join(sorted(s.unique()))
    )
    per_signal = per_signal.join(roles, on=["base_folder", "edf_filename"])
    per_subject = grouped_metrics(df, ["subject_id"])

    # --- Save CSVs ---
    overall.to_csv(report_dir / "metrics_overall.csv", index=False)
    bool_tab.to_csv(report_dir / "metrics_boolean.csv", index=False)
    cat_tab.to_csv(report_dir / "metrics_categorical.csv", index=False)
    bins_tab.to_csv(report_dir / "metrics_numerical_bins.csv", index=False)
    corr_tab.to_csv(report_dir / "metrics_numerical_spearman.csv", index=False)
    border_tab.to_csv(report_dir / "metrics_numerical_borders.csv", index=False)
    per_signal.to_csv(report_dir / "metrics_per_signal.csv", index=False)
    per_subject.to_csv(report_dir / "metrics_per_subject.csv", index=False)

    # --- Report ---
    fails_md = (
        md_table(fail_counts.rename_axis("pred_status").reset_index(name="n"))
        if len(fail_counts) else "None.\n"
    )
    num_figs = "\n".join(
        f"### {c}\n\n![{c}](figures/numerical_{c}.png)\n\n"
        + md_table(bins_tab.loc[bins_tab.flag == c,
                                ["bin", "x_median", "n", "mae", "rmse",
                                 "bias", "mape"]])
        for c in corr_tab.get("flag", [])
    )
    sections = [
        ("Data",
         f"- Prediction folders: {', '.join(args.pred_dir)}\n"
         f"- Segments loaded: {n_total}\n"
         f"- Dropped (NaN `hr_pred`): {int(failed.sum())}\n"
         f"- Segments analyzed: {len(df)}\n"
         f"- Signals (EDF files): {len(per_signal)}, subjects: {len(per_subject)}\n"
         f"- Skipped constant columns: {', '.join(constant) or 'none'}\n\n"
         "Failure reasons:\n\n" + fails_md),
        ("Overall (bpm)",
         md_table(overall) + "\n![error histogram](figures/error_histogram.png)\n"),
        ("Boolean flags (sorted by |MAE difference|, True − False)",
         md_table(bool_tab)),
        ("Categorical integer flags", md_table(cat_tab)),
        (f"Numerical flags: suggested borders for {args.error_limit:g} % relative error",
         "MAPE = mean of |HR_pred − HR_true| / HR_true. The kept range is the "
         f"contiguous run of quantile bins with MAPE ≤ {args.error_limit:g} % "
         "around the best bin; segments outside it are rejected. Sorted by "
         "MAPE of rejected − kept segments (most useful metric first). Each "
         "metric is evaluated on its own.\n\n" + md_table(border_tab)),
        ("Numerical flags: Spearman ρ (flag vs absolute error)",
         md_table(corr_tab) + "\n" + num_figs),
        ("Worst 30 signals (by MAE)",
         "Full ranking: `metrics_per_signal.csv`.\n\n"
         + md_table(per_signal, 30)),
        ("Best 15 signals (by MAE)", md_table(per_signal.iloc[::-1], 15)),
        ("Subjects (by MAE)",
         "Full table: `metrics_per_subject.csv`.\n\n" + md_table(per_subject, 40)),
    ]
    write_report(report_dir / "report.md", sections)

    print(f"Overall MAE={overall.loc[0, 'mae']:.3f} bpm, "
          f"RMSE={overall.loc[0, 'rmse']:.3f} bpm, N={overall.loc[0, 'n']}")
    print(f"Report written to {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
