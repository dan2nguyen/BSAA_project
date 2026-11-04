"""
Runs `predict_training_segments.py` for every CV fold one after another
and, after each fold, `analyze_predictions.py` on its output.

Output per fold:
    <output-dir>/fold<k>/predictions/   mirrored parquet files with predictions
    <output-dir>/fold<k>/report/        report.md, CSVs, figures

Default paths are the Kaggle datasets when running on Kaggle
(/kaggle/input exists), otherwise the local repo folders.

Usage (Kaggle notebook cell):
    !python prototyping_dan/run_commands.py
    !python prototyping_dan/run_commands.py --folds 1 2 --subset all
Local:
    python prototyping_dan/run_commands.py --skip-existing
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
PREDICT_SCRIPT = root_dir / "prototyping_dan" / "predict_training_segments.py"
ANALYZE_SCRIPT = root_dir / "prototyping_dan" / "analyze_predictions.py"

ON_KAGGLE = Path("/kaggle/input").exists()
if ON_KAGGLE:
    KAGGLE_DATA = Path("/kaggle/input/datasets/dannguyen2/bsaa-ppg-project")
    DEFAULTS = {
        "models_dir": Path("/kaggle/input/datasets/dannguyen2/cnn-basic-models"),
        "edf_dir": KAGGLE_DATA / "harmonized_ecg_ppg_database_filtered",
        "parquet_dir": Path("/kaggle/input/datasets/dannguyen2/hr-parts/seg10s_step5s"),
        "output_dir": Path("/kaggle/working/fold_analysis"),
    }
else:
    DEFAULTS = {
        "models_dir": root_dir / "data" / "training_outputs",
        "edf_dir": root_dir / "data" / "harmonized_ecg_ppg_database_filtered",
        "parquet_dir": root_dir / "data" / "hr_parts" / "seg10s_step5s",
        "output_dir": root_dir / "data" / "training_outputs",
    }


def resolve_parquet_dir(path: Path) -> Path:
    """
    The scripts expect <dir>/<base_folder>/<file>.parquet. If the given
    folder is one level higher (e.g. hr_parts/ instead of
    hr_parts/seg10s_step5s/), the single subfolder with that layout is used.
    """
    if any(path.glob("*/*.parquet")):
        return path
    candidates = sorted(
        d for d in path.iterdir() if d.is_dir() and any(d.glob("*/*.parquet"))
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No <folder>/<file>.parquet found in {path}")
    raise ValueError(
        f"Several parquet databases in {path}: "
        + ", ".join(d.name for d in candidates)
        + " - choose one with --parquet-dir"
    )


def find_model(models_dir: Path, fold: int, pattern: str) -> Path:
    """
    Finds the checkpoint of one fold anywhere below models_dir, e.g.
    best_model_10s_fold3.pt or fold_3/best_model.pt.
    """
    fold_re = re.compile(rf"fold_?{fold}(?!\d)")
    matches = sorted(
        p for p in models_dir.rglob(pattern)
        if fold_re.search(str(p.relative_to(models_dir)))
    )
    if len(matches) != 1:
        found = [str(p) for p in models_dir.rglob(pattern)]
        raise FileNotFoundError(
            f"Expected 1 checkpoint for fold {fold} in {models_dir}, "
            f"found {len(matches)}. All checkpoints: {found}"
        )
    return matches[0]


def run(args: list[str]) -> tuple[bool, float]:
    """Runs a command in the repo root; returns (success, duration in s)."""
    print("\n>>> " + subprocess.list2cmdline(args), flush=True)
    t0 = time.time()
    ok = subprocess.run(args, cwd=root_dir).returncode == 0
    return ok, time.time() - t0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and analyze every CV fold."
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULTS["models_dir"])
    parser.add_argument("--edf-dir", type=Path, default=DEFAULTS["edf_dir"])
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULTS["parquet_dir"],
        help="Parquet database (or its parent folder with one database)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULTS["output_dir"])
    parser.add_argument(
        "--model-pattern",
        default="*.pt",
        help="Glob for checkpoints; the fold is taken from 'fold<k>' in the path",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--subset", choices=["train", "val", "all"], default="train"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the prediction if its output folder already has parquet files",
    )
    parser.add_argument(
        "--analyze-args",
        default="",
        help="Extra arguments for analyze_predictions.py, "
        "e.g. \"--error-limit 5 --n-bins 20\"",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = resolve_parquet_dir(args.parquet_dir)
    print(f"Running on {'Kaggle' if ON_KAGGLE else 'local machine'}")
    print(f"Parquet DB: {parquet_dir}\nEDF dir:    {args.edf_dir}")
    print(f"Models:     {args.models_dir}\nOutput:     {args.output_dir}")

    # Find all checkpoints first so a missing one fails before hours of work
    models = {
        k: find_model(args.models_dir, k, args.model_pattern) for k in args.folds
    }
    for k, path in models.items():
        print(f"  fold {k}: {path}")

    results = []
    for k in args.folds:
        print(f"\n{'=' * 70}\nFold {k}")
        pred_dir = args.output_dir / f"fold{k}" / "predictions"
        report_dir = args.output_dir / f"fold{k}" / "report"

        if args.skip_existing and any(pred_dir.rglob("*.parquet")):
            print(f"Predictions exist in {pred_dir}, prediction skipped")
            pred_status = "skipped (exists)"
        else:
            ok, dt = run([
                sys.executable, str(PREDICT_SCRIPT),
                "--parquet-dir", str(parquet_dir),
                "--edf-dir", str(args.edf_dir),
                "--model-path", str(models[k]),
                "--output-dir", str(pred_dir),
                "--n-splits", str(args.n_splits),
                "--fold", str(k),
                "--subset", args.subset,
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
            ])
            pred_status = f"ok ({dt / 60:.1f} min)" if ok else "FAILED"
            if not ok:
                results.append((k, pred_status, "not run"))
                continue

        ok, _ = run([
            sys.executable, str(ANALYZE_SCRIPT),
            "--pred-dir", str(pred_dir),
            "--report-dir", str(report_dir),
            *shlex.split(args.analyze_args),
        ])
        results.append((k, pred_status, f"ok -> {report_dir}" if ok else "FAILED"))

    print(f"\n{'=' * 70}\nSummary")
    for k, pred_status, analysis_status in results:
        print(f"  fold {k}: prediction: {pred_status:<20} analysis: {analysis_status}")


if __name__ == "__main__":
    main()
