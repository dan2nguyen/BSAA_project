"""
Runs a trained HR model on the training-database segments of one CV fold
and writes per-segment predictions as a mirror of the input parquet tree.

Segment selection, preprocessing and the model are imported from
`prototyping_dan/training.py` so they stay identical to training.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from prototyping_dan.training import (
    MODEL_REGISTRY,
    build_model,
    load_training_segments,
    make_cv_splits,
    normalize_batch,
)
from utils.read_edf import read_edf_signals

STATUS_OK = "ok"


# -----------------------------
# 0. Model loading
# -----------------------------


def load_model(
    model_path: str, model_name: str, device: torch.device
) -> tuple[torch.nn.Module, str]:
    """
    Loads a checkpoint (state_dict saved by training.py).

    With model_name == "auto" the architecture is detected from the
    checkpoint keys by trying every model in MODEL_REGISTRY.

    Returns:
        (model in eval mode on device, name of the architecture)
    """
    state_dict = torch.load(model_path, map_location=device)
    names = list(MODEL_REGISTRY) if model_name == "auto" else [model_name]

    errors = {}
    for name in names:
        model = build_model(name, in_channels=1)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            errors[name] = str(e).splitlines()[0]
            continue
        return model.to(device).eval(), name

    raise RuntimeError(
        f"Checkpoint {model_path} does not match "
        + ", ".join(f"{n} ({msg})" for n, msg in errors.items())
    )


# -----------------------------
# 1. Segment selection
# -----------------------------


def select_fold_segments(
    parquet_dir: str,
    n_splits: int,
    fold: int,
    subset: str,
    target_col: str = "feasible_bpm",
) -> pd.DataFrame:
    """
    Selects the segments of one CV fold exactly as training.py does.

    Args:
        parquet_dir: Root of the parquet database.
        n_splits: Number of CV splits used during training.
        fold: 1-based fold number (as in training.py / checkpoint names).
        subset: 'train', 'val' or 'all'.
        target_col: Training target (must match training.py --target_col).

    Returns:
        Selected rows with '_source_relpath', '_source_row' and 'cv_role'.
    """
    if not 1 <= fold <= n_splits:
        raise ValueError(f"--fold must be in 1..{n_splits}, got {fold}")

    train_df = load_training_segments(
        parquet_dir, n_splits, keep_source_info=True, target_col=target_col
    )
    train_idx, val_idx = make_cv_splits(train_df, n_splits)[fold - 1]

    parts = []
    if subset in ("train", "all"):
        parts.append(train_df.iloc[train_idx].assign(cv_role="train"))
    if subset in ("val", "all"):
        parts.append(train_df.iloc[val_idx].assign(cv_role="val"))
    selected = pd.concat(parts, ignore_index=True)

    # Safety net: never include test-database segments
    assert (selected["split"] == "train").all()
    return selected


# -----------------------------
# 2. Per-EDF-file dataset
# -----------------------------


class EdfFileSegments(Dataset):
    """
    One item = all selected segments of one EDF file.

    The PPG channel is read once per file and sliced in memory, so the file
    is not reopened per segment. Failed segments are returned as NaN with a
    status string instead of raising.
    """

    def __init__(self, df: pd.DataFrame, edf_dir: str):
        self.edf_dir = edf_dir
        self.groups = [
            (key, idx.to_numpy())
            for key, idx in df.groupby(
                ["base_folder", "edf_filename"], sort=True
            ).groups.items()
        ]
        self.starts = df["start"].to_numpy()
        self.ends = df["end"].to_numpy()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int):
        (base_folder, edf_filename), rows = self.groups[i]
        edf_path = os.path.join(self.edf_dir, base_folder, edf_filename)
        starts, ends = self.starts[rows], self.ends[rows]
        seg_len = int((ends - starts).max())
        segments = np.full((len(rows), seg_len), np.nan, dtype=np.float32)
        statuses = []

        ppg_full, load_error = None, None
        if not os.path.exists(edf_path):
            load_error = "edf_missing"
        else:
            try:
                _, ppg_full, _ = read_edf_signals(
                    edf_path, start_sample=0, end_sample=None
                )
            except Exception as e:  # noqa: BLE001
                load_error = f"edf_load_failed: {type(e).__name__}"
            else:
                if ppg_full is None:
                    load_error = "edf_load_failed_or_no_ppg"

        for k, (start, end) in enumerate(zip(starts, ends)):
            if load_error is not None:
                statuses.append(load_error)
                continue
            seg = ppg_full[start:end]
            if end - start != seg_len or len(seg) != seg_len:
                statuses.append(f"too_short ({len(seg)}/{seg_len})")
            elif not np.all(np.isfinite(seg)):
                statuses.append("nan_or_inf_in_signal")
            else:
                segments[k] = seg
                statuses.append(STATUS_OK)

        return rows, segments, statuses


def _keep_as_is(item):
    """Collate function that keeps numpy arrays (no tensor conversion)."""
    return item


# -----------------------------
# 3. Inference
# -----------------------------


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module, x: np.ndarray, device: torch.device
) -> np.ndarray:
    """Normalizes like training.py, runs the model and returns HR in bpm."""
    xt = torch.from_numpy(x).unsqueeze(1).to(device)
    xt = normalize_batch(xt)
    # Training/eval used autocast (fp16) on GPU; mirror it when on CUDA
    with torch.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        out = model(xt)
    return out.float().cpu().numpy()


def run_inference(
    df: pd.DataFrame,
    model: torch.nn.Module,
    edf_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, list[str]]:
    """Predicts HR for every row of df; failed rows get NaN and a status."""
    hr_pred = np.full(len(df), np.nan, dtype=np.float64)
    status = np.full(len(df), "not_processed", dtype=object)

    loader = DataLoader(
        EdfFileSegments(df, edf_dir),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_keep_as_is,
    )

    # Buffers keyed by segment length so batches can span several files
    buffers: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    buffered: dict[int, int] = {}

    def flush(length: int) -> None:
        rows = np.concatenate([r for r, _ in buffers[length]])
        x = np.concatenate([s for _, s in buffers[length]])
        for i in range(0, len(rows), batch_size):
            hr_pred[rows[i : i + batch_size]] = predict_batch(
                model, x[i : i + batch_size], device
            )
        buffers[length], buffered[length] = [], 0

    pbar = tqdm(total=len(df), desc="Predicting", unit="seg")
    for rows, segments, statuses in loader:
        statuses = np.asarray(statuses, dtype=object)
        status[rows] = statuses
        ok = statuses == STATUS_OK
        if ok.any():
            length = segments.shape[1]
            buffers.setdefault(length, []).append((rows[ok], segments[ok]))
            buffered[length] = buffered.get(length, 0) + int(ok.sum())
            if buffered[length] >= batch_size:
                flush(length)
        pbar.update(len(rows))
    for length in list(buffers):
        if buffered[length]:
            flush(length)
    pbar.close()

    return hr_pred, status.tolist()


# -----------------------------
# 4. Output (mirror of the parquet tree)
# -----------------------------


def write_mirror(
    selected: pd.DataFrame, parquet_dir: str, output_dir: str
) -> tuple[int, int]:
    """
    Writes one output parquet per input parquet that has selected rows,
    keeping all original columns and adding the prediction columns.

    Input files without selected rows are skipped (not written).

    Returns:
        (number of files written, number of input files skipped)
    """
    pred_cols = [
        "hr_pred",
        "hr_error",
        "hr_abs_error",
        "cv_role",
        "model_fold",
        "model_path",
        "pred_status",
    ]
    all_inputs = sorted(
        os.path.relpath(p, parquet_dir)
        for p in Path(parquet_dir).glob("*/*.parquet")
    )
    by_file = dict(tuple(selected.groupby("_source_relpath", sort=False)))

    written = 0
    for relpath in tqdm(all_inputs, desc="Writing parquet", unit="file"):
        if relpath not in by_file:
            continue
        part = by_file[relpath].sort_values("_source_row")
        original = pd.read_parquet(os.path.join(parquet_dir, relpath))
        out = original.iloc[part["_source_row"].to_numpy()].reset_index(
            drop=True
        )
        for col in pred_cols:
            out[col] = part[col].to_numpy()

        out_path = Path(output_dir) / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        written += 1

    return written, len(all_inputs) - written


# -----------------------------
# 5. Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict HR on training-DB segments of one CV fold."
    )
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--edf-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="1-based CV fold of the checkpoint (as in training.py)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset", choices=["train", "val", "all"], default="train"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits used in training (must match!)",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", *MODEL_REGISTRY.keys()],
        help="Architecture of the checkpoint; 'auto' detects it",
    )
    parser.add_argument(
        "--target-col",
        choices=["feasible_bpm", "hr_bpm"],
        default="feasible_bpm",
        help="Training target of the checkpoint (must match training.py"
        " --target_col, otherwise the folds differ)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug: random subset of N selected segments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if Path(args.output_dir).resolve() == Path(args.parquet_dir).resolve():
        raise ValueError("--output-dir must differ from --parquet-dir")

    device = torch.device(args.device)

    # Load the model first so a wrong checkpoint fails before data loading
    model, model_name = load_model(args.model_path, args.model, device)
    print(f"Model: {model_name} ({args.model_path})")

    print(f"Selecting fold {args.fold}/{args.n_splits} ({args.subset})...")
    selected = select_fold_segments(
        args.parquet_dir, args.n_splits, args.fold, args.subset,
        args.target_col,
    )
    if args.limit is not None and args.limit < len(selected):
        selected = selected.sample(n=args.limit, random_state=0)
        selected = selected.reset_index(drop=True)
        print(f"--limit: using {len(selected)} random segments")
    print(
        f"Selected segments: {len(selected)} "
        f"{selected['cv_role'].value_counts().to_dict()}"
    )

    hr_pred, status = run_inference(
        selected,
        model,
        args.edf_dir,
        device,
        args.batch_size,
        args.num_workers,
    )

    selected["hr_pred"] = hr_pred
    selected["hr_error"] = selected["hr_pred"] - selected["hr_bpm"]
    selected["hr_abs_error"] = selected["hr_error"].abs()
    selected["model_fold"] = args.fold
    selected["model_path"] = str(args.model_path)
    selected["pred_status"] = status

    written, skipped = write_mirror(
        selected, args.parquet_dir, args.output_dir
    )

    # --- Summary ---
    failed = selected["hr_pred"].isna()
    print("\n" + "=" * 40)
    print(f"Fold {args.fold} | subset={args.subset} | model={model_name}")
    print(f"Segments: {len(selected)}  Failed: {int(failed.sum())}")
    if failed.any():
        reasons = Counter(selected.loc[failed, "pred_status"])
        for reason, n in reasons.most_common():
            print(f"  {reason}: {n}")
    for role, g in [("all", selected)] + list(selected.groupby("cv_role")):
        err = g["hr_error"].dropna()
        if len(err):
            print(
                f"[{role:5s}] N={len(err):7d}  MAE={err.abs().mean():.3f} bpm  "
                f"RMSE={np.sqrt((err**2).mean()):.3f} bpm"
            )
    print(
        f"Parquet files written: {written}, "
        f"skipped (no selected rows): {skipped}"
    )
    print(f"Output: {args.output_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()
