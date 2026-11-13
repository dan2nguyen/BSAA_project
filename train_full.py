"""
Trains the final model on ALL training segments (no validation split).

The settings were chosen with the 5-fold cross-validation (training.py);
this script then trains one model on the whole training split with those
settings, for a fixed number of epochs. The result is the model that is
evaluated once on the test split (evaluate_test.py).

Same data, model, loss weights and optimizer as training.py. Without a
validation set, the learning-rate schedule uses the training loss and the
model after the last epoch is saved.

Usage (Kaggle / repo root):
    python prototyping_dan/train_full.py --db_path <hr_parts/seg10s_step5s>
        --edf_dir <harmonized_ecg_ppg_database_filtered>
        --output_dir training_full --epochs 15 --lr 1e-3
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prototyping_dan.training import (  # noqa: E402
    add_training_args,
    load_segment_array,
    load_training_segments,
    plot_learning_curves,
    train_fold,
)


def main():
    p = argparse.ArgumentParser(description="Final model on all training data")
    add_training_args(p)
    p.add_argument("--db_path", required=True)
    p.add_argument("--edf_dir", required=True)
    p.add_argument("--output_dir", default="training_full")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--target_col", default="feasible_bpm",
                   choices=["feasible_bpm", "hr_bpm"])
    p.set_defaults(epochs=15, min_epochs=15)
    args = p.parse_args()
    # Fixed number of epochs: no early stopping
    args.min_epochs = args.epochs

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_df = load_training_segments(args.db_path, n_splits=5,
                                      target_col=args.target_col)
    segments, ok = load_segment_array(train_df, args.edf_dir, args.workers,
                                      args.cache_dir)
    X = torch.from_numpy(segments).to(device)
    y = torch.tensor(train_df["hr_bpm"].to_numpy(), dtype=torch.float32,
                     device=device)
    del segments

    idx = ok.nonzero()[0]
    print(f"Training on all {len(idx)} training segments,"
          f" {train_df['cv_group'].nunique()} subjects, {args.epochs} epochs")

    # train_fold needs a validation set: the training set itself is used,
    # so the schedule follows the training loss and the last (lowest) epoch
    # is kept. These 'Val' numbers are training numbers, not a test.
    result = train_fold(X, y, train_df["hr_bpm"].to_numpy(), idx, idx,
                        args, device, device)

    out = Path(args.output_dir) / args.model / "full"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result["state_dict"], out / "best_model.pt")
    plot_learning_curves(result["history"], out / "learning_curves_full.png")
    print(f"Saved {out / 'best_model.pt'} (epoch {result['best_epoch']}"
          f" of {result['epochs_run']}, {result['seconds'] / 60:.0f} min)")


if __name__ == "__main__":
    main()
