import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Use tqdm for progress bars
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
# Import database loading utilities
from prototyping_dan.load_database import (
    add_cv_groups,
    load_parquet_files,
)

# Import the optimized EDF reader
from utils.read_edf import read_edf_signals

# -----------------------------
# 1. Modular Architectures
# -----------------------------


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            1,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock(32, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)


class BiLSTM1D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=64,
        num_layers=2,
    ):
        super().__init__()
        # CRNN Architecture: CNN front-end for local feature extraction
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(
                in_channels, 32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # PyTorch LSTM expects (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # 1. CNN extraction: (Batch, 1, Time) -> (Batch, 64, Time')
        x = self.cnn_frontend(x)
        # 2. Transpose for LSTM: (Batch, 64, Time') -> (Batch, Time', 64)
        x = x.transpose(1, 2)
        # 3. LSTM processing
        out, _ = self.lstm(x)
        # 4. Use final hidden state for regression
        return self.fc(out[:, -1, :]).squeeze(-1)


# -----------------------------
# 2. Model Factory
# -----------------------------

MODEL_REGISTRY = {
    "cnn1d": SimpleCNN1D,
    "bilstm": BiLSTM1D,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


# -----------------------------
# 3. Dataset Template
# -----------------------------


class SignalDataset(Dataset):
    def __init__(self, df, base_dir):
        """
        Lazy-loading dataset for signal segments.

        Args:
            df (pd.DataFrame): DataFrame containing segment metadata.
            base_dir (str): Base directory where EDF files are stored.
        """
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path to the EDF file
        edf_path = os.path.join(
            self.base_dir, row["base_folder"], row["edf_filename"]
        )

        # Lazily load only the required segment
        # We assume we use PPG for training (in_channels=1)
        _, ppg_segment, _ = read_edf_signals(
            edf_path,
            start_sample=row["start"],
            end_sample=row["end"],
        )

        # if ppg_segment is None:
        #     # Fallback for corrupted/missing files: return a zero vector
        #     ppg_segment = np.zeros(500, dtype=np.float32)
        # else:
        #     # --- 1. Length Enforcement (Crop/Pad to 500) ---
        #     target_len = 500
        #     if len(ppg_segment) > target_len:
        #         ppg_segment = ppg_segment[:target_len]
        #     elif len(ppg_segment) < target_len:
        #         ppg_segment = np.pad(
        #             ppg_segment,
        #             (0, target_len - len(ppg_segment)),
        #             mode="constant",
        #         )

        # --- 2. Z-Score Normalization ---
        # (x - mean) / (std + epsilon)
        mean = np.mean(ppg_segment)
        std = np.std(ppg_segment)
        ppg_segment = (ppg_segment - mean) / (std + 1e-8)

        # Target: Exact HR value for regression
        label = row["hr_bpm"]

        # Return as tensors (Shape: [1, Time])
        signal_tensor = torch.tensor(
            ppg_segment, dtype=torch.float32
        ).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return signal_tensor, label_tensor


# -----------------------------
# 4. Train / Eval Steps
# -----------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_error, total_acc, total = 0.0, 0.0, 0.0, 0
    # Wrap loader in tqdm for per-batch progress
    pbar = tqdm(loader, desc="  Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * len(y)
        total_error += torch.abs(out - y).sum().item()

        # Accuracy: % within 5 BPM
        acc = (torch.abs(out - y) <= 5.0).float().mean().item()
        total_acc += acc * len(y)

        total += len(y)

        # Update progress bar with current batch loss
        pbar.set_postfix(
            {"loss": f"{batch_loss:.4f}", "acc": f"{acc:.2%}"}
        )

    return total_loss / total, total_error / total, total_acc / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_error, total_acc, total = 0.0, 0.0, 0.0, 0
    all_preds = []
    all_targets = []
    # Wrap loader in tqdm for per-batch progress
    pbar = tqdm(loader, desc="  Evaluating", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        batch_loss = loss.item()
        total_loss += batch_loss * len(y)
        total_error += torch.abs(out - y).sum().item()

        # Accuracy: % within 5 BPM
        acc = (torch.abs(out - y) <= 5.0).float().mean().item()
        total_acc += acc * len(y)

        total += len(y)

        all_preds.append(out.cpu().numpy())
        all_targets.append(y.cpu().numpy())

        # Update progress bar with current batch loss
        pbar.set_postfix(
            {"loss": f"{batch_loss:.4f}", "acc": f"{acc:.2%}"}
        )

    metrics = (
        total_loss / total,
        total_error / total,
        total_acc / total,
    )
    return (
        metrics,
        np.concatenate(all_preds).flatten(),
        np.concatenate(all_targets).flatten(),
    )


def plot_learning_curves(history, save_path):
    """Plots Train vs Val Loss, MAE, and Accuracy."""
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("loss", "MSE Loss"),
        ("mae", "Mean Absolute Error (BPM)"),
        ("acc", "Accuracy (within 5 BPM)"),
    ]

    for i, (key, label) in enumerate(metrics):
        axes[i].plot(
            history[f"train_{key}"], label="Train", color="blue"
        )
        axes[i].plot(history[f"val_{key}"], label="Val", color="red")
        axes[i].set_title(label)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# 5. Main Execution
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Signal NN Training with Patient-Wise CV and Real Data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_splits", type=int, default=5)

    # Data paths
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to Parquet files folder",
    )
    parser.add_argument(
        "--edf_dir",
        type=str,
        required=True,
        help="Root directory for EDF files",
    )
    parser.add_argument(
        "--hr_threshold",
        type=float,
        default=100.0,
        help="HR threshold for labels",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Load Real Data ---
    print(f"Loading database from {args.db_path}...")
    train_df, _ = load_parquet_files(
        args.db_path,
        columns=[
            "base_folder",
            "edf_filename",
            "start",
            "end",
            "hr_bpm",
            "split",
        ],
        split_stats=True,
    )

    # Assign patient-wise CV groups
    train_df = add_cv_groups(train_df, n_folds=args.n_splits)

    # Create targets for StratifiedGroupKFold
    y_cv = pd.qcut(train_df["hr_bpm"], q=5, labels=False)
    groups_cv = train_df["cv_group"]

    # --- Patient-Wise Stratified K-Fold ---
    sgkf = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=42
    )

    all_oof_dfs = []

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(train_df, y_cv, groups=groups_cv)
    ):
        print(
            f"\n========== Fold {fold + 1}/{args.n_splits} =========="
        )

        # Sliced DataFrames for the current fold
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        # Lazy loading datasets
        train_ds = SignalDataset(train_fold_df, args.edf_dir)
        val_ds = SignalDataset(val_fold_df, args.edf_dir)

        # Use num_workers > 0 to parallelize EDF reading
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Dynamic model instantiation
        model = build_model(args.model, in_channels=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # Create structured output directory
        output_dir = Path(
            f"training_outputs/{args.model}/fold_{fold + 1}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        patience_limit = 5

        # History tracking for plotting
        history = {
            "train_loss": [],
            "train_mae": [],
            "train_acc": [],
            "val_loss": [],
            "val_mae": [],
            "val_acc": [],
        }

        for epoch in range(1, args.epochs + 1):
            t_loss, t_mae, t_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            (v_loss, v_mae, v_acc), v_preds, v_targets = evaluate(
                model, val_loader, criterion, device
            )

            # Record history
            history["train_loss"].append(t_loss)
            history["train_mae"].append(t_mae)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_mae"].append(v_mae)
            history["val_acc"].append(v_acc)

            # Scheduler step
            scheduler.step(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(
                    model.state_dict(),
                    output_dir / "best_model.pt",
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            print(
                f"Epoch {epoch:02d} | Train MAE: {t_mae:.3f} Acc: {t_acc:.2%} | Val MAE: {v_mae:.3f} Acc: {v_acc:.2%} | Val Loss: {v_loss:.4f}"
            )

        # Save learning curves for the fold
        plot_learning_curves(
            history, output_dir / f"learning_curves_fold_{fold + 1}.png"
        )

        # OOF Collection: Load best model and get predictions for the fold
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        (_, _, _), fold_preds, fold_targets = evaluate(
            model, val_loader, criterion, device
        )

        fold_results = pd.DataFrame(
            {
                "target_hr": fold_targets,
                "predicted_hr": fold_preds,
                "fold": fold + 1,
            }
        )
        all_oof_dfs.append(fold_results)
        print(
            f"Fold {fold + 1} complete. Model and plots saved to {output_dir}"
        )

    # --- Final OOF Summary ---
    if all_oof_dfs:
        oof_df = pd.concat(all_oof_dfs, ignore_index=True)
        oof_save_path = Path(
            f"training_outputs/{args.model}/oof_predictions_{args.model}.csv"
        )
        oof_df.to_csv(oof_save_path, index=False)

        y_true = oof_df["target_hr"]
        y_pred = oof_df["predicted_hr"]

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        acc = np.mean(np.abs(y_true - y_pred) <= 5.0)

        print("\n" + "=" * 30)
        print(f"FINAL CV SUMMARY ({args.model})")
        print("=" * 30)
        print(f"Overall MAE: {mae:.3f} BPM")
        print(f"Overall RMSE: {rmse:.3f} BPM")
        print(f"Overall Acc (±5 BPM): {acc:.2%}")
        print(f"OOF Predictions saved to: {oof_save_path}")
        print("=" * 30)


if __name__ == "__main__":
    main()
