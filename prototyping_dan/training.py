import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
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
    """
    Original CNN. Receptive field is only 29 samples (0.29 s at 100 Hz),
    i.e. shorter than one beat interval. Kept to load old checkpoints.
    """

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


class DilatedResidualBlock(nn.Module):
    """Two dilated convolutions with a residual connection (same length)."""

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.body(x))


class DilatedCNN1D(nn.Module):
    """
    Residual CNN with exponentially growing dilation.

    The stem halves the time resolution (100 Hz -> 50 Hz), then blocks with
    dilations 1, 2, 4, ..., 64 give a receptive field of ~1030 samples,
    i.e. every feature sees the whole 10 s window (6-30 beats), so the model
    can use the beat-to-beat periodicity instead of counting local peaks.

    The output is de-normalized (hr_mean + hr_std * head), so the network
    learns a standardized target but returns HR in bpm like the other models.
    """

    def __init__(
        self,
        in_channels=1,
        channels=64,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.1,
        hr_mean=80.0,
        hr_std=20.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[DilatedResidualBlock(channels, d, dropout=dropout)
              for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Start from predicting hr_mean
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("hr_mean", torch.tensor(float(hr_mean)))
        self.register_buffer("hr_std", torch.tensor(float(hr_std)))

    def forward(self, x):
        x = self.blocks(self.stem(x))
        # Average + max pooling over time
        x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.hr_mean + self.hr_std * self.head(x).squeeze(-1)


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
    "dilated_cnn": DilatedCNN1D,
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
# 3. Data: segment selection and in-memory segment array
# -----------------------------


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """Per-segment z-score normalization along the time axis."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def load_training_segments(
    db_path: str,
    n_splits: int,
    keep_source_info: bool = False,
    target_col: str = "feasible_bpm",
) -> pd.DataFrame:
    """
    Loads the train-split segments and assigns patient-wise CV groups.

    Only rows with a non-NaN target_col are kept, and 'hr_bpm' is replaced
    by it, so everything downstream trains on target_col. With the default
    'feasible_bpm' (see feasible_bpm.py) these are the segments inside the
    ECG/PPG quality borders of the fold report.
    """
    columns = ["base_folder", "edf_filename", "start", "end", "hr_bpm",
               "split"]
    if target_col not in columns:
        columns.append(target_col)
    train_df, _ = load_parquet_files(
        db_path,
        columns=columns,
        split_stats=True,
        keep_source_info=keep_source_info,
    )
    n_before = len(train_df)
    train_df = train_df[train_df[target_col].notna()].reset_index(drop=True)
    train_df["hr_bpm"] = train_df[target_col]
    print(f"Target '{target_col}': kept {len(train_df)}/{n_before}"
          f" train segments ({len(train_df) / max(n_before, 1):.1%})")
    return add_cv_groups(train_df, n_folds=n_splits)


def make_cv_splits(
    train_df: pd.DataFrame, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-wise stratified K-fold split; returns (train_idx, val_idx) per fold."""
    # Create targets for StratifiedGroupKFold
    y_cv = pd.qcut(train_df["hr_bpm"], q=5, labels=False)
    groups_cv = train_df["cv_group"]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )
    return list(sgkf.split(train_df, y_cv, groups=groups_cv))


def make_lds_weighter(
    targets: np.ndarray,
    alpha: float,
    sigma: float,
    max_weight: float,
    bin_width: float = 1.0,
):
    """
    Label distribution smoothing (Yang et al., 2021) for imbalanced
    regression: the HR histogram of the training targets is smoothed with
    a Gaussian, and each sample gets weight density(hr) ** -alpha
    (alpha = 0.5: inverse square root, 1: inverse frequency).

    Weights are normalized to mean 1 on `targets` and capped at max_weight,
    so the loss scale and the learning rate stay comparable to unweighted
    training.

    Args:
        targets: HR of the training samples (bpm), used for the density.
        alpha: Strength of the reweighting.
        sigma: Gaussian smoothing of the histogram in bpm.
        max_weight: Upper cap of a weight (relative to the mean of 1).
        bin_width: Histogram bin width in bpm.

    Returns:
        Function hr array -> weight array (also for HR outside the
        training range, e.g. validation samples).
    """
    from scipy.ndimage import gaussian_filter1d

    edges = np.arange(np.floor(targets.min()),
                      np.ceil(targets.max()) + bin_width, bin_width)
    counts, _ = np.histogram(targets, bins=edges)
    density = gaussian_filter1d(counts.astype(np.float64),
                                sigma / bin_width, mode="constant")
    centers = (edges[:-1] + edges[1:]) / 2
    floor = density[density > 0].min()

    def raw(hr):
        d = np.interp(hr, centers, density, left=floor, right=floor)
        return np.maximum(d, floor) ** -alpha

    scale = 1.0 / raw(targets).mean()
    # Rescale once after capping so the mean is ~1 again
    scale /= np.minimum(raw(targets) * scale, max_weight).mean()

    def weigh(hr):
        return np.minimum(raw(hr) * scale, max_weight).astype(np.float32)

    return weigh


def per_bin_errors(
    targets: np.ndarray, preds: np.ndarray, width: float = 10.0
) -> tuple[pd.DataFrame, float]:
    """
    MAE and bias per HR bin of the targets, and the balanced MAE (mean of
    the per-bin MAEs), which weighs every HR range equally.
    """
    df = pd.DataFrame({"target": targets, "err": preds - targets})
    df["bin"] = (np.floor(df["target"] / width) * width).astype(int)
    table = df.groupby("bin").agg(
        n=("err", "size"),
        mae=("err", lambda e: e.abs().mean()),
        bias=("err", "mean"),
    )
    return table, float(table["mae"].mean())


def _load_file_segments(task):
    """
    Worker: reads the PPG of one EDF file once and returns its segments,
    z-scored like `normalize_batch`, as float16.

    Returns:
        (row positions, segments [k, seg_len] float16, ok mask [k])
    """
    edf_path, rows, starts, ends, seg_len = task
    segments = np.zeros((len(rows), seg_len), dtype=np.float16)
    ok = np.zeros(len(rows), dtype=bool)
    try:
        _, ppg, _ = read_edf_signals(edf_path)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading {edf_path}: {e}")
        ppg = None
    if ppg is None:
        return rows, segments, ok

    for k, (start, end) in enumerate(zip(starts, ends)):
        seg = ppg[start:end].astype(np.float64)
        if len(seg) != seg_len or not np.all(np.isfinite(seg)):
            continue
        seg = (seg - seg.mean()) / (seg.std(ddof=1) + 1e-8)
        segments[k] = seg.astype(np.float16)
        ok[k] = True
    return rows, segments, ok


def _segments_key(df: pd.DataFrame) -> str:
    """Hash of the segment list, to check that a cache matches the rows."""
    keys = df[["base_folder", "edf_filename", "start", "end"]]
    return hashlib.md5(
        pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()
    ).hexdigest()


def load_segment_array(
    df: pd.DataFrame,
    edf_dir: str,
    workers: int,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads all segments of df into one array, reading every EDF file only once
    (in parallel). Segments are z-scored per segment and stored as float16
    (~2 KB per 10 s segment at 100 Hz).

    Args:
        df: Segment rows (base_folder, edf_filename, start, end).
        edf_dir: Root folder of the EDF files.
        workers: Number of parallel reader processes.
        cache_dir: If given, the array is saved there and reused on the next
            run with the same segment list.

    Returns:
        (segments [N, seg_len] float16, ok mask [N] bool)
    """
    seg_len = int((df["end"] - df["start"]).max())
    key = _segments_key(df)
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"segments_{key}.npz"
        if cache_path.exists():
            print(f"Loading segment cache {cache_path}")
            data = np.load(cache_path)
            return data["segments"], data["ok"]

    segments = np.zeros((len(df), seg_len), dtype=np.float16)
    ok = np.zeros(len(df), dtype=bool)
    starts, ends = df["start"].to_numpy(), df["end"].to_numpy()
    # One task per EDF file; idx are row positions in df
    groups = df.groupby(["base_folder", "edf_filename"]).indices
    tasks = [
        (os.path.join(edf_dir, folder, fname), idx, starts[idx], ends[idx],
         seg_len)
        for (folder, fname), idx in groups.items()
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows, segs, file_ok in tqdm(
            pool.map(_load_file_segments, tasks),
            total=len(tasks), desc="Reading EDF files", unit="file",
        ):
            segments[rows] = segs
            ok[rows] = file_ok
    print(
        f"Loaded {ok.sum()}/{len(df)} segments from {len(tasks)} files in"
        f" {time.perf_counter() - t0:.0f} s ({segments.nbytes / 1e9:.2f} GB)"
    )

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, segments=segments, ok=ok)
        print(f"Saved segment cache to {cache_path}")
    return segments, ok


# -----------------------------
# 4. Train / Eval Steps
# -----------------------------


def autocast_ctx(device: torch.device):
    """Mixed precision (fp16) on CUDA, no-op on CPU."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def iterate_batches(idx: torch.Tensor, batch_size: int, shuffle: bool):
    """Yields index batches (on the device of idx)."""
    if shuffle:
        idx = idx[torch.randperm(len(idx), device=idx.device)]
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def get_batch(X, y, batch_idx, device):
    """Gathers a batch from the in-memory arrays -> ([B, 1, T] fp32, [B])."""
    xb = X[batch_idx.to(X.device)].to(device, non_blocking=True)
    yb = y[batch_idx.to(y.device)].to(device, non_blocking=True)
    # Segments are already z-scored; fp16 -> fp32 for the model input
    return xb.float().unsqueeze(1), yb


def weighted_loss(criterion, out, yb, wb):
    """Mean of the per-sample loss, weighted by wb if given."""
    loss = criterion(out, yb)
    return (loss * wb).mean() if wb is not None else loss.mean()


def train_one_epoch(model, X, y, idx, batch_size, criterion, optimizer,
                    device, scaler, w=None):
    """w: per-sample loss weights (same indexing as y) or None."""
    model.train()
    # Metrics are accumulated on the GPU, synced once per epoch
    sums = torch.zeros(3, device=device)  # loss*n, abs error, within 5 bpm
    n_batches = (len(idx) + batch_size - 1) // batch_size
    pbar = tqdm(iterate_batches(idx, batch_size, shuffle=True),
                total=n_batches, desc="  Training", leave=False)
    for step, batch_idx in enumerate(pbar):
        xb, yb = get_batch(X, y, batch_idx, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            out = model(xb)
        out = out.float()
        wb = (w[batch_idx.to(w.device)].to(device, non_blocking=True)
              if w is not None else None)
        loss = weighted_loss(criterion, out, yb, wb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            err = (out - yb).abs()
            sums += torch.stack(
                [loss.detach() * len(yb), err.sum(), (err <= 5.0).sum()]
            )
        if step % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.2f}"})

    loss_sum, err_sum, acc_sum = sums.tolist()
    n = len(idx)
    return loss_sum / n, err_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(model, X, y, idx, batch_size, criterion, device, w=None):
    """w: per-sample loss weights (same indexing as y) or None."""
    model.eval()
    preds = []
    for batch_idx in tqdm(iterate_batches(idx, batch_size, shuffle=False),
                          total=(len(idx) + batch_size - 1) // batch_size,
                          desc="  Evaluating", leave=False):
        xb, _ = get_batch(X, y, batch_idx, device)
        with autocast_ctx(device):
            preds.append(model(xb).float())
    preds = torch.cat(preds)
    targets = y[idx.to(y.device)].to(device)

    err = (preds - targets).abs()
    wv = w[idx.to(w.device)].to(device) if w is not None else None
    metrics = (
        weighted_loss(criterion, preds, targets, wv).item(),
        err.mean().item(),
        (err <= 5.0).float().mean().item(),
    )
    return metrics, preds.cpu().numpy(), targets.cpu().numpy()


def train_fold(X, y, hr, train_idx, val_idx, args, device, store):
    """
    Trains one model on the rows train_idx of (X, y) with early stopping on
    the validation loss of the rows val_idx. Used by main() and by
    sweep_borders.py, so both train exactly the same way.

    Args:
        X, y: Segment array [N, T] and targets [N] (on `store`).
        hr: Targets [N] as numpy (for the loss weights).
        train_idx, val_idx: numpy row indices of the loadable segments.
        args: Namespace with the options of add_training_args().
        device: Training device; store: device of X and y.

    Returns:
        dict with model (best weights loaded), state_dict (best, on CPU),
        history, best_val_loss, best_epoch, epochs_run, seconds,
        criterion and w (loss weights tensor or None).
    """
    tr = torch.from_numpy(train_idx).to(store)
    va = torch.from_numpy(val_idx).to(store)
    print(f"Train segments: {len(tr)}, val segments: {len(va)}")

    # Dynamic model instantiation
    model = build_model(args.model, in_channels=1).to(device)
    train_model = torch.compile(model) if args.compile else model
    criterion = (
        nn.HuberLoss(delta=5.0, reduction="none")
        if args.loss == "huber" else nn.MSELoss(reduction="none")
    )

    # Loss weights from the HR distribution of this fold's train part
    w = None
    if args.reweight != "none":
        alpha = {"sqrt_inv": 0.5, "inv": 1.0}[args.reweight]
        weigh = make_lds_weighter(
            hr[train_idx], alpha, args.lds_sigma, args.max_weight,
        )
        w = torch.from_numpy(weigh(hr)).to(store)
        for v in (50, 60, 80, 100, 120, 140):
            print(f"  weight at {v} bpm: {weigh(np.array([v]))[0]:.2f}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda"
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "train_mae": [], "train_acc": [],
        "val_loss": [], "val_mae": [], "val_acc": [],
        "epoch_seconds": [],
    }
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        t_loss, t_mae, t_acc = train_one_epoch(
            train_model, X, y, tr, args.batch_size, criterion,
            optimizer, device, scaler, w,
        )
        (v_loss, v_mae, v_acc), v_preds, v_targets = evaluate(
            train_model, X, y, va, args.batch_size, criterion, device, w
        )
        _, v_bmae = per_bin_errors(v_targets, v_preds)

        history["train_loss"].append(t_loss)
        history["train_mae"].append(t_mae)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(v_mae)
        history["val_acc"].append(v_acc)
        history["epoch_seconds"].append(time.perf_counter() - t0)

        scheduler.step(v_loss)

        print(
            f"Epoch {epoch:02d} | Train MAE: {t_mae:.3f} Acc: {t_acc:.2%}"
            f" | Val MAE: {v_mae:.3f} bMAE: {v_bmae:.3f} Acc: {v_acc:.2%}"
            f" | Val Loss: {v_loss:.4f}"
            f" | lr {optimizer.param_groups[0]['lr']:.1e}"
            f" | {time.perf_counter() - t0:.0f} s"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            # Uncompiled module, so the checkpoint loads anywhere
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "state_dict": best_state,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "seconds": time.perf_counter() - t_start,
        "criterion": criterion,
        "w": w,
    }


def plot_learning_curves(history, save_path):
    """Plots Train vs Val Loss, MAE, and Accuracy."""
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("loss", "Loss"),
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


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Model / optimization options (shared with sweep_borders.py)."""
    parser.add_argument(
        "--model",
        type=str,
        default="dilated_cnn",
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", "--batch-size", type=int,
                        default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "--weight-decay", type=float,
                        default=1e-4)
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
        help="huber is less sensitive to wrong labels (delta = 5 bpm)",
    )
    parser.add_argument(
        "--reweight",
        choices=["sqrt_inv", "inv", "none"],
        default="sqrt_inv",
        help="Loss weights against HR imbalance (label distribution"
        " smoothing): sqrt_inv = density^-0.5, inv = density^-1",
    )
    parser.add_argument(
        "--lds_sigma", type=float, default=3.0,
        help="Gaussian smoothing of the HR histogram for --reweight (bpm)",
    )
    parser.add_argument(
        "--max_weight", type=float, default=10.0,
        help="Cap of a loss weight (weights have mean 1)",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (needs a GPU with compute capability >= 7.0)",
    )
    parser.add_argument("--seed", type=int, default=42)


def main():
    parser = argparse.ArgumentParser(
        description="Signal NN Training with Patient-Wise CV and Real Data"
    )
    add_training_args(parser)
    parser.add_argument(
        "--target_col",
        choices=["feasible_bpm", "hr_bpm"],
        default="feasible_bpm",
        help="feasible_bpm: only segments inside the quality borders"
        " (NaN rows dropped); hr_bpm: all valid segments (old behaviour)",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="1-based folds to train, e.g. '1' or '1,2' (data split unchanged)",
    )

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
        "--output_dir",
        type=str,
        default="training_outputs",
        help="Folder for checkpoints, curves and OOF predictions",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Folder to cache the loaded segments (reused on the next run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Processes for reading the EDF files",
    )
    parser.add_argument(
        "--data_on_cpu",
        action="store_true",
        help="Keep the segment array in CPU RAM instead of GPU memory",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load Real Data ---
    print(f"Loading database from {args.db_path}...")
    # Train split with patient-wise CV groups
    train_df = load_training_segments(
        args.db_path, args.n_splits, target_col=args.target_col
    )

    # --- Load all segments once into memory ---
    segments, ok = load_segment_array(
        train_df, args.edf_dir, args.workers, args.cache_dir
    )
    store = torch.device("cpu") if args.data_on_cpu else device
    X = torch.from_numpy(segments).to(store)
    if store.type == "cpu" and device.type == "cuda":
        X = X.pin_memory()
    y = torch.tensor(train_df["hr_bpm"].to_numpy(), dtype=torch.float32,
                     device=store)
    del segments
    if not ok.all():
        print(f"Warning: {int((~ok).sum())} segments could not be loaded"
              " and are skipped")

    if args.folds.lower() == "all":
        selected_folds = set(range(1, args.n_splits + 1))
    else:
        selected_folds = {int(f) for f in args.folds.split(",")}

    all_oof_dfs = []

    # --- Patient-Wise Stratified K-Fold ---
    for fold, (train_idx, val_idx) in enumerate(
        make_cv_splits(train_df, args.n_splits)
    ):
        if fold + 1 not in selected_folds:
            continue
        print(
            f"\n========== Fold {fold + 1}/{args.n_splits} =========="
        )

        # Indices of loadable segments, kept on the device of the data
        train_idx = train_idx[ok[train_idx]]
        val_idx = val_idx[ok[val_idx]]

        # Create structured output directory
        output_dir = Path(args.output_dir) / args.model / f"fold_{fold + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = train_fold(
            X, y, train_df["hr_bpm"].to_numpy(), train_idx, val_idx,
            args, device, store,
        )
        model, history = result["model"], result["history"]
        criterion, w = result["criterion"], result["w"]
        torch.save(result["state_dict"], output_dir / "best_model.pt")
        va = torch.from_numpy(val_idx).to(store)

        # Save learning curves for the fold
        plot_learning_curves(
            history, output_dir / f"learning_curves_fold_{fold + 1}.png"
        )

        # OOF Collection: Load best model and get predictions for the fold
        (_, _, _), fold_preds, fold_targets = evaluate(
            model, X, y, va, args.batch_size, criterion, device, w
        )

        fold_results = train_df.iloc[val_idx][
            ["base_folder", "edf_filename", "start", "end", "cv_group"]
        ].reset_index(drop=True)
        fold_results["target_hr"] = fold_targets
        fold_results["predicted_hr"] = fold_preds
        fold_results["fold"] = fold + 1
        all_oof_dfs.append(fold_results)
        print(
            f"Fold {fold + 1} complete. Model and plots saved to {output_dir}"
        )

    # --- Final OOF Summary ---
    if all_oof_dfs:
        oof_df = pd.concat(all_oof_dfs, ignore_index=True)
        oof_save_path = (
            Path(args.output_dir) / args.model
            / f"oof_predictions_{args.model}.csv"
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
        bins, bmae = per_bin_errors(y_true.to_numpy(), y_pred.to_numpy())
        print(f"Balanced MAE (mean over 10 bpm bins): {bmae:.3f} BPM")
        print(bins.round(3).to_string())
        print(f"OOF Predictions saved to: {oof_save_path}")
        print("=" * 30)


if __name__ == "__main__":
    main()
