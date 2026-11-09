"""
Predicts heart rate from the PPG of one EDF recording, or NONE where the
signal is not feasible.

Pipeline (each step reuses the code that built the training data):
  1. preprocessing: utils/filt_data_folder.filter_edf (notch/band-pass
     filters + EDF export/re-read, i.e. the same 16-bit quantization as the
     filtered database),
  2. segmentation: 10 s windows with 5 s step, start = 0, 500, 1000, ...
     samples (as databasing/pan_tompkins/build_hr_database.py),
  3. quality metrics per segment: compute_metrics of
     databasing/evaluation_signal_segments.py via build_hr_database's
     compute_segment_metrics (flags on the raw segment, values on the
     z-scored segment),
  4. feasibility: the borders of prototyping_dan/feasible_bpm.py, which
     define the training target of training.py:
       --feasibility ppg  (default) not ppg_is_flat and inside PPG_BOUNDS;
                          needs only the PPG, as in real use
       --feasibility full not is_flat and inside FEASIBILITY_BOUNDS
                          (ECG + PPG borders, exactly the training
                          selection; needs the ECG channel)
  5. model: per-segment z-score -> float16 (the training segment cache)
     -> trained network (training.py checkpoint).

Output per segment: HR in bpm, or NONE with the failed metrics.
The recording HR is the median over the feasible segments (or NONE).

Usage:
    python predict.py recording.edf
    python predict.py recording.edf --model-path data/training_out_cnn/dilated_cnn/fold_2/best_model.pt
    python predict.py recording.edf --feasibility full --output preds.csv
    python predict.py data/harmonized_ecg_ppg_database_filtered/x/y.edf --input-filtered
"""

import argparse
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from databasing.pan_tompkins.build_hr_database import (
    compute_segment_metrics,
    normalize_segment,
)
from prototyping_dan.feasible_bpm import FEASIBILITY_BOUNDS, PPG_BOUNDS
from prototyping_dan.predict_training_segments import load_model
from prototyping_dan.training import MODEL_REGISTRY, autocast_ctx
from utils import read_edf
from utils.filt_data_folder import filter_edf

DEFAULT_MODEL = (
    ROOT / "data" / "training_out_cnn" / "dilated_cnn" / "fold_1"
    / "best_model.pt"
)
MODEL_FS = 100  # sampling rate of the training database
SEGMENT_S = 10.0
STEP_S = 5.0
FEASIBILITY_MODES = ("ppg", "full")


# -----------------------------
# 1. Preprocessing
# -----------------------------


def _read_and_forget(path: Path, with_ecg: bool):
    """read_edf_signals without keeping the file in its Raw cache."""
    ecg, ppg, meta = read_edf.read_edf_signals(
        str(path), ecg_keywords=None if with_ecg else []
    )
    read_edf._RAW_CACHE.pop(str(path), None)
    return ecg, ppg, meta


def preprocess_edf(
    edf_path: str | Path, with_ecg: bool = False,
    input_filtered: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """
    Reads one EDF file and preprocesses it like the filtered database.

    Args:
        edf_path: Raw recording (as in data/harmonized_ecg_ppg_databases).
        with_ecg: Also return the filtered ECG (for --feasibility full).
        input_filtered: The file is already from the filtered database;
            it is only read.

    Returns:
        (ecg or None, ppg, fs); resampled to MODEL_FS if needed.
    """
    edf_path = Path(edf_path)
    if input_filtered:
        ecg, ppg, meta = _read_and_forget(edf_path, with_ecg)
    else:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            filtered = Path(tmp) / f"{edf_path.stem}_filtered.edf"
            if not filter_edf(str(edf_path), str(filtered),
                              require_ecg=False):
                raise ValueError(f"{edf_path}: no PPG channel")
            read_edf._RAW_CACHE.pop(str(edf_path), None)
            ecg, ppg, meta = _read_and_forget(filtered, with_ecg)
    if ppg is None:
        raise ValueError(f"{edf_path}: no PPG channel")
    if with_ecg and ecg is None:
        raise ValueError(
            f"{edf_path}: no ECG channel, use --feasibility ppg")

    fs = float(meta["sampling_rate"])
    if round(fs) != MODEL_FS:
        # The model and the borders were made at 100 Hz
        g = gcd(int(round(fs)), MODEL_FS)
        up, down = MODEL_FS // g, int(round(fs)) // g
        ppg = scipy.signal.resample_poly(ppg, up, down)
        if ecg is not None:
            ecg = scipy.signal.resample_poly(ecg, up, down)
        print(f"Warning: {edf_path.name} resampled {fs:g} -> {MODEL_FS} Hz;"
              " the database was recorded at 100 Hz")
        fs = float(MODEL_FS)
    return ecg, ppg, fs


# -----------------------------
# 2. Segments, metrics, feasibility
# -----------------------------


def segment_starts(n_samples: int, fs: float, segment_s: float = SEGMENT_S,
                   step_s: float = STEP_S) -> np.ndarray:
    """Segment starts in samples, laid out like the database."""
    win, step = round(segment_s * fs), round(step_s * fs)
    return np.arange(0, n_samples - win + 1, step)


def bounds_for(mode: str) -> dict:
    return PPG_BOUNDS if mode == "ppg" else FEASIBILITY_BOUNDS


def quality_table(
    ppg: np.ndarray, fs: float, starts: np.ndarray, ends: np.ndarray,
    ecg: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Database metric columns per segment ("ppg_<metric>", "ppg_is_flat",
    and the ECG ones if ecg is given), computed like build_hr_database.
    """
    cols = {}
    for name, sig in (("ppg", ppg), ("ecg", ecg)):
        if sig is None:
            continue
        flags, values = compute_segment_metrics(sig, fs, starts, ends, name)
        cols.update(flags)
        cols.update(values)
    return pd.DataFrame(cols)


def check_feasibility(q: pd.DataFrame, mode: str) -> tuple[np.ndarray, list]:
    """
    Feasible = not flat and strictly inside every border of the mode
    (NaN fails), like within_bounds() and the is_flat filter of training.

    Returns:
        (feasible mask, reason per segment: failed checks joined by ';')
    """
    failed = {}
    failed["ppg_is_flat"] = q["ppg_is_flat"].to_numpy(bool)
    if mode == "full":
        failed["ecg_is_flat"] = q["ecg_is_flat"].to_numpy(bool)
    for col, (lower, upper) in bounds_for(mode).items():
        v = q[col].to_numpy(float)
        ok = np.isfinite(v)
        if lower is not None:
            ok &= v > lower
        if upper is not None:
            ok &= v < upper
        failed[col] = ~ok
    names = np.array(list(failed))
    table = np.column_stack(list(failed.values()))
    reasons = [";".join(names[row]) for row in table]
    return ~table.any(axis=1), reasons


def prepare_signal(
    ppg: np.ndarray, fs: float, ecg: np.ndarray | None = None,
    starts: np.ndarray | None = None, mode: str = "ppg",
    segment_s: float = SEGMENT_S, step_s: float = STEP_S,
    model_input: str = "feasible",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    CPU part of the prediction: segments, metrics, feasibility and the
    model input of a preprocessed recording.

    Args:
        starts: Segment starts in samples (default: all database segments).
        model_input: "feasible" = model input only for feasible segments,
            "all" = for every segment (evaluation of the rejected ones).

    Returns:
        (one row per segment, model input [k, win] float16 for the rows
         with 'to_model' True, in row order)
    """
    win = round(segment_s * fs)
    if starts is None:
        starts = segment_starts(len(ppg), fs, segment_s, step_s)
    starts = np.asarray(starts, dtype=np.int64)
    ends = starts + win
    inside = (starts >= 0) & (ends <= len(ppg))
    st, en = starts[inside], ends[inside]
    finite = np.array([np.isfinite(ppg[s:e]).all() for s, e in zip(st, en)],
                      dtype=bool)

    seg = pd.DataFrame({"start": starts, "end": ends,
                        "start_t": starts / fs, "end_t": ends / fs})
    seg["feasible"] = False
    seg["reason"] = "outside_signal"
    if finite.any():
        q = quality_table(ppg, fs, st[finite], en[finite], ecg)
        feasible, reasons = check_feasibility(q, mode)
        rows = np.flatnonzero(inside)[finite]
        q.index = rows
        seg = seg.join(q)
        seg.loc[rows, "feasible"] = feasible
        seg.loc[rows, "reason"] = reasons
    seg.loc[np.flatnonzero(inside)[~finite], "reason"] = "nan_in_signal"

    usable = np.zeros(len(seg), dtype=bool)
    usable[np.flatnonzero(inside)[finite]] = True
    seg["to_model"] = usable & (
        seg["feasible"].to_numpy() if model_input == "feasible" else True)

    # Model input exactly like the training segment cache
    x = np.zeros((int(seg["to_model"].sum()), win), dtype=np.float16)
    for k, (s, e) in enumerate(
        seg.loc[seg["to_model"], ["start", "end"]].to_numpy()
    ):
        x[k] = normalize_segment(ppg[s:e].astype(np.float64)).astype(
            np.float16)
    return seg, x


def prepare_edf(
    edf_path: str | Path, mode: str = "ppg", input_filtered: bool = False,
    starts: np.ndarray | None = None, segment_s: float = SEGMENT_S,
    step_s: float = STEP_S, model_input: str = "feasible",
) -> tuple[pd.DataFrame, np.ndarray]:
    """preprocess_edf + prepare_signal (picklable, for worker processes)."""
    ecg, ppg, fs = preprocess_edf(edf_path, with_ecg=mode == "full",
                                  input_filtered=input_filtered)
    return prepare_signal(ppg, fs, ecg, starts, mode, segment_s, step_s,
                          model_input)


# -----------------------------
# 3. Model
# -----------------------------


class HRPredictor:
    """
    Trained HR model + feasibility check.

    >>> p = HRPredictor()                      # default fold_1 model
    >>> seg = p.predict_edf("recording.edf")   # one row per segment
    >>> recording_hr(seg)                      # float or None
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 model_name: str = "auto", device: str | None = None,
                 feasibility: str = "ppg", batch_size: int = 512):
        if feasibility not in FEASIBILITY_MODES:
            raise ValueError(f"feasibility must be one of {FEASIBILITY_MODES}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.model_name = load_model(
            str(model_path), model_name, self.device)
        self.model_path = Path(model_path)
        self.feasibility = feasibility
        self.batch_size = batch_size

    @torch.no_grad()
    def run_model(self, x: np.ndarray) -> np.ndarray:
        """x: z-scored segments [k, win] (float16) -> HR in bpm [k]."""
        out = []
        for i in range(0, len(x), self.batch_size):
            xb = torch.from_numpy(x[i : i + self.batch_size]).to(
                self.device).float().unsqueeze(1)
            with autocast_ctx(self.device):
                out.append(self.model(xb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def finish(self, seg: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
        """
        Adds 'hr_model' (network output wherever it was run) and 'hr_bpm'
        (the prediction: hr_model on feasible segments, NaN = NONE).
        """
        seg = seg.copy()
        seg["hr_model"] = np.nan
        seg.loc[seg["to_model"], "hr_model"] = self.run_model(x)
        seg["hr_bpm"] = seg["hr_model"].where(seg["feasible"])
        return seg.drop(columns="to_model")

    def predict_signal(self, ppg, fs, ecg=None, starts=None,
                       segment_s=SEGMENT_S, step_s=STEP_S,
                       model_input="feasible") -> pd.DataFrame:
        """Prediction for an already preprocessed PPG (100 Hz)."""
        return self.finish(*prepare_signal(
            ppg, fs, ecg, starts, self.feasibility, segment_s, step_s,
            model_input))

    def predict_edf(self, edf_path, input_filtered=False, starts=None,
                    segment_s=SEGMENT_S, step_s=STEP_S,
                    model_input="feasible") -> pd.DataFrame:
        """Prediction for one EDF recording (raw unless input_filtered)."""
        return self.finish(*prepare_edf(
            edf_path, self.feasibility, input_filtered, starts, segment_s,
            step_s, model_input))


def recording_hr(seg: pd.DataFrame) -> float | None:
    """Median HR over the feasible segments, None if there are none."""
    hr = seg["hr_bpm"].dropna()
    return float(hr.median()) if len(hr) else None


def fmt_hr(value) -> str:
    return "NONE" if value is None or not np.isfinite(value) else \
        f"{value:.1f} bpm"


# -----------------------------
# 4. CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HR from the PPG of one EDF file, or NONE where the"
        " signal is not feasible")
    p.add_argument("edf", help="EDF recording (raw, unfiltered)")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--model", default="auto",
                   choices=["auto", *MODEL_REGISTRY])
    p.add_argument("--feasibility", default="ppg", choices=FEASIBILITY_MODES,
                   help="ppg: PPG borders only (default, needs no ECG);"
                   " full: ECG + PPG borders as in training")
    p.add_argument("--input-filtered", action="store_true",
                   help="the EDF is already from the filtered database")
    p.add_argument("--segment-s", type=float, default=SEGMENT_S)
    p.add_argument("--step-s", type=float, default=STEP_S)
    p.add_argument("--output", default=None,
                   help="CSV with one row per segment")
    p.add_argument("--max-print", type=int, default=40,
                   help="print at most this many segment lines")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    predictor = HRPredictor(args.model_path, args.model, args.device,
                            args.feasibility, args.batch_size)
    print(f"Model: {predictor.model_name} ({args.model_path}),"
          f" feasibility: {args.feasibility}")
    seg = predictor.predict_edf(args.edf, args.input_filtered,
                                segment_s=args.segment_s,
                                step_s=args.step_s)
    if seg.empty:
        print(f"{args.edf}: shorter than one segment ({args.segment_s:g} s)"
              " -> NONE")
        return

    for row in seg.head(args.max_print).itertuples():
        reason = "" if row.feasible else f"  ({row.reason})"
        print(f"{row.start_t:8.1f}-{row.end_t:8.1f} s: "
              f"{fmt_hr(row.hr_bpm)}{reason}")
    if len(seg) > args.max_print:
        print(f"... {len(seg) - args.max_print} more segments"
              " (use --output to save all)")

    n_ok = int(seg["feasible"].sum())
    print(f"\nFeasible segments: {n_ok}/{len(seg)} ({n_ok / len(seg):.1%})")
    print(f"Recording HR (median of feasible segments):"
          f" {fmt_hr(recording_hr(seg))}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        seg.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
