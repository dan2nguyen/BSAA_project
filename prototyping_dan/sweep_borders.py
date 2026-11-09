"""
Sweep of the feasibility borders of feasible_bpm.py on one CV fold.

Only the borders change between configs. Model, hyperparameters,
preprocessing, epochs, seed and data split are the same for every config;
the training code is imported from training.py (train_fold, evaluate) and
the border check from feasible_bpm.py (within_bounds).

Data: the train split of the parquet database (hr_valid & ~is_flat, as in
training.py). The patient-wise CV split is made ONCE on all these segments
(add_cv_groups + make_cv_splits of training.py), so the validation part is
the same for every config. Borders choose the training data of the fold's
training part and define the feasible segments of its validation part.
The test split is never loaded.

Sweep levels are defined by coverage: a level r means the varied border
alone rejects a share r of the fold's training segments (a two-sided border
splits r between its sides like the current border does).

Strategy:
    1. baselines: no filtering, current FEASIBILITY_BOUNDS
    2. one-at-a-time: each border at --levels, the others at current
    3. grid: <= --max-configs combinations around the best levels of 2
    4. random-subset controls: for every tested training coverage (rounded
       to --rand-bucket), a model on a random subset of the same size
    5. --seeds: the best configs repeated with more seeds

Results are written per config (results/<hash>.json, preds/<hash>.npy),
so an interrupted run resumes by skipping finished configs.

Usage:
    python prototyping_dan/sweep_borders.py --dry-run
    python prototyping_dan/sweep_borders.py --fold 0 --strategy both
    python prototyping_dan/sweep_borders.py --max-configs 2 --epochs 1 \
        --subset-frac 0.02 --output-dir data/sweep_smoke   # smoke test
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from prototyping_dan.feasible_bpm import FEASIBILITY_BOUNDS, within_bounds
from prototyping_dan.load_database import (
    add_cv_groups,
    extract_patient_id,
    load_parquet_files,
)
from prototyping_dan.report_borders import (
    borders_mask,
    build_report,
    compute_metrics,
    load_results,
    add_control_metrics,
    ppg_only,
    rank_configs,
    aurc_table,
)

ON_KAGGLE = Path("/kaggle/input").exists()
BORDER_COLS = list(FEASIBILITY_BOUNDS)
# Training options that define a sweep; they are part of every config hash
SIGNATURE_KEYS = ("model", "epochs", "batch_size", "lr", "weight_decay",
                  "loss", "reweight", "lds_sigma", "max_weight", "patience",
                  "fold", "n_splits")


# -----------------------------
# Data: index + segment cache
# -----------------------------


def cache_tag(args) -> str:
    if args.subset_frac >= 1:
        return "full"
    return f"subset{args.subset_frac:g}_seed{args.seed}"


def build_index(args) -> pd.DataFrame:
    """Train-split segments, sorted, with subject, recording and CV fold."""
    cols = ["base_folder", "edf_filename", "start", "end", "hr_bpm", "split"]
    cols += BORDER_COLS
    extra = ["feasible_bpm"]
    try:
        df, _ = load_parquet_files(args.parquet_dir, columns=cols + extra,
                                   split_stats=True)
    except Exception:  # noqa: BLE001  (older databases without feasible_bpm)
        df, _ = load_parquet_files(args.parquet_dir, columns=cols,
                                   split_stats=True)
    # os.listdir order differs between systems -> sort for a stable split
    df = df.sort_values(["base_folder", "edf_filename", "start"]).reset_index(
        drop=True)
    if args.subset_frac < 1:
        recs = df[["base_folder", "edf_filename"]].drop_duplicates()
        recs = recs.sample(frac=args.subset_frac, random_state=args.seed)
        df = df.merge(recs, on=["base_folder", "edf_filename"]).reset_index(
            drop=True)
    df["source_file"] = df.pop("_source_file")
    df["subject"] = (df["base_folder"] + "/"
                     + df["source_file"].map(extract_patient_id))
    df["recording"] = df["base_folder"] + "/" + df["edf_filename"]
    return assign_folds(df, args.n_splits)


def assign_folds(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Patient-wise folds exactly as training.py (on all segments)."""
    from prototyping_dan.training import make_cv_splits

    df = df.copy()
    tmp = add_cv_groups(df.assign(_source_file=df["source_file"]),
                        n_folds=n_splits)
    df["cv_group"] = tmp["cv_group"].to_numpy()
    df["fold"] = -1
    for k, (_, va) in enumerate(make_cv_splits(df, n_splits)):
        df.loc[va, "fold"] = k
    df.attrs["n_splits"] = n_splits
    return df


def prepare_data(args) -> tuple[pd.DataFrame, Path]:
    """
    Builds (or loads) the cache: <cache-dir>/<tag>/index.parquet and
    segments.npy (float16 [N, T], z-scored by training.py's reader).
    EDF files are read only here, never inside the sweep loop.
    """
    from prototyping_dan.training import load_segment_array

    cdir = Path(args.cache_dir) / cache_tag(args)
    index_path, seg_path = cdir / "index.parquet", cdir / "segments.npy"
    meta_path = cdir / "meta.json"
    if index_path.exists() and seg_path.exists():
        print(f"Using cache {cdir}")
        df = pd.read_parquet(index_path)
        meta = json.loads(meta_path.read_text())
        if meta["n_splits"] != args.n_splits:
            df = assign_folds(df, args.n_splits)
        return df, seg_path

    if not args.parquet_dir or not Path(args.parquet_dir).exists():
        raise FileNotFoundError(
            f"No cache in {cdir} and --parquet-dir not found: {args.parquet_dir}")
    df = build_index(args)
    print(f"Index: {len(df)} train segments, {df['subject'].nunique()}"
          f" subjects, {df['recording'].nunique()} recordings")
    if args.dry_run:
        df["ok"] = True  # no EDF reading for a dry run
        return df, None
    segments, ok = load_segment_array(df, args.edf_dir, args.workers,
                                      cache_dir=None)
    df["ok"] = ok
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(seg_path, segments)
    df.to_parquet(index_path, index=False)
    meta_path.write_text(json.dumps({
        "n_splits": args.n_splits, "n": len(df), "seg_len": segments.shape[1],
        "subset_frac": args.subset_frac,
    }))
    print(f"Saved cache to {cdir} ({segments.nbytes / 1e9:.2f} GB)")
    return df, seg_path


# -----------------------------
# Borders defined by coverage
# -----------------------------


def _threshold(sorted_v: np.ndarray, uniq: np.ndarray, r: float, side: str):
    """
    Border whose rejection share (strict borders: value <= lo or >= hi
    rejected) is closest to r. Midpoints between distinct values are used,
    so discrete variables (e.g. ppg_zero_dev_dur) snap to reachable shares.
    """
    n = len(sorted_v)
    if side == "lo":
        rej = np.searchsorted(sorted_v, uniq, side="right") / n  # v <= u
        j = int(np.argmin(np.abs(rej - r)))
        nxt = uniq[j + 1] if j + 1 < len(uniq) else uniq[j] + 1e-6
        return float((uniq[j] + nxt) / 2), float(rej[j])
    rej = 1 - np.searchsorted(sorted_v, uniq, side="left") / n  # v >= u
    j = int(np.argmin(np.abs(rej - r)))
    prv = uniq[j - 1] if j > 0 else uniq[j] - 1e-6
    return float((prv + uniq[j]) / 2), float(rej[j])


def _round(v):
    return None if v is None else float(f"{v:.6g}")


class BorderLevels:
    """Quantile-based border values of each variable on the training part."""

    def __init__(self, train_df: pd.DataFrame):
        self.n = len(train_df)
        self.sorted = {c: np.sort(train_df[c].to_numpy()) for c in BORDER_COLS}
        self.uniq = {c: np.unique(v) for c, v in self.sorted.items()}
        self.df = train_df

    def rejection(self, col, lo, hi) -> float:
        return float(1 - within_bounds(self.df[[col]], {col: (lo, hi)}).mean())

    def side_shares(self, col):
        lo, hi = FEASIBILITY_BOUNDS[col]
        v = self.df[col]
        p_lo = float((v <= lo).mean()) if lo is not None else 0.0
        p_hi = float((v >= hi).mean()) if hi is not None else 0.0
        if p_lo + p_hi == 0:
            k = (lo is not None) + (hi is not None)
            return (lo is not None) / k, (hi is not None) / k
        return p_lo / (p_lo + p_hi), p_hi / (p_lo + p_hi)

    def border(self, col, r):
        """(lo, hi) rejecting ~r of the training part on its own."""
        if r <= 0:
            return None
        f_lo, f_hi = self.side_shares(col)
        lo = hi = None
        if f_lo > 0:
            lo, _ = _threshold(self.sorted[col], self.uniq[col], r * f_lo, "lo")
        if f_hi > 0:
            hi, _ = _threshold(self.sorted[col], self.uniq[col], r * f_hi, "hi")
        return [_round(lo), _round(hi)]


def current_borders() -> dict:
    return {c: [lo, hi] for c, (lo, hi) in FEASIBILITY_BOUNDS.items()}


def border_text(b) -> str:
    if b is None:
        return "none"
    f = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
    return f"{f(b[0])}/{f(b[1])}"


# -----------------------------
# Configs
# -----------------------------


def borders_key(borders: dict, kind: str, random_frac=None) -> str:
    payload = {"b": {c: [_round(x) if x is not None else None for x in b]
                     for c, b in sorted(borders.items())}}
    if kind == "random":
        payload = {"random": round(random_frac, 4)}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def make_config(name, kind, borders, seed, signature, var=None, level=None,
                random_frac=None, phase="oat"):
    bk = borders_key(borders, kind, random_frac)
    h = hashlib.md5(f"{bk}|{seed}|{signature}".encode()).hexdigest()[:16]
    return {"name": name, "kind": kind, "var": var, "level": level,
            "borders": borders, "seed": seed, "random_frac": random_frac,
            "phase": phase, "borders_key": bk, "hash": h}


def signature_of(args) -> str:
    return json.dumps({k: getattr(args, k) for k in SIGNATURE_KEYS} | {
        "cache": cache_tag(args)}, sort_keys=True)


def plan_baseline_oat(args, levels: BorderLevels, seed: int) -> list[dict]:
    sig = signature_of(args)
    cur = current_borders()
    cfgs = [make_config("no_filter", "none", {}, seed, sig, phase="oat"),
            make_config("current", "current", cur, seed, sig, phase="oat")]
    if args.strategy not in ("one-at-a-time", "both"):
        return cfgs
    for col in args.vars:
        cur_rej = levels.rejection(col, *FEASIBILITY_BOUNDS[col])
        seen = set()
        for r in args.levels:
            if r > 0 and abs(r - cur_rej) < args.level_tolerance:
                continue  # ~ the current border, covered by "current"
            b = levels.border(col, r)
            borders = {c: v for c, v in cur.items() if c != col}
            if b is not None:
                borders[col] = b
            cfg = make_config(f"{col}@{r:g}", "oat", borders, seed, sig,
                              var=col, level=r)
            if cfg["borders_key"] in seen or cfg["borders_key"] == \
                    borders_key(cur, "current"):
                continue
            seen.add(cfg["borders_key"])
            cfgs.append(cfg)
    return cfgs


def plan_controls(args, cfgs, ctx, seed) -> list[dict]:
    """Random-subset controls at every (bucketed) training coverage."""
    sig = signature_of(args)
    buckets = set()
    for c in cfgs:
        if c["kind"] in ("none", "random"):
            continue
        cov = ctx.train_mask(c).mean()
        b = round(round(cov / args.rand_bucket) * args.rand_bucket, 4)
        if 0 < b < 0.999:
            buckets.add(b)
    return [make_config(f"random@{b:g}", "random", {}, seed, sig,
                        random_frac=b, phase=cfgs[0]["phase"] if cfgs else "oat")
            for b in sorted(buckets)]


def plan_grid(args, ctx, levels: BorderLevels, seed) -> list[dict]:
    """
    Joint refinement around the one-at-a-time results: the variables with
    the strongest effect get {best level, neighbouring levels}, all other
    variables sit at their best one-at-a-time level; the product is capped
    at --max-configs.
    """
    sig = signature_of(args)
    df = finished_table(args, ctx)
    if df is None or (df["kind"] == "oat").sum() == 0:
        print("Grid: no one-at-a-time results yet - run with"
              " --strategy one-at-a-time (or both) first")
        return []
    aurc = aurc_table(df)
    cur = current_borders()
    best_border, cand_borders, impact = {}, {}, {}
    for var in args.vars:
        pts = df[((df["kind"] == "oat") & (df["var"] == var))
                 | (df["kind"] == "current")].copy()
        if pts.empty:
            continue
        pts["lvl"] = pts["level"].where(pts["kind"] == "oat", -1.0)
        ranked, _ = rank_configs(pts, args.max_mae, args.min_coverage,
                                 args.min_subject_coverage)
        top = ranked.iloc[0]
        best_border[var] = top["borders"].get(var)
        # neighbours of the best level in level order
        order = pts.assign(cov=pts["coverage_val"]).sort_values("cov")
        keys = list(order["hash"])
        i = keys.index(top["hash"])
        neigh = order.iloc[max(i - 1, 0): i + 2]
        cand_borders[var] = [r["borders"].get(var) for _, r in neigh.iterrows()]
        impact[var] = float(pts["mae_feasible"].max() - pts["mae_feasible"].min())
    if not best_border:
        return []
    ranked_vars = sorted(impact, key=impact.get, reverse=True)
    print("Grid: variable impact (MAE range over levels): "
          + ", ".join(f"{v}={impact[v]:.2f}" for v in ranked_vars))
    if not aurc.empty:
        print("Grid: variables marked not useful: "
              + ", ".join(aurc.loc[~aurc["useful"], "variable"]))

    # choose varied variables while the product stays <= max_configs
    varied, size = [], 1
    for v in ranked_vars:
        k = len({json.dumps(b) for b in cand_borders[v]})
        if k > 1 and size * k <= args.max_configs:
            varied.append(v)
            size *= k
    base = {}
    for v in BORDER_COLS:
        b = best_border.get(v, cur.get(v))
        if b is not None:
            base[v] = b
    combos = [base]
    for v in varied:
        opts = []
        for b in cand_borders[v]:
            if json.dumps(b) not in [json.dumps(o) for o in opts]:
                opts.append(b)
        new = []
        for c in combos:
            for b in opts:
                d = dict(c)
                if b is None:
                    d.pop(v, None)
                else:
                    d[v] = b
                new.append(d)
        combos = new
    cfgs, seen = [], set()
    for i, borders in enumerate(combos[: args.max_configs]):
        cfg = make_config(f"grid{i:02d}", "grid", borders, seed, sig,
                          phase="grid")
        if cfg["borders_key"] not in seen:
            seen.add(cfg["borders_key"])
            cfgs.append(cfg)
    print(f"Grid: {len(cfgs)} configs, varied: {varied}")
    return cfgs


# -----------------------------
# Fold context
# -----------------------------


class FoldContext:
    def __init__(self, df: pd.DataFrame, seg_path, args):
        ok = df["ok"].to_numpy()
        self.df = df
        self.seg_path = seg_path
        self.train_rows = np.flatnonzero((df["fold"] != args.fold).to_numpy() & ok)
        self.val_rows = np.flatnonzero((df["fold"] == args.fold).to_numpy() & ok)
        self.train_df = df.iloc[self.train_rows].reset_index(drop=True)
        self.val_df = df.iloc[self.val_rows].reset_index(drop=True)
        self.levels = BorderLevels(self.train_df)
        self.args = args
        self._mask_cache = {}

    def _random_mask(self, n, frac, seed, salt):
        rng = np.random.default_rng([seed, salt, int(round(frac * 1e4))])
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, size=max(int(round(frac * n)), 1), replace=False)] = True
        return m

    def train_mask(self, cfg) -> np.ndarray:
        key = ("tr", cfg["borders_key"], cfg["seed"])
        if key not in self._mask_cache:
            if cfg["kind"] == "random":
                m = self._random_mask(len(self.train_df), cfg["random_frac"],
                                      cfg["seed"], 1)
            else:
                m = borders_mask(self.train_df, cfg["borders"])
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def val_mask(self, cfg) -> np.ndarray:
        if cfg["kind"] == "random":
            return self._random_mask(len(self.val_df), cfg["random_frac"],
                                     cfg["seed"], 2)
        return borders_mask(self.val_df, cfg["borders"])


def coverage_preview(ctx: FoldContext, cfgs) -> pd.DataFrame:
    from prototyping_dan.report_borders import coverage_stats

    rows = []
    for c in cfgs:
        tm, vm = ctx.train_mask(c), ctx.val_mask(c)
        lvl = c["level"]
        b = c["borders"].get(c["var"]) if c["var"] else None
        r = {"name": c["name"], "kind": c["kind"], "phase": c["phase"],
             "border": border_text(b) if c["var"] else "",
             "level": lvl if lvl is not None else np.nan,
             "coverage_train": tm.mean(), "n_train": int(tm.sum()),
             "coverage_val": vm.mean(),
             "coverage_val_ppg_only": (borders_mask(ctx.val_df, ppg_only(c["borders"])).mean()
                                       if c["kind"] != "random" else np.nan)}
        r.update(coverage_stats(vm, ctx.val_df["subject"].to_numpy(), "subject"))
        rows.append(r)
    return pd.DataFrame(rows)


def finished_table(args, ctx) -> pd.DataFrame | None:
    out = Path(args.output_dir)
    if not (out / "results").exists() or not any((out / "results").glob("*.json")):
        return None
    df = load_results(out)
    df["is_main_seed"] = df["seed"] == args.seed
    df = df[df["is_main_seed"]]
    return add_control_metrics(df, ctx.val_df, out, args.rand_bucket)


# -----------------------------
# Training worker
# -----------------------------


class Trainer:
    """Holds the segment array on the device; trains configs."""

    def __init__(self, ctx: FoldContext, args):
        import torch

        self.torch = torch
        self.ctx, self.args = ctx, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            p = torch.cuda.get_device_properties(0)
            self.gpu = f"{p.name} ({p.total_memory / 1e9:.1f} GB)"
        else:
            self.gpu = "cpu"
        segments = np.load(ctx.seg_path, mmap_mode="r")
        self.store = torch.device("cpu") if args.data_on_cpu else self.device
        # Only rows of this fold are moved to the device
        rows = np.concatenate([ctx.train_rows, ctx.val_rows])
        X = torch.from_numpy(np.ascontiguousarray(segments[rows]))
        if self.store.type == "cpu" and self.device.type == "cuda":
            X = X.pin_memory()
        self.X = X.to(self.store)
        hr = ctx.df["hr_bpm"].to_numpy()[rows].astype(np.float32)
        self.hr = hr
        self.y = torch.from_numpy(hr).to(self.store)
        n_tr = len(ctx.train_rows)
        self.tr_pos = np.arange(n_tr)  # positions in X of the train part
        self.va_pos = np.arange(n_tr, len(rows))
        print(f"[{self.gpu}] fold data on {self.store}:"
              f" {self.X.numel() * 2 / 1e9:.2f} GB, {n_tr} train /"
              f" {len(self.va_pos)} val segments")
        self.benchmark()

    def benchmark(self):
        """Seconds per epoch of the full training part (a few timed steps)."""
        from prototyping_dan.training import build_model

        torch = self.torch
        model = build_model(self.args.model, in_channels=1).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(self.device.type,
                                      enabled=self.device.type == "cuda")
        bs = min(self.args.batch_size, len(self.tr_pos))
        idx = torch.arange(bs, device=self.store)
        steps = 6 if self.device.type == "cuda" else 2
        for i in range(steps + 1):
            if i == 1:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            xb = self.X[idx].to(self.device).float().unsqueeze(1)
            with torch.autocast(self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                loss = model(xb).float().mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        step = (time.perf_counter() - t0) / steps
        n_steps = len(self.tr_pos) / bs
        print(f"[{self.gpu}] ~{step * 1000:.0f} ms/step at batch {bs}"
              f" -> ~{step * n_steps:.0f} s/epoch on the full training part"
              f" ({len(self.tr_pos)} segments)")
        del model, opt

    def run(self, cfg) -> dict:
        from prototyping_dan.training import evaluate, train_fold

        torch, ctx, args = self.torch, self.ctx, self.args
        tm, vm = ctx.train_mask(cfg), ctx.val_mask(cfg)
        train_pos = self.tr_pos[tm]
        es_pos = self.va_pos[vm] if vm.any() else self.va_pos
        print(f"\n=== {cfg['name']} [{cfg['hash']}] seed {cfg['seed']}:"
              f" train {tm.sum()}/{len(tm)} ({tm.mean():.1%}),"
              f" val feasible {vm.sum()}/{len(vm)} ({vm.mean():.1%})")
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        a = copy.copy(args)
        a.seed = cfg["seed"]
        res = train_fold(self.X, self.y, self.hr, train_pos, es_pos, a,
                         self.device, self.store)
        va = torch.from_numpy(self.va_pos).to(self.store)
        _, preds, _ = evaluate(res["model"], self.X, self.y, va,
                               args.batch_size, res["criterion"], self.device)
        metrics = compute_metrics(ctx.val_df, preds, vm)
        metrics_ppg = (compute_metrics(
            ctx.val_df, preds, borders_mask(ctx.val_df, ppg_only(cfg["borders"])))
            if cfg["kind"] != "random" else {})
        var = cfg["var"]
        level_actual = (ctx.levels.rejection(var, *cfg["borders"][var])
                        if var and var in cfg["borders"] else 0.0)
        if cfg["kind"] == "current":
            level_actual = np.nan  # filled per variable in the report
        return {
            **{k: cfg[k] for k in ("hash", "borders_key", "name", "kind", "var",
                                   "level", "seed", "random_frac", "phase",
                                   "borders")},
            "level_actual": level_actual,
            "border_text": border_text(cfg["borders"].get(var)) if var else "",
            "fold": args.fold, "n_splits": args.n_splits,
            "coverage_train": float(tm.mean()), "n_train": int(tm.sum()),
            "n_train_total": int(len(tm)),
            "epochs_run": res["epochs_run"], "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "train_seconds": res["seconds"],
            "seconds_per_epoch": float(np.mean(res["history"]["epoch_seconds"])),
            "gpu": self.gpu, "metrics": metrics, "metrics_ppg": metrics_ppg,
            "history": res["history"], "_preds": preds,
            "_state": res["state_dict"],
        }


def save_result(out: Path, r: dict):
    import torch

    preds, state = r.pop("_preds"), r.pop("_state")
    for d in ("results", "preds", "ckpt"):
        (out / d).mkdir(parents=True, exist_ok=True)
    np.save(out / "preds" / f"{r['hash']}.npy", preds.astype(np.float32))
    torch.save(state, out / "ckpt" / f"{r['hash']}.pt")
    tmp = out / "results" / f"{r['hash']}.json.tmp"

    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    tmp.write_text(json.dumps(clean(r), indent=1))
    os.replace(tmp, out / "results" / f"{r['hash']}.json")  # atomic


def is_done(out: Path, cfg) -> bool:
    return (out / "results" / f"{cfg['hash']}.json").exists()


def claim(out: Path, cfg) -> bool:
    (out / "locks").mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out / "locks" / f"{cfg['hash']}.lock",
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def work(cfgs, ctx, args, trainer_holder: dict):
    """Trains every unfinished, unclaimed config until the deadline."""
    out = Path(args.output_dir)
    last = 0.0
    for cfg in cfgs:
        if is_done(out, cfg) or not claim(out, cfg):
            continue
        if args.deadline and time.time() + last > args.deadline:
            print("Time budget reached - stopping (resume later)")
            (out / "locks" / f"{cfg['hash']}.lock").unlink(missing_ok=True)
            return False
        if "trainer" not in trainer_holder:
            trainer_holder["trainer"] = Trainer(ctx, args)
        t0 = time.time()
        r = trainer_holder["trainer"].run(cfg)
        save_result(out, r)
        last = time.time() - t0
        m = r["metrics"]
        print(f"--> {cfg['name']}: cov_val {m['coverage_val']:.3f}"
              f" mae_feasible {m['mae_feasible']:.3f} mae_all {m['mae_all']:.3f}"
              f" mae_rejected {m['mae_rejected']:.3f} ({last / 60:.1f} min)")
    return True


def run_phase(cfgs, ctx, args, trainer_holder) -> bool:
    out = Path(args.output_dir)
    todo = [c for c in cfgs if not is_done(out, c)]
    print(f"\nPhase: {len(cfgs)} configs, {len(cfgs) - len(todo)} already done,"
          f" {len(todo)} to train")
    if not todo:
        return True
    # stale locks of an interrupted run
    for p in (out / "locks").glob("*.lock") if (out / "locks").exists() else []:
        p.unlink()
    plan_path = out / "plan_current.json"
    plan_path.write_text(json.dumps(todo))
    if args.gpus <= 1:
        return work(todo, ctx, args, trainer_holder)
    procs = []
    for g in range(args.gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TQDM_DISABLE="1")
        cmd = [sys.executable, "-u", __file__, *sys.argv[1:],
               "--worker-plan", str(plan_path), "--deadline", str(args.deadline or 0)]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(5)  # stagger startup (data loading)
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"Warning: worker exit codes {codes}")
    return all(is_done(out, c) for c in todo)


def prune_checkpoints(args, ctx):
    """Keep only no-filter, current and the top-3 configs."""
    out = Path(args.output_dir)
    df = finished_table(args, ctx)
    if df is None:
        return
    ranked, _ = rank_configs(df, args.max_mae, args.min_coverage,
                             args.min_subject_coverage)
    keep = set(df.loc[df["kind"].isin(["none", "current"]), "hash"])
    keep |= set(ranked.head(3)["hash"])
    removed = 0
    for p in (out / "ckpt").glob("*.pt"):
        if p.stem not in keep:
            p.unlink()
            removed += 1
    print(f"Checkpoints kept: {len(keep)}, removed: {removed}")


# -----------------------------
# Main
# -----------------------------


def default_paths():
    if ON_KAGGLE:
        return {"parquet": None, "edf": None,
                "out": "/kaggle/working/sweep_results",
                "cache": "/kaggle/working/cache"}
    return {"parquet": str(root_dir / "data" / "hr_parts" / "seg10s_step5s"),
            "edf": str(root_dir / "data" / "harmonized_ecg_ppg_database_filtered"),
            "out": str(root_dir / "data" / "sweep_borders" / "fold0"),
            "cache": str(root_dir / "data" / "sweep_borders" / "cache")}


def parse_args(argv=None):
    from prototyping_dan.training import add_training_args

    d = default_paths()
    p = argparse.ArgumentParser(
        description="Feasibility-border sweep on one CV fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_training_args(p)  # --model --epochs --batch-size --lr ... --seed
    p.add_argument("--parquet-dir", default=d["parquet"])
    p.add_argument("--edf-dir", default=d["edf"])
    p.add_argument("--output-dir", default=d["out"])
    p.add_argument("--cache-dir", default=d["cache"])
    p.add_argument("--fold", type=int, default=0, help="0-based fold")
    p.add_argument("--n-splits", dest="n_splits", type=int, default=5,
                   help="as training.py --n_splits")
    p.add_argument("--strategy", choices=["one-at-a-time", "grid", "both"],
                   default="both")
    p.add_argument("--levels", default="0,0.025,0.05,0.1,0.15,0.2,0.3,0.4",
                   help="share of training segments the varied border rejects"
                   " on its own (0 = border removed)")
    p.add_argument("--level-tolerance", type=float, default=0.01,
                   help="levels this close to the current border's rejection"
                   " are replaced by the current config")
    p.add_argument("--vars", default="all",
                   help="comma-separated border variables to sweep")
    p.add_argument("--max-configs", type=int, default=20,
                   help="cap of grid configs; with --strategy one-at-a-time"
                   " also caps the number of border configs (smoke tests)")
    p.add_argument("--max-mae", type=float, default=5.0)
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--min-subject-coverage", type=float, default=0.5)
    p.add_argument("--rand-bucket", type=float, default=0.025,
                   help="training-coverage resolution of the random controls")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--seeds", default="",
                   help="extra seeds for the top configs, e.g. '43,44'")
    p.add_argument("--n-seed-configs", type=int, default=3)
    p.add_argument("--gpus", type=int, default=None,
                   help="parallel worker processes, one per GPU"
                   " (default: number of GPUs)")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="processes for reading the EDF files")
    p.add_argument("--data-on-cpu", dest="data_on_cpu", action="store_true")
    p.add_argument("--subset-frac", type=float, default=1.0,
                   help="use a random share of the recordings (smoke tests)")
    p.add_argument("--time-budget-hours", type=float, default=None,
                   help="stop starting configs after this (Kaggle: ~11.3)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--build-cache-only", action="store_true",
                   help="read the EDF files into the cache, then exit")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--worker-plan", default=None, help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.levels = [float(x) for x in args.levels.split(",") if x.strip()]
    args.vars = (BORDER_COLS if args.vars == "all"
                 else [v.strip() for v in args.vars.split(",")])
    unknown = set(args.vars) - set(BORDER_COLS)
    if unknown:
        p.error(f"unknown --vars {unknown}; choose from {BORDER_COLS}")
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.gpus is None:
        try:
            import torch
            args.gpus = max(torch.cuda.device_count(), 1)
        except ImportError:
            args.gpus = 1
    return args


def verify_current(ctx: FoldContext, df: pd.DataFrame):
    """The current config must select the same segments as feasible_bpm.py."""
    cur = current_borders()
    if "feasible_bpm" not in df:
        print("Verification skipped: no feasible_bpm column in the database")
        return
    mine = borders_mask(df, cur)
    orig = df["feasible_bpm"].notna().to_numpy()
    key = ["base_folder", "edf_filename", "start"]
    a = set(map(tuple, df.loc[mine, key].to_numpy()))
    b = set(map(tuple, df.loc[orig, key].to_numpy()))
    print(f"Verification: current borders select {len(a)} segments,"
          f" feasible_bpm column {len(b)}; identical: {a == b}")
    tr = ctx.train_mask(make_config("current", "current", cur, 0, ""))
    tr_orig = ctx.train_df["feasible_bpm"].notna().to_numpy()
    print(f"Verification (fold training part): {tr.sum()} vs {tr_orig.sum()},"
          f" identical: {bool((tr == tr_orig).all())}")


def print_distribution(ctx: FoldContext):
    t = ctx.train_df
    q = np.round(np.arange(0, 1.0001, 0.05), 2)
    Q = t[BORDER_COLS].quantile(q).T
    Q.columns = [f"{int(c * 100)}%" for c in q]
    print("\nQuantiles of the border variables (fold training part):")
    print(Q.to_string(float_format=lambda v: f"{v:.4g}"))
    rows = []
    for c, (lo, hi) in FEASIBILITY_BOUNDS.items():
        rows.append({"variable": c, "lower": lo, "upper": hi,
                     "rejects_alone": ctx.levels.rejection(c, lo, hi),
                     "side_split_lo/hi": "{:.2f}/{:.2f}".format(
                         *ctx.levels.side_shares(c))})
    print("\nCurrent borders (keep if lower < value < upper, AND of all):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    allm = borders_mask(t, current_borders())
    print(f"All current borders together keep {allm.mean():.1%} of the"
          f" training part")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    df, seg_path = prepare_data(args)
    if args.build_cache_only:
        print(f"Cache ready: {seg_path.parent}")
        return
    ctx = FoldContext(df, seg_path, args)
    holder = {}

    if args.worker_plan:  # child process on one GPU
        work(json.loads(Path(args.worker_plan).read_text()), ctx, args, holder)
        return

    print(f"Fold {args.fold}/{args.n_splits}: {len(ctx.train_df)} training"
          f" segments ({ctx.train_df['subject'].nunique()} subjects),"
          f" {len(ctx.val_df)} validation segments"
          f" ({ctx.val_df['subject'].nunique()} subjects:"
          f" {', '.join(sorted(ctx.val_df['subject'].unique()))})")
    print_distribution(ctx)
    verify_current(ctx, df)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "sweep_manifest.json"
    sig = signature_of(args)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("signature") != sig:
            print("Warning: this output folder holds a sweep with other"
                  " settings; their results are kept but not reused.")
    manifest_path.write_text(json.dumps({
        "signature": sig, "fold": args.fold, "n_splits": args.n_splits,
        "seed": args.seed, "epochs": args.epochs, "rand_bucket": args.rand_bucket,
        "levels": args.levels, "vars": args.vars,
        "current_levels": {c: ctx.levels.rejection(c, *b)
                           for c, b in FEASIBILITY_BOUNDS.items()},
        "current_borders": current_borders(),
    }, indent=1))
    ctx.val_df[["base_folder", "edf_filename", "start", "subject",
                "recording", "hr_bpm", *BORDER_COLS]].to_parquet(
        out / "val_index.parquet", index=False)

    if args.time_budget_hours:
        args.deadline = time.time() + args.time_budget_hours * 3600
    seed = args.seed

    # Phase 1: baselines + one-at-a-time (+ controls)
    plan = plan_baseline_oat(args, ctx.levels, seed)
    if args.strategy == "one-at-a-time" and len(plan) > args.max_configs:
        plan = plan[: args.max_configs]
    controls = [] if args.no_controls else plan_controls(args, plan, ctx, seed)
    preview = coverage_preview(ctx, plan + controls)
    pd.set_option("display.width", 250)
    print(f"\nPlanned phase 1: {len(plan)} border configs + {len(controls)}"
          " random controls")
    print(preview.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    preview.to_csv(out / "planned_configs.csv", index=False)

    if args.dry_run:
        if args.strategy in ("grid", "both"):
            grid = plan_grid(args, ctx, ctx.levels, seed)
            if grid:
                print(coverage_preview(ctx, grid).to_string(index=False))
        return

    ok = run_phase(plan + controls, ctx, args, holder)

    # Phase 2: joint grid
    if ok and args.strategy in ("grid", "both"):
        grid = plan_grid(args, ctx, ctx.levels, seed)
        if grid:
            gcontrols = [] if args.no_controls else plan_controls(args, grid, ctx, seed)
            print(coverage_preview(ctx, grid + gcontrols).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
            ok = run_phase(grid + gcontrols, ctx, args, holder)

    # Phase 3: extra seeds for the best configs
    if ok and args.seeds:
        table = finished_table(args, ctx)
        ranked, _ = rank_configs(table, args.max_mae, args.min_coverage,
                                 args.min_subject_coverage)
        base = pd.concat([table[table["kind"].isin(["none", "current"])],
                          ranked.head(args.n_seed_configs)]).drop_duplicates("hash")
        sig_ = signature_of(args)
        reps = [make_config(r["name"], r["kind"], r["borders"], s, sig_,
                            var=r["var"], level=r["level"], phase="seeds")
                for _, r in base.iterrows() for s in args.seeds]
        ok = run_phase(reps, ctx, args, holder)

    prune_checkpoints(args, ctx)
    if not args.no_report:
        build_report(out, out / "report", args.max_mae, args.min_coverage,
                     args.min_subject_coverage, args.rand_bucket)
    print("Sweep complete." if ok else
          "Sweep stopped early (time budget) - rerun the same command to resume.")


if __name__ == "__main__":
    main()
