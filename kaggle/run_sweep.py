"""
Kaggle kernel: feasibility-border sweep (fold 0) + report.

1. rebuilds the code packages from the bsaa-code dataset
2. installs missing dependencies (internet on)
3. finds the parquet / EDF / optional cache / optional resume inputs
4. runs prototyping_dan/sweep_borders.py (builds the cache if needed,
   resumes finished configs, stops before the session limit)
5. writes the report and zips results + report for download

Inputs are found by searching /kaggle/input, so both mount layouts
(/kaggle/input/<slug>/ and /kaggle/input/datasets/<user>/<slug>/) work.
Runs locally too (python kaggle/run_sweep.py) with the repo paths.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ---- sweep settings (edit here, then `kaggle kernels push`) ----
SWEEP_ARGS = [
    "--fold",
    "0",
    "--strategy",
    "both",
    "--levels",
    "0,0.025,0.05,0.1,0.15,0.2,0.3,0.4",
    "--max-configs",
    "20",
    "--epochs",
    "15",
    "--batch-size",
    "512",  # = training.py; larger is faster but changes training
    "--seeds",
    "43,44",
    "--max-mae",
    "5",
    "--min-subject-coverage",
    "0.0",
]
TIME_BUDGET_HOURS = 11.0  # Kaggle GPU sessions end after 12 h
KEEP_CACHE_IN_OUTPUT = True  # cache (~1.3 GB) in /kaggle/working/cache
# Optional manual paths (None = search /kaggle/input). The parquet folder
# must contain the metric columns (ecg_kurtosis, ppg_kurtosis, ...).
PARQUET_DIR = None  # e.g. "/kaggle/input/datasets/dannguyen2/hr-parts/seg10s_step5s"
EDF_DIR = None
REQUIRED_COLS = ("ecg_kurtosis", "ppg_kurtosis", "ppg_zero_dev_dur")

INPUT = Path("/kaggle/input")
ON_KAGGLE = INPUT.exists()
REPO = Path(__file__).resolve().parent.parent
WORK = (
    Path("/kaggle/working")
    if ON_KAGGLE
    else REPO / "data" / "sweep_borders"
)
TMP = Path("/kaggle/tmp") if ON_KAGGLE else WORK / "_tmp"
CODE_FILES = {
    "training.py": "prototyping_dan",
    "feasible_bpm.py": "prototyping_dan",
    "load_database.py": "prototyping_dan",
    "sweep_borders.py": "prototyping_dan",
    "report_borders.py": "prototyping_dan",
    "read_edf.py": "utils",
}


def walk_inputs(max_depth=6):
    """(dir, subdirs, files) under /kaggle/input, depth-limited."""
    base = len(INPUT.parts)
    for d, subdirs, files in os.walk(INPUT, followlinks=True):
        if len(Path(d).parts) - base >= max_depth:
            subdirs[:] = []
        yield Path(d), subdirs, files


def setup_code() -> Path:
    if not ON_KAGGLE:
        return REPO
    code = TMP / "code"
    found = {}
    for d, _, files in walk_inputs():
        for f in files:
            if f in CODE_FILES and f not in found:
                found[f] = d / f
            if f.endswith(".zip") and "code" in d.name and not found:
                with zipfile.ZipFile(d / f) as z:
                    z.extractall(TMP / "code_zip")
    if len(found) < len(CODE_FILES):
        for f in CODE_FILES:  # extracted zip layout
            for p in (
                (TMP / "code_zip").rglob(f)
                if (TMP / "code_zip").exists()
                else []
            ):
                found.setdefault(f, p)
    missing = set(CODE_FILES) - set(found)
    if missing:
        raise FileNotFoundError(
            f"code files not found in /kaggle/input: {missing}"
        )
    for f, pkg in CODE_FILES.items():
        (code / pkg).mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[f], code / pkg / f)
    print(f"Code from {found['sweep_borders.py'].parent} -> {code}")
    return code


def ensure_deps():
    need = []
    for mod, pkg in (
        ("mne", "mne"),
        ("sklearn", "scikit-learn"),
        ("pyarrow", "pyarrow"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print(f"Installing {need}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *need],
            check=True,
        )


def parquet_has_columns(d: Path) -> tuple[bool, str]:
    """Does the first parquet file below d have REQUIRED_COLS?"""
    import pyarrow.parquet as pq

    f = next(d.glob("*/*.parquet"), None)
    if f is None:
        return False, "parquet files"
    names = set(pq.read_schema(f).names)
    missing = [c for c in REQUIRED_COLS if c not in names]
    return not missing, ",".join(missing)


def find_inputs() -> dict:
    if not ON_KAGGLE:
        return {
            "parquet": REPO / "data" / "hr_parts" / "seg10s_step5s",
            "edf": REPO
            / "data"
            / "harmonized_ecg_ppg_database_filtered",
            "cache": None,
            "resume": [],
        }
    res = {"parquet": None, "edf": None, "cache": None, "resume": []}
    parquet_cands = []
    for d, subdirs, files in walk_inputs():
        if any(next(Path(d, s).glob("*.parquet"), None) for s in subdirs[:3]):
            parquet_cands.append(d)  # <dir>/<base_folder>/<file>.parquet
        if (
            res["edf"] is None
            and d.name == "harmonized_ecg_ppg_database_filtered"
        ):
            res["edf"] = d
        if "segments.npy" in files and "index.parquet" in files:
            res["cache"] = d
        if "sweep_manifest.json" in files:
            res["resume"].append(d)
        res["resume"] += [
            d / f for f in files if f == "sweep_results.zip"
        ]
    # Several copies of the database may be attached (e.g. an old one
    # without metric columns inside bsaa-ppg-project): use the first
    # folder whose parquet files have the border columns.
    for d in sorted(parquet_cands, key=lambda d: d.name != "seg10s_step5s"):
        ok, cols = parquet_has_columns(d)
        print(f"Parquet candidate {d}: {'OK' if ok else 'missing ' + cols}")
        if ok and res["parquet"] is None:
            res["parquet"] = d
    if PARQUET_DIR:
        res["parquet"] = Path(PARQUET_DIR)
    if EDF_DIR:
        res["edf"] = Path(EDF_DIR)
    if res["parquet"] is None and res["cache"] is None:
        raise FileNotFoundError(
            "No parquet folder with the metric columns"
            f" {REQUIRED_COLS} in /kaggle/input - attach the current"
            " hr-parts dataset or set PARQUET_DIR"
        )
    print("Inputs:", {k: str(v) for k, v in res.items()})
    return res


def link_cache(cache_src: Path | None) -> Path:
    """sweep_borders.py expects <cache-dir>/full/{index.parquet,segments.npy}."""
    cache_dir = WORK / "cache"
    if cache_src is None:
        return cache_dir
    if cache_src.name == "full" or cache_src.name.startswith("subset"):
        return cache_src.parent  # already <cache-dir>/<tag>/
    dst = TMP / "cache" / "full"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("index.parquet", "segments.npy", "meta.json"):
        if (cache_src / f).exists() and not (dst / f).exists():
            os.symlink(cache_src / f, dst / f)
    return dst.parent


def restore_results(sources: list[Path], out: Path):
    """Copies finished configs of an earlier session (never overwrites)."""
    for src in sources:
        if src.suffix == ".zip":
            tmp = TMP / "resume_zip"
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            found = list(tmp.rglob("sweep_manifest.json"))
            if not found:
                continue
            src = found[0].parent
        n = 0
        for p in src.rglob("*"):
            if p.is_file():
                dst = out / p.relative_to(src)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
                    n += p.parent.name == "results"
        print(f"Restored {n} finished configs from {src}")


def zip_dir(src: Path, dst: Path, skip=()):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if p.is_file() and not any(s in p.parts for s in skip):
                z.write(p, p.relative_to(src.parent))
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    code = setup_code()
    ensure_deps()
    subprocess.run(["nvidia-smi"], check=False)
    inp = find_inputs()
    out = WORK / "sweep_results"
    out.mkdir(parents=True, exist_ok=True)
    restore_results(inp["resume"], out)
    cache_dir = link_cache(inp["cache"])

    cmd = [
        sys.executable,
        "-u",
        str(code / "prototyping_dan" / "sweep_borders.py"),
        "--output-dir",
        str(out),
        "--cache-dir",
        str(cache_dir),
        "--time-budget-hours",
        str(TIME_BUDGET_HOURS),
        *SWEEP_ARGS,
    ]
    if inp["parquet"]:
        cmd += ["--parquet-dir", str(inp["parquet"])]
    if inp["edf"]:
        cmd += ["--edf-dir", str(inp["edf"])]
    print(" ".join(cmd), flush=True)
    env = dict(os.environ, TQDM_DISABLE="1")
    code_rc = subprocess.run(cmd, env=env).returncode
    if code_rc != 0 and (out / "results").exists():
        print(
            f"sweep exited with {code_rc}; building the report from what is done"
        )
        subprocess.run(
            [
                sys.executable,
                str(code / "prototyping_dan" / "report_borders.py"),
                "--results-dir",
                str(out),
            ],
            env=env,
        )

    zip_dir(out, WORK / "sweep_results.zip", skip=("locks",))
    if (out / "report").exists():
        zip_dir(out / "report", WORK / "sweep_report.zip")
    if ON_KAGGLE and not KEEP_CACHE_IN_OUTPUT:
        shutil.rmtree(WORK / "cache", ignore_errors=True)


if __name__ == "__main__":
    main()
