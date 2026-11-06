"""
Copies the code needed by the sweep into kaggle/_build/<name>/ with a
dataset-metadata.json, ready for `kaggle datasets create/version`.

Files are staged flat (no subfolders), so no --dir-mode is needed;
run_sweep.py rebuilds the prototyping_dan/ and utils/ packages on Kaggle.

Usage:
    python kaggle/stage_code.py                  # code dataset
    python kaggle/stage_code.py --cache          # + metadata for the cache dataset
    python kaggle/stage_code.py --resume DIR     # metadata for a resume dataset
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path(__file__).resolve().parent / "_build"
USER = "dannguyen2"
CODE_FILES = [
    "prototyping_dan/training.py",
    "prototyping_dan/feasible_bpm.py",
    "prototyping_dan/load_database.py",
    "prototyping_dan/sweep_borders.py",
    "prototyping_dan/report_borders.py",
    "utils/read_edf.py",
]


def write_meta(folder: Path, slug: str, title: str):
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{USER}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", action="store_true",
                   help="write dataset-metadata.json into the local cache"
                   " folder data/sweep_borders/cache/full")
    p.add_argument("--resume", default=None,
                   help="folder with downloaded sweep_results to upload")
    a = p.parse_args()

    code = BUILD / "bsaa-code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    for f in CODE_FILES:
        shutil.copy2(ROOT / f, code / Path(f).name)
    write_meta(code, "bsaa-code", "bsaa-code")
    print(f"Code dataset staged in {code}")

    if a.cache:
        cache = ROOT / "data" / "sweep_borders" / "cache" / "full"
        if not (cache / "segments.npy").exists():
            raise FileNotFoundError(
                f"{cache} has no segments.npy - build it with"
                " sweep_borders.py --build-cache-only")
        write_meta(cache, "bsaa-sweep-cache", "bsaa-sweep-cache")
        print(f"Cache dataset metadata written to {cache}")
    if a.resume:
        write_meta(Path(a.resume), "bsaa-sweep-resume", "bsaa-sweep-resume")
        print(f"Resume dataset metadata written to {a.resume}")


if __name__ == "__main__":
    main()
