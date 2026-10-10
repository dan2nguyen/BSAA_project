"""Subject-wise 80/20 train/test split of the harmonized EDF databases.

Expected layout:  ROOT/<database>/<subject>_<anything>.edf
  e.g. 21_1.edf -> subject 21 | P9_watch_6.edf, P9_ring_6.edf -> P9 | s1_walk.edf -> s1

All files of one subject go to the same set (no leakage), and the split is
done per database so every database is ~80/20.
Test files are MOVED to ROOT_test/<database>/; train files stay where they are.

Usage:  python split_data.py ROOT [--test-size 0.2] [--seed 42] [--dry-run]
        python split_data.py ROOT --from-manifest split_manifest.csv   # reproduce a teammate's split
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


def subject_of(path: Path) -> str:
    return path.stem.split("_")[0]


def apply_manifest(root, test_root, manifest, dry_run):
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    listed = {r["file"] for r in rows}
    present = {
        f.relative_to(root).as_posix() for f in root.rglob("*.edf")
    }
    already = (
        {
            f.relative_to(test_root).as_posix()
            for f in test_root.rglob("*.edf")
        }
        if test_root.exists()
        else set()
    )
    missing = listed - present - already
    extra = present - listed
    if missing or extra:
        raise SystemExit(
            f"Your data differs from the manifest.\n  missing: {sorted(missing)[:10]}\n"
            f"  not in manifest: {sorted(extra)[:10]}"
        )
    moved = 0
    for r in rows:
        src = root / r["file"]
        if r["set"] == "test" and src.exists():
            moved += 1
            if not dry_run:
                dst = test_root / r["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    n_test = sum(r["set"] == "test" for r in rows)
    print(
        f"{len(rows) - n_test} train / {n_test} test files; {'would move' if dry_run else 'moved'} {moved} to {test_root}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="only print, move nothing",
    )
    ap.add_argument(
        "--from-manifest",
        type=Path,
        help="apply an existing split_manifest.csv",
    )
    a = ap.parse_args()

    root = a.root.resolve()
    test_root = root.with_name(root.name + "_test")
    if a.from_manifest:
        apply_manifest(root, test_root, a.from_manifest, a.dry_run)
        return
    manifest = root.with_name("split_manifest.csv")
    if manifest.exists() and not a.dry_run:
        raise SystemExit(
            f"{manifest} exists - data already split. Not splitting twice."
        )

    rows = []
    for db in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(db.rglob("*.edf"))
        subjects = sorted({subject_of(f) for f in files})
        if not subjects:
            continue
        rng = random.Random(
            f"{a.seed}-{db.name}"
        )  # reproducible, independent per database
        rng.shuffle(subjects)
        n_test = max(1, round(len(subjects) * a.test_size))
        test = set(subjects[:n_test])
        print(
            f"{db.name}: {len(subjects)} subjects -> {len(subjects) - n_test} train / {n_test} test"
            f"   test = {sorted(test)}"
        )
        for f in files:
            s = subject_of(f)
            part = "test" if s in test else "train"
            rel = f.relative_to(root)
            rows.append([db.name, s, rel.as_posix(), part])
            if part == "test" and not a.dry_run:
                dst = test_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))

    if a.dry_run:
        print("dry run - nothing moved")
        return
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["database", "subject", "file", "set"])
        w.writerows(rows)
    print(f"test files moved to {test_root}\nmanifest: {manifest}")


if __name__ == "__main__":
    main()
