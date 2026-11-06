# Border sweep on Kaggle

Runs `prototyping_dan/sweep_borders.py` (fold 0) on a Kaggle GPU kernel,
then `report_borders.py`. All commands run from the repo root in PowerShell
or bash. CLI syntax was checked against the Kaggle CLI docs
(github.com/Kaggle/kaggle-cli, `docs/`), November 2026.

## 0. One-time setup

```bash
pip install kaggle            # or: uv tool install kaggle
kaggle auth login             # or put kaggle.json into ~/.kaggle/
```

Limits (Kaggle docs / account page, Nov 2026): GPU sessions stop after
**12 h**, and the weekly GPU quota is **~30 h** (see *Settings → Quota*).
`machine_shape: NvidiaTeslaT4` = **2× T4 (16 GB each)**. P100 is no longer
listed as a CLI accelerator. A T4×2 session counts once against the quota,
and the sweep runs one config per GPU, so you get twice the throughput.

## 1. Datasets on Kaggle

| dataset | content | status |
| --- | --- | --- |
| `dannguyen2/hr-parts` | parquet database, `seg10s_step5s/<base_folder>/*.parquet` (needs the metric columns; `feasible_bpm` is not needed) | already exists |
| `dannguyen2/bsaa-ppg-project` | `harmonized_ecg_ppg_database_filtered/` EDF files | already exists (only needed if the cache is built on Kaggle) |
| `dannguyen2/bsaa-code` | the 6 code files (flat) | **new**, upload below |
| `dannguyen2/bsaa-sweep-cache` | prebuilt cache: `index.parquet`, `segments.npy` (~1.3 GB), `meta.json` | **optional but recommended** |
| `dannguyen2/bsaa-sweep-resume` | `sweep_results.zip` of an earlier session | only for resuming |

### Code (each time the code changes)

```bash
python kaggle/stage_code.py
kaggle datasets create  -p kaggle/_build/bsaa-code          # first time
kaggle datasets version -p kaggle/_build/bsaa-code -m "sweep code update"   # later
```

### Cache (optional: saves reading 14 GB of EDF on Kaggle)

```bash
python prototyping_dan/sweep_borders.py --build-cache-only     # -> data/sweep_borders/cache/full
python kaggle/stage_code.py --cache
kaggle datasets create -p data/sweep_borders/cache/full
```
Then add `"dannguyen2/bsaa-sweep-cache"` to `dataset_sources` in
`kaggle/kernel-metadata.json`. You can then drop `bsaa-ppg-project` too.
Without the cache dataset, the kernel builds the cache itself in
`/kaggle/working/cache` (EDF reading on Kaggle's 4 CPUs is slow and counts
against the GPU session).

Wait until the datasets are ready: `kaggle datasets status dannguyen2/bsaa-code`.

## 2. Run

Sweep settings are at the top of `kaggle/run_sweep.py` (`SWEEP_ARGS`,
`TIME_BUDGET_HOURS`).

Preview the plan locally first (no training):
```bash
python prototyping_dan/sweep_borders.py --dry-run
```

Push and run the kernel:
```bash
kaggle kernels push -p kaggle
kaggle kernels status dannguyen2/bsaa-border-sweep
```
`status` shows `queued` / `running` / `complete` / `error`. The log is on
the kernel page (kaggle.com/code/dannguyen2/bsaa-border-sweep). At startup
it prints the GPU (`nvidia-smi`), the GPU memory, and a
`~N s/epoch on the full training part` benchmark per GPU.

## 3. Download the results

```bash
kaggle kernels output dannguyen2/bsaa-border-sweep -p data/sweep_borders/kaggle_out --file-pattern "sweep_.*\.zip$"
```
That downloads only `sweep_report.zip` (report.md, figures, CSVs) and
`sweep_results.zip` (all raw results, used for resuming), not the 1.3 GB
cache. Unzip, then open `sweep_results/report/report.md`. To rebuild the
report locally (e.g. with another selection rule):
```bash
python prototyping_dan/report_borders.py --results-dir data/sweep_borders/kaggle_out/sweep_results --max-mae 6
python prototyping_dan/report_borders.py --results-dir data/sweep_borders/kaggle_out/sweep_results --min-coverage 0.4
```

## 4. Resume an interrupted run

Each finished config is saved immediately (`results/<hash>.json`), and the
sweep stops starting new configs after `TIME_BUDGET_HOURS` (11 h), then
writes the report and the zips. If the session hits the 12 h limit
anyway, the last finished configs are still in the output. Then:

1. Download `sweep_results.zip` (step 3).
2. Upload it as the resume dataset:
   ```bash
   mkdir -p kaggle/_build/resume
   cp data/sweep_borders/kaggle_out/sweep_results.zip kaggle/_build/resume/
   python kaggle/stage_code.py --resume kaggle/_build/resume
   kaggle datasets create  -p kaggle/_build/resume          # first time
   kaggle datasets version -p kaggle/_build/resume -m "resume"   # later
   ```
3. Add `"dannguyen2/bsaa-sweep-resume"` to `dataset_sources` in
   `kaggle/kernel-metadata.json` (once), then `kaggle kernels push -p kaggle` again.

`run_sweep.py` copies the earlier results into `/kaggle/working/sweep_results`.
The sweep skips every config whose hash already has a result, and then
continues (e.g. with the grid phase). Never change `--epochs`, the
hyperparameters or `--fold` between sessions. They are part of the
config hash, so changed settings start a new sweep, and the folder warns about it.

## Locally

The same scripts run locally (CPU or GPU) with the repo's `data/` paths:
```bash
python prototyping_dan/sweep_borders.py --strategy one-at-a-time --max-configs 2 --epochs 1 --subset-frac 0.03 --output-dir data/sweep_smoke
python kaggle/run_sweep.py    # full kernel logic with local paths
```
