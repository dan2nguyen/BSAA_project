# BSAA_project
DREAM TEAM

Heart rate (HR) estimation from PPG with a 1D CNN. The reference HR comes from
the simultaneously recorded ECG (Pan-Tompkins). Segments whose PPG is not
usable are rejected by quality borders, and the model returns **NONE** for
them instead of a wrong HR.

**This project is managed by uv.** Set up the environment:
```bash
pip install uv
uv sync          # create / update the venv
uv add <package> # add a library
```
All commands are run from the repository root (`uv run python ...`).

<img width="1129" height="676" alt="image" src="https://github.com/user-attachments/assets/e342d970-db9f-4e0b-9e9b-57fd63340bd9" />

## Pipeline and key scripts

| # | Step | Script | Output |
|---|---|---|---|
| 0 | Subject-wise 80/20 train/test split | `utils/split_data.py` | `databasing/references/split_manifest.csv` |
| 1 | Filtering of the raw EDF database | `utils/filt_data_folder.py` (filters in `exploratory_analysis/filters.py`) | `data/harmonized_ecg_ppg_database_filtered/` |
| 2 | Segment database: quality metrics + ECG reference HR per 10 s segment (step 5 s) | `databasing/pan_tompkins/build_hr_database.py` (metrics: `databasing/evaluation_signal_segments.py`, HR: `detection.py`, `heart_rate.py`) | `data/hr_parts/seg10s_step5s/<subset>/<file>.parquet` |
| 3 | Finding the quality borders: model trained on all segments, its errors analysed per metric | `prototyping_dan/predict_training_segments.py`, `run_commands.py`, `analyze_predictions.py`, `analyze_folds_combined.py` | `data/fold_analysis/combined_report/` |
| 4 | Feasibility column (`feasible_bpm` = HR inside all borders, else NaN) | `prototyping_dan/feasible_bpm.py` (`FEASIBILITY_BOUNDS`) | column in the parquet files |
| 5 | Training (patient-wise 5-fold CV, target `feasible_bpm`) | `prototyping_dan/training.py` | `data/training_out_cnn/dilated_cnn/fold_<k>/best_model.pt`, OOF predictions |
| 6 | Tuning the borders (sweep on Kaggle GPUs) | `prototyping_dan/sweep_borders.py`, `report_borders.py`, `kaggle/` | sweep report (see the last section) |
| 7 | Prediction for one EDF recording: HR or NONE | `predict.py` | printed / CSV |
| 8 | Evaluation on the test database | `evaluate_test.py` | `data/test_evaluation/<model>_<fold>_<mode>/report.md` |

Shared helpers: `utils/read_edf.py` (reads the ECG/PPG channels of an EDF
file), `prototyping_dan/load_database.py` (reads the parquet database, patient
IDs, CV groups).

# Database filtering

The raw database is filtered before anything else (power-line noise,
baseline drift):
- `exploratory_analysis/filters.py`: `apply_standard_filters` applies
  - **ECG**: 50 Hz notch + Butterworth band-pass 0.5-40 Hz,
  - **PPG**: Butterworth band-pass 0.5-8 Hz.
- `utils/filt_data_folder.py`: filters a whole database folder and saves it
  as the filtered database in EDF format (`filter_edf()` does one file; it is
  reused by `predict.py`).

```bash
uv run python utils/filt_data_folder.py data/harmonized_ecg_ppg_databases
```
The results are saved to `data/harmonized_ecg_ppg_database_filtered/`. To
re-filter, delete the output folder or its `filtered_progress.csv` first
(the script tracks progress to resume interrupted runs).

# Pan-Tompkins HR and segment database (`databasing/pan_tompkins/`)

Ground-truth heart rate from ECG (Pan-Tompkins QRS detection) together with the signal quality flags of every segment. All commands are run from the repository root.

## Files

| File | What it does |
|---|---|
| `detection.py` | `detect_qrs(ecg, fs)`: QRS peak detection. Also `pan_tompkins_stages` and `pan_tompkins_threshold`, used only for the illustration plot. |
| `heart_rate.py` | `compute_hr_segments(...)`: HR per segment from the peaks. `estimate_hr(ecg, fs, segment_s, step_s)`: detection + HR + flat-ECG check in one call. |
| `build_hr_database.py` | Builds the segment database (quality flags + HR) for all EDF files. |
| `plots.py` | Slide plots: ECG with detected peaks, HR over time, Pan-Tompkins stages. |

## How it works

1. **QRS detection.** The peaks are detected by `sleepecg.detect_heartbeats` (a modified Pan-Tompkins: bandpass, derivative, squaring, moving window integration, adaptive threshold). `detect_qrs` runs it on 300 s chunks with 10 s overlap, z-scores every chunk (amplitudes differ a lot between files), keeps only the peaks from the chunk core (no duplicates) and skips chunks where the signal does not change at all.
2. **HR per segment.** For every segment, `HR = 60 / mean RR` from the RR intervals inside the segment. The median RR, RR standard deviation and `rr_cv = std / mean` are stored as well.
3. **Invalid HR (HR = NaN).** A segment gets no HR when:

   | `invalid_reason` | Condition |
   |---|---|
   | `flat_ecg` | ECG is flat (`ecg_is_flat` from `compute_metrics` in `databasing/evaluation_signal_segments.py`) |
   | `too_few_beats` | fewer than 3 peaks (`MIN_BEATS`) |
   | `hr_out_of_range` | HR outside 30–220 bpm (`HR_RANGE_BPM`) |
   | `irregular_rr` | `rr_cv` > 0.15 (`MAX_RR_CV`): noise, missed or extra beats |

   The thresholds are constants at the top of `heart_rate.py`.

## Input data

The scripts read the **filtered** database `data/harmonized_ecg_ppg_database_filtered/`, created by:

```bash
uv run python utils/filt_data_folder.py data/harmonized_ecg_ppg_databases
```

- To re-filter, move or delete the old filtered folder first. Otherwise `filtered_progress.csv` marks every file as done and nothing is re-filtered.
- The 4 files `multisite-ppg/P3_*_7.edf` are very long (about 1.4 GB each). Filtering them needs a lot of memory and can take about 30 minutes per file on a laptop with 8 GB RAM.

## HR for one file: `heart_rate.py`

```bash
uv run python databasing/pan_tompkins/heart_rate.py data/harmonized_ecg_ppg_database_filtered/wrist-ppg-during-exercise-1.0.0/s9_walk.edf
```

Prints the table of segments and the median HR.

| Option | Meaning | Default |
|---|---|---|
| `--segment-s` | segment length in seconds | 10 |
| `--step-s` | step between segment starts in seconds | equal to the segment (no overlap) |

## Plots: `plots.py`

```bash
# all three plots for the default file (wrist-finger-ppg/01_4)
uv run python databasing/pan_tompkins/plots.py

# a chosen file, ECG window from 60 s to 75 s, only the ECG plot, also shown on screen
uv run python databasing/pan_tompkins/plots.py data/harmonized_ecg_ppg_database_filtered/wrist-ppg-during-exercise-1.0.0/s9_walk.edf --start-s 60 --duration-s 15 --plots ecg_peaks --show
```

The PNG files are saved as `data/plots/<file>_<plot>.png`.

| Option | Meaning | Default |
|---|---|---|
| `--plots` | which plots: `ecg_peaks`, `hr`, `pt_stages` | all three |
| `--start-s` | start of the ECG window, seconds from the recording start | 0 (or the slide preset) |
| `--duration-s` | length of the ECG window | 10 (or the slide preset) |
| `--hr-start-s`, `--hr-end-s` | time range of the HR curve | whole recording |
| `--segment-s`, `--step-s` | segment length and step for HR | 10, no overlap |
| `--output-dir` | folder for the PNG files | `data/plots` |
| `--show` | also open the plots in a window | off |

What the plots show:

- **`ecg_peaks`**: z-scored ECG in the chosen window. Orange dots are peaks in segments with valid HR, grey crosses are rejected peaks, grey background marks segments without HR, dashed lines are segment borders.
- **`hr`**: HR over time, one point per segment (at the segment centre). Grey bands are segments without HR. The time axis switches to hours when the plotted range is at least 1 hour; long flat stretches are labelled "no ECG signal".
- **`pt_stages`**: the classic Pan-Tompkins steps (5–15 Hz bandpass, derivative, squaring, 150 ms integration) and the adaptive threshold with the detected peaks, on at most the first 5 s of the window. This is an illustration only; the detection itself uses `sleepecg`.

`SLIDE_PRESETS` in `plots.py` stores the default window and labels for `multisite-ppg/P10_necklace_4` (44455–44500 s: dead signal, transient, valid ECG). They are used when neither `--start-s` nor `--duration-s` is given.

## Segment database: `build_hr_database.py`

```bash
# test run on 10 random files (prints a summary and a time estimate, writes no database file)
uv run python databasing/pan_tompkins/build_hr_database.py --limit 10

# full database, 10 s segments with 50 % overlap
uv run python databasing/pan_tompkins/build_hr_database.py
```

| Option | Meaning | Default |
|---|---|---|
| `--input` | filtered database folder | `data/harmonized_ecg_ppg_database_filtered` |
| `--segment-s` | segment length in seconds | 10 |
| `--step-s` | step between segments | half of the segment (50 % overlap) |
| `--manifest` | train/test split file | `databasing/references/split_manifest.csv` |
| `--output` | final parquet file | `data/hr_database_seg<segment>s_step<step>s.parquet` |
| `--parts-dir` | folder for the per-file results | `data/hr_parts/seg<segment>s_step<step>s/` |
| `--workers` | files processed in parallel | 4 |
| `--large-file-gb` | files at least this large are processed at the end, one at a time | 0.5 |
| `--limit N` | test run on N random files, no final file | off |
| `--seed` | seed for `--limit` | 0 |
| `--overwrite` | recompute files that already have a result | off |

- **Resuming:** every file is stored as its own parquet in the parts folder. If the run is interrupted (Ctrl+C), running the same command again continues with the remaining files.
- **After changing a threshold** in `heart_rate.py`, run with `--overwrite`; otherwise the old per-file results are reused.
- **Memory:** every worker holds one whole file in memory. Files of at least `--large-file-gb` (only the four `P3_*_7` files, about 1.4 GB on disk and about 5.6 GB in memory each) are therefore processed after all other files, one at a time. On a laptop with 8 GB RAM they still use swap and take a while.
- **Time:** about 1 hour 20 minutes for all other files with 4 workers (estimate from a 10-file test run), plus the four `P3_*_7` files.

### Columns of the database

One row per segment. `section_id`, `start` and `end` match the output of `databasing/evaluation_signal_segments.py` for the same window and step.

| Column | Meaning |
|---|---|
| `base_folder`, `edf_filename` | subset folder and EDF file name |
| `section_id` | index of the segment within the file |
| `split` | `train` or `test` (from the split manifest) |
| `start`, `end` | segment borders in samples |
| `start_t`, `end_t` | segment borders in seconds from the recording start |
| `segment_s`, `sampling_rate` | segment length (s) and sampling rate (Hz) |
| `is_flat` | `ecg_is_flat` OR `ppg_is_flat`: the segment is unusable |
| `ecg_is_flat`, `ppg_is_flat` | flat signal |
| `ecg_/ppg_mean_absolute_derivative_spike` | spike flag |
| `ecg_/ppg_LF_ratio_motion` | low-frequency (motion) flag |
| `ecg_/ppg_InBand_power_ratio_out` | too little power in the signal band |
| `ecg_/ppg_kurtosis_out` | kurtosis outside the expected range |
| `ecg_/ppg_Skewness_out` | skewness outside the expected range |
| `hr_bpm` | Pan-Tompkins HR (NaN if invalid) |
| `hr_valid` | HR passed all checks |
| `invalid_reason` | why HR is NaN (empty if valid) |
| `n_beats`, `hr_median_bpm`, `rr_mean_s`, `rr_median_s`, `rr_std_s`, `rr_cv` | beat and RR statistics (also stored for `irregular_rr` segments, so the RR threshold can be changed without recomputing) |

The flag columns come from `compute_metrics` in `databasing/evaluation_signal_segments.py`; `True` means the problem is present.

A segment is usable as ground truth when `is_flat` is False and `hr_valid` is True:

```python
import pandas as pd

db = pd.read_parquet("data/hr_database_seg10s_step5s.parquet")
usable = db[~db.is_flat & db.hr_valid]
train = usable[usable.split == "train"]
```

# Quality borders (`prototyping_dan/`)

The model only gets segments whose signal metrics lie inside fixed borders.
They were found from the errors of a model trained on all usable segments:

1. `predict_training_segments.py`: predictions of each fold's model on its
   training-database segments (`run_commands.py` runs all folds, then
   `analyze_predictions.py` makes a per-fold report).
2. `analyze_folds_combined.py`: for every metric, the border that rejects a
   fixed share of segments where the error is largest
   (`data/fold_analysis/combined_report/report.md`).
3. The combined borders are stored in `FEASIBILITY_BOUNDS` in
   `feasible_bpm.py` (strict `lower < value < upper`, NaN fails).
   `PPG_BOUNDS` is the PPG part, the only one available at prediction time.
4. Add the `feasible_bpm` column to the parquet files (after changing the
   borders, run it again):
   ```bash
   uv run python prototyping_dan/feasible_bpm.py --parquet_dir data/hr_parts/seg10s_step5s
   ```

Helper plots: `plot_feasible_hist.py` (HR histogram of the feasible
segments), `plot_hr_scatter.py` (reference vs predicted HR).

# Training (`prototyping_dan/training.py`)

```bash
uv run python prototyping_dan/training.py --db_path data/hr_parts/seg10s_step5s --edf_dir data/harmonized_ecg_ppg_database_filtered --output_dir data/training_out_cnn --model dilated_cnn
```

- Data: train-split segments with a valid HR, not flat, and (default
  `--target_col feasible_bpm`) inside the borders. The PPG is z-scored per
  segment and kept in GPU memory.
- CV: patient-wise `StratifiedGroupKFold` (5 folds); `--folds 1` trains only
  fold 1.
- Model: `dilated_cnn` (residual CNN with dilations 1-64, sees the whole
  10 s window). The architecture and training choices are explained in
  `prototyping_dan/architecture_and_training.md`.
- Important options: `--epochs 15`, `--batch_size 512`, `--lr 1e-3`,
  `--reweight sqrt_inv` (LDS weighting of rare HR ranges), `--patience 5`,
  `--cache_dir` (cache of the loaded segments).
- Output: `data/training_out_cnn/dilated_cnn/fold_<k>/best_model.pt`,
  learning curves, `oof_predictions_dilated_cnn.csv`.

# Prediction (`predict.py`)

Input: one **raw** EDF recording (like `data/harmonized_ecg_ppg_databases`).
Output: HR in bpm for every 10 s segment (step 5 s), or **NONE**, plus one HR
for the whole recording (median of the feasible segments, or NONE).

```bash
uv run python predict.py data/harmonized_ecg_ppg_databases/wrist-finger-ppg/14_0.edf
uv run python predict.py <file.edf> --model-path data/training_out_cnn/dilated_cnn/fold_2/best_model.pt --output preds.csv
```

Steps, each with the same code that made the training data:
1. filtering with `filter_edf()` of `utils/filt_data_folder.py`, including
   the EDF export (16-bit quantization; it changes e.g. `zero_dev_dur`),
2. segments laid out like the database (start 0, 500, 1000, ... samples),
3. metrics with `compute_segment_metrics` of `build_hr_database.py`,
4. feasibility: not flat and inside the borders,
   - `--feasibility ppg` (default): `PPG_BOUNDS`, needs only the PPG, as in
     real use,
   - `--feasibility full`: all borders (ECG + PPG), exactly the training
     selection, needs the ECG channel,
5. per-segment z-score -> model (default
   `data/training_out_cnn/dilated_cnn/fold_1/best_model.pt`).

Other options: `--input-filtered` (file is already from the filtered
database), `--output` (CSV per segment with the metrics and the reason for
NONE), `--device`. Checked: the metrics from the raw EDF are identical to the
database values (difference 0) and the feasibility agrees 100 %.

In code: `HRPredictor(model_path).predict_edf(path)` returns the segment
table, `recording_hr(table)` the recording HR.

# Test evaluation (`evaluate_test.py`)

Runs `predict.py` on every test-split recording (raw EDF files) and compares
it with the ECG reference HR on the test segments with a valid HR that are
not flat (the same population as training).

```bash
uv run python evaluate_test.py
uv run python evaluate_test.py --model-path data/training_out_cnn/dilated_cnn/fold_2/best_model.pt --feasibility full
uv run python evaluate_test.py --limit-files 2 --output-dir data/test_evaluation/smoke   # quick test
```

Report (`data/test_evaluation/<model>_<fold>_<mode>/report.md`, plus
`summary.json`, `segments.parquet`, figures):
- **prediction**: coverage (share of test segments that get an HR), MAE,
  RMSE, accuracy within ±5 bpm, bias, balanced MAE (mean over 10 bpm bins),
- the model on all segments (no filter) and on the rejected ones, the
  full-borders selection and the fold's CV result for comparison,
- per dataset, per subject, per HR bin, the reasons for NONE, and the
  pipeline check (recomputed metrics vs the database).

Result of the current model (`fold_1`, PPG borders, 6 Nov 2026,
`data/test_evaluation/dilated_cnn_fold_1_ppg/report.md`): HR for 66.5 % of
the 171.7k test segments, MAE 4.71 bpm, RMSE 8.48 bpm, 72.6 % within ±5 bpm
(no filter: MAE 5.86 bpm; full ECG + PPG borders: 31.8 % coverage, MAE
3.99 bpm; CV fold 1: MAE 3.32 bpm).

Options: `--workers 2` (RAM: ~1-2 GB per worker; recordings longer than
`--big-file-samples` run one at a time), `--input-filtered` (faster, same
signals), `--all-segments` (also the segments without a reference HR).

# Other scripts

Every script describes itself at the top of the file.

| Script | Purpose |
|---|---|
| `databasing/evaluation_signal_segments.py` | `compute_metrics()` (the quality metrics, used by the database); run directly it writes the old 20 s evaluation table |
| `databasing/pan_tompkins/plots.py` | presentation plots of the Pan-Tompkins HR (see above) |
| `databasing/read_parquet.py` | prints a slice of a parquet file |
| `exploratory_analysis/explore_plot.py` | interactive viewer of raw vs filtered signals with spectra (`time_frequency.py`) |
| `exploratory_analysis/plot_signals_section.py`, `exclude_corupted_sections.py`, `analyze_corruption_stats.py` | old exploration based on the 20 s evaluation table |
| `plot_single_edf.py` | raw vs filtered ECG/PPG of one file |
| `prototyping_dan/predict_hr.py` | old version of `predict.py` (no filtering, older checkpoint paths) |
| `prototyping_dan/train_signals_fisebility.py` | old version of `training.py` |
| `prototyping_dan/commands.txt` | old command list for `predict_training_segments.py` |

# Border sweep on Kaggle (status: **running**)

Goal: find the borders in `FEASIBILITY_BOUNDS` with the best trade-off
between HR accuracy and coverage. Only the borders change; data, model and
training settings stay as in `training.py`. The plan is in
`planning/cnn_borders_testing`.

- `prototyping_dan/sweep_borders.py`: trains the model on CV fold 0 once per
  border configuration and evaluates it on the same fixed validation
  subjects. Levels are defined by coverage: a level `r` is the border that
  alone rejects the share `r` of the training part
  (`0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4`). Phases:
  1. baselines (no filter, current borders) + one border at a time,
  2. joint grid of the most important borders (at most 20 configurations),
  3. extra seeds (43, 44) for no filter, current and the best
     configurations.

  Random-subset controls of the same size show whether a border helps by
  picking better training data or only by making the validation set easier.
  Every finished configuration is saved at once (`results/<hash>.json`), so
  an interrupted run resumes.
- `prototyping_dan/report_borders.py`: the report: risk-coverage curves, AURC
  per variable, recommended `FEASIBILITY_BOUNDS` (highest validation
  coverage with feasible MAE <= 5 bpm, better than its random control),
  per-subject coverage, learning curves, seed spread.
- `kaggle/`: `run_sweep.py` (kernel script; settings at the top),
  `kernel-metadata.json`, `stage_code.py` (uploads the code as a dataset),
  `README_kaggle.md` (all commands: upload, push, download, resume).

Status (6 Nov 2026): kernel `dannguyen2/bsaa-border-sweep` (version 2) is
**still running** on 2× T4, one configuration per GPU (about 1 min per epoch
for the current borders, 2 min without a filter). Phase 1 alone takes about
8-11 h, so the session (12 h limit, the sweep stops starting new
configurations after 11 h) will probably end before the grid and the seeds;
these then continue in a resumed session (`kaggle/README_kaggle.md`, section
4). Results: `sweep_report.zip` / `sweep_results.zip` in the kernel output.

When it is done: copy the recommended borders into `FEASIBILITY_BOUNDS`,
re-run `feasible_bpm.py`, retrain (`training.py`) and evaluate with
`evaluate_test.py`.
