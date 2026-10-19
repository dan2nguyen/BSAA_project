# BSAA_project
DREAM TEAM

**This project is managet by UV** <br>
How to set project enviroment:<br>
>pip install uv<br>

For venv initialization and updating:
>uv sync<br>

To add pip library to uv
> uv add ...



Project dscription
<img width="1129" height="676" alt="image" src="https://github.com/user-attachments/assets/e342d970-db9f-4e0b-9e9b-57fd63340bd9" />



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
- With the current filter, 7 files in `wrist-finger-ppg` (`14_0`, `14_1`, `14_2`, `14_5`, `22_1`, `22_4`, `25_0`) cannot be exported to EDF and are missing from the filtered database.
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
