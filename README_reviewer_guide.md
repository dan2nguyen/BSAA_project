## Guide for the reviewer (algorithm optimisation)

**Method:** a 1D CNN (`dilated_cnn`) that estimates HR from 10 s PPG
segments, with a quality check in front of it that returns **NONE** for
segments whose PPG is not usable.

All optimisation uses the **training split only** (subject-wise 80/20 split,
`databasing/references/split_manifest.csv`). The test split is not used for
any choice; it is evaluated once, at the end, with `evaluate_test.py`.

| What was optimised | How | Where |
|---|---|---|
| Training loop | Patient-wise 5-fold cross-validation, early stopping, learning-rate schedule, re-weighting of rare HR ranges | `prototyping_dan/training.py`, explained in `prototyping_dan/architecture_and_training.md` |
| Plausibility checks / quality borders | Model errors analysed per signal metric; borders reject segments where the error is large | `prototyping_dan/analyze_predictions.py`, `analyze_folds_combined.py`, `feasible_bpm.py` |
| Border tuning (grid search) | Sweep over border levels, trade-off between MAE and coverage, random-subset controls, several seeds. Result: no clear gain over the current borders, so they were kept | `prototyping_dan/sweep_borders.py`, `report_borders.py`, `kaggle/` |
| Reference HR plausibility | At least 3 beats, 30–220 bpm, RR variation `rr_cv` ≤ 0.15 | `databasing/pan_tompkins/heart_rate.py` |
| Investigation of large errors | Error per dataset, subject, HR range and quality metric | `prototyping_dan/analyze_predictions.py`, `plot_hr_scatter.py` |

### Results on the training set (cross-validation)

Source: `data/training_out_cnn/dilated_cnn/oof_predictions_dilated_cnn.csv`.
Patient-wise 5-fold cross-validation on the training split: every segment is
predicted by the model of the fold in which its subject was **not** used for
training (out-of-fold). Only segments inside the quality borders are
included (the others get NONE). Reference: ECG Pan-Tompkins HR.
Limits of agreement (LoA) = bias ± 1.96 SD (Bland-Altman).

| | Segments | MAE (bpm) | RMSE (bpm) | Bias (bpm) | 95 % LoA (bpm) | Within ±5 bpm |
|---|---|---|---|---|---|---|
| **All** | 314,289 | **4.04** | 7.17 | +0.79 | −13.2 to +14.8 | 77.1 % |
| multisite-ppg | 311,327 | 4.06 | 7.20 | +0.81 | −13.2 to +14.8 | 76.9 % |
| wrist-finger-ppg | 2,918 | 2.14 | 2.81 | −0.56 | −6.0 to +4.8 | 93.3 % |
| wrist-ppg-during-exercise | 44 | 9.60 | 11.67 | −9.18 | −23.5 to +5.1 | 31.8 % |

| Fold | Segments | MAE (bpm) | RMSE (bpm) | Bias (bpm) | Within ±5 bpm |
|---|---|---|---|---|---|
| 1 | 67,364 | 3.32 | 6.90 | −0.45 | 81.9 % |
| 2 | 61,469 | 4.03 | 6.28 | −0.97 | 78.0 % |
| 3 | 61,447 | 3.98 | 6.56 | +1.13 | 75.5 % |
| 4 | 61,832 | 3.22 | 5.97 | +0.52 | 82.4 % |
| 5 | 62,177 | 5.70 | 9.56 | +3.82 | 67.3 % |

Observations:
- The CNN is most accurate on the wrist-finger data and good on the
  multisite data.
- The quality borders reject almost all exercise segments: only 44
  segments of one subject remain, and these are still inaccurate. Strong
  motion is the main limitation.
- The folds differ mainly because their validation subjects differ (folds 1,
  4 and 5 contain only 2–3 multisite subjects each), not because one model
  is clearly better.

#### Where the model has problems

Figures in `results/cnn_validation/`, made with
`prototyping_dan/plot_cnn_validation.py` from the out-of-fold predictions:

```bash
uv run python prototyping_dan/plot_cnn_validation.py data/training_out_cnn/dilated_cnn/oof_predictions_dilated_cnn.csv --output-dir results/cnn_validation
```

| Figure | Finding |
|---|---|
| `1_error_by_site.png` | Sensor site matters: earring MAE 1.8 bpm and wrist-finger 2.1 bpm, ring 3.6, watch 5.5, necklace 7.0, exercise 9.6 bpm. |
| `2_error_by_hr.png` | Most accurate at 70–90 bpm (MAE about 3 bpm). Errors grow at low HR (40–60 bpm: 7.3) and strongly at high HR (100–120 bpm: 8.1; above 120 bpm: 30.4, few segments). |
| `3_pred_vs_true.png` | The model rarely predicts more than about 120–125 bpm, so high heart rates are underestimated; high HR is rare in the training data. |
| `4_bland_altman.png` | Bias +0.79 bpm, 95 % limits of agreement −13.2 to +14.8 bpm. |
| `5_error_by_subject.png` | A few subjects cause most of the large errors (exercise s3, multisite P14 and P2); all wrist-finger subjects are below 4.1 bpm. |

Possible improvements: more training data or re-weighting for high heart
rates, and site-specific quality borders (necklace and watch).

**Effect of the quality borders** (model trained on all segments, analysed in
`data/fold_analysis/combined_report/report.md`): keeping the segments inside
all borders (43 %) lowers the MAE from 7.1 to 4.1 bpm. The border sweep found
no configuration that was clearly better than the current borders.


### Training experiments

Training length and learning rate were tested with the same patient-wise
5-fold cross-validation on the same segments (run on Kaggle with
`prototyping_dan/training.py`):

| | Original | Higher LR | Lower LR |
|---|---|---|---|
| Learning rate | 1·10⁻³ | 2·10⁻³ | 5·10⁻⁴ |
| Epochs | 15 | 40 (min. 30) | 40 (min. 30) |
| MAE (bpm) | **4.04** | 4.12 | 4.20 |
| RMSE (bpm) | **7.17** | 7.29 | 7.44 |
| Within ±5 bpm | **77.1 %** | 76.9 % | 76.8 % |
| Balanced MAE (bpm) | 20.7 | 21.9 | **19.4** |

- Neither change improved the overall result. The differences are small,
  so training length and learning rate are not what limits the model.
- With the higher learning rate the validation curves jump strongly in the
  first ~10 epochs. In all runs the validation error levels off after about
  10–15 epochs while the training error keeps decreasing, so longer training
  brings no benefit.
- The lower learning rate gives smoother curves and slightly smaller errors
  above 110 bpm, but slightly larger errors at normal heart rates.
- The remaining errors come mainly from heart rates above about 110 bpm,
  which are rare in the training data.

**Final model:** the original settings (learning rate 1·10⁻³, 15 epochs),
trained once on all training subjects with `prototyping_dan/train_full.py`
and then evaluated once on the test split with `evaluate_test.py`.

### Results on the test set

Final model (all training subjects, original settings), evaluated once on
the test split with `evaluate_test.py`: 106 recordings of 10 subjects,
171,679 segments with a valid ECG reference. Only the PPG quality borders
are applied, as in real use.

| | Coverage | MAE (bpm) | RMSE (bpm) | Bias (bpm) | Within ±5 bpm |
|---|---|---|---|---|---|
| **Prediction (HR or NONE)** | **66.5 %** | **4.80** | 8.85 | −1.81 | 73.1 % |
| Model on all segments (no quality check) | 100 % | 5.98 | 10.02 | −1.95 | 64.4 % |
| Rejected segments only (model output) | 33.5 % | 8.30 | 12.02 | −2.24 | 46.9 % |
| Training selection (ECG + PPG borders) | 31.8 % | 4.11 | 7.31 | −1.50 | 76.5 % |

95 % limits of agreement (Bland-Altman): −18.8 to +15.2 bpm.

| Dataset | Coverage | MAE (bpm) | Within ±5 bpm |
|---|---|---|---|
| wrist-finger-ppg (5 subjects) | 97.0 % | 0.95 | 99.6 % |
| multisite-ppg (4 subjects) | 66.3 % | 4.84 | 72.9 % |
| wrist-ppg-during-exercise (1 subject) | 91.8 % | 8.79 | 41.6 % |

- The quality check works: the segments it rejects have a much larger
  error (8.3 bpm) than the ones it keeps (4.8 bpm).
- With the same selection as in training, the test MAE (4.11 bpm) is close
  to the cross-validation MAE (4.04 bpm), so the model generalises to new
  subjects.
- Below 90 bpm the MAE is about 3–5 bpm. Above 100 bpm the model
  underestimates the heart rate more and more (bias −10 bpm at 100–110 bpm,
  −28 bpm at 120–130 bpm), as already seen in the cross-validation.
- The exercise result comes from one test subject only and is therefore
  uncertain.
- Pipeline check: the quality metrics recomputed by the prediction code
  match the database (max. difference 3·10⁻¹⁴, 100 % agreement of the
  quality decision).

### Model file

The trained model is not stored in GitHub. Download the final model
(trained on all training subjects) from **https://drive.google.com/file/d/1bLnSyhCZEOF6aIwmPDOPnhNPf6Nlr8MG/view?usp=sharing** and place it at
`data/training_out_cnn/dilated_cnn/fold_1/best_model.pt` (the default path
of `predict.py`, `estimate_heartrate.py` and `test_interface.py`), or pass
its path with `--model-path`.

It was trained with:

```bash
uv run python prototyping_dan/train_full.py --db_path data/hr_parts/seg10s_step5s --edf_dir data/harmonized_ecg_ppg_database_filtered --output_dir data/training_full --epochs 15 --lr 1e-3
```

## Test interface

**Entry point** (interface specification of the course), in
`estimate_heartrate.py`:

```python
from estimate_heartrate import estimate_heartrate
heartrate = estimate_heartrate(ppg, fs)
```

`ppg` is the raw PPG signal and `fs` its sampling rate. The result is the
heart rate in bpm at 2 Hz (two values per second) over the whole duration of
the input. Inside, the PPG runs through the same pipeline as the training
data: filtering, 10 s segments every 5 s, quality check, CNN. Segments with
unusable PPG give no estimate; gaps of up to 15 s are interpolated, longer
gaps stay NaN. The model is loaded from
`data/training_out_cnn/dilated_cnn/fold_1/best_model.pt` (or `model_path=` /
the variable `HR_MODEL_PATH`).

To test it on EDF recordings, `test_interface.py` runs the entry point on one
file or a whole folder:

```bash
uv sync
uv run python test_interface.py <file.edf or folder> --model-path <best_model.pt>
```

For each recording it saves the 2 Hz heart rate, the HR per 10 s segment
(with the reason when there is no estimate) and, if the file has an ECG
channel, the comparison with the ECG heart rate (MAE, bias, limits of
agreement, Bland-Altman plot). Results go to `results/test_interface/`, with
an overview in `summary.csv`.
