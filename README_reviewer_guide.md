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

**Final model:** fold 1 (`fold_1/best_model.pt`, the default of
`predict.py`). Since the fold differences come from the validation subjects,
the default fold was kept rather than picking the fold with the lowest error.
Generalisation to new subjects is checked once on the test split with
`evaluate_test.py`.

### Model file

The trained model is not stored in GitHub. Download it from **https://drive.google.com/file/d/1GLl-3rkm2yNHLgGoG_pan9HTYS5JX0Cf/view?usp=sharing** and
place it at `data/training_out_cnn/dilated_cnn/fold_1/best_model.pt`
(the default path of `predict.py` and `evaluate_test.py`), or pass its path
with `--model-path`.

The submitted model was trained on 6 Nov 2026 with the settings of that
day (15 epochs). Since 7 Nov `training.py` defaults to 40 epochs (minimum
30), so retraining gives a slightly different model.

## Test interface

`test_interface.py` runs the model on one EDF file or a whole folder:

```bash
uv sync
uv run python test_interface.py <file.edf or folder> --model-path <best_model.pt>
```

The recordings should be raw EDF files with a PPG channel. For each
recording the script saves the HR per 10 s segment (NONE where the PPG is
not usable) and an HR value per second, where short gaps are interpolated.
If the file also has an ECG channel, the result is compared with the ECG
heart rate (MAE, bias, limits of agreement, Bland-Altman plot). Results go
to `results/test_interface/`, with an overview in `summary.csv`.
