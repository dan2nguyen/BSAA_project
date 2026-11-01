# Model Architecture and Training Configuration

This document describes the updated production-ready architecture of the neural networks used for heart rate (HR) regression from PPG signals and the arguments available to customize the training process.

## 1. Model Architectures

The system implements a modular model factory. All models are designed for regression, outputting a single continuous value representing the Heart Rate in BPM.

### ResidualCNN1D (`cnn1d`)
An upgraded 1D Convolutional Neural Network utilizing residual learning to extract deep spatial-temporal features without losing signal context.
- **Architecture**:
    - **Stem**: Initial Conv1D (32 filters, kernel 7) $\rightarrow$ BatchNorm $\rightarrow$ ReLU.
    - **Residual Blocks**: Three successive blocks consisting of:
        - `Conv1D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv1D` $\rightarrow$ `BatchNorm` $\rightarrow$ **Skip Connection** $\rightarrow$ `ReLU`.
        - The blocks gradually increase filter depth (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256) while reducing temporal resolution via stride-2 convolutions.
    - **Global Pooling**: `AdaptiveAvgPool1d(1)` to flatten the feature map into a global descriptor.
    - **Output**: A Linear layer mapping the 256 features to a single HR scalar.
- **Key Improvement**: Residual connections prevent gradient vanishing and allow the model to learn more complex morphological patterns in the PPG waveform.

### CRNN (`bilstm`)
A Convolutional Recurrent Neural Network (CRNN) that combines local feature extraction with long-range temporal dependency tracking.
- **Architecture**:
    - **CNN Front-end**: Two layers of `Conv1D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)`. This reduces the sequence length and extracts high-level local features.
    - **Recurrent Back-end**: A Bidirectional LSTM (Bi-LSTM) that processes the CNN feature maps in both forward and backward directions.
    - **Configuration**: Default hidden dimension of 64 and 2 LSTM layers.
    - **Output**: The final hidden state of the Bi-LSTM is passed through a Linear layer to produce the HR prediction.
- **Key Improvement**: The CNN front-end reduces the computational load on the LSTM and provides a more robust feature representation of the signal.

---

## 2. Signal Preprocessing Pipeline

To ensure training stability and model generalization, every signal segment undergoes the following pipeline in the `SignalDataset`:
1. **Length Enforcement**: Every segment is forced to a fixed length of **500 samples**.
   - If too long: Cropped to the first 500.
   - If too short: Zero-padded to reach 500.
2. **Z-Score Normalization**: Per-segment normalization is applied: $x_{norm} = \frac{x - \mu}{\sigma + 1e-8}$.
   - This removes amplitude offsets and scaling differences between different patients or sensors.

---

## 3. Training Process and Arguments

The training script (`training.py`) incorporates professional deep learning practices for regression.

### Model & Hyperparameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `cnn1d` | Choice of architecture: `cnn1d` (ResidualCNN) or `bilstm` (CRNN). |
| `--epochs` | `int` | `15` | Maximum number of training epochs. |
| `--batch_size` | `int` | `32` | Number of segments per gradient update. |
| `--lr` | `float` | `1e-3` | Initial learning rate for the Adam optimizer. |

### Data & Paths
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--db_path` | `str` | *Required* | Path to folder containing Parquet metadata files. |
| `--edf_dir` | `str` | *Required* | Root directory where original `.edf` files are stored. |
| `--hr_threshold` | `float` | `100.0` | Threshold for heart rate labels. |

### Validation Strategy
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--n_splits` | `int` | `5` | Folds for Patient-Wise Stratified K-Fold cross-validation. |

### Advanced Implementation Details
- **Loss Function**: Mean Squared Error (`MSELoss`).
- **Optimizer**: Adam.
- **Training Stability**:
    - **Learning Rate Scheduler**: `ReduceLROnPlateau` monitors validation loss and reduces LR by factor 0.5 if no improvement is seen for 2 epochs.
    - **Early Stopping**: Stops training if validation loss does not improve for 5 consecutive epochs.
- **Evaluation Framework**:
    - **Patient-Wise CV**: Uses `StratifiedGroupKFold` to prevent data leakage (all segments of one patient stay in the same fold).
    - **Out-of-Fold (OOF) Tracking**: Predictions for every validation fold are collected and saved to `oof_predictions_{model}.csv`.
    - **Metrics**: The final performance is reported as **Overall MAE**, **Overall RMSE**, and **Accuracy ($\pm 5$ BPM)** across the entire OOF dataset.
- **Outputs**: Each fold's best model and learning curves are saved in `training_outputs/{model}/fold_{i}/`.
