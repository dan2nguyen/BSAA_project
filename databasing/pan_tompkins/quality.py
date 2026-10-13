import numpy as np

# Thresholds are scale independent on purpose: amplitudes differ by many
# orders of magnitude between files, so no absolute values are used.
MIN_UNIQUE_RATIO = 0.1
MAX_ZERO_DERIVATIVE_FRACTION = 0.5


def compute_quality_metrics(ecg_segment: np.ndarray) -> dict:
    """
    Computes scale-independent quality metrics of an ECG segment.

    Args:
        ecg_segment (np.ndarray): 1D ECG segment.

    Returns:
        dict: unique_ratio (number of unique values / number of samples)
            and zero_derivative_fraction (share of consecutive samples
            with identical values).
    """
    x = np.asarray(ecg_segment, dtype=float)
    if len(x) < 2:
        return {"unique_ratio": 0.0, "zero_derivative_fraction": 1.0}
    return {
        "unique_ratio": len(np.unique(x)) / len(x),
        "zero_derivative_fraction": float(np.mean(np.diff(x) == 0)),
    }


def check_segment_quality(
    ecg_segment: np.ndarray,
    min_unique_ratio: float = MIN_UNIQUE_RATIO,
    max_zero_derivative_fraction: float = MAX_ZERO_DERIVATIVE_FRACTION,
) -> tuple[bool, str]:
    """
    Decides whether an ECG segment is usable for HR estimation.

    Rejects dead/flat stretches, which after filtering look like a few
    repeating quantized levels (unique_ratio ~0.002, zero derivative ~1.0,
    compared to >0.4 and <0.15 for real ECG).

    Args:
        ecg_segment (np.ndarray): 1D ECG segment.
        min_unique_ratio (float): Minimum accepted unique_ratio.
        max_zero_derivative_fraction (float): Maximum accepted share of
            zero-derivative samples.

    Returns:
        tuple: (valid, reason), reason is an empty string for valid segments.
    """
    if not np.all(np.isfinite(ecg_segment)):
        return False, "non_finite"
    metrics = compute_quality_metrics(ecg_segment)
    if metrics["zero_derivative_fraction"] > max_zero_derivative_fraction:
        return False, "flat_zero_derivative"
    if metrics["unique_ratio"] < min_unique_ratio:
        return False, "flat_few_unique_values"
    return True, ""
