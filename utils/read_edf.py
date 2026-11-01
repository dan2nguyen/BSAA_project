import warnings
from functools import lru_cache

import mne

# Cache for MNE Raw objects to avoid repeated header parsing and disk seek overhead
_RAW_CACHE = {}

def get_raw_edf(file_path: str):
    """
    Retrieves a Raw EDF object from cache or loads it with preload=False.
    """
    if file_path not in _RAW_CACHE:
        try:
            # preload=False keeps the data on disk, loading only metadata into RAM
            _RAW_CACHE[file_path] = mne.io.read_raw_edf(
                file_path, preload=False, verbose=False
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Error reading EDF file {file_path}: {e}")
            return None
    return _RAW_CACHE[file_path]

def read_edf_signals(
    file_path: str,
    ecg_keywords: list | None = None,
    ppg_keywords: list | None = None,
    start_sample: int | None = None,
    end_sample: int | None = None,
):
    """
    Reads an EDF file and extracts ECG and PPG records.
    Supports lazy loading of specific segments.

    Args:
        file_path (str): Absolute path to the .edf file.
        ecg_keywords (list, optional): List of strings to identify ECG channels.
        ppg_keywords (list, optional): List of strings to identify PPG channels.
        start_sample (int, optional): Start index for the segment.
        end_sample (int, optional): End index for the segment.

    Returns:
        tuple: (ecg_data, ppg_data, metadata)
            - ecg_data: numpy array or None
            - ppg_data: numpy array or None
            - metadata: Dictionary containing sampling rate, channel names, and other EDF header info.
    """
    # Default keywords for signal detection
    if ecg_keywords is None:
        ecg_keywords = ["ECG"]
    if ppg_keywords is None:
        ppg_keywords = ["PPG"]

    raw = get_raw_edf(file_path)
    if raw is None:
        return None, None, None

    ch_names = raw.ch_names

    def find_channel(keywords):
        matches = [
            name
            for name in ch_names
            if any(kw.lower() in name.lower() for kw in keywords)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            warnings.warn(
                f"Multiple channels matched keywords {keywords}: {matches}. Using the first one: {matches[0]}"
            )
        return matches[0]

    ecg_channel = find_channel(ecg_keywords)
    ppg_channel = find_channel(ppg_keywords)

    # Extract data
    ecg_data = None
    if ecg_channel:
        # get_data returns (n_channels, n_times)
        # If start/end are provided, only that slice is read from disk
        ecg_data = raw.get_data(
            picks=[ecg_channel], start=start_sample, stop=end_sample
        )[0]

    ppg_data = None
    if ppg_channel:
        ppg_data = raw.get_data(
            picks=[ppg_channel], start=start_sample, stop=end_sample
        )[0]

    # Extract metadata
    metadata = {
        "sampling_rate": raw.info["sfreq"],
        "duration": raw.n_times / raw.info["sfreq"],
        "channel_mapping": {"ecg": ecg_channel, "ppg": ppg_channel},
        "all_channels": ch_names,
        "n_times": raw.n_times,
    }

    return ecg_data, ppg_data, metadata
