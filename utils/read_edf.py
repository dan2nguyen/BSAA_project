import warnings

import mne


def read_edf_signals(
    file_path: str,
    ecg_keywords: list | None = None,
    ppg_keywords: list | None = None,
):
    """
    Reads an EDF file and extracts ECG and PPG records.

    Args:
        file_path (str): Absolute path to the .edf file.
        ecg_keywords (list, optional): List of strings to identify ECG channels.
        ppg_keywords (list, optional): List of strings to identify PPG channels.

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

    try:
        # Load EDF file. preload=True loads data into memory for easier access.
        # verbose=False suppresses MNE's detailed logging
        raw = mne.io.read_raw_edf(
            file_path, preload=True, verbose=False
        )
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error reading EDF file {file_path}: {e}")
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
        # get_data returns (n_channels, n_times), we take [0] to get a 1D array
        ecg_data = raw.get_data(picks=[ecg_channel])[0]

    ppg_data = None
    if ppg_channel:
        ppg_data = raw.get_data(picks=[ppg_channel])[0]

    # Extract metadata
    metadata = {
        "sampling_rate": raw.info["sfreq"],
        "duration": raw.n_times / raw.info["sfreq"],
        "channel_mapping": {"ecg": ecg_channel, "ppg": ppg_channel},
        "all_channels": ch_names,
        "n_times": raw.n_times,
    }

    return ecg_data, ppg_data, metadata
