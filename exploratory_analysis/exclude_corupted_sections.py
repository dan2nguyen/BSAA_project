import os

import pandas as pd


def generate_exclusion_file():
    input_path = r"references\evaluation_results.parquet"
    output_path = r"references\corrupted_sections.parquet"

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    try:
        print(f"Loading {input_path}...")
        df = pd.read_parquet(input_path)

        # Define columns for flatness check
        ecg_flags = [
            "ecg_std_flat",
            "ecg_zero_dev_flat",
            "ecg_dynamic_range_flat",
        ]
        ppg_flags = [
            "ppg_std_flat",
            "ppg_zero_dev_flat",
            "ppg_dynamic_range_flat",
        ]

        # Verify columns exist
        missing_cols = [
            col
            for col in ecg_flags + ppg_flags
            if col not in df.columns
        ]
        if missing_cols:
            print(
                f"Error: Missing columns in Parquet file: {missing_cols}"
            )
            return

        # 1. Calculate notFlat
        # A signal is flat if at least 2 of its 3 flags are True
        ecg_is_flat = df[ecg_flags].sum(axis=1) >= 2
        ppg_is_flat = df[ppg_flags].sum(axis=1) >= 2

        # Section is corrupted if either ECG or PPG is flat
        is_corrupted = ecg_is_flat | ppg_is_flat
        df["notFlat"] = ~is_corrupted

        # 2. Calculate no_abnormal_waves
        # True if ALL the following are False
        abnormal_cols = [
            "ppg_mean_absolute_derivative_spike",
            "ppg_LF_ratio_motion",
            "ppg_InBand_power_ratio",
            "ppg_kurtosis_out",
            "ppg_Skewness_out",
            "ecg_mean_absolute_derivative_spike",
            "ecg_LF_ratio_motion",
            "ecg_InBand_power_ratio",
            "ecg_kurtosis_out",
            "ecg_Skewness_out",
        ]

        # Filter to columns that actually exist in df to avoid KeyError
        existing_abnormal_cols = [
            col for col in abnormal_cols if col in df.columns
        ]

        if existing_abnormal_cols:
            # sum(axis=1) == 0 means all boolean values in those columns are False
            df["no_abnormal_waves"] = (
                df[existing_abnormal_cols].sum(axis=1) == 0
            )
        else:
            print(
                "Warning: None of the abnormal wave columns were found. Setting no_abnormal_waves to True."
            )
            df["no_abnormal_waves"] = True

        # Keep identifiers for usability
        meta_cols = [
            "base_folder",
            "edf_filename",
            "section_id",
            "start",
            "end",
        ]
        meta_cols = [c for c in meta_cols if c in df.columns]

        result_df = df[meta_cols + ["notFlat", "no_abnormal_waves"]]

        # Save as Parquet (faster than CSV)
        result_df.to_parquet(output_path, index=False)
        print(f"Successfully saved exclusion file to {output_path}")

        # Reporting
        total = len(df)
        corrupted_count = is_corrupted.sum()
        abnormal_count = (~df["no_abnormal_waves"]).sum()
        print(f"Total sections: {total}")
        print(
            f"Corrupted sections (flat): {corrupted_count} ({corrupted_count / total:.2%})"
        )
        print(
            f"Sections with abnormal waves: {abnormal_count} ({abnormal_count / total:.2%})"
        )
        print(
            f"Valid sections (notFlat=True): {total - corrupted_count}"
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_exclusion_file()
