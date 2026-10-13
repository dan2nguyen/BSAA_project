import pandas as pd
import pyarrow.parquet as pq

# Force pandas to display every column without truncating
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def read_parquet_slice(
    file_path: str, start: int = 0, stop: int | None = None
):
    if stop is None:
        stop = start + 10

    table = pq.read_table(file_path)
    sliced_table = table.slice(
        offset=start, length=max(0, stop - start)
    )
    return sliced_table.to_pandas()


# Examples:
# df = read_parquet_slice("data.parquet", start=20)          # 10 rows from index 20 to 30
# df = read_parquet_slice("data.parquet", start=50, stop=75)  # Index 50 to 75
# print(df)

if __name__ == "__main__":
    # Example usage
    df = read_parquet_slice(
        r"databasing\references\evaluation_results_win2000.parquet",
        start=0,
    )  # 10 rows from index 20 to 30
    print(df)
