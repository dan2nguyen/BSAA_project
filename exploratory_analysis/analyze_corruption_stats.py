import pandas as pd
import os

def analyze_corruption():
    file_path = r"references\corrupted_sections.parquet"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        print(f"Loading {file_path}...")
        df = pd.read_parquet(file_path)
        
        # Identify boolean columns
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if not bool_cols:
            print("No boolean columns found in the file.")
            return

        print("\n" + "="*40)
        print("INDIVIDUAL COLUMN STATISTICS")
        print("="*40)
        for col in bool_cols:
            true_pct = df[col].mean() * 100
            false_pct = 100 - true_pct
            print(f"Column: {col}")
            print(f"  True:  {true_pct:6.2f}%")
            print(f"  False: {false_pct:6.2f}%")
            print("-" * 20)

        print("\n" + "="*40)
        print("COMBINATION STATISTICS")
        print("="*40)
        # Group by all boolean columns to get combinations
        combinations = df.groupby(bool_cols).size().reset_index(name='count')
        total_rows = len(df)
        
        # Calculate percentage for each combination
        combinations['percentage'] = (combinations['count'] / total_rows) * 100
        
        # Sort by count descending for better readability
        combinations = combinations.sort_values(by='count', ascending=False)
        
        # Rename columns for pretty printing
        print(combinations.to_string(index=False, formatters={'percentage': '{:6.2f}%'.format}))
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_corruption()
