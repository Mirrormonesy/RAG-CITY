"""Quick viewer for parquet files. Usage: python scripts/view_parquet.py <path> [n]"""
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python scripts/view_parquet.py <path> [n_rows]")
    sys.exit(1)

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

df = pd.read_parquet(path)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

print(f"File: {path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\n--- First {n} rows ---")
print(df.head(n))
print(f"\n--- dtypes ---")
print(df.dtypes)
