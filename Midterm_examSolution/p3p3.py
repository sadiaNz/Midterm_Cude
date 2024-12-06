import pandas as pd
import os
import time

# File paths
parquet_file = "/data/csc59866_f24/tlcdata/yellow_tripdata_2009-01.parquet"
csv_file = "/home/nawa24/Midterm_Cude/yellow_tripdata_2009-01.csv"  # Writable directory

# ---------------- Read Parquet and Save as CSV ----------------
print("Step 1: Reading Parquet file...")
start_time_parquet = time.time()  # Start time for Parquet
parquet_df = pd.read_parquet(parquet_file, engine='pyarrow')
end_time_parquet = time.time()    # End time for Parquet

print("Step 2: Saving DataFrame to CSV file...")
parquet_df.to_csv(csv_file, index=False)

# ---------------- Read CSV into DataFrame ----------------
print("Step 3: Reading CSV file...")
start_time_csv = time.time()  # Start time for CSV
csv_df = pd.read_csv(csv_file)
end_time_csv = time.time()    # End time for CSV

# ---------------- Calculate File Sizes ----------------
parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)  # Size in MB
csv_size = os.path.getsize(csv_file) / (1024 * 1024)          # Size in MB

# ---------------- Report Results ----------------
print("\n--------------- Results ----------------\n")
print(f"Parquet file size: {parquet_size:.2f} MB")
print(f"CSV file size: {csv_size:.2f} MB")
print(f"Time to load Parquet: {(end_time_parquet - start_time_parquet) * 1000:.2f} ms")
print(f"Time to load CSV: {(end_time_csv - start_time_csv) * 1000:.2f} ms")
