import pandas as pd
import time
import os

# ==============================================================================
# Step 1: Define the file paths
# ==============================================================================
parquet_file_path = "/data/csc59866_f24/tlcdata/your_file.parquet"  # Replace with actual file path
csv_file_path = "converted_file.csv"

# ==============================================================================
# Step 2: Read the Parquet file and measure load time
# ==============================================================================
start_time_parquet = time.time()
df_parquet = pd.read_parquet(parquet_file_path)
end_time_parquet = time.time()

parquet_load_time = (end_time_parquet - start_time_parquet) * 1000  # Convert to milliseconds

# ==============================================================================
# Step 3: Save the DataFrame to a CSV file
# ==============================================================================
df_parquet.to_csv(csv_file_path, index=False)

# ==============================================================================
# Step 4: Read the CSV file and measure load time
# ==============================================================================
start_time_csv = time.time()
df_csv = pd.read_csv(csv_file_path)
end_time_csv = time.time()

csv_load_time = (end_time_csv - start_time_csv) * 1000  # Convert to milliseconds

# ==============================================================================
# Step 5: Report file sizes
# ==============================================================================
parquet_file_size = os.path.getsize(parquet_file_path) / (1024 * 1024)  # Convert to MB
csv_file_size = os.path.getsize(csv_file_path) / (1024 * 1024)  # Convert to MB

# ==============================================================================
# Step 6: Print the results
# ==============================================================================
print(f"Parquet file size: {parquet_file_size:.2f} MB")
print(f"CSV file size: {csv_file_size:.2f} MB")
print(f"Parquet file load time: {parquet_load_time:.2f} ms")
print(f"CSV file load time: {csv_load_time:.2f} ms")
