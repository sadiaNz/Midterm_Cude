import pandas as pd
import os
import time

# File paths
parquet_file = '/data/csc59866_f24/tlcdata/yellow_tripdata_2009-01.parquet'  # Use the actual file name
csv_file = 'yellow_tripdata_2009-01.csv'  # Save the CSV in the current directory

# Step 1: Read the Parquet file and measure runtime
start_time_parquet = time.time()
df_parquet = pd.read_parquet(parquet_file)
end_time_parquet = time.time()

# Save the Parquet runtime
parquet_runtime = (end_time_parquet - start_time_parquet) * 1000  # Convert to milliseconds

# Step 2: Save the DataFrame to a CSV file
df_parquet.to_csv(csv_file, index=False)

# Step 3: Measure the file sizes
parquet_file_size = os.path.getsize(parquet_file)  # Size in bytes
csv_file_size = os.path.getsize(csv_file)  # Size in bytes

# Step 4: Read the CSV file and measure runtime
start_time_csv = time.time()
df_csv = pd.read_csv(csv_file)
end_time_csv = time.time()

# Save the CSV runtime
csv_runtime = (end_time_csv - start_time_csv) * 1000  # Convert to milliseconds

# Step 5: Report results
print("File Sizes:")
print(f"Parquet file size: {parquet_file_size / (1024 * 1024):.2f} MB")
print(f"CSV file size: {csv_file_size / (1024 * 1024):.2f} MB")

print("\nRuntimes:")
print(f"Time to load Parquet file: {parquet_runtime:.2f} ms")
print(f"Time to load CSV file: {csv_runtime:.2f} ms")
