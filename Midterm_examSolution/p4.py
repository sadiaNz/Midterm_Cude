import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ---------------- Step 1: Load Data from Local File ----------------
print("\nStep 1: Loading data from local file...\n")
file_path = "Midterm_examSolution/world_cities.csv"  # Local file path

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully!\n")
    print("Raw Data Sample:\n", df.head(), "\n")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# ---------------- Step 2: Filter Cities by Population ----------------
print("\nStep 2: Filtering cities with population > 1 million...\n")
if 'population' in df.columns:
    filtered_df = df[df['population'] > 1_000_000].reset_index(drop=True)
    print(f"Number of cities with population > 1 million: {len(filtered_df)}\n")
    print("Filtered Data Sample:\n", filtered_df.head(), "\n")
else:
    print("Error: 'population' column not found in dataset.")
    exit()

# ---------------- Step 3: Compute Pairwise Haversine Distances ----------------
print("\nStep 3: Computing pairwise haversine distances...\n")
if 'lat' in filtered_df.columns and 'lng' in filtered_df.columns:
    coordinates = np.radians(filtered_df[['lat', 'lng']].values)
    distance_matrix = haversine_distances(coordinates) * 6371  # Earth radius in km
    print("Pairwise Haversine Distance Matrix Computed (in km).\n")
else:
    print("Error: 'lat' or 'lng' column not found in dataset.")
    exit()

# ---------------- Step 4: Apply Clustering ----------------
print("\nStep 4: Applying DBSCAN clustering algorithm...\n")
clustering = DBSCAN(eps=500, min_samples=3, metric='precomputed')  # 500 km radius
clusters = clustering.fit_predict(distance_matrix)
filtered_df['cluster'] = clusters

print("Clustering completed!")
print(f"Number of unique clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}\n")
print("Sample of clustered data:\n", filtered_df.head(), "\n")

# ---------------- Step 5: Visualize Results ----------------
print("\nStep 5: Visualizing results...\n")
plt.figure(figsize=(12, 8))
for cluster in set(clusters):
    cluster_points = filtered_df[filtered_df['cluster'] == cluster]
    plt.scatter(cluster_points['lng'], cluster_points['lat'], label=f"Cluster {cluster}" if cluster != -1 else "Noise")

plt.title("Clusters of Cities with Population > 1 Million")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

# Explicitly show the plot
plt.show(block=True)

# ---------------- Step 6: Save Clustered Data ----------------
output_file = "clustered_cities.csv"
print(f"\nStep 6: Saving clustered data to '{output_file}'...\n")
try:
    filtered_df.to_csv(output_file, index=False)
    print(f"Clustered data saved successfully to '{output_file}'!")
except Exception as e:
    print(f"Error saving clustered data: {e}")
    exit()

# ---------------- Step 7: Analyze Outliers ----------------
print("\nStep 7: Analyzing outliers (noise points)...\n")
noise_points = filtered_df[filtered_df['cluster'] == -1]
print(f"Number of outlier cities (noise points): {len(noise_points)}\n")
if len(noise_points) > 0:
    print("Outlier Cities:\n", noise_points[['city', 'lat', 'lng', 'population']], "\n")
else:
    print("No noise points detected.\n")
