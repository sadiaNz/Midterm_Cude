import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time


# Project 2 formula (Pure Python) ---------
def haversine_distance_python(lat1, lon1, lat2, lon2):
    """
    Compute the haversine distance between two points in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def pairwise_haversine_python(coords):
    """
    Compute pairwise haversine distances for a set of coordinates.
    """
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = haversine_distance_python(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return distances


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

# ---------------- Step 3: Compute Pairwise Haversine Distances (Sklearn) ----------------
print("\nStep 3: Computing pairwise haversine distances with sklearn...\n")
if 'lat' in filtered_df.columns and 'lng' in filtered_df.columns:
    coordinates = np.radians(filtered_df[['lat', 'lng']].values)

    # Measure runtime for sklearn
    start = time.time()
    sklearn_distance_matrix = haversine_distances(coordinates) * 6371  # Earth radius in km
    sklearn_time = time.time() - start
    print(f"Runtime (sklearn): {sklearn_time:.4f} seconds\n")
else:
    print("Error: 'lat' or 'lng' column not found in dataset.")
    exit()

# ---------------- Compute Pairwise Haversine Distances (Pure Python) ----------------
print("\nStep 4: Computing pairwise haversine distances with Pure Python...\n")
start = time.time()
python_distance_matrix = pairwise_haversine_python(filtered_df[['lat', 'lng']].values)
python_time = time.time() - start
print(f"Runtime (Pure Python): {python_time:.4f} seconds\n")

# ---------------- Step 5: Apply Clustering ----------------
print("\nStep 5: Applying DBSCAN clustering algorithm...\n")
clustering = DBSCAN(eps=500, min_samples=3, metric='precomputed')  # 500 km radius
clusters = clustering.fit_predict(sklearn_distance_matrix)  # Use sklearn matrix for clustering
filtered_df['cluster'] = clusters

print("Clustering completed!")
print(f"Number of unique clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}\n")
print("Sample of clustered data:\n", filtered_df.head(), "\n")

# ---------------- Step 6: Visualize Results ----------------
print("\nStep 6: Visualizing results...\n")
plt.figure(figsize=(12, 8))
for cluster in set(clusters):
    cluster_points = filtered_df[filtered_df['cluster'] == cluster]
    plt.scatter(cluster_points['lng'], cluster_points['lat'], label=f"Cluster {cluster}" if cluster != -1 else "Noise")

plt.title("Clusters of Cities with Population > 1 Million")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

# Save the plot to a file
visualization_output = "clusters_visualization.png"
plt.savefig(visualization_output)
print(f"Visualization saved to '{visualization_output}'\n")

plt.show(block=True)

# ---------------- Step 7: Save Clustered Data ----------------
output_file = "clustered_cities.csv"
print(f"\nStep 7: Saving clustered data to '{output_file}'...\n")
try:
    filtered_df.to_csv(output_file, index=False)
    print(f"Clustered data saved successfully to '{output_file}'!")
except Exception as e:
    print(f"Error saving clustered data: {e}")
    exit()

# ---------------- Step 8: Compare Runtimes ----------------
print("\nStep 8: Comparing runtimes...\n")
print(f"Runtime for sklearn haversine distances: {sklearn_time:.4f} seconds")
print(f"Runtime for Pure Python haversine distances: {python_time:.4f} seconds\n")

# ---------------- Step 9: Analyze Outliers ----------------
print("\nStep 9: Analyzing outliers (noise points)...\n")
noise_points = filtered_df[filtered_df['cluster'] == -1]
print(f"Number of outlier cities (noise points): {len(noise_points)}\n")
if len(noise_points) > 0:
    print("Outlier Cities:\n", noise_points[['city', 'lat', 'lng', 'population']], "\n")
else:
    print("No noise points detected.\n")
