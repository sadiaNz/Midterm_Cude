import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from math import radians

# ---------------- Step 1: Fetch and Load Data ----------------
print("\nStep 1: Fetching data from URL and loading into Pandas DataFrame...\n")
url = "https://gist.githubusercontent.com/jlewis8756/6b83a54351e91012b9fd541356a347c9/raw/54dbb430227d4d44a7262e51194ff702ec0fa410/worldcities.csv"

try:
    df = pd.read_csv(url)
    print("Data fetched and loaded successfully!\n")
    print("Raw Data Sample:\n", df.head(), "\n")
except Exception as e:
    print(f"Error fetching data: {e}")
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
    # Convert lat and lng to radians for haversine_distances
    coordinates = np.radians(filtered_df[['lat', 'lng']].values)
    distance_matrix = haversine_distances(coordinates) * 6371  # Multiply by Earth radius (6371 km)
    print("Pairwise Haversine Distance Matrix Computed (in km).\n")
else:
    print("Error: 'lat' or 'lng' column not found in dataset.")
    exit()

# ---------------- Step 4: Apply Clustering ----------------
print("\nStep 4: Applying clustering algorithm...\n")

# Choose either DBSCAN or AgglomerativeClustering
# Uncomment one of the following options:

# Option 1: DBSCAN
clustering = DBSCAN(eps=500, min_samples=3, metric='precomputed')  # Using 500 km radius for eps
clusters = clustering.fit_predict(distance_matrix)

# Option 2: AgglomerativeClustering
# clustering = AgglomerativeClustering(n_clusters=10, linkage='average', affinity='precomputed')
# clusters = clustering.fit_predict(distance_matrix)

filtered_df['cluster'] = clusters
print("Clustering completed!")
print(f"Number of unique clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}\n")
print("Sample of clustered data:\n", filtered_df.head(), "\n")

# ---------------- Step 5: Visualize Results ----------------
print("\nStep 5: Visualizing results...\n")

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in set(clusters):
    cluster_points = filtered_df[filtered_df['cluster'] == cluster]
    plt.scatter(cluster_points['lng'], cluster_points['lat'], label=f"Cluster {cluster}" if cluster != -1 else "Noise")

plt.title("Clusters of Cities with Population > 1 Million")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()
