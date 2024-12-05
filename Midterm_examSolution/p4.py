import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import matplotlib.pyplot as plt

#Step 1: import the “raw” CSV file from https://gist.github.com/jlewis8756/6b83a54351e91012b9fd541356a347c9 and read it as a Pandas dataframe. Filter it by keeping only cities that have more than 1 million population (you should get less than 500 rows).
# Correct raw file URL for reading the CSV
try:
  url = "https://gist.githubusercontent.com/jlewis8756/6b83a54351e91012b9fd541356a347c9/raw/world_cities.csv"
except:
  url = 'world_cities.csv'


# Read the CSV filse into a DataFrame
df = pd.read_csv(url)

filtered_cities = df[df['population'] > 1000000]
print('Result of cities that have more than 1 million population')
print(filtered_cities)
# Step 2:  using the “lat” (for latitude) and “lng” (for longitude) columns for haversine_distances API and compute pairwise haversine distances among these big cities.
lat_lng = np.radians(filtered_cities[['lat', 'lng']].values)
earth_radius = 6371000/1000
distances = haversine_distances(lat_lng) * earth_radius  # Convert radians to kilometres
print('\n\\n pairwise haversine distances among these big cities')
print(distances)
# Step 3: apply either DBSCAN or AgglomerativeClustering, to cluster the big cities into clusters using the parameters of your choice. Note that the two important parameters for AgglomerativeClustering are n_clusters (e.g. 10) and linkage( e.g., "average"), and  those for DBSCAN are eps (e.g., 10km) and min_samples (e.g., 5). Plot your results for visual examination using any visualization package (e.g., Matplotlib).
# using DBSCAN
dbscan = DBSCAN(eps=250, min_samples=5, metric="precomputed")  # eps in kilometers
labels_dbscan = dbscan.fit_predict(distances)

#Visualization of clusters
cities_data = filtered_cities.copy() # because of warning A value is trying to be set on a copy of a slice from a DataFrame.

cities_data['Cluster_DBSCAN'] = labels_dbscan


# Plot using DBSCAN labels
plt.figure(figsize=(10, 6))
plt.scatter(cities_data['lng'], cities_data['lat'], c=labels_dbscan, cmap="tab10", s=50, edgecolor='k')
plt.title("City clusters with pairwise distance", fontsize=16)
plt.xlabel("Longitude", fontsize=11)
plt.ylabel("Latitude", fontsize=11)
plt.colorbar(label="Cluster")
plt.show()
plt.savefig('fig.png')

