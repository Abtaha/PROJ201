import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and prepare data
df = pd.read_csv("export_data.csv", delimiter=',')#, header=0)
features = [
    "Duration",
    "Peak Energy Bin",
    "Peak Energy In Bin",
    "Skewness",
    "Kurtosis",
    "Centroid",
    "Rise Time",
    "Decay Time",
    "Total Energy Released",
]
X = df[features]
clusterdf = df[['Event ID']]
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
clusterdf["KMeans Cluster"] = kmeans_labels

# DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(X_scaled)
clusterdf["DBScan Cluster"] = dbscan_labels

clusterdf.to_csv("clusters.csv", index=False)

# Hierarchical clustering
linkage_matrix = linkage(X_scaled, method="ward")

# Plot results
plt.figure(figsize=(20, 10))

# PCA variance explained
plt.subplot(231)
explained_variance_ratio = pca.explained_variance_ratio_
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio)
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

# First two PCs with K-means clusters
plt.subplot(232)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis")
plt.title("K-means Clustering (PCA)")
plt.xlabel("First PC")
plt.ylabel("Second PC")

# First two PCs with DBSCAN clusters
plt.subplot(233)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="viridis")
plt.title("DBSCAN Clustering (PCA)")
plt.xlabel("First PC")
plt.ylabel("Second PC")

# Dendrogram
plt.subplot(234)
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")

# Feature importance (PCA components)
plt.subplot(235)
component_df = pd.DataFrame(
    pca.components_[:2].T, columns=["PC1", "PC2"], index=features
)
component_df.plot(kind="bar")
plt.title("Feature Importance in First Two PCs")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

# Print summary statistics
print("\nCluster Sizes:")
print("K-means clusters:", pd.Series(kmeans_labels).value_counts().to_dict())
print("DBSCAN clusters:", pd.Series(dbscan_labels).value_counts().to_dict())

# Calculate and print silhouette scores
from sklearn.metrics import silhouette_score

print("\nSilhouette Scores:")
print("K-means:", silhouette_score(X_scaled, kmeans_labels))
print("DBSCAN:", silhouette_score(X_scaled, dbscan_labels))
