from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Load the dataset
df = pd.read_csv("export_data.csv")

# # Define the features
# features = [
#     "Duration",
#     "Peak Intensity",
#     "Peak Energy Bin",
#     "Peak Energy In Bin",
#     "Skewness",
#     "Kurtosis",
#     "Centroid",
#     "Rise Time",
#     "Decay Time",
#     "Mean Time",
#     "Std Time",
#     "Peak Time",
#     "Mean Energy",
#     "Std Energy",
#     "Total Energy Released",
# ]


features = [
    "Duration",
    "Peak Intensity",
    "Peak Energy Bin",
    "Peak Energy In Bin",
    "Skewness",
    "Kurtosis",
    "Rise Time",
    "Decay Time",
    "Centroid",
    "Total Energy Released",
]

# Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])


# PCA-based feature ranking
def pca_feature_ranking(data, feature_names):
    pca = PCA(n_components=len(feature_names))  # Perform PCA
    pca.fit(data)
    # Calculate feature importance as the sum of absolute contributions to all principal components
    feature_importance = np.abs(pca.components_).sum(axis=0)
    feature_ranking = sorted(
        zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
    )
    return feature_ranking


# Get ranked features
feature_ranking = pca_feature_ranking(df_scaled, features)
print("\nFeature Ranking based on PCA contributions:")
for feature, importance in feature_ranking:
    print(f"{feature}: {importance:.4f}")

# Select the top 5 features
top_5_features = [feature for feature, _ in feature_ranking[:5]]
print("\nTop 5 Features Selected:")
print(top_5_features)

# Clustering with top 5 features
df_top_features = scaler.fit_transform(df[top_5_features])

# Elbow Method to find the optimal k
sse = []
k_rng = range(1, 15)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_top_features)
    sse.append(km.inertia_)

kneedle = KneeLocator(k_rng, sse, curve="convex", direction="decreasing")
optimal_k = kneedle.knee

if not optimal_k:
    optimal_k = 4

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_rng, sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.xticks(range(1, len(k_rng) + 1))
plt.axvline(optimal_k, color="red", linestyle="--", label="Optimal k")
plt.show()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_top_features)

# Add cluster labels to the DataFrame
df["Cluster"] = clusters

# Visualize clusters in 3D using PCA
pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_top_features)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot each cluster in a different color
for cluster in range(optimal_k):
    ax.scatter(
        df_pca[df["Cluster"] == cluster, 0],
        df_pca[df["Cluster"] == cluster, 1],
        df_pca[df["Cluster"] == cluster, 2],
        label=f"Cluster {cluster}",
    )

# Mark the centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    centroids_pca[:, 2],
    s=200,
    c="black",
    marker="*",
    label="Centroids",
)

# Labels and legend
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.legend()
ax.set_title("3D Clusters Visualized with PCA")
plt.show()
