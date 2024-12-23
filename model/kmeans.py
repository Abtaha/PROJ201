from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Load the dataset
df = pd.read_csv("export_data.csv")

# Define the features
features = [
    "Duration",
    "Peak Intensity",
    "Peak Energy Bin",
    "Peak Energy In Bin",
    "Skewness",
    "Kurtosis",
    "Centroid",
    "Rise Time",
    "Decay Time",
    "Mean Time",
    "Std Time",
    "Peak Time",
    "Mean Energy",
    "Std Energy",
    "Total Energy Released",
]


# features = [
#     "Duration",
#     "Peak Intensity",
#     "Peak Energy Bin",
#     "Peak Energy In Bin",
#     "Skewness",
#     "Kurtosis",
#     "Rise Time",
#     "Decay Time",
#     "Centroid",
#     "Total Energy Released",
# ]

# Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])


# PCA-based feature ranking
def pca_feature_ranking(data, feature_names):
    pca = PCA(n_components=len(feature_names))  # Perform PCA
    pca.fit(data)
    # Calculate feature importance as the sum of absolute contributions to all principal components
    feature_importance = np.abs(pca.components_).sum(axis=0)
    print(np.abs(pca.components_))
    print(np.abs(pca.components_).sum(axis=0))
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


# Compute summary statistics for each cluster
cluster_summary = df.groupby("Cluster")[features].agg(["mean", "std"])
print("\nCluster Summary (Mean and Std):")
print(cluster_summary)


# Visualize the distribution of top features
for feature in top_5_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Cluster", y=feature, data=df)
    plt.title(f"{feature} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.show()


# Pairwise visualization of top features with clusters
sns.pairplot(df, hue="Cluster", vars=top_5_features, palette="Set2", diag_kind="kde")
plt.suptitle("Pairwise Plot of Top Features by Cluster", y=1.02)
plt.show()


# Transform centroids back to original feature space
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

# Create a DataFrame for easier interpretation
centroids_df = pd.DataFrame(centroids_original, columns=top_5_features)
centroids_df["Cluster"] = range(optimal_k)
print("\nCluster Centroids in Original Feature Space:")
print(centroids_df)

from scipy.stats import f_oneway

# Perform ANOVA for each top feature
print("\nANOVA Results:")
for feature in top_5_features:
    groups = [df[df["Cluster"] == cluster][feature] for cluster in range(optimal_k)]
    f_stat, p_value = f_oneway(*groups)
    print(f"{feature}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4e}")
