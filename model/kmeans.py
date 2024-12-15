from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools

df = pd.read_csv("export_data.csv")
print(df.head())
for col in df.columns:
    print(col)
    print(df[col])

features = [
    "Duration",
    "Peak Intensity",
    #"Peak Energy Bin",
    #"Peak Energy In Bin",
    "Skewness",
    #"Kurtosis",
    "Rise Time",
    #"Decay Time",
    #"Centroid",
    "Total Energy Released",
]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])


sse = []
k_rng = range(1, 15)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    sse.append(km.inertia_)




#threshold = max(sse) * 0.01
optimal_k = -1

for k in k_rng:
    if k == 0:
        continue

    diff = sse[k - 1] - sse[k]
    if abs(diff) < sse[k-1]*0.15:
        optimal_k = k
        break

# optimal_k = 4
print("Optimal k:", optimal_k)

plt.figure(figsize=(10, 6))
plt.plot(k_rng, sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.xticks(range(1, len(k_rng)+1))
plt.axvline(optimal_k, color="red", linestyle='--', label="Optimal k")
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df["Cluster"] = clusters

pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_scaled)

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
        s=50,  # Size of points
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

# pca = PCA(n_components=2)
# df_pca = pca.fit_transform(df_scaled)
#
# plt.figure(figsize=(10, 6))
# for cluster in range(optimal_k):
#     plt.scatter(
#         df_pca[df["Cluster"] == cluster, 0],
#         df_pca[df["Cluster"] == cluster, 1],
#         label=f"Cluster {cluster}",
#     )
#
# # Mark cluster centroids
# centroids_pca = pca.transform(kmeans.cluster_centers_)
# plt.scatter(
#     centroids_pca[:, 0],
#     centroids_pca[:, 1],
#     s=200,
#     c="black",
#     marker="*",
#     label="Centroids",
# )
#
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.legend()
# plt.title("Clusters Visualized in 2D (PCA)")
# plt.show()
