from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("export_data.csv")
print(df.head())

features = [
    "Duration",
    "Peak Intensity",
    "Skewness",
    "Kurtosis",
    "Rise Time",
    "Decay Time",
    "Centroid",
    "Skewness",
    "Kurtosis",
    "Total Energy Released",
]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

sse = []
k_rng = range(1, 30)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_rng, sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()


threshold = max(sse) * 0.01
optimal_k = -1

for k in k_rng:
    if k == 0:
        continue

    diff = sse[k - 1] - sse[k]
    print(diff, threshold)
    if abs(diff) < threshold:
        optimal_k = k
        break

# optimal_k = 4
print("optimal", optimal_k)

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df["Cluster"] = clusters

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    plt.scatter(
        df_pca[df["Cluster"] == cluster, 0],
        df_pca[df["Cluster"] == cluster, 1],
        label=f"Cluster {cluster}",
    )

# Mark cluster centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    s=200,
    c="black",
    marker="*",
    label="Centroids",
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("Clusters Visualized in 2D (PCA)")
plt.show()

# df = pd.read_csv("export_data.csv")
# print(df.head())
#
#
# # duration | kurtosis
# # duration | peak intensity
# # duration | total energy
# # skewness | kurtosis
# # Skewness | total energy
#
# features = [
#     # ['Duration', 'Peak Time'],
#     # ['Duration', 'Kurtosis',],
#     ["Duration", "Peak Intensity"],
#     ["Duration", "Total Energy Released"],
#     ["Skewness", "Kurtosis"],
#     ["Skewness", "Total Energy Released"],
#     ["Kurtosis", "Total Energy Released"],
#     # ['Skewness', 'Peak Intensity'],
#     # ['Kurtosis', 'Peak Intensity']
# ]
#
# # fig, axes = plt.subplots(
# #    len(features), 1
# # )
#
#
#
# scaler = MinMaxScaler()
#
# for couple in features:
#     x = couple[0]
#     y = couple[1]
#
#     scaler.fit(df[[x]])
#     df[x] = scaler.transform(df[[x]])
#
#     scaler.fit(df[[y]])
#     df[y] = scaler.transform(df[[y]])
#
#     sse = []
#     k_rng = range(1, 13)
#     for k in k_rng:
#         km = KMeans(n_clusters=k)
#         km.fit(df[[x, y]])
#         sse.append(km.inertia_)
#
#     threshold = max(sse) * 0.15
#     optimal_k = -1
#     for i in range(len(k_rng) - 1):
#         diff = sse[i - 1] - sse[i]
#         if abs(diff) < threshold:
#             optimal_k = k_rng[i]
#             break
#
#     print("Optimal k is", optimal_k, "with SSE", sse[i], "and threshold", threshold)
#
#     plt.xlabel("K")
#     plt.ylabel("Sum of squared error")
#     plt.plot(k_rng, sse)
#     plt.show()
#
#     km = KMeans(n_clusters=optimal_k)
#     y_predicted = km.fit_predict(df[[x, y]])
#
#     df["cluster"] = y_predicted
#     df1 = df[df.cluster == 0]
#     df2 = df[df.cluster == 1]
#     df3 = df[df.cluster == 2]
#     df4 = df[df.cluster == 3]
#     df5 = df[df.cluster == 4]
#     df6 = df[df.cluster == 5]
#
#     plt.scatter(df1[x], df1[y], color="green")
#     plt.scatter(df2[x], df2[y], color="red")
#     plt.scatter(df3[x], df3[y], color="blue")
#     plt.scatter(df4[x], df4[y], color="black")
#     plt.scatter(df5[x], df5[y], color="yellow")
#     plt.scatter(df6[x], df6[y], color="cyan")
#     plt.scatter(
#         km.cluster_centers_[:, 0],
#         km.cluster_centers_[:, 1],
#         color="purple",
#         marker="*",
#         label="centroid",
#     )
#     plt.xlabel(x)
#     plt.ylabel(y)
#     plt.legend()
#     plt.show()
#
