from sklearn.decomposition import pca
from sklearn.preprocessing import minmaxscaler
from sklearn.cluster import kmeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import kneelocator
import itertools
import json

# load the dataset
df = pd.read_csv("export_data.csv")

# define the features
features = [
    "duration",
    "peak intensity",
    "peak energy bin",
    "peak energy in bin",
    "skewness",
    "kurtosis",
    "centroid",
    "rise time",
    "decay time",
    "mean time",
    "std time",
    "peak time",
    "mean energy",
    "std energy",
    "total energy released",
]


# features = [
#     "duration",
#     "peak intensity",
#     "peak energy bin",
#     "peak energy in bin",
#     "skewness",
#     "kurtosis",
#     "rise time",
#     "decay time",
#     "centroid",
#     "total energy released",
# ]

# scale the data
scaler = minmaxscaler()
df_scaled = scaler.fit_transform(df[features])
# k means k range
k_rng = 15


# elbow method to find the optimal k
def optimal(scaler, flist, k_rng):
    sse = []
    k_rng = range(1, k_rng)
    df_scaled_narrower = scaler.fit_transform(df[flist])

    for k in k_rng:
        km = kmeans(n_clusters=k, random_state=42)
        km.fit(df_scaled_narrower)
        sse.append(km.inertia_)

    kneedle = kneelocator(k_rng, sse, curve="convex", direction="decreasing")
    optimal_k = int(kneedle.knee)
    if not optimal_k:
        optimal_k = 4

    return sse, optimal_k


def generate_combinations(flist: list, scaler: object, combinations: int) -> dict:
    combinations = list(itertools.combinations(flist, combinations))
    combinations = {x: none for x in combinations}
    print(f"total combinations: {len(combinations)}")

    for i, comb in enumerate(combinations.keys()):
        sse, optimal_k = optimal(scaler, list(comb), k_rng)
        combinations[comb] = optimal_k

    combinations = {"-".join(key): val for key, val in combinations.items()}
    sorted_combs = dict(sorted(combinations.items(), key=lambda item: item[1]))
    print(json.dumps(sorted_combs, indent=4))

    return sorted_combs


def select_distinct_features(features: list, scaler: object, combinations: int) -> list:
    sorted = generate_combinations(features, scaler, 4)
    # select the last one, optional selection no rational thinking here.
    most_features = list(sorted)[-1]
    most_k_val = sorted[most_features]
    optimal_k = most_k_val
    flist = most_features.split("-")
    print("\none of the feature combination that has the most distinct clustering: ")
    for feature in flist:
        print(feature)
    return flist


# pca-based feature ranking
def pca_feature_ranking(data, feature_names):
    pca = pca(n_components=len(feature_names))  # perform pca
    pca.fit(data)
    # calculate feature importance as the sum of absolute contributions to all principal components
    feature_importance = np.abs(pca.components_).sum(axis=0)
    print(np.abs(pca.components_))
    print(np.abs(pca.components_).sum(axis=0))
    feature_ranking = sorted(
        zip(feature_names, feature_importance), key=lambda x: x[1], reverse=true
    )
    return feature_ranking


# get ranked features
feature_ranking = pca_feature_ranking(df_scaled, features)
print("\nfeature ranking based on pca contributions:")
for feature, importance in feature_ranking:
    print(f"{feature}: {importance:.4f}")

# select the top 5 features
flist = [feature for feature, _ in feature_ranking[:5]]
print("\ntop 5 features selected:")
print(flist)

# an alternative approach for feature selection
# flist = select_distinct_features(features, scaler, combinations=5)
# df_top_features = scaler.fit_transform(df[flist])

df_top_features = scaler.fit_transform(df[flist])

# elbow method application
sse, optimal_k = optimal(scaler, flist, k_rng)

# plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, k_rng), sse, marker="o")
plt.xlabel("number of clusters (k)")
plt.ylabel("sum of squared errors (sse)")
plt.title("elbow method for optimal k")
plt.xticks(range(1, k_rng))
plt.axvline(optimal_k, color="red", linestyle="--", label="optimal k")
plt.show()

# perform kmeans clustering
kmeans = kmeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_top_features)

# add cluster labels to the dataframe
df["cluster"] = clusters

# visualize clusters in 3d using pca
pca = pca(n_components=3)
df_pca = pca.fit_transform(df_top_features)

# 3d plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# plot each cluster in a different color
for cluster in range(optimal_k):
    ax.scatter(
        df_pca[df["cluster"] == cluster, 0],
        df_pca[df["cluster"] == cluster, 1],
        df_pca[df["cluster"] == cluster, 2],
        label=f"cluster {cluster}",
    )

# mark the centroids in pca space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    centroids_pca[:, 2],
    s=200,
    c="black",
    marker="*",
    label="centroids",
)

# labels and legend
ax.set_xlabel("pca component 1")
ax.set_ylabel("pca component 2")
ax.set_zlabel("pca component 3")
ax.legend()
ax.set_title("3d clusters visualized with pca")
plt.show()


# compute summary statistics for each cluster
cluster_summary = df.groupby("cluster")[features].agg(["mean", "std"])
print("\ncluster summary (mean and std):")
print(cluster_summary)


# visualize the distribution of top features
for feature in flist:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="cluster", y=feature, data=df)
    plt.title(f"{feature} by cluster")
    plt.xlabel("cluster")
    plt.ylabel(feature)
    plt.show()


# pairwise visualization of top features with clusters
sns.pairplot(df, hue="cluster", vars=flist, palette="set2", diag_kind="kde")
plt.suptitle("pairwise plot of top features by cluster", y=1.02)
plt.show()


# transform centroids back to original feature space
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

# create a dataframe for easier interpretation
centroids_df = pd.dataframe(centroids_original, columns=flist)
centroids_df["cluster"] = range(optimal_k)
print("\ncluster centroids in original feature space:")
print(centroids_df)

from scipy.stats import f_oneway

# perform anova for each top feature
print("\nanova results:")
for feature in flist:
    groups = [df[df["cluster"] == cluster][feature] for cluster in range(optimal_k)]
    f_stat, p_value = f_oneway(*groups)
    print(f"{feature}: f-statistic = {f_stat:.4f}, p-value = {p_value:.4e}")
