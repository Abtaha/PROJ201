import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("export_data.csv")

# Select the features
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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA on the full data
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


# 1. **Explained Variance Ratio Plot** for PCA components
def plot_explained_variance(pca, title="Explained Variance Ratio"):
    plt.figure(figsize=(8, 6))
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        color="b",
        label="Variance Ratio",
    )
    plt.title(title)
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.show()


# Plot explained variance for the original dataset
plot_explained_variance(pca)


# 2. **Effect of Varying Each Feature on PCA Variance**
def vary_feature_effect_on_pca(X_scaled, feature_idx, variation_range, pca_base):
    explained_variances = []

    for value in variation_range:
        X_varied = X_scaled.copy()
        X_varied[:, feature_idx] = value  # Vary the specific feature
        pca_varied = PCA()
        pca_varied.fit(X_varied)
        explained_variances.append(pca_varied.explained_variance_ratio_)

    return np.array(explained_variances)


# Range of values to vary the feature
variation_range = np.linspace(-3, 3, 10)

# 2. **Effect of Varying Each Feature on PCA Variance**
for i, feature in enumerate(features):
    explained_variances = vary_feature_effect_on_pca(X_scaled, i, variation_range, pca)

    plt.figure(figsize=(10, 6))
    for j in range(explained_variances.shape[1]):
        plt.plot(variation_range, explained_variances[:, j], label=f"PC{j+1}")

    plt.title(f"Explained Variance Ratio When Varying {feature}")
    plt.xlabel("Feature Value (scaled)")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()
    plt.show()


# 3. **Biplot for Visualizing Loadings and Scores**
def biplot(pca, X_scaled, features, title="PCA Biplot"):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5, label="Data points")
    for i in range(len(features)):
        # Scale arrows for better visibility
        arrow_length = 2  # You can adjust this for better visualization
        plt.arrow(
            0,
            0,
            pca.components_[0, i] * arrow_length,
            pca.components_[1, i] * arrow_length,
            color="r",
            alpha=0.5,
            head_width=0.05,
        )
        plt.text(
            pca.components_[0, i] * arrow_length,
            pca.components_[1, i] * arrow_length,
            features[i],
            color="r",
            ha="center",
            va="center",
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


# Visualize the biplot for original data
biplot(pca, X_scaled, features)


# 4. **Heatmap of Feature Correlations with Principal Components**
def plot_feature_pca_correlation(pca, X_scaled, features):
    # Number of principal components
    n_components = pca.components_.shape[0]

    # Calculate correlation between each feature and each principal component
    correlation_matrix = np.dot(X_scaled, pca.components_.T) / (X_scaled.shape[0] - 1)

    # Plot the correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        # annot=True,
        cmap="coolwarm",
        xticklabels=features,
        yticklabels=[f"PC{i+1}" for i in range(n_components)],
        annot_kws={"size": 10},  # Adjusting annotation font size
    )
    plt.title("Feature vs Principal Component Correlation")
    plt.show()


# Plot feature correlations with principal components
plot_feature_pca_correlation(pca, X_scaled, features)
