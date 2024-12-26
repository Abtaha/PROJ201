### Data Analysis and Clustering of Magnetar Bursts

#### Introduction
In this report, we analyze a dataset containing features extracted from magnetar bursts. The dataset includes several measurements, such as peak intensity, energy, skewness, kurtosis, and various time-related features. The objective of this analysis is to apply dimensionality reduction, clustering techniques, and explore feature relationships to better understand the underlying structure of the data.

#### Data Preprocessing
We begin by loading the data, which consists of various columns representing burst characteristics. The following features were selected for analysis:

- **Duration**
- **Peak Energy Bin**
- **Peak Energy in Bin**
- **Skewness**
- **Kurtosis**
- **Centroid**
- **Rise Time**
- **Decay Time**
- **Total Energy Released**

The data was then standardized using `StandardScaler` to ensure that all features contribute equally to the analysis.

#### Principal Component Analysis (PCA)
PCA was applied to reduce the dimensionality of the dataset and identify the most significant directions of variance. The explained variance ratio plot showed that the first few principal components (PCs) account for a substantial portion of the variance, highlighting the importance of these components for capturing the structure of the data.

##### Explained Variance Ratio
The explained variance ratio plot provides insight into how much variance each principal component accounts for. The first two PCs captured the majority of the variance, indicating that most of the information in the data is contained within these components.

#### Clustering Analysis
To explore potential groupings within the data, three clustering algorithms were applied: K-means, DBSCAN, and Hierarchical Clustering. Each of these methods offers different perspectives on the data's structure.

1. **K-means Clustering**
   K-means clustering with three clusters revealed distinct groupings in the data. The scatter plot of the first two PCs colored by K-means labels showed clear separations between the clusters, suggesting that the data may naturally divide into three categories.

2. **DBSCAN Clustering**
   DBSCAN, a density-based clustering algorithm, was used to identify regions of high data density. Unlike K-means, DBSCAN can detect noise and find clusters of arbitrary shapes. The results showed a mixture of dense clusters and noise, indicating that some data points may not belong to any particular cluster.

3. **Hierarchical Clustering**
   The dendrogram from hierarchical clustering provided a hierarchical view of the data, revealing how the data points are grouped at various levels. This method offers a different perspective compared to K-means and DBSCAN, where the tree structure helps understand the relationships between clusters.

#### Feature Analysis
To gain a deeper understanding of the data, several analyses were performed:

1. **Effect of Varying Each Feature on PCA Variance**
   We investigated the impact of varying individual features on the explained variance in PCA. By systematically altering the value of each feature, we observed how the principal components' variance changes. This analysis helps understand the relative importance of each feature in determining the data's variance.

2. **PCA Biplot**
   A biplot was created to visualize the loadings of the features in the first two principal components. This plot provided insight into the relationships between features and how they contribute to the principal components.

3. **Feature vs. Principal Component Correlation Heatmap**
   A heatmap was generated to show the correlations between features and principal components. This visualization helped identify which features are most strongly associated with each principal component, providing valuable insight into the underlying data structure.

#### Clustering and Evaluation
The clustering methods were evaluated using **Silhouette Scores**, which measure how similar each point is to its own cluster compared to other clusters. The silhouette score for K-means clustering was found to be higher than that of DBSCAN, suggesting that the K-means clusters are more compact and well-separated. However, DBSCAN was able to identify noise points, which K-means could not.

- **K-means Silhouette Score**: The higher score indicates that K-means has successfully separated the data into meaningful clusters.
- **DBSCAN Silhouette Score**: The lower score suggests that while DBSCAN can identify dense regions, its results are more sensitive to noise.

#### Results
- **Cluster Sizes**: K-means identified three clusters, while DBSCAN found fewer, with some data points categorized as noise.
- **PCA**: The explained variance plot confirmed that most of the information in the data is captured by the first few principal components.
- **Feature Importance**: The biplot and correlation heatmap revealed which features contribute most significantly to the principal components, helping to interpret the clusters.
- **Silhouette Scores**: K-means performed better in terms of cluster cohesion, but DBSCAN provided additional insights by detecting outliers.

#### Conclusion
This analysis provided a detailed exploration of the magnetar burst data using PCA, clustering techniques, and feature analysis. By leveraging K-means, DBSCAN, and hierarchical clustering, we identified meaningful patterns and groups within the data. The PCA analysis also revealed that most of the variance in the data is captured by a few components, which helps simplify the problem for clustering and further analysis.

In future work, further tuning of DBSCAN parameters and experimenting with other clustering algorithms may provide additional insights.
