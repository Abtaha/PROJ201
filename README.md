# Morphological Classification of Magnetar Bursts
## Codebase outline
__Click on their name to see the source codes.__
#### Event handling, visualization, data extraction
- [Event/Event.py](Event/Event.py) : The object-oriented source code for analyzing event records, extracting features, visualizing nearly everything about any event on their own. It's a custom package!
- [Event/test.py](Event/test.py) : Code required for testing the functionality of Event.py
- [features.py](features.py) : Handles event files, acts as a bridge between Event.py and event record files, writes every event's features to a file named [export_data.csv](export_data.csv). A total of 221 pulse features is extracted from 101 event records.

#### Data analysis, model training, further visualization of clusters/events
- [clusters.csv](clusters.csv) : The data file including KMeans and DBSCAN clusterings of all the events.
- [model/kmeans.py](model/kmeans.py) : A code that applies various analysis techniques and machine learning algorithms to the extracted event data.
  -  Analysis Techniques: Principal Component Variance Analysis, Feature importance sorting, Silhouette Scores of clusterings
  -  Machine Learning Algorithms: K-means clustering, DBSCAN clustering, Hierarchial Clustering 
- [model/kmeans1.py](model/kmeans1.py) : A statistical analysis and 3D visualization code for understanding the behaviours of features and visualizing 3D view of clusters.
- [model/simulation.py](model/simulation.py) : A simulation for understanding how varying specific features change explained variance ratio, also checks for correlations between Principal Components and features, and visualizes a BiPlot.
- [model/test3ax.py](model/test3ax.py) : A test for visualizing 5D/4D/3D clusters in 3D space.

## Running the code
[requirements.txt](requirements.txt) file includes all the necessary packages to develop and run this project properly.
The main packages used in this project are:
```
matplotlib
pandas
sklearn
scipy
numpy
seaborn
kneed
```
First clone the repository:
```bash
git clone https://github.com/Abtaha/PROJ201.git
```
Then move into the repository and install the required packages:
```bash
pip install -r requirements.txt
```
Then run the following python files from the root directory of the repository to see the visualizations and analysis:
```
model/kmeans.py
model/simulation.py
```
