from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("export_data.csv")
print(df.head())


# duration | kurtosis
# duration | peak intensity
# duration | total energy
# skewness | kurtosis
# Skewness | total energy

plt.scatter(df['Id'],df['Kurtosis'])
plt.xlabel('Id')
plt.ylabel('Kurtosis')
plt.show()
