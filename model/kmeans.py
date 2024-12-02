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

features=[
    #['Duration', 'Peak Time'],
    #['Duration', 'Kurtosis',],
    ['Duration', 'Peak Intensity'],
    ['Duration', 'Total Energy Released'],
    ['Skewness', 'Kurtosis'],
    ['Skewness', 'Total Energy Released'],
    ['Kurtosis', 'Total Energy Released'],
    #['Skewness', 'Peak Intensity'],
    #['Kurtosis', 'Peak Intensity']
]
#fig, axes = plt.subplots(
#    len(features), 1
#)
scaler = MinMaxScaler()

for couple in features:
    x = couple[0]
    y = couple[1]

    scaler.fit(df[[x]])
    df[x] = scaler.transform(df[[x]])

    scaler.fit(df[[y]])
    df[y] = scaler.transform(df[[y]])

    sse = []
    k_rng = range(1,13)
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(df[[x,y]])
        sse.append(km.inertia_)
    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(k_rng,sse)
    plt.show()

    km = KMeans(n_clusters=4)
    y_predicted = km.fit_predict(df[[x,y]])
    
    df['cluster']=y_predicted
    df1 = df[df.cluster==0]
    df2 = df[df.cluster==1]
    df3 = df[df.cluster==2]
    df4 = df[df.cluster==3]
    df5 = df[df.cluster==4]
    df6 = df[df.cluster==5]

    plt.scatter(df1[x],df1[y],color='green')
    plt.scatter(df2[x],df2[y],color='red')
    plt.scatter(df3[x],df3[y],color='blue')
    plt.scatter(df4[x],df4[y],color='black')
    plt.scatter(df5[x],df5[y],color='yellow')
    plt.scatter(df6[x],df6[y],color='cyan')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()

    #plt.scatter(df[x],df[y])
    #plt.show()

