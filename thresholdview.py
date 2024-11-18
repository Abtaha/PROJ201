import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns

def compute_threshold(energy, factor=3):
    """
    Compute a threshold for detecting bursts in the energy data.

    Parameters:
    - energy: array-like, the energy values to analyze
    - factor: float, the number of standard deviations above the mean to set the threshold (default is 3)

    Returns:
    - threshold: float, the computed threshold
    """
    # Calculate the mean and standard deviation of the energy values
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # Compute the threshold as mean + factor * standard deviation
    threshold = mean_energy + factor * std_energy

    return threshold

fig, (ax1, ax1s, ax1_, ax1s_) = plt.subplots(4, 1, figsize=(12, 6))
#data = np.random.rand(100)
data1 = pandas.read_csv("events/10.csv")
counts1, bins1, bars1 = ax1.hist(data1["times"], bins=400)
data2 = pandas.read_csv("events/10s.csv")
counts2, bins2, bars2 = ax1s.hist(data2["times"], bins=400)
thresholde = 0.1 * max(counts1)
thresholds = 0.1 * max(counts1)
p1= ax1.patches
#p2 = ax1s.patches
heights1 = [i.get_height() for i in p1]
#heights2 = [i.get_height() for i in p2]
indexs1 = next(x[0] for x in enumerate(heights1) if x[1] > thresholde)
indexe1 = next(len(heights1)-x[0] for x in enumerate(heights1[::-1]) if x[1] > thresholds)
times1 = bins1[indexs1]
timee1 = bins1[indexe1]
#print(time)
filtered1 = [i for i in data1["times"] if i > times1 and i < timee1]
#filtered2 = [i for i in data2["times"] if i > time2]
ax1_.hist(filtered1, bins=400)
#ax2.hist(filtered2, bins=100)
sns.distplot(filtered1, bins=400, color="green")
#ax1_.hist(filtered1, bins=400)
#ax1s_.hist(filtered2, bins=400)
#plt.bar(range(len(counts[index:])), counts[index:])
#print(index)
#print(heights)
#p[0].get_height()
#print(bins)
#print([i for i in bars])
#print(data)
#print(float(counts[0]))
#print(threshold)
#threshold_time = min([time for time in data["times"]])
#filtered = [time for time in data["times"] if time > 1]
#plt.hist(data["times"])
plt.tight_layout()
plt.show()
