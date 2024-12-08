import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

bin_size = 2


photon_times = np.array(
    [
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]
)

print(photon_times)

# Compute bin edges based on photon times
min_time, max_time = np.min(photon_times), np.max(photon_times)
bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

# Compute histogram of photon counts
time_counts, time_bins = np.histogram(photon_times, bins=bin_edges)


filtered_bins = np.ones_like(time_counts, dtype=bool)

counts, bins = time_counts[filtered_bins], time_bins[:-1][filtered_bins]

print(time_counts, len(counts))
print(time_bins, len(bins))

print(time_bins[:-1])

plt.bar(
    time_bins[:-1],  # Left edges of the bins
    time_counts,  # Photon counts
    width=bin_size,  # Bar width matches bin size
    align="edge",  # Align bars with edges
    color="skyblue",
    edgecolor="black",
)
plt.xticks(bin_edges)
# If a threshold is provided, add a line to indicate the threshold
plt.xlabel("Time")
plt.ylabel("Number of Photons")
plt.legend()
plt.title("Photon Arrival Times (with Threshold)")
plt.show()

photons_per_bin = []

for i in range(len(bin_edges) - 1):
    start = bin_edges[i]
    end = bin_edges[i + 1]

    # Include the upper edge for the last bin
    if i == len(bin_edges) - 2:
        photons_in_bin = photon_times[(photon_times >= start) & (photon_times <= end)]
    else:
        photons_in_bin = photon_times[(photon_times >= start) & (photon_times < end)]

    photons_per_bin.append(photons_in_bin)

    print(start, end, photons_in_bin)
