import json
import pandas
import matplotlib.pyplot as plt
import numpy as np



def compute_threshold(photon_times, bin_size=0.002, factor=2):
    """
    Compute a valid threshold based on the statistical distribution of photon arrivals.
    
    Parameters:
    - photon_times: array-like, arrival times of each photon
    - bin_size: float, width of each time bin (in seconds)
    - factor: float, the number of standard deviations above the mean to set the threshold
    
    Returns:
    - time_bins: array of time bins
    - photon_counts: array of photon counts per bin
    - thresholded_bins: list of bins where photon count exceeds the computed threshold
    """
    # Define time bins for photon counts
    time_bins = np.arange(min(photon_times), max(photon_times) + bin_size, bin_size)
    photon_counts, _ = np.histogram(photon_times, bins=time_bins)
    
    # Calculate mean and standard deviation of photon counts
    mean_counts = np.mean(photon_counts)
    std_counts = np.std(photon_counts)
    
    # Compute threshold based on mean + factor * standard deviation
    threshold = mean_counts + factor * std_counts
    
    # Identify bins where photon count exceeds the computed threshold
    thresholded_bins = time_bins[:-1][photon_counts > threshold]
    
    return time_bins, photon_counts, thresholded_bins, threshold


def extract_features(time, energy):
    time_bins, photon_counts, thresholded_bins = compute_threshold(time)
    peak_photon_count = np.max(photon_counts)
    duration = time[-1] - time[0]
    fluence = np.trapezoid(energy, time)  # Energy fluence calculation
    rise_time = time[np.argmax(photon_counts)] - time[0]
    fall_time = time[-1] - time[np.argmax(photon_counts)]
    burst_frequency = len(thresholded_bins)  # Count of bins exceeding threshold

    return {
        "peak_photon_count": peak_photon_count,
        "duration": duration,
        "fluence": fluence,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "burst_frequency": burst_frequency,
    }


# Read in the data
data1 = pandas.read_csv("Event List for PROJ dersi.csv")
data2 = pandas.read_csv("Event Lists for PROJ Dersi (1).csv")
data3 = pandas.read_csv("Event List for PROJ Dersi.csv")

dfs = [data1, data2, data3]
colors = ["red", "green", "blue"]

for i, df in enumerate(dfs):
    X = np.array(dfs[i])
    BIN_SIZE = np.arange(min(dfs[i]["times"]), max(dfs[i]["times"]), 0.002)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    ax1.hist(dfs[i]["times"], bins=BIN_SIZE, label="Full Spectrum", color=colors[i])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of Photons")
    ax1.set_title("Overlay of Gamma-Ray Bursts")

    lower_band = [x[0] for x in X if x[1] < 50]
    ax2.hist(lower_band, bins=BIN_SIZE, label="Low Energy Band", color=colors[i])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Number of Photons")
    ax2.set_title("Gamma-Ray Bursts: Low Energy Band (Energy < 50)")

    high_band = [x[0] for x in X if 50 < x[1]]
    ax3.hist(high_band, bins=BIN_SIZE, label="High Energy Band", color=colors[i])
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of Photons")
    ax3.set_title("Gamma-Ray Bursts: High Energy Band (Energy > 50)")

    # plt.tight_layout()
    # plt.show()

    # Step 2: Compute threshold based on photon count
    time_bins, photon_counts, thresholded_bins, threshold = compute_threshold(
        np.array(df["times"]), bin_size=0.002
    )

    print(thresholded_bins)

    # # Output for thresholding (optional)
    # print("Time bins:", time_bins)
    # print("Photon counts per bin:", photon_counts)
    # print("Thresholded bins (significant bursts):", thresholded_bins)

    # # Step 3: Extract features for the burst analysis
    # features = extract_features(np.array(df["times"]), np.array(df["energies"]))

    # # Output extracted features
    # print("Extracted Features:")
    # for key, value in features.items():
    #     print(f"{key}: {value}")
