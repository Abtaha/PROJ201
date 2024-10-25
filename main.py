import json
import pandas
import matplotlib.pyplot as plt
import numpy as np


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


def extract_features(time, energy):
    threshold = compute_threshold(energy)
    peak_energy = np.max(energy)
    duration = time[-1] - time[0]
    fluence = np.trapezoid(energy, time)
    rise_time = time[np.argmax(energy)] - time[0]
    fall_time = time[-1] - time[np.argmax(energy)]
    burst_frequency = len(np.where(energy > threshold)[0])
    print(np.min(energy))

    return {
        "peak_energy": peak_energy,
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

    plt.tight_layout()
    plt.show()