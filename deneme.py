import pandas
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.stats import skew
from scipy.integrate import trapezoid
from scipy.ndimage import binary_erosion, binary_dilation, label

def extract_features(time_data, energy_data, peak_time=None):
    """
    Extracts a variety of features from the time-series data of a photon burst.

    Parameters:
    - time_data (np.array): Array of photon arrival times.
    - energy_data (np.array): Array of photon energy levels corresponding to the time_data.
    - peak_time (float, optional): Time of the peak event if known; otherwise, the peak is calculated.

    Returns:
    - features (dict): Dictionary of extracted features.
    """

    # 1. Amplitude-based Features
    peak_intensity = np.max(energy_data)  # Peak intensity is the max energy level

    # Calculate the mean and standard deviation of the time and energy
    mean_time = np.mean(time_data)
    std_time = np.std(time_data)
    mean_energy = np.mean(energy_data)
    std_energy = np.std(energy_data)

    # Signal-to-Noise Ratio (SNR) calculation
    if peak_time is None:
        peak_time = time_data[
            np.argmax(energy_data)
        ]  # If no peak time is given, calculate it from energy
    background_noise = np.std(
        energy_data[(time_data < peak_time - 0.1) | (time_data > peak_time + 0.1)]
    )
    SNR = peak_intensity / background_noise

    # 2. Time-based Features
    # Rise time (time taken to reach peak intensity)
    rise_time = time_data[np.argmax(energy_data)] - time_data[0]

    # Decay time (time taken to return to 10% of the peak intensity)
    decay_time_idx = np.where(energy_data <= 0.1 * peak_intensity)[0]
    decay_time = (
        time_data[decay_time_idx[0]] - time_data[np.argmax(energy_data)]
        if decay_time_idx.size > 0
        else np.nan
    )

    # Duration (total time of the burst)
    duration = time_data[-1] - time_data[0]

    # Centroid (weighted average of time based on energy)
    centroid = np.sum(time_data * energy_data) / np.sum(energy_data)

    # Skewness (measure of asymmetry)
    skewness = skew(energy_data)

    # 3. Morphological Features
    # Area under the curve (AUC) for the energy vs. time graph
    auc = trapezoid(energy_data, time_data)

    # Binary signal for morphological analysis (above 10% of peak intensity)
    binary_signal = energy_data > 0.1 * peak_intensity
    dilated_signal = binary_dilation(binary_signal)
    eroded_signal = binary_erosion(binary_signal)

    # Number of distinct regions (connected components after dilation)
    regions, num_regions = label(dilated_signal)

    # Collecting all features into a dictionary
    features = {
        "Peak Intensity": peak_intensity,
        "Mean Time": mean_time,
        "Std Time": std_time,
        "Mean Energy": mean_energy,
        "Std Energy": std_energy,
        "SNR": SNR,
        "Rise Time": rise_time,
        "Decay Time": decay_time,
        "Duration": duration,
        "Centroid": centroid,
        "Skewness": skewness,
        "AUC": auc,
        "Number of Regions": num_regions,
    }

    return features

feature_types = ["Peak Intensity","Mean Time","Std Time","Mean Energy",
            "Std Energy","SNR","Rise Time","Decay Time","Duration",
            "Centroid","Skewness","AUC","Number of Regions"]
dfs=[]
for i in range(12):
    exec(f"m{i+1} = pandas.read_csv('events/{i+1}.csv')")
    #exec(f"s{i} = {i}s.csv")
    dfs.append(eval(f"m{i+1}"))
    i+=1

for i, df in enumerate(dfs, start=1):
    exec(f"features{i} = extract_features(np.array(list(df['times'])), np.array(list(df['energies'])))")

colors = ["orangered", "goldenrod", "lightseagreen", "navy", "deeppink"]
for i, feature in enumerate(feature_types):
    fig = plt.figure(figsize = (10, 5))
    ilgili = []
    for k in range(12):
        ilgili.append(eval(f"features{k+1}"))
    names = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    values = [x[feature] for x in ilgili]
    plt.bar(names, values, color = colors[i%5], 
        width = 0.9)
    plt.xlabel("Burst sources")
    plt.ylabel(feature)
    plt.title("Graph of features")
    plt.show()

