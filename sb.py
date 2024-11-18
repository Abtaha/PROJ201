import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re

import scipy
import scipy.signal

import numpy as np
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
        # "Mean Time": mean_time,
        # "Std Time": std_time,
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


# Define folder path
folder = os.path.join(os.getcwd(), "Sgr1935 Event List")
data = {}

for file in os.listdir(folder):
    if file.endswith("_sb.csv") or file.endswith(".csv"):
        # Check if it's an '_sb.csv' file
        is_sb = file.endswith("_sb.csv")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder, file))

        # Extract the SGR number from the filename (e.g., 'sgr1935_488409650')
        sgr_number_match = re.search(r"(sgr\d+_\d+)", file)
        base_name = sgr_number_match.group(1) if sgr_number_match else None

        # Ensure the SGR number key exists in the data dictionary
        if base_name not in data:
            data[base_name] = {"main": None, "sb": None}

        # Convert the columns to NumPy arrays for time and energy
        time_array = df.iloc[:, 0].to_numpy()  # Time values (assuming first column)
        energy_array = df.iloc[
            :, 1
        ].to_numpy()  # Energy values (assuming second column)

        # Store the time and energy arrays under 'sb' or 'main' based on the file type
        if is_sb:
            data[base_name]["sb"] = {
                "time": time_array,
                "energy": energy_array,
            }
        else:
            data[base_name]["main"] = {
                "time": time_array,
                "energy": energy_array,
            }

all_fts = []
for base_name, datasets in data.items():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Set the bin size
    bin_size = 0.002

    # if datasets["main"] is not None:
    #     datasets["main"]["time"] = datasets["main"]["time"][
    #         datasets["main"]["time"] >= 0
    #     ]

    # if datasets["sb"] is not None:
    #     datasets["sb"]["time"] = datasets["sb"]["time"][datasets["sb"]["time"] >= 0]

    # Plot histogram for 'main' time data using np.histogram
    if datasets["main"] is not None:
        # Calculate the histogram data
        time_counts, time_bins = np.histogram(
            datasets["main"]["time"],
            bins=np.arange(
                min(datasets["main"]["time"]),
                max(datasets["main"]["time"]) + bin_size,
                bin_size,
            ),
        )

        # Plot the histogram
        ax1.hist(
            datasets["main"]["time"],
            bins=1000,
            label="Main Times",
            color="blue",
            edgecolor="black",
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Photons")
        ax1.set_title("Main Times")

    # Plot histogram for 'sb' time data using np.histogram
    if datasets["sb"] is not None:
        # Calculate the histogram data
        time_counts, time_bins = np.histogram(
            datasets["sb"]["time"],
            bins=np.arange(
                min(datasets["sb"]["time"]),
                max(datasets["sb"]["time"]) + bin_size,
                bin_size,
            ),
        )

        # Plot the histogram
        ax2.hist(
            datasets["sb"]["time"],
            bins=time_bins,
            label="SB Times",
            color="red",
            edgecolor="black",
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of Photons")
        ax2.set_title("SB Times")

    # Plot the intersection of 'main' and 'sb' times in the third chart
    if datasets["main"] is not None and datasets["sb"] is not None:
        common_data = []

        for i, mtime in enumerate(datasets["main"]["time"]):
            for stime in datasets["sb"]["time"]:
                if abs(mtime - stime) <= 0.02:
                    common_data.append(
                        {"time": mtime, "energy": datasets["main"]["energy"][i]}
                    )

        # Use broadcasting to calculate pairwise differences and find matches within tolerance
        common_times = datasets["main"]["time"][
            np.any(
                np.abs(datasets["main"]["time"][:, None] - datasets["sb"]["time"])
                <= 0.02,
                axis=1,
            )
        ]

        # Calculate bins based on the bin size for the intersection
        common_time_bins = np.arange(
            min(common_times), max(common_times) + bin_size, bin_size
        )

        # Create histogram data
        time_counts, _ = np.histogram(common_times, bins=common_time_bins)

        # Identify peaks using stricter up-then-down condition
        valid_peaks = []
        peak_start = None  # Track the start of the peak

        # Find the maximum histogram value
        max_value = np.max(time_counts)

        for i in range(1, len(time_counts) - 1):
            # Check if the current bin is greater than the previous bin (gradual increase)
            if time_counts[i] > time_counts[i - 1]:
                if peak_start is None:
                    peak_start = i  # Mark the start of the rise

            # After the peak starts, check for the decrease to 10% of the highest bin value
            if peak_start is not None and time_counts[i] < time_counts[i - 1]:
                # Ensure it drops to 10% of the maximum value
                if time_counts[i] <= 0.1 * max_value:
                    # Peak detected, add it and reset the start
                    valid_peaks.append(peak_start)
                    peak_start = None  # Reset peak detection

        ax3.hist(
            common_times,
            bins=common_time_bins,
            label="Intersection of Main and SB Times",
            color="purple",
            edgecolor="black",
        )
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Number of Photons")
        ax3.set_title("Intersection of Main and SB Times")

        # Now, pass common_times and common_energies to the extract_features function
        features = extract_features(
            np.array([entry["time"] for entry in common_data]),
            np.array([entry["energy"] for entry in common_data]),
        )
        # print(json.dumps(features, indent=4))
        all_fts.append(features)

        # # Mark valid peaks
        # ax3.plot(
        #     common_time_bins[valid_peaks], time_counts[valid_peaks], "rx", label="Peaks"
        # )

    # Add titles and legends
    plt.suptitle(f"{base_name} Data")
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()

rise_times = [ft["Rise Time"] for ft in all_fts]

# Create a bar chart
plt.bar(range(len(rise_times)), rise_times)

# Add labels and title
plt.xlabel("Index")
plt.ylabel("Rise Time")
plt.title("Rise Times Bar Graph")

# Show the plot
plt.show()


"""
main_times = []
sb_times = []
common_times = []

tolerance = 0.001

main_times_sorted = np.sort(main_times)
sb_times_sorted = np.sort(sb_times)

# Initialize pointers for both arrays
i, j = 0, 0
common_times = []

# Use a two-pointer technique to find common times within tolerance
while i < len(main_times_sorted) and j < len(sb_times_sorted):
    main_time = main_times_sorted[i]
    sb_time = sb_times_sorted[j]

    # Check if the sb_time is within the tolerance of the main_time
    if np.abs(main_time - sb_time) <= tolerance:
        # If they are close enough, add the main_time to common_times
        common_times.append(main_time)
        i += 1  # Move the pointer in main_times
        j += 1  # Move the pointer in sb_times
    elif main_time < sb_time:
        # If main_time is smaller, move the main_times pointer forward
        i += 1
    else:
        # If sb_time is smaller, move the sb_times pointer forward
        j += 1

BIN_SIZE = np.arange(min(main_times), max(main_times), 0.002)

# Assuming 'common_times' is your list/array of times
min_time = np.min(common_times)
max_time = np.max(common_times)

# Calculate the bin edges with a bin size of 0.002
bin_size = 0.002
bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

# Create the histogram with the defined bin edges
counts, _ = np.histogram(common_times, bins=bin_edges)

# Calculate mean and standard deviation of the histogram counts
mean_counts = np.mean(counts)
std_counts = np.std(counts)

# Set the threshold to be 2 standard deviations above the mean (you can adjust this factor)
threshold_height = mean_counts + 2 * std_counts

# Use the calculated threshold to find peaks
peaks, _ = scipy.signal.find_peaks(counts, height=threshold_height, prominence=50)
num_peaks = len(peaks)
print("Peaks: ", num_peaks)

# Find the index of the bin with the highest count
max_count_index = np.argmax(counts)

# The energy (or time) range of the bin with the highest count
max_energy_bin_range = (bin_edges[max_count_index], bin_edges[max_count_index + 1])

# The highest count value (number of photons) in that bin
max_count_value = counts[max_count_index]

# Output the results
print(
    f"The bin with the highest count is between {max_energy_bin_range[0]:.3f} and {max_energy_bin_range[1]:.3f} with {max_count_value} photons."
)


# Plot histogram of the collected main times
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.hist(main_times, bins=BIN_SIZE, label="Main Times")
ax1.set_xlabel("Time")
ax1.set_ylabel("Number of Photons")
ax1.set_title("Main Times")


ax2.hist(sb_times, bins=BIN_SIZE, label="SB Times")
ax2.set_xlabel("Time")
ax2.set_ylabel("Number of Photons")
ax2.set_title("SB Times")

ax3.hist(common_times, bins=BIN_SIZE, label="Common Times")
ax3.scatter(
    bin_edges[peaks], counts[peaks], color="red", label=f"Peaks ({num_peaks})", zorder=5
)
ax3.set_xlabel("Time")
ax3.set_ylabel("Number of Photons")
ax3.set_title("Common Times")
plt.tight_layout()
plt.show()
"""
