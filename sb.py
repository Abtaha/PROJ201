import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re

import scipy
import scipy.signal

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

for base_name, datasets in data.items():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Set the bin size
    bin_size = 0.002

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

        # Detect peaks in the histogram
        peaks, _ = scipy.signal.find_peaks(time_counts, prominence=70)

        # Plot the histogram
        ax1.hist(
            datasets["main"]["time"],
            bins=time_bins,
            label="Main Times",
            color="b",
            edgecolor="black",
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Photons")
        ax1.set_title("Main Times")
        ax1.set_xlim(left=0)

        # Mark the peaks on the histogram
        ax1.plot(time_bins[peaks], time_counts[peaks], "rx", label="Peaks")

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

        # Detect peaks in the histogram
        peaks, _ = scipy.signal.find_peaks(time_counts)

        # Plot the histogram
        ax2.hist(
            datasets["sb"]["time"],
            bins=time_bins,
            label="SB Times",
            color="r",
            edgecolor="black",
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of Photons")
        ax2.set_title("SB Times")
        ax2.set_xlim(left=0)

        # Mark the peaks on the histogram
        ax2.plot(time_bins[peaks], time_counts[peaks], "rx", label="Peaks")

    # Plot the intersection of 'main' and 'sb' times in the third chart
    if datasets["main"] is not None and datasets["sb"] is not None:
        # Find the intersection of 'main' and 'sb' times (common time values)
        common_times = np.intersect1d(datasets["main"]["time"], datasets["sb"]["time"])

        # Calculate bins based on the bin size for the intersection
        common_time_bins = np.arange(
            min(common_times), max(common_times) + bin_size, bin_size
        )
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
        ax3.set_xlim(left=0)

    # Add titles and legends
    plt.suptitle(f"{base_name} Data")
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
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
