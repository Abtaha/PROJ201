from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np

from scipy import stats
from scipy.integrate import trapezoid
from scipy.ndimage import binary_erosion, binary_dilation, label


class Photon:
    def __init__(self, energy, time):
        self.energy = energy
        self.time = time

    def __str__(self):
        return f"Photon(energy={self.energy}, time={self.time})"

    def __repr__(self):
        return self.__str__()


class Event:
    # features = {
    #     "Peak Intensity": "keV",
    #     # "Mean Time": "s",
    #     # "Std Time": "s",
    #     "Mean Energy": "keV",
    #     "Std Energy": "keV",
    #     # "SNR": None,
    #     "Rise Time": "s",
    #     "Decay Time": "s",
    #     "Duration": "s",
    #     "Centroid": None,
    #     "Skewness": None,
    #     "Kurtosis": None,
    #     "Total Energy Released": "keV",
    #     # "Number of Regions": None,
    # }

    def __init__(
        self, photons, type, name="Unnamed", threshold: np.float64 = np.float64(0)
    ):
        self.photons = photons
        self.type = type
        self.name = name

        self.threshold = threshold
        self.features = {}

    def __str__(self):
        return f"Event(photons={self.photons})"

    def __repr__(self):
        return self.__str__()

    def get_bins(
        self, get_thresholded: bool = True, bin_size: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group photons into time bins or filter bins based on the threshold.

        Parameters:
        - get_thresholded (bool): If True, return only bins where counts > threshold.
        - bin_size (float): Size of each bin (default 0.02).

        Returns:
        - time_counts (np.ndarray): Counts of photons in each bin.
        - time_bins (np.ndarray): Bin edges.
        """
        if not self.photons:
            raise ValueError("No photons provided for binning.")

        # Extract photon times
        photon_times = np.array([photon.time for photon in self.photons])

        # Compute bin edges based on photon times
        min_time, max_time = np.min(photon_times), np.max(photon_times)
        bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

        # Compute histogram of photon counts
        time_counts, time_bins = np.histogram(photon_times, bins=bin_edges)

        filtered_bins = (
            time_counts > self.threshold
            if self.threshold > 0 and get_thresholded
            else np.ones_like(time_counts, dtype=bool)
        )

        # Return the original bins and counts without filtering
        return time_counts[filtered_bins], time_bins[:-1][filtered_bins]

    def compute_threshold(self, bin_size: float = 0.02, factor: int = 5) -> np.float64:
        """
        Compute a threshold for detecting bursts based on photon counts in time bins.

        The threshold is set as the mean photon count plus a factor times the standard deviation
        of photon counts.

        Args:
            bin_size (float): Size of the time bins for histogramming (default is 0.02).
            factor (int): Multiplier for the standard deviation to set the threshold (default is 5).

        Returns:
            np.float64: The computed threshold value.
        """
        time_counts, time_bins = self.get_bins(get_thresholded=False)
        print(time_counts)

        mean_counts = np.mean(time_counts)
        std_counts = np.std(time_counts)

        # Compute threshold based on mean + factor * standard deviation
        threshold = mean_counts + factor * std_counts
        self.threshold = threshold
        print(threshold)

        return threshold

    def plot_event(self, bin_size=0.02):
        """
        Plot the events based on a threshold, focusing on photon times.
        Exclude bins with fewer photons than the threshold.

        Parameters:
        - bin_size: float, the size of the time bins for histogramming (default is 0.02)
        """

        counts, bins = self.get_bins(
            get_thresholded=True if self.threshold > 0 else False, bin_size=bin_size
        )

        # Plot only the bins that exceed the threshold
        plt.bar(
            bins,  # Bin edges for the x-axis
            counts,  # Counts for the y-axis
            width=bin_size,
            color="skyblue",
            edgecolor="black",
        )

        # If a threshold is provided, add a line to indicate the threshold
        if self.threshold is not None:
            plt.axhline(
                float(self.threshold), color="red", linestyle="--", label="Threshold"
            )

        plt.xlabel("Time")
        plt.ylabel("Number of Photons")
        plt.legend()
        plt.title("Photon Arrival Times (with Threshold)")
        plt.show()

    def extract_features(self, peak_time=None) -> dict:
        """
        Extracts a variety of features from the time-series data of a photon burst.

        Parameters:
        - peak_time (float, optional): Time of the peak event if known; otherwise, the peak is calculated.

        Returns:
        - features (dict): Dictionary of extracted features.
        """
        time_counts, time_bins = self.get_bins(get_thresholded=True)

        time_data = np.array([photon.time for photon in self.photons])
        energy_data = np.array([photon.energy for photon in self.photons])
        # find the histogram starting time
        for td in time_data:
            if td >= time_bins[0]:
                start_index = np.where(time_data == td)[0][0]
                break
        for td in time_data:
            if td >= time_bins[-1]:
                end_index = np.where(time_data == td)[0][0] - 1
                break
        time_data = time_data[start_index : end_index + 1]
        energy_data = energy_data[start_index : end_index + 1]

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
        # histograma ihtiyaç var, ilgili bin - 0 benim için rise edilmiş timeı verecek
        peak_index = np.argmax(time_counts)
        peak_bin = time_bins[peak_index]
        # print(peak_bin)
        # print(time_data[0])
        # print(time_bins[0])
        # print(time_bins)

        rise_time = peak_bin - time_data[0]
        decay_time = time_data[-1] - peak_bin

        # Decay time (time taken to return to 10% of the peak intensity)
        # decay_time_idx = np.where(energy_data <= 0.1 * peak_intensity)[0]
        # decay_time = (
        #    time_data[decay_time_idx[0]] - time_data[np.argmax(energy_data)]
        #    if decay_time_idx.size > 0
        #    else np.nan
        # )

        # Duration (total time of the burst)
        duration = time_data[-1] - time_data[0]

        # Centroid (weighted average of time based on energy)
        centroid = np.sum(time_data * energy_data) / np.sum(energy_data)

        # Skewness (measure of asymmetry) and Kurtosis (Tailedness)
        # skewness = skew(energy_data)
        skewness = stats.skew(time_data)
        kurtosis = stats.kurtosis(time_data)

        # 3. Morphological Features
        # Area under the curve (AUC) for the energy vs. time graph
        # auc = trapezoid(energy_data, time_data)
        # Alternative approach, directly summing the energies, in keV unit
        auc = sum(energy_data)

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
            # "SNR": SNR,
            "Rise Time": rise_time,
            "Decay Time": decay_time,
            "Duration": duration,
            "Centroid": centroid,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Total Energy Released": auc,
            # "Number of Regions": num_regions,
        }

        self.features = features
        return features


class EventList:
    def __init__(self, main, sb, combined=None):
        self.main = main
        self.sb = sb
        self.combined = combined

    def __str__(self):
        return f"EventList(main={self.main} sb={self.sb})"

    def __repr__(self):
        return self.__str__()

    def combine_events(self, tolerance: float = 0.0002) -> Event:
        """
        Combine the photons from two events into a single event.
        """
        main_event = self.main
        sb_event = self.sb

        # Extract photon times from both events
        main_times = np.array([photon.time for photon in main_event.photons])
        sb_times = np.array([photon.time for photon in sb_event.photons])

        if tolerance == 0:
            # Use np.intersect1d to find common times efficiently
            common_times = np.intersect1d(main_times, sb_times)

            # Return photons with common times
            common_photons = [
                photon for photon in main_event.photons if photon.time in common_times
            ]
            return Event(common_photons, type="combined")

        # If threshold is not zero, we want to find photons within the threshold range
        common_photons = []

        # Efficiently find all main photons within threshold of sb photons
        for main_photon in main_event.photons:
            # Calculate time differences
            time_diffs = np.abs(sb_times - main_photon.time)

            # Find indices where time differences are less than or equal to the threshold
            matching_indices = np.where(time_diffs <= tolerance)[0]

            # Collect the corresponding sb_photons using the matching indices
            close_photons = [sb_event.photons[i] for i in matching_indices]

            if close_photons:  # If any close photons exist, add the main photon
                common_photons.append(main_photon)

        self.combined = Event(
            common_photons, type="combined", name=main_event.name + " Combined"
        )
        return self.combined

    def plot_events_multiple_axes(self, labels=None, threshold=None, bin_size=0.02):
        events = (
            [self.main, self.sb, self.combined]
            if self.combined
            else [self.main, self.sb]
        )

        if labels is None:
            labels = [f"Event {events[i].name}" for i in range(len(events))]

        if len(events) != len(labels):
            raise ValueError("The number of labels must match the number of events.")

        fig, axes = plt.subplots(
            len(events), 1, figsize=(8, 3 * len(events)), sharex=True
        )
        if len(events) == 1:
            axes = [axes]

        for i, (event, label) in enumerate(zip(events, labels)):
            all_times = [photon.time for photon in event.photons]
            time_counts, time_bins = np.histogram(
                all_times,
                bins=np.arange(min(all_times), max(all_times) + bin_size, bin_size),
            )
            axes[i].hist(
                all_times,
                bins=np.arange(min(all_times), max(all_times) + bin_size, bin_size),
                edgecolor="black",
                color="skyblue",
                alpha=0.7,
            )
            if threshold is not None:
                axes[i].axhline(
                    threshold, color="red", linestyle="--", label="Threshold"
                )
            axes[i].set_title(label, fontsize=10)
            axes[i].set_ylabel("Count", fontsize=8)
            axes[i].tick_params(axis="both", labelsize=8)
            axes[i].legend(fontsize=8)

        axes[-1].set_xlabel("Time", fontsize=10)
        plt.tight_layout(pad=1.0, h_pad=0.5)
        plt.show()
