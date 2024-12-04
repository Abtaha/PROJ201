from typing import Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np

from scipy.ndimage import label
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

    def __init__(self, photons, type, name="Unnamed", threshold: int = 0):
        self.photons = photons
        self.type = type
        self.name = name

        self.threshold = threshold
        self.features = {}

    def __str__(self):
        return f"Event(photons={self.photons})"

    def __repr__(self):
        return self.__str__()

    def split_pulses(self) -> list:
        """
        Split the event into pulses based on photon count exceeding the threshold.

        Returns:
            list[Event]: A list of pulse events.
        """
        # Compute the threshold and bin the photon counts
        threshold = self.compute_threshold()
        counts, bins = self.get_bins(get_thresholded=False)

        # Identify contiguous regions where counts exceed the threshold
        above_threshold = counts > threshold
        labeled_regions, num_regions = label(above_threshold)

        pulses = []
        for region_id in range(1, num_regions + 1):
            # Find indices of the bins in the current region
            region_indices = np.where(labeled_regions == region_id)[0]
            if len(region_indices) == 0:
                continue

            # Determine the time range of the region
            start_time = bins[region_indices[0]]
            end_time = (
                bins[region_indices[-1] + 1]
                if region_indices[-1] + 1 < len(bins)
                else bins[-1]
            )

            # Extract photons within this time range
            pulse_photons = [
                photon
                for photon in self.photons
                if start_time <= photon.time < end_time
            ]

            if pulse_photons:
                # Create a new Event for the pulse
                pulses.append(
                    Event(
                        photons=pulse_photons,
                        type=self.type,
                        name=f"{self.name}_Pulse_{region_id}",
                    )
                )

        return pulses

    def get_bins(
        self,
        get_thresholded: bool = True,
        max_time: float = float("inf"),
        bin_size: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group photons into time bins or filter bins based on the threshold.

        Parameters:
        - get_thresholded (bool): If True, return only bins where counts > threshold.
        - max_time (float): Maximum time for photons to be included in the binning (default is -inf).
        - bin_size (float): Size of each bin (default 0.02).

        Returns:
        - time_counts (np.ndarray): Counts of photons in each bin.
        - time_bins (np.ndarray): Bin edges.
        """
        if not self.photons:
            raise ValueError("No photons provided for binning.")

        # Extract photon times
        photon_times = np.array(
            [photon.time for photon in self.photons if photon.time < max_time]
        )

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

    def compute_threshold(self, max_time: int = 0, factor: int = 5) -> np.float64:
        """
        Compute a threshold for detecting bursts based on photon counts in time bins.

        The threshold is set as the mean photon count plus a factor times the standard deviation
        of photon counts.

        Args:
            factor (int): Multiplier for the standard deviation to set the threshold (default is 5).
            max_time (int): Maximum time to consider in the computation of the threshold (default is 0).

        Returns:
            np.float64: The computed threshold value.
        """
        time_counts, time_bins = self.get_bins(get_thresholded=False, max_time=max_time)

        # Background noise detection
        # background_bins_idx = []
        # for index, time_bin in enumerate(time_bins):
        #    if time_bin < 0:
        #        background_bins_idx.append(index)
        #    else:
        #        break
        # background_counts = [time_counts[i] for i in background_bins_idx]

        mean_counts = np.mean(time_counts)
        std_counts = np.std(time_counts)
        # 5 std + mean of background noise
        # Compute threshold based on mean + factor * standard deviation
        threshold = mean_counts + factor * std_counts
        self.threshold = threshold
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

    def extract_features(self) -> Dict[str, float]:
        """
        Extracts relevant features from the light curve based on photon time and energy.

        Parameters:
        - peak_time (float, optional): Time of the peak event if known; otherwise, the peak is calculated.

        Returns:
        - features (dict): Dictionary of extracted features including rise time, decay time, duration,
                            energy-based features, and time-based features.
        """
        # Get binned photon counts
        time_counts, time_bins = self.get_bins(get_thresholded=True)

        # Extract photon times and energies
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

        # Time and energy-based features
        peak_time = time_data[
            np.argmax(energy_data)
        ]  # Use the time corresponding to the max energy
        peak_intensity = self._get_peak_intensity(energy_data)
        mean_time, std_time = self._get_time_statistics(time_data)
        mean_energy, std_energy = self._get_energy_statistics(energy_data)

        # Signal-to-Noise Ratio (SNR)
        SNR = self._calculate_snr(time_data, energy_data, peak_time)

        # Time-based features (rise time, decay time, duration)
        rise_time, decay_time = self._get_rise_decay_time(
            time_bins, time_counts, time_data, energy_data
        )
        duration = self._get_duration(time_data)

        # Centroid, skewness, and kurtosis for time data
        centroid = self._get_centroid(time_data, energy_data)
        skewness = stats.skew(time_data)
        kurtosis = stats.kurtosis(time_data)

        # Total energy released and number of regions
        auc = self._get_total_energy_released(energy_data)
        num_regions = self._get_number_of_regions(energy_data, peak_intensity)

        # Compile all features into a dictionary
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
            "Kurtosis": kurtosis,
            "Total Energy Released": auc,
            "Number of Regions": num_regions,
            "Peak Time": peak_time,  # Include peak time in the features
        }

        self.features = features
        return features

    def _get_peak_intensity(self, energy_data: np.ndarray) -> float:
        """
        Calculates the peak intensity, defined as the maximum energy in the event.
        """
        return np.max(energy_data)

    def _get_time_statistics(self, time_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the mean and standard deviation of the photon arrival times.
        """
        mean_time = np.mean(time_data)
        std_time = np.std(time_data)
        return mean_time, std_time

    def _get_energy_statistics(self, energy_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the mean and standard deviation of the photon energies.
        """
        mean_energy = np.mean(energy_data)
        std_energy = np.std(energy_data)
        return mean_energy, std_energy

    def _calculate_snr(
        self, time_data: np.ndarray, energy_data: np.ndarray, peak_time: float
    ) -> float:
        """
        Calculates the Signal-to-Noise Ratio (SNR) based on energy data and peak time.
        """
        # Calculate the background noise (standard deviation) around the peak time
        background_noise = np.std(
            energy_data[(time_data < peak_time - 0.1) | (time_data > peak_time + 0.1)]
        )
        peak_intensity = np.max(energy_data)
        return peak_intensity / background_noise if background_noise > 0 else np.inf

    def _get_rise_decay_time(
        self,
        time_bins: np.ndarray,
        time_counts: np.ndarray,
        time_data: np.ndarray,
        energy_data: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates rise time and decay time based on the time bins and photon data.
        """
        peak_index = np.argmax(time_counts)
        peak_bin = time_bins[peak_index]

        rise_time = peak_bin - time_data[0]
        decay_time = time_data[-1] - peak_bin

        return rise_time, decay_time

    def _get_duration(self, time_data: np.ndarray) -> float:
        """
        Calculates the duration of the event, defined as the difference between the first and last photon times.
        """
        return time_data[-1] - time_data[0]

    def _get_centroid(self, time_data: np.ndarray, energy_data: np.ndarray) -> float:
        """
        Calculates the centroid of the event, defined as the weighted average of photon times based on energy.
        """
        return np.sum(time_data * energy_data) / np.sum(energy_data)

    def _get_total_energy_released(self, energy_data: np.ndarray) -> float:
        """
        Calculates the total energy released during the event, defined as the sum of the photon energies.
        """
        return np.sum(energy_data)

    def _get_number_of_regions(
        self, energy_data: np.ndarray, peak_intensity: float
    ) -> int:
        """
        Calculates the number of distinct regions where photon intensity exceeds a threshold.
        """
        binary_signal = energy_data > self.threshold * peak_intensity
        dilated_signal = binary_dilation(binary_signal)
        eroded_signal = binary_erosion(binary_signal)
        _, num_regions = label(dilated_signal)
        return num_regions


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
