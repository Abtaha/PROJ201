from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


class Photon:
    def __init__(self, energy, time):
        self.energy = energy
        self.time = time

    def __str__(self):
        return f"Photon(energy={self.energy}, time={self.time})"

    def __repr__(self):
        return self.__str__()


class Event:
    def __init__(self, photons, type, name="Unnamed"):
        self.photons = photons
        self.type = type
        self.name = name

    def __str__(self):
        return f"Event(photons={self.photons})"

    def __repr__(self):
        return self.__str__()


class EventList:
    def __init__(self, main, sb):
        self.main = main
        self.sb = sb

    def __str__(self):
        return f"EventList(main={self.main} sb={self.sb})"

    def __repr__(self):
        return self.__str__()


def read_event(filename, type):
    photons = []
    with open(filename, "r") as f:
        df = pd.read_csv(f)

        time = df.iloc[:, 0].to_numpy()
        energy = df.iloc[:, 1].to_numpy()

        for e, t in zip(energy, time):
            photons.append(Photon(e, t))
    return Event(photons, type, filename)


def get_bins(event: Event, bin_size: float = 0.02):
    """
    Group the photons in the event into bins based on their times, with each bin having a size of `bin_size`.

    Parameters:
    - event: Event, the event containing photons
    - bin_size: float, the size of each bin (default is 0.025)

    Returns:
    - bins: list of lists, where each sublist contains photons in a specific time interval
    - bin_edges: array, the edges of the time bins
    """
    # Extract photon times from the event
    photon_times = np.array([photon.time for photon in event.photons])

    # Determine the bin edges based on the time range of the photons
    min_time = np.min(photon_times)
    max_time = np.max(photon_times)

    # Calculate the bin edges starting from min_time, with steps of bin_size
    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

    # Digitize the photon times into bin indices based on the bin edges
    bin_indices = (
        np.digitize(photon_times, bin_edges) - 1
    )  # Subtract 1 to match the index (0-based)

    # Group photons into bins based on their indices
    bins = [[] for _ in range(len(bin_edges) - 1)]
    for photon, bin_index in zip(event.photons, bin_indices):
        bins[bin_index].append(photon)

    return bins, bin_edges


def filter_event(event, threshold):
    """
    Filter the events based on a threshold.

    Parameters:
    - event_list: EventList, the events to filter
    - threshold: float, the threshold to use for filtering

    Returns:
    - filtered_events: EventList, the filtered events
    """
    filtered_events = []
    filtered_photons = [photon for photon in event.photons if photon.energy > threshold]
    if filtered_photons:
        return Event(filtered_photons, event.type)
    return filtered_events


def plot_events_multiple_axes(
    events: list[Event], labels=None, threshold=None, bin_size=0.02
):
    if labels is None:
        labels = [f"Event {events[i].name}" for i in range(len(events))]

    if len(events) != len(labels):
        raise ValueError("The number of labels must match the number of events.")

    fig, axes = plt.subplots(len(events), 1, figsize=(8, 3 * len(events)), sharex=True)
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
            axes[i].axhline(threshold, color="red", linestyle="--", label="Threshold")
        axes[i].set_title(label, fontsize=10)
        axes[i].set_ylabel("Count", fontsize=8)
        axes[i].tick_params(axis="both", labelsize=8)
        axes[i].legend(fontsize=8)

    axes[-1].set_xlabel("Time", fontsize=10)
    plt.tight_layout(pad=1.0, h_pad=0.5)
    plt.show(block=True)  # Keeps the window open


def plot_event(event: Event, threshold=None, bin_size=0.02):
    """
    Plot the events based on a threshold, focusing on photon times.
    Exclude bins with fewer photons than the threshold.

    Parameters:
    - event: Event, the event containing photon data
    - threshold: float, the threshold to use for filtering and plotting (default is None)
    - bin_size: float, the size of the time bins for histogramming (default is 0.02)
    """
    # Extract photon times from the event
    all_times = [photon.time for photon in event.photons]

    # Create the time bins based on the bin size
    time_counts, time_bins = np.histogram(
        all_times,
        bins=np.arange(min(all_times), max(all_times) + bin_size, bin_size),
    )

    # Filter out bins with photon counts less than the threshold
    filtered_bins = (
        time_counts > threshold
        if threshold is not None
        else np.ones_like(time_counts, dtype=bool)
    )

    # Plot only the bins that exceed the threshold
    plt.bar(
        time_bins[:-1][filtered_bins],  # Bin edges for the x-axis
        time_counts[filtered_bins],  # Counts for the y-axis
        width=bin_size,
        color="skyblue",
        edgecolor="black",
    )

    # If a threshold is provided, add a line to indicate the threshold
    if threshold is not None:
        plt.axhline(threshold, color="red", linestyle="--", label="Threshold")

    plt.xlabel("Time")
    plt.ylabel("Number of Photons")
    plt.legend()
    plt.title("Photon Arrival Times (with Threshold)")
    plt.show()


def combine_events(event_list: EventList, threshold: float = 0.0002) -> Event:
    """
    Combine the photons from two events into a single event.
    """
    main_event = event_list.main
    sb_event = event_list.sb

    # Extract photon times from both events
    main_times = np.array([photon.time for photon in main_event.photons])
    sb_times = np.array([photon.time for photon in sb_event.photons])

    if threshold == 0:
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
        matching_indices = np.where(time_diffs <= threshold)[0]

        # Collect the corresponding sb_photons using the matching indices
        close_photons = [sb_event.photons[i] for i in matching_indices]

        if close_photons:  # If any close photons exist, add the main photon
            common_photons.append(main_photon)

    return Event(common_photons, type="combined")


def compute_threshold(event, bin_size=0.02, percentile=10):
    """
    Compute a threshold for detecting bursts in the time data (based on time bins).

    Parameters:
    - event: Event, the event containing photon data
    - bin_size: float, the size of the time bins to use for histogramming (default is 0.02)
    - percentile: float, the percentage of bins to be considered as having "very few" photons (default is 5)

    Returns:
    - threshold: float, the computed threshold for time bins
    """
    # Extract photon times from the event
    all_times = [photon.time for photon in event.photons]

    # Create time bins based on the bin size
    time_counts, time_bins = np.histogram(
        all_times, bins=np.arange(min(all_times), max(all_times) + bin_size, bin_size)
    )

    # Sort the bin counts to identify the sparsest bins
    sorted_counts = np.sort(time_counts)

    # Compute the threshold based on the `percentile` of the sorted photon counts
    # Percentile gives the value below which a given percentage of observations fall
    threshold = np.percentile(sorted_counts, percentile)

    return threshold


def compute_threshold2(event, bin_size=0.02, percentile=10):
    """
    Compute a threshold for detecting bursts in the time data by identifying the bin with the highest photon count
    and setting the threshold to 10% of the maximum count.

    Parameters:
    - event: Event, the event containing photon data
    - bin_size: float, the size of the time bins to use for histogramming (default is 0.02)
    - percentile: float, the percentage of bins to be considered as having "very few" photons (default is 10)

    Returns:
    - threshold: float, the computed threshold for time bins
    """
    # Extract photon times from the event
    all_times = [photon.time for photon in event.photons]

    # Create time bins based on the bin size
    time_counts, time_bins = np.histogram(
        all_times, bins=np.arange(min(all_times), max(all_times) + bin_size, bin_size)
    )

    # Find the bin with the maximum photon count
    max_count = np.max(time_counts)

    # Set the threshold as 10% of the highest bin count
    threshold = max_count * (percentile / 100)

    return threshold


if __name__ == "__main__":
    BIN_SIZE = 0.02

    events_list: list[EventList] = []

    for i in range(1, 13):
        main_event = read_event(f"events/{i}.csv", "main")
        sb_event = read_event(f"events/{i}s.csv", "sb")
        events_list.append(EventList(main_event, sb_event))

    for i, events in enumerate(events_list):
        # threshold = compute_threshold(events.main)
        # filtered = filter_event(events.main, 0.1)
        combined = combine_events(events, 0.0002)
        # combined = events.main
        # threshold = compute_threshold(combined, bin_size=BIN_SIZE, percentile=10)
        threshold = compute_threshold2(combined, bin_size=BIN_SIZE, percentile=10)
        print(threshold)

        # plot_events_multiple_axes([events.main, events.sb, combined], bin_size=BIN_SIZE)
        plot_event(combined, threshold=threshold, bin_size=BIN_SIZE)

    # # Example events
    # event1 = Event([Photon(time, time) for time in np.random.uniform(0, 10, 100)], "Type A")
    # event2 = Event([Photon(time, time) for time in np.random.uniform(5, 15, 150)], "Type B")
    # event3 = Event([Photon(time, time) for time in np.random.uniform(10, 20, 120)], "Type C")

    # # Plot each event on its own axis
    # plot_events_multiple_axes(
    #     [event1, event2, event3],
    #     labels=["Event 1", "Event 2", "Event 3"],
    #     threshold=10,
    #     bin_size=0.5,
    # )

    # for photon in events.main.photons:
    #     print(photon.time)
    # break
    # print(events)
