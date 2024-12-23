import json
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from Event.Event import Event, EventList, Photon


def read_event(filename, type):
    photons = []
    with open(filename, "r") as f:
        df = pd.read_csv(f)

        time = df.iloc[:, 0].to_numpy()
        energy = df.iloc[:, 1].to_numpy()

        for e, t in zip(energy, time):
            photons.append(Photon(e, t))
    return Event(photons, type, filename)


def export_features(feature_dict: dict) -> None:
    df = pd.DataFrame.from_dict(feature_dict, orient="index")
    df.to_csv("export_data.csv")


def plot_features(events_features: dict) -> None:
    # For every frature in existing features
    for i, feature in enumerate(events_features[1].keys()):
        event_ids = [event_id for event_id, features in events_features.items()]
        values = [features[feature] for event_id, features in events_features.items()]
        plt.bar(
            event_ids,  # Bin edges for the x-axis
            values,  # Counts for the y-axis
            width=0.5,
            color="coral",
            edgecolor="black",
        )

        plt.xlabel("Event ID")
        # ylabel = feature
        # if Event.features[feature] != None:
        #     unit = Event.features[feature]
        #     ylabel += f" ({unit})"
        plt.xticks(range(1, len(event_ids) + 1))
        # plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"Comparison of Events based on {feature}")
        plt.show()


if __name__ == "__main__":
    BIN_SIZE = 0.02

    events_list: list[EventList] = []
    x = [22]
    for i in range(1, 102):
        main_event = read_event(f"events/{i}.csv", "main")
        sb_event = read_event(f"events/{i}s.csv", "sb")
        events_list.append(EventList(main_event, sb_event))

    i = 1
    feature_dict = {}
    for events in events_list:
        combined = events.combine_events()
        threshold, sigma, mean = combined.compute_threshold()

        pulses = combined.split_pulses()

        pulse_counter = 0
        for pulse in pulses:
            pulse_counter += 1
            counts, bins = pulse.get_bins(get_thresholded=False)

            if len(bins) <= 3:
                continue

            # pulse.plot_event()

            features = pulse.extract_features()
            feature_dict[i] = features

            if features["Rise Time"] < 0.001:
                pulse.plot_event()
            i += 1

    # export_features(feature_dict)
    # plot_features(feature_dict)
