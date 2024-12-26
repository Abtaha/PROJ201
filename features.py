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
    events_list: list[EventList] = []
    x = [22]
    for i in range(1, 102):
        main_event = read_event(f"events/{i}.csv", "main")
        sb_event = read_event(f"events/{i}s.csv", "sb")
        events_list.append(EventList(main_event, sb_event))

    i = 1
    feature_dict = {}
    for event_counter, events in enumerate(events_list):
        combined = events.combine_events()
        threshold, sigma, mean = combined.compute_threshold()

        pulses = combined.split_pulses()

        pulse_counter = 1
        for pulse in pulses:
            counts, bins = pulse.get_bins(get_thresholded=False)
            if len(bins) <= 3:
                continue

            print(f"Event: {event_counter+1} | Pulse: {pulse_counter}")
            pulse_counter += 1

            features = pulse.extract_features()
            if features["Rise Time"] == 0:
                first_bin_start, first_bin_end = bins[0], bins[1]
                features["Rise Time"] = combined.rset(first_bin_start, first_bin_end)
            if features["Decay Time"] == 0:
                reversed = bins.reverse()
                last_bin_end, last_bin_start = reversed[0], reversed[1]

            feature_dict[i] = features

            i += 1

    export_features(feature_dict)
    # plot_features(feature_dict)
