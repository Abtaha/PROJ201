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


if __name__ == "__main__":
    events_list: list[EventList] = []
    x = [22]
    for i in range(5, 82):
        main_event = read_event(f"events/{i}.csv", "main")
        sb_event = read_event(f"events/{i}s.csv", "sb")
        events_list.append(EventList(main_event, sb_event))

    i = 1
    feature_dict = {}
    for events in events_list:
        combined = events.combine_events()
        pulses = combined.split_pulses()

        pulse_counter = 0
        for pulse in pulses:
            pulse_counter += 1
            counts, bins = pulse.get_bins(get_thresholded=False)

            if len(bins) <= 3:
                continue

            feature_dict[i] = pulse.extract_features()

            i += 1

    df = pd.DataFrame.from_dict(feature_dict, orient="index")
    df.to_csv("export_data.csv")
