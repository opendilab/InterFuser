import os
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

routes = []
SAMPLE_RATE = 0.02
STAT_DISCRETE_KEYS = [
    "command",
    "weather_id",
    "should_brake",
    "should_slow",
    "is_junction",
    "is_vehicle_present",
    "is_bike_present",
    "is_lane_vehicle_present",
    "is_junction_vehicle_present",
    "is_pedestrian_present",
    "is_red_light_present",
    "is_stop_sign_present",
    "stop_sign",
]
STAT_CONTINUOUS_KEYS = ["speed", "steer"]
STAT_KEYS = STAT_CONTINUOUS_KEYS + STAT_DISCRETE_KEYS

for i in range(14):
    subs = os.listdir(os.path.join("dataset", "weather-%d" % i, "data"))
    for sub in subs:
        if not os.path.isdir(os.path.join("dataset", "weather-%d" % i, "data", sub)):
            continue
        routes.append(os.path.join("dataset", "weather-%d" % i, "data", sub))


def process(route_dir):
    frames = len(os.listdir(os.path.join(route_dir, "measurements_full")))
    data = {}
    for key in STAT_KEYS:
        data[key] = []
    for i in range(frames):
        if np.random.random() > SAMPLE_RATE:
            continue
        json_data = json.load(
            open(os.path.join(route_dir, "measurements_full", "%04d.json" % i))
        )
        for key in STAT_KEYS:
            if "present" not in key:
                data[key].append(json_data[key])
            else:
                data[key].append(len(json_data[key]))
    return data


if __name__ == "__main__":
    data = {}
    for key in STAT_DISCRETE_KEYS:
        data[key] = defaultdict(int)
    with Pool(16) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
        for t in r:
            for i in range(len(t["weather_id"])):
                for key in STAT_DISCRETE_KEYS:
                    data[key][t[key][i]] += 1

        for key in STAT_DISCRETE_KEYS:
            print(key)
            print(data[key])
