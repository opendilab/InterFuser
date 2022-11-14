import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

routes = []
for i in range(14):
    subs = os.listdir(os.path.join("dataset", "weather-%d" % i, "data"))
    for sub in subs:
        if not os.path.isdir(os.path.join("dataset", "weather-%d" % i, "data", sub)):
            continue
        routes.append(os.path.join("dataset", "weather-%d" % i, "data", sub))


def process(route_dir):
    frames = len(os.listdir(os.path.join(route_dir, "measurements")))
    stop = 0
    max_stop = 0
    last_actors_num = 0
    res = []
    for i in range(frames):
        json_data = json.load(
            open(os.path.join(route_dir, "measurements", "%04d.json" % i))
        )
        actors_data = json.load(
            open(os.path.join(route_dir, "actors_data", "%04d.json" % i))
        )
        actors_num = len(actors_data)
        light = json_data["is_red_light_present"]
        speed = json_data["speed"]
        junction = json_data["is_junction"]
        brake = json_data["should_brake"]
        if speed < 0.1 and len(light) == 0 and brake == 1:
            stop += 1
            max_stop = max(max_stop, stop)
        else:
            if stop >= 10 and actors_num < last_actors_num:
                res.append((route_dir, i, stop))
            stop = 0
        last_actors_num = actors_num
    if stop >= 10:
        res.append((route_dir, frames - 1, stop))
    return res


if __name__ == "__main__":
    with Pool(16) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
    with open("blocked_stat.txt", "w") as f:
        for t in r:
            print(t)
            if len(t) == 0:
                continue
            for tt in t:
                f.write("%s %d %d\n" % tt)
