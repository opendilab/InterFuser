import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

dt = {}
dt["topdown"] = "%04d.jpg"
dt["seg_right"] = "%04d.jpg"
dt["seg_left"] = "%04d.jpg"
dt["seg_front"] = "%04d.jpg"
dt["rgb_right"] = "%04d.jpg"
dt["rgb_left"] = "%04d.jpg"
dt["rgb_front"] = "%04d.jpg"
dt["rgb_rear"] = "%04d.jpg"
dt["depth_right"] = "%04d.jpg"
dt["depth_left"] = "%04d.jpg"
dt["depth_front"] = "%04d.jpg"
dt["measurements"] = "%04d.json"
dt["lidar"] = "%04d.npy"
dt["birdview"] = "%04d.jpg"
dt["affordances"] = "%04d.npy"
dt["actors_data"] = "%04d.json"
dt["3d_bbs"] = "%04d.npy"
dt["2d_bbs_left"] = "%04d.npy"
dt["2d_bbs_right"] = "%04d.npy"
dt["2d_bbs_front"] = "%04d.npy"

tasks = []
with open("haze_stat.txt") as f:
    for line in f.readlines():
        line = line.split()
        tasks.append([line[0], int(line[1]), int(line[2])])


def process(task):
    route_dir, end_id, length = task
    frames = len(os.listdir(os.path.join(route_dir, "measurements")))
    for i in range(end_id - length + 6, end_id - 3):
        for key in dt:
            os.remove(os.path.join(route_dir, key, dt[key] % i))


if __name__ == "__main__":
    with Pool(16) as p:
        r = list(tqdm(p.imap(process, tasks), total=len(tasks)))
