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

routes = []

with open("haze_routes.txt") as f:
    for line in f.readlines():
        routes.append(line.strip())


def process(route):
    frames = len(os.listdir(os.path.join(route, "measurements")))
    for folder in dt:
        temp = dt[folder]
        files = os.listdir(os.path.join(route, folder))
        fs = []
        for file in files:
            fs.append(int(file[:4]))
        fs.sort()
        for i in range(len(fs)):
            if i == fs[i]:
                continue
            try:
                os.rename(
                    os.path.join(route, folder, temp % fs[i]),
                    os.path.join(route, folder, temp % i),
                )
            except Exception as e:
                print(e)


if __name__ == "__main__":
    with Pool(24) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
