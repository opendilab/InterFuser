import os
import time
import sys
import json
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import numpy as np


def process(route):
    try:
        frames = len(os.listdir(os.path.join(route, "measurements")))
        if not os.path.exists(os.path.join(route, "rgb_full")):
            os.mkdir(os.path.join(route, "rgb_full"))
        if not os.path.exists(os.path.join(route, "measurements_full")):
            os.mkdir(os.path.join(route, "measurements_full"))
        for i in range(frames):
            img_front = Image.open(os.path.join(route, "rgb_front/%04d.jpg" % i))
            img_left = Image.open(os.path.join(route, "rgb_left/%04d.jpg" % i))
            img_right = Image.open(os.path.join(route, "rgb_right/%04d.jpg" % i))
            new = Image.new(img_front.mode, (800, 1800))
            new.paste(img_front, (0, 0))
            new.paste(img_left, (0, 600))
            new.paste(img_right, (0, 1200))
            new.save(os.path.join(route, "rgb_full", "%04d.jpg" % i))

            measurements = json.load(
                open(os.path.join(route, "measurements/%04d.json" % i))
            )
            actors_data = json.load(
                open(os.path.join(route, "actors_data/%04d.json" % i))
            )
            affordances = np.load(
                os.path.join(route, "affordances/%04d.npy" % i), allow_pickle=True
            )

            measurements["actors_data"] = actors_data
            measurements["stop_sign"] = affordances.item()["stop_sign"]
            json.dump(
                measurements,
                open(os.path.join(route, "measurements_full/%04d.json" % i), "w"),
            )
    except Exception as e:
        print(e)
        print(route)


if __name__ == "__main__":
    list_file = sys.argv[1]
    routes = []
    for line in open(list_file, "r").readlines():
        routes.append(line.strip())
    with Pool(24) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
