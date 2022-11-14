import sys
import os
import json
import random
from tqdm import tqdm

src_dir = sys.argv[1]
target_dir = sys.argv[2]

# Move data(data & result) to dataset dir
for w in range(14):
    subs = os.listdir("%s/weather-%d/data" % (src_dir, w))
    for sub in subs:
        route_dir = os.path.join("%s/weather-%d" % (src_dir, w), "data", sub)
        if not os.path.isdir(route_dir):
            continue
        dst_dir = os.path.join("%s/weather-%d" % (target_dir, w), "data")
        os.system("mv %s %s" % (route_dir, dst_dir))
        print("mv %s %s" % (route_dir, dst_dir))

for w in range(14):
    subs = os.listdir("%s/weather-%d/results" % (src_dir, w))
    for sub in subs:
        route_dir = os.path.join("%s/weather-%d" % (src_dir, w), "results", sub)
        if os.path.isdir(route_dir):
            continue
        dst_dir = os.path.join("%s/weather-%d" % (target_dir, w), "results")
        os.system("mv %s %s" % (route_dir, dst_dir))
        print("mv %s %s" % (route_dir, dst_dir))
