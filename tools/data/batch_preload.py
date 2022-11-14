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
    if frames < 6:
        print(route_dir)
        return
    if not os.path.exists(os.path.join(route_dir, "waypoints.npy")):
        preload_dicts = {}
        keys = ["gps_x", "gps_y", "theta"]
        for key in keys:
            preload_dicts[key] = []
        for i in range(frames):
            json_data = json.load(
                open(os.path.join(route_dir, "measurements", "%04d.json" % i))
            )
            for key in ["gps_x", "gps_y"]:
                preload_dicts[key].append(json_data[key])
            if np.isnan(json_data["theta"]):
                preload_dicts["theta"].append(0)
            else:
                preload_dicts["theta"].append(json_data["theta"])

        all_waypoints = []
        for i in range(frames):
            waypoints = []
            ego_x = preload_dicts["gps_x"][i]
            ego_y = preload_dicts["gps_y"][i]
            ego_theta = preload_dicts["theta"][i]
            for j in range(6):
                k = min(frames - 1, i + j)
                local_waypoint = transform_2d_points(
                    np.zeros((1, 3)),
                    np.pi / 2 - preload_dicts["theta"][k],
                    -preload_dicts["gps_x"][k],
                    -preload_dicts["gps_y"][k],
                    np.pi / 2 - ego_theta,
                    -ego_x,
                    -ego_y,
                )
                waypoints.append(local_waypoint[0, :2])
            waypoints = np.vstack(waypoints)
            if np.sum(np.isnan(waypoints)) > 0:
                print(route_dir, i)
            all_waypoints.append(waypoints)
        all_waypoints = np.stack(all_waypoints, axis=0)
        np.save(os.path.join(route_dir, "waypoints.npy"), all_waypoints)


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


if __name__ == "__main__":
    with Pool(16) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))
