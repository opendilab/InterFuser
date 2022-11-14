import numpy as np
import math
from team_code.render import find_peak_box

reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw


class TrackedObject():
    def __init__(self):
        self.last_step = 0
        self.last_pos = [0, 0]
        self.historical_pos = []
        self.historical_steps = []
        self.historical_features = []

    def update(self, step, object_info):
        self.last_step = step
        self.last_pos = object_info[:2]
        self.feature = object_info[2]

        self.historical_pos.append(self.last_pos)
        self.historical_steps.append(step)
        self.historical_features.append(self.feature)


class Tracker():
    def __init__(self, frequency=10):
        self.tracks = []
        self.alive_ids = []
        self.frequency = frequency

    def convert_grid_to_xy(self, i, j):
        x = j - 9.5
        y = 17.5 - i
        return x, y

    def update_and_predict(self, det_data, pos, theta, step, merge_precent=0.4):
        det_data = det_data * reweight_array
        box_ids, box_info = find_peak_box(det_data)
        objects_info = []
        R = np.array(
        [[np.cos(-theta), -np.sin(-theta)],
         [np.sin(-theta), np.cos(-theta)]])

        for poi in box_ids:
            i, j = poi
            center_y, center_x = self.convert_grid_to_xy(i, j)
            center_x = center_x + det_data[i,j,1]
            center_y = center_y + det_data[i,j,2]

            loc = R.T.dot(np.array([center_x, center_y]))
            objects_info.append([loc[0]+pos[0], loc[1]+pos[1], det_data[poi][1:]])

        updates_ids = self._update(objects_info, step)
        speed_results, speed_confidence, heading_results, heading_confidence = self._predict(updates_ids)


        for k, poi in enumerate(box_ids):
            i, j = poi
            if heading_results[k] is not None:
                factor = merge_precent * heading_confidence[k] * 0.1
                heading_results[k] = (heading_results[k] + theta / np.pi) % 2
                det_data[i,j,3] = heading_results[k] * factor + det_data[i,j,3] * (1- factor)
            if speed_results[k] is not None:
                factor = merge_precent * speed_confidence[k] * 0.1
                det_data[i,j,6] = speed_results[k] * factor + det_data[i,j,3] * (1- factor)
        det_data = det_data / reweight_array
        return det_data

    def _update(self, objects_info, step):
        latest_ids = []
        if len(self.tracks) == 0:
            for object_info in objects_info:
                to = TrackedObject()
                to.update(step, object_info)
                self.tracks.append(to)
                self.alive_ids.append(len(self.tracks)-1)
                latest_ids.append(len(self.tracks)-1)
        else:
            to_ids = self._match(objects_info)
            for i, to_id in enumerate(to_ids):
                if to_id == -1:
                    to = TrackedObject()
                    self.tracks.append(to)
                    to_id = len(self.tracks) - 1
                    self.alive_ids.append(to_id)
                latest_ids.append(to_id)
                self.tracks[to_id].update(step, objects_info[i])

        self.alive_ids = [x for x in self.alive_ids if self.tracks[x].last_step > step -6]
        return latest_ids


    def _match(self, objects_info):
        results = []
        matched_ids = []
        for object_info in objects_info:
            min_id, min_error = -1, 0x7FFFFFFF
            pos_x, pos_y = object_info[:2]
            for _id in self.alive_ids:

                if _id in matched_ids:
                    continue
                track_pos = self.tracks[_id].last_pos
                distance = np.sqrt((track_pos[0]-pos_x)**2 + (track_pos[1]-pos_y)**2)
                if distance < min_error:
                    min_error = distance
                    min_id = _id

            if min_error > 2:
                results.append(-1)
            else:
                results.append(min_id)
                matched_ids.append(min_id)
        return results


    def _predict(self, updates_ids):
        speed_results = []
        heading_results = []
        speed_confidence = []
        heading_confidence = []
        for each_id in updates_ids:
            to = self.tracks[each_id]
            avg_speed = []
            avg_heading = []
            speed_data_point = 0
            heading_data_point = 0
            for i in range(len(to.historical_steps)-3, -1, -1):
                if to.historical_steps[i] < to.last_step - self.frequency - 1 - 1:
                    break
                speed_data_point += 1
                prev_pos, cur_pos = to.historical_pos[i], to.historical_pos[i+2]
                speed = 0.5 * self.frequency * np.sqrt((prev_pos[0]-cur_pos[0])**2+(prev_pos[1]-cur_pos[1])**2) / (to.historical_steps[i+1]-to.historical_steps[i])
                speed = np.clip(speed-0.2, 0, 7)
                avg_speed.extend([speed]*(to.historical_steps[i+1]-to.historical_steps[i]))
                if speed > 1:
                    heading_data_point += 1
                    heading = np.array([cur_pos[0]-prev_pos[0], cur_pos[1]-prev_pos[1]])
                    heading = heading / np.linalg.norm(heading)
                    avg_heading.append(heading)

            if len(avg_speed) == 0:
                speed_results.append(None)
                speed_confidence.append(None)
            else:
                speed_results.append(np.mean(avg_speed))
                speed_confidence.append(np.sqrt(speed_data_point / 10))

            if len(avg_heading) == 0:
                heading_results.append(None)
                heading_confidence.append(None)
            else:
                avg_heading = np.mean(np.stack(avg_heading, 0), 0)
                avg_heading = (4 - get_yaw_angle(avg_heading) / np.pi) % 2
                heading_results.append(avg_heading)
                heading_confidence.append(np.sqrt(heading_data_point / 10))

        return speed_results, speed_confidence, heading_results, heading_confidence
