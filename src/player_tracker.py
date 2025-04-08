import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, initial_pos):
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])  # [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.R *= 0.5  # Measurement noise
        self.kf.P *= 10  # Initial covariance
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update(self, pos):
        self.kf.update(pos)
        self.hits += 1
        self.time_since_update = 0

    def get_velocity(self):
        return [self.kf.x[2], self.kf.x[3]]

    def set_velocity(self, avg_velocity):
        # if self.kf.x[2] > avg_velocity or self.kf.x[3] > avg_velocity:
        self.kf.x[2] = avg_velocity
        self.kf.x[3] = avg_velocity


class Tracker:
    def __init__(self, max_age=125, min_hits=25):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.min_hits = min_hits
        self.avg_velocity = 0
        self.velocity_sum = 0
        self.velocity_num = 0

    def update(self, detections):

        # Predict existing tracks
        for track in self.tracks:
            track.predict()
            track.time_since_update += 1

        # Cost matrix using Euclidean distance
        if len(self.tracks) > 0 and len(detections) > 0:
            cost = distance.cdist([track.kf.x[:2] for track in self.tracks], detections, 'euclidean')
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = [], []

        # Update matched tracks
        matched_tracks = set()
        matched_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 3:  # Gating threshold
                self.tracks[r].update(detections[c])
                self.avg_velocity_update(self.tracks[r].get_velocity())
                # print(self.avg_velocity*2)
                self.tracks[r].set_velocity(self.avg_velocity)
                matched_tracks.add(r)
                matched_detections.add(c)

        # Create new tracks for unmatched detections
        for d in range(len(detections)):
            if d not in matched_detections:
                self.tracks.append(Track(self.next_id, detections[d]))
                self.next_id += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age and (t.hits >= self.min_hits or t.age < 10)
        ]

        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def avg_velocity_update(self, velocity):
        self.velocity_sum += np.average(velocity)
        self.velocity_num += 1
        self.avg_velocity = self.velocity_sum / self.velocity_num
