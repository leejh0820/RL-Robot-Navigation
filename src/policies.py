import numpy as np


class GoalSeekBrake:

    def __init__(self, n_rays=16, brake_dist=0.18, turn_gain=1.0):
        self.n_rays = n_rays
        self.brake_dist = brake_dist
        self.turn_gain = turn_gain

    def predict(self, obs):
        lidar = obs[: self.n_rays]
        sin_d, cos_d = obs[self.n_rays + 2 : self.n_rays + 4]

        w = np.clip(self.turn_gain * sin_d, -1.0, 1.0)

        min_l = float(np.min(lidar))
        v = 1.0
        if min_l < self.brake_dist:
            v = 0.0

        v_raw = 2.0 * v - 1.0
        return np.array([v_raw, w], dtype=np.float32)
