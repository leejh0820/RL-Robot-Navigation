import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def ray_circle_intersect(ray_o, ray_d, c, r, t_max):
    oc = ray_o - c
    b = 2.0 * float(np.dot(oc, ray_d))
    cterm = float(np.dot(oc, oc) - r * r)
    disc = b * b - 4.0 * cterm
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    t = None
    if 0.0 <= t1 <= t_max:
        t = t1
    elif 0.0 <= t2 <= t_max:
        t = t2
    return t


def ray_aabb_intersect(ray_o, ray_d, lo, hi, t_max):
    tmin = 0.0
    tmax_local = t_max
    for i in range(2):
        if abs(ray_d[i]) < 1e-9:
            if ray_o[i] < lo[i] or ray_o[i] > hi[i]:
                return None
        else:
            inv = 1.0 / float(ray_d[i])
            t1 = (lo[i] - ray_o[i]) * inv
            t2 = (hi[i] - ray_o[i]) * inv
            t_near, t_far = (t1, t2) if t1 <= t2 else (t2, t1)
            tmin = max(tmin, float(t_near))
            tmax_local = min(tmax_local, float(t_far))
            if tmin > tmax_local:
                return None
    return tmin if 0.0 <= tmin <= t_max else None


class ObstacleAvoidEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        seed=0,
        world_size=5.0,
        n_obs=8,
        obs_radius_range=(0.18, 0.35),
        robot_radius=0.18,
        n_rays=16,
        lidar_max=3.5,
        dt=0.1,
        v_max=1.0,
        w_max=2.0,
        max_steps=250,
        goal_radius=0.25,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.world_size = float(world_size)
        self.robot_r = float(robot_radius)
        self.n_obs = int(n_obs)
        self.obs_r_lo, self.obs_r_hi = obs_radius_range
        self.n_rays = int(n_rays)
        self.lidar_max = float(lidar_max)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.max_steps = int(max_steps)
        self.goal_r = float(goal_radius)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_dim = self.n_rays + 2 + 2 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.ray_angles = np.linspace(
            -math.pi, math.pi, self.n_rays, endpoint=False
        ).astype(np.float32)
        self.pos = np.zeros(2, dtype=np.float32)
        self.theta = 0.0
        self.vw = np.zeros(2, dtype=np.float32)
        self.obstacles = []
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.prev_goal_dist = None

    def _sample_free_point(self, margin):
        for _ in range(2000):
            p = self.rng.uniform(margin, self.world_size - margin, size=(2,)).astype(
                np.float32
            )
            ok = True
            for c, r in self.obstacles:
                if np.linalg.norm(p - c) <= (r + margin):
                    ok = False
                    break
            if ok:
                return p
        return self.rng.uniform(margin, self.world_size - margin, size=(2,)).astype(
            np.float32
        )

    def _spawn_obstacles(self):
        self.obstacles = []
        lo = np.array([0.0, 0.0], dtype=np.float32)
        hi = np.array([self.world_size, self.world_size], dtype=np.float32)
        for _ in range(self.n_obs):
            r = float(self.rng.uniform(self.obs_r_lo, self.obs_r_hi))
            margin = r + self.robot_r + 0.2
            c = self.rng.uniform(margin, self.world_size - margin, size=(2,)).astype(
                np.float32
            )
            good = True
            for c2, r2 in self.obstacles:
                if np.linalg.norm(c - c2) < (r + r2 + 0.35):
                    good = False
                    break
            if not good:
                continue
            self.obstacles.append((c, r))

        self.aabb_lo = lo
        self.aabb_hi = hi

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._spawn_obstacles()

        self.pos = self._sample_free_point(margin=self.robot_r + 0.2)
        self.theta = float(self.rng.uniform(-math.pi, math.pi))
        self.vw[:] = 0.0

        for _ in range(2000):
            g = self._sample_free_point(margin=self.robot_r + 0.2)
            if np.linalg.norm(g - self.pos) > 2.0:
                self.goal = g
                break
        self.step_count = 0
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
        obs = self._get_obs()
        info = {}
        return obs, info

    def _lidar(self):
        o = self.pos
        hits = np.full((self.n_rays,), self.lidar_max, dtype=np.float32)

        for i, a in enumerate(self.ray_angles):
            ang = self.theta + float(a)
            d = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)

            best_t = self.lidar_max
            t_wall = ray_aabb_intersect(
                o, d, self.aabb_lo, self.aabb_hi, self.lidar_max
            )
            tx = self.lidar_max
            ty = self.lidar_max
            if abs(d[0]) > 1e-9:
                if d[0] > 0:
                    tx = (self.aabb_hi[0] - o[0]) / d[0]
                else:
                    tx = (self.aabb_lo[0] - o[0]) / d[0]
            if abs(d[1]) > 1e-9:
                if d[1] > 0:
                    ty = (self.aabb_hi[1] - o[1]) / d[1]
                else:
                    ty = (self.aabb_lo[1] - o[1]) / d[1]
            t_exit = min(tx, ty)
            if 0.0 <= t_exit <= best_t:
                best_t = float(t_exit)

            for c, r in self.obstacles:
                t = ray_circle_intersect(o, d, c, r + self.robot_r, best_t)
                if t is not None and t < best_t:
                    best_t = float(t)

            hits[i] = best_t

        return hits

    def _check_collision(self):

        if (self.pos[0] - self.robot_r) < 0.0 or (
            self.pos[0] + self.robot_r
        ) > self.world_size:
            return True
        if (self.pos[1] - self.robot_r) < 0.0 or (
            self.pos[1] + self.robot_r
        ) > self.world_size:
            return True

        for c, r in self.obstacles:
            if np.linalg.norm(self.pos - c) <= (self.robot_r + r):
                return True
        return False

    def _get_obs(self):
        lidar = self._lidar() / self.lidar_max
        goal_vec = (self.goal - self.pos).astype(np.float32)
        goal_dist = float(np.linalg.norm(goal_vec) + 1e-8)

        R = rot2d(-self.theta)
        goal_rf = (R @ goal_vec).astype(np.float32)
        goal_rf_n = np.clip(goal_rf / self.world_size, -1.0, 1.0)

        goal_ang = math.atan2(goal_vec[1], goal_vec[0])
        dth = wrap_pi(goal_ang - self.theta)
        sc = np.array([math.sin(dth), math.cos(dth)], dtype=np.float32)

        vw_n = np.array(
            [self.vw[0] / self.v_max, self.vw[1] / self.w_max], dtype=np.float32
        )
        vw_n = np.clip(vw_n, -1.0, 1.0)

        obs = np.concatenate([lidar.astype(np.float32), goal_rf_n, sc, vw_n], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        self.step_count += 1
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        v = (a[0] + 1.0) * 0.5 * self.v_max
        w = a[1] * self.w_max
        self.vw[:] = (v, w)
        self.pos[0] += float(v * math.cos(self.theta) * self.dt)
        self.pos[1] += float(v * math.sin(self.theta) * self.dt)
        self.theta = wrap_pi(self.theta + float(w * self.dt))

        collided = self._check_collision()
        goal_dist = float(np.linalg.norm(self.goal - self.pos))
        reached = goal_dist < self.goal_r
        truncated = self.step_count >= self.max_steps
        terminated = collided or reached

        progress = float(self.prev_goal_dist - goal_dist)
        r = 1.0 * progress
        r += -0.01
        r += -0.001 * abs(float(w))
        if reached:
            r += 10.0
        if collided:
            r += -10.0

        self.prev_goal_dist = goal_dist
        obs = self._get_obs()
        info = {"goal_dist": goal_dist, "reached": reached, "collided": collided}
        return obs, float(r), bool(terminated), bool(truncated), info

    def render(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import numpy as np
        import math

        fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for c, r in self.obstacles:
            ax.add_patch(Circle((c[0], c[1]), r, fill=False))

        ax.add_patch(Circle((self.goal[0], self.goal[1]), self.goal_r, fill=False))
        ax.add_patch(Circle((self.pos[0], self.pos[1]), self.robot_r, fill=False))

        hx = self.pos[0] + self.robot_r * 1.5 * math.cos(self.theta)
        hy = self.pos[1] + self.robot_r * 1.5 * math.sin(self.theta)
        ax.plot([self.pos[0], hx], [self.pos[1], hy])

        fig.canvas.draw()

        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[..., :3].copy()

        plt.close(fig)
        return img
