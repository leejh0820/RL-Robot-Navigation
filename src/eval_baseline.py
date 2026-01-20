import numpy as np
from env import ObstacleAvoidEnv
from policies import GoalSeekBrake


def run_eval(policy, n_episodes=200, seed=123):
    env = ObstacleAvoidEnv(seed=seed)
    success = 0
    collision = 0
    steps_list = []
    ret_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_ret = 0.0
        ep_steps = 0

        while not (done or trunc):
            action = policy.predict(obs)
            obs, r, done, trunc, info = env.step(action)
            ep_ret += float(r)
            ep_steps += 1

        if info.get("reached", False):
            success += 1
        if info.get("collided", False):
            collision += 1

        steps_list.append(ep_steps)
        ret_list.append(ep_ret)

    return {
        "episodes": n_episodes,
        "success_rate": success / n_episodes,
        "collision_rate": collision / n_episodes,
        "avg_steps": float(np.mean(steps_list)),
        "avg_return": float(np.mean(ret_list)),
    }


if __name__ == "__main__":
    policy = GoalSeekBrake(n_rays=16, brake_dist=0.2, turn_gain=2.0)
    print(run_eval(policy))
