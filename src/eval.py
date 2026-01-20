import numpy as np
from stable_baselines3 import PPO
from env import ObstacleAvoidEnv


def run_eval(model, n_episodes=200, seed=123):
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
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_ret += float(r)
            ep_steps += 1

        if info.get("reached", False):
            success += 1
        if info.get("collided", False):
            collision += 1

        steps_list.append(ep_steps)
        ret_list.append(ep_ret)

    out = {
        "episodes": n_episodes,
        "success_rate": success / n_episodes,
        "collision_rate": collision / n_episodes,
        "avg_steps": float(np.mean(steps_list)),
        "avg_return": float(np.mean(ret_list)),
    }
    return out


if __name__ == "__main__":
    model = PPO.load("checkpoints/ppo_nav")
    metrics = run_eval(model, n_episodes=200)
    print(metrics)
