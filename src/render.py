import os
import imageio.v2 as imageio

from stable_baselines3 import PPO
from env import ObstacleAvoidEnv
from policies import GoalSeekBrake


def rollout(env, policy_fn, max_steps=300):
    frames = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, r, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)

    # 1) PPO policy

    ppo_env = ObstacleAvoidEnv(seed=42, n_obs=12)
    model = PPO.load("checkpoints/ppo_nav")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    frames_ppo = rollout(ppo_env, ppo_policy)
    out_ppo = "assets/demo_ppo_hard.gif"
    imageio.mimsave(out_ppo, frames_ppo, fps=20)
    print(f"[OK] saved -> {out_ppo}")

    # 2) Baseline policy

    base_env = ObstacleAvoidEnv(seed=42, n_obs=12)
    baseline = GoalSeekBrake(n_rays=16, brake_dist=0.2, turn_gain=2.0)

    def baseline_policy(obs):
        return baseline.predict(obs)

    frames_base = rollout(base_env, baseline_policy)
    out_base = "assets/demo_baseline_hard.gif"
    imageio.mimsave(out_base, frames_base, fps=20)
    print(f"[OK] saved -> {out_base}")
