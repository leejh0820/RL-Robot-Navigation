import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env import ObstacleAvoidEnv


def make_env(seed=0, n_obs=8, reward_mode="full"):
    def _thunk():
        env = ObstacleAvoidEnv(seed=seed, n_obs=n_obs, reward_mode=reward_mode)
        return Monitor(env)

    return _thunk


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    experiments = [
        ("full", "full"),
        ("no_progress", "no_progress"),
        ("weak_collision", "weak_collision"),
        ("no_time", "no_time"),
    ]

    seed = 0
    n_obs = 8
    total_timesteps = 250_000

    for exp_name, reward_mode in experiments:
        print(f"\n=== Training: {exp_name} (reward_mode={reward_mode}) ===")

        env = DummyVecEnv([make_env(seed=seed, n_obs=n_obs, reward_mode=reward_mode)])

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.0,
            clip_range=0.2,
        )

        model.learn(total_timesteps=total_timesteps)
        save_path = f"checkpoints/ppo_{exp_name}"
        model.save(save_path)
        print(f"[OK] saved -> {save_path}.zip")
