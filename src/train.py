import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env import ObstacleAvoidEnv


def make_env(seed=0):
    def _thunk():
        env = ObstacleAvoidEnv(seed=seed)
        return Monitor(env)

    return _thunk


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    env = DummyVecEnv([make_env(0)])

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

    model.learn(total_timesteps=250_000)
    model.save("checkpoints/ppo_nav")
    print("[OK] saved -> checkpoints/ppo_nav.zip")
