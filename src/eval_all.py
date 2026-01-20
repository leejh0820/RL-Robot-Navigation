import json
import numpy as np

from stable_baselines3 import PPO
from env import ObstacleAvoidEnv
from policies import GoalSeekBrake


def eval_policy(env, policy_fn, n_episodes=200, seed=123):
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
        last_info = {}

        while not (done or trunc):
            action = policy_fn(obs)
            obs, r, done, trunc, info = env.step(action)
            ep_ret += float(r)
            ep_steps += 1
            last_info = info

        if last_info.get("reached", False):
            success += 1
        if last_info.get("collided", False):
            collision += 1

        steps_list.append(ep_steps)
        ret_list.append(ep_ret)

    return {
        "episodes": int(n_episodes),
        "success_rate": float(success / n_episodes),
        "collision_rate": float(collision / n_episodes),
        "avg_steps": float(np.mean(steps_list)),
        "avg_return": float(np.mean(ret_list)),
    }


def fmt_pct(x):
    return f"{100.0 * x:.1f}%"


def make_md_table(rows):
    header = (
        "| Map | Policy | Success | Collision | Avg Steps |\n|---|---|---:|---:|---:|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['map']} | {r['policy']} | {fmt_pct(r['success_rate'])} | "
            f"{fmt_pct(r['collision_rate'])} | {r['avg_steps']:.1f} |\n"
        )
    return header + body


if __name__ == "__main__":
    N_EP = 200
    SEED = 123

    model = PPO.load("checkpoints/ppo_nav")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    baseline = GoalSeekBrake(n_rays=16, brake_dist=0.2, turn_gain=2.0)

    def base_policy(obs):
        return baseline.predict(obs)

    configs = [
        ("Easy (n_obs=8)", 8),
        ("Hard (n_obs=12)", 12),
    ]

    rows = []
    raw = {}

    for map_name, n_obs in configs:
        env = ObstacleAvoidEnv(seed=SEED, n_obs=n_obs)

        res_ppo = eval_policy(env, ppo_policy, n_episodes=N_EP, seed=SEED)
        rows.append({"map": map_name, "policy": "PPO", **res_ppo})

        env2 = ObstacleAvoidEnv(seed=SEED, n_obs=n_obs)
        res_base = eval_policy(env2, base_policy, n_episodes=N_EP, seed=SEED)
        rows.append({"map": map_name, "policy": "Rule-based Baseline", **res_base})

        raw[map_name] = {"ppo": res_ppo, "baseline": res_base}

    md = make_md_table(rows)
    print("\n=== Markdown Table (copy into README) ===\n")
    print(md)

    out = {"episodes": N_EP, "seed": SEED, "results": raw}
    with open("assets/eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[OK] saved -> assets/eval_results.json")
