# RL Robot Navigation with LiDAR

This project implements a 2D single-robot navigation environment with LiDAR sensing,
where a mobile robot must reach a target while avoiding obstacles.

We train a reinforcement learning policy using **Proximal Policy Optimization (PPO)**
and compare it against a **rule-based reactive baseline** under varying obstacle density.
Experimental results show that the learned PPO policy achieves over **90% success rate**
on simple maps and maintains **86%+ success rate** in cluttered environments,
while the rule-based baseline fails to generalize as complexity increases.

---

## Overview

Autonomous navigation is a fundamental problem in robotics, requiring agents to make
long-horizon decisions under partial observability.
In this project, we study single-robot navigation in a continuous 2D environment using
LiDAR-based perception.

The robot receives distance measurements from a fixed number of LiDAR rays and must
control its linear and angular velocity to reach a goal while avoiding collisions with
walls and circular obstacles.

We compare:
- a **learned policy** trained with PPO, and
- a **hand-crafted rule-based baseline** based on goal-seeking and obstacle braking,

to highlight the advantages of learning-based navigation over reactive heuristics.

---

## Environment & Setup

### Environment
- 2D bounded square world
- Circular robot with unicycle dynamics
- Circular obstacles with random positions and sizes
- Goal location sampled at a sufficient distance from the robot

### Observation Space
Each observation consists of:
- **LiDAR distances** (16 rays, normalized)
- **Goal position** in the robot’s local frame
- **Heading error** to the goal (sin, cos)
- **Previous control inputs** (linear and angular velocity)

### Action Space
- Continuous action: `[v, ω]`
  - Linear velocity `v ∈ [0, v_max]`
  - Angular velocity `ω ∈ [-ω_max, ω_max]`

### Reward Function
The reward is designed to encourage progress while penalizing unsafe behavior:
- Positive reward for reducing distance to the goal
- Large positive reward for reaching the goal
- Large negative penalty for collision
- Small time penalty to discourage stalling
- Small angular velocity penalty to reduce spinning

### Training
- Algorithm: **Proximal Policy Optimization (PPO)**
- Library: Stable-Baselines3
- Training steps: 250k timesteps
- Deterministic evaluation policy

---

## Results

### Qualitative Comparison (Visualization)

**Easy Map (n_obs = 8)**  
PPO learns smooth, goal-directed navigation and proactively avoids obstacles.  
In contrast, the rule-based baseline frequently stalls due to conservative braking behavior.

- **PPO**
  ![PPO Easy](assets/demo_ppo.gif)
- **Rule-based Baseline**
  ![Baseline Easy](assets/demo_baseline.gif)

**Hard Map (n_obs = 12)**  
As obstacle density increases, PPO remains robust and continues to reach the goal reliably.  
The rule-based baseline, however, almost always fails to make progress in cluttered regions.

- **PPO**
  ![PPO Hard](assets/demo_ppo_hard.gif)
- **Rule-based Baseline**
  ![Baseline Hard](assets/demo_baseline_hard.gif)

---

### Quantitative Evaluation

Each policy is evaluated over **200 episodes** with fixed random seeds.

| Map             | Policy              | Success Rate | Collision Rate | Avg Steps |
| --------------- | ------------------- | -----------: | -------------: | --------: |
| Easy (n_obs=8)  | PPO                 | **90.0%**    | 0.0%           | 61.9      |
| Easy (n_obs=8)  | Rule-based Baseline | 0.5%         | 0.0%           | 248.8     |
| Hard (n_obs=12) | PPO                 | **86.5%**    | 1.5%           | 65.9      |
| Hard (n_obs=12) | Rule-based Baseline | 0.0%         | 0.0%           | 250.0     |

---

## Discussion

On the Easy map, PPO significantly outperforms the reactive rule-based baseline,
achieving a **90.0% success rate** while reaching the goal in substantially fewer steps.

As obstacle density increases in the Hard map, the baseline almost completely fails,
often stalling near obstacles due to overly conservative heuristics.
In contrast, PPO maintains a high success rate (**86.5%**), demonstrating strong robustness
and generalization to more complex environments.

These results highlight the advantage of learning-based navigation over hand-crafted rules,
particularly in cluttered settings that require long-horizon decision making.

---

## Future Work

This project can be extended in several directions:
- **Trajectory visualization** to better analyze policy behavior
- **Reward ablation studies** to understand the contribution of each reward term
- **Curriculum learning**, gradually increasing obstacle density during training
- **Multi-robot navigation** and collision avoidance
- Integration with **ROS / real-world simulators** for sim-to-real transfer

---
