## Results

### Qualitative Comparison (Visualization)

**Easy Map (n_obs = 8)**  
PPO learns smooth, goal-directed navigation while avoiding obstacles early.  
The rule-based baseline frequently stalls near obstacles due to conservative braking.

- **PPO**
  ![PPO Easy](assets/demo_ppo.gif)
- **Rule-based Baseline**
  ![Baseline Easy](assets/demo_baseline.gif)

**Hard Map (n_obs = 12)**  
With increased obstacle density, PPO remains robust and continues to reach the goal,  
whereas the baseline almost always fails to make progress.

- **PPO**
  ![PPO Hard](assets/demo_ppo_hard.gif)
- **Rule-based Baseline**
  ![Baseline Hard](assets/demo_baseline_hard.gif)

---

### Quantitative Evaluation

We evaluate each policy over **200 episodes** with fixed random seeds.

| Map             | Policy              | Success Rate | Collision Rate | Avg Steps |
| --------------- | ------------------- | -----------: | -------------: | --------: |
| Easy (n_obs=8)  | PPO                 |    **90.0%** |           0.0% |      61.9 |
| Easy (n_obs=8)  | Rule-based Baseline |         0.5% |           0.0% |     248.8 |
| Hard (n_obs=12) | PPO                 |    **86.5%** |           1.5% |      65.9 |
| Hard (n_obs=12) | Rule-based Baseline |         0.0% |           0.0% |     250.0 |

---

### Discussion

On the Easy map, PPO significantly outperforms the reactive rule-based baseline,
achieving a **90% success rate** while reaching the goal in fewer steps.

As obstacle density increases (Hard map), the baseline almost completely fails,
often stalling due to conservative obstacle avoidance heuristics.
In contrast, PPO maintains a high success rate (**86.5%**), demonstrating strong robustness
to environmental complexity.

These results highlight the advantage of learning-based navigation over hand-crafted rules,
especially in cluttered environments where long-horizon planning is required.
