<!-- hyperparameter_setting.md -->

# Hyperparameter Configuration Reference

All hyperparameters are defined in `source_code/ordered_policy/hyperparameters.py` (and similarly under `source_code/ordered_policy_queuing/`). You can select and tune the following settings:

---

## 1. Core Flags

| Parameter               | Description                                                                                                                             |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `current_problem`       | Simulation to run. Options:                                                                                                             |
|                         | - `inventory_control`                                                                                                                   |
|                         | - `dual_index`                                                                                                                          |
|                         | - `M1L` (refers to the queuing model from `ordered_policy_queuing`)                                                                      |
| `hyper_algo_set`        | List of algorithms to evaluate. By default:                                                                                             |
|                         | `['our_algo', 'random', 'PPO', 'heuristic', 'optimal', 'feedback_graph', 'empirical_hindsight']`. Uncomment and edit to choose which to run. |
| `time_horizon_list`     | List of training horizons (T) to run. You can specify multiple values (e.g. `[100, 1000, 10000, 100000]`).                              |
| `testing_horizon`       | Single horizon length (H) for policy evaluation (e.g. `100000`).                                                                        |
| `exp_repeat_times`      | Number of independent trials (repetitions) per configuration (e.g. `20`).                                                               |
| `testing_flag`          | If `1`, run in testing mode (load existing models / skip training). Recommend `0` for fresh runs.                                       |
| `saving_flag`           | If `1`, save results to disk.                                                                                                           |
| `plotting_flag`         | If `1`, skip experiments and only produce plots from existing results. Default `0`.                                                      |

---

## 2. Feedback Graph Control

| Parameter                    | Description                                                                                                 |
|------------------------------|-------------------------------------------------------------------------------------------------------------|
| `feedbackgraph_update_fraction` | Fraction of Q-values updated per iteration to speed up feedback graph training (e.g. `0.01`).            |
| `bonus_scale_factor`         | Scale factor for the bonus term in the feedback graph algorithm (e.g. `1e-4`).                             |

---

## 3. Problem-Specific Hyperparameter Sets

Three settings are provided via internal helper methods. See **Table 4** of the paper for reference.

### 3.1 Inventory Control (`_inventory_problem_setting`)
```python
{
  'policy_set': [[0, 300]],         # Base-stock policy range
  'time_horizon': 100000,           # Internal training horizon
  'holding_cost': 1,
  'shortage_penalty': 10,
  'purchasing_cost': 0,
  'purchasing_cost_expedit': 0,
  'L': 6,                           # Long-lead lead time
  'l': 0,                           # Short-lead lead time
  'distribution': 'uniform',
  'maximum_demand': 40,
  'discretization_radius_Qlearning': 1,
  'demand_zero_rate': 0.3,
  'H_for_testing': 20               # Internal test horizon override
}
```

### 3.2 Dual-Index (`_dual_index_setting`)
```python
{
  'policy_set': [[0, 6], [0, 6]],   # [[short-term levels], [long-term levels]]
  'time_horizon': 10000,
  'holding_cost': 1,
  'shortage_penalty': 10,
  'purchasing_cost': 0,
  'purchasing_cost_expedit': 0.5,
  'L': 1,
  'l': 0,
  'distribution': 'normal',
  'maximum_demand': 3,
  'discretization_radius_Qlearning': 1,
  'demand_zero_rate': 0.3,
  'H_for_testing': 20
}
```

### 3.3 M1L Queuing Model (`_M1L`)
```python
{
  'policy_set': None,
  'time_horizon': 10000,
  'S': 2,
  'Amax': 3,
  'lambda_rate': 6,
  'lambda_max': 10,
  'mu_rate': 3,
  'mu_max': 10,
  'C': 100
}
```

---

## 4. Additional Notes

- **Discretization Radius for Q-Learning:** The default `discretization_radius_Qlearning` is `1` in the provided settings; a finer discretization (e.g., `0.1`) is possible but **not** recommended unless necessary—refer to `hyperparameters.py`.

- **PPO Hyperparameters:** PPO’s internal hyperparameters (learning rate, clip range, etc.) use Stable-Baselines3 defaults. We do **not** suggest modifying these within this repository; to tune PPO, consult SB3 documentation or adjust settings in `stable_baselines3` directly.

- **Demand Distribution Control:** All demand distributions are parameterized solely by `maximum_demand`. The mapping to actual distributions is implemented in `ic.py` via `demand_func()`, which applies **only** to inventory control and dual-sourcing simulations. The queuing case study (`M1L`) has no distribution ambiguity.

- **BASA Hyperparameter Tuning:** For BASA, please refer to the original authors for hyperparameter settings. Our only modification is in `main3.m` at line 86, where we introduce `clip_and_mask_demand(d, 0, 40, 0.3)`. The third and fourth arguments specify the demand upper bound and the probability of demand being zero, respectively. We also introduced tests on the normal distribution.

---
