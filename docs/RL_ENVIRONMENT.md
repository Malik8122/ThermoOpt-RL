# ThermoOpt-RL — RL Environment (Phase 1, Step 5)

`DataCenterEnv` (`backend/rl/environment.py`, registered as **`ThermoOpt-DataCenter-v0`**) is the Gymnasium
interface to the simulated data-centre digital twin described in [MODEL_CALIBRATION.md](MODEL_CALIBRATION.md).
It contains **no physics**. It issues commands to `DataCentre` and reads back what the simulator produced.

```python
import gymnasium as gym
import backend.rl                                   # registers the environment

env = gym.make("ThermoOpt-DataCenter-v0", episode_minutes=60)                 # DQN / PPO
env = gym.make("ThermoOpt-DataCenter-v0", representation="tabular")          # PI / VI / Q-Learning
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
print(env.unwrapped.explain_reward())
```

## 1. Architecture

| Module | Responsibility |
|---|---|
| `environment.py` | `DataCenterEnv`: Gymnasium API, episode bookkeeping, `info`, render, `explain_reward()` |
| `actions.py` | Action encodings (18 primitive, 9 tabular), read-only validation, macro-action resolution, application |
| `state_encoder.py` | 144-state tabular abstraction |
| `observations.py` | 30-feature normalised observation, plus an optional 12-feature forecast extension |
| `reward.py` | `calculate_reward()`: a pure function returning a `RewardBreakdown` with six components |
| `benchmark_random.py` | Random-policy benchmark |

Each `step(action)`:

1. **Decode.** A tabular macro-action is first resolved to one primitive action for the current state.
2. **Validate.** The check is read-only and atomic. Both the placement part and the cooling part must be valid.
3. **Apply.** If the action is valid, the environment calls `DataCentre.place_workload`, `delay_workload`,
   `migrate_workload` and/or `set_cooling_target`.
4. **Advance.** It calls `DataCentre.step()`. The simulator's own step order is unchanged.
5. **Read outcomes.** Energy, carbon, cost, temperatures and job events come from the `StepRecord`, the accountant
   and the job states. Completed and failed jobs are identified by comparing job-status sets before and after the
   step, so the simulator needs no extra hooks.
6. **Score.** `calculate_reward(RewardInputs, config)` is applied to those outcomes.

The only simulator additions in Step 5 are **commands**:
- `DataCentre.migrate_workload()`, `Rack.receive_migrated_workload()`, `Rack.check_capacity()` and
  `Workload.mark_migrated()`. These are needed for the *migrate* action defined in PHASE1_DESIGN §4.2.
- `CoolingSystem.level_above()` and `level_below()`, read-only queries used to validate cooling commands.

No equation, parameter, step-order detail or random stream was changed. The seed-42 baseline CSV is still
byte-identical (SHA-256 `e7fe90be…cafab`, pinned by `tests/test_rl_reward.py::test_baseline_simulation_unchanged_by_step5`).

## 2. State representations

| `representation=` | Observation space | Action space | Used by |
|---|---|---|---|
| `"features"` (default) | `Box(30,)` float32, or `Box(42,)` with forecasts | `Discrete(18)` | DQN, PPO, Predictive PPO |
| `"tabular"` | `Discrete(144)` | `Discrete(9)` | Policy Iteration, Value Iteration, Q-Learning |

## 3. Tabular abstraction: 144 states

| Factor | Simulator source | Levels (edges from `config.yaml → discretisation`) |
|---|---|---|
| temperature | hottest rack temperature | LOW < 25 ≤ MEDIUM < 27 ≤ HIGH < 30 ≤ CRITICAL |
| workload | used CPU+GPU units / total capacity | LOW < 40 % ≤ MEDIUM < 70 % ≤ HIGH |
| cooling | cooling **target** (set-point) | LOW < 40 % ≤ MEDIUM < 70 % ≤ HIGH |
| carbon | grid carbon intensity now | LOW < 400 ≤ HIGH gCO₂/kWh |
| queue | waiting jobs | EMPTY (0), PENDING (≥ 1) |

$$s = \big(((\,T\cdot 3 + W)\cdot 3 + C)\cdot 2 + K\big)\cdot 2 + Q \in [0, 144)$$

This is implemented by `TabularStateEncoder.encode / decode` and the module functions `encode_tabular_state` /
`decode_tabular_state`. The tests enumerate all 4·3·3·2·2 combinations and check that they map one-to-one onto
0–143. Example: the initial twin at 08:00 (hottest rack 28.1 °C, 461/800 units used, cooling 65 %, carbon 440 g,
empty queue) is HIGH/MEDIUM/MEDIUM/HIGH/EMPTY = **state 90**.

## 4. Tabular actions: 9

`a = p·3 + c`, with p ∈ {0 place on coolest feasible rack, 1 place on least-utilised feasible rack, 2 delay} and
c ∈ {0 decrease, 1 maintain, 2 increase}.

A macro-action resolves to exactly one primitive action:
- *coolest* means the minimum temperature among racks that can host the head job;
- *least-utilised* means the minimum combined CPU+GPU utilisation among those racks;
- ties go to the lowest rack index.

If a job is waiting but no rack can host it, the action is **invalid** (`NO_FEASIBLE_RACK`). With an empty queue,
every placement macro is a no-op.

## 5. Deep-RL actions: 18

`a = placement·3 + cooling`

| a | placement | | cooling (a mod 3) |
|---|---|---|---|
| 0–11 | place the head-of-queue job on Rack ⌊a/3⌋+1 | | 0 decrease one level |
| 12–14 | delay the head job (status DELAYED, moved to the back of the queue) | | 1 maintain |
| 15–17 | migrate one running job from the hottest rack to the coolest cooler rack that fits it | | 2 increase one level |

**Migration rule** (deterministic):
- The source is the hottest rack.
- Its jobs are tried in order of decreasing thermal-load contribution $0.6\,\text{cpu}/100 + 0.4\,\text{gpu}/100$.
- Each job goes to the coolest rack that is *strictly cooler* than the source and has capacity.
- The job keeps its progress. Migration is modelled as instantaneous, and its cost appears in the action penalty.

## 6. Observation: 30 features

| # | Feature | Normalisation | Range |
|---|---|---|---|
| 0–3 | rack temperature | (T − 18) / 22 | [0, 1] |
| 4–7 | rack CPU utilisation | fraction | [0, 1] |
| 8–11 | rack GPU utilisation | fraction | [0, 1] |
| 12 | cooling target | fraction | [0, 1] |
| 13 | delivered cooling | fraction | [0, 1] |
| 14 | queue length | min(1, n / 20) | [0, 1] |
| 15 | head job CPU | / 100 units | [0, 1] |
| 16 | head job GPU | / 100 units | [0, 1] |
| 17 | head job urgency | clip(1 − slack_left / 240, 0, 1) | [0, 1] |
| 18 | head job flexible | 0/1 | {0, 1} |
| 19 | a job is waiting | 0/1 | {0, 1} |
| 20 | electricity price | / ₹9.50 (profile max) | [0, 1] |
| 21 | carbon intensity | / 520 g (profile max) | [0, 1] |
| 22 | carbon trend | (I(t+60 min) − I(t)) / 520 | [−1, 1] |
| 23–24 | time of day | sin, cos(2πh/24) | [−1, 1] |
| 25 | ambient temperature | (T − 18) / 22 | [0, 1] |
| 26–29 | head job fits rack r | 0/1 | {0, 1} |

Head-job features are 0 when the queue is empty. Free capacity is 1 − utilisation, and power is a deterministic
function of utilisation and delivered cooling (MODEL_CALIBRATION §6–7), so both are implicitly represented. The
`Box` bounds are exactly the ranges above, and the tests check `observation_space.contains(obs)` on every step of a
240-minute episode.

## 7. Predictive extension: 42 features (Step 12)

`DataCenterEnv(prediction_provider=f)`, where `f(dc) -> array (4, 3)` of forecast temperatures (°C) for
+5, +10 and +15 min, appends 12 features in rack-major order (`rack1_forecast_t+5`, `…t+10`, `…t+15`,
`rack2_…`). Each is clipped to [18, 40] and normalised like feature 0. The observation space becomes `Box(42,)`.
No forecasting model exists yet. The tests use a *persistence* stand-in (forecast = current temperature) only to
exercise the interface.

## 8. Reward equation

$$
r_t = -\,0.30\,E_t \;-\; 0.20\,C_t \;-\; 0.30\,\Theta_t \;-\; 0.15\,S_t \;-\; \big(0.05\,M_t + 0.1\cdot\mathbb 1[\text{invalid}]\big) \;+\; 0.05\sum_{j\in\text{done}_t}\text{priority}_j
$$

| Component | `info` key | Contribution |
|---|---|---|
| Energy | `reward_energy` | $-0.30\,E_t$ |
| Carbon | `reward_carbon` | $-0.20\,C_t$ |
| Thermal | `reward_thermal` | $-0.30\,\Theta_t$ |
| SLA | `reward_sla` | $-0.15\,S_t$ |
| Action / migration | `reward_action` | $-0.05\,M_t - 0.1\cdot\mathbb 1[\text{invalid}]$ |
| Completion | `reward_completion` | $+0.05\sum \text{priority}$ |

`reward_total` and the returned reward are the exact sum of the six contributions (tested). The weights and
constants are in `config.yaml → reward`. `calculate_reward` is a pure function: identical inputs give an identical
breakdown.

## 9. Normalisation

Let $P_\text{max} = 25.6 + 16 = 41.6$ kW (max IT + max cooling), $I_\text{max} = 520$ gCO₂/kWh and Δt = 1 min.

$$E_t = \frac{E^\text{sim}_t\ [\text{kWh}]}{P_\text{max}\,\Delta t/60} \in [0,1] \qquad
C_t = \frac{\text{CO}_2{}^\text{sim}_t\ [\text{g}]}{P_\text{max}\,\Delta t/60 \cdot I_\text{max}} \in [0,1]$$

$E^\text{sim}_t$ and $\text{CO}_2{}^\text{sim}_t$ are the accountant's values for the step. Carbon is
$E^\text{sim}_t \times I(t)$, computed once, by the simulator (tested against `info`).

$$M_t = \frac{\mathbb 1[\text{cooling target changed}] + N_\text{migrations}}{2} \in [0, 1]$$

## 10. Thermal penalty

$$
\Theta_t = \frac1{4}\sum_r \theta(T_r),\qquad
\theta(T)=\begin{cases}0 & T<27\\[2pt] 1.0\cdot\dfrac{T-27}{3} & 27\le T<30\\[4pt] 10 + 2\,(T-30) & T\ge 30\end{cases}
$$

$T_r$ is the temperature at the *end* of the step. θ rises from 0 to 1 across WARNING, then jumps to 10 at CRITICAL.
**Dominance:** one critical rack contributes $-0.30 \times 10/4 = -0.75$ per step. That is more than the largest
possible energy + carbon penalty ($-0.30 - 0.20 = -0.50$), so no energy saving can compensate for a critical rack.
The test `test_critical_rack_outweighs_any_energy_saving` checks this.

## 11. SLA penalty

$$S_t = \frac{30\cdot N^\text{failed}_t \;+\; 0.5\sum_{j\in\text{started}_t}\dfrac{\text{waited}_j}{\text{slack}_j}}{1}$$

- **Deadline violation** (FAILED: deadline can no longer be met, or rejected because the queue was full) costs
  $0.15 \times 30 = 4.5$ per job.
- **Delayed but on time**: when a job starts, it pays $0.15 \times 0.5 \times$ the fraction of its slack it spent
  waiting. The maximum is 0.075, for a job that used all of its slack. A job started immediately pays nothing.
  Delay is therefore never treated as a failure (tested).
- **Calibration (Step 5).** The first constants (violation 1.0, normalisation 3) made a violation cost only 0.05.
  Running the largest configured job costs up to **4.25** in energy + carbon penalty
  (`reward.max_job_run_cost()`: AI_TRAINING with 50 CPU, 100 GPU, 120 min). An energy-minimising agent would
  therefore have learned to let jobs expire. With 4.5 per violation, missing a deadline never pays off, which
  `test_missing_a_deadline_never_pays_off` checks. The delay penalty was kept small so that shifting flexible work
  into low-carbon hours can still be worth it. The five component *weights* are unchanged.

## 12. Action penalty

A cooling target change or a migration costs 0.05 × ½ each, which discourages oscillating set-points and needless
migrations. An **invalid** action adds −0.1 (`environment.invalid_action_penalty`) and changes nothing in the
simulator.

## 13. Completion reward

$+0.05 \times \text{priority}$ (1–3) for each job **placed by the agent** that completes during the step. Background
jobs, which were already running at reset, are excluded because the agent did not schedule them. Once started, a job
always finishes before its deadline, so every completion is an on-time completion.

## 14. Episode mechanics

- One step is one simulated minute (`simulation.timestep_minutes`).
- `episode_minutes` defaults to 1440 (24 h). Any positive multiple of the timestep works, for example 60, 120 or 240
  for training.
- The data centre never reaches a terminal state, so `terminated` is always `False` and `truncated` becomes `True`
  after `episode_minutes`. Stepping after that raises an error until `reset()` is called.
- The start time comes from `config.simulation.start_hour` (08:00), from the `start_hour=` argument, from
  `reset(options={"start_hour": h})`, or is drawn at random (`randomize_start_hour=True`, whole minutes, from the
  environment's seeded RNG).

## 15. Seeding

- `reset(seed=s)` builds `DataCentre(seed=s)`. The simulator's independent streams (workloads, thermal,
  background) derive from *s*. So `env.reset(seed=42)` starts from exactly the state of the seed-42 baseline.
- `reset()` without a seed draws the next simulator seed from the environment's `np_random`. That makes a sequence
  of unseeded resets reproducible from the first seeded one (tested).
- Identical seed, configuration and action sequence give identical observations, rewards and `info` (tested).
  Different seeds give different trajectories.

## 16. Invalid-action handling

| Reason | When |
|---|---|
| `INSUFFICIENT_CPU` / `INSUFFICIENT_GPU` | target rack lacks free capacity for the head job |
| `NO_FEASIBLE_RACK` | tabular placement, but no rack can host the head job |
| `NO_MIGRATION_CANDIDATE` | no job on the hottest rack fits on a strictly cooler rack |
| `COOLING_AT_MAXIMUM` / `COOLING_AT_MINIMUM` | increase at 100 % / decrease at 0 % |

- Validation is read-only, and a joint action is applied **all or nothing**. An invalid placement with a valid
  cooling change applies *neither*.
- The environment never substitutes another action. `info["primitive_action"]` is always the action that was
  requested.
- Time still advances, and the step is scored with the −0.1 penalty.
- `info["action_valid"]` and `info["invalid_action_reason"]` report the outcome.
- `env.action_masks()` returns the valid actions for the current state. It is tested to agree with `step()` for all
  18 actions and can be used for masked policies.
- Integers outside the action space raise `ValueError`. They are not actions.

## 17. Random-policy benchmark

```powershell
python -m backend.rl.benchmark_random          # 10 episodes x 60 min, seeds 42-51, start 08:00
```

Results are in `experiments/random_benchmark_10x60min.json`, which includes per-episode values and per-component
reward totals. The table shows mean ± sample std over 10 episodes:

| Metric | Uniform over 18 actions | Uniform over valid actions |
|---|---|---|
| Episode reward | −82.54 ± 34.56 | −79.48 ± 47.56 |
| Energy (kWh) | 28.27 ± 3.94 | 28.24 ± 2.61 |
| Carbon (kg CO₂) | 12.26 ± 1.71 | 12.25 ± 1.13 |
| Cost (₹) | 230.05 ± 32.05 | 229.77 ± 21.28 |
| Peak temperature (°C) | 30.37 ± 3.25 | 31.22 ± 2.87 |
| WARNING-or-worse rack-minutes | 51.7 ± 65.3 | 80.5 ± 65.8 |
| CRITICAL rack-minutes | 24.4 ± 38.9 | 31.0 ± 40.8 |
| SLA failures | 7.3 ± 3.6 | 6.1 ± 3.1 |
| Completed jobs (incl. background) | 11.4 ± 1.5 | 13.1 ± 1.5 |
| Invalid-action rate | 0.472 ± 0.090 | 0.000 |

Mean reward by component, uniform / valid-only:
- energy −12.23 / −12.22
- carbon −6.80 / −6.80
- thermal −27.32 / −32.13
- SLA −33.07 / −27.65
- action −3.56 / −1.27
- completion +0.44 / +0.60

The random walk in cooling pushes some episodes into CRITICAL, and missed deadlines are the other large cost.
These numbers are the reference point for later agents. They are not a judgement of the policy.

For context, the Steps 2–4 open-loop reference (first-fit placement, cooling fixed at 65 %, seed 42) used
29.74 kWh and emitted 12.90 kg CO₂ in the same hour with no CRITICAL minutes. That run uses a different dispatcher
and cannot be compared directly with the 10-seed average.
