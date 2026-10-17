# ThermoOpt-RL

**AI-driven data-centre cooling, workload optimisation and carbon-aware scheduling using reinforcement learning**
(B.Tech AI/ML final-year project, Phase 1)

> ThermoOpt-RL runs on a **simulated digital twin** of a small data centre. It is not connected to any real
> facility. All grid prices and carbon intensities are synthetic.

## Objective

Build a simulated data-centre digital twin in which reinforcement learning dynamically optimises workload
scheduling, workload placement and cooling control. The goal is to minimise energy consumption and carbon emissions
while keeping racks within thermal limits and meeting job deadlines (SLAs).

Phase 1 covers:

1. **Carbon-aware workload shifting**: delay flexible jobs into low-carbon periods.
2. **Thermal-aware workload placement**: place and migrate jobs away from hotspots.
3. **Predictive cooling + RL**: an LSTM forecasts rack temperatures and PPO acts before a rack goes critical.

Algorithms: Random, Round-robin and Rule-based baselines; Policy Iteration; Value Iteration; Q-Learning; DQN;
PPO; Predictive PPO.

The full design is in **[docs/PHASE1_DESIGN.md](docs/PHASE1_DESIGN.md)**: architecture, thermal/power/cooling
equations and their calibration, state and action spaces, reward function, evaluation protocol, API and roadmap.

## Status

| Step | Component | Status |
|---|---|---|
| 1 | Architecture, configuration, reproducibility utilities | ✅ done |
| 2–4 | Digital twin: racks, workloads, thermal / power / cooling models | ✅ done |
| 5 | Gymnasium environment + reward | ✅ done |
| 6 | Baseline controllers | ✅ done |
| 7 | Policy Iteration | ✅ done |
| 8 | Value Iteration | ✅ done |
| 9 | Q-Learning | ✅ done |
| 10 | DQN | ✅ done |
| 11 | PPO | ✅ done |
| 12 | LSTM temperature prediction | ✅ done |
| 13 | Predictive PPO | ⏳ next |
| 14 | Evaluation framework | — |
| 15–17 | FastAPI backend, React dashboard | ✅ done (built ahead of Step 13) |
| 18 | End-to-end testing | — |

This README grows with each step. Sections for the state space, action space, reward, each algorithm, the
frontend and the API will be filled in as those parts are implemented.

## Digital twin (Steps 2–4)

`backend/simulation/` contains the simulated facility. Each module has one job:

| Module | Responsibility |
|---|---|
| `datacentre.py` | `DataCentre`: composes everything below and advances time with `step()` |
| `rack.py` | `Rack`: integer CPU/GPU accounting, never over-allocates, validates temperature and cooling |
| `hardware.py` | `Server`, `GPUNode` |
| `workload.py` | `Workload` record and its legal status transitions |
| `workload_generator.py` | Poisson arrivals (constant or time-varying), per-type resource and duration distributions, background jobs |
| `power.py` | Utilisation-dependent device power |
| `cooling.py` | Shared cooling plant: discrete targets, 3-minute response ramp, electrical power |
| `thermal.py` | Rack temperature dynamics, ambient profile, NORMAL / WARNING / CRITICAL zones |
| `grid.py`, `profiles.py` | Synthetic electricity price and carbon-intensity profile |
| `accounting.py` | Energy (kWh), carbon (gCO₂) and cost (₹), per step and cumulative |

Equations, units, calibration and the baseline results are in
**[docs/MODEL_CALIBRATION.md](docs/MODEL_CALIBRATION.md)**.

Run the 60-minute reference simulation (seed 42, no RL, first-fit placement, cooling fixed at 65 %):

```powershell
python -m backend.simulation.run_baseline                 # -> data/baseline_60min.csv
python -m backend.simulation.run_baseline --minutes 1440 --out data/baseline_24h.csv
```

## Project structure

```
ThermoOpt/
├── config.yaml          # every tunable parameter (single source of truth)
├── requirements.txt
├── docs/                # design documentation
├── backend/
│   ├── config.py        # validated config loader
│   ├── api/             # FastAPI routers
│   ├── models/          # API request/response schemas
│   ├── simulation/      # digital twin
│   ├── rl/              # Gymnasium environment, reward, baselines and RL agents
│   ├── prediction/      # LSTM temperature forecasting
│   ├── evaluation/      # experiments and metrics
│   └── utils/           # seeding, logging
├── frontend/            # React + Vite + Tailwind dashboard
├── data/                # generated datasets
├── models/              # trained agents / predictors
├── experiments/         # results and plots
├── notebooks/
└── tests/
```

## Installation

Requires Python 3.11+ (tested with 3.13) and Node.js 18+ (for the frontend, from Step 16).

```powershell
cd ThermoOpt
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

# CPU-only PyTorch (smaller download). Skip this line if you have a CUDA GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Configuration

Every parameter lives in [`config.yaml`](config.yaml): rack layout and initial state, thermal / cooling / power
coefficients, workload distributions, the grid price and carbon profile, reward weights, discretisation bins,
RL hyperparameters, LSTM settings, evaluation seeds and API settings. The file is validated on load: typos,
wrong list lengths, probabilities that don't sum to 1 and similar mistakes raise an error immediately.

```powershell
python -m backend.config     # print a summary of the active configuration
```

To run with a different configuration file:

```powershell
$env:THERMOOPT_CONFIG = "configs\my_experiment.yaml"
```

Default random seed: **42**. Every stochastic component gets its own seeded generator, so runs are reproducible.

## Tests

```powershell
python -m pytest -q
```

## RL environment (Step 5)

`ThermoOpt-DataCenter-v0` is a Gymnasium environment over the digital twin, with two representations:
- 30 normalised features and 18 actions, for DQN and PPO;
- 144 discrete states and 9 macro-actions, for Policy Iteration, Value Iteration and Q-Learning.

The reward has six parts (energy, carbon, thermal, SLA, action, completion), and `env.unwrapped.explain_reward()`
explains each one. Full specification: **[docs/RL_ENVIRONMENT.md](docs/RL_ENVIRONMENT.md)**.

```python
import gymnasium as gym, backend.rl
env = gym.make("ThermoOpt-DataCenter-v0", episode_minutes=60)
obs, info = env.reset(seed=42)
```

```powershell
python -m backend.rl.benchmark_random      # random-policy benchmark -> experiments/random_benchmark_10x60min.json
```

## Baseline controllers (Step 6)

`backend/rl/baselines.py` provides three reference controllers: **random** (uniform over valid actions),
**round-robin** (cycle racks, cooling held at 60 %) and **rule-based** (carbon-aware delay, placement on the
least-utilised cool rack, threshold cooling). Like every later agent, each exposes
`reset(seed)` and `act(observation, action_mask) -> action`, and reads only the observation.

```powershell
python -m backend.evaluation.run_baselines     # 5 seeds x 24 h -> experiments/baselines_5x1440min.json
```

| 5 × 24 h | Random | Round-robin | Rule-based |
|---|---|---|---|
| Reward | −1588 | −669 | −688 |
| Energy (kWh) | 557 | 614 | 567 |
| Carbon (kg) | 240 | 263 | 239 |
| Critical rack-minutes | 717 | 26 | 25 |
| SLA failures | 96 | 56 | 63 |

Rules, thresholds, the full metric table and the findings are in **[docs/BASELINES.md](docs/BASELINES.md)**.

## Policy Iteration (Step 7)

Classical tabular Policy Iteration (γ = 0.95, θ = 1e-6) on the 144-state / 9-action tabular environment. Its model
P(s'|s,a), R(s,a,s') is **estimated from the real environment**. Concrete simulator states are collected under a
random policy, and 100 reseeded one-step transitions are sampled for every modelled (state, action) pair. Nothing
is hard-coded. The policy converges in 6 iterations.

```powershell
python -m backend.rl.policy_iteration              # build model + train -> models/policy_iteration/ (~12 min)
python -m backend.rl.evaluate_policy_iteration     # 5 x 24 h -> experiments/policy_iteration_5x1440min.json + plots
```

| 5 × 24 h | Random | Round-robin | Rule-based | **Policy Iteration** |
|---|---|---|---|---|
| Reward | −1588 | −669 | −688 | −1208 |
| Energy (kWh) | 557 | 614 | 567 | 499 |
| Carbon (kg) | 240 | 263 | 239 | 218 |
| Critical rack-minutes | 717 | 26 | 25 | 0.4 |
| SLA failures | 96 | 56 | 63 | 191 |

Policy Iteration is the best controller on thermal safety, energy and carbon, but it fails about 3× as many jobs.
The 144-state abstraction cannot see how urgent a waiting job is, and the model, built under a random policy,
underestimates the SLA cost of the delays it chooses. Details, diagnosis and limitations are in
**[docs/POLICY_ITERATION.md](docs/POLICY_ITERATION.md)**.

## Value Iteration (Step 8)

Value Iteration solves **the same saved empirical model** as Policy Iteration. It never samples: it loads
`models/policy_iteration/model.npz` and checks its SHA-256 `model_hash`. It converges in 250 Bellman-optimality
sweeps.

```powershell
python -m backend.rl.value_iteration              # -> models/value_iteration/ (under a second)
python -m backend.rl.evaluate_value_iteration     # PI vs VI comparison + 5 x 24 h -> experiments/value_iteration_5x1440min.json
```

PI and VI agree completely: values within 3.0e-5 (bound 3.8e-5), the same action in 132/132 modelled states, and
bit-identical 24-hour episodes. Both return the optimal policy *of the empirical 144-state model*, and that policy
still scores −1208 in the simulator. The limiting factor is the model and its abstraction, not the planner. See
**[docs/VALUE_ITERATION.md](docs/VALUE_ITERATION.md)**.

## Q-Learning (Step 9)

Tabular Q-Learning on the same 144 states / 9 actions. It learns only from `env.step` experience, starting from
Q = 0, and never reads the PI/VI model. Training is 1000 simulated days (1.44 M steps, α 0.1, γ 0.95,
ε 1.0 → 0.05), with actions chosen only among those valid in the concrete state.

```powershell
python -m backend.rl.q_learning               # ~27 min -> models/q_learning/ + experiments/q_learning_training.csv
python -m backend.rl.evaluate_q_learning      # greedy 5 x 24 h + ablations -> experiments/q_learning_5x1440min.json
```

| 5 × 24 h | Round-robin | Rule-based | PI / VI | **Q-Learning** |
|---|---|---|---|---|
| Reward | −669 | −688 | −1208 | −751 |
| Energy (kWh) | 614 | 567 | 499 | 576 |
| Critical rack-minutes | 26 | 25 | 0.4 | 6 |
| SLA failures | 56 | 63 | 191 | 80 |

Q-Learning recovers most of the PI/VI shortfall. A 2 × 2 ablation shows the gain needs both learning from its own
experience *and* acting on the live action mask. It still trails the two hand-written baselines, which is consistent
with the 144-state abstraction, not the learning method, being the remaining limit. See
**[docs/Q_LEARNING.md](docs/Q_LEARNING.md)**.

## DQN (Step 10)

A hand-written PyTorch DQN on the **30-feature observation** and the **18-action** space: a 30-128-128-18 MLP,
replay of 10 000 transitions storing both action masks, a target network synced every 100 updates, Huber loss and
Adam. The live action mask is used for exploration, the greedy choice and the Bellman target.

```powershell
python -m backend.rl.dqn                  # 300 x 24 h (432 k steps, ~63 min CPU) -> models/dqn/
python -m backend.rl.dqn --no-mask        # masking ablation -> models/dqn_nomask/
python -m backend.rl.evaluate_dqn         # greedy 5 x 24 h, comparison, feature and Q diagnostics
```

| 5 × 24 h | Round-robin | Rule-based | Q-Learning | **DQN** | DQN, no mask |
|---|---|---|---|---|---|
| Reward | −669 | −688 | −751 | **−693** | −1614 |
| Energy (kWh) | 614 | 567 | 576 | 575 | 393 |
| Critical rack-minutes | 26 | 25 | 6 | 31 | 31 |
| SLA failures | 56 | 63 | 80 | 61 | 277 |

DQN is the best learned controller: it beats Q-Learning on all 5 days, and is statistically level with the
hand-written baselines. Without masking it collapses. It is not a one-variable comparison with Q-Learning, because
state, action space and budget all differ. See **[docs/DQN.md](docs/DQN.md)**.

## PPO (Step 11)

A hand-written PyTorch PPO on the 30-feature observation and 18 actions: separate actor and critic MLPs
(128-128, tanh), a masked categorical policy, GAE (λ = 0.95), the clipped surrogate (ε = 0.2), 2048-step rollouts and
10 epochs of minibatches of 64.

```powershell
python -m backend.rl.ppo                                    # design protocol, 199 k steps (~12 min CPU) -> models/ppo/
python -m backend.rl.ppo --no-mask                          # masking ablation
python -m backend.rl.ppo --timesteps 432000 --tag ppo_432k  # DQN-matched budget
python -m backend.rl.evaluate_ppo                           # 5 x 24 h, comparison, diagnostics
```

| 5 × 24 h | Round-robin | Rule-based | Q-Learning | DQN | **PPO** | PPO, no mask |
|---|---|---|---|---|---|---|
| Reward | −669 | −688 | −751 | −693 | **−720** | −780 |
| Energy (kWh) | 614 | 567 | 576 | 575 | 588 | 583 |
| Critical rack-minutes | 26 | 25 | 6 | 31 | 36 | 25 |
| SLA failures | 56 | 63 | 80 | 61 | 65 | 82 |

PPO (evaluated deterministically) is statistically level with DQN and the hand-written baselines, using less than
half of DQN's environment steps. Sampling from the policy does better (−698; −677 with a DQN-matched budget), and
unlike DQN it barely needs the action mask. See **[docs/PPO.md](docs/PPO.md)**.

## Temperature forecasting (Step 12)

A 2-layer LSTM (hidden 64) reads the last 10 minutes of six per-rack signals and forecasts the rack's temperature
5, 10 and 15 minutes ahead. It is trained on 20 simulated days driven by four different controllers and tested on
held-out days.

```powershell
python -m backend.prediction.dataset      # 20 simulated days -> data/temperature_history.npz
python -m backend.prediction.model        # train (~13 min CPU) -> models/temperature_predictor/
python -m backend.prediction.evaluate     # metrics, baselines, plots -> experiments/temperature_prediction.json
```

| Test days | +5 min | +10 min | +15 min |
|---|---|---|---|
| LSTM MAE (°C) | 1.24 | 1.79 | 1.91 |
| Persistence MAE (°C) | 1.91 | 2.19 | 2.12 |
| LSTM R² | 0.71 | 0.47 | 0.42 |
| Warning-crossing recall | 31 % | 7 % | 0.8 % |

The forecaster is clearly useful at 5 minutes and only marginally better than "no change" at 15. The rack's 4-minute
thermal time constant means the temperature 15 minutes ahead depends on jobs and control actions that have not
happened yet. `ForecastProvider` plugs the forecasts into the environment as the 42-feature observation for Step 13.
See **[docs/TEMPERATURE_PREDICTION.md](docs/TEMPERATURE_PREDICTION.md)**.

## Dashboard (Steps 15–17)

A live web dashboard: a real-time 3D view of the data centre, any of the eight controllers driving it, a feed of
their decisions with explanations, and every saved result (algorithm comparison, training curves, forecaster).

```powershell
.\start_dashboard.ps1            # builds the UI the first time, then opens http://127.0.0.1:8000
```

For development with hot reload, run `python -m backend.api.main` and, in `frontend/`, `npm install; npm run dev`
(UI on http://127.0.0.1:5173). The stack is FastAPI + WebSocket, React, Vite, Tailwind, Motion, React Three Fiber and
Recharts. Every number shown comes from the backend. See **[docs/DASHBOARD.md](docs/DASHBOARD.md)**.
