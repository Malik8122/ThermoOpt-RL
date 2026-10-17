# ThermoOpt-RL — Phase 1 Design

**AI-driven data-centre cooling, workload optimisation and carbon-aware scheduling using reinforcement learning**

> ThermoOpt-RL is a **simulated digital twin**. It is not connected to any real data centre, and none of its
> numbers are measurements. All parameters live in [`config.yaml`](../config.yaml).

Phase 1 objective: *develop a simulated data-centre digital twin in which reinforcement learning dynamically
optimises workload scheduling, workload placement and cooling control while minimising energy consumption and
carbon emissions and maintaining thermal and SLA constraints.*

It covers three ideas:

| Idea | What the agent learns | Where it shows up |
|---|---|---|
| 1. Carbon-aware workload shifting | Delay flexible jobs from high-carbon to low-carbon periods without missing deadlines | `delay` action, carbon and carbon-trend features, carbon term in the reward |
| 3. Thermal-aware workload placement | Put jobs on racks that can absorb the heat (not Rack 3, which is poorly cooled) and migrate load away from hotspots | `place on rack k` and `migrate` actions |
| 4. Predictive cooling + RL | Raise cooling *before* a rack becomes critical, using LSTM forecasts | Predicted temperatures in the Predictive PPO observation |

Phase 2 (not built here): cooling failures and degradation, sensor faults, workload spikes, failure recovery, safe RL, multi-agent control.

---

## 1. Architecture

```
┌────────────────────────── frontend (React + Vite + Tailwind + Recharts) ──────────────────────────┐
│  Overview │ Digital Twin │ Workloads │ RL Training │ Algorithm Comparison │ Predictive Cooling      │
└───────────────▲─────────────────────────────────────────────────▲───────────────────────────────────┘
                │ REST (Axios)                                     │ WebSocket /ws/simulation (live state)
┌───────────────┴─────────────────────── backend/api (FastAPI) ────┴───────────────────────────────────┐
│  SimulationService (holds one live twin + active controller)   TrainingService (background jobs)    │
└───────┬───────────────────────────────┬──────────────────────────────┬───────────────────────────────┘
        │                               │                              │
┌───────▼──────────────┐   ┌────────────▼─────────────┐   ┌───────────▼────────────┐
│ backend/simulation   │   │ backend/rl (Step 5)      │   │ backend/rl (agents)    │
│  DataCentre (twin)   │◄──│  DataCenterEnv (Gym)     │◄──│  baselines             │
│  Rack, Job           │   │  reward.calculate_reward │   │  policy/value iteration│
│  thermal / power /   │   │  observation builders    │   │  q_learning, dqn, ppo  │
│  cooling models      │   │  tabular wrapper         │   └───────────▲────────────┘
│  WorkloadGenerator   │   └────────────▲─────────────┘               │
│  GridSignals         │                │ predicted temps             │
└──────────────────────┘   ┌────────────┴─────────────┐   ┌───────────┴────────────┐
                           │ backend/prediction (LSTM)│   │ backend/evaluation     │
                           └──────────────────────────┘   │  experiments, metrics  │
                                                          └────────────────────────┘
      config.yaml ──► backend/config.py (validated, read by every module)
      data/ (datasets)   models/ (trained weights)   experiments/ (curves, results, plots)
```

**Separation of concerns**

- `simulation/` knows nothing about RL. It advances the physical twin by one timestep given a *decision*
  (placement, cooling set-point). It can be driven by a human, a baseline or an agent.
- `rl/environment.py` and its helpers turn the twin into a Gymnasium MDP: observations, action decoding, reward.
- `rl/` contains controllers only. Every controller exposes `act(observation) -> action`, so the evaluation
  harness and the live dashboard treat a rule-based baseline and PPO the same way.
- `api/` holds no simulation logic. It owns a running twin and forwards commands.
- The frontend only renders what the backend sends. It generates no numbers of its own.

## 2. Folder structure

```
ThermoOpt/
├── config.yaml                  # every tunable parameter
├── requirements.txt
├── README.md
├── docs/PHASE1_DESIGN.md        # this document
├── backend/
│   ├── config.py                # typed + validated config loader
│   ├── main.py                  # FastAPI app entry point            (Step 15)
│   ├── api/                     # routers: simulation, training, comparison, predictions (Step 15)
│   ├── models/                  # Pydantic API schemas               (Step 15)
│   ├── simulation/              # twin, rack, job, thermal, power, cooling, workloads, grid (Steps 2-4)
│   ├── rl/                      # DataCenterEnv, actions, observations, reward (Step 5); agents (Steps 6-11, 13)
│   ├── prediction/              # lstm.py, dataset generation        (Step 12)
│   ├── evaluation/              # experiment runner, metrics, plots  (Step 14)
│   └── utils/                   # seeding, logging
├── frontend/                    # React app                          (Steps 16-17)
├── data/                        # generated datasets (git-ignored)
├── models/                      # trained agents / predictors (git-ignored)
├── experiments/                 # metrics JSON/CSV and plots (git-ignored)
├── notebooks/                   # optional analysis notebooks
└── tests/                       # pytest suite, grows every step
```

## 3. Digital-twin models

### 3.1 Racks

4 racks, each with 10 servers + 2 GPU nodes, 100 CPU units + 100 GPU units → 400 CPU / 400 GPU units in total.
Each rack tracks CPU %, GPU %, temperature, IT power, delivered cooling, active jobs and free capacity.
Initial state comes from the brief: temperatures 24.5 / 23.8 / 28.1 / 23.5 °C, CPU 72 / 51 / 88 / 43 %,
GPU 55 / 40 / 82 / 30 %, cooling 65 %. At reset, *background jobs* are created that add up to exactly these
utilisations, so the initial load drains away naturally rather than being a fixed number.

### 3.2 Thermal model

Full derivation, units and calibration are in **[MODEL_CALIBRATION.md](MODEL_CALIBRATION.md)**. All state
variables are 0–1 fractions. α and β are defined **per percentage point**, which is why they are multiplied by 100:

$$
T_r(t{+}\Delta t) = \mathrm{clip}\Big(T_r + \Delta t\big[100\alpha\,L_r - 100\beta\,\eta_r\,C + \kappa\,(T_\text{amb}(h) - T_r)\big] + \sigma\sqrt{\Delta t}\,\varepsilon,\ 18,\ 40\Big)
$$

with $L_r = 0.6\,\text{cpu}_r + 0.4\,\text{gpu}_r$, α = 0.035, β = 0.045, κ = 0.25 /min, η = (1, 1, **0.75**, 1),
$T_\text{amb}(h) = 24 + 2\cos(2\pi(h-15)/24)$ and σ = 0.05 °C. Steady state: $T^* = T_\text{amb} + 100(\alpha L - \beta\eta C)/\kappa$.

| Steady state under the initial loads | Rack 1 | Rack 2 | Rack 3 | Rack 4 |
|---|---|---|---|---|
| 08:00, cooling 65 % | 20.9 | 18.3 | 26.7 | 17.1 |
| 15:00, cooling 65 % | 23.4 | 20.8 | **29.2** (warning) | 19.6 |
| 15:00, cooling 40 % | 27.9 | 25.3 | **32.6** (critical) | 24.1 |

Rack 3's lower cooling effectiveness (η = 0.75, end-of-aisle position) makes it the structural hotspot, and
the afternoon ambient peak pushes it towards critical. The other racks are over-cooled at the same set-point.
Raising cooling for everyone is expensive, so the efficient answer is to *place and migrate work away from
Rack 3*. That makes the scheduling problem non-trivial.

### 3.3 Cooling model

- One shared cooling plant with target $C_\text{set} \in \{0,20,40,60,80,100\}\%$ (starts at 65 %;
  the first *increase* goes to 80 % and the first *decrease* to 60 %).
- **Response delay:** the delivered level ramps *linearly* to a new target over `response_time_minutes` = 3.
  Together with the 4-minute thermal time constant, a purely reactive controller is late by several minutes.
  Predictive control is designed to recover that time.
- Cooling power $= 16\text{ kW} \times C_\text{delivered}$. Rack effectiveness scales the cooling *effect*, never the electrical power.
- **16 kW is a scaled-down simulation parameter** matched to the 25.6 kW maximum IT load (PUE ≈ 1.3–1.6). The
  brief's 400 kW would imply a PUE of about 10 for this fleet.

### 3.4 Power model

$P = P_\text{idle} + (P_\text{max} - P_\text{idle})\,u^{k}$ with u in 0–1 and k = 1. Server 150–500 W, GPU node 100–700 W.
Facility IT power ranges from 6.8 kW (idle) to 25.6 kW (full). IT, cooling and total power are reported separately.
Energy per step = P × Δt/60, carbon = energy × intensity, cost = energy × price.

### 3.5 Workloads

Poisson arrivals with a synthetic daily profile (LOW at night, MEDIUM in the day, HIGH in the afternoon peak), averaging
0.30 jobs/min. A constant-rate pattern is also available.

| Type | Share | CPU units | GPU units | Duration (log-normal mean ± std, clipped) | Flexible |
|---|---|---|---|---|---|
| NORMAL_CPU | 50 % | triangular 10–40 (mode 20) | 0 | 30 ± 12 min [5, 90] | 50 % |
| GPU_INFERENCE | 30 % | triangular 10–30 (mode 15) | triangular 20–60 (mode 35) | 10 ± 4 min [2, 30] | 10 % |
| AI_TRAINING | 20 % | uniform 20–50 | beta(2, 2) on 60–100 | 45 ± 15 min [15, 120] | 70 % |

Deadline = arrival + duration + slack. Slack is 0–10 min for strict jobs and 60–240 min for flexible ones.
Statuses: `QUEUED`, `DELAYED` (deferred by a controller), `RUNNING`, `COMPLETED` and `FAILED`. A job FAILS when it
can no longer finish by its deadline or is rejected because the queue is full. Only RUNNING jobs hold resources.
Over a simulated day with a naive first-fit dispatcher, mean utilisation is 47–50 % for both CPU and GPU, with
afternoon peaks around 75–80 %.

### 3.6 Grid signals (synthetic)

Hourly anchors are interpolated with periodic PCHIP, which is smooth and never overshoots. The 10:00 / 12:00 / 14:00 /
16:00 anchors are the brief's values. The rest are illustrative: solar dip in the afternoon, evening peak,
coal-heavy night.

| Hour | 0 | 3 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ₹/kWh | 6.80 | 6.20 | 7.00 | 8.00 | **8.50** | **9.20** | **7.10** | **6.50** | 8.80 | 9.50 | 7.80 |
| gCO₂/kWh | 480 | 500 | 470 | 440 | **420** | **450** | **310** | **250** | 380 | 520 | 500 |

The low-carbon window is roughly 13:30–17:00. A flexible training job arriving at 11:00 with 240 min of slack can
be shifted into it.

## 4. MDP formulation

### 4.1 Time and decisions

One step = 1 simulated minute. At each step the agent makes **one joint decision**:

1. what to do with the job at the **head of the queue** (if any), and
2. how to adjust the shared **cooling set-point**.

Arrivals average 0.30 jobs/min, so one placement per minute is enough capacity. Jobs waiting behind the head accumulate
queueing time, and that waiting is what an SLA violation eventually penalises.

### 4.2 Action space: `Discrete(18)`

`action = placement * 3 + cooling`

| placement | meaning | | cooling | meaning |
|---|---|---|---|---|
| 0–3 | place head job on Rack 1–4 | | 0 | decrease one level |
| 4 | delay (move head job to back of queue) | | 1 | maintain |
| 5 | migrate one job from the hottest rack to the coolest rack that fits it | | 2 | increase one level |

- Placing on a rack without enough free CPU/GPU is **invalid**. The job stays queued and a small penalty (0.1) is applied.
  The observation includes per-rack "fits" flags, so agents can learn to avoid invalid placements.
- With an empty queue the placement part is ignored, except for *migrate*.
- A single combined 18-action space is small enough for DQN and PPO, and it lets one policy trade cooling against
  placement. That trade-off is the core of Ideas 3 and 4. Splitting it into two environments would hide the trade-off.

### 4.3 Continuous observation (DQN, PPO): 30 features, all normalised to about [0, 1]

| Index | Feature | Normalisation |
|---|---|---|
| 0–3 | rack temperatures | (T − 18) / 22 |
| 4–7 | rack CPU utilisation | / 100 |
| 8–11 | rack GPU utilisation | / 100 |
| 12 | cooling set-point | / 100 |
| 13 | delivered cooling | / 100 |
| 14 | queue length | / 20, clipped |
| **15** | FastAPI backend + WebSocket | ✅ 24 API tests; all 8 controllers drive the live twin |
| **16** | React dashboard (6 pages) | ✅ builds with `npm run build`; [DASHBOARD.md](DASHBOARD.md) |
| **17** | Frontend ↔ backend wiring | ✅ live 3D twin, controls, decision feed, saved results |
| 18 | head job is flexible | 0/1 |
| 19 | a job is waiting | 0/1 |
| 20 | electricity price | / max price |
| 21 | carbon intensity | / max carbon |
| 22 | carbon trend, carbon(t+60) − carbon(t) | / max carbon |
| 23–24 | time of day sin / cos | [−1, 1] |
| 25 | ambient temperature | (T − 18) / 22 |
| 26–29 | head job fits on rack 1–4 | 0/1 |

Free capacity is `1 − utilisation`, so it is already represented.
**Predictive PPO** appends 12 features: LSTM forecasts for each rack at +5, +10 and +15 min, normalised like
temperature (42 features).

### 4.4 Discrete state (Policy Iteration, Value Iteration, Q-Learning): 144 states

| Factor | Levels |
|---|---|
| hottest-rack temperature | LOW < 25, MEDIUM 25–27, HIGH 27–30, CRITICAL ≥ 30 |
| mean utilisation | LOW < 40, MEDIUM 40–70, HIGH ≥ 70 |
| cooling set-point | LOW < 40, MEDIUM 40–70, HIGH ≥ 70 |
| carbon intensity | LOW < 400, HIGH ≥ 400 |
| queue | EMPTY, PENDING |

4 × 3 × 3 × 2 × 2 = **144**. The discrete state does not record which rack is which, so tabular agents use
9 **macro-actions**: {place on coolest feasible rack, place on least-utilised feasible rack, delay} ×
{decrease, maintain, increase}. A wrapper maps each macro-action to one of the 18 primitive actions, so all
agents run in the same simulator and are compared on the same metrics.

Policy and Value Iteration need a model $P(s'|s,a), R(s,a)$. It is **estimated empirically** by sampling the
simulator from many concrete states belonging to each abstract state (`model_samples_per_state_action` = 100),
then solved exactly. This is certainty-equivalence planning. It is documented as an approximation: the aggregated
state is not exactly Markov.

### 4.5 Reward: `calculate_reward()`

> The implemented definitions, normalisations and SLA calibration are in **[RL_ENVIRONMENT.md](RL_ENVIRONMENT.md)**
> §8–13 and supersede this summary where they differ. SLA is 30 per violation plus 0.5 × the fraction of slack
> waited, and M is normalised by 2.

$$
r_t = -\big(0.30\,E_t + 0.20\,C_t + 0.30\,\Theta_t + 0.15\,S_t + 0.05\,M_t\big) + 0.05\sum_{j\,\in\,\text{completed}_t}\text{priority}_j - 0.1\cdot\mathbb 1[\text{invalid}]
$$

| Term | Definition | Range |
|---|---|---|
| $E_t$ energy | total power / max total power | [0, 1] |
| $C_t$ carbon | (total power × carbon intensity) / (max total power × max carbon) | [0, 1] |
| $\Theta_t$ thermal | mean over racks of: 0 if T < 27; (T − 27)/3 in the warning band; **10 + 2·(T − 30)** if T ≥ 30 | [0, 1) normally, ≥ 2.5 once any rack is critical |
| $S_t$ SLA | violations this step / 3 | ≥ 0 |
| $M_t$ switching | 1 if cooling set-point changed + 1 per migration | {0, 1, 2} |

The thermal term jumps sharply at the critical threshold. With weight 0.30, a single critical rack costs
0.30 × 10 / 4 = 0.75 per step, more than the maximum energy + carbon penalty combined (0.5). The agent can never profit by
letting a rack go critical to save energy. All weights and constants are in `config.yaml → reward`.

## 5. Algorithms

| Algorithm | Env | Type | Key settings |
|---|---|---|---|
| Random | primitive | baseline | uniform over valid actions |
| Round-robin | primitive | baseline | cycle racks, skip full ones, cooling fixed at 60 % |
| Rule-based | primitive | baseline | avoid racks ≥ 27 °C, prefer least utilised, +cooling if any rack > 27 °C, −cooling if all < 24 °C, delay flexible jobs when carbon ≥ 400 and deadline permits (≥ 30 min slack, no other job waiting; see BASELINES.md) |
| Policy Iteration | tabular | model-based DP | γ = 0.95, θ = 1e-6 |
| Value Iteration | tabular | model-based DP | γ = 0.95, θ = 1e-6 |
| Q-Learning | tabular | model-free TD | α = 0.1, γ = 0.95, ε 1.0 → 0.05 (×0.995/episode) |
| DQN | continuous | deep value-based | 128-128 MLP, replay 10 000, batch 64, target update 100, lr 1e-3 |
| PPO | continuous | actor-critic | lr 3e-4, γ = 0.99, λ = 0.95, clip 0.2, entropy 0.01 |
| Predictive PPO | continuous + forecasts | actor-critic | same as PPO, 42-dim observation |

All RL logic (DP sweeps, TD updates, replay, target network, GAE, clipped surrogate) is written by hand. No
Stable-Baselines3.

**Temperature predictor:** a 2-layer LSTM (hidden 64) takes the last 10 minutes of [temperature, CPU %, GPU %, IT power,
cooling, queue] for one rack and outputs +5 / +10 / +15-minute temperatures. It is trained on 20 simulated days,
split chronologically 70 / 15 / 15, and reported with MAE, RMSE and R² plus actual-vs-predicted plots.

## 6. Evaluation protocol

Every controller runs on the **same 5 seeds (42–46) × 24-hour episodes**, so each sees identical job arrivals and grid
profiles. Results are reported as mean ± std across seeds:

total / IT / cooling energy (kWh) · electricity cost (₹) · carbon (kg CO₂) · mean and peak temperature ·
thermal violations (rack-minutes ≥ 30 °C) · warning rack-minutes · SLA violations · completed jobs · delayed jobs ·
mean completion time · cumulative reward.

Nothing is hard-coded. Every number in the comparison page is read from `experiments/comparison.json`, which the
evaluation script generates.

## 7. API (Step 15)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | clock, power split, grid signals, temperatures, SLA counters, active controller |
| GET | `/api/racks` | per-rack state |
| GET | `/api/workloads` | queue, running and finished jobs; recent agent decisions with explanations |
| POST | `/api/simulation/start` · `/pause` · `/reset` · `/step` | control the live twin (`step` accepts minutes: 1 / 5 / 15 / 60) |
| POST | `/api/simulation/controller` | choose which controller drives the twin |
| POST | `/api/workload` | inject a user-defined job |
| POST | `/api/train/{algorithm}` | start training in the background |
| GET | `/api/training/{algorithm}/metrics` | live and saved training curves |
| GET | `/api/comparison` | experiment results |
| GET | `/api/predictions` | LSTM forecasts for each rack |
| WS | `/ws/simulation` | pushes the full state after every simulated minute |

**Explainability.** Every decision is logged with the observation features that drove it, for example
*"Rack 3 forecast 30.1 °C at +15 min (now 26.5 °C) → PPO raised cooling 60 % → 80 %"* or
*"Carbon 450 g/kWh, falling to 310 within 2 h; job J-0142 is flexible with 190 min slack → delayed"*.
The explanation is built from the actual state and action. It is not free text invented by the frontend.

## 8. Implementation roadmap

| Step | Deliverable | Done when |
|---|---|---|
| **1** | Architecture, `config.yaml`, validated loader, seeding, folder layout | ✅ `python -m backend.config` prints the summary; config tests pass |
| **2** | `simulation/`: `Rack`, `Server`, `GPUNode`, `Workload`, `DataCentre`, grid signals | ✅ twin initialises with brief values; PCHIP profile hits all anchors |
| **3** | `WorkloadGenerator` + background jobs | ✅ seeded arrivals reproducible; type mix / durations match config statistically |
| **4** | thermal, power, cooling models + `DataCentre.step()` | ✅ steady states match §3.2 table; energy/carbon/cost tests; `data/baseline_60min.csv` reproducible |
| **5** | `DataCenterEnv`, `calculate_reward`, observation builders, tabular wrapper | ✅ passes `gymnasium.utils.env_checker`; 68 Step 5 tests; random benchmark |
| **6** | Random, round-robin, rule-based controllers | ✅ 5 × 24 h runs, `experiments/baselines_5x1440min.json`, [BASELINES.md](BASELINES.md) |
| **7** | Policy Iteration | ✅ converges in 6 iterations; plots in `experiments/plots/policy_iteration/`; [POLICY_ITERATION.md](POLICY_ITERATION.md) |
| **8** | Value Iteration | ✅ same model hash as PI; 250 sweeps; values within 3e-5 and 100 % policy agreement with PI; [VALUE_ITERATION.md](VALUE_ITERATION.md) |
| **9** | Q-Learning | ✅ 1000 × 24 h training; reward, ε, ΔQ, SLA, energy and carbon curves; [Q_LEARNING.md](Q_LEARNING.md) |
| **10** | DQN | ✅ 300 × 24 h, masked + no-mask ablation; loss, Q, ε, reward, SLA, energy, carbon and invalid-rate curves; [DQN.md](DQN.md) |
| **11** | PPO | ✅ design run (199 k steps) + no-mask ablation + 430 k run; reward, policy/value loss, entropy, KL curves; [PPO.md](PPO.md) |
| **12** | LSTM predictor | ✅ MAE 1.24 / 1.79 / 1.91 °C at +5 / +10 / +15 min, with baselines and ablations; live 42-feature provider; [TEMPERATURE_PREDICTION.md](TEMPERATURE_PREDICTION.md) |
| 13 | Predictive PPO | trained with forecasts in the observation; proactive-cooling episodes logged |
| 14 | Evaluation framework | `experiments/comparison.json` + comparison plots for all 9 controllers |
| 15 | FastAPI backend + WebSocket | all endpoints tested with TestClient |
| 16 | React dashboard (6 pages) | builds with `npm run build` |
| 17 | Frontend ↔ backend wiring | live twin visible, controls work |
| 18 | End-to-end test + demo script | the 15-point demo from the brief runs start to finish |

## 9. Deviations from the brief and the reasons for them

1. **Max cooling power 16 kW instead of 400 kW.** This is a scaled-down simulation parameter that keeps PUE realistic for a 25.6 kW IT load (§3.3). It is configurable.
2. **Cooling response delay and per-rack cooling effectiveness added.** Without the lag, prediction has no value.
   Without per-rack effectiveness, all racks look alike and placement barely matters.
3. **Ambient coupling κ = 0.25 /min.** The brief leaves the ambient term unspecified. This value gives a 4-minute
   thermal time constant and a realistic 14 °C span between idle and full load.
4. **Migration included as a placement option.** The final demo asks for "RL moves workload if necessary".
   It costs a switching penalty so the agent doesn't migrate needlessly.
5. **Tabular agents use macro-actions** because the 144-state abstraction cannot tell racks apart (§4.4).
