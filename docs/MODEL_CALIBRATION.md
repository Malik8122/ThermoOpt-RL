# ThermoOpt-RL — Model Calibration Report (Phase 1, Steps 2–4)

> **Everything in this document describes a simulated, scaled-down digital twin.** The parameters are
> engineering assumptions chosen so that the simulation is physically plausible, internally consistent and
> suitable for a reinforcement-learning study. They are **not** measurements from, or claims about, any real
> data centre. See §15.

All values below come from [`config.yaml`](../config.yaml). The code lives in `backend/simulation/`. Every
equation is implemented once, in the module named next to it.

## 1–3. Racks, servers and GPU nodes

| Quantity | Per rack | Facility |
|---|---|---|
| Racks | — | **4** |
| Servers | 10 | **40** |
| GPU nodes | 2 | **8** |
| CPU capacity | 100 units | 400 units |
| GPU capacity | 100 units | 400 units |

Resources are **integer units** (`rack.py`), so accounting is exact. A job can start on a rack only if
`available CPU ≥ required CPU` **and** `available GPU ≥ required GPU`. Otherwise the rack returns an
`AssignmentResult` whose reason is `INSUFFICIENT_CPU` or `INSUFFICIENT_GPU`, and nothing is allocated.
Within a rack, load is assumed to be spread evenly over its servers (CPU) and GPU nodes (GPU). This is a
simplifying assumption, documented in `hardware.py`.

## 4–5. Initial state

| | Rack 1 | Rack 2 | Rack 3 | Rack 4 |
|---|---|---|---|---|
| Temperature (°C) | 24.5 | 23.8 | 28.1 | 23.5 |
| CPU utilisation | 72 % | 51 % | 88 % | 43 % |
| GPU utilisation | 55 % | 40 % | 82 % | 30 % |
| Cooling effectiveness η | 1.00 | 1.00 | **0.75** | 1.00 |

Shared cooling starts at **65 %**. The initial utilisation is not a hard-coded number. At reset, *background
jobs* are created whose CPU and GPU units add up exactly to the values above (72 CPU units on Rack 1, and so on). Each
has a remaining run time drawn from U(10, 120) min. The initial load therefore drains away realistically as those
jobs finish.

## 6. Power model — `power.py`

Per device, with utilisation $u \in [0, 1]$:

$$P(u) = P_\text{idle} + (P_\text{max} - P_\text{idle})\,u^{k}\qquad k = 1 \text{ (linear)}$$

| Device | $P_\text{idle}$ | $P_\text{max}$ |
|---|---|---|
| Server | 150 W | 500 W |
| GPU node | 100 W | 700 W |

- Rack IT power = Σ server power + Σ GPU-node power, converted W → kW once (`units.w_to_kw`).
- Facility IT power: 6.8 kW (all idle) to **25.6 kW** (all at 100 %).
- Example, Rack 1 at t = 0: servers 10 × (150 + 350 × 0.72) = 4.02 kW; GPUs 2 × (100 + 600 × 0.55) = 0.86 kW;
  IT = **4.88 kW** (unit-tested).
- Reported separately: server power, GPU power, IT power, cooling power, total power = IT + cooling.

## 7. Cooling model — `cooling.py`

$$P_\text{cooling} = P_\text{max}\cdot C^{\,n},\qquad P_\text{max} = 16\text{ kW},\ n = 1$$

$C$ is the **delivered** cooling level (0–1). Examples: 60 % → 9.6 kW, 65 % → 10.4 kW, 100 % → 16 kW.

**Why 16 kW (scaled-down simulation parameter).** The twin contains 40 servers and 8 GPU nodes. Their maximum
IT power is 25.6 kW. A 16 kW cooling capacity gives a PUE of about 1.3–1.6 at typical load, which is in the
range usually quoted for air-cooled facilities. This value is used as the cooling capacity of a *small*
digital-twin environment. It is not a claim about a real facility. It is configurable
(`cooling.max_cooling_power_kw`) and should be scaled together with the IT fleet if the twin is enlarged. With the
brief's original 400 kW, this fleet would run at a PUE of about 10, and cooling cost would dominate every decision.

Discrete targets: 0, 20, 40, 60, 80, 100 %. *Increase* and *decrease* move to the next level above or below the
current target, so from 65 % they go to 80 % and 60 %.

## 8. Thermal model — `thermal.py`

Explicit Euler integration with timestep Δt = 1 min, per rack r:

$$
T_r(t{+}\Delta t) = \operatorname{clip}\Big(T_r + \Delta t\,\big[\underbrace{100\,\alpha\,L_r}_{\text{IT heating}} \;-\; \underbrace{100\,\beta\,\eta_r\,C}_{\text{cooling}} \;+\; \underbrace{\kappa\,(T_\text{amb}(h) - T_r)}_{\text{ambient exchange}}\big] + \sigma\sqrt{\Delta t}\,\varepsilon,\ \ 18,\ 40\Big)
$$

| Symbol | Meaning | Unit | Value |
|---|---|---|---|
| $L_r$ | thermal load $= 0.6\,\text{cpu}_r + 0.4\,\text{gpu}_r$ | fraction 0–1 | — |
| $C$ | delivered shared cooling level | fraction 0–1 | — |
| $\eta_r$ | fraction of the shared cooling reaching rack r | fraction 0–1 | 1, 1, 0.75, 1 |
| α | heating rate | °C · min⁻¹ per **percentage point** of load | 0.035 |
| β | cooling rate | °C · min⁻¹ per **percentage point** of cooling | 0.045 |
| κ | ambient exchange rate | min⁻¹ | 0.25 |
| σ | process/sensor noise | °C per √min | 0.05 |
| ε | standard normal draw, from the twin's own seeded stream | — | — |

**Units, stated exactly.** All state variables (utilisation, cooling) are 0–1 fractions. α and β keep the brief's
values, but their unit is *per percentage point*, so the model multiplies them by 100 once (`ThermalModel.__init__`).
This was checked rather than assumed. If 0.035 and 0.045 were applied directly to 0–1 fractions, a fully loaded,
uncooled rack would settle only **0.14 °C** above ambient, and WARNING and CRITICAL could never be reached. The model
would be useless for thermal control. `tests/test_thermal.py::test_coefficient_units` guards this.

**Steady state.** Setting the bracket to zero gives

$$T^* = T_\text{amb} + \frac{100\,(\alpha L - \beta\,\eta\,C)}{\kappa}$$

So a rack moves 14 °C between 0 % and 100 % load (100α/κ), and 18 °C × η between 0 % and 100 % cooling (100β/κ).
The time constant is 1/κ = **4 min**. Explicit Euler is stable because κΔt = 0.25 < 1. The config validator
rejects κΔt ≥ 1.

**Bounds.** Temperatures are clipped to [18, 40] °C. The 18 °C floor represents the supply-air temperature:
cooling cannot push air below the temperature at which it is supplied.

**Zones** (`TemperatureStatus` enum, `get_temperature_status()`): NORMAL < 27 °C ≤ WARNING < 30 °C ≤ CRITICAL.

Calibrated steady states under the initial loads (unit-tested in `test_calibration_scenarios`):

| | Rack 1 | Rack 2 | Rack 3 | Rack 4 |
|---|---|---|---|---|
| 08:00, cooling 65 % | 20.9 | 18.3 | 26.7 | 17.1 |
| 15:00, cooling 65 % | 23.4 | 20.8 | **29.2** WARNING | 19.6 |
| 15:00, cooling 40 % | 27.9 | 25.3 | **32.6** CRITICAL | 24.1 |

## 9. Cooling response delay — `cooling.py`

When the target changes at time $t_0$ while the delivered level is $C_0$:

$$C(t) = C_0 + (C_\text{target} - C_0)\cdot\min\!\Big(1,\ \frac{t - t_0}{\tau}\Big),\qquad \tau = 3\text{ min}$$

Example (unit-tested): 65 % → 100 % gives 76.7 % after 1 min, 88.3 % after 2 min and 100 % after 3 min. The system
exposes `current_level`, `target_level` and `progress` (0–1). If the target changes mid-ramp, a new ramp starts
from the level delivered at that moment. **Both cooling power and the thermal effect use the delivered level**,
not the requested one.

A linear ramp was chosen over a first-order lag because it reaches the target in a defined time and gives an
interpretable progress value. `response_time_minutes` is configurable; 0 makes cooling instantaneous. This delay,
together with the 4-minute thermal time constant, is what makes predictive control worthwhile.

## 10. Rack 3 cooling asymmetry

Rack 3 is modelled at the end of the aisle, furthest from the cooling unit, so it receives only
**η = 0.75** of the shared cooling *effect*. At a global setting of 80 %, the effective cooling is 80 / 80 / **60** / 80 %
(unit-tested).
The **electrical** cooling power is not reduced. The plant draws 0.8 × 16 = 12.8 kW regardless, and each rack is
attributed an equal share (3.2 kW) for reporting. Under identical load and cooling, Rack 3 settles
100β·C·(1 − η)/κ = 3.6 °C hotter at C = 0.8 (unit-tested against a 200-step simulation).

## 11. Ambient model — `thermal.py`

$$T_\text{amb}(h) = 24 + 2\cos\!\big(2\pi(h - 15)/24\big)\ \text{°C}$$

The profile is deterministic: 24 °C mean, warmest (26 °C) at 15:00, coolest (22 °C) at 03:00. Setting
`ambient_daily_amplitude_c: 0` gives a constant 24 °C. The ambient term $\kappa\,(T_\text{amb} - T)$ is
Newton's law of heat exchange. It is proportional to the temperature *difference*, so it warms a rack that is
cooler than the room, cools one that is warmer, and is zero at $T = T_\text{amb}$. It is not a fixed per-minute
increment.

## 12–14. Energy, carbon and cost — `accounting.py`

For each step of Δt minutes:

| Quantity | Equation | Unit |
|---|---|---|
| IT energy | $E_\text{IT} = P_\text{IT}\,\Delta t / 60$ | kWh |
| Cooling energy | $E_\text{cool} = P_\text{cool}\,\Delta t / 60$ | kWh |
| Total energy | $E = E_\text{IT} + E_\text{cool}$ | kWh |
| Carbon | $E \times I_\text{carbon}(h)$ | gCO₂ |
| Cost | $E \times \text{price}(h)$ | ₹ (INR) |

Values are stored per step and cumulatively. Carbon and cost apply to total energy, because the grid supplies both
IT and cooling. Price and carbon intensity come from a **synthetic** daily profile. It passes exactly through the
brief's anchors (10:00, 12:00, 14:00 and 16:00) and is interpolated with periodic PCHIP, so it never overshoots
(`grid.py`, `profiles.py`).

Check: 65 % cooling for 60 min = 10.4 kW × 1 h = **10.4 kWh**. This is unit-tested and matches the baseline run below.

## Workload model — `workload.py`, `workload_generator.py`

| Type | Share | CPU units | GPU units | Duration | Flexible |
|---|---|---|---|---|---|
| NORMAL_CPU | 0.50 | triangular(10, 20, 40) | 0 | lognormal 30 ± 12 min, [5, 90] | 50 % |
| GPU_INFERENCE | 0.30 | triangular(10, 15, 30) | triangular(20, 35, 60) | lognormal 10 ± 4 min, [2, 30] | 10 % |
| AI_TRAINING | 0.20 | uniform(20, 50) | beta(2, 2) scaled to [60, 100] | lognormal 45 ± 15 min, [15, 120] | 70 % |

- Arrivals are Poisson. `time_varying` (default) follows a synthetic daily intensity: LOW about 0.4 at 04:00,
  MEDIUM about 1.0 at 09:00, HIGH about 1.4 at 14:00. It is normalised so the daily mean is 0.30 jobs/min, the same
  as `constant`.
- Jobs arriving during [t, t+1) are released at t+1. Deadline = release + duration + slack, with slack drawn from
  U(0, 10) min for strict jobs and U(60, 240) min for flexible ones. Priority is 1, 2 or 3 with probabilities
  0.50, 0.35 and 0.15.
- Statuses: `QUEUED → (DELAYED) → RUNNING → COMPLETED`, or `QUEUED/DELAYED → FAILED` (reasons
  `DEADLINE_MISSED` or `QUEUE_FULL`). Illegal transitions raise `WorkloadStateError`. Only RUNNING jobs hold
  resources, and `check_invariants()` verifies this after every step.
- Why the arrival rate is 0.30 jobs/min. A 24 h open-loop run (first-fit placement, cooling fixed at 65 %) gives
  mean utilisation of 46.7 % CPU / 46.1 % GPU (seed 42) and 49.7 % / 50.4 % (seed 43), with afternoon peaks
  around 75–82 %. Racks spend 226–337 rack-minutes in WARNING and none in CRITICAL, and 15–40 strict jobs fail
  per day. That leaves room for a smarter controller to improve.

## Simulation step order — `datacentre.py`

For the interval [t, t+Δt): (1) arrivals join the queue → (2) utilisation of the jobs running in the interval →
(3) IT power → (4) cooling ramps towards its target, giving cooling power → (5) rack temperatures → (6) price and
carbon at t → (7) energy, carbon and cost → (8) running jobs progress, and finished ones complete and release
resources at t+Δt → (9) waiting jobs that can no longer meet their deadline fail.

Physics is evaluated **before** completions, so a job of duration d contributes load to exactly d minutes. This is
unit-tested. Placement, delay and cooling targets are commands issued *between* steps by a controller.

## Baseline: 60 minutes, seed 42, no RL

Command:

```powershell
python -m backend.simulation.run_baseline            # -> data/baseline_60min.csv
```

Reference controller: FIFO first-fit placement (lowest-numbered rack that fits) and cooling held at 65 %. This is a
fixed rule used for calibration, not one of the Step 6 baselines. The episode starts at 08:00.

| | Rack 1 | Rack 2 | Rack 3 | Rack 4 |
|---|---|---|---|---|
| T at 08:00 (initial) | 24.50 | 23.80 | 28.10 | 23.50 |
| T at 09:00 (end) | 23.39 | 24.35 | 24.04 | 18.00 |
| T min / max | 21.03 / 24.50 | 18.43 / 24.35 | 23.37 / 28.10 | 18.00 / 23.50 |
| Mean CPU / GPU | 86.3 % / 55.0 % | 63.3 % / 61.5 % | 71.3 % / 82.0 % | 52.6 % / 48.8 % |
| Status at end | NORMAL | NORMAL | NORMAL | NORMAL |

Selected rows of `data/baseline_60min.csv` (temperature at the end of each minute):

| Minute | T1 | T2 | T3 | T4 | CPU % (R1–R4) | IT kW | Cooling kW | Total kW | Cumulative kWh | Cumulative gCO₂ | Cumulative ₹ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 23.57 | 22.42 | 27.79 | 21.84 | 72 / 51 / 88 / 43 | 18.17 | 10.40 | 28.57 | 0.476 | 209.5 | 3.81 |
| 9 | 21.11 | 18.75 | 26.85 | 18.00 | 72 / 51 / 88 / 43 | 18.17 | 10.40 | 28.57 | 4.762 | 2 091.0 | 38.21 |
| 19 | 22.47 | 18.59 | 23.72 | 18.00 | 92 / 51 / 47 / 43 | 17.44 | 10.40 | 27.84 | 9.396 | 4 115.6 | 75.63 |
| 29 | 22.90 | 23.08 | 24.98 | 22.53 | 92 / 95 / 95 / 89 | 23.40 | 10.40 | 33.80 | 14.559 | 6 360.2 | 117.56 |
| 39 | 22.55 | 21.86 | 27.36 | 20.26 | 72 / 35 / 95 / 61 | 19.22 | 10.40 | 29.62 | 19.861 | 8 653.4 | 160.84 |
| 59 | 23.39 | 24.35 | 24.04 | 18.00 | 96 / 97 / 47 / 29 | 18.87 | 10.40 | 29.27 | 29.743 | 12 894.7 | 242.09 |

**Totals over the hour:** IT 19.343 kWh · cooling 10.400 kWh · total **29.743 kWh** · mean PUE 1.538 ·
**12.895 kg CO₂** · **₹242.09**. Jobs: 13 arrived, 14 completed (including background jobs), 0 failed, and at the
end 9 running and 1 queued. There were 16 WARNING rack-minutes (all on Rack 3: minutes 0–4 and 35–45) and no
CRITICAL minutes.

**Reproducibility.** The CSV has SHA-256 `e7fe90bec76dafbc1411ed821f86a59a2d8f54bdf13d4d445abd5639a89cafab` for seed 42.
Repeated runs are byte-identical. `tests/test_datacentre.py::test_seed_42_run_is_reproducible` checks this and also
checks that seed 43 differs. The thermal noise, arrivals and background jobs each use their own seeded stream
(`derive_seed(seed, "thermal" | "workloads" | "background")`).

## Calibration concerns (open issues)

1. **Hard 18 °C floor.** The cooling term is linear and does not weaken as a rack approaches the supply-air
   temperature, so over-cooled, lightly loaded racks hit the clip. Rack 4 does so from minute 6 of the baseline
   and spends 16 of the 60 minutes there. The energy penalty will discourage over-cooling, but the plateau is
   non-physical and a forecaster will see flat segments. A possible refinement is cooling proportional to
   $(T - T_\text{supply})$. It is not implemented, to keep the brief's linear form.
2. **Fast dynamics.** With a 4-minute time constant, temperatures change by up to about 0.9 °C in the first
   minute when the initial state is far from equilibrium (Rack 1: 24.5 → 23.57 °C). That is plausible for rack
   exhaust air but quick. The +15 min forecast horizon in Step 12 is therefore about four time constants, so
   forecasts will mostly reflect load and cooling changes rather than thermal inertia.
3. **The initial temperatures are above the 08:00 steady state at 65 % cooling**, so the twin cools during the first
   10–15 minutes, and Rack 3 leaves WARNING after 5 minutes. Hotspots come back when load builds up and the ambient
   peaks in the afternoon. For demos that should *start* with a hotspot, use a later `start_hour` (for example 13:00).
4. **The first-fit reference is intentionally naive.** It packs Rack 1 to 86–96 % CPU. The numbers above calibrate
   the physics. They are not a benchmark to compare RL against. That comparison uses the Step 6 baselines.
5. **Even intra-rack load balancing** is assumed. With a linear power curve it does not change rack power, but it
   would with `utilisation_exponent ≠ 1`.

## 15. Why these are simulation assumptions, not measurements

- The fleet (40 servers, 8 GPU nodes, 16 kW of cooling) is deliberately small so that RL training runs on a laptop.
- The thermal model is a lumped, first-order, one-temperature-per-rack approximation. Real rooms have airflow
  recirculation, rack-inlet and rack-outlet gradients, humidity, and chiller and CRAH dynamics.
- α, β and κ were chosen to give plausible magnitudes: a 14 °C span between idle and full load, a 4-minute response,
  and a hotspot on the worst-cooled rack. They were not fitted to sensor data.
- Workload mixes, arrival profiles, grid prices and carbon intensities are synthetic, except for the four anchor
  values given in the project brief, which are themselves illustrative.
- The simulation is useful for **comparing control strategies against each other under identical conditions**.
  Absolute kWh, CO₂ and ₹ figures should not be quoted as predictions for a real facility.
