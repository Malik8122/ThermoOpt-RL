# Baseline controllers (Step 6)

Three hand-written controllers give later RL agents a reference point. The code is in `backend/rl/baselines.py`,
the settings are in `config.yaml → baselines`, and the episode runner is in `backend/evaluation/runner.py`.

## Interface

Every controller, including the RL agents from Step 7 onwards, implements:

```python
controller.reset(seed)                        # once per episode
action = controller.act(observation, mask)    # once per step; mask = env.action_masks()
```

The baselines act in the 18-action primitive space and read **only the feature observation**. They never inspect
the simulator directly, so the evaluation harness and the live dashboard can drive them exactly like PPO.
`ObservationReader` reverses the observation's normalisation to recover temperatures in °C, rack utilisation,
the cooling set-point, head-job slack, queue length and carbon intensity. Only the random controller uses the mask.

## The controllers

| Controller | Placement | Cooling |
|---|---|---|
| `random` | uniform over the currently valid actions (placement and cooling together) | (same draw) |
| `round_robin` | cycle through racks from where the last placement stopped; skip racks the head job doesn't fit on; delay the job if it fits nowhere | step to 60 % and hold it there |
| `rule_based` | see below | +1 level if any rack > 27 °C; −1 level if every rack < 24 °C; otherwise maintain |

Rule-based placement, applied in this order:

1. No job waiting, or the head job fits on no rack → delay (a no-op on an empty queue).
2. **Carbon-aware.** The head job is flexible, carbon ≥ 400 g/kWh (`grid.high_carbon_threshold`) and the job
   *can wait* → delay it.
3. **Thermal-aware.** Place the job on the least-utilised rack that fits it and is below 27 °C.
4. If the only racks that fit are hot: delay the job if it *can wait*, otherwise place it on the coolest rack
   that fits.

A job **can wait** if it has at least 30 minutes of slack left **and no other job is waiting behind it**
(`max_queue_to_delay: 1`).

### Why the queue condition exists

Each step allows one placement action, and "delay" sends the head job to the back of the queue. The rule as first
written in PHASE1_DESIGN §5 (delay any flexible job while carbon is high) spent 214 of the first 240 minutes
rotating flexible jobs through the queue. Strict jobs, which have only 0–10 minutes of slack, never reached the
front: all 31 failures in that run were strict jobs that were never delayed themselves.

The sweep below applies the queue limit to both delay rules (seeds 42–44, 24 h):

| `max_queue_to_delay` | SLA failures/day | carbon (kg) | reward |
|---|---|---|---|
| 0 (never delay deliberately) | 49.7 | 241.1 | −629.5 |
| **1 (default)** | 58.7 | 239.1 | −669.7 |
| 2 | 68.0 | 235.2 | −714.0 |
| 3 | 82.0 | 232.9 | −769.9 |

Before the queue limit was also applied to the thermal "wait for a cool rack" rule, a limit of 50 (effectively
none) gave 191 failures/day. Delaying only when carbon is forecast to fall within the hour (the observation's
`carbon_trend` feature) did not change these trade-offs. The default of 1 keeps the design's carbon rule
without letting it starve other jobs. It is deliberately not tuned further, because the rule-based controller
is a reference point, not a competitor.

## Results: 5 seeds (42–46) × 24 h, start 08:00

Generated with `python -m backend.evaluation.run_baselines` → `experiments/baselines_5x1440min.json`. Values are
mean ± sample std across seeds.

| Metric | Random | Round-robin | Rule-based |
|---|---|---|---|
| Total reward | −1588 ± 193 | **−669 ± 113** | −688 ± 117 |
| Energy (kWh) | 557 ± 16 | 614 ± 11 | 567 ± 20 |
| of which cooling (kWh) | 187 ± 12 | 230 ± 0 | 185 ± 9 |
| Cost (₹) | 4297 ± 125 | 4737 ± 88 | 4409 ± 164 |
| Carbon (kg CO₂) | 240 ± 7 | 263 ± 5 | **239 ± 9** |
| Peak temperature (°C) | 36.5 ± 0.5 | 30.8 ± 0.2 | 31.4 ± 0.7 |
| Critical rack-minutes (≥ 30 °C) | 717 ± 150 | 26 ± 7 | 25 ± 7 |
| Warning-or-worse rack-minutes | 1601 ± 179 | **450 ± 86** | 617 ± 94 |
| SLA failures | 96 ± 25 | **56 ± 23** | 63 ± 23 |
| Completed jobs | 336 ± 14 | 376 ± 13 | 368 ± 11 |
| Delayed jobs | 142 ± 13 | 76 ± 24 | 142 ± 22 |
| Mean arrival → completion (min) | 32.7 ± 2.5 | 29.5 ± 2.0 | 31.1 ± 1.9 |
| Cooling set-point changes | 893 ± 26 | 1 ± 0 | 648 ± 60 |
| Migrations | 272 ± 13 | 0 | 0 |
| Invalid-action rate | 0 | 0 | 0 |

### What the numbers say

- **Round-robin and rule-based finish almost level on reward, with opposite strengths.** Rule-based saves about
  8 % of energy (48 kWh/day) and 9 % of carbon, mostly by running cooling lower. It pays for that with 37 % more
  warning-zone rack-minutes and 7 more SLA failures per day.
- **Reactive threshold cooling hunts.** The rule-based controller changes the cooling set-point 648 times a day,
  about every other minute. The 3-minute cooling ramp and the roughly 4-minute thermal time constant mean it
  always reacts to conditions that have already changed (PHASE1_DESIGN §3). This is the gap a predictive
  controller (Step 13) is meant to close.
- **About 50 SLA failures a day are not caused by the rules.** Round-robin never delays a job deliberately and
  still fails 56. On seed 42, 11 of its 17 failures arrived between 13:00 and 17:00, the arrival peak, when strict jobs find no free capacity
  within their 0–10 minutes of slack.
- **Carbon shifting barely pays off with this action space.** Carbon is at or above 400 g/kWh for about 18 hours of the synthetic day, while
  flexible slack is at most 4 hours and only one job can be delayed per step. A learned policy that uses the
  `carbon_trend` feature and the time of day has room to do better, but the one-action-per-step queue is the
  binding constraint for any controller.
- **Random is a floor, not a competitor.** It spends 717 rack-minutes a day in the critical zone.
