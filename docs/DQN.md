# Deep Q-Network (Step 10)

| | |
|---|---|
| Code | `backend/rl/dqn.py` (network, agent, training loop, controller); `backend/rl/replay_buffer.py` |
| Train | `python -m backend.rl.dqn` → `models/dqn/`, `experiments/dqn_training.csv` (63 min, CPU) · `python -m backend.rl.dqn --no-mask` → `models/dqn_nomask/` |
| Evaluate | `python -m backend.rl.evaluate_dqn` → `experiments/dqn_5x1440min.json`, `experiments/plots/dqn/` |
| Tests | `tests/test_dqn.py`: 37 tests, including a toy contextual bandit (masked and unmasked) |

> **Summary.** DQN with the live action mask scores **−693 ± 102**. That makes it the best *learned* controller:
> it beats Q-Learning on all 5 evaluation days (by 58 ± 42 on average) and has 61 SLA failures a day. It is
> statistically level with rule-based (−688; paired difference −5 ± 23) and slightly behind round-robin (−669;
> −24 ± 26, 1 win in 5). Without the mask, the same network, budget and seed collapses to **−1614**, worse than
> random, with 48 % invalid actions and 277 SLA failures. DQN is **not** a one-variable replacement for Q-Learning,
> because it changes both the state (30 continuous features instead of 144 states) and the actions (18 instead of 9).

## 1. Why DQN

Steps 7–9 showed that the 144-state abstraction hides information that matters: job urgency, which rack is hot, time
of day, and the cooling ramp. DQN reads the environment's **30-feature observation** instead. That observation
includes per-rack temperatures and utilisation, the head job's size, urgency and flexibility, whether the head job
fits on each rack, carbon, price, the carbon trend and time of day. A neural network approximates Q over this
continuous input, which no table could represent.

## 2. Observation: 30 features

This is the Step 5 `ObservationBuilder` output, unchanged (RL_ENVIRONMENT.md §6):
- rack temperatures ×4, CPU ×4 and GPU ×4;
- cooling set-point and delivered cooling;
- queue length;
- head job's CPU, GPU, urgency and flexible flag, and whether a job is waiting;
- price, carbon, carbon trend, time sin/cos and ambient temperature;
- head-job-fits-rack ×4.

The agent validates the shape (30,) for every observation, and `observation_space.contains` at every reset. All 7200
evaluation observations were inside the declared bounds. The tabular state id is never computed or used: the DQN
module doesn't import the state encoder, and a test checks this.

## 3. Actions: 18

The primitive action space, unchanged: `a = placement × 3 + cooling`. Placement is: rack 1–4, delay, or migrate
from the hottest rack. Cooling is: decrease, maintain or increase. Unlike the tabular agents, DQN chooses the
**specific rack** and can **migrate**.

## 4. Network

```
QNetwork: Linear(30,128) → ReLU → Linear(128,128) → ReLU → Linear(128,18)     (22 802 parameters)
```

There are an online network and a target network. Both are initialised from `derive_seed(seed, "dqn", "init")`,
and the target starts as an exact copy of the online network.

## 5. Replay buffer

A fixed-capacity ring buffer of 10 000 transitions (`replay_buffer.py`), about 7 simulated days. Each entry stores
`(s, a, r, s′, terminated, truncated, M(s), M(s′))`, where M is the action mask **observed at the time**. Masks are
copied into the buffer, so later simulator states can never change a stored target (tested). Sampling is uniform with
replacement, from a generator seeded with `derive_seed(seed, "dqn", "replay")`.

## 6. Target network

The target is a hard copy of the online network every **100 gradient updates**. The final run made 4310 syncs.

## 7. Bellman target

$$y = r + \gamma\,(1-d)\max_{a' \in M(s')} Q_{\text{target}}(s', a'), \qquad d = \text{terminated} \lor \text{truncated}$$

Invalid next actions are set to −∞ before the max, so they cannot affect y. The `(1 − d)` is applied with
`torch.where`, so a terminal row never multiplies −∞ by 0; a test covers this.

**Truncation.** Following the Step 10 specification (`truncation_is_terminal: true`), the time-limit truncation at
the end of each 24-hour episode is treated as terminal. That is not strictly correct for a continuing task, but it
affects only 1 in 1440 transitions. Q-Learning (Step 9) bootstrapped through truncation instead. The flag is
configurable, and both settings are tested.

**Loss and optimiser.** Huber loss (`SmoothL1Loss`, δ = 1) is used rather than MSE: it is less sensitive to rare
large TD errors, such as SLA-failure spikes of −4.5. Adam with lr 1e-3, and gradient-norm clipping at 10.

## 8. Action masking

The live mask `env.action_masks()` is used in three places:
1. exploration draws uniformly from valid actions only;
2. exploitation takes the argmax over valid actions;
3. the target's max ranges over the stored M(s′).

Result: **0 invalid actions** in 432 000 training steps and 7200 evaluation steps. Exact ties go to the lowest action
index.

## 9. Exploration

Exploration is ε-greedy with a **linear, per-environment-step** schedule. This is deliberately not Q-Learning's
per-episode multiplicative decay:

$$\varepsilon(t) = \max\big(0.05,\ 1.0 - 0.95 \cdot t / 50\,000\big)$$

ε reaches 0.05 at step 50 000, part-way through episode 35, and stays there for the remaining 382 000 steps. Over
the run, 45 468 steps (10.5 %) explored and 386 532 exploited.

## 10. Hyperparameters

| | Value | | Value |
|---|---|---|---|
| learning rate (Adam) | 1e-3 | batch | 64 |
| γ | 0.95 | replay capacity | 10 000 |
| target sync | every 100 updates | warm-up (`learning_starts`) | 1000 transitions |
| train frequency | 1 update per step | gradient clip | 10 |
| ε | 1.0 → 0.05 linear over 50 000 steps | loss | Huber |
| hidden layers | 128, 128 | seed | 42 |

All values are in `config.yaml → rl.dqn`.

## 11. Training protocol

| | DQN (masked) | DQN (no mask) | Q-Learning (Step 9) |
|---|---|---|---|
| episodes × length | **300 × 1440 min** | 300 × 1440 min | 1000 × 1440 min |
| environment steps | 432 000 | 432 000 | 1 440 000 |
| gradient updates / target syncs | 431 001 / 4310 | 431 001 / 4310 | n/a |
| wall-clock | 3782 s | 3856 s | 1592 s |
| device | CPU, 1 thread, PyTorch 2.14.0+cpu | same | n/a |

**Budget.** DQN saw 30 % of Q-Learning's environment steps. The 300 × 1440 protocol is the value fixed in
PHASE1_DESIGN. With one gradient update per step, a DQN step costs about 5 ms against about 1.1 ms for Q-Learning,
so a 1.44 M-step run would take about 3.5 hours per variant. The comparison with Q-Learning is therefore **not
budget-matched**: DQN reached a better result with fewer environment steps, but more compute.

Episode *e* uses environment seed `derive_seed(42, "dqn", "episode", e)` and a random start hour. Every episode is
logged to `experiments/dqn_training.csv` (26 columns).

## 12. Training diagnostics

| | first 30 episodes | last 30 episodes |
|---|---|---|
| reward (while exploring) | −1118 | −889 |
| SLA failures | 112 | 105 |
| critical rack-minutes | 242 | 40 |
| energy (kWh) | 511 | 538 |
| mean Huber loss | 0.31 | 0.33 |
| mean predicted Q (batch) | −3.4 | −7.0 |

Training reward rises for about 30 episodes, then **oscillates** without a clear trend (`reward.png`). The 20-episode
mean peaks at about −810 near episode 140, dips to about −1000 near episode 200, and ends at about −870. The loss
stays flat at about 0.3 (`loss.png`); the largest batch loss was 1.21. Mean Q drifts down to about −7 and settles
(`mean_q.png`). No NaN, infinity or exploding values occurred, and the training loop raises an error on any non-finite
loss or Q. The evaluated model is the **final** network. No checkpoint was selected, because selecting on the
evaluation days would leak them.

Training reward, measured with ε = 0.05, is well below greedy evaluation reward (−889 vs −693). At this ε about 72
exploratory actions a day are random, and random delays, migrations and cooling changes are expensive.

## 13. Evaluation protocol

The shared runner is used with seeds 42–46, 24 hours from 08:00, the same as Steps 6–9. The controller is greedy
(ε = 0). A test checks that evaluation draws no random numbers. The masked model uses the live mask; the no-mask
model ignores it. Earlier controllers' numbers are **inherited** from their result files.

## 14. Results

### 5 × 24 h, greedy

| Metric | DQN (live mask) | DQN no-mask ablation |
|---|---|---|
| Total reward | **−693.1 ± 101.5** | −1613.7 ± 102.5 |
| Energy / IT / cooling (kWh) | 574.7 ± 22.9 / 381.8 / 193.0 | 393.3 ± 41.4 / 271.1 / 122.2 |
| Carbon (kg) · cost (₹) | 240.7 ± 10.9 · 4501.8 ± 182.6 | 167.0 · 2991.9 |
| Mean / peak temperature (°C) | 22.7 ± 0.2 / 31.7 ± 1.3 | 22.4 / 33.2 |
| Warning / critical rack-minutes | 434.4 ± 65.6 / 31.2 ± 23.7 | 231.8 / 30.8 |
| Completed / SLA failures | 371.0 ± 10.4 / **60.8 ± 23.6** | 154.6 / 277.0 |
| Delayed jobs · mean arrival → completion | 93.6 · 30.4 min | 104.4 · 65.7 min |
| Cooling changes · migrations | 838 ± 85 · 245 ± 46 | 198 · 138 |
| Invalid-action rate | **0.000** | 0.481 ± 0.06 |

### All controllers

| Controller | Source | Reward | Energy kWh | Carbon kg | Cost ₹ | Critical min | SLA failures | Invalid |
|---|---|---|---|---|---|---|---|---|
| Random | Step 6, inherited | −1588 ± 193 | 557 | 240 | 4297 | 717 | 96 | 0 |
| Round-robin | Step 6, inherited | **−669 ± 113** | 614 | 263 | 4737 | 26 | **56** | 0 |
| Rule-based | Step 6, inherited | −688 ± 117 | 567 | 239 | 4410 | 25 | 63 | 0 |
| Policy Iteration | Step 7, inherited | −1208 ± 108 | **499** | **218** | **3901** | **0.4** | 191 | 0.131 |
| Value Iteration | Step 8, inherited | −1208 ± 108 | **499** | **218** | **3901** | **0.4** | 191 | 0.131 |
| Q-Learning | Step 9, inherited | −751 ± 121 | 576 | 244 | 4479 | 6.0 | 80 | 0 |
| **DQN** | **Step 10, new** | **−693 ± 102** | 575 | 241 | 4502 | 31.2 | 61 | 0 |
| *DQN, no mask* | Step 10, new | −1614 ± 102 | 393 | 167 | 2992 | 30.8 | 277 | 0.481 |

**Paired per-seed differences** (same 5 days for every controller):

| DQN minus… | per seed (42…46) | mean ± sd | DQN better on |
|---|---|---|---|
| Q-Learning | +32.9, +13.7, +51.2, +71.4, +121.7 | **+58.2 ± 41.5** | 5 / 5 |
| Rule-based | −24.8, −7.7, +20.0, −29.1, +16.0 | −5.1 ± 22.6 | 2 / 5 |
| Round-robin | −53.0, −18.0, −42.5, −22.5, +15.4 | −24.1 ± 26.3 | 1 / 5 |

With 5 days, DQN vs rule-based is a tie, and DQN vs round-robin is a small deficit that is not clearly distinguishable
from zero. The advantage over Q-Learning is consistent.

- **Energy and carbon:** essentially the same as Q-Learning and rule-based (575 kWh / 241 kg), and below round-robin
  (614 / 263). PI/VI and the no-mask model use less only because they complete far fewer jobs.
- **Thermal:** 31 critical rack-minutes a day, similar to round-robin and rule-based (about 25) but worse than
  Q-Learning (6), with 434 warning minutes a day. DQN migrates 245 times a day and changes cooling 838 times, so it
  is as restless as Q-Learning and rule-based.
- **SLA:** 61 failures, between round-robin (56) and rule-based (63), and 24 % fewer than Q-Learning (80).

## 15. Masking ablation

Same architecture, hyperparameters, seed, 432 000-step budget and evaluation days. Only `use_action_mask = false`:
exploration, argmax and the target's max all range over all 18 actions. The environment's invalid-action semantics
are unchanged (a no-op plus a −0.1 penalty).

| | masked | no mask |
|---|---|---|
| greedy reward | −693 | **−1614** (worse than random, −1588) |
| training reward, first → last 30 episodes | −1118 → −889 | −1334 → **−1613** (gets worse) |
| invalid actions (evaluation) | 0 % | 48 % |
| SLA failures | 61 | 277 |
| Q range on evaluation states | −37.6 … −0.7 | −65.7 … **+15.2** |

Invalid actions of the no-mask policy on seed 42 (1440 steps):

| Reason | Count |
|---|---|
| placing a job that doesn't fit (CPU 155, GPU 129) | 284 |
| migrating with no migration candidate (160, 11 of these also cooling-at-max) | 160 |
| decreasing cooling already at 0 % | 115 |
| increasing cooling already at 100 % | 20 |

**Interpretation, different from Step 9.** For DQN the mask adds little *information*:
- whether the head job fits on each rack is in the observation (`head_fits_rack1–4`);
- the cooling limits follow from `cooling_setpoint`;
- only migration validity, which depends on per-job sizes, is not directly observable.

So the ablation mostly measures the mask as a **learning constraint**, not an information channel. Without it:
- exploration and greedy choices frequently land on no-ops (33 % of training steps were invalid in the first 30
  episodes, 36 % in the last 30);
- the target's max includes actions that are never actually valid in s′. Their Q-values are poorly trained, and the
  maximum over them is biased upwards. That is visible as Q-values reaching +15, when almost every reward is
  negative;
- an invalid placement is a *free hold*: the job stays at the head of the queue without being rotated, and no energy
  is spent. The short-term reward looks better, and the deadline cost arrives later.

These effects are consistent with the training reward getting worse over time. Proving which one dominates would need
further ablations. The ablation was not used to tune the main model.

## 16. Does DQN use the richer observation?

The measurements below are descriptive, over the 7200 greedy evaluation steps of the masked model. They are not
causal feature attributions.

- **Permutation sensitivity** (shuffle one feature across all recorded states, keep the recorded masks, count
  changed greedy actions). The most influential features are time of day (`time_sin` 30 %, `time_cos` 30 %), cooling
  set-point and delivered cooling (29 %, 27 %), head-job GPU size (23 %), queue length (23 %), rack CPU utilisation
  (15–21 %), and head-job flexibility and urgency (19 %, 17 %). The rack temperatures come low (8–9 %), as do carbon
  and price (6–9 %). Features are correlated (time, carbon, price and ambient all follow the clock), so the low scores
  for carbon and price don't mean they are unused. **Time and head-job urgency are not in the 144-state abstraction,
  and DQN's decisions depend on them.**
- **Rack choice.** When a job is waiting and DQN places it, the chosen rack is the coolest feasible one 52 % of the
  time and the least-utilised feasible one 47 % of the time. Rack 3, the badly cooled end-of-aisle rack, receives the
  fewest placements: 14 % of them, against 30 %, 21 % and 35 % for racks 1, 2 and 4. A tabular agent cannot express a
  rack preference at all.
- **Cooling vs hottest rack.** The share of *decrease* falls as the hottest rack warms: 38 % below 25 °C, 22 % at
  25–27 °C, 16 % at 27–30 °C and above. The share of *increase* is 28–32 % in every band. This is a weak relationship,
  consistent with the low permutation scores for temperature.
- **Delay.** With a job waiting, DQN places it in 44 % of steps, delays it in 41 % and migrates in the rest.

## 17. Q-value diagnostics (evaluation states)

| | masked | no mask |
|---|---|---|
| all Q finite | yes | yes |
| Q over all actions: mean ± std, [min, max] | −6.30 ± 2.73, [−37.6, −0.7] | −8.61 ± 3.84, [−65.7, +15.2] |
| best valid Q: mean, [min, max] | −6.12, [−19.6, −1.7] | −7.72, [−23.6, +15.2] |
| spread over valid actions: median / max | 0.27 / 20.5 | 0.47 / 55.1 |
| best − second-best: median; share < 0.01 | 0.030; 28 % | 0.026; 20 % |

The masked Q-values are negative, bounded and not saturated. Their magnitude is consistent with about 20-step
discounted sums of per-step rewards around −0.3. Many decisions are close calls: 28 % are within 0.01 of the
runner-up. The representative states below (seed 42, at quantiles of the hottest-rack temperature) are listed in the
results JSON. Cool states choose "place on rack 4 + maintain". At the 75th percentile (27.0 °C) and the maximum
(31.5 °C) the top action becomes **delay + increase cooling**, with *migrate from hottest rack* within 0.002 of it at
31.5 °C.

## 18. Comparison with Q-Learning: not a one-variable change

> **DQN is not a direct one-variable replacement for Q-Learning, because it uses both a richer state representation
> and a larger action space.**

| | Q-Learning | DQN |
|---|---|---|
| state | 144 abstract states | 30 continuous features |
| actions | 9 macro-actions (coolest / least-utilised / delay × cooling) | 18 (specific rack / delay / **migrate** × cooling) |
| Q representation | table | neural network (22 802 parameters) |
| target | bootstraps through truncation | truncation treated as terminal |
| budget | 1.44 M steps | 432 k steps |

DQN's +58 per day over Q-Learning cannot be assigned to any single one of these. The feature-dependence evidence in
§16 (time, urgency and rack choice) makes it *plausible* that the richer observation contributes. Isolating it would
need, for example, DQN on a one-hot of the 144 states with the 9 macro-actions.

## 19. Reproducibility

CPU training is bit-for-bit deterministic for a given seed:
- one thread, `torch.use_deterministic_algorithms(True)`;
- separate seeded streams for initialisation, exploration and replay sampling;
- a seeded environment for every episode.

The tests check that the same seed gives identical initial weights, training rows and final weights after a short run,
and that a different seed gives different weights. A **second full 300 × 1440 masked run** produced bit-identical
network weights, optimiser state and training CSV (see the final report). CUDA is supported via `device: cuda` but was
not used, and GPU kernels may not be bit-reproducible.

## 20. Limitations

1. **Not budget-matched with Q-Learning** (30 % of its environment steps), and not a one-variable comparison (§18).
2. **One training seed per variant.** The ± values are evaluation-day variation, not training-run variation. The
   non-monotone training curve suggests the final policy depends on when training stops.
3. **Final-network evaluation.** There is no checkpoint selection and no held-out validation days.
4. **Small replay buffer** (about 7 days) and constant ε = 0.05: the policy keeps drifting (§12).
5. **Restless control:** 838 cooling changes and 245 migrations a day. The switching penalty is too small to
   discourage this, and the reward was deliberately not changed.
6. **Thermal regression** compared with Q-Learning (31 vs 6 critical minutes a day).
7. **Environment constraints** are unchanged: one placement per minute; delay rotates the queue.
8. **Truncation treated as terminal** (1 / 1440 transitions), as specified. This differs from Q-Learning's handling.
