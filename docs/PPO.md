# Proximal Policy Optimisation (Step 11)

| | |
|---|---|
| Code | `backend/rl/ppo.py` (actor-critic, masked policy, GAE, clipped update, training loop, controller) |
| Train | `python -m backend.rl.ppo` → `models/ppo/` (12 min, CPU) · `--no-mask` → `models/ppo_nomask/` · `--timesteps 432000 --tag ppo_432k` → `models/ppo_432k/` |
| Evaluate | `python -m backend.rl.evaluate_ppo` → `experiments/ppo_5x1440min.json`, `experiments/plots/ppo/` |
| Tests | `tests/test_ppo.py`: 30 tests, including toy bandits and a truncation-bootstrap test |

> **Summary.** The design-protocol PPO (199 k steps, live mask), evaluated deterministically, scores **−720 ± 148**.
> On the same five days that is statistically level with DQN (paired −27 ± 84), rule-based (−32 ± 70) and round-robin
> (−51 ± 63), and ahead of Q-Learning on 4 of 5 days. Three further findings:
> 1. **PPO's policy is meant to be sampled.** Sampling from it scores −698. With a DQN-matched budget (430 k steps)
>    sampling scores **−677**, beating the same policy's deterministic action on all five days (+71 on average).
> 2. **PPO barely needs the mask.** Without it PPO scores −780 with 11 % invalid actions. DQN without the mask
>    collapsed to −1614 with 48 % invalid.
> 3. **It learns fast.** The sampled training reward reaches about −800 within 25 000 steps.
>
> No learned controller is clearly better than the two hand-written baselines.

## 1. Why PPO

DQN learns action values and acts greedily on them. PPO learns the policy directly: a probability distribution over
actions, improved by gradient ascent on a clipped surrogate objective, with a separate critic estimating state values
to reduce variance. It is on-policy (each batch of experience is used for a few epochs and then discarded) and is
usually more stable than DQN. It is also the base for the forecast-driven controller of Step 13.

## 2. Observation, actions and networks

The same inputs and outputs as DQN: the **30-feature observation** and the **18 primitive actions**, both unchanged.

```
actor   Linear(30,128) → Tanh → Linear(128,128) → Tanh → Linear(128,18)    logits
critic  Linear(30,128) → Tanh → Linear(128,128) → Tanh → Linear(128,1)     V(s)
```

The actor and critic are separate networks with orthogonal initialisation. The actor's last layer uses gain 0.01, so
the initial policy is almost uniform (tested: the largest and smallest probabilities differ by less than 0.02).

## 3. Masked policy

$$\pi(a|s) = \frac{\exp z_a}{\sum_{a' \in M(s)} \exp z_{a'}}\ \text{for } a \in M(s), \qquad \pi(a|s) = 0 \text{ otherwise}$$

Invalid actions get logit −∞ before the softmax, so they have probability exactly 0, in three places: when sampling
during rollouts, when re-evaluating log-probabilities during the update (using the mask **stored with each
transition**), and at evaluation. Entropy is computed with the zero-probability terms removed, so it is never NaN.
Result: 0 invalid actions in training and evaluation.

## 4. Advantages (GAE)

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad A_t = \delta_t + \gamma\lambda A_{t+1}, \qquad R_t = A_t + V(s_t)$$

The recursion is reset at episode ends. $V(s_{t+1})$ is 0 after a terminal step. After the 24-hour **time-limit
truncation** it is the critic's value of the final observation (`truncation_is_terminal: false`), because the data
centre keeps running. A toy test shows the difference: with reward 1 per step and γ = 0.9, the critic learns a value
above 8.5 (true value 10) when bootstrapping, and below 6.5 when the limit is treated as terminal. This matches
Q-Learning's handling. DQN treated truncation as terminal, as its specification asked. Either way it affects 1 in
1440 transitions.

## 5. Update

For each rollout of 2048 steps: 10 epochs over shuffled minibatches of 64 (320 gradient steps per rollout).

$$L = -\mathbb E\big[\min(\rho A,\ \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon) A)\big] + c_v \cdot \tfrac12 (R - V)^2 - c_e H[\pi],
\qquad \rho = \frac{\pi_\text{new}(a|s)}{\pi_\text{old}(a|s)}$$

Advantages are normalised per minibatch. The optimiser is Adam with gradient-norm clipping. Rewards and returns are
**not** normalised, so the critic learns values around −50 directly.

## 6. Hyperparameters

| | Value | | Value |
|---|---|---|---|
| learning rate (Adam, constant) | 3e-4 | rollout steps | 2048 |
| γ | **0.99** | update epochs | 10 |
| GAE λ | 0.95 | minibatch | 64 |
| clip ε | 0.2 | gradient clip | 0.5 |
| entropy coefficient $c_e$ | 0.01 | value coefficient $c_v$ | 0.5 |
| hidden layers | 128, 128 (tanh) | seed | 42 |

All values are from PHASE1_DESIGN and `config.yaml → rl.ppo`. **PPO uses γ = 0.99, while the value-based agents use
0.95.** The evaluation metrics are undiscounted, so results are comparable. The agents do optimise slightly
different horizons: about 100 minutes against 20.

## 7. Training protocol

Three runs, fixed before any result was seen:

| | PPO (design) | PPO no mask | PPO 430 k | DQN (Step 10) |
|---|---|---|---|---|
| requested timesteps | 200 000 | 200 000 | 432 000 | 432 000 |
| environment steps (whole rollouts) | **198 656** | 198 656 | 430 080 | 432 000 |
| rollouts × gradient steps | 97 × 320 = 31 040 | 31 040 | 210 × 320 = 67 200 | 431 001 updates |
| complete 24 h episodes | 137 | 137 | 298 | 300 |
| wall-clock (CPU, 1 thread) | 709 s | 687 s | 1812 s | 3782 s |

- **Design run** is the primary result: the protocol in PHASE1_DESIGN.
- **430 k run** matches DQN's environment-step budget. It is reported alongside the design run, not instead of it.
  It shares the seed, so its first 97 rollouts are identical to the design run.
- Rollouts continue across episode boundaries. Episode *e* uses environment seed
  `derive_seed(42, "ppo", "episode", e)` and a random start hour.
- Logs: `experiments/ppo_training.csv` (per episode) and `experiments/ppo_updates.csv` (per rollout), and the same
  pair for `ppo_nomask` and `ppo_432k`.

## 8. Training diagnostics

| | first 10 updates | last 10 (design) | last 10 (430 k) |
|---|---|---|---|
| policy loss | −0.020 | −0.018 | −0.020 |
| value loss | 34.9 | 5.9 | 5.2 |
| critic explained variance | 0.76 | 0.95 | 0.95 |
| policy entropy (nats) | 2.24 | 1.55 | 1.48 |
| approximate KL | 0.008 | 0.010 | 0.010 |
| clip fraction | 0.12 | 0.14 | 0.14 |

- **Fast, stable learning.** The 10-episode average of the sampled reward rises from about −1250 to about −800 within
  25 000 steps and then drifts between −700 and −830 (`reward.png`). The largest KL in any update was 0.015 (design)
  and 0.017 (430 k), so updates stayed small. There were no NaNs, and the update raises an error on any non-finite
  statistic.
- **Not converged.** Entropy is still falling at the end of both runs (`entropy.png`). After 430 k steps the policy
  still has 66 % of the entropy of a uniform policy over the valid actions.
- **More steps changed little:** the last-20-episode sampled reward is −774 (design) and −744 (430 k).
- Episode means, first → last 20 episodes (design run): reward −928 → −774, SLA failures 79 in the last 20, critical
  rack-minutes 9.9 in the last 20.

Plots (x-axis: environment steps): `reward`, `sla_failures`, `energy`, `carbon`, `critical_minutes`, `invalid_rate`,
`policy_loss`, `value_loss`, `entropy`, `approx_kl`, `explained_variance`.

## 9. Evaluation protocol

The shared runner is used with seeds 42–46, 24 hours from 08:00, the same as Steps 6–10. Earlier controllers' numbers
are inherited from their result files.

- **Primary: deterministic.** The most probable valid action (exact ties → lowest index). This matches the greedy
  protocol of Steps 9–10 and was chosen before training. A test checks it draws no random numbers.
- **Secondary: stochastic.** Actions sampled from the policy, with a generator reseeded per evaluation day, so it is
  reproducible.

## 10. Results

### 5 × 24 h

| Metric | PPO design, deterministic | PPO design, stochastic | PPO 430 k, deterministic | PPO 430 k, stochastic |
|---|---|---|---|---|
| Total reward | **−719.8 ± 147.6** | −698.1 ± 134.6 | −748.0 ± 138.6 | **−677.0 ± 123.3** |
| Energy (kWh) | 588.5 ± 22.3 | 584.2 | 588.4 | 590.2 |
| Carbon (kg) · cost (₹) | 246.1 · 4605 | 244.6 · 4570 | 245.0 · 4616 | 246.6 · 4626 |
| Critical rack-minutes | 35.6 ± 57.7 | 12.6 ± 7.6 | 74.6 ± 64.6 | 10.0 ± 10.7 |
| SLA failures | 65.0 ± 26.6 | 64.6 | 61.0 | 59.0 |
| Invalid-action rate | 0 | 0 | 0 | 0 |

Design run, deterministic, remaining metrics: IT / cooling energy 383.4 / 205.0 kWh; mean / peak temperature
22.3 / 31.3 °C; warning-or-worse 250 rack-minutes; 367 completed jobs; 68 delayed jobs; mean arrival → completion
31.9 min; 658 cooling changes and 373 migrations a day.

### All controllers

| Controller | Source | Reward | Energy kWh | Carbon kg | Cost ₹ | Critical min | SLA failures | Invalid |
|---|---|---|---|---|---|---|---|---|
| Random | Step 6, inherited | −1588 ± 193 | 557 | 240 | 4297 | 717 | 96 | 0 |
| Round-robin | Step 6, inherited | **−669 ± 113** | 614 | 263 | 4737 | 26 | **56** | 0 |
| Rule-based | Step 6, inherited | −688 ± 117 | 567 | 239 | 4410 | 25 | 63 | 0 |
| Policy / Value Iteration | Steps 7–8, inherited | −1208 ± 108 | **499** | **218** | **3901** | **0.4** | 191 | 0.131 |
| Q-Learning | Step 9, inherited | −751 ± 121 | 576 | 244 | 4479 | 6.0 | 80 | 0 |
| DQN | Step 10, inherited | −693 ± 102 | 575 | 241 | 4502 | 31.2 | 61 | 0 |
| **PPO (design, deterministic)** | **Step 11, new** | **−720 ± 148** | 588 | 246 | 4605 | 35.6 | 65 | 0 |
| *PPO design, stochastic* | Step 11, new | −698 ± 135 | 584 | 245 | 4570 | 12.6 | 65 | 0 |
| *PPO 430 k, deterministic* | Step 11, new | −748 ± 139 | 588 | 245 | 4616 | 74.6 | 61 | 0 |
| *PPO 430 k, stochastic* | Step 11, new | −677 ± 123 | 590 | 247 | 4626 | 10.0 | 59 | 0 |
| *PPO no mask, deterministic* | Step 11, new | −780 ± 184 | 583 | 248 | 4520 | 24.6 | 82 | 0.106 |

### Paired per-seed reward differences (same 5 days)

| PPO variant minus… | Round-robin | Rule-based | DQN | Q-Learning |
|---|---|---|---|---|
| design, deterministic (primary) | −51 ± 63 (1/5) | −32 ± 70 (2/5) | −27 ± 84 (2/5) | +32 ± 58 (4/5) |
| design, stochastic | −29 ± 37 (0/5) | −10 ± 19 (2/5) | −5 ± 39 (3/5) | — |
| 430 k, deterministic | −79 ± 71 (0/5) | −60 ± 68 (0/5) | −55 ± 73 (1/5) | — |
| 430 k, stochastic | −8 ± 39 (2/5) | +11 ± 18 (3/5) | +16 ± 38 (2/5) | — |

The cells show mean ± standard deviation and, in brackets, the days on which PPO was better. With five days, none of
the differences against round-robin, rule-based or DQN is clearly distinguishable from zero, except that the 430 k
deterministic policy is consistently behind the baselines.

- **Energy and carbon:** 588 kWh / 246 kg, about 2 % above DQN and Q-Learning and below round-robin (614 / 263).
- **Thermal:** the deterministic policy has 36 critical rack-minutes a day, but with a very wide spread
  (28, 10, 1, 2 and **137** across the five days). Sampling gives 13.
- **SLA:** 65 failures a day; DQN 61, rule-based 63, round-robin 56.
- **Control behaviour:** 373 migrations and 658 cooling changes a day. PPO migrates more than DQN (245).

## 11. Deterministic vs stochastic execution

| | design | 430 k |
|---|---|---|
| stochastic − deterministic reward, per seed | +8, +24, −41, −13, +130 (mean +22) | +43, +171, +44, +58, +40 (**mean +71, 5/5**) |
| critical rack-minutes, deterministic | 28, 10, 1, 2, 137 | 75, 170, 97, 22, 9 |
| critical rack-minutes, stochastic | 19, 14, 2, 8, 20 | 10, 28, 8, 1, 3 |

PPO optimises the *sampled* policy. The trained policy is still spread out: the most probable action has an average
probability of 0.45 (0.48 after 430 k steps), and in 65 % of states no action reaches 0.5. Always taking the most
probable action is then a different policy from the one that was trained. For example, a state in which "increase
cooling" has probability 0.3 never increases cooling under the deterministic rule. That is consistent with the
deterministic policy's much higher critical minutes, and with the gap widening as training continues. The primary
result stays deterministic, as declared in advance, but **PPO's natural mode of execution is sampling**, and
the dashboard and Step 13 should probably use it. That is a decision to make, not a result to select after the fact.

## 12. Masking ablation

Same network, hyperparameters, seed and budget, with `use_action_mask = false`: the policy spreads probability over
all 18 actions, and the environment applies its own invalid-action semantics.

| | PPO masked | PPO no mask | DQN masked | DQN no mask |
|---|---|---|---|---|
| reward | −720 | −780 | −693 | −1614 |
| invalid actions (evaluation) | 0 % | 10.6 % | 0 % | 48 % |
| SLA failures | 65 | 82 | 61 | 277 |
| cooling changes · migrations per day | 658 · 373 | 175 · 36 | 838 · 245 | 198 · 138 |

Without the mask PPO loses 60 points and gains 17 SLA failures, but it does not collapse. Its invalid-action rate
during training falls as it learns, to 12 % over the last 20 episodes (`invalid_rate.png`). Two differences from
DQN are consistent with this, though neither was tested separately:
- PPO has no bootstrapped maximum over actions, so poorly estimated values of never-valid actions cannot leak into its
  targets.
- An invalid action directly lowers that action's own advantage through the −0.1 penalty, which pushes its probability
  down.

The no-mask policy migrates far less (36 a day against 373). "Migrate" is the action whose validity is not visible in
the observation, so avoiding it is the simplest way to avoid the penalty.

## 13. Does PPO use the richer observation?

Descriptive measurements over the 7200 deterministic evaluation steps of the design model. They are not causal.

- **Permutation sensitivity** (share of deterministic actions that change when one feature is shuffled): time of day
  35 % and 34 %, job-waiting flag 30 %, cooling set-point 26 %, head-job urgency 22 %, rack 3 GPU load 21 %,
  head-job flexibility 19 %. Rack temperatures score 6–8 %, carbon intensity 4 %, price 2 % and ambient 0.4 %. As
  with DQN, time and urgency, which the 144-state abstraction lacks, are among the most influential inputs. Carbon,
  price and ambient are strongly correlated with time of day, so their low scores don't show they are unused.
- **Rack choice:** when it places a waiting job, PPO picks the coolest feasible rack 53 % of the time. Rack 3, the
  badly cooled rack, gets the fewest placements: 13 %, against 42 %, 23 % and 21 % for racks 1, 2 and 4.
- **Cooling vs hottest rack:** the share of *decrease* falls from 27 % below 25 °C to 13 % at 27–30 °C and 2 % at
  30 °C and above. The share of *increase* rises from 20 % to 34 % at 27–30 °C. This is a clearer relationship than
  DQN's.
- **Migration** is used most between 25 and 30 °C (37–43 % of steps), and less below 25 °C (18 %).

## 14. Policy and value diagnostics (evaluation states)

| | design | no mask | 430 k |
|---|---|---|---|
| all probabilities and values finite | yes | yes | yes |
| policy entropy, mean (nats) | 1.56 | 1.83 | 1.48 |
| entropy relative to uniform over valid actions | 0.71 | 0.63 | 0.66 |
| top-action probability, mean | 0.45 | 0.41 | 0.48 |
| states where top action > 0.9 | 2.2 % | 3.1 % | 8.1 % |
| V(s): mean ± std, [min, max] | −51.8 ± 25.2, [−102, −20] | −56.7 ± 29.1, [−119, −21] | −48.9 ± 20.1, [−101, −20] |

The value estimates are about 8× DQN's Q-values (about −6). That is expected: γ = 0.99 sums about 100 future steps,
γ = 0.95 about 20. Representative states (seed 42, at quantiles of hottest-rack temperature) are in the results JSON.
At the coolest state (18 °C) the policy puts 0.73 on "place on rack 4 + decrease cooling". At the 75th percentile
(25.8 °C) its top three actions are all migrations (0.41, 0.31, 0.12). At the hottest state (32.8 °C) "increase
cooling" appears in its two most probable actions.

## 15. Comparison with DQN and Q-Learning

| | Q-Learning | DQN | PPO |
|---|---|---|---|
| input / actions | 144 states / 9 | 30 features / 18 | 30 features / 18 |
| learns | action values (table) | action values (network) | policy + state values (networks) |
| data use | off-policy, online | off-policy, replay | on-policy rollouts |
| γ | 0.95 | 0.95 | 0.99 |
| environment steps | 1 440 000 | 432 000 | 198 656 (and 430 080) |
| wall-clock | 27 min | 63 min | 12 min (30 min) |
| reward, primary evaluation | −751 | −693 | −720 |
| without the live mask | −1079 (table) | −1614 | −780 |

PPO and DQN share inputs and actions, so this is a closer comparison than DQN against Q-Learning. They still differ
in algorithm, γ, activation function, initialisation and budget, so it is not a one-variable comparison either. On
the evidence here they are equivalent in reward. PPO reaches that level with less than half the environment steps
and a fifth of the compute, and it is far more robust to losing the mask.

## 16. Reproducibility

CPU training is bit-for-bit deterministic for a given seed: one thread, deterministic kernels, and separate seeded
streams for initialisation, action sampling and minibatch shuffling. A second **full** design-protocol run reproduced
the network weights, the optimiser state and both CSVs exactly. The 430 k run's first 97 rollouts match the design
run. The tests check same-seed equality and different-seed inequality on short runs.

## 17. Limitations

1. **One training seed per variant.** The ± values are evaluation-day variation, not training-run variation.
2. **Not converged.** Entropy is still falling; the result is a snapshot of an unfinished policy.
3. **Deterministic execution does not suit this policy** (§11). The primary number understates what the trained
   policy does when sampled.
4. **No reward or return normalisation** and a constant learning rate: plain PPO, as designed, with no tuning.
5. **Different γ** from the value-based agents (0.99 against 0.95).
6. **Wide day-to-day spread** in critical minutes for the deterministic policy (1 to 137).
7. **Restless control:** 658 cooling changes and 373 migrations a day.
8. **Environment constraints** are unchanged: one placement per minute; delay rotates the queue.
