# Policy Iteration (Step 7)

| | |
|---|---|
| Code | `backend/rl/tabular_model.py` (empirical model), `backend/rl/policy_iteration.py` (algorithm, controller, save/load) |
| Train | `python -m backend.rl.policy_iteration` → `models/policy_iteration/{policy.json, model.npz}` |
| Evaluate | `python -m backend.rl.evaluate_policy_iteration` → `experiments/policy_iteration_5x1440min.json`, `experiments/plots/policy_iteration/` |
| Tests | `tests/test_policy_iteration.py` (27 tests: 12 on toy MDPs with known solutions, 15 on the data-centre model) |

> **Summary.** Policy Iteration converges to a stable policy in 6 iterations. It is the best controller so far on
> thermal safety (0.4 critical rack-minutes a day against about 25), energy (−19 % vs round-robin) and carbon
> (−17 %). However, it fails 191 jobs a day against 56–63, so its total reward (−1208) is worse than both hand-written
> baselines (about −680), though better than random (−1588). The failures come from the state abstraction, not the
> algorithm: the 144-state representation cannot see how urgent a waiting job is (§11).

## 1. Why Policy Iteration

Policy Iteration is the classical exact method for a known finite MDP. It alternates two steps: evaluate the current
policy exactly, then act greedily on those values. It converges in a finite number of iterations to a policy that is
optimal *for the model it is given*. That makes it a clean first learned controller. It separates "is the
algorithm correct?" (checked on toy MDPs with known answers) from "is the model of the data centre good enough?"
(checked by running the policy in the real simulator).

## 2. State representation: 144 states

The existing `TabularStateEncoder` (`backend/rl/state_encoder.py`) is used unchanged:

| Factor | Levels |
|---|---|
| hottest-rack temperature | LOW < 25 °C, MEDIUM 25–27, HIGH 27–30, CRITICAL ≥ 30 |
| mean CPU+GPU utilisation | LOW < 40 %, MEDIUM 40–70, HIGH ≥ 70 |
| cooling set-point | LOW < 40 %, MEDIUM 40–70, HIGH ≥ 70 |
| carbon intensity | LOW < 400, HIGH ≥ 400 g/kWh |
| queue | EMPTY, PENDING |

`s = (((T·3 + W)·3 + C)·2 + K)·2 + Q ∈ [0, 144)`

**This is an approximate, aggregated representation and it is not Markov.** Many different simulator states share
one abstract state, and the abstract state omits things that change what happens next:
- which rack is hot;
- the head job's size, flexibility and remaining slack;
- how many jobs are waiting;
- the clock time, which drives arrivals, carbon, price and ambient temperature;
- the delivered cooling level while the cooling ramps.

## 3. Action representation: 9 actions

The existing tabular macro-actions (`backend/rl/actions.py`) are used unchanged: `a = placement·3 + cooling`.

| a | placement | cooling |
|---|---|---|
| 0, 1, 2 | place head job on the coolest rack that fits it | decrease, maintain, increase |
| 3, 4, 5 | place head job on the least-utilised rack that fits it | decrease, maintain, increase |
| 6, 7, 8 | delay the head job | decrease, maintain, increase |

The environment resolves each macro-action to one of the 18 primitive actions for the current concrete state.

## 4. Empirical transition model

The model is not written by hand. No probability, reward or policy entry is hard-coded. Every number comes from
calling the unmodified `DataCenterEnv.step` (`build_empirical_model`):

1. **Collect concrete states.** Run 30 tabular episodes of 240 minutes each. The behaviour policy picks uniformly
   among valid actions (`env.action_masks()`), and each episode starts at a random hour, so both carbon levels and the
   whole daily cycle are covered. Every 5 minutes, a deep copy of the simulator is stored under its abstract state,
   keeping at most 50 per state (reservoir sampling). This reached **98 abstract states**.
2. **Sample transitions.** For each state *s* and each of the 9 actions *a*, take **100 one-step transitions**
   (`model_samples_per_state_action`). Sample *k* copies stored snapshot *k* mod *n_s*. It then reseeds that copy's
   arrival and thermal-noise random streams with `derive_seed(seed, s, a, k)`, so copies of the same snapshot have
   independent futures. Finally it calls `env.step(a)` and records *s′*, *r* and whether the action was valid.
3. **Closure.** If a sampled successor *s′* has no stored snapshot, the concrete successor state becomes its
   snapshot and *s′* is modelled in turn. This added **34 states**, giving **132 modelled states** and
   132 × 9 × 100 = **118 800 sampled transitions** (about 12 minutes on a laptop CPU).

The estimates are maximum-likelihood:

$$\hat P(s'|s,a) = \frac{N(s,a,s')}{N(s,a)}, \qquad \hat R(s,a,s') = \frac{1}{N(s,a,s')}\sum r .$$

The 12 states that were never reached (for example HIGH workload with a LOW temperature, or CRITICAL temperature with LOW workload and HIGH cooling) get an absorbing self-loop
with reward 0, and only action 7 (delay + maintain) is allowed there. Closure guarantees no modelled state can
transition into them, and none was visited during evaluation.

**Termination and truncation.** The environment never terminates; it is a continuing task truncated only by a time
limit. Each one-step sample starts with a fresh step counter, so there are no terminal transitions and every
successor is bootstrapped. The builder raises an error if a terminal transition ever appears.

**Model identity.** `model_hash()` (SHA-256 of the sample arrays) is stored in `policy.json` and checked on load. Value Iteration (Step 8) uses it to prove it solved exactly this model.

**Reproducibility.** Collection, snapshot choice and every sample's random streams are derived from a single seed
(`derive_seed`), so the model is a deterministic function of (config, seed, sample count).

## 5. Bellman expectation equation (policy evaluation)

For a deterministic policy π:

$$V^\pi(s) = \sum_{s'} \hat P(s'|s,\pi(s)) \big[\hat R(s,\pi(s),s') + \gamma V^\pi(s')\big]$$

Evaluation uses synchronous sweeps, $V_{k+1} = R_\pi + \gamma P_\pi V_k$, warm-started from the previous policy's
values. It stops when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$ or after `max_evaluation_sweeps`. The maximum change
of every sweep is recorded. On a random 5-state MDP the tests check the result against the exact linear solve
$(I - \gamma P_\pi)^{-1} R_\pi$.

## 6. Policy improvement

$$Q(s,a) = \sum_{s'} \hat P(s'|s,a)\big[\hat R(s,a,s') + \gamma V(s')\big], \qquad
\pi'(s) = \arg\max_{a \in \mathcal A(s)} Q(s,a)$$

The argmax covers only admissible actions (§9). Q-values within `tie_tolerance` = 1e-4 of the maximum count as ties.
If the current action is among the ties it is kept; otherwise the lowest action index wins. Keeping the current
action is the standard safeguard that stops Policy Iteration cycling between equally good policies. The tolerance is
larger than the evaluation error bound $\gamma\theta/(1-\gamma) \approx 1.9\times10^{-5}$, so evaluation noise
cannot produce a spurious "improvement".

**Initial policy:** π₀(s) = the lowest-index admissible action. This is deterministic and needs no random numbers.

**Loop:** evaluate → improve → stop when no state's action changes, or after `max_iterations`.

## 7. Discount factor

γ = 0.95, giving an effective horizon of $1/(1-\gamma)$ = 20 minutes. Values are therefore discounted sums over about
20 steps. They are not comparable in scale to the 1440-step episode returns in §14.

## 8. Convergence criterion and hyperparameters

| Parameter | Value | Config key |
|---|---|---|
| γ | 0.95 | `rl.policy_iteration.gamma` |
| θ (evaluation tolerance) | 1e-6 | `rl.policy_iteration.theta` |
| max outer iterations | 100 | `rl.policy_iteration.max_iterations` |
| max evaluation sweeps | 1000 | `rl.policy_iteration.max_evaluation_sweeps` |
| samples per (s, a) | 100 | `rl.policy_iteration.model_samples_per_state_action` |
| tie tolerance | 1e-4 | `rl.policy_iteration.tie_tolerance` |
| collection | 30 episodes × 240 min, snapshot every 5 min, ≤ 50 per state | `rl.tabular_model.*` |

Policy Iteration is **converged** when an improvement step changes no action. Each evaluation is **converged** when
the maximum value change drops below θ. Both are recorded per iteration in `policy.json → training.history`.

## 9. Invalid actions

Whether an action is valid depends on the concrete state, not the abstract one. HIGH cooling covers both 80 % and
100 %, and only one of those can be increased. Whether the head job fits on any rack depends on each rack's free
capacity. So the model handles invalid actions the way the environment does, in every calculation:

- **Every** sampled transition, valid or not, goes into $\hat P$ and $\hat R$. An invalid action is a no-op that
  still advances time, carries the environment's −0.1 invalid-action penalty and leads to wherever the simulator
  goes. Its estimated value therefore already includes its risk of being invalid.
- The **admissible set** $\mathcal A(s)$ holds the actions that were valid in at least one of the 100 samples.
  Actions that were never valid in *s* can never be chosen. Action 7 (delay + maintain) is valid in every concrete
  state, so $\mathcal A(s)$ is never empty.
- Inadmissible actions get $Q = -\infty$ in improvement. Their rows are never used by evaluation, because the
  policy never selects them.

This was preferred over "allow only actions valid in every sample". That rule would forbid placement in any state
where a large job ever failed to fit, forcing the policy to delay indefinitely in busy states.

**Consequence:** the policy can choose an action that turns out invalid in the concrete state. This happened on
13 % of evaluation steps. The model predicted the rate correctly (for example, 64 % validity predicted vs 58 %
observed in state 55), and it costs 0.1 per occurrence.

## 10. Time-varying environment

Arrival rate, electricity price, carbon intensity and ambient temperature all follow the clock. The abstract state
carries only the carbon level (LOW/HIGH), not the time. The estimated $\hat P$ and $\hat R$ are therefore **averages
over the times of day at which the collection policy visited each state**. Policy Iteration solves this stationary
average, not the true time-dependent MDP. For example, it cannot learn that a HIGH-carbon state at 11:00 is about
to become LOW-carbon, while the same state at 20:00 is not.

## 11. Approximate-MDP limitation, and what it does to the policy

Because the abstraction is not Markov, $\hat P$ and $\hat R$ depend on *which* concrete states stand behind each
abstract state. That in turn depends on the policy that generated the data. The model is built from a random policy,
but it is used to evaluate a very different policy. The evaluation shows the consequences directly. The table below
shows the most-used state-action pairs on seeds 42–43, the model's predicted one-step reward, and what actually
happened when the learned policy ran:

| State (T/W/C/K/Q) | Action | Steps | Model $\hat R(s,a)$ | Observed mean reward | Observed failures / step | Validity: model → observed |
|---|---|---|---|---|---|---|
| 5 LOW/LOW/MED/LOW/PENDING | delay + maintain | 610 | −0.288 | **−1.836** | **0.37** | 1.00 → 1.00 |
| 2 LOW/LOW/LOW/HIGH/EMPTY | place coolest + maintain | 346 | −0.125 | −0.149 | 0 | 1.00 → 1.00 |
| 55 MED/MED/MED/HIGH/PENDING | place coolest + maintain | 265 | −0.413 | −0.869 | 0.12 | 0.64 → 0.58 |
| 43 MED/LOW/MED/HIGH/PENDING | delay + maintain | 255 | −0.215 | −0.723 | 0.12 | 1.00 → 1.00 |
| 54 MED/MED/MED/HIGH/EMPTY | place coolest + maintain | 207 | −0.252 | −0.274 | 0 | 1.00 → 1.00 |

In states with an empty queue the model is accurate. In PENDING states the observed reward is 2–6× worse than predicted, because of SLA failures the model did not anticipate.
Under the random collection policy, a waiting job was usually placed within a few minutes, so a delay step rarely
coincided with a deadline expiring. Under the learned policy, jobs in state 5 are delayed repeatedly, and the same
abstract state now hides jobs close to their deadline. In state 5, placing has the *better* one-step reward (−0.212
vs −0.288). Policy Iteration still delays, because the model's future values say that keeping utilisation LOW is
worth more than the modelled SLA cost. This is a correct solution of the wrong MDP: an instance of state aliasing
combined with distribution shift between the data-collecting policy and the evaluated policy.

This result is reported, not fixed. Fixing it would mean adding head-job urgency or time to the state, or
re-collecting the model under the learned policy. The first changes the 144-state abstraction, which Step 7 keeps
fixed. The second is a separate method (iterated model-based RL) and could be a later ablation. Q-Learning
(Step 9) learns from its own on-policy experience and is a natural point of comparison.

## 12. Model sampling procedure (summary)

| | |
|---|---|
| Behaviour policy | uniform over valid tabular actions, random start hour |
| Collection | 30 × 240 min, snapshot every 5 min, ≤ 50 snapshots per state |
| States from collection / closure / total | 98 / 34 / 132 of 144 |
| Samples per (s, a) | 100, for all 9 actions of every modelled state |
| Total transitions | 118 800 |
| Wall-clock | 705 s in total; solving Policy Iteration itself takes 0.07 s |

## 13. Evaluation methodology

The saved policy runs in the tabular environment through the shared runner (`backend/evaluation/runner.py`), with
exactly the Step 6 protocol: seeds 42–46, 24 hours each, starting at 08:00, with the same metric definitions. The
baseline figures are **read from `experiments/baselines_5x1440min.json` (measured in Step 6) and not re-run**. The
evaluation script checks that both files use the same seeds and episode length. The model-collection episodes use
seeds derived from 42 through `derive_seed`, not the evaluation seeds themselves.

## 14. Results

### Convergence

| Outer iteration | Evaluation sweeps | Final max ΔV | Actions changed |
|---|---|---|---|
| 1 | 300 | 9.6e-7 | 129 |
| 2 | 296 | 9.6e-7 | 89 |
| 3 | 232 | 9.6e-7 | 44 |
| 4 | 204 | 9.6e-7 | 20 |
| 5 | 160 | 9.6e-7 | 7 |
| 6 | 145 | 9.5e-7 | **0 → stable** |

Every evaluation reached θ. There were 1337 sweeps in total. Plots: `value_delta_per_sweep.png` and
`policy_changes_per_iteration.png`.

### Learned policy

- **Value function** over the 132 modelled states: min −28.6, max −3.5, mean −8.8, std 4.7
  (`value_distribution.png`).
- **State coverage in evaluation:** 61 of 144 states visited. All 61 were modelled; no unmodelled state was ever
  reached.
- **Actions in the policy table** (132 modelled states): a spread over all 9 actions. *Delay + increase* is the most
  common, in 25 states.
- **Actions taken during evaluation** (`action_frequency.png`): place coolest + maintain 41.4 %, delay + maintain
  38.7 %, delay + increase 4.7 %, delay + decrease 4.4 %, place coolest + decrease or increase 2.9 % each,
  place least-utilised 1.5–1.9 % each.

The representative states below are the **10 most-visited states during evaluation**. They were chosen by that rule,
not by hand:

| State | Temp | Workload | Cooling | Carbon | Queue | Action | V(s) | Visits |
|---|---|---|---|---|---|---|---|---|
| 5 | LOW | LOW | MEDIUM | LOW | PENDING | delay + maintain | −5.8 | 1361 |
| 55 | MEDIUM | MEDIUM | MEDIUM | HIGH | PENDING | place coolest + maintain | −6.2 | 767 |
| 2 | LOW | LOW | LOW | HIGH | EMPTY | place coolest + maintain | −3.5 | 733 |
| 43 | MEDIUM | LOW | MEDIUM | HIGH | PENDING | delay + maintain | −3.8 | 665 |
| 19 | LOW | MEDIUM | MEDIUM | HIGH | PENDING | place coolest + maintain | −6.6 | 554 |
| 54 | MEDIUM | MEDIUM | MEDIUM | HIGH | EMPTY | place coolest + maintain | −5.8 | 514 |
| 42 | MEDIUM | LOW | MEDIUM | HIGH | EMPTY | delay + maintain | −3.6 | 360 |
| 3 | LOW | LOW | LOW | HIGH | PENDING | place coolest + maintain | −3.7 | 239 |
| 67 | MEDIUM | HIGH | MEDIUM | HIGH | PENDING | delay + maintain | −7.1 | 238 |
| 38 | MEDIUM | LOW | LOW | HIGH | EMPTY | delay + increase | −4.0 | 209 |

With an EMPTY queue, "place" and "delay" are both no-ops, so only the cooling part matters in states 2, 54, 42 and 38.

### 5 × 24 h evaluation

| Metric | Policy Iteration (mean ± std) |
|---|---|
| Total reward | −1207.9 ± 108.5 |
| Energy / IT / cooling (kWh) | 499.2 ± 12.3 / 321.2 ± 10.7 / 178.0 ± 4.3 |
| Carbon (kg) | 217.9 ± 5.2 |
| Cost (₹) | 3900.8 ± 94.7 |
| Mean / peak temperature (°C) | 21.7 ± 0.3 / 29.1 ± 0.7 |
| Critical / warning-or-worse rack-minutes | 0.4 ± 0.9 / 68.8 ± 24.8 |
| SLA failures | 191.2 ± 23.2 |
| Completed jobs | 241.4 ± 18.4 |
| Delayed jobs | 143.0 ± 12.0 |
| Mean arrival → completion (min) | 46.4 ± 4.8 |
| Cooling changes | 241 ± 31 |
| Migrations | 0 (not in the tabular action set) |
| Invalid-action rate | 0.13 ± 0.05 |

### Comparison with the Step 6 baselines

| Controller | Source | Reward | Energy kWh | Carbon kg | Cost ₹ | Critical min | SLA failures |
|---|---|---|---|---|---|---|---|
| Random | Step 6, measured earlier | −1588 ± 193 | 557 ± 16 | 240 ± 7 | 4297 ± 125 | 717 ± 150 | 96 ± 25 |
| Round-robin | Step 6, measured earlier | **−669 ± 113** | 614 ± 11 | 263 ± 5 | 4737 ± 88 | 26 ± 7 | **56 ± 23** |
| Rule-based | Step 6, measured earlier | −688 ± 117 | 567 ± 20 | 239 ± 9 | 4410 ± 164 | 25 ± 7 | 63 ± 23 |
| **Policy Iteration** | **Step 7, this run** | −1208 ± 108 | **499 ± 12** | **218 ± 5** | **3901 ± 95** | **0.4 ± 0.9** | 191 ± 23 |

**Trade-off.** Policy Iteration is best on every energy, carbon, cost and thermal metric. Part of that is genuine:
cooling is well matched to load, with near-zero critical minutes. Part of it is an artefact of doing less work: it
completes 241 jobs a day against 376 for round-robin, and a job that fails uses no energy. Its SLA penalty
(−768 on seed 42, against −78 for round-robin) outweighs everything it saves, so on the project's reward it ranks
third of four. The per-component reward breakdown for every episode is in the results JSON.

## 15. Reproducibility

Two full training runs with the same configuration, seed and sample count produced **bit-identical** results: the
sample counts, reward sums, validity counts, policy, value function and convergence history. The tests check the
same property at small scale, and check that a different seed gives a different model.

## 16. Limitations

1. **Not Markov (§2, §11).** The abstraction hides job urgency, rack identity, time and cooling-ramp state. That is
   the main cause of the SLA failures.
2. **Model depends on the collection policy (§11).** The model is estimated under a uniform-random policy and
   evaluated under a very different one.
3. **Stationary average of a time-varying system (§10).**
4. **Coverage.** 12 of 144 states were never reached during collection or closure. They are absorbing placeholders; none was
   visited in evaluation.
5. **Sampling error.** With 100 samples per (s, a), rare events such as a job failure in one particular minute
   have wide confidence intervals. The ranking of close Q-values can depend on the model seed.
6. **Invalid actions (§9).** Abstract admissibility doesn't guarantee concrete validity: 13 % of steps were invalid
   no-ops.
7. **No migration.** The 9 tabular macro-actions don't include migration, unlike the 18-action space used by the
   baselines.
