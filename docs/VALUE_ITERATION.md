# Value Iteration (Step 8)

| | |
|---|---|
| Code | `backend/rl/value_iteration.py`; shared helpers `q_values`, `greedy_actions` and `model_hash` in `backend/rl/tabular_model.py` |
| Train | `python -m backend.rl.value_iteration` → `models/value_iteration/policy.json` (0.04 s; loads the Step 7 model) |
| Evaluate | `python -m backend.rl.evaluate_value_iteration` → `experiments/value_iteration_5x1440min.json`, `experiments/plots/value_iteration/` |
| Tests | `tests/test_value_iteration.py`: 23 tests (11 on toy MDPs, 12 on the saved Step 7 model) |

> **Summary.** Value Iteration was run on the *same* saved empirical model as Policy Iteration, with model hash
> `6adc8b7a…563c`. It converges in 250 Bellman-optimality sweeps. Its value function differs from Policy Iteration's
> by at most 3.0 × 10⁻⁵ and always by a positive amount, as theory predicts. It chooses the same action in all 132
> modelled states. Its 5 × 24 h evaluation is therefore bit-identical to Policy Iteration's. This shows that both
> algorithms solve the same approximate MDP consistently. It says nothing new about how good that MDP's optimal
> policy is in the real simulator: it still fails 191 jobs a day.

## 1. Theory

Value Iteration computes the optimal value function directly. It repeatedly applies the Bellman *optimality*
operator T, a γ-contraction in the max-norm, so $V_k \to V^*$ from any starting point. The greedy policy with
respect to $V^*$ is optimal. Policy Iteration reaches the same fixed point differently: each outer iteration fully
evaluates one policy, then improves it. One VI iteration is one backup. One PI iteration is a full policy evaluation
(here 145–300 sweeps) plus an improvement step. **Iteration counts are not a speed comparison.**

## 2. Bellman optimality equation

$$V_{k+1}(s) = \max_{a \in \mathcal A(s)} \sum_{s'} \hat P(s'|s,a)\big[\hat R(s,a,s') + \gamma V_k(s')\big]$$

Updates are synchronous (Jacobi): every state uses $V_k$. After convergence:

$$Q(s,a) = \sum_{s'} \hat P(s'|s,a)\big[\hat R(s,a,s') + \gamma V(s')\big], \qquad
\pi(s) = \arg\max_{a\in\mathcal A(s)} Q(s,a)$$

Q-values within `tie_tolerance` = 1e-4 of the maximum count as ties, and the **lowest action index** wins. Unlike
Policy Iteration there is no current action to keep, so VI needs no "keep current" rule. `q_values` and
`greedy_actions` are the same functions Policy Iteration uses.

**Initial values:** $V_0(s) = 0$ for every state. It is deliberately *not* initialised from Policy Iteration's
values, so the two runs are independent.

## 3. State and action representation

This is unchanged from Step 7 (POLICY_ITERATION.md §2–3): 144 aggregated states (hottest-rack temperature ×
utilisation × cooling set-point × carbon × queue) and 9 macro-actions ({place coolest, place least-utilised, delay}
× {decrease, maintain, increase}).

## 4. Reused empirical model

Value Iteration **does not build, resample or modify** the model. It loads `models/policy_iteration/model.npz`
(configured by `rl.value_iteration.model_dir`) with `load_source_model`:

| | |
|---|---|
| Model hash (SHA-256 of counts, reward sums, validity counts, snapshots per state) | `6adc8b7a9cac1fa5a9c44a58f0dce178fdfd305ddd07a110985fd1a4d59d563c` |
| Modelled states / sampled transitions | 132 / 118 800 (132 × 9 × 100) |
| Built by | Step 7, `build_empirical_model`, seed 42 |

**Artifact-format change (the smallest one possible).** The Step 7 `policy.json` did not record a hash. So:
1. `model_hash()` was added.
2. Policy Iteration's `save()` now writes `model.model_hash`, and its `load()` checks the hash when one is present.
3. The hash of the existing `model.npz` was written into the existing `models/policy_iteration/policy.json`. The
   `model.npz` file itself was not touched (its SHA-256 was checked before and after), and nothing was retrained.

Value Iteration's `policy.json` records the source directory and the same hash, and `load()` refuses a model whose
hash differs. A test checks that the VI module cannot reach the model builder: it doesn't import it, and patching
the builder to raise changes nothing. Training also refuses to run if `value_iteration.gamma` differs from the gamma
stored with the source model.

## 5. Hyperparameters

| | Value Iteration | Policy Iteration (Step 7) |
|---|---|---|
| γ | 0.95 | 0.95 |
| θ | 1e-6 (Bellman sweep) | 1e-6 (policy-evaluation sweep) |
| max iterations | 1000 sweeps | 100 outer iterations, ≤ 1000 sweeps each |
| tie tolerance | 1e-4 | 1e-4 |
| initialisation | V₀ = 0 | π₀ = lowest admissible action, V = 0 |

## 6. Convergence criterion and results

The loop stops when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$. Every iteration records the max and mean change, how
many modelled states changed by at least θ, and the min, mean and max of V.

| | |
|---|---|
| Iterations | **250** (converged) |
| Final max change | 9.61e-7 |
| Final Bellman residual $\max_s \lvert (TV)(s) - V(s)\rvert$ | 9.13e-7 (Policy Iteration's values: 9.04e-7) |
| Contraction | The max change shrinks by about 0.95 per sweep after the first ~15 sweeps (`bellman_delta.png`) |
| Monotone | Mean and max V decrease at every iteration (`value_summary.png`). All rewards are negative, so starting from 0, $V_k$ decreases towards V\* |

## 7. Invalid-action handling

This is identical to Policy Iteration, because it is the same `EmpiricalModel.to_mdp()` output:
- Every sampled transition, including invalid ones (a no-op plus the environment's −0.1 penalty), is in $\hat P$ and
  $\hat R$.
- $\mathcal A(s)$ holds the actions that were valid in at least one sample.
- Inadmissible actions get Q = −∞ and are never selected; a test checks this.
- The 12 unmodelled states are absorbing, with only *delay + maintain* allowed.

**Limitation (unchanged):** action validity depends partly on simulator details hidden by the 144-state abstraction.
Examples are whether HIGH cooling means 80 % or 100 %, and whether the head job fits on any rack. So an admissible
action can still be invalid in a particular concrete state. This happened on 13 % of evaluation steps for both
algorithms.

## 8. PI vs VI methodology

The two algorithms receive the same `TabularMDP` (same model hash), the same γ, the same tie tolerance and the same
Q-value function. Only the dynamic-programming procedure differs. Comparison:
1. Values on the 132 modelled states: max and mean |ΔV|, and the sign of the difference.
2. The action chosen in each modelled state.
3. Q-values of each chosen action under each algorithm's own V, plus the best competing actions.
4. How close each decision is: the gap between the best and second-best Q.

## 9. Policy and value agreement

| Measure | Result |
|---|---|
| max / mean \|V_VI − V_PI\| (132 states) | **2.99e-5 / 2.55e-5** |
| Theoretical bound $2\gamma\theta/(1-\gamma)$ | 3.8e-5 |
| Sign of V_VI − V_PI | positive in all 132 states (1.3e-5 to 3.0e-5) |
| Same action | **132 / 132 (100 %)**; 0 disagreements |
| Value statistics, VI (PI) | min −28.6179 (−28.6179), max −3.4832 (−3.4832), mean −8.7902 (−8.7903), std 4.7172 (4.7172) |

**Why the difference is positive and about 3e-5.** Rewards are negative, so VI starting from 0 approaches V\* from
above, and stops within $\gamma\theta/(1-\gamma) \approx 1.9\times10^{-5}$ of it. PI's final policy evaluation is
warm-started from the previous, worse policy's values. It therefore approaches $V^{\pi^*} = V^*$ from below and
stops within the same bound. The two errors have opposite signs and add up, which is why the gap is below
3.8e-5 but not close to 0. This is a stopping-tolerance effect, not a modelling disagreement. A test enforces both
the bound and the sign.

**Disagreements.** There are none, so this section lists the **10 closest decisions** instead, chosen by a fixed rule
(the smallest gap between the best and second-best Q under VI). These are where tie-breaking, tolerance or sampling
noise could flip an action:

| State (T/W/C/K/Q) | PI = VI action | Q (VI) | Runner-up | Gap |
|---|---|---|---|---|
| 66 MED/HIGH/MED/HIGH/EMPTY | place coolest + maintain | −7.0959 | place least-utilised + maintain | 3.2e-5 |
| 70 MED/HIGH/HIGH/HIGH/EMPTY | place least-utilised + decrease | −7.0068 | delay + decrease | 6.0e-4 |
| 78 HIGH/LOW/MED/HIGH/EMPTY | place coolest + maintain | −4.1667 | place coolest + increase | 1.0e-3 |
| 10 LOW/LOW/HIGH/HIGH/EMPTY | place coolest + decrease | −3.7625 | delay + decrease | 1.8e-3 |
| 32 LOW/HIGH/HIGH/LOW/EMPTY | place coolest + maintain | −8.7819 | place coolest + decrease | 2.1e-3 |
| 27 LOW/HIGH/LOW/HIGH/PENDING | place least-utilised + increase | −6.5893 | place coolest + increase | 2.3e-3 |
| 34 LOW/HIGH/HIGH/HIGH/EMPTY | place least-utilised + decrease | −6.9174 | delay + decrease | 2.6e-3 |
| 131 CRIT/MED/HIGH/HIGH/PENDING | place coolest + maintain | −7.0367 | place coolest + increase | 3.0e-3 |
| 46 MED/LOW/HIGH/HIGH/EMPTY | place least-utilised + decrease | −3.7157 | delay + decrease | 3.2e-3 |
| 72 HIGH/LOW/LOW/LOW/EMPTY | delay + maintain | −4.8646 | place coolest + maintain | 3.3e-3 |

Only state 66 is a tie within the 1e-4 tolerance. Both algorithms resolve it to the lowest index (action 1). PI
would keep its current action there, but its current action was already action 1, so the two rules happen to agree.

**Finding: many close calls are sampling noise between physically identical actions.** Six of the ten (states 66, 70, 10, 34, 46
and 72) are EMPTY-queue states whose two best actions differ only in placement: place coolest, place
least-utilised or delay. With no
waiting job, all three placement choices are the same no-op. The model still estimates them from independent sets of
100 samples, so their Q-values differ only by sampling noise. The chosen placement in those states is therefore
arbitrary but harmless. In those six states the two best candidates choose the same cooling command, which is the part that matters.
The other four close calls (78, 32, 27, 131) are genuine trade-offs, either between cooling commands or between
two racks for a waiting job, and are 1–3 × 10⁻³ apart.

## 10. 24-hour evaluation

The VI policy was run through `backend/evaluation/runner.py` on seeds 42–46, 24 h each from 08:00, exactly as in
Steps 6 and 7. Because the VI and PI policies are identical and the environment is deterministic per seed, **all
five episodes are bit-identical to Policy Iteration's** (checked in the results JSON as
`episodes_identical_to_policy_iteration: true`).

| Metric | Value Iteration (mean ± std) |
|---|---|
| Total reward | −1207.9 ± 108.5 |
| Energy / IT / cooling (kWh) | 499.2 ± 12.3 / 321.2 ± 10.7 / 178.0 ± 4.3 |
| Carbon (kg) · cost (₹) | 217.9 ± 5.2 · 3900.8 ± 94.7 |
| Peak temperature (°C) | 29.1 ± 0.7 |
| Warning / critical rack-minutes | 68.8 ± 24.8 / 0.4 ± 0.9 |
| Completed jobs / SLA failures | 241.4 ± 18.4 / 191.2 ± 23.2 |
| Delayed jobs · mean arrival → completion | 143.0 ± 12.0 · 46.4 ± 4.8 min |
| Cooling changes · migrations · invalid-action rate | 241 ± 31 · 0 · 0.13 ± 0.05 |
| States visited | 61 of 144, all modelled |

Action frequency during evaluation (steps over 5 × 1440):

| Action | PI | VI | Difference |
|---|---|---|---|
| place coolest + decrease | 209 | 209 | 0 |
| place coolest + maintain | 2981 | 2981 | 0 |
| place coolest + increase | 208 | 208 | 0 |
| place least-utilised + decrease | 116 | 116 | 0 |
| place least-utilised + maintain | 110 | 110 | 0 |
| place least-utilised + increase | 134 | 134 | 0 |
| delay + decrease | 315 | 315 | 0 |
| delay + maintain | 2789 | 2789 | 0 |
| delay + increase | 338 | 338 | 0 |

## 11. Comparison with the baselines

| Controller | Source | Reward | Energy kWh | Carbon kg | Cost ₹ | Critical min | SLA failures |
|---|---|---|---|---|---|---|---|
| Random | Step 6, inherited | −1588 ± 193 | 557 ± 16 | 240 ± 7 | 4297 ± 125 | 717 ± 150 | 96 ± 25 |
| Round-robin | Step 6, inherited | **−669 ± 113** | 614 ± 11 | 263 ± 5 | 4737 ± 88 | 26 ± 7 | **56 ± 23** |
| Rule-based | Step 6, inherited | −688 ± 117 | 567 ± 20 | 239 ± 9 | 4410 ± 164 | 25 ± 7 | 63 ± 23 |
| Policy Iteration | Step 7, inherited | −1208 ± 108 | **499 ± 12** | **218 ± 5** | **3901 ± 95** | **0.4 ± 0.9** | 191 ± 23 |
| **Value Iteration** | **Step 8, this run** | −1208 ± 108 | **499 ± 12** | **218 ± 5** | **3901 ± 95** | **0.4 ± 0.9** | 191 ± 23 |

The inherited rows are read from `experiments/baselines_5x1440min.json` and
`experiments/policy_iteration_5x1440min.json`. The script checks that both used the same seeds and episode length.

## 12. Approximate-MDP limitations

Everything in POLICY_ITERATION.md §10–11 and §16 still applies, because the model is the same:
- job urgency is hidden;
- rack identity is compressed;
- time of day is averaged;
- the cooling ramp is only partly represented;
- the model depends on the random collection policy;
- 12 states are unobserved;
- there is no migration action.

Value Iteration does not and cannot change any of these. The same state-5 "delay" decision drives the same 191 SLA
failures a day.

## 13. Interpretation

- **Agreement is the expected result, and it validates both implementations.** Two different algorithms, started
  independently, reach the same fixed point of the same Bellman operator. Their values agree to within the combined
  stopping tolerance, with the predicted sign, and their policies agree exactly.
- **Neither algorithm is "better".** VI reaches the answer in 250 cheap sweeps and PI in 6 expensive iterations
  (1337 sweeps). Both take well under a second on 144 states, and they return the same policy.
- **Optimal for the empirical tabular model ≠ optimal for the simulator.** Both algorithms return the optimal policy
  of $\hat P, \hat R$ on the 144-state abstraction. That policy scores −1208 in the real simulator, below two
  hand-written baselines. The gap between the model and the simulator (POLICY_ITERATION.md §11) is the limiting
  factor, not the choice of planner. A better planner cannot improve on this; only better information can: a richer
  state, an on-policy model, or model-free learning from the controller's own experience (Step 9).
