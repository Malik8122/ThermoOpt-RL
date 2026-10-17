# Tabular Q-Learning (Step 9)

| | |
|---|---|
| Code | `backend/rl/q_learning.py` (agent, training loop, controller, save/load) |
| Train | `python -m backend.rl.q_learning` → `models/q_learning/{agent.json, q_table.npz}`, `experiments/q_learning_training.csv` (about 27 min) |
| Evaluate | `python -m backend.rl.evaluate_q_learning` → `experiments/q_learning_5x1440min.json`, `experiments/plots/q_learning/` |
| Tests | `tests/test_q_learning.py`: 30 tests, including 3 toy MDPs with known solutions |

> **Summary.** Q-Learning uses the same 144-state / 9-action abstraction as PI/VI, learns only from `env.step`
> and starts from Q = 0. Its greedy policy scores **−751 ± 121**. That is 457 better than PI/VI (−1208), and SLA
> failures drop from 191 to **80** a day. It is still behind round-robin (−669, 56 failures) and rule-based
> (−688, 63 failures). A 2 × 2 ablation shows that two things contribute to the gain over PI/VI:
> - **learning from its own experience**: in state 5, where PI/VI delayed, it now places the job;
> - **acting on the live action mask**, which reveals concrete information the abstract state hides.
>
> Neither factor alone comes close to −751. Direct experience removes the *sampled-model* error, but the
> *state-aliasing* limit remains. With the same abstraction, no tabular method beats the two hand-written baselines.

## 1. Theory

Q-Learning is a model-free, off-policy temporal-difference method. It estimates the optimal action-value function
from sampled transitions: each update moves Q(s,a) towards a one-step bootstrapped target built from the greedy value
of the next state. Under the usual conditions (a Markov state, every pair visited infinitely often, decaying step
sizes), Q converges to Q\*. **None of those conditions fully hold here** (§12): the state is aliased and the step
size is constant. So Q is an estimate that keeps fluctuating, not a converged Q\*.

## 2. Update rule

For a transition (s, a, r, s′), with M(s′) the environment's actual valid-action set in s′:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma \max_{a' \in M(s')} Q(s',a') - Q(s,a)\Big]$$

For a terminal transition, the target is just r. The data-centre environment never terminates: the end of a 24-hour
episode is a time-limit truncation, so the last transition is still bootstrapped. The terminal branch is exercised
by the toy MDP tests.

## 3. State representation

The existing `TabularStateEncoder`, unchanged: 144 states from hottest-rack temperature (4 levels) × utilisation (3)
× cooling set-point (3) × carbon (2) × queue (2). No urgency, rack identity, time, cooling-ramp state or forecast was
added.

## 4. Action representation

The 9 existing macro-actions, unchanged: {place coolest, place least-utilised, delay} × {decrease, maintain,
increase}. The Q-table has shape **144 × 9**, and the code asserts it.

## 5. Epsilon-greedy exploration

With probability ε the agent picks uniformly among the **currently valid** actions (exploration). Otherwise it picks
the greedy action among them (exploitation). Every decision is counted as one or the other in the training CSV. The
schedule is per episode:

$$\varepsilon_e = \max(\varepsilon_{\min},\ \varepsilon_0 \cdot d^{\,e}) = \max(0.05,\ 0.995^{e})$$

ε reaches 0.05 at episode 598 and stays there for the remaining 402 episodes. All random draws come from one
generator seeded with `derive_seed(42, "q-learning", "exploration")`. Nothing uses global NumPy or Python
randomness.

## 6. Hyperparameters

| Parameter | Value | Config key (`rl.q_learning`) |
|---|---|---|
| learning rate α | 0.10 (constant) | `learning_rate` |
| γ | 0.95 | `gamma` |
| ε start / min / decay | 1.0 / 0.05 / 0.995 per episode | `epsilon_start`, `epsilon_min`, `epsilon_decay` |
| episodes × length | **1000 × 1440 min = 1.44 M steps** | `episodes`, `episode_minutes` |
| start hour | random per episode (seeded) | `randomize_start_hour` |
| training seed | 42 | `seed` |
| tie tolerance | 1e-4, ties → lowest index (as PI/VI) | `tie_tolerance` |
| initial Q | 0 for all 1296 entries | fixed |

Before this step the config held an unused placeholder (`alpha: 0.1`, `episodes: 500`). It was renamed to
`learning_rate` and set to the Step 9 protocol of 1000 × 1440. Nothing changes during training. The final run used
the full configured protocol; the training script labels any shorter run a "development run".

**Initial values.** Q = 0 is deterministic and independent of PI, VI and the baselines; tests check this. Per-step
rewards here are almost always negative; only the small completion bonus can make one positive. So Q = 0 is *optimistic*: an action
not yet tried looks better than a tried one. That encourages exploration, a side-effect of the specified
initialisation that is worth knowing about.

## 7. Masking

Action validity depends on the concrete simulator state, not the abstract one. The agent therefore asks the
environment for `env.action_masks()` at every step:
- Exploration and exploitation both choose only among valid actions. As a result **no invalid action was taken in
  training or evaluation**.
- The target maximises over the valid actions of s′ only, so an invalid action's value can never enter a target.
  Tests check this with a deliberately huge Q on a masked action.
- Q-values of invalid actions stay in the table, but they are selected only when the mask allows them.
- The saved **policy table** π(s) uses the same admissible-set rule as PI/VI: actions observed valid at least once
  in s. States never visited fall back to *delay + maintain*. In practice every state was visited.

**Consequence: Q-values are conditional on validity.** Q(s, *place*) is only updated on steps where placing was
valid, that is, where the head job fitted on some rack. It therefore estimates "the value of placing, *given that it
is possible*". Acting greedily over the live mask uses that information correctly. Following the policy table blindly
does not: it chose invalid actions **50 %** of the time (§11). PI/VI's model instead averages the invalid outcomes
into Q.

## 8. Training protocol

1000 episodes of 24 hours. Episode *e* resets the environment with seed `derive_seed(42, "q-learning", "episode", e)`,
a different simulated day from the evaluation seeds 42–46, and a random start hour drawn from the environment's
seeded generator. Every episode is logged to `experiments/q_learning_training.csv`, one row per episode with 24
columns. The columns cover the episode's reward, energy, IT and cooling energy, carbon, cost, SLA failures and
completed jobs, critical and warning minutes, ε, explore and exploit steps, invalid actions, Q-update statistics,
and cumulative coverage.

The training loop is separate from the PI/VI model. It never imports or calls the model loader or builder; a test
patches all three to raise an error and still trains. The Q-table starts at zero.

## 9. Convergence diagnostics

| | first 100 episodes | last 100 episodes |
|---|---|---|
| total reward (while exploring) | −1199.6 | −808.1 |
| critical rack-minutes | 394.6 | 28.2 |
| SLA failures | 87.8 | 87.3 |
| energy (kWh) · carbon (kg) | 562.8 · 239.3 | 570.6 · 241.8 |
| max \|ΔQ\| in an episode | 1.37 | 1.12 |
| mean \|ΔQ\| per update | 0.095 | 0.086 |
| Q entries changed per episode | 457 | 183 |

The moving-average reward rises for about 250 episodes and then **plateaus** at about −810 (`reward.png`). Almost
all of the improvement is thermal: critical minutes fall from 395 to 28. SLA failures during training stay at about
87 throughout (`sla_failures.png`). The per-update |ΔQ| does **not** decay (`q_updates.png`), for two reasons:
- **Constant α with noisy targets.** With α = 0.1 held constant, each update keeps moving Q by about 10 % of the
  current TD error, and that error never shrinks to zero because the outcomes of a given (s, a) are random. In
  particular, one abstract state covers many concrete ones, so the "same" (s, a) produces different outcomes.
- **ε = 0.05** keeps injecting exploratory transitions.

The Q-table is therefore a **noisy, non-converged estimate**, which is expected under state aliasing. The greedy
policy, not the Q-values, is what is evaluated. Plots in `experiments/plots/q_learning/`: `reward.png`,
`epsilon.png`, `q_updates.png`, `sla_failures.png`, `energy.png`, `carbon.png`, `critical_minutes.png` (raw
per-episode data plus a 50-episode moving average; the raw data is in the CSV).

## 10. Evaluation protocol

The shared runner is used with seeds 42–46, 24 hours from 08:00, exactly as in Steps 6–8. The controller is
**greedy (ε = 0)**. A test checks that evaluation makes no random draws at all. It chooses the argmax of Q over the
live action mask. Earlier methods' numbers are **inherited** from their result files and were not re-run. The script
checks they used the same seeds and episode length.

Two **ablations**, newly measured in this step, separate the causes (§11):
- **Q-Learning policy table, no mask:** π(s) from the Q-table, ignoring the live mask, exactly as PI/VI act.
- **PI/VI Q-values with the live mask:** greedy over the live mask using Q computed from the Step 7 model and PI's
  value function. The Step 7 model is read here for *evaluation only*.

## 11. Results and comparison with PI/VI

### 5 × 24 h evaluation (greedy, live mask)

| Metric | Q-Learning (mean ± std) |
|---|---|
| Total reward | **−751.2 ± 121.2** |
| Energy / IT / cooling (kWh) | 575.9 ± 17.5 / 374.8 ± 9.6 / 201.1 ± 8.6 |
| Carbon (kg) · cost (₹) | 244.2 ± 8.5 · 4478.7 ± 135.5 |
| Mean / peak temperature (°C) | 22.1 ± 0.2 / 30.4 ± 0.3 |
| Warning / critical rack-minutes | 147.8 ± 62.1 / 6.0 ± 7.0 |
| Completed jobs / SLA failures | 352.2 ± 15.2 / **80.2 ± 24.4** |
| Delayed jobs · mean arrival → completion | 128.0 ± 13.5 · 31.3 ± 2.6 min |
| Cooling changes · migrations · invalid-action rate | 881 ± 71 · 0 · 0.000 |
| States visited in evaluation | 110 of 144 (PI/VI: 61) |

### All controllers

| Controller | Source | Reward | Energy kWh | Carbon kg | Cost ₹ | Critical min | SLA failures | Invalid |
|---|---|---|---|---|---|---|---|---|
| Random | Step 6, inherited | −1588 ± 193 | 557 | 240 | 4297 | 717 | 96 | 0 |
| Round-robin | Step 6, inherited | **−669 ± 113** | 614 | 263 | 4737 | 26 | **56** | 0 |
| Rule-based | Step 6, inherited | −688 ± 117 | 567 | 239 | 4410 | 25 | 63 | 0 |
| Policy Iteration | Step 7, inherited | −1208 ± 108 | **499** | 218 | **3901** | **0.4** | 191 | 0.131 |
| Value Iteration | Step 8, inherited | −1208 ± 108 | **499** | 218 | **3901** | **0.4** | 191 | 0.131 |
| **Q-Learning** | **Step 9, new** | **−751 ± 121** | 576 | 244 | 4479 | 6.0 | 80 | 0 |
| *ablation: QL table, no mask* | Step 9, new | −1079 ± 154 | 515 | **215** | 4064 | 17.4 | 145 | 0.503 |
| *ablation: PI/VI Q + live mask* | Step 9, new | −1126 ± 75 | 501 | 219 | 3917 | 1.8 | 176 | 0 |

### The 2 × 2: learning from experience vs. using the live mask

| | Policy table (no live mask) | Greedy over live mask |
|---|---|---|
| **Model-based (PI/VI Q, random-policy model)** | −1208 · SLA 191 | −1126 · SLA 176 |
| **Model-free (Q-Learning, own experience)** | −1079 · SLA 145 | **−751 · SLA 80** |

- **Learning from experience helps under either execution mode:** +129 without the mask, +375 with it.
- **The live mask helps both methods**, but Q-Learning far more (+328 vs +82). This interaction is what §7
  predicts. Q-Learning's values are conditional on validity, so they pair naturally with a mask. PI/VI's Q already
  averages the invalid outcomes in, so the mask adds less.
- Only the combination reaches −751. Attributing the gain to "model-free learning" alone would be wrong.

### SLA performance

Q-Learning fails 80 jobs a day: 58 % fewer than PI/VI (191), but 43 % more than round-robin (56) and 27 % more than
rule-based (63). The failure mode Step 7 diagnosed has changed. In **state 5** (LOW / LOW / MEDIUM / LOW / PENDING)
and **state 43** (MEDIUM / LOW / MEDIUM / HIGH / PENDING), where PI/VI *delayed* the job, Q-Learning **places it**:

| State | PI/VI action | Q-Learning action | Q-Learning Q-values (best · delay + maintain) | Training visits |
|---|---|---|---|---|
| 5 | delay + maintain | place coolest + decrease | −13.76 · −15.60 | 2 887 |
| 43 | delay + maintain | place coolest + decrease | −7.10 · −8.13 | 15 387 |
| 67 (MED/HIGH/MED/HIGH/PENDING) | delay + maintain | delay + increase | −11.21 · −12.77 | 12 604 |

Q-Learning's own estimate of *delay + maintain* in state 5 is −15.6, against V(5) = −5.8 in the PI/VI model. That
gap is the model error Step 7 diagnosed, now measured from direct experience. In state 67 (HIGH utilisation)
Q-Learning still delays, but raises cooling.

### Energy and carbon

Q-Learning spends more energy (576 kWh) and carbon (244 kg) than PI/VI (499, 218), because it completes 111 more jobs
a day (352 vs 241). It uses slightly more energy than rule-based (567 kWh / 239 kg) and less than round-robin
(614 kWh / 263 kg). Its cooling is **restless: 881 set-point changes a day**, more than rule-based (648). The switching
penalty is small, so nothing in the reward discourages it much.

### Policy agreement with PI/VI

On the 132 states both methods cover, the Q-Learning table and the PI/VI policy choose the same action in **31
(23.5 %)**. Evaluation action mix, over 7200 steps: Q-Learning *delay + maintain* 2062 against PI/VI's 2789, and
*place coolest + maintain* 579 against 2981. Q-Learning spreads its choices far more across cooling up and down
(full table in the results JSON, `action_frequency_in_evaluation`).

## 12. Model-free vs model-based, and on-policy coverage

| | |
|---|---|
| States visited by Q-Learning (training) | **144 / 144 (100 %)** |
| State–action pairs taken / observed valid | 1294 / 1296 taken; all 1296 were valid at least once |
| States in the PI/VI model | 132 (98 from random collection + 34 from closure) |
| Common to both | 132 (every PI/VI state was also visited by Q-Learning) |
| Visited only by Q-Learning | 12: 24, 25, 28, 40, 44, 60, 116–119, 128, 142 (exactly the 12 the PI/VI model never reached) |
| Q-Learning states missing from the *random collection* | 46 states, together only **2.9 %** of Q-Learning's training steps |
| States visited in evaluation | Q-Learning 110, PI/VI 61 |

The on-policy distribution discovers the 12 missing states, but they are rare and rarely matter. The states that
dominate training (for example 2, 6, 19, 38, 18, 55, 23, 54, each 3–9 % of steps) were all in the random collection.
**Coverage was therefore not the problem.** What differs is *which concrete situations stand behind each abstract
state*. The PI/VI model estimated each state's outcomes from situations created by the random collection policy.
Q-Learning's targets come from the situations its own behaviour creates, and they record what actually followed.
In state 5 that includes the waiting jobs that expired. Discovering a state does not by itself make the policy better. Here the benefit comes from
**on-policy outcome statistics in states PI/VI already had**, not from the new states.

## 13. State-aliasing limitation

**Q-Learning removes the sampled-transition-model limitation. It does not remove state aliasing.** Q(s,a) is still
indexed by the same 144 aggregated states, so it averages over every concrete situation that maps to s: which rack
is hot, how urgent the head job is, the time of day, and the cooling ramp. The evidence that aliasing still limits
the result:
- Training SLA failures never improve (about 87 per episode from start to finish). The agent cannot see which
  waiting jobs are about to expire, so it cannot learn to prioritise them.
- The Q-table does not settle: the mean |ΔQ| stays at about 0.09.
- The largest single gain comes from the **live mask**, the one channel through which concrete-state information
  reaches the agent.
- Two very simple heuristics that look at concrete state (round-robin checks whether the job fits each rack;
  rule-based reads per-rack temperatures and head-job slack) still beat it.

These observations are *consistent with* state aliasing being the dominant remaining limitation. They do not prove
it. A controlled ablation, such as adding head-job urgency to the state, would be the direct test.

## 14. Reproducibility

Training is a deterministic function of (config, seed). The tests train twice with seed 5 and get bit-identical
Q-tables and per-episode rows, and seed 6 gives a different Q-table. A second **full** 1000 × 1440 training run
reproduced the saved Q-table, the counters and all 1000 CSV rows **bit-for-bit**. Greedy evaluation makes no random
draws.

## 15. Limitations

1. **State aliasing** (§13): the 144-state abstraction hides job urgency, rack identity, time and cooling-ramp state.
2. **Constant α, ε floor 0.05**: the Q-table never converges, so the greedy policy is a snapshot of a still-moving
   table (the one after episode 1000).
3. **Validity-conditional Q-values** (§7): the learned table is only safe with the live mask. Without it, half the
   actions are invalid.
4. **One training seed.** Seed-to-seed variation of the *learned policy* was not measured, because a full run takes
   about 27 minutes. The ± values reflect evaluation-day variation only.
5. **One placement per minute and no migration** in the tabular actions (unchanged).
6. **Restless cooling** (881 changes a day): the reward's switching penalty is too small to discourage it. This was
   deliberately not changed.
