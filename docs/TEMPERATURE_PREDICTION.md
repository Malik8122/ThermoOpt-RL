# Temperature Forecasting (Step 12)

| | |
|---|---|
| Code | `backend/prediction/dataset.py` (features, history, windows) · `model.py` (network, training, live provider) · `evaluate.py` |
| Data | `python -m backend.prediction.dataset` → `data/temperature_history.npz` (20 simulated days) |
| Train | `python -m backend.prediction.model` → `models/temperature_predictor/` (13 min, CPU) |
| Evaluate | `python -m backend.prediction.evaluate` → `experiments/temperature_prediction.json`, `experiments/plots/temperature_prediction/` |
| Tests | `tests/test_prediction.py`: 29 tests |

> **Summary.** The configured 2-layer LSTM forecasts each rack's temperature 5, 10 and 15 minutes ahead. On held-out
> days its mean absolute error is **1.24 / 1.79 / 1.91 °C**. That beats "the temperature stays as it is"
> (persistence) by 35 % at 5 minutes, but by only 18 % at 10 and **10 % at 15 minutes**. It almost never predicts a
> rack entering the warning zone 15 minutes ahead (recall 0.8 %). The cause is physical: with a 4-minute thermal time
> constant, a rack's temperature 15 minutes from now is set by the load and cooling *then*, which depend on jobs and
> control actions that have not happened yet. Giving the model three extra signals that are already known (commanded
> cooling, jobs about to finish, ambient) improves the 15-minute error only to 1.80 °C. **For Step 13, the 5-minute
> forecast carries real information; the 10- and 15-minute forecasts carry little.**

## 1. Task

For one rack at minute *t*: given the last 10 minutes of its signals, predict its temperature at *t* + 5, *t* + 10
and *t* + 15 minutes. One model serves all four racks. The forecasts feed the 42-feature observation of Predictive
PPO (Step 13) and the dashboard.

## 2. Inputs

Six features per rack per minute, from `config.yaml → prediction.features`:

| Feature | Meaning | Unit |
|---|---|---|
| `temperature` | rack temperature | °C |
| `cpu_utilisation`, `gpu_utilisation` | used / capacity | 0–1 |
| `it_power` | rack IT power | kW |
| `cooling_level` | cooling the rack actually receives: delivered level × the rack's effectiveness | 0–1 |
| `queue_length` | jobs waiting (the same for every rack) | jobs |

`cooling_level` uses received cooling, not the shared plant level, so the model can tell the badly cooled Rack 3
(effectiveness 0.75) from the others. `rack_features(dc, names)` is the single definition of these inputs. The offline
dataset and the live provider both call it.

## 3. Data

**History.** 20 simulated 24-hour days, recorded once per minute (1441 frames × 4 racks per day). The design did not
say who controls the data centre during those days, and a forecaster only learns the control patterns it sees. Day *i*
is therefore driven by `behaviour_policies[i % 4]`: **random, round-robin, rule-based, and the Step 11 PPO policy
(sampled)**. Day *i* uses environment seed `derive_seed(42, "prediction", "day", i)`.

**Split: chronological, by whole days.** Days 0–13 train, 14–16 validate, 17–19 test (70 / 15 / 15). Windows never
cross a day boundary, so no test minute appears in a training window.

| | Days | Behaviour policies | Windows |
|---|---|---|---|
| train | 0–13 | all four, 3–4 days each | 79 352 |
| validation | 14–16 | rule-based, PPO, random | 17 004 |
| test | 17–19 | round-robin, rule-based, PPO | 17 004 |

**Generalisation days.** Five further days with new seeds, one per policy: the four above plus **DQN**, whose
behaviour is not in the training data.

**Target.** The model predicts the temperature *change* from now to each horizon. The forecast is the current
temperature plus that change. A zero output is then exactly the persistence forecast, so the model starts from the
strongest simple baseline and only has to learn the departure from it. Inputs are standardised with training-set
statistics, and each horizon's change with its training-set standard deviation.

## 4. Model and training

```
LSTM(6 → 64, 2 layers, dropout 0.1 between layers) → last time step → Linear(64 → 3)      51 907 parameters
```

| | Value |
|---|---|
| sequence length · horizons | 10 minutes · 5, 10, 15 minutes |
| optimiser · learning rate · batch | Adam · 1e-3 · 128 |
| loss | MSE of the standardised change |
| epochs | 30; the weights of the epoch with the lowest **validation** loss are kept |
| seed · device | 42 · CPU, 1 thread |

**Training curve** (`training_curve.png`). Validation loss is lowest at **epoch 5** (0.694) and rises afterwards
(0.849 at epoch 30), while training loss keeps falling (0.57 → 0.33). The model overfits quickly, so the best-epoch
checkpoint matters: the last-epoch weights would be clearly worse. The test days were not used for this choice.

## 5. Results on the test days

Two baselines need no model: **persistence**, $T(t+h) = T(t)$, and **linear trend**,
$T(t+h) = T(t) + h\,(T(t) - T(t-1))$.

| Horizon | LSTM MAE | LSTM RMSE | LSTM R² | Persistence MAE | Trend MAE | MAE cut vs persistence | Skill (MSE removed) |
|---|---|---|---|---|---|---|---|
| +5 min | **1.24 °C** | 1.70 | 0.713 | 1.91 | 2.85 | 35.0 % | 0.61 |
| +10 min | **1.79 °C** | 2.32 | 0.465 | 2.19 | 5.97 | 18.2 % | 0.42 |
| +15 min | **1.91 °C** | 2.42 | 0.421 | 2.12 | 7.85 | 9.8 % | 0.31 |

- R² is on the absolute temperature. Persistence itself scores R² of 0.26, 0.08 and 0.16, so the temperature now says
  little about the temperature even 5 minutes later.
- The forecasts are unbiased: mean error between −0.02 and +0.06 °C.
- The standard deviation of the actual change is 2.7–3.0 °C at every horizon. Temperatures here move by several
  degrees within minutes.
- Linear extrapolation is far worse than persistence. The changes are jumps towards a new equilibrium, not trends.
- Plots: `actual_vs_predicted.png`, `mae_by_horizon.png`, `example_forecast.png`.

### By rack

MAE at +15 minutes: 1.84, 1.97, 1.97 and 1.87 °C for racks 1–4. The model is equally (in)accurate on every rack.

### By behaviour policy (MAE in °C; LSTM / persistence)

| Day's controller | +5 | +10 | +15 |
|---|---|---|---|
| round-robin (test) | 0.71 / 0.72 | 1.24 / 1.16 | 1.50 / 1.45 |
| rule-based (test) | **1.36 / 3.06** | 2.00 / 2.97 | 2.05 / 2.39 |
| PPO (test) | 1.65 / 1.93 | 2.13 / 2.44 | 2.19 / 2.52 |

The gain over persistence is very uneven. On the rule-based day, cooling follows a predictable threshold rule and the
model more than halves the 5-minute error. On the round-robin day, cooling is fixed and temperature moves only when
jobs start or finish; the model has **no advantage at any horizon**.

### Generalisation days (new seeds; MAE at +5 / +10 / +15, LSTM vs persistence)

| Controller | LSTM | Persistence |
|---|---|---|
| random | 1.75 / 2.75 / 3.05 | 1.96 / 2.83 / 3.30 |
| round-robin | 0.67 / 1.14 / 1.39 | 0.71 / 1.16 / 1.45 |
| rule-based | 1.23 / 1.84 / 1.86 | 2.81 / 2.93 / 2.09 |
| PPO | 1.46 / 1.80 / 1.81 | 1.76 / 2.17 / 2.13 |
| **DQN (not in training data)** | 1.44 / 2.05 / 2.18 | 1.53 / 2.14 / 2.44 |

Errors on fresh days match the test days, so the test result is not a fluke of three days. On DQN-driven days the
model still beats persistence, but by less (6 % at 5 minutes, against 17 % for PPO). **The forecaster's edge depends
on the controller it is paired with**, and a controller trained *on* the forecasts (Step 13) will behave differently
again.

## 6. Why 15 minutes ahead is hard here

**The thermal time constant is 4 minutes.** A rack settles to within a few percent of a new equilibrium in about 12
minutes. That equilibrium is $T^* = T_\text{amb} + 100(\alpha L - \beta\eta C)/\kappa$. A change of 0.3 in thermal load
(one large job) moves it by about 4 °C, and one cooling level (20 %) by about 3.6 °C. The temperature 15 minutes from
now is therefore decided by the load and cooling at that time, and the last 10 minutes of history say little about
them:

| Test windows where received cooling… | Share | LSTM MAE +5 / +10 / +15 | Persistence |
|---|---|---|---|
| changed by more than 0.02 within 15 min | 57 % | 1.54 / 2.11 / 2.18 | 2.59 / 2.72 / 2.58 |
| stayed the same | 43 % | 0.84 / 1.37 / 1.55 | 0.99 / 1.47 / 1.51 |

Even when cooling does not change, the 15-minute error is 1.55 °C and no better than persistence. The remaining
uncertainty comes from jobs: arrivals are random, and placements are chosen by the controller.

## 7. Warning-zone detection

Among test windows that start below 27 °C, does the forecast flag the ones that end at or above 27 °C?

| Horizon | Actual crossings | LSTM precision | LSTM recall | LSTM flags | Trend precision / recall |
|---|---|---|---|---|---|
| +5 min | 899 | 0.65 | **0.31** | 426 | 0.22 / 0.48 |
| +10 min | 986 | 0.53 | 0.07 | 128 | 0.06 / 0.21 |
| +15 min | 898 | 0.44 | **0.008** | 16 | 0.12 / 0.54 |

The LSTM minimises average error, so under uncertainty it predicts changes close to zero and rarely forecasts a
crossing. At 5 minutes it catches 31 % of crossings with 65 % precision. At 15 minutes it catches 7 of 898.
Persistence can never flag a crossing. The linear trend flags many, but is wrong 78–94 % of the time. **As an early
warning for predictive cooling, only the 5-minute forecast is usable.**

## 8. Ablations

Both are trained on the same history and split, and reported for comparison. Neither replaces the configured model.

| Model | Parameters | Best epoch | MAE +5 / +10 / +15 | Skill +5 / +10 / +15 | +15 crossing recall |
|---|---|---|---|---|---|
| **LSTM, 6 features (configured)** | 51 907 | 5 | 1.238 / 1.791 / 1.913 | 0.61 / 0.42 / 0.31 | 0.008 |
| GRU, 6 features | 38 979 | 4 | 1.243 / 1.782 / 1.899 | 0.61 / 0.42 / 0.32 | 0.012 |
| LSTM, 9 features | 52 675 | 4 | 1.187 / 1.681 / 1.798 | 0.64 / 0.47 / 0.38 | 0.037 |

- **GRU vs LSTM:** indistinguishable. The architecture is not the limit.
- **Three extra inputs** that an operator already knows at time *t*:
  - `cooling_target`: the set-point the plant is ramping towards, × the rack's effectiveness;
  - `load_ending_soon`: the thermal load of jobs on the rack that will finish within 15 minutes;
  - `ambient_temperature`.

  They cut the 15-minute MAE by 6 % (1.91 → 1.80 °C) and help most on round-robin days (1.50 → 1.25 °C), where job
  completions are the main source of change. Crossing recall at 15 minutes rises only from 0.8 % to 3.7 %. Better
  inputs help a little. They do not remove the dependence on future arrivals and future control.

## 9. Live provider

`ForecastProvider(forecaster)` implements the environment's `PredictionProvider` interface. It keeps the last 10
minutes of `rack_features` itself and returns a (4 racks × 3 horizons) array in °C.

```python
env = DataCenterEnv(prediction_provider=ForecastProvider(TemperatureForecaster.load("models/temperature_predictor")))
obs, _ = env.reset(seed=42)        # obs.shape == (42,): the 30 base features + 12 forecasts
```

- A new episode clears the history. Until 10 frames exist, the earliest frame is repeated, so the first forecasts of
  an episode are close to "no change".
- Calling it twice in the same minute does not add a frame.
- A test checks that it matches the offline forecast on the same frames, and that the 42-feature observation stays
  inside the declared observation space. The 30-feature observation is unchanged.

## 10. Reproducibility

Dataset generation and training are deterministic for a given seed. Regenerating the 20-day history reproduced the six
configured feature columns bit-for-bit. A second full training run reproduced the saved LSTM weights, best epoch and
validation loss exactly.

## 11. Limitations

1. **Low skill beyond 5 minutes** (§5–6). This is a property of the simulated plant: the 4-minute time constant is
   short compared with the forecast horizons.
2. **The forecast is a conditional average.** It under-predicts large moves, so it rarely predicts threshold
   crossings (§7). A quantile or classification head would suit early warning better; that was not built here.
3. **Policy dependence.** Accuracy depends on the controller generating the data (§5). Predictive PPO will change the
   control pattern the forecaster was trained on.
4. **No knowledge of future actions.** The forecaster does not see the action about to be taken. An
   action-conditioned model ("what if cooling rises one level?") would be more useful for control, and is a different
   design.
5. **One training seed.** The split is three validation and three test days, so each lacks one of the four policies.
6. **Temperatures are clipped at 18 °C** by the simulator (visible as a vertical line in the scatter plots). Many
   windows end at that floor.

## 12. What this means for Step 13

The design assumes forecasts let the controller act before a rack becomes critical. The measured picture:
- The **+5-minute** forecast is informative: 35 % better than persistence, and it catches 31 % of warning crossings.
  The cooling plant needs 3 minutes to respond, so a 5-minute warning is still actionable.
- The **+10 and +15-minute** forecasts add little beyond the current state, which PPO already observes.

Predictive PPO can still be trained as designed, with all 12 forecast features. An honest expectation is a small
effect, if any. Options to decide before Step 13:
1. Proceed as designed and report whatever the comparison with PPO shows.
2. Use the 9-feature forecaster (slightly better, same interface).
3. Shorten the horizons (for example 2, 5 and 8 minutes) to where the forecaster has skill.

Options 2 and 3 change the design and are not applied here.
