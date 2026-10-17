# Dashboard (Steps 15–17, built ahead of Step 13)

A live web dashboard for the digital twin and all saved results: a FastAPI backend and a React frontend.

| | |
|---|---|
| Backend | `backend/api/main.py` (routes, WebSocket), `service.py` (live twin + controllers), `results.py` (saved results) |
| Frontend | `frontend/` — React 19, Vite, TypeScript, Tailwind CSS 4 |
| Tests | `tests/test_api.py`: 24 tests |

## Run it

**One command** (builds the frontend the first time, then serves everything on one port):

```powershell
.\start_dashboard.ps1            # http://127.0.0.1:8000
```

**For development** (hot reload), use two terminals:

```powershell
.venv\Scripts\python -m backend.api.main        # API on http://127.0.0.1:8000  (interactive docs at /docs)
cd frontend; npm install; npm run dev           # UI on http://127.0.0.1:5173, proxies /api and /ws to the API
```

Nothing needs to be downloaded by hand. `npm install` fetches the frontend libraries, all free and open source.

## Principle: the frontend invents no numbers

Every value on screen comes from the backend. Live values are read from the simulator after each simulated minute.
Result pages read the files the evaluation scripts wrote to `experiments/`. The frontend only formats, sorts and
animates. The one quantity computed in the browser is the "live error" on the Predictive Cooling page, which is the
mean difference between two series the backend sent.

## Pages

| Page | What it shows |
|---|---|
| **Overview** | Hero with the live 3D twin, six live KPIs, the controller leaderboard, the 24-hour carbon profile with a "now" marker, and a card for each controller that loads it into the twin. |
| **Digital Twin** | Full 3D scene, transport controls, live HUD, the decision feed with explanations and reward breakdown, a card per rack (temperature gauge, CPU / GPU load, forecast chips), and streaming charts for temperature, power and cooling. |
| **Workloads** | Queue, jobs running on each rack with progress, finished jobs, and a form to inject your own job. |
| **RL Training** | Saved training curves for Q-Learning, DQN and PPO: reward, SLA failures, critical minutes, energy, and each algorithm's own diagnostics. Runs can be toggled. |
| **Algorithms** | Ranking by any of ten metrics with mean, ± 1 standard deviation and one dot per evaluation day; optional ablations; an energy-vs-reliability chart; a scorecard. |
| **Predictive Cooling** | Live forecasts per rack, "forecast in hindsight" against what actually happened, and the saved evaluation of the LSTM against simple baselines. |

## The 3D scene

Built with React Three Fiber. Everything animated is driven by backend state:

| Visual | Driven by |
|---|---|
| Rack glow colour, floor halo, light | rack temperature (one thermal colour ramp, also used by gauges and chips) |
| Pulsing glow | rack in the critical zone |
| Lit server slots (10) and GPU slots (2) | CPU and GPU utilisation |
| Fan speed, ring brightness, air-particle speed | delivered cooling level |
| Rising heat particles | rack temperature |
| Floor rings | the last decision: teal = placed, amber → teal = migrated, red = rejected |

Labels are ordinary DOM elements positioned each frame by projecting 3D anchor points to the screen.

## Live controllers

Any of the eight controllers can drive the twin and be swapped mid-run: rule-based, round-robin, random, Policy
Iteration, Value Iteration, Q-Learning, DQN and PPO. The tabular controllers receive the 144-state id computed from
the same simulator, and their macro-action is resolved to a primitive action exactly as the tabular environment does.
PPO is run sampled, the mode in which it performed best (PPO.md §11). A controller whose model folder is missing is
shown as unavailable.

**Decision explanations** are assembled from simulator facts and the action actually taken, for example
*"Placed J-00231 (50 CPU / 93 GPU) on Rack 3, 3rd coolest rack at 18.0 °C. Lowered cooling 100 % → 80 % with Rack 1
at 18.0 °C (normal); its +5 min forecast was 19.2 °C."* They state what happened and the conditions at the time.
They do not claim to know why a learned controller chose it.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` · `/api/racks` · `/api/workloads` · `/api/history` | live state, per-rack state, jobs and decisions, last 6 hours |
| GET | `/api/controllers` | the eight controllers, whether each is available and which is active |
| POST | `/api/simulation/start` · `/pause` · `/speed` · `/step` · `/reset` | run, pause, set simulated minutes per second, step 1–240 minutes, restart with a seed |
| POST | `/api/simulation/controller` | switch the controller without resetting |
| POST | `/api/workload` | inject a job |
| GET | `/api/predictions` | current LSTM forecasts |
| GET | `/api/meta` | twin configuration and the 24-hour grid profile |
| GET | `/api/comparison` | saved evaluation of every controller and ablation |
| GET | `/api/training/{algorithm}/metrics` | saved training curves (`q_learning`, `dqn`, `ppo`), down-sampled to ≤ 400 points |
| GET | `/api/predictions/summary` | saved forecaster evaluation |
| WS | `/ws/simulation` | `init` on connect, then a `tick` (state, chart point, decision) after every simulated minute |

When a day ends, the twin rolls over to the next seed and keeps running.

**Not implemented:** `POST /api/train/{algorithm}` from the original design (starting training from the browser).
Training runs take minutes to an hour, and are started from the command line.

## Libraries (all free and open source)

React, Vite, TypeScript, Tailwind CSS · Motion (animation and page transitions) · three.js with React Three Fiber,
drei and postprocessing (3D scene and bloom) · Recharts (charts) · Lucide (icons) · Zustand (state) · Fontsource
(Sora, Manrope and JetBrains Mono, bundled locally) · FastAPI and Uvicorn (backend).

## Limitations

- Desktop-first. It is usable on a laptop screen and reflows on narrower widths, but was not designed for phones.
- The 3D scene needs WebGL. On a machine without a GPU it will be slow.
- One shared twin: every browser tab connected to the same backend sees and controls the same simulation.
- The JavaScript bundle is about 1.9 MB (545 kB gzipped), mostly three.js; it is not code-split.
