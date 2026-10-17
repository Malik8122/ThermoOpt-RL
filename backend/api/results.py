"""Read-only access to saved experiment results for the dashboard.

Every number served here comes from a file written by an evaluation or training script in ``experiments/``.
Nothing is recomputed or invented; a missing file simply leaves that part out.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from backend.config import Config
from backend.simulation.grid import GridSignals

METRIC_INFO = {
    "total_reward": ("Reward", "", "higher"),
    "energy_kwh": ("Energy", "kWh", "lower"),
    "it_energy_kwh": ("IT energy", "kWh", "lower"),
    "cooling_energy_kwh": ("Cooling energy", "kWh", "lower"),
    "cost_inr": ("Cost", "₹", "lower"),
    "carbon_kg": ("Carbon", "kg CO₂", "lower"),
    "mean_temperature_c": ("Mean temperature", "°C", "lower"),
    "peak_temperature_c": ("Peak temperature", "°C", "lower"),
    "critical_rack_minutes": ("Critical rack-minutes", "min", "lower"),
    "warning_or_worse_rack_minutes": ("Warning rack-minutes", "min", "lower"),
    "sla_failures": ("SLA failures", "jobs", "lower"),
    "completed_jobs": ("Completed jobs", "jobs", "higher"),
    "delayed_jobs": ("Delayed jobs", "jobs", "lower"),
    "mean_completion_minutes": ("Mean completion time", "min", "lower"),
    "cooling_changes": ("Cooling changes", "per day", "lower"),
    "migrations": ("Migrations", "per day", "lower"),
    "invalid_action_rate": ("Invalid-action rate", "share", "lower"),
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _clean(value):
    """JSON-safe: NaN / inf become None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _entry(key: str, label: str, family: str, step: int, summary: dict, episodes: list[dict] | None,
           note: str = "", primary: bool = True) -> dict:
    return {"key": key, "label": label, "family": family, "step": step, "primary": primary, "note": note,
            "metrics": {m: summary[m] for m in METRIC_INFO if m in summary},
            "per_seed": [{"seed": e["seed"], **{m: e[m] for m in METRIC_INFO if m in e}} for e in episodes or []]}


def comparison(config: Config) -> dict:
    exp = config.paths.experiments
    entries: list[dict] = []
    protocol = None
    base = _load(exp / "baselines_5x1440min.json")
    if base:
        protocol = {"seeds": base["config"]["seeds"], "episode_minutes": base["config"]["episode_minutes"]}
        labels = {"random": "Random", "round_robin": "Round-robin", "rule_based": "Rule-based"}
        for key, data in base["controllers"].items():
            entries.append(_entry(key, labels.get(key, key), "baseline", 6, data["summary"], data["episodes"]))
    for key, label, step, fname in (("policy_iteration", "Policy Iteration", 7, "policy_iteration_5x1440min.json"),
                                    ("value_iteration", "Value Iteration", 8, "value_iteration_5x1440min.json")):
        d = _load(exp / fname)
        if d:
            entries.append(_entry(key, label, "planning", step, d["summary"], d["episodes"]))
    ql = _load(exp / "q_learning_5x1440min.json")
    if ql:
        entries.append(_entry("q_learning", "Q-Learning", "tabular-rl", 9, ql["summary"], ql["episodes"]))
        entries.append(_entry("q_learning_table", "Q-Learning, no live mask", "tabular-rl", 9,
                              ql["ablation_policy_table_without_live_mask"]["summary"],
                              ql["ablation_policy_table_without_live_mask"]["episodes"],
                              "Ablation: the learned table used without the live action mask.", primary=False))
        entries.append(_entry("pi_vi_masked", "PI / VI + live mask", "planning", 9,
                              ql["ablation_pi_vi_q_with_live_mask"]["summary"],
                              ql["ablation_pi_vi_q_with_live_mask"]["episodes"],
                              "Ablation: Policy Iteration's values acted on over the live mask.", primary=False))
    dqn = _load(exp / "dqn_5x1440min.json")
    if dqn:
        entries.append(_entry("dqn", "DQN", "deep-rl", 10, dqn["summary"], dqn["episodes"]))
        entries.append(_entry("dqn_nomask", "DQN, no mask", "deep-rl", 10, dqn["ablation_no_mask"]["summary"],
                              dqn["ablation_no_mask"]["episodes"], "Ablation: trained and run without action masking.",
                              primary=False))
    ppo = _load(exp / "ppo_5x1440min.json")
    if ppo:
        m = ppo["models"]
        entries.append(_entry("ppo", "PPO", "deep-rl", 11, m["ppo"]["deterministic"]["summary"],
                              m["ppo"]["deterministic"]["episodes"], "Design protocol, most-probable action."))
        variants = (("ppo_sampled", "PPO, sampled", "ppo", "stochastic", "Same policy, actions sampled."),
                    ("ppo_nomask", "PPO, no mask", "ppo_nomask", "deterministic", "Ablation: no action masking."),
                    ("ppo_432k", "PPO, 430k steps", "ppo_432k", "deterministic", "DQN-matched training budget."),
                    ("ppo_432k_sampled", "PPO, 430k steps, sampled", "ppo_432k", "stochastic",
                     "DQN-matched budget, actions sampled."))
        for key, label, tag, mode, note in variants:
            if tag in m:
                entries.append(_entry(key, label, "deep-rl", 11, m[tag][mode]["summary"], m[tag][mode]["episodes"],
                                      note, primary=False))
    return _clean({"protocol": protocol, "controllers": entries,
                   "metrics": [{"key": k, "label": v[0], "unit": v[1], "better": v[2]} for k, v in METRIC_INFO.items()]})


def _read_csv(path: Path, columns: list[str], max_points: int = 400) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    step = max(1, math.ceil(len(rows) / max_points))
    picked = rows[::step]
    out = {"rows": len(rows), "stride": step}
    for c in columns:
        if c in rows[0]:
            out[c] = [_clean(float(r[c])) if r[c] not in ("", "nan") else None for r in picked]
    return out


EPISODE_COLUMNS = ["episode", "env_steps", "epsilon", "total_reward", "energy_kwh", "carbon_kg", "sla_failures",
                   "critical_rack_minutes", "invalid_action_rate", "invalid_actions", "mean_loss", "mean_q",
                   "max_abs_q_update", "mean_abs_q_update", "mean_entropy", "mean_value", "states_visited_so_far"]
UPDATE_COLUMNS = ["update", "env_steps", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                  "explained_variance"]

TRAINING_RUNS = {
    "q_learning": {"label": "Q-Learning", "runs": [("main", "1000 × 24 h, live mask", "q_learning_training.csv", None)]},
    "dqn": {"label": "DQN", "runs": [("main", "300 × 24 h, live mask", "dqn_training.csv", None),
                                     ("nomask", "300 × 24 h, no mask", "dqn_nomask_training.csv", None)]},
    "ppo": {"label": "PPO", "runs": [("main", "199k steps, live mask", "ppo_training.csv", "ppo_updates.csv"),
                                     ("nomask", "199k steps, no mask", "ppo_nomask_training.csv", "ppo_nomask_updates.csv"),
                                     ("long", "430k steps, live mask", "ppo_432k_training.csv", "ppo_432k_updates.csv")]},
}


def training(config: Config, algorithm: str) -> dict | None:
    if algorithm not in TRAINING_RUNS:
        return None
    exp = config.paths.experiments
    spec = TRAINING_RUNS[algorithm]
    runs = []
    for key, label, episodes_csv, updates_csv in spec["runs"]:
        episodes = _read_csv(exp / episodes_csv, EPISODE_COLUMNS)
        if episodes is None:
            continue
        runs.append({"key": key, "label": label, "episodes": episodes,
                     "updates": _read_csv(exp / updates_csv, UPDATE_COLUMNS) if updates_csv else None})
    return {"algorithm": algorithm, "label": spec["label"], "runs": runs}


def prediction(config: Config) -> dict | None:
    d = _load(config.paths.experiments / "temperature_prediction.json")
    if d is None:
        return None
    curve = _read_csv(config.paths.experiments / "temperature_predictor_training.csv",
                      ["epoch", "train_loss", "val_loss", "val_mae_c"])
    return _clean({"model": d["model"], "split": d["split"],
                   "training": {"best_epoch": d["training"]["best_epoch"], "seconds": d["training"]["seconds"],
                                "train_windows": d["training"]["train_windows"], "curve": curve},
                   "test": d["test"], "test_by_policy": d["test_by_policy"], "test_by_rack": d["test_by_rack"],
                   "generalisation_days": d["generalisation_days"],
                   "warning_crossing_detection": d["warning_crossing_detection"],
                   "test_by_cooling_change": d["test_by_cooling_change"],
                   "other_architectures": {k: {"test": v["test"], "parameters": v["parameters"],
                                               "feature_names": v.get("feature_names")}
                                           for k, v in d["other_architectures"].items()}})


def meta(config: Config) -> dict:
    grid = GridSignals(config.grid)
    hours = np.linspace(0, 24, 97)
    dc, th, rw = config.datacenter, config.thermal, config.reward
    return {
        "project": config.project.name,
        "racks": dc.num_racks, "servers": dc.total_servers, "gpu_nodes": dc.total_gpu_nodes,
        "rack_names": dc.rack_names, "rack_cooling_effectiveness": th.rack_cooling_effectiveness,
        "zones": {"warning": th.zones.warning_c, "critical": th.zones.critical_c,
                  "min": th.min_temperature_c, "max": th.max_temperature_c},
        "cooling_levels_pct": config.cooling.levels_pct, "max_cooling_kw": config.cooling.max_cooling_power_kw,
        "max_it_kw": config.max_it_power_kw,
        "reward_weights": rw.weights.model_dump(),
        "high_carbon_threshold": config.grid.high_carbon_threshold,
        "grid_profile": {"hour": hours.round(3).tolist(),
                         "carbon": [float(grid.carbon_at(h % 24)) for h in hours],
                         "price": [float(grid.price_at(h % 24)) for h in hours]},
        "workload_types": {k: {"probability": v.probability, "cpu": [v.cpu.min, v.cpu.max],
                               "gpu": [v.gpu.min, v.gpu.max], "duration": [v.duration_minutes.min, v.duration_minutes.max]}
                           for k, v in config.workloads.types.items()},
        "horizons": config.prediction.horizons_minutes,
    }
