"""Run a controller for one episode and collect the PHASE1_DESIGN §6 metrics.

Used by the baseline evaluation (Step 6) and, from Step 14, by the full comparison.
Metrics are read from the simulator and the environment's episode statistics, never
recomputed from the reward.
"""

from __future__ import annotations

import numpy as np

from backend.rl.baselines import Controller
from backend.rl.environment import DataCenterEnv
from backend.simulation.enums import WorkloadStatus

METRICS = (
    "total_reward", "energy_kwh", "it_energy_kwh", "cooling_energy_kwh", "cost_inr", "carbon_kg",
    "mean_temperature_c", "peak_temperature_c", "critical_rack_minutes", "warning_or_worse_rack_minutes",
    "sla_failures", "completed_jobs", "delayed_jobs", "mean_completion_minutes",
    "cooling_changes", "migrations", "invalid_action_rate",
)


def run_episode(env: DataCenterEnv, controller: Controller, seed: int) -> dict:
    """One full episode on ``seed``. The controller gets the observation and the action mask each step."""
    obs, _ = env.reset(seed=seed)
    controller.reset(seed)
    truncated, info = False, {}
    temperature_sum = 0.0
    while not truncated:
        action = controller.act(obs, env.action_masks())
        obs, _, _, truncated, info = env.step(action)
        temperature_sum += info["avg_temperature"]
    return episode_metrics(env, info, temperature_sum, seed)


def episode_metrics(env: DataCenterEnv, info: dict, temperature_sum: float, seed: int) -> dict:
    """The §6 metrics of a finished episode: ``info`` of its last step, sum of per-step mean temperatures."""
    ep, totals = info["episode"], env.dc.accountant.totals
    submitted = [w for w in env.dc.workloads.values() if not w.is_background]
    completed = [w for w in submitted if w.status is WorkloadStatus.COMPLETED]
    return {
        "seed": seed,
        "total_reward": ep["total_reward"],
        "energy_kwh": totals.total_kwh,
        "it_energy_kwh": totals.it_kwh,
        "cooling_energy_kwh": totals.cooling_kwh,
        "cost_inr": totals.cost_inr,
        "carbon_kg": totals.carbon_g / 1000.0,
        "mean_temperature_c": temperature_sum / ep["steps"],
        "peak_temperature_c": ep["peak_temperature_c"],
        "critical_rack_minutes": ep["critical_rack_minutes"],
        "warning_or_worse_rack_minutes": ep["warning_or_worse_rack_minutes"],
        "sla_failures": ep["jobs_failed"],
        "completed_jobs": ep["jobs_completed"],
        "delayed_jobs": sum(w.delay_count > 0 for w in submitted),
        # arrival -> completion, for jobs submitted during the episode (background jobs excluded)
        "mean_completion_minutes": float(np.mean([w.completed_at - w.arrival_time for w in completed]))
                                   if completed else 0.0,
        "cooling_changes": ep["cooling_changes"],
        "migrations": ep["migrations"],
        "invalid_action_rate": ep["invalid_actions"] / ep["steps"],
        "reward_by_component": ep["reward_by_component"],
    }


def summarise(episodes: list[dict], metrics: tuple[str, ...] = METRICS) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each metric across episodes."""
    return {m: {"mean": float(np.mean([e[m] for e in episodes])),
                "std": float(np.std([e[m] for e in episodes], ddof=1)) if len(episodes) > 1 else 0.0}
            for m in metrics}
