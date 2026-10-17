"""Random-policy benchmark: the first reference point for every later RL agent.

Two random policies are measured on the same episodes (seeds 42, 43, ...):
  * uniform     - samples uniformly from all 18 actions, including invalid ones
  * valid-only  - samples uniformly from the actions valid in the current state (action mask)

Usage:
    python -m backend.rl.benchmark_random                          # 10 episodes x 60 min
    python -m backend.rl.benchmark_random --episodes 20 --minutes 240
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.config import get_config
from backend.evaluation.runner import summarise
from backend.rl.environment import DataCenterEnv
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng, set_global_seed

logger = get_logger("benchmark_random")

POLICIES = ("uniform", "valid-only")
METRICS = ("total_reward", "energy_kwh", "carbon_kg", "cost_inr", "peak_temperature_c",
           "warning_or_worse_rack_minutes", "critical_rack_minutes", "sla_failures",
           "completed_jobs", "invalid_action_rate")


def run_episode(env: DataCenterEnv, policy: str, seed: int) -> dict[str, float]:
    rng = make_rng(derive_seed(seed, "random-policy", policy))
    env.reset(seed=seed)
    truncated = False
    info: dict = {}
    while not truncated:
        if policy == "uniform":
            action = int(rng.integers(env.action_space.n))
        else:
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, _, _, truncated, info = env.step(action)
    ep, totals = info["episode"], env.dc.accountant.totals
    return {
        "seed": seed,
        "total_reward": ep["total_reward"],
        "energy_kwh": totals.total_kwh,
        "carbon_kg": totals.carbon_g / 1000.0,
        "cost_inr": totals.cost_inr,
        "peak_temperature_c": ep["peak_temperature_c"],
        "warning_or_worse_rack_minutes": ep["warning_or_worse_rack_minutes"],
        "critical_rack_minutes": ep["critical_rack_minutes"],
        "sla_failures": ep["jobs_failed"],
        "completed_jobs": ep["jobs_completed"],
        "invalid_action_rate": ep["invalid_actions"] / ep["steps"],
        "reward_by_component": ep["reward_by_component"],
    }


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=cfg.project.seed)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or cfg.paths.experiments / f"random_benchmark_{args.episodes}x{args.minutes}min.json"

    set_global_seed(args.seed)
    env = DataCenterEnv(episode_minutes=args.minutes)
    seeds = [args.seed + i for i in range(args.episodes)]
    results = {"config": {"episodes": args.episodes, "episode_minutes": args.minutes, "seeds": seeds,
                          "start_hour": cfg.simulation.start_hour}, "policies": {}}
    for policy in POLICIES:
        episodes = [run_episode(env, policy, s) for s in seeds]
        results["policies"][policy] = {"summary": summarise(episodes, METRICS), "episodes": episodes}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Random-policy benchmark: {args.episodes} episodes x {args.minutes} min, seeds {seeds[0]}-{seeds[-1]}")
    print(f"{'metric':<32}" + "".join(f"{p:>26}" for p in POLICIES))
    for m in METRICS:
        cells = "".join(f"{results['policies'][p]['summary'][m]['mean']:>14.3f} +/- {results['policies'][p]['summary'][m]['std']:<9.3f}"
                        for p in POLICIES)
        print(f"{m:<32}{cells}")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
