"""Evaluate the three baseline controllers (Step 6).

Every controller runs on the same seeds and episode length (config.yaml → evaluation:
seeds 42-46 x 24 h by default), so each sees identical job arrivals and grid signals.

Usage:
    python -m backend.evaluation.run_baselines                     # 5 seeds x 1440 min
    python -m backend.evaluation.run_baselines --minutes 240 --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import get_config
from backend.evaluation.runner import METRICS, run_episode, summarise
from backend.rl.baselines import BASELINES, make_baseline
from backend.rl.environment import DataCenterEnv
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import set_global_seed

logger = get_logger("run_baselines")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--controllers", nargs="+", choices=sorted(BASELINES), default=list(BASELINES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or cfg.paths.experiments / f"baselines_{len(args.seeds)}x{args.minutes}min.json"

    set_global_seed(args.seeds[0])
    env = DataCenterEnv(episode_minutes=args.minutes)
    results = {"config": {"episode_minutes": args.minutes, "seeds": args.seeds,
                          "start_hour": cfg.simulation.start_hour,
                          "baselines": cfg.baselines.model_dump()},
               "controllers": {}}
    for name in args.controllers:
        controller = make_baseline(name)
        episodes = []
        for seed in args.seeds:
            episodes.append(run_episode(env, controller, seed))
            logger.info("%s seed %d: reward %.2f", name, seed, episodes[-1]["total_reward"])
        results["controllers"][name] = {"summary": summarise(episodes), "episodes": episodes}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Baselines: {len(args.seeds)} episodes x {args.minutes} min, seeds {args.seeds}")
    print(f"{'metric':<32}" + "".join(f"{n:>24}" for n in args.controllers))
    for m in METRICS:
        s = {n: results["controllers"][n]["summary"][m] for n in args.controllers}
        print(f"{m:<32}" + "".join(f"{s[n]['mean']:>13.2f} +/- {s[n]['std']:<7.2f}" for n in args.controllers))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
