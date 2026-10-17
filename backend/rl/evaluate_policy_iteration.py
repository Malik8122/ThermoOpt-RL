"""Evaluate the trained Policy Iteration policy (Step 7) and compare it with the Step 6 baselines.

Runs the saved policy in the tabular environment with the shared runner (backend/evaluation/runner.py)
on the Step 6 protocol: seeds 42-46 x 24 h from 08:00. The baseline numbers are *read* from
experiments/baselines_5x1440min.json (measured in Step 6), not re-run.

Outputs:
    experiments/policy_iteration_5x1440min.json      metrics, comparison, policy analysis
    experiments/plots/policy_iteration/*.png         convergence and policy plots

Usage:
    python -m backend.rl.evaluate_policy_iteration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from backend.config import get_config  # noqa: E402
from backend.evaluation.runner import METRICS, run_episode, summarise  # noqa: E402
from backend.rl.actions import N_TABULAR_ACTIONS  # noqa: E402
from backend.rl.environment import DataCenterEnv  # noqa: E402
from backend.rl.policy_iteration import PolicyIterationController, describe_action, load  # noqa: E402
from backend.rl.state_encoder import TabularStateEncoder  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402
from backend.utils.seeding import set_global_seed  # noqa: E402

logger = get_logger("evaluate_policy_iteration")

COMPARISON = (("total_reward", "Reward"), ("energy_kwh", "Energy kWh"), ("carbon_kg", "Carbon kg"),
              ("cost_inr", "Cost INR"), ("critical_rack_minutes", "Critical min"), ("sla_failures", "SLA failures"))
N_EXAMPLES = 10

# Reference palette (dataviz skill): one hue per chart, text in ink tokens, recessive grid.
SERIES = "#2a78d6"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def value_stats(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "n": int(values.size)}


def analyse_policy(loaded, controller: PolicyIterationController, encoder: TabularStateEncoder) -> dict:
    policy, V = loaded.policy, loaded.value_function
    modelled = loaded.model.modelled
    visits = controller.state_visits
    visited = np.flatnonzero(visits > 0)
    training = loaded.payload["training"]

    def state_row(s: int) -> dict:
        levels = encoder.describe(encoder.decode(s))
        return {"state": int(s), **levels, "action": int(policy[s]), "action_name": describe_action(policy[s]),
                "value": float(V[s]), "evaluation_visits": int(visits[s]), "modelled": bool(modelled[s])}

    # Examples are chosen by a fixed rule, not by hand: the most-visited states during evaluation.
    by_visits = sorted(visited, key=lambda s: (-visits[s], s))
    return {
        "outer_iterations": training["iterations"],
        "converged": training["converged"],
        "evaluation_sweeps_total": sum(h["evaluation_sweeps"] for h in training["history"]),
        "policy_changes_per_iteration": [h["policy_changes"] for h in training["history"]],
        "states": {"total": int(len(policy)), "modelled": int(modelled.sum()),
                   "visited_in_evaluation": int(len(visited)),
                   "visited_but_unmodelled": [int(s) for s in visited if not modelled[s]]},
        "action_frequency_in_evaluation": {describe_action(a): int(controller.action_counts[a])
                                           for a in range(N_TABULAR_ACTIONS)},
        "action_frequency_in_policy_table": {describe_action(a): int(np.sum(policy[modelled] == a))
                                             for a in range(N_TABULAR_ACTIONS)},
        "value_function_modelled_states": value_stats(V[modelled]),
        "most_visited_states": [state_row(s) for s in by_visits[:N_EXAMPLES]],
        "all_visited_states": [state_row(s) for s in by_visits],
    }


def comparison_table(baselines: dict, pi_summary: dict) -> list[dict]:
    rows = [{"controller": name, "source": "Step 6 (measured earlier)",
             **{m: data["summary"][m] for m, _ in COMPARISON}}
            for name, data in baselines["controllers"].items()]
    rows.append({"controller": "policy_iteration", "source": "Step 7 (this run)",
                 **{m: pi_summary[m] for m, _ in COMPARISON}})
    return rows


# ------------------------------------------------------------------ plots
def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def make_plots(loaded, analysis: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = loaded.payload["training"]["history"]
    theta = loaded.payload["hyperparameters"]["theta"]
    paths = []

    # 1. Max value change per evaluation sweep, sweeps of all outer iterations laid end to end.
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    x0 = 0
    for h in history:
        xs = np.arange(x0 + 1, x0 + len(h["sweep_deltas"]) + 1)
        ax.plot(xs, h["sweep_deltas"], color=SERIES, linewidth=2)
        ax.axvline(x0 + 0.5, color=GRID, linewidth=1)
        ax.text(x0 + 1, 1.0, f"it {h['iteration']}", transform=ax.get_xaxis_transform(), va="top",
                fontsize=7, color=INK_2)
        x0 += len(h["sweep_deltas"])
    ax.axhline(theta, color=INK_2, linewidth=1, linestyle="--")
    ax.text(0, theta, f" θ = {theta:g}", va="bottom", ha="left", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    style_axes(ax, "Policy evaluation: max |V_new − V_old| per sweep", "evaluation sweep (cumulative)",
           "max value change")
    paths.append(out_dir / "value_delta_per_sweep.png")

    # 2. Policy actions changed per outer iteration.
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor=SURFACE)
    its = [h["iteration"] for h in history]
    changes = [h["policy_changes"] for h in history]
    ax2.bar(its, changes, color=SERIES, width=0.6)
    for i, c in zip(its, changes):
        ax2.text(i, c, str(c), ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_xticks(its)
    style_axes(ax2, "Policy improvement: states whose action changed", "policy iteration", "states changed")
    paths.append(out_dir / "policy_changes_per_iteration.png")

    # 3. Value-function distribution over modelled states.
    fig3, ax3 = plt.subplots(figsize=(6, 3.5), facecolor=SURFACE)
    ax3.hist(loaded.value_function[loaded.model.modelled], bins=20, color=SERIES, edgecolor=SURFACE, linewidth=2)
    style_axes(ax3, "V(s) over modelled states", "V(s)", "states")
    paths.append(out_dir / "value_distribution.png")

    # 4. Action frequency during evaluation.
    fig4, ax4 = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    names = list(analysis["action_frequency_in_evaluation"])
    counts = np.array(list(analysis["action_frequency_in_evaluation"].values()))
    share = counts / max(1, counts.sum()) * 100
    ax4.barh(names, share, color=SERIES, height=0.6)
    for y, v in enumerate(share):
        ax4.text(v, y, f" {v:.1f}%", va="center", fontsize=8, color=INK)
    ax4.invert_yaxis()
    style_axes(ax4, "Actions chosen during evaluation (5 × 24 h)", "% of steps", "")
    paths.append(out_dir / "action_frequency.png")

    for f, p in zip((fig, fig2, fig3, fig4), paths):
        f.tight_layout()
        f.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(f)
    return paths


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.paths.models / "policy_iteration")
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--baselines", type=Path, default=cfg.paths.experiments / "baselines_5x1440min.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or cfg.paths.experiments / f"policy_iteration_{len(args.seeds)}x{args.minutes}min.json"

    set_global_seed(args.seeds[0])
    loaded = load(args.model_dir)
    controller = PolicyIterationController(loaded.policy, cfg)
    env = DataCenterEnv(representation="tabular", episode_minutes=args.minutes)
    episodes = [run_episode(env, controller, seed) for seed in args.seeds]
    summary = summarise(episodes)
    analysis = analyse_policy(loaded, controller, TabularStateEncoder(cfg.discretisation))

    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    same_protocol = (baselines["config"]["seeds"] == args.seeds
                     and baselines["config"]["episode_minutes"] == args.minutes)
    comparison = comparison_table(baselines, summary)
    plots = make_plots(loaded, analysis, cfg.paths.experiments / "plots" / "policy_iteration")

    results = {"config": {"episode_minutes": args.minutes, "seeds": args.seeds,
                          "start_hour": cfg.simulation.start_hour, "model_dir": str(args.model_dir),
                          "baselines_file": str(args.baselines), "baselines_same_protocol": same_protocol},
               "summary": summary, "episodes": episodes, "comparison": comparison,
               "policy_analysis": analysis, "plots": [str(p) for p in plots]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Policy Iteration: {len(args.seeds)} episodes x {args.minutes} min, seeds {args.seeds}")
    for m in METRICS:
        print(f"  {m:<32}{summary[m]['mean']:>12.2f} +/- {summary[m]['std']:.2f}")
    if not same_protocol:
        print("WARNING: baseline file used a different protocol; comparison is not like-for-like")
    print(f"\n{'controller':<18}{'source':<28}" + "".join(f"{label:>18}" for _, label in COMPARISON))
    for row in comparison:
        print(f"{row['controller']:<18}{row['source']:<28}"
              + "".join(f"{row[m]['mean']:>10.1f} ±{row[m]['std']:>6.1f}" for m, _ in COMPARISON))
    a = analysis
    print(f"\nstates: {a['states']['modelled']} modelled, {a['states']['visited_in_evaluation']} visited, "
          f"visited but unmodelled: {a['states']['visited_but_unmodelled']}")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
