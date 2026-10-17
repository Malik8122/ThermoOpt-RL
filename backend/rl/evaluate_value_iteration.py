"""Evaluate Value Iteration (Step 8) and compare it with Policy Iteration and the Step 6 baselines.

Two kinds of comparison:
  * algorithmic, with no simulator involved: VI and PI solve the same saved empirical model (checked via
    model_hash), so their values, policies and Q-values are compared state by state;
  * behavioural: the VI policy runs through the shared runner on the Step 6/7 protocol (seeds 42-46 x 24 h).
    Baseline and Policy Iteration numbers are read from their saved results and not re-run.

Outputs:
    experiments/value_iteration_5x1440min.json
    experiments/plots/value_iteration/*.png

Usage:
    python -m backend.rl.evaluate_value_iteration
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
from backend.rl.evaluate_policy_iteration import (  # noqa: E402
    COMPARISON,
    GRID,
    INK,
    INK_2,
    SERIES,
    SURFACE,
    style_axes,
    value_stats,
)
from backend.rl.policy_iteration import describe_action  # noqa: E402
from backend.rl.policy_iteration import load as load_pi  # noqa: E402
from backend.rl.state_encoder import TabularStateEncoder  # noqa: E402
from backend.rl.tabular_model import q_values  # noqa: E402
from backend.rl.value_iteration import ValueIterationController  # noqa: E402
from backend.rl.value_iteration import load as load_vi  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402
from backend.utils.seeding import set_global_seed  # noqa: E402

logger = get_logger("evaluate_value_iteration")

SERIES_2 = "#eb6834"      # categorical slot 2 of the reference palette (VI when shown next to PI)
N_CLOSEST = 10


# --------------------------------------------------------------------------- #
# Algorithmic comparison (no simulator)
# --------------------------------------------------------------------------- #
def compare_algorithms(pi, vi, encoder: TabularStateEncoder) -> dict:
    """PI vs VI on the shared model. Both value functions are judged by the same Bellman operator."""
    model = vi.source.model
    mdp = model.to_mdp()
    gamma = vi.payload["gamma"]
    tie = vi.payload["tie_tolerance"]
    m = model.modelled
    states = np.flatnonzero(m)
    Q_pi, Q_vi = q_values(mdp, pi.value_function, gamma), q_values(mdp, vi.value_function, gamma)
    diff = np.abs(pi.value_function - vi.value_function)[m]
    agree = pi.policy[m] == vi.policy[m]

    def residual(V):
        return float(np.max(np.abs(q_values(mdp, V, gamma).max(axis=1) - V)))

    def competitors(Q, s, chosen, k=2):
        order = [a for a in np.argsort(-Q[s], kind="stable") if a != chosen and np.isfinite(Q[s, a])]
        return [{"action": int(a), "name": describe_action(a), "q": float(Q[s, a]),
                 "gap_to_selected": float(Q[s, chosen] - Q[s, a])} for a in order[:k]]

    def state_row(s):
        return {"state": int(s), **encoder.describe(encoder.decode(s)),
                "pi_action": describe_action(pi.policy[s]), "vi_action": describe_action(vi.policy[s]),
                "pi_q_selected": float(Q_pi[s, pi.policy[s]]), "vi_q_selected": float(Q_vi[s, vi.policy[s]]),
                "vi_best_competitors": competitors(Q_vi, s, vi.policy[s]),
                "n_admissible": int(mdp.admissible[s].sum())}

    # Gap between the best and second-best admissible action (VI's Q). Small gaps are where tie-breaking,
    # tolerance or sampling noise could flip the decision.
    gaps = {}
    for s in states:
        q = np.sort(Q_vi[s][np.isfinite(Q_vi[s])])[::-1]
        gaps[int(s)] = float(q[0] - q[1]) if len(q) > 1 else float("inf")
    closest = sorted(gaps, key=lambda s: (gaps[s], s))[:N_CLOSEST]
    n_tied = int(sum(g <= tie for g in gaps.values()))

    pi_hist, vi_hist = pi.payload["training"], vi.payload["training"]
    return {
        "model_hash": {"policy_iteration": pi.payload["model"]["model_hash"],
                       "value_iteration": vi.payload["model"]["model_hash"],
                       "identical": pi.payload["model"]["model_hash"] == vi.payload["model"]["model_hash"]},
        "modelled_states": int(m.sum()),
        "value_function": {"policy_iteration": value_stats(pi.value_function[m]),
                           "value_iteration": value_stats(vi.value_function[m]),
                           "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
                           "theoretical_bound": 2 * gamma * vi.payload["theta"] / (1 - gamma)},
        "policy": {"states_compared": int(m.sum()), "identical_actions": int(agree.sum()),
                   "different_actions": int((~agree).sum()), "agreement_pct": float(100 * agree.mean()),
                   "states_with_q_ties_within_tolerance": n_tied,
                   "action_frequency_policy_table": [
                       {"action": describe_action(a), "pi": int(np.sum(pi.policy[m] == a)),
                        "vi": int(np.sum(vi.policy[m] == a)),
                        "difference": int(np.sum(vi.policy[m] == a) - np.sum(pi.policy[m] == a))}
                       for a in range(N_TABULAR_ACTIONS)]},
        "disagreements": [state_row(s) for s in states[~agree]],
        "closest_decisions": [{**state_row(s), "q_gap_best_vs_second": gaps[s]} for s in closest],
        "convergence": {
            "vi_iterations": vi_hist["iterations"], "vi_converged": vi_hist["converged"],
            "pi_outer_iterations": pi_hist["iterations"], "pi_converged": pi_hist["converged"],
            "pi_evaluation_sweeps": sum(h["evaluation_sweeps"] for h in pi_hist["history"]),
            "bellman_optimality_residual": {"policy_iteration": residual(pi.value_function),
                                            "value_iteration": residual(vi.value_function)},
            "note": "A VI iteration is one Bellman-optimality backup; a PI iteration is a full policy evaluation "
                    "(many sweeps) plus an improvement step. Iteration counts are not comparable as a speed measure.",
        },
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(pi, vi, algo: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = vi.payload["training"]["history"]
    it = np.array([h["iteration"] for h in hist])
    theta = vi.payload["theta"]
    m = vi.source.model.modelled
    figs, paths = [], []

    # 1. Bellman delta per iteration (max and mean over modelled states), log scale.
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.plot(it, [h["max_delta"] for h in hist], color=SERIES, linewidth=2, label="max over states")
    ax.plot(it, [h["mean_delta"] for h in hist], color=SERIES_2, linewidth=2, label="mean over modelled states")
    ax.axhline(theta, color=INK_2, linewidth=1, linestyle="--")
    ax.text(it[0], theta, f" θ = {theta:g}", va="bottom", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    style_axes(ax, "Value Iteration: |V(k+1) − V(k)| per iteration", "iteration k", "value change (reward units)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    figs.append(fig), paths.append(out_dir / "bellman_delta.png")

    # 2. Value-function summary per iteration.
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.fill_between(it, [h["v_min"] for h in hist], [h["v_max"] for h in hist], color=SERIES, alpha=0.18,
                    linewidth=0, label="min – max")
    ax.plot(it, [h["v_mean"] for h in hist], color=SERIES, linewidth=2, label="mean")
    style_axes(ax, "V_k(s) over the 132 modelled states", "iteration k", "V (discounted reward)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    figs.append(fig), paths.append(out_dir / "value_summary.png")

    # 3. Final values: PI vs VI per modelled state.
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    v_pi, v_vi = pi.value_function[m], vi.value_function[m]
    lo, hi = min(v_pi.min(), v_vi.min()), max(v_pi.max(), v_vi.max())
    ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.5)
    ax.scatter(v_pi, v_vi, s=28, color=SERIES, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    vf = algo["value_function"]
    ax.text(0.03, 0.97, f"max |ΔV| = {vf['max_abs_difference']:.1e}\nmean |ΔV| = {vf['mean_abs_difference']:.1e}",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    style_axes(ax, "Final V(s): Policy vs Value Iteration", "Policy Iteration V(s)", "Value Iteration V(s)")
    figs.append(fig), paths.append(out_dir / "pi_vs_vi_values.png")

    # 4. Policy agreement: per-action counts in the two policy tables.
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    rows = algo["policy"]["action_frequency_policy_table"]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["pi"] for r in rows], height=0.36, color=SERIES, label="Policy Iteration")
    ax.barh(y + 0.2, [r["vi"] for r in rows], height=0.36, color=SERIES_2, label="Value Iteration")
    ax.set_yticks(y, [r["action"] for r in rows])
    ax.invert_yaxis()
    p = algo["policy"]
    style_axes(ax, f"Policy agreement: {p['identical_actions']}/{p['states_compared']} modelled states "
                   f"({p['agreement_pct']:.1f} %)", "modelled states choosing the action", "")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    figs.append(fig), paths.append(out_dir / "pi_vs_vi_policy.png")

    for f, path in zip(figs, paths):
        f.tight_layout()
        f.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(f)
    return paths


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vi-dir", type=Path, default=cfg.paths.models / "value_iteration")
    parser.add_argument("--pi-dir", type=Path, default=cfg.paths.models / "policy_iteration")
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--baselines", type=Path, default=cfg.paths.experiments / "baselines_5x1440min.json")
    parser.add_argument("--pi-results", type=Path, default=cfg.paths.experiments / "policy_iteration_5x1440min.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or cfg.paths.experiments / f"value_iteration_{len(args.seeds)}x{args.minutes}min.json"

    vi, pi = load_vi(args.vi_dir), load_pi(args.pi_dir)   # loading only; nothing is rebuilt or resampled
    algo = compare_algorithms(pi, vi, TabularStateEncoder(cfg.discretisation))
    if not algo["model_hash"]["identical"]:
        raise SystemExit("PI and VI were trained on different models; refusing to compare")

    set_global_seed(args.seeds[0])
    controller = ValueIterationController(vi.policy, cfg)
    env = DataCenterEnv(representation="tabular", episode_minutes=args.minutes)
    episodes = [run_episode(env, controller, seed) for seed in args.seeds]
    summary = summarise(episodes)

    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    pi_results = json.loads(args.pi_results.read_text(encoding="utf-8"))
    protocol = {"seeds": args.seeds, "episode_minutes": args.minutes}
    same_protocol = all(src["config"][k] == v for src in (baselines, pi_results) for k, v in protocol.items())
    comparison = [{"controller": n, "source": "Step 6 (inherited)", **{m: d["summary"][m] for m, _ in COMPARISON}}
                  for n, d in baselines["controllers"].items()]
    comparison.append({"controller": "policy_iteration", "source": "Step 7 (inherited)",
                       **{m: pi_results["summary"][m] for m, _ in COMPARISON}})
    comparison.append({"controller": "value_iteration", "source": "Step 8 (this run)",
                       **{m: summary[m] for m, _ in COMPARISON}})

    pi_freq = pi_results["policy_analysis"]["action_frequency_in_evaluation"]
    vi_freq = {describe_action(a): int(controller.action_counts[a]) for a in range(N_TABULAR_ACTIONS)}
    action_comparison = [{"action": a, "pi": pi_freq[a], "vi": vi_freq[a], "difference": vi_freq[a] - pi_freq[a]}
                         for a in vi_freq]
    same_as_pi = pi_results["episodes"] == episodes

    plots = make_plots(pi, vi, algo, cfg.paths.experiments / "plots" / "value_iteration")
    results = {
        "config": {**protocol, "start_hour": cfg.simulation.start_hour, "vi_dir": str(args.vi_dir),
                   "pi_dir": str(args.pi_dir), "same_protocol_as_inherited_results": same_protocol},
        "algorithmic_comparison": algo,
        "summary": summary, "episodes": episodes, "comparison": comparison,
        "action_frequency_in_evaluation": action_comparison,
        "episodes_identical_to_policy_iteration": same_as_pi,
        "states_visited_in_evaluation": int(np.count_nonzero(controller.state_visits)),
        "plots": [str(p) for p in plots],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    a, vf, p = algo, algo["value_function"], algo["policy"]
    print(f"Model hash (PI = VI): {a['model_hash']['value_iteration']}  identical={a['model_hash']['identical']}")
    print(f"VI: {a['convergence']['vi_iterations']} iterations | PI: {a['convergence']['pi_outer_iterations']} outer, "
          f"{a['convergence']['pi_evaluation_sweeps']} sweeps | Bellman residual PI "
          f"{a['convergence']['bellman_optimality_residual']['policy_iteration']:.2e}, VI "
          f"{a['convergence']['bellman_optimality_residual']['value_iteration']:.2e}")
    print(f"|V_PI - V_VI| on {a['modelled_states']} modelled states: max {vf['max_abs_difference']:.2e}, "
          f"mean {vf['mean_abs_difference']:.2e} (bound {vf['theoretical_bound']:.1e})")
    print(f"Policy agreement: {p['identical_actions']}/{p['states_compared']} ({p['agreement_pct']:.1f} %); "
          f"states with Q ties within tolerance: {p['states_with_q_ties_within_tolerance']}")
    print(f"\nValue Iteration: {len(args.seeds)} episodes x {args.minutes} min, seeds {args.seeds}"
          f"{'' if same_protocol else '  WARNING: inherited results used a different protocol'}")
    for mname in METRICS:
        print(f"  {mname:<32}{summary[mname]['mean']:>12.2f} +/- {summary[mname]['std']:.2f}")
    print(f"episodes identical to Policy Iteration's: {same_as_pi}")
    print(f"\n{'controller':<18}{'source':<22}" + "".join(f"{label:>18}" for _, label in COMPARISON))
    for row in comparison:
        print(f"{row['controller']:<18}{row['source']:<22}"
              + "".join(f"{row[mm]['mean']:>10.1f} ±{row[mm]['std']:>6.1f}" for mm, _ in COMPARISON))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
