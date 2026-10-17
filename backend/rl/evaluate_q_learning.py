"""Evaluate tabular Q-Learning (Step 9) and compare it with the baselines, Policy Iteration and Value Iteration.

Main evaluation: the greedy policy (epsilon = 0) chooses argmax_a Q(s, a) over the *live* action mask, through the
shared runner, on the Step 6-8 protocol (seeds 42-46 x 24 h). Earlier methods are read from their result files.

Secondary evaluation (ablation): the extracted policy *table* pi(s), which ignores the live mask exactly as the
PI/VI policies do. This separates what Q-Learning learned from the benefit of consulting the mask at run time.

Second ablation: PI/VI's own Q-values (from the saved Step 7 model and PI's value function) acted on greedily
over the same live mask. Together the four evaluations form a 2 x 2: {model-based, model-free} x {policy table,
live mask}. The Step 7 model is loaded here for this *evaluation* only; Q-Learning's training never reads it.

The model-gap analysis reads only the *list* of states in the Step 7 model (models/policy_iteration/policy.json).

    python -m backend.rl.evaluate_q_learning
"""

from __future__ import annotations

import argparse
import csv
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
from backend.rl.evaluate_policy_iteration import COMPARISON, INK, SERIES, SURFACE, style_axes  # noqa: E402
from backend.rl.policy_iteration import PolicyIterationController, describe_action  # noqa: E402
from backend.rl.policy_iteration import load as load_pi  # noqa: E402
from backend.rl.q_learning import QLearning, QLearningController  # noqa: E402
from backend.rl.baselines import Controller  # noqa: E402
from backend.rl.tabular_model import q_values  # noqa: E402
from backend.rl.state_encoder import TabularStateEncoder  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402
from backend.utils.seeding import set_global_seed  # noqa: E402

logger = get_logger("evaluate_q_learning")

SERIES_2 = "#eb6834"
MA_WINDOW = 50
# The PENDING-queue states in which PI/VI delay and which dominated their SLA failures (POLICY_ITERATION.md §11).
FOCUS_STATES = (5, 43, 67)


class MaskedGreedyController(Controller):
    """Greedy over a fixed Q matrix restricted to the live action mask (ties -> lowest index)."""

    name = "masked_greedy"

    def __init__(self, q: np.ndarray, tie_tolerance: float, config=None) -> None:
        super().__init__(config)
        self.q, self.tie = q, tie_tolerance

    def act(self, observation, action_mask=None) -> int:
        q = np.where(action_mask, self.q[int(observation)], -np.inf)
        return int(np.argmax(q >= q.max() - self.tie))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = np.convolve(x, np.ones(w) / w, mode="valid")
    return out


# --------------------------------------------------------------------------- #
def coverage_and_model_gap(agent: QLearning, pi_payload: dict, pi_results: dict) -> dict:
    ql_states = set(np.flatnonzero(agent.state_counts).tolist())
    model = pi_payload["model"]
    modelled = set(model["modelled_states"])
    collected = set(model["states_from_collection"])
    pi_onpolicy = {r["state"]: r["evaluation_visits"] for r in pi_results["policy_analysis"]["all_visited_states"]}
    share = agent.state_counts / agent.state_counts.sum()
    top = np.argsort(-agent.state_counts, kind="stable")[:15]
    return {
        "states_visited": len(ql_states), "states_visited_pct": 100 * len(ql_states) / agent.n_states,
        "state_action_pairs_visited": int((agent.visit_counts > 0).sum()),
        "state_action_pairs_observed_valid": int((agent.valid_counts > 0).sum()),
        "state_action_pairs_total": int(agent.q_table.size),
        "pi_vi_model": {"modelled": len(modelled), "from_random_collection": len(collected),
                        "from_closure": len(modelled - collected)},
        "states_in_both": len(ql_states & modelled),
        "states_only_q_learning": sorted(ql_states - modelled),
        "states_only_pi_vi_model": sorted(modelled - ql_states),
        "q_learning_states_absent_from_random_collection": sorted(ql_states - collected),
        "training_visit_share_of_states_only_via_closure_or_missing": float(share[sorted(ql_states - collected)].sum()),
        "pi_on_policy_states": len(pi_onpolicy),
        "top_training_states": [{"state": int(s), "share_pct": float(100 * share[s]),
                                 "in_random_collection": int(s) in collected, "in_pi_vi_model": int(s) in modelled,
                                 "pi_evaluation_visits": int(pi_onpolicy.get(int(s), 0))} for s in top],
    }


def focus_states(agent: QLearning, pi_policy: np.ndarray, encoder: TabularStateEncoder, states) -> list[dict]:
    policy = agent.get_policy()
    rows = []
    for s in states:
        adm = agent.admissible()[s]
        rows.append({"state": int(s), **encoder.describe(encoder.decode(s)),
                     "pi_vi_action": describe_action(pi_policy[s]), "q_learning_action": describe_action(policy[s]),
                     "training_visits": int(agent.state_counts[s]),
                     "q_values": {describe_action(a): float(agent.q_table[s, a]) for a in range(N_TABULAR_ACTIONS)
                                  if adm[a]},
                     "times_taken": {describe_action(a): int(agent.visit_counts[s, a]) for a in range(N_TABULAR_ACTIONS)
                                     if adm[a]}})
    return rows


def training_summary(rows: list[dict]) -> dict:
    def block(sl):
        part = rows[sl]
        return {k: float(np.mean([r[k] for r in part])) for k in
                ("total_reward", "energy_kwh", "carbon_kg", "sla_failures", "completed_jobs",
                 "critical_rack_minutes", "epsilon", "max_abs_q_update", "mean_abs_q_update", "q_entries_changed")}
    n = len(rows)
    return {"episodes": n, "first_100": block(slice(0, min(100, n))), "last_100": block(slice(max(0, n - 100), n)),
            "epsilon_initial": rows[0]["epsilon"], "epsilon_final": rows[-1]["epsilon"],
            "explore_steps": int(sum(r["explore_steps"] for r in rows)),
            "exploit_steps": int(sum(r["exploit_steps"] for r in rows)),
            "invalid_actions_total": int(sum(r["invalid_actions"] for r in rows)),
            "invalid_action_rate": float(sum(r["invalid_actions"] for r in rows)
                                         / sum(r["explore_steps"] + r["exploit_steps"] for r in rows))}


# --------------------------------------------------------------------------- #
def make_plots(rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = np.array([r["episode"] for r in rows]) + 1
    paths = []

    def curve(key, title, ylabel, fname, log=False):
        y = np.array([r[key] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
        ax.plot(ep, y, color=SERIES, alpha=0.25, linewidth=1, label="per episode (raw)")
        ax.plot(ep, moving_average(y, MA_WINDOW), color=SERIES, linewidth=2, label=f"{MA_WINDOW}-episode moving average")
        if log:
            ax.set_yscale("log")
        style_axes(ax, title, "training episode", ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(out_dir / fname)

    curve("total_reward", "Episode reward (24 h, exploring)", "total reward", "reward.png")
    curve("sla_failures", "SLA failures per training episode", "failed jobs", "sla_failures.png")
    curve("energy_kwh", "Energy per training episode", "kWh", "energy.png")
    curve("carbon_kg", "Carbon per training episode", "kg CO₂", "carbon.png")
    curve("critical_rack_minutes", "Critical rack-minutes per training episode", "rack-minutes ≥ 30 °C",
          "critical_minutes.png")

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    ax.plot(ep, [r["epsilon"] for r in rows], color=SERIES, linewidth=2)
    style_axes(ax, "Exploration rate ε (per episode)", "training episode", "ε")
    fig.tight_layout(), fig.savefig(out_dir / "epsilon.png", dpi=150, facecolor=SURFACE), plt.close(fig)
    paths.append(out_dir / "epsilon.png")

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.plot(ep, [r["max_abs_q_update"] for r in rows], color=SERIES, linewidth=1.5, label="max |ΔQ| in episode")
    ax.plot(ep, [r["mean_abs_q_update"] for r in rows], color=SERIES_2, linewidth=1.5, label="mean |ΔQ| per update")
    ax.set_yscale("log")
    style_axes(ax, "Q-table updates per episode", "training episode", "|ΔQ| (reward units)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(), fig.savefig(out_dir / "q_updates.png", dpi=150, facecolor=SURFACE), plt.close(fig)
    paths.append(out_dir / "q_updates.png")
    return paths


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.paths.models / "q_learning")
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    exp = cfg.paths.experiments
    out = args.out or exp / f"q_learning_{len(args.seeds)}x{args.minutes}min.json"

    agent = QLearning.load(args.model_dir)
    assert agent.get_q_table().shape == (144, N_TABULAR_ACTIONS)
    rows = read_csv(Path(agent.metadata["training"]["training_csv"]))
    encoder = TabularStateEncoder(cfg.discretisation)
    set_global_seed(args.seeds[0])
    env = DataCenterEnv(representation="tabular", episode_minutes=args.minutes)

    # Main: greedy over the live mask.
    controller = QLearningController(agent, cfg)
    episodes = [run_episode(env, controller, s) for s in args.seeds]
    summary = summarise(episodes)
    # Ablation: the extracted policy table, no live mask (as PI/VI act).
    table_controller = PolicyIterationController(agent.get_policy(), cfg)
    table_episodes = [run_episode(env, table_controller, s) for s in args.seeds]
    table_summary = summarise(table_episodes)
    # Ablation 2: PI/VI's Q-values, greedy over the live mask (model-based + mask).
    pi_loaded = load_pi(cfg.paths.models / "policy_iteration")
    q_pi = q_values(pi_loaded.model.to_mdp(), pi_loaded.value_function, pi_loaded.payload["gamma"])
    pi_mask_episodes = [run_episode(env, MaskedGreedyController(q_pi, agent.tie_tolerance, cfg), s)
                        for s in args.seeds]
    pi_mask_summary = summarise(pi_mask_episodes)

    baselines = json.loads((exp / "baselines_5x1440min.json").read_text(encoding="utf-8"))
    pi_results = json.loads((exp / "policy_iteration_5x1440min.json").read_text(encoding="utf-8"))
    vi_results = json.loads((exp / "value_iteration_5x1440min.json").read_text(encoding="utf-8"))
    pi_payload = json.loads((cfg.paths.models / "policy_iteration" / "policy.json").read_text(encoding="utf-8"))
    protocol = {"seeds": args.seeds, "episode_minutes": args.minutes}
    same_protocol = all(src["config"][k] == v for src in (baselines, pi_results, vi_results)
                        for k, v in protocol.items())

    def row(name, source, summ):
        return {"controller": name, "source": source, **{m: summ[m] for m, _ in COMPARISON},
                "invalid_action_rate": summ["invalid_action_rate"]}
    comparison = [row(n, "Step 6 (inherited)", d["summary"]) for n, d in baselines["controllers"].items()]
    comparison += [row("policy_iteration", "Step 7 (inherited)", pi_results["summary"]),
                   row("value_iteration", "Step 8 (inherited)", vi_results["summary"]),
                   row("q_learning", "Step 9 (this run)", summary),
                   row("q_learning_table_no_mask", "Step 9 ablation (this run)", table_summary),
                   row("pi_vi_q_with_live_mask", "Step 9 ablation (this run)", pi_mask_summary)]

    pi_policy = np.asarray(pi_payload["policy"])
    q_policy = agent.get_policy()
    both = np.flatnonzero((agent.state_counts > 0) & np.isin(np.arange(144), pi_payload["model"]["modelled_states"]))
    pi_freq = pi_results["policy_analysis"]["action_frequency_in_evaluation"]
    action_freq = [{"action": describe_action(a), "q_learning": int(controller.action_counts[a]),
                    "pi_vi": int(pi_freq[describe_action(a)]),
                    "difference": int(controller.action_counts[a] - pi_freq[describe_action(a)])}
                   for a in range(N_TABULAR_ACTIONS)]
    eval_visits = controller.state_visits
    top_eval = [int(s) for s in np.argsort(-eval_visits, kind="stable")[:10] if eval_visits[s] > 0]

    results = {
        "config": {**protocol, "start_hour": cfg.simulation.start_hour, "model_dir": str(args.model_dir),
                   "same_protocol_as_inherited_results": same_protocol, "greedy_epsilon": controller.epsilon},
        "hyperparameters": agent.hyperparameters(), "training": agent.metadata["training"],
        "training_summary": training_summary(rows),
        "summary": summary, "episodes": episodes, "comparison": comparison,
        "ablation_policy_table_without_live_mask": {"summary": table_summary, "episodes": table_episodes},
        "ablation_pi_vi_q_with_live_mask": {"summary": pi_mask_summary, "episodes": pi_mask_episodes},
        "action_frequency_in_evaluation": action_freq,
        "coverage_and_model_gap": coverage_and_model_gap(agent, pi_payload, pi_results),
        "policy_vs_pi_vi": {"states_compared": int(len(both)),
                            "same_action": int(np.sum(q_policy[both] == pi_policy[both])),
                            "agreement_pct": float(100 * np.mean(q_policy[both] == pi_policy[both]))},
        "focus_states_pi_vi_delay": focus_states(agent, pi_policy, encoder, FOCUS_STATES),
        "most_visited_states_in_evaluation": [{**r, "evaluation_visits": int(eval_visits[r["state"]])}
                                              for r in focus_states(agent, pi_policy, encoder, top_eval)],
        "states_visited_in_evaluation": int(np.count_nonzero(eval_visits)),
        "plots": [str(p) for p in make_plots(rows, exp / "plots" / "q_learning")],
    }
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    t = results["training_summary"]
    c = results["coverage_and_model_gap"]
    print(f"Training: {t['episodes']} episodes, eps {t['epsilon_initial']:.3f} -> {t['epsilon_final']:.4f}, "
          f"explore/exploit {t['explore_steps']}/{t['exploit_steps']}, invalid rate {t['invalid_action_rate']:.4f}")
    print(f"  reward first/last 100: {t['first_100']['total_reward']:.1f} / {t['last_100']['total_reward']:.1f}; "
          f"SLA {t['first_100']['sla_failures']:.1f} / {t['last_100']['sla_failures']:.1f}; "
          f"mean |dQ| {t['first_100']['mean_abs_q_update']:.4f} / {t['last_100']['mean_abs_q_update']:.4f}")
    print(f"Coverage: {c['states_visited']}/144 states ({c['states_visited_pct']:.1f} %), "
          f"{c['state_action_pairs_visited']}/{c['state_action_pairs_total']} pairs; in PI/VI model too: "
          f"{c['states_in_both']}; only Q-Learning: {c['states_only_q_learning']}; only model: {c['states_only_pi_vi_model']}")
    p = results["policy_vs_pi_vi"]
    print(f"Policy table vs PI/VI: {p['same_action']}/{p['states_compared']} ({p['agreement_pct']:.1f} %)")
    print(f"\nQ-Learning greedy (eps=0, live mask), {len(args.seeds)} x {args.minutes} min:")
    for m in METRICS:
        print(f"  {m:<32}{summary[m]['mean']:>12.2f} +/- {summary[m]['std']:.2f}")
    print(f"\n{'controller':<26}{'source':<28}" + "".join(f"{label:>17}" for _, label in COMPARISON) + "   invalid")
    for r in comparison:
        print(f"{r['controller']:<26}{r['source']:<28}"
              + "".join(f"{r[m]['mean']:>10.1f} ±{r[m]['std']:>5.1f}" for m, _ in COMPARISON)
              + f"   {r['invalid_action_rate']['mean']:.3f}")
    for f in results["focus_states_pi_vi_delay"]:
        print(f"state {f['state']}: PI/VI {f['pi_vi_action']} | QL {f['q_learning_action']} | visits {f['training_visits']}")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
