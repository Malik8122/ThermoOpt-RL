"""Evaluate PPO (Step 11): 5 x 24 h for the design-protocol model, the no-mask ablation and the budget-matched
(432 k-step) model; comparison with all earlier controllers (inherited results); state-information analysis;
policy and value diagnostics.

Primary evaluation is deterministic (most probable valid action), matching the greedy protocol of Steps 9-10.
Stochastic evaluation (sampling from the policy, seeded per episode) is reported as secondary.

    python -m backend.rl.evaluate_ppo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.config import get_config  # noqa: E402
from backend.evaluation.runner import METRICS, run_episode, summarise  # noqa: E402
from backend.rl.actions import ActionCodec  # noqa: E402
from backend.rl.dqn import configure_torch  # noqa: E402
from backend.rl.environment import DataCenterEnv  # noqa: E402
from backend.rl.evaluate_dqn import state_information  # noqa: E402
from backend.rl.evaluate_policy_iteration import COMPARISON, INK, SERIES, SURFACE, style_axes  # noqa: E402
from backend.rl.observations import ObservationBuilder  # noqa: E402
from backend.rl.ppo import PPOAgent, PPOController, entropy_of, masked_log_probs  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402
from backend.utils.seeding import set_global_seed  # noqa: E402

logger = get_logger("evaluate_ppo")

COLORS = {"ppo": SERIES, "ppo_nomask": "#eb6834", "ppo_432k": "#1baf7a"}     # categorical slots 1-3
LABELS = {"ppo": "PPO (design, 199 k steps)", "ppo_nomask": "PPO no mask (199 k)", "ppo_432k": "PPO (430 k steps)"}
MA_WINDOW = 10


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def batch_log_probs(agent: PPOAgent, obs: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        m = torch.as_tensor(masks) if agent.use_action_mask else None
        return masked_log_probs(agent.model.logits(torch.as_tensor(obs, dtype=torch.float32)), m)


def greedy_fn(agent: PPOAgent):
    return lambda obs, masks: batch_log_probs(agent, obs, masks).argmax(dim=1).numpy()


def policy_diagnostics(agent: PPOAgent, ctrl: PPOController, builder: ObservationBuilder, codec: ActionCodec) -> dict:
    obs, masks = np.stack(ctrl.observations), np.stack(ctrl.masks)
    lp = batch_log_probs(agent, obs, masks)
    probs = lp.exp().numpy().astype(np.float64)
    ent = entropy_of(lp).numpy()
    with torch.no_grad():
        values = agent.model.value(torch.as_tensor(obs)).numpy().astype(np.float64)
    n_valid = masks.sum(axis=1) if agent.use_action_mask else np.full(len(obs), agent.n_actions)
    top = probs.max(axis=1)
    hottest = builder.t_min + obs[:, 0:4].max(axis=1) * builder.t_span
    day = slice(0, min(1440, len(obs)))
    reps = []
    for qtl in (0, 25, 50, 75, 100):          # representative states: quantiles of hottest-rack temperature, first day
        i = int(np.argmin(np.abs(hottest[day] - np.percentile(hottest[day], qtl))))
        order = np.argsort(-probs[i], kind="stable")[:3]
        reps.append({"hottest_rack_quantile": qtl, "step": i, "hottest_rack_c": float(hottest[i]),
                     "valid_actions": int(n_valid[i]), "value": float(values[i]), "entropy": float(ent[i]),
                     "top_actions": [{"action": codec.decode(int(a)).describe(), "probability": float(probs[i, a])}
                                     for a in order]})
    stat = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)),  # noqa: E731
                      "max": float(np.max(x))}
    return {"all_finite": bool(np.isfinite(probs).all() and np.isfinite(values).all()),
            "entropy_nats": stat(ent),
            "entropy_relative_to_uniform_over_valid": stat(ent / np.log(np.maximum(n_valid, 2))),
            "top_action_probability": {**stat(top), "pct_above_0.9": float(100 * np.mean(top > 0.9)),
                                       "pct_below_0.5": float(100 * np.mean(top < 0.5))},
            "value_estimate": stat(values),
            "representative_states": reps}


def training_summary(episodes: list[dict], updates: list[dict], n: int = 20) -> dict:
    ek = ("total_reward", "energy_kwh", "carbon_kg", "sla_failures", "critical_rack_minutes", "invalid_action_rate",
          "mean_entropy", "mean_value")
    uk = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance")
    blk = lambda rows, keys: {k: float(np.nanmean([r[k] for r in rows])) for k in keys}  # noqa: E731
    return {"episodes_completed": len(episodes), "rollouts": len(updates),
            f"episodes_first_{n}": blk(episodes[:n], ek), f"episodes_last_{n}": blk(episodes[-n:], ek),
            "updates_first_10": blk(updates[:10], uk), "updates_last_10": blk(updates[-10:], uk),
            "max_approx_kl": float(max(r["approx_kl"] for r in updates))}


def paired(new: list[dict], other: list[dict]) -> dict:
    d = np.array([a["total_reward"] - b["total_reward"] for a, b in zip(new, other)])
    return {"per_seed": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "wins": int((d > 0).sum())}


# --------------------------------------------------------------------------- #
def make_plots(runs: dict[str, tuple[list[dict], list[dict]]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def ma(y):
        out = np.full(len(y), np.nan)
        if len(y) >= MA_WINDOW:
            out[MA_WINDOW - 1:] = np.convolve(y, np.ones(MA_WINDOW) / MA_WINDOW, mode="valid")
        return out

    def curve(which, key, title, ylabel, fname, log=False, smooth=True):
        fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
        for tag, data in runs.items():
            rows = data[which]
            x = np.array([r["env_steps"] for r in rows]) / 1000
            y = np.array([r[key] for r in rows])
            if smooth:
                ax.plot(x, y, color=COLORS[tag], alpha=0.2, linewidth=1)
                ax.plot(x, ma(y), color=COLORS[tag], linewidth=2, label=LABELS[tag])
            else:
                ax.plot(x, y, color=COLORS[tag], linewidth=1.8, label=LABELS[tag])
        if log:
            ax.set_yscale("log")
        suffix = f" ({MA_WINDOW}-episode average; faint = raw)" if smooth else ""
        style_axes(ax, title + suffix, "environment steps (thousands)", ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(out_dir / fname)

    curve(0, "total_reward", "Episode reward while sampling", "total reward", "reward.png")
    curve(0, "sla_failures", "SLA failures per episode", "failed jobs", "sla_failures.png")
    curve(0, "energy_kwh", "Energy per episode", "kWh", "energy.png")
    curve(0, "carbon_kg", "Carbon per episode", "kg CO₂", "carbon.png")
    curve(0, "critical_rack_minutes", "Critical rack-minutes per episode", "rack-minutes ≥ 30 °C", "critical_minutes.png")
    curve(0, "invalid_action_rate", "Invalid-action rate per episode", "share of steps", "invalid_rate.png")
    curve(1, "policy_loss", "Policy (clipped surrogate) loss per update", "loss", "policy_loss.png", smooth=False)
    curve(1, "value_loss", "Value loss per update", "0.5 · (R − V)²", "value_loss.png", log=True, smooth=False)
    curve(1, "entropy", "Policy entropy per update", "nats", "entropy.png", smooth=False)
    curve(1, "approx_kl", "Approximate KL per update", "KL", "approx_kl.png", smooth=False)
    curve(1, "explained_variance", "Critic explained variance per update", "1 − Var(R − V) / Var(R)",
          "explained_variance.png", smooth=False)
    return paths


# --------------------------------------------------------------------------- #
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--tags", nargs="+", default=["ppo", "ppo_nomask", "ppo_432k"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    exp = cfg.paths.experiments
    out = args.out or exp / f"ppo_{len(args.seeds)}x{args.minutes}min.json"

    configure_torch(cfg.rl.ppo.torch_threads)
    set_global_seed(args.seeds[0])
    agents = {tag: PPOAgent.load(cfg.paths.models / tag) for tag in args.tags}
    env = DataCenterEnv(representation="features", episode_minutes=args.minutes)
    codec, builder = ActionCodec(cfg.datacenter.num_racks), ObservationBuilder(cfg)

    evals: dict[str, dict] = {}
    for tag, agent in agents.items():
        det = PPOController(agent, cfg, deterministic=True, record=True)
        det_eps = [run_episode(env, det, s) for s in args.seeds]
        sto = PPOController(agent, cfg, deterministic=False)
        sto_eps = [run_episode(env, sto, s) for s in args.seeds]
        evals[tag] = {"controller": det, "deterministic": {"episodes": det_eps, "summary": summarise(det_eps)},
                      "stochastic": {"episodes": sto_eps, "summary": summarise(sto_eps)},
                      "stochastic_action_counts": sto.action_counts.tolist()}

    load = lambda name: json.loads((exp / name).read_text(encoding="utf-8"))  # noqa: E731
    base, ql, dqn = load("baselines_5x1440min.json"), load("q_learning_5x1440min.json"), load("dqn_5x1440min.json")
    inherited = {**{n: ("Step 6 (inherited)", d["summary"], d["episodes"]) for n, d in base["controllers"].items()},
                 "policy_iteration": ("Step 7 (inherited)", load("policy_iteration_5x1440min.json")["summary"], None),
                 "value_iteration": ("Step 8 (inherited)", load("value_iteration_5x1440min.json")["summary"], None),
                 "q_learning": ("Step 9 (inherited)", ql["summary"], ql["episodes"]),
                 "dqn": ("Step 10 (inherited)", dqn["summary"], dqn["episodes"])}
    row = lambda name, src, s: {"controller": name, "source": src, **{m: s[m] for m, _ in COMPARISON},  # noqa: E731
                                "invalid_action_rate": s["invalid_action_rate"]}
    comparison = [row(n, src, s) for n, (src, s, _) in inherited.items()]
    names = {"ppo": "ppo", "ppo_nomask": "ppo_no_mask_ablation", "ppo_432k": "ppo_432k_steps"}
    for tag in agents:
        comparison.append(row(names[tag], "Step 11 (this run)", evals[tag]["deterministic"]["summary"]))
        comparison.append(row(names[tag] + "_stochastic", "Step 11 (this run)", evals[tag]["stochastic"]["summary"]))

    main_tag = args.tags[0]
    main_eps = evals[main_tag]["deterministic"]["episodes"]
    train = {tag: (read_csv(Path(a.metadata["training"]["episode_csv"])), read_csv(Path(a.metadata["training"]["update_csv"])))
             for tag, a in agents.items()}
    results = {
        "config": {"seeds": args.seeds, "episode_minutes": args.minutes, "start_hour": cfg.simulation.start_hour,
                   "primary_evaluation": "deterministic (argmax of the masked policy)",
                   "device": str(agents[main_tag].device)},
        "hyperparameters": agents[main_tag].hyperparameters(),
        "architecture": agents[main_tag].metadata["architecture"],
        "training": {tag: {**a.metadata["training"], "summary": training_summary(*train[tag])}
                     for tag, a in agents.items()},
        "summary": evals[main_tag]["deterministic"]["summary"], "episodes": main_eps,
        "models": {tag: {"deterministic": evals[tag]["deterministic"], "stochastic": evals[tag]["stochastic"]}
                   for tag in agents},
        "comparison": comparison,
        "paired_reward_difference_vs": {n: paired(main_eps, eps) for n, (_, _, eps) in inherited.items() if eps},
        "action_frequency_in_evaluation": [
            {"action": codec.decode(a).describe(),
             **{tag: int(evals[tag]["controller"].action_counts[a]) for tag in agents}} for a in range(codec.n_actions)],
        "state_information": state_information(greedy_fn(agents[main_tag]), evals[main_tag]["controller"], builder, codec),
        "policy_diagnostics": {tag: policy_diagnostics(agents[tag], evals[tag]["controller"], builder, codec)
                               for tag in agents},
        "plots": [str(p) for p in make_plots(train, exp / "plots" / "ppo")],
    }
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for tag in agents:
        t = results["training"][tag]
        s = t["summary"]
        print(f"[{tag}] {t['total_env_steps']} steps, {t['rollouts']} rollouts, {t['gradient_steps']} gradient steps, "
              f"{t['seconds']:.0f} s; reward first/last 20 ep: {s['episodes_first_20']['total_reward']:.1f} / "
              f"{s['episodes_last_20']['total_reward']:.1f}; entropy {s['updates_first_10']['entropy']:.2f} -> "
              f"{s['updates_last_10']['entropy']:.2f}; EV {s['updates_last_10']['explained_variance']:.2f}")
    print(f"\nPPO deterministic, {len(args.seeds)} x {args.minutes} min:")
    for m in METRICS:
        print(f"  {m:<32}" + "".join(f"{evals[t]['deterministic']['summary'][m]['mean']:>12.2f} ±{evals[t]['deterministic']['summary'][m]['std']:>7.2f}"
                                    for t in agents) + "   (" + ", ".join(agents) + ")")
    print(f"\n{'controller':<30}{'source':<22}" + "".join(f"{label:>17}" for _, label in COMPARISON) + "  invalid")
    for r in comparison:
        print(f"{r['controller']:<30}{r['source']:<22}"
              + "".join(f"{r[m]['mean']:>10.1f} ±{r[m]['std']:>5.1f}" for m, _ in COMPARISON)
              + f"  {r['invalid_action_rate']['mean']:.3f}")
    print("\npaired (PPO - other):", {k: (round(v["mean"], 1), round(v["sd"], 1), f"{v['wins']}/5")
                                      for k, v in results["paired_reward_difference_vs"].items()})
    si = results["state_information"]
    print("placement:", si["placement"])
    print("top sensitivity:", [(r["feature"], round(r["action_change_pct"], 1)) for r in si["permutation_sensitivity"][:8]])
    for r in si["actions_by_hottest_rack_temperature"]:
        print(r)
    print("policy:", {k: v for k, v in results["policy_diagnostics"][main_tag].items() if k != "representative_states"})
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
