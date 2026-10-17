"""Evaluate DQN (Step 10): greedy 5 x 24 h for the masked model and the no-mask ablation, comparison with all
earlier controllers (inherited results), state-information analysis and Q-value diagnostics.

    python -m backend.rl.evaluate_dqn
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
from backend.rl.actions import ActionCodec, CoolingCommand, PlacementKind  # noqa: E402
from backend.rl.dqn import DQNAgent, DQNController, configure_torch  # noqa: E402
from backend.rl.environment import DataCenterEnv  # noqa: E402
from backend.rl.evaluate_policy_iteration import COMPARISON, INK, SERIES, SURFACE, style_axes  # noqa: E402
from backend.rl.observations import ObservationBuilder  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402
from backend.utils.seeding import derive_seed, make_rng, set_global_seed  # noqa: E402

logger = get_logger("evaluate_dqn")

SERIES_2 = "#eb6834"
MA_WINDOW = 20
TEMP_BINS = (("< 25 °C", -np.inf, 25.0), ("25–27 °C", 25.0, 27.0), ("27–30 °C", 27.0, 30.0), ("≥ 30 °C", 30.0, np.inf))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def masked_greedy(agent: DQNAgent, obs: np.ndarray, masks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Batch greedy actions and Q-values (Q of invalid actions set to -inf when masks are given)."""
    with torch.no_grad():
        q = agent.online(torch.as_tensor(obs, dtype=torch.float32)).numpy().astype(np.float64)
    qm = np.where(masks, q, -np.inf) if masks is not None else q
    return np.argmax(qm, axis=1), qm


# --------------------------------------------------------------------------- #
def state_information(greedy, ctrl, builder: ObservationBuilder, codec: ActionCodec) -> dict:
    """``greedy(obs_batch, mask_batch) -> actions`` is the controller's deterministic policy (DQN or PPO);
    ``ctrl`` supplies the recorded observations, masks and actions."""
    obs, masks, acts = np.stack(ctrl.observations), np.stack(ctrl.masks), np.asarray(ctrl.actions)
    names = builder.feature_names
    temps = builder.t_min + obs[:, 0:4] * builder.t_span
    hottest = temps.max(axis=1)
    decoded = [codec.decode(int(a)) for a in acts]
    cooling = np.array([int(d.cooling) for d in decoded])
    kind = np.array([d.kind.value for d in decoded])

    by_temperature = []
    for label, lo, hi in TEMP_BINS:
        sel = (hottest >= lo) & (hottest < hi)
        n = int(sel.sum())
        by_temperature.append({
            "hottest_rack": label, "steps": n,
            "cooling_pct": {c.name.lower(): float(100 * np.mean(cooling[sel] == int(c))) if n else 0.0
                            for c in CoolingCommand},
            "placement_pct": {k.value.lower(): float(100 * np.mean(kind[sel] == k.value)) if n else 0.0
                              for k in PlacementKind}})

    # Where does it place jobs? Compare with the coolest / least-utilised feasible rack.
    fits = obs[:, names.index("head_fits_rack1"):names.index("head_fits_rack1") + 4] > 0.5
    util = 0.5 * obs[:, 4:8] + 0.5 * obs[:, 8:12]
    waiting = obs[:, names.index("job_waiting")] > 0.5
    place = np.flatnonzero((kind == PlacementKind.PLACE.value) & waiting)       # real placements only
    place_noop = int(np.sum((kind == PlacementKind.PLACE.value) & ~waiting))   # "place" on an empty queue
    coolest = least = 0
    for i in place:
        feas = np.flatnonzero(fits[i])
        rack = decoded[i].rack_index
        coolest += rack == feas[np.argmin(temps[i, feas])]
        least += rack == feas[np.argmin(util[i, feas])]
    rack_share = {f"rack{r + 1}": float(100 * np.mean([decoded[i].rack_index == r for i in place])) if len(place) else 0
                  for r in range(4)}

    # Permutation sensitivity: shuffle one feature across the recorded observations, keep the recorded masks,
    # count how often the greedy action changes. Descriptive only; features are correlated, so not causal.
    rng = make_rng(derive_seed(0, "dqn-permutation"))
    base = greedy(obs, masks)
    sensitivity = []
    for j, name in enumerate(names):
        shuffled = obs.copy()
        shuffled[:, j] = obs[rng.permutation(len(obs)), j]
        changed = greedy(shuffled, masks)
        sensitivity.append({"feature": name, "action_change_pct": float(100 * np.mean(changed != base))})
    sensitivity.sort(key=lambda r: -r["action_change_pct"])

    return {
        "observations": int(len(obs)),
        "feature_ranges": [{"feature": n, "min": float(obs[:, j].min()), "max": float(obs[:, j].max()),
                            "mean": float(obs[:, j].mean()), "std": float(obs[:, j].std())}
                           for j, n in enumerate(names)],
        "all_within_observation_space": bool(((obs >= builder.low) & (obs <= builder.high)).all()),
        "actions_by_hottest_rack_temperature": by_temperature,
        "placement": {"place_steps_with_job_waiting": int(len(place)), "place_steps_on_empty_queue": place_noop,
                      "steps_with_job_waiting": int(waiting.sum()),
                      "delay_steps_with_job_waiting": int(np.sum((kind == PlacementKind.DELAY.value) & waiting)),
                      "chose_coolest_feasible_rack_pct": float(100 * coolest / max(1, len(place))),
                      "chose_least_utilised_feasible_rack_pct": float(100 * least / max(1, len(place))),
                      "share_by_rack_pct": rack_share},
        "permutation_sensitivity": sensitivity,
    }


def q_diagnostics(agent: DQNAgent, ctrl: DQNController, builder: ObservationBuilder, codec: ActionCodec) -> dict:
    obs, masks = np.stack(ctrl.observations), np.stack(ctrl.masks)
    with torch.no_grad():
        q = agent.online(torch.as_tensor(obs)).numpy().astype(np.float64)
    valid_q = np.where(masks, q, np.nan)
    best = np.nanmax(valid_q, axis=1)
    spread = best - np.nanmin(valid_q, axis=1)
    srt = np.sort(np.where(masks, q, -np.inf), axis=1)[:, ::-1]
    gap = np.where(masks.sum(1) > 1, srt[:, 0] - srt[:, 1], np.nan)
    hottest = builder.t_min + obs[:, 0:4].max(axis=1) * builder.t_span
    first_day = slice(0, 1440)
    reps = []
    for qtl in (0, 25, 50, 75, 100):          # representative states: quantiles of hottest-rack temperature, seed 42
        target = np.percentile(hottest[first_day], qtl)
        i = int(np.argmin(np.abs(hottest[first_day] - target)))
        order = [a for a in np.argsort(-q[i], kind="stable") if masks[i, a]][:3]
        reps.append({"hottest_rack_quantile": qtl, "step": i, "hottest_rack_c": float(hottest[i]),
                     "valid_actions": int(masks[i].sum()),
                     "top_actions": [{"action": codec.decode(int(a)).describe(), "q": float(q[i, a])} for a in order],
                     "chosen": codec.decode(int(ctrl.actions[i])).describe()})
    return {"all_finite": bool(np.isfinite(q).all()),
            "q_all_actions": {"mean": float(q.mean()), "std": float(q.std()), "min": float(q.min()), "max": float(q.max())},
            "best_valid_q": {"mean": float(best.mean()), "std": float(best.std()), "min": float(best.min()),
                             "max": float(best.max())},
            "valid_action_spread": {"mean": float(np.mean(spread)), "median": float(np.median(spread)),
                                    "max": float(spread.max())},
            "best_vs_second_gap": {"median": float(np.nanmedian(gap)),
                                   "pct_below_0.01": float(100 * np.nanmean(gap < 0.01))},
            "representative_states": reps}


def training_summary(rows: list[dict], n: int = 30) -> dict:
    keys = ("total_reward", "energy_kwh", "carbon_kg", "sla_failures", "critical_rack_minutes", "invalid_action_rate",
            "mean_loss", "mean_q", "epsilon")
    blk = lambda part: {k: float(np.nanmean([r[k] for r in part])) for k in keys}  # noqa: E731
    return {"episodes": len(rows), f"first_{n}": blk(rows[:n]), f"last_{n}": blk(rows[-n:]),
            "env_steps": int(rows[-1]["env_steps"]), "training_updates": int(rows[-1]["training_updates"]),
            "target_updates": int(rows[-1]["target_updates"]),
            "explore_steps": int(sum(r["explore_steps"] for r in rows)),
            "exploit_steps": int(sum(r["exploit_steps"] for r in rows)),
            "max_loss": float(np.nanmax([r["max_loss"] for r in rows]))}


# --------------------------------------------------------------------------- #
def make_plots(runs: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"masked": SERIES, "no mask": SERIES_2}
    paths = []

    def ma(y):
        out = np.full(len(y), np.nan)
        if len(y) >= MA_WINDOW:
            out[MA_WINDOW - 1:] = np.convolve(y, np.ones(MA_WINDOW) / MA_WINDOW, mode="valid")
        return out

    def curve(key, title, ylabel, fname, log=False, smooth=True):
        fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
        for label, rows in runs.items():
            x = np.array([r["episode"] for r in rows]) + 1
            y = np.array([r[key] for r in rows])
            if smooth:
                ax.plot(x, y, color=colors[label], alpha=0.22, linewidth=1)
                ax.plot(x, ma(y), color=colors[label], linewidth=2, label=f"{label} ({MA_WINDOW}-ep. avg; faint = raw)")
            else:
                ax.plot(x, y, color=colors[label], linewidth=2, label=label)
        if log:
            ax.set_yscale("log")
        style_axes(ax, title, "training episode (24 h)", ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(out_dir / fname)

    curve("total_reward", "Episode reward (exploring)", "total reward", "reward.png")
    curve("energy_kwh", "Energy per episode", "kWh", "energy.png")
    curve("carbon_kg", "Carbon per episode", "kg CO₂", "carbon.png")
    curve("sla_failures", "SLA failures per episode", "failed jobs", "sla_failures.png")
    curve("critical_rack_minutes", "Critical rack-minutes per episode", "rack-minutes ≥ 30 °C", "critical_minutes.png")
    curve("epsilon", "Exploration rate ε (end of episode)", "ε", "epsilon.png", smooth=False)
    curve("mean_loss", "Huber loss (mean per episode)", "loss", "loss.png", log=True)
    curve("mean_q", "Mean predicted Q of taken actions (training batches)", "Q", "mean_q.png")
    curve("invalid_action_rate", "Invalid-action rate per episode", "share of steps", "invalid_rate.png")
    return paths


# --------------------------------------------------------------------------- #
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.paths.models / "dqn")
    parser.add_argument("--nomask-dir", type=Path, default=cfg.paths.models / "dqn_nomask")
    parser.add_argument("--minutes", type=int, default=cfg.evaluation.episode_minutes)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg.evaluation.seeds)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    exp = cfg.paths.experiments
    out = args.out or exp / f"dqn_{len(args.seeds)}x{args.minutes}min.json"

    configure_torch(cfg.rl.dqn.torch_threads)
    set_global_seed(args.seeds[0])
    agents = {"masked": DQNAgent.load(args.model_dir), "no mask": DQNAgent.load(args.nomask_dir)}
    assert agents["masked"].use_action_mask and not agents["no mask"].use_action_mask
    env = DataCenterEnv(representation="features", episode_minutes=args.minutes)
    codec, builder = ActionCodec(cfg.datacenter.num_racks), ObservationBuilder(cfg)

    evaluations = {}
    for label, agent in agents.items():
        ctrl = DQNController(agent, cfg, record=True)
        episodes = [run_episode(env, ctrl, s) for s in args.seeds]
        evaluations[label] = {"controller": ctrl, "episodes": episodes, "summary": summarise(episodes)}

    inherited = {
        **{n: ("Step 6 (inherited)", d["summary"]) for n, d in
           json.loads((exp / "baselines_5x1440min.json").read_text(encoding="utf-8"))["controllers"].items()},
        "policy_iteration": ("Step 7 (inherited)", json.loads((exp / "policy_iteration_5x1440min.json")
                                                               .read_text(encoding="utf-8"))["summary"]),
        "value_iteration": ("Step 8 (inherited)", json.loads((exp / "value_iteration_5x1440min.json")
                                                             .read_text(encoding="utf-8"))["summary"]),
        "q_learning": ("Step 9 (inherited)", json.loads((exp / "q_learning_5x1440min.json")
                                                        .read_text(encoding="utf-8"))["summary"]),
    }
    comparison = [{"controller": n, "source": src, **{m: s[m] for m, _ in COMPARISON},
                   "invalid_action_rate": s["invalid_action_rate"]} for n, (src, s) in inherited.items()]
    for label, name in (("masked", "dqn"), ("no mask", "dqn_no_mask_ablation")):
        s = evaluations[label]["summary"]
        comparison.append({"controller": name, "source": "Step 10 (this run)", **{m: s[m] for m, _ in COMPARISON},
                           "invalid_action_rate": s["invalid_action_rate"]})

    train_rows = {label: read_csv(Path(a.metadata["training"]["training_csv"])) for label, a in agents.items()}
    results = {
        "config": {"seeds": args.seeds, "episode_minutes": args.minutes, "start_hour": cfg.simulation.start_hour,
                   "greedy_epsilon": 0.0, "device": str(agents["masked"].device)},
        "training": {label: {**a.metadata["training"], "summary": training_summary(train_rows[label])}
                     for label, a in agents.items()},
        "hyperparameters": agents["masked"].hyperparameters(),
        "architecture": agents["masked"].metadata["architecture"],
        "summary": evaluations["masked"]["summary"], "episodes": evaluations["masked"]["episodes"],
        "ablation_no_mask": {"summary": evaluations["no mask"]["summary"],
                             "episodes": evaluations["no mask"]["episodes"]},
        "comparison": comparison,
        "action_frequency_in_evaluation": [
            {"action": codec.decode(a).describe(),
             "masked": int(evaluations["masked"]["controller"].action_counts[a]),
             "no_mask": int(evaluations["no mask"]["controller"].action_counts[a])} for a in range(codec.n_actions)],
        "state_information": state_information(lambda o, m: masked_greedy(agents["masked"], o, m)[0],
                                               evaluations["masked"]["controller"], builder, codec),
        "q_diagnostics": {label: q_diagnostics(agents[label], evaluations[label]["controller"], builder, codec)
                          for label in agents},
        "plots": [str(p) for p in make_plots({"masked": train_rows["masked"], "no mask": train_rows["no mask"]},
                                              exp / "plots" / "dqn")],
    }
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for label in agents:
        t = results["training"][label]
        ts = t["summary"]
        print(f"[{label}] {t['episodes']} x {t['episode_minutes']} min = {ts['env_steps']} steps, "
              f"{ts['training_updates']} updates, {ts['target_updates']} syncs, {t['seconds']:.0f} s; "
              f"reward first/last 30: {ts['first_30']['total_reward']:.1f} / {ts['last_30']['total_reward']:.1f}")
    print(f"\nDQN greedy (eps=0, live mask), {len(args.seeds)} x {args.minutes} min:")
    for m in METRICS:
        s, a = evaluations["masked"]["summary"][m], evaluations["no mask"]["summary"][m]
        print(f"  {m:<32}{s['mean']:>11.2f} +/- {s['std']:<8.2f}   no-mask {a['mean']:>10.2f} +/- {a['std']:.2f}")
    print(f"\n{'controller':<22}{'source':<22}" + "".join(f"{label:>17}" for _, label in COMPARISON) + "  invalid")
    for r in comparison:
        print(f"{r['controller']:<22}{r['source']:<22}"
              + "".join(f"{r[m]['mean']:>10.1f} ±{r[m]['std']:>5.1f}" for m, _ in COMPARISON)
              + f"  {r['invalid_action_rate']['mean']:.3f}")
    si = results["state_information"]
    print("\nplacement:", si["placement"])
    print("top permutation sensitivity:", si["permutation_sensitivity"][:6])
    for row in si["actions_by_hottest_rack_temperature"]:
        print(row)
    print("Q (masked):", {k: v for k, v in results["q_diagnostics"]["masked"].items() if k != "representative_states"})
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
