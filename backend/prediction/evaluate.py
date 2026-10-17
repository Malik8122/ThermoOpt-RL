"""Evaluate the temperature forecaster (Step 12).

Test set: the last days of the recorded history (never used for training or model selection).
Reported per horizon: MAE, RMSE, R² of the temperature, and skill against two baselines that need no model:
    persistence   T(t+h) = T(t)
    linear trend  T(t+h) = T(t) + h * (T(t) - T(t-1))
Also: accuracy by rack and by behaviour policy, accuracy when cooling did / did not change inside the horizon,
warning-threshold crossing detection, and fresh "generalisation" days including a controller (DQN) whose
behaviour is not in the training data.

    python -m backend.prediction.evaluate
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

from backend.config import get_config  # noqa: E402
from backend.prediction.dataset import History, Windows, generate_history, make_windows, split_days  # noqa: E402
from backend.prediction.model import ForecastProvider, TemperatureForecaster  # noqa: E402
from backend.rl.dqn import configure_torch  # noqa: E402
from backend.rl.environment import DataCenterEnv  # noqa: E402
from backend.rl.evaluate_policy_iteration import GRID, INK, INK_2, SERIES, SURFACE, style_axes  # noqa: E402
from backend.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("prediction.evaluate")

SERIES_2, SERIES_3 = "#eb6834", "#1baf7a"
GENERALISATION_POLICIES = ["random", "round_robin", "rule_based", "ppo", "dqn"]   # dqn is not in the training data
COOLING_CHANGE = 0.02      # |change in received cooling| inside the horizon that counts as "cooling changed"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = predicted - actual
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
            "bias": float(err.mean()), "n": int(len(err))}


def baseline_predictions(w: Windows, temp_index: int, horizons: list[int]) -> dict[str, np.ndarray]:
    now = w.x[:, -1, temp_index]
    slope = now - w.x[:, -2, temp_index]
    return {"persistence": np.repeat(now[:, None], len(horizons), axis=1),
            "linear_trend": now[:, None] + slope[:, None] * np.asarray(horizons)[None, :]}


def evaluate_windows(forecaster: TemperatureForecaster, w: Windows) -> dict:
    """Per-horizon metrics for the model and the baselines on a set of windows."""
    actual = w.target_temperature
    preds = {"model": forecaster.predict(w.x), **baseline_predictions(w, forecaster.temp_index, forecaster.horizons)}
    out = {}
    for j, h in enumerate(forecaster.horizons):
        row = {name: regression_metrics(actual[:, j], p[:, j]) for name, p in preds.items()}
        mse = {k: v["rmse"] ** 2 for k, v in row.items()}
        row["skill_vs_persistence"] = 1 - mse["model"] / mse["persistence"]      # share of persistence MSE removed
        row["mae_reduction_vs_persistence_pct"] = 100 * (1 - row["model"]["mae"] / row["persistence"]["mae"])
        row["actual_change_std_c"] = float(w.y[:, j].std())
        out[f"+{h}"] = row
    return out


def subset(w: Windows, keep: np.ndarray) -> Windows:
    return Windows(w.x[keep], w.y[keep], w.last_temperature[keep], w.day[keep], w.rack[keep], w.minute[keep])


def crossing_detection(forecaster: TemperatureForecaster, w: Windows, threshold: float) -> dict:
    """Among windows that start below ``threshold``: does the forecast flag the ones that end at or above it?"""
    start_below = w.last_temperature < threshold
    actual = w.target_temperature[start_below]
    preds = {"model": forecaster.predict(w.x[start_below]),
             **{k: v[start_below] for k, v in baseline_predictions(w, forecaster.temp_index, forecaster.horizons).items()}}
    out = {}
    for j, h in enumerate(forecaster.horizons):
        truth = actual[:, j] >= threshold
        row = {"windows_starting_below": int(start_below.sum()), "actual_crossings": int(truth.sum())}
        for name, p in preds.items():
            flag = p[:, j] >= threshold
            tp = int(np.sum(flag & truth))
            row[name] = {"precision": tp / int(flag.sum()) if flag.any() else float("nan"),
                         "recall": tp / int(truth.sum()) if truth.any() else float("nan"), "flagged": int(flag.sum())}
        out[f"+{h}"] = row
    return out


def provider_consistency(forecaster: TemperatureForecaster, history: History, day: int, config) -> dict:
    """Replay one recorded day through the live provider and compare with the offline forecasts."""
    env = DataCenterEnv(representation="features", episode_minutes=120, config=config,
                        prediction_provider=ForecastProvider(forecaster))
    obs, _ = env.reset(seed=history.seeds[day])
    columns = [history.feature_names.index(n) for n in forecaster.feature_names]
    first = history.frames[day, 0][:, columns]
    offline = forecaster.predict(np.stack([first] * forecaster.sequence_length, axis=1))
    live = env.observation_builder.prediction_provider.last_forecast
    return {"observation_size": int(obs.shape[0]), "minute0_max_abs_difference_c": float(np.abs(offline - live).max()),
            "observation_in_space": bool(env.observation_space.contains(obs))}


# --------------------------------------------------------------------------- #
def make_plots(forecaster, test: Windows, results: dict, rows: list[dict], history: History, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons, paths = forecaster.horizons, []
    pred, actual = forecaster.predict(test.x), test.target_temperature

    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["train_loss"] for r in rows], color=SERIES, linewidth=2, label="training")
    ax.plot(ep, [r["val_loss"] for r in rows], color=SERIES_2, linewidth=2, label="validation")
    best = min(rows, key=lambda r: r["val_loss"])
    ax.axvline(best["epoch"], color=GRID, linewidth=1.5)
    ax.text(best["epoch"], 0.98, f" best epoch {int(best['epoch'])}", transform=ax.get_xaxis_transform(), va="top",
            fontsize=8, color=INK_2)
    style_axes(ax, "Training curve (MSE of standardised temperature change)", "epoch", "loss")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    paths.append((fig, out_dir / "training_curve.png"))

    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), size=min(4000, len(test)), replace=False)
    lo, hi = float(actual.min()), float(actual.max())
    for j, (h, ax) in enumerate(zip(horizons, np.atleast_1d(axes))):
        ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.5)
        ax.scatter(actual[idx, j], pred[idx, j], s=5, color=SERIES, alpha=0.25, linewidths=0)
        m = results["test"][f"+{h}"]["model"]
        ax.text(0.04, 0.96, f"MAE {m['mae']:.2f} °C\nRMSE {m['rmse']:.2f} °C\nR² {m['r2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        style_axes(ax, f"+{h} min", "actual temperature (°C)", "forecast (°C)" if j == 0 else "")
    fig.suptitle("Actual vs forecast rack temperature, test days", x=0.01, ha="left", fontsize=11, color=INK)
    paths.append((fig, out_dir / "actual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURFACE)
    x = np.arange(len(horizons))
    for k, (name, label, color) in enumerate((("model", forecaster.architecture.upper(), SERIES),
                                              ("persistence", "persistence", SERIES_2),
                                              ("linear_trend", "linear trend", SERIES_3))):
        vals = [results["test"][f"+{h}"][name]["mae"] for h in horizons]
        ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=color, label=label)
        for xi, v in zip(x + (k - 1) * 0.27, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [f"+{h} min" for h in horizons])
    style_axes(ax, "Mean absolute error by horizon, test days", "forecast horizon", "MAE (°C)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    paths.append((fig, out_dir / "mae_by_horizon.png"))

    # Example: the first test day, Rack 3 (the badly cooled rack), longest horizon, a 6-hour stretch.
    d, r, j = int(test.day.min()), min(2, history.frames.shape[2] - 1), len(horizons) - 1
    sel = (test.day == d) & (test.rack == r) & (test.minute >= 300) & (test.minute < 660)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    t_target = test.minute[sel] + horizons[j]
    ax.plot(t_target, actual[sel, j], color=INK_2, linewidth=1.5, label="actual")
    ax.plot(t_target, pred[sel, j], color=SERIES, linewidth=1.8, label=f"forecast made {horizons[j]} min earlier")
    ax.plot(t_target, test.last_temperature[sel], color=SERIES_2, linewidth=1.2, alpha=0.8,
            label=f"persistence (temperature {horizons[j]} min earlier)")
    style_axes(ax, f"Rack {r + 1}, test day {d} ({history.policies[d]}): +{horizons[j]} min forecast",
               "minute of the day being forecast", "temperature (°C)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    paths.append((fig, out_dir / "example_forecast.png"))

    out = []
    for fig, path in paths:
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        out.append(path)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = get_config()
    p = cfg.prediction
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.paths.models / "temperature_predictor")
    parser.add_argument("--history", type=Path, default=cfg.paths.data / "temperature_history.npz")
    parser.add_argument("--out", type=Path, default=cfg.paths.experiments / "temperature_prediction.json")
    args = parser.parse_args()

    configure_torch(p.torch_threads)
    forecaster = TemperatureForecaster.load(args.model_dir)
    history = History.load(args.history)
    train_days, val_days, test_days = split_days(len(history.policies), p.split)
    windows = lambda f, hist, days: make_windows(hist, days, p.sequence_length, p.horizons_minutes,  # noqa: E731
                                                 f.feature_names)
    test, val = windows(forecaster, history, test_days), windows(forecaster, history, val_days)
    cool = history.feature_names.index("cooling_level")

    # Cooling changed inside the longest horizon? (difference in received cooling between now and t + h_max)
    h_max = max(p.horizons_minutes)
    future = history.frames[test.day, test.minute + h_max, test.rack, cool]
    changed = np.abs(future - history.frames[test.day, test.minute, test.rack, cool]) > COOLING_CHANGE

    results = {
        "model": {k: forecaster.metadata[k] for k in ("architecture", "hidden_size", "num_layers", "dropout",
                                                      "sequence_length", "horizons_minutes", "feature_names",
                                                      "parameters", "target")},
        "training": forecaster.metadata["training"],
        "split": {"train_days": list(train_days), "val_days": list(val_days), "test_days": list(test_days),
                  "test_policies": [history.policies[d] for d in test_days], "test_windows": len(test)},
        "test": evaluate_windows(forecaster, test),
        "validation": evaluate_windows(forecaster, val),
        "test_by_rack": {f"rack{r + 1}": evaluate_windows(forecaster, subset(test, test.rack == r))
                         for r in range(history.frames.shape[2])},
        "test_by_policy": {history.policies[d]: evaluate_windows(forecaster, subset(test, test.day == d))
                           for d in test_days},
        "test_by_cooling_change": {
            "threshold": COOLING_CHANGE, "share_changed_pct": float(100 * changed.mean()),
            "cooling_changed": evaluate_windows(forecaster, subset(test, changed)),
            "cooling_steady": evaluate_windows(forecaster, subset(test, ~changed))},
        "warning_crossing_detection": crossing_detection(forecaster, test, cfg.thermal.zones.warning_c),
        "provider": provider_consistency(forecaster, history, test_days[0], cfg),
    }

    # Fresh days (new seeds), one per policy; DQN's behaviour was never in the training data.
    extra = generate_history(cfg, days=len(GENERALISATION_POLICIES), policies=GENERALISATION_POLICIES,
                             label="generalisation")
    results["generalisation_days"] = {
        pol: evaluate_windows(forecaster, windows(forecaster, extra, [i]))
        for i, pol in enumerate(GENERALISATION_POLICIES)}

    # Other architectures trained on the same data (comparison only).
    others = {}
    for d in sorted(cfg.paths.models.glob("temperature_predictor_*")):
        other = TemperatureForecaster.load(d)
        o_test, o_val = windows(other, history, test_days), windows(other, history, val_days)
        others[d.name.removeprefix("temperature_predictor_")] = {
            "architecture": other.architecture, "feature_names": other.feature_names,
            "test": evaluate_windows(other, o_test), "validation": evaluate_windows(other, o_val),
            "test_by_policy": {history.policies[x]: evaluate_windows(other, subset(o_test, o_test.day == x))
                               for x in test_days},
            "generalisation_days": {pol: evaluate_windows(other, windows(other, extra, [i]))
                                    for i, pol in enumerate(GENERALISATION_POLICIES)},
            "warning_crossing_detection": crossing_detection(other, o_test, cfg.thermal.zones.warning_c),
            "parameters": other.metadata["parameters"],
            "best_epoch": other.metadata["training"]["best_epoch"]}
    results["other_architectures"] = others

    with Path(forecaster.metadata["training"]["training_csv"]).open(encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    results["plots"] = [str(x) for x in make_plots(forecaster, test, results, rows, history,
                                                   cfg.paths.experiments / "plots" / "temperature_prediction")]
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{forecaster.architecture.upper()} ({forecaster.metadata['parameters']} parameters), best epoch "
          f"{forecaster.metadata['training']['best_epoch']}; test days {list(test_days)} "
          f"({results['split']['test_policies']}), {len(test)} windows")
    print(f"{'horizon':<8}{'MAE':>7}{'RMSE':>7}{'R²':>7} | {'persist MAE':>12}{'trend MAE':>11} | {'skill':>6}{'MAE cut':>9}")
    for h, r in results["test"].items():
        print(f"{h:<8}{r['model']['mae']:>7.3f}{r['model']['rmse']:>7.3f}{r['model']['r2']:>7.3f} | "
              f"{r['persistence']['mae']:>12.3f}{r['linear_trend']['mae']:>11.3f} | "
              f"{r['skill_vs_persistence']:>6.2f}{r['mae_reduction_vs_persistence_pct']:>8.1f}%")
    last = f"+{h_max}"
    c = results["test_by_cooling_change"]
    print(f"cooling changed within {h_max} min in {c['share_changed_pct']:.0f}% of windows: MAE {last} "
          f"{c['cooling_changed'][last]['model']['mae']:.2f} (changed) vs {c['cooling_steady'][last]['model']['mae']:.2f} (steady)")
    print("by rack", {k: round(v[last]["model"]["mae"], 2) for k, v in results["test_by_rack"].items()})
    print("by policy (test)", {k: (round(v[last]["model"]["mae"], 2), round(v[last]["persistence"]["mae"], 2))
                               for k, v in results["test_by_policy"].items()})
    print("fresh days", {k: (round(v[last]["model"]["mae"], 2), round(v[last]["persistence"]["mae"], 2))
                         for k, v in results["generalisation_days"].items()})
    print("warning crossing", last, results["warning_crossing_detection"][last])
    for k, v in others.items():
        print(f"other [{k}] MAE", {h: round(v["test"][h]["model"]["mae"], 3) for h in v["test"]},
              "skill", {h: round(v["test"][h]["skill_vs_persistence"], 2) for h in v["test"]},
              "crossing recall/precision", last, round(v["warning_crossing_detection"][last]["model"]["recall"], 3),
              v["warning_crossing_detection"][last]["model"]["precision"])
    print("provider", results["provider"])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
