"""Temperature-forecasting data (Step 12): per-rack features, simulated history, and sliding windows.

``rack_features(dc)`` is the single definition of the model's inputs. The offline dataset and the live
``ForecastProvider`` both call it, so training and serving see identical features.

History: ``dataset_episodes`` simulated 24-hour days. Day i is driven by
``behaviour_policies[i % len(behaviour_policies)]`` with environment seed
``derive_seed(seed, "prediction", "day", i)``. The features of every rack are recorded once per minute
(1441 frames a day, including minute 0).

Windows: for a rack at minute t, the input is the last ``sequence_length`` frames (t-L+1 .. t) and the target is
the temperature *change* to t+h for each horizon h. A window never crosses a day boundary.

    python -m backend.prediction.dataset        # -> data/temperature_history.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import Config, get_config
from backend.rl.baselines import Controller, make_baseline
from backend.rl.environment import DataCenterEnv
from backend.simulation.datacentre import DataCentre
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed

logger = get_logger("prediction.dataset")

ENDING_SOON_MINUTES = 15

# name -> value for one rack of the simulator. All are read from the current state only.
FEATURE_READERS = {
    "temperature": lambda dc, rack: rack.temperature_c,                          # °C
    "cpu_utilisation": lambda dc, rack: rack.get_cpu_utilisation(),              # 0-1
    "gpu_utilisation": lambda dc, rack: rack.get_gpu_utilisation(),              # 0-1
    "it_power": lambda dc, rack: rack.get_power_consumption().it_kw,             # kW
    "cooling_level": lambda dc, rack: rack.effective_cooling,                    # delivered level x effectiveness, 0-1
    "queue_length": lambda dc, rack: float(len(dc.queue)),                       # jobs waiting (same for all racks)
    # Extra signals for the feature ablation (not in the configured model). Each is known to an operator now:
    "cooling_target": lambda dc, rack: dc.cooling.target_level * rack.cooling_effectiveness,   # commanded, 0-1
    "load_ending_soon": lambda dc, rack: _thermal_load_ending(dc, rack, ENDING_SOON_MINUTES),  # 0-1
    "ambient_temperature": lambda dc, rack: dc.ambient.temperature_at(dc.hour_of_day),         # °C
}
EXTRA_FEATURES = ["cooling_target", "load_ending_soon", "ambient_temperature"]


def _thermal_load_ending(dc: DataCentre, rack, within_minutes: int) -> float:
    """Thermal load (0-1, thermal-model weights) of this rack's jobs that will finish within ``within_minutes``."""
    return float(sum(dc.thermal.w_cpu * w.cpu_requirement / rack.cpu_capacity
                     + dc.thermal.w_gpu * w.gpu_requirement / rack.gpu_capacity
                     for w in rack.workloads if w.remaining_time <= within_minutes))


def rack_features(dc: DataCentre, names: list[str]) -> np.ndarray:
    """(num_racks, n_features) float32 snapshot of the simulator's current state."""
    unknown = [n for n in names if n not in FEATURE_READERS]
    if unknown:
        raise ValueError(f"unknown prediction features {unknown}; supported: {sorted(FEATURE_READERS)}")
    return np.array([[FEATURE_READERS[n](dc, rack) for n in names] for rack in dc.racks], dtype=np.float32)


def make_behaviour_controller(name: str, config: Config) -> Controller:
    """A controller for the feature environment. Learned policies are loaded from models/ (never trained here)."""
    if name == "ppo":
        from backend.rl.ppo import PPOAgent, PPOController
        return PPOController(PPOAgent.load(config.paths.models / "ppo"), config, deterministic=False)
    if name == "dqn":
        from backend.rl.dqn import DQNAgent, DQNController
        return DQNController(DQNAgent.load(config.paths.models / "dqn"), config)
    return make_baseline(name, config)


def record_day(env: DataCenterEnv, controller: Controller, seed: int, names: list[str]) -> np.ndarray:
    """Run one episode and return (steps + 1, num_racks, n_features): the state at every minute boundary."""
    obs, _ = env.reset(seed=seed)
    controller.reset(seed)
    frames = [rack_features(env.dc, names)]
    truncated = False
    while not truncated:
        obs, _, _, truncated, _ = env.step(controller.act(obs, env.action_masks()))
        frames.append(rack_features(env.dc, names))
    return np.stack(frames)


@dataclass
class History:
    frames: np.ndarray          # (days, minutes + 1, racks, features) float32
    policies: list[str]         # behaviour policy of each day
    seeds: list[int]
    feature_names: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=self.frames, policies=np.array(self.policies),
                            seeds=np.array(self.seeds, dtype=np.int64), feature_names=np.array(self.feature_names))

    @classmethod
    def load(cls, path: Path) -> "History":
        with np.load(path) as d:
            return cls(d["frames"], [str(p) for p in d["policies"]], [int(s) for s in d["seeds"]],
                       [str(n) for n in d["feature_names"]])


def generate_history(config: Config, days: int | None = None, policies: list[str] | None = None,
                     minutes: int | None = None, label: str = "day") -> History:
    """Simulate ``days`` days. ``label`` namespaces the seeds, so extra evaluation days never repeat training days.

    Every supported feature is recorded (the configured ones first); models pick their columns by name.
    """
    p = config.prediction
    names = list(p.features) + [n for n in FEATURE_READERS if n not in p.features]
    days = days or p.dataset_episodes
    policies = policies or list(p.behaviour_policies)
    env = DataCenterEnv(representation="features", episode_minutes=minutes or config.simulation.episode_minutes,
                        config=config)
    controllers = {name: make_behaviour_controller(name, config) for name in set(policies)}
    frames, used, seeds = [], [], []
    for i in range(days):
        name = policies[i % len(policies)]
        seed = derive_seed(p.seed, "prediction", label, i)
        frames.append(record_day(env, controllers[name], seed, names))
        used.append(name)
        seeds.append(seed)
        logger.info("%s %d/%d (%s): hottest rack %.1f °C max", label, i + 1, days, name,
                    float(frames[-1][..., names.index("temperature")].max()))
    return History(np.stack(frames), used, seeds, names)


def split_days(n_days: int, split: list[float]) -> tuple[range, range, range]:
    """Chronological split by whole days: the first days train, the next validate, the last test."""
    n_train = int(round(split[0] * n_days))
    n_val = int(round(split[1] * n_days))
    if not (0 < n_train and 0 < n_val and n_train + n_val < n_days):
        raise ValueError(f"cannot split {n_days} days as {split}")
    return range(0, n_train), range(n_train, n_train + n_val), range(n_train + n_val, n_days)


@dataclass
class Windows:
    x: np.ndarray            # (N, L, F) raw features
    y: np.ndarray            # (N, H) temperature change in °C from the last input frame to each horizon
    last_temperature: np.ndarray   # (N,) °C at the last input frame
    day: np.ndarray          # (N,) index into the History
    rack: np.ndarray         # (N,)
    minute: np.ndarray       # (N,) minute of the last input frame

    def __len__(self) -> int:
        return len(self.x)

    @property
    def target_temperature(self) -> np.ndarray:
        return self.last_temperature[:, None] + self.y


def make_windows(history: History, days, sequence_length: int, horizons: list[int],
                 features: list[str] | None = None) -> Windows:
    """Windows over the named feature columns (default: all recorded), in that order."""
    features = list(features) if features is not None else list(history.feature_names)
    columns = [history.feature_names.index(n) for n in features]
    temp_idx = features.index("temperature")
    n_frames, n_racks = history.frames.shape[1], history.frames.shape[2]
    L, h_max = sequence_length, max(horizons)
    ts = np.arange(L - 1, n_frames - h_max)                 # minute of the last input frame
    xs, ys, last, day_idx, rack_idx, minute_idx = [], [], [], [], [], []
    for d in days:
        for r in range(n_racks):
            series = history.frames[d, :, r, :][:, columns]   # (frames, F)
            x = np.stack([series[t - L + 1:t + 1] for t in ts])
            now = series[ts, temp_idx]
            y = np.stack([series[ts + h, temp_idx] - now for h in horizons], axis=1)
            xs.append(x), ys.append(y), last.append(now)
            day_idx.append(np.full(len(ts), d)), rack_idx.append(np.full(len(ts), r)), minute_idx.append(ts)
    return Windows(np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
                   np.concatenate(last).astype(np.float32), np.concatenate(day_idx), np.concatenate(rack_idx),
                   np.concatenate(minute_idx))


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=cfg.paths.data / "temperature_history.npz")
    args = parser.parse_args()
    history = generate_history(cfg)
    history.save(args.out)
    print(f"{history.frames.shape[0]} days x {history.frames.shape[1]} frames x {history.frames.shape[2]} racks x "
          f"{history.frames.shape[3]} features -> {args.out}")


if __name__ == "__main__":
    main()
