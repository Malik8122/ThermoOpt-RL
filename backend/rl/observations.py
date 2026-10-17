"""Normalised feature observation for DQN / PPO (PHASE1_DESIGN §4.3).

30 base features, every one read from the simulator's current state:

    0-3    rack temperature              (T - T_min) / (T_max - T_min)          [0, 1]
    4-7    rack CPU utilisation          fraction                               [0, 1]
    8-11   rack GPU utilisation          fraction                               [0, 1]
    12     cooling set-point (target)    fraction                               [0, 1]
    13     delivered cooling             fraction                               [0, 1]
    14     queue length                  min(1, len / queue_normalisation)      [0, 1]
    15     head job CPU requirement      / rack CPU capacity                    [0, 1]
    16     head job GPU requirement      / rack GPU capacity                    [0, 1]
    17     head job urgency              clip(1 - slack_left / max_flexible_slack, 0, 1)
    18     head job is flexible          0/1
    19     a job is waiting              0/1
    20     electricity price             / max price in profile                 [0, 1]
    21     carbon intensity              / max carbon in profile                [0, 1]
    22     carbon trend                  (carbon(t+60 min) - carbon(t)) / max carbon    [-1, 1]
    23-24  time of day                   sin, cos of 2*pi*hour/24               [-1, 1]
    25     ambient temperature           (T - T_min) / (T_max - T_min)          [0, 1]
    26-29  head job fits on rack r       0/1

Features 15-18 and 26-29 are 0 when no job is waiting.

Predictive extension (Step 12): when a ``PredictionProvider`` is supplied, 12
features are appended: the forecast temperature of each rack at each horizon
in config.prediction.horizons_minutes (+5, +10, +15 min), rack-major order,
clipped to the simulator's temperature bounds and normalised like feature 0.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from backend.config import Config
from backend.simulation.datacentre import DataCentre
from backend.utils.units import HOURS_PER_DAY, MINUTES_PER_HOUR

# (num_racks, n_horizons) array of forecast temperatures in °C for the current state.
PredictionProvider = Callable[[DataCentre], np.ndarray]


class ObservationBuilder:
    def __init__(self, config: Config, prediction_provider: PredictionProvider | None = None) -> None:
        self.config = config
        self.num_racks = config.datacenter.num_racks
        self.t_min = config.thermal.min_temperature_c
        self.t_span = config.thermal.max_temperature_c - self.t_min
        self.max_price = config.grid.max_price
        self.max_carbon = config.grid.max_carbon
        self.queue_norm = config.environment.queue_normalisation
        self.lookahead_h = config.environment.carbon_lookahead_minutes / MINUTES_PER_HOUR
        self.urgency_horizon = config.workloads.flexible_slack_minutes[1]
        self.horizons = list(config.prediction.horizons_minutes)
        self.prediction_provider = prediction_provider
        self.feature_names = self._names()
        self.low, self.high = self._bounds()

    @property
    def include_predictions(self) -> bool:
        return self.prediction_provider is not None

    @property
    def size(self) -> int:
        return len(self.feature_names)

    @property
    def prediction_size(self) -> int:
        return self.num_racks * len(self.horizons) if self.include_predictions else 0

    @property
    def base_size(self) -> int:
        return self.size - self.prediction_size

    def _names(self) -> list[str]:
        r = range(1, self.num_racks + 1)
        names = [f"rack{i}_temperature" for i in r] + [f"rack{i}_cpu" for i in r] + [f"rack{i}_gpu" for i in r]
        names += ["cooling_setpoint", "cooling_delivered", "queue_length",
                  "head_cpu", "head_gpu", "head_urgency", "head_flexible", "job_waiting",
                  "electricity_price", "carbon_intensity", "carbon_trend",
                  "time_sin", "time_cos", "ambient_temperature"]
        names += [f"head_fits_rack{i}" for i in r]
        if self.include_predictions:
            names += [f"rack{i}_forecast_t+{h}" for i in r for h in self.horizons]
        return names

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.zeros(self.size, dtype=np.float32)
        high = np.ones(self.size, dtype=np.float32)
        for name in ("carbon_trend", "time_sin", "time_cos"):
            low[self.feature_names.index(name)] = -1.0
        return low, high

    def _norm_temperature(self, t: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(t, dtype=float) - self.t_min) / self.t_span

    def build(self, dc: DataCentre) -> np.ndarray:
        hour = dc.hour_of_day
        head = dc.queue[0] if dc.queue else None
        rack0 = dc.racks[0]
        if head is not None:
            head_features = [
                head.cpu_requirement / rack0.cpu_capacity,
                head.gpu_requirement / rack0.gpu_capacity,
                min(1.0, max(0.0, 1.0 - head.slack_at(dc.minute) / self.urgency_horizon)),
                float(head.flexible),
                1.0,
            ]
            fits = [float(r.can_accept_workload(head).success) for r in dc.racks]
        else:
            head_features = [0.0] * 5
            fits = [0.0] * self.num_racks
        carbon_now = dc.grid.carbon_at(hour)
        angle = 2.0 * math.pi * hour / HOURS_PER_DAY
        parts = [
            self._norm_temperature(dc.temperatures()),
            dc.cpu_utilisation(),
            dc.gpu_utilisation(),
            [dc.cooling.target_level, dc.cooling.current_level, min(1.0, len(dc.queue) / self.queue_norm)],
            head_features,
            [dc.grid.price_at(hour) / self.max_price,
             carbon_now / self.max_carbon,
             (dc.grid.carbon_at(hour + self.lookahead_h) - carbon_now) / self.max_carbon,
             math.sin(angle), math.cos(angle),
             self._norm_temperature(dc.ambient.temperature_at(hour))],
            fits,
        ]
        if self.prediction_provider is not None:
            forecast = np.asarray(self.prediction_provider(dc), dtype=float)
            expected = (self.num_racks, len(self.horizons))
            if forecast.shape != expected:
                raise ValueError(f"prediction provider returned shape {forecast.shape}, expected {expected}")
            clipped = np.clip(forecast, self.t_min, self.t_min + self.t_span)
            parts.append(self._norm_temperature(clipped).reshape(-1))
        obs = np.concatenate([np.atleast_1d(np.asarray(p, dtype=float)) for p in parts]).astype(np.float32)
        if obs.shape != (self.size,):
            raise RuntimeError(f"observation has shape {obs.shape}, expected ({self.size},)")
        return obs

    def as_dict(self, observation: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.feature_names, observation)}
