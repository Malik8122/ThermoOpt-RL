"""Tabular state abstraction for Policy Iteration, Value Iteration and Q-Learning.

Five factors, each discretised with the edges in config.yaml -> discretisation
(a value v falls in bin i where edge[i-1] <= v < edge[i]):

    factor        source in the simulator                                  default levels
    temperature   hottest rack temperature (°C)                            LOW <25 | MEDIUM 25-27 | HIGH 27-30 | CRITICAL >=30
    workload      mean utilisation of all CPU+GPU capacity (%)             LOW <40 | MEDIUM 40-70 | HIGH >=70
    cooling       cooling set-point / target (%)                           LOW <40 | MEDIUM 40-70 | HIGH >=70
    carbon        grid carbon intensity (gCO2/kWh)                         LOW <400 | HIGH >=400
    queue         number of waiting jobs                                   EMPTY 0 | PENDING >=1

Mixed-radix encoding in that order (C order):

    s = (((temperature * 3 + workload) * 3 + cooling) * 2 + carbon) * 2 + queue      in [0, 144)
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

from backend.config import DiscretisationConfig, get_config
from backend.simulation.datacentre import DataCentre
from backend.utils.units import fraction_to_pct

LEVEL_NAMES: dict[str, tuple[str, ...]] = {
    "temperature": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "workload": ("LOW", "MEDIUM", "HIGH"),
    "cooling": ("LOW", "MEDIUM", "HIGH"),
    "carbon": ("LOW", "HIGH"),
    "queue": ("EMPTY", "PENDING"),
}


@dataclass(frozen=True)
class TabularState:
    temperature: int
    workload: int
    cooling: int
    carbon: int
    queue: int


class TabularStateEncoder:
    FACTORS = ("temperature", "workload", "cooling", "carbon", "queue")

    def __init__(self, cfg: DiscretisationConfig) -> None:
        self.edges = {
            "temperature": np.asarray(cfg.temperature_edges_c, dtype=float),
            "workload": np.asarray(cfg.workload_edges_pct, dtype=float),
            "cooling": np.asarray(cfg.cooling_edges_pct, dtype=float),
            "carbon": np.asarray(cfg.carbon_edges, dtype=float),
            "queue": np.asarray(cfg.queue_edges, dtype=float),
        }
        self.shape = tuple(len(self.edges[f]) + 1 for f in self.FACTORS)
        self.n_states = int(np.prod(self.shape))

    # ---------------------------------------------------------- discretisation
    def _bin(self, factor: str, value: float) -> int:
        return int(np.digitize(value, self.edges[factor], right=False))

    def discretise(self, hottest_temperature_c: float, mean_utilisation_pct: float, cooling_setpoint_pct: float,
                   carbon_g_per_kwh: float, queue_length: int) -> TabularState:
        return TabularState(
            temperature=self._bin("temperature", hottest_temperature_c),
            workload=self._bin("workload", mean_utilisation_pct),
            cooling=self._bin("cooling", cooling_setpoint_pct),
            carbon=self._bin("carbon", carbon_g_per_kwh),
            queue=self._bin("queue", queue_length),
        )

    def observe(self, dc: DataCentre) -> TabularState:
        """Abstract the simulator's current state."""
        used = sum((r.cpu_capacity - r.get_available_cpu()) + (r.gpu_capacity - r.get_available_gpu())
                   for r in dc.racks)
        capacity = sum(r.cpu_capacity + r.gpu_capacity for r in dc.racks)
        return self.discretise(
            hottest_temperature_c=float(dc.temperatures().max()),
            mean_utilisation_pct=fraction_to_pct(used / capacity),
            cooling_setpoint_pct=fraction_to_pct(dc.cooling.target_level),
            carbon_g_per_kwh=dc.grid.carbon_at(dc.hour_of_day),
            queue_length=len(dc.queue),
        )

    # ---------------------------------------------------------------- encoding
    def encode(self, state: TabularState) -> int:
        levels = astuple(state)
        for factor, level, size in zip(self.FACTORS, levels, self.shape):
            if not 0 <= level < size:
                raise ValueError(f"{factor} level {level} outside [0, {size})")
        return int(np.ravel_multi_index(levels, self.shape))

    def decode(self, index: int) -> TabularState:
        if not 0 <= index < self.n_states:
            raise ValueError(f"state index must be in [0, {self.n_states}), got {index}")
        return TabularState(*(int(v) for v in np.unravel_index(index, self.shape)))

    def describe(self, state: TabularState) -> dict[str, str]:
        out = {}
        for factor, level, size in zip(self.FACTORS, astuple(state), self.shape):
            names = LEVEL_NAMES[factor]
            out[factor] = names[level] if len(names) == size else f"L{level}"
        return out


def _default_encoder() -> TabularStateEncoder:
    return TabularStateEncoder(get_config().discretisation)


def encode_tabular_state(state: TabularState, encoder: TabularStateEncoder | None = None) -> int:
    return (encoder or _default_encoder()).encode(state)


def decode_tabular_state(index: int, encoder: TabularStateEncoder | None = None) -> TabularState:
    return (encoder or _default_encoder()).decode(index)
