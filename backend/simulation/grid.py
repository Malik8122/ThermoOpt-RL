"""Electricity price and grid carbon intensity over a simulated day.

The profile is a SYNTHETIC assumption (see config.yaml -> grid), not real grid data.
"""

from __future__ import annotations

from backend.config import GridConfig
from backend.simulation.profiles import PeriodicProfile


class GridSignals:
    def __init__(self, cfg: GridConfig) -> None:
        hours = [p[0] for p in cfg.profile]
        self._price = PeriodicProfile(hours, [p[1] for p in cfg.profile], cfg.interpolation)
        self._carbon = PeriodicProfile(hours, [p[2] for p in cfg.profile], cfg.interpolation)
        self.high_carbon_threshold = cfg.high_carbon_threshold

    def price_at(self, hour: float) -> float:
        """Electricity price in INR per kWh at clock time ``hour`` (0-24)."""
        return self._price(hour)

    def carbon_at(self, hour: float) -> float:
        """Grid carbon intensity in gCO2 per kWh at clock time ``hour``."""
        return self._carbon(hour)

    def is_high_carbon(self, hour: float) -> bool:
        return self.carbon_at(hour) >= self.high_carbon_threshold
