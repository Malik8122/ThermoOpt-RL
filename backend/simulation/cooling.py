"""Shared cooling plant: set-point, delayed response, and electrical power.

Response model (linear ramp). When the target changes at time t0 while the
delivered level is C0, the delivered level follows

    C(t) = C0 + (target - C0) * min(1, (t - t0) / response_time)

so it reaches any new target exactly ``response_time_minutes`` later. If the
target changes again mid-transition, the ramp restarts from the level
delivered at that moment. A response time of 0 means instantaneous.

A linear ramp was chosen over a first-order lag because it reaches the target
in a defined time and exposes an interpretable ``progress`` value (0-1) for the
dashboard. The delay is what gives a predictive controller its advantage: a
reactive controller that raises cooling only once a rack is hot gets the full
effect several minutes too late.

Electrical power depends on the delivered level only:

    P_cooling = P_max * C ** n         (n = 1: linear, as in the brief)

Per-rack cooling effectiveness (Rack 3 = 0.75) scales the thermal *effect* in
the thermal model; it never changes the electrical power drawn here.
"""

from __future__ import annotations

from backend.config import CoolingConfig
from backend.simulation.enums import SimulationInvariantError
from backend.utils.units import pct_to_fraction

_LEVEL_TOLERANCE = 1e-9


class CoolingSystem:
    def __init__(
        self,
        levels: list[float],
        initial_level: float,
        max_power_kw: float,
        power_exponent: float,
        response_time_minutes: float,
    ) -> None:
        self.levels = sorted(levels)
        self.max_power_kw = max_power_kw
        self.power_exponent = power_exponent
        self.response_time_minutes = response_time_minutes
        self._check_level(initial_level)
        self.current_level = initial_level   # delivered cooling, fraction 0-1
        self.target_level = initial_level    # requested set-point, fraction 0-1
        self._ramp_start = initial_level
        self._elapsed = 0.0

    @classmethod
    def from_config(cls, cfg: CoolingConfig) -> "CoolingSystem":
        return cls(
            levels=[pct_to_fraction(p) for p in cfg.levels_pct],
            initial_level=pct_to_fraction(cfg.initial_level_pct),
            max_power_kw=cfg.max_cooling_power_kw,
            power_exponent=cfg.power_exponent,
            response_time_minutes=cfg.response_time_minutes,
        )

    # ------------------------------------------------------------- set-point
    def set_target(self, level: float) -> None:
        """Request a new cooling level (fraction 0-1). Takes effect gradually."""
        self._check_level(level)
        if abs(level - self.target_level) < _LEVEL_TOLERANCE:
            return
        self.target_level = level
        self._ramp_start = self.current_level
        self._elapsed = 0.0
        if self.response_time_minutes == 0:
            self.current_level = level

    def level_above(self) -> float | None:
        """Next discrete level above the current target, or None at the top. Read-only."""
        above = [lv for lv in self.levels if lv > self.target_level + _LEVEL_TOLERANCE]
        return above[0] if above else None

    def level_below(self) -> float | None:
        """Next discrete level below the current target, or None at the bottom. Read-only."""
        below = [lv for lv in self.levels if lv < self.target_level - _LEVEL_TOLERANCE]
        return below[-1] if below else None

    def increase(self) -> float:
        """Move the target to the next discrete level above it (no-op at the top)."""
        level = self.level_above()
        if level is not None:
            self.set_target(level)
        return self.target_level

    def decrease(self) -> float:
        """Move the target to the next discrete level below it (no-op at the bottom)."""
        level = self.level_below()
        if level is not None:
            self.set_target(level)
        return self.target_level

    # --------------------------------------------------------------- dynamics
    @property
    def progress(self) -> float:
        """Fraction (0-1) of the current transition that has been completed."""
        if self.response_time_minutes == 0 or abs(self.target_level - self._ramp_start) < _LEVEL_TOLERANCE:
            return 1.0
        return min(1.0, self._elapsed / self.response_time_minutes)

    @property
    def in_transition(self) -> bool:
        return self.progress < 1.0

    def advance(self, minutes: float) -> float:
        """Advance the response by ``minutes``; returns the delivered level."""
        self._elapsed += minutes
        self.current_level = self._ramp_start + (self.target_level - self._ramp_start) * self.progress
        self._check_level(self.current_level)
        return self.current_level

    def power_kw(self, level: float | None = None) -> float:
        """Electrical power of the plant at ``level`` (default: delivered level)."""
        level = self.current_level if level is None else level
        self._check_level(level)
        return self.max_power_kw * level ** self.power_exponent

    @staticmethod
    def _check_level(level: float) -> None:
        if not -_LEVEL_TOLERANCE <= level <= 1 + _LEVEL_TOLERANCE:
            raise SimulationInvariantError(f"cooling level must be within [0, 1], got {level}")
