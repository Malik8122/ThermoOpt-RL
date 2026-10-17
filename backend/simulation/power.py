"""Utilisation-dependent IT power model.

    P(u) = P_idle + (P_max - P_idle) * u ** k        u in [0, 1], P in W

k = 1 gives the linear model from the brief. Device power is in W; everything
aggregated to rack or facility level is converted to kW via backend.utils.units.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import PowerConfig
from backend.simulation.enums import SimulationInvariantError

_UTILISATION_TOLERANCE = 1e-9


def _validated_utilisation(u: float) -> float:
    if not -_UTILISATION_TOLERANCE <= u <= 1 + _UTILISATION_TOLERANCE:
        raise SimulationInvariantError(f"utilisation must be within [0, 1], got {u}")
    return min(max(u, 0.0), 1.0)


@dataclass(frozen=True)
class PowerModel:
    server_idle_w: float
    server_max_w: float
    gpu_idle_w: float
    gpu_max_w: float
    utilisation_exponent: float = 1.0

    @classmethod
    def from_config(cls, cfg: PowerConfig) -> "PowerModel":
        return cls(cfg.server_idle_w, cfg.server_max_w, cfg.gpu_idle_w, cfg.gpu_max_w, cfg.utilisation_exponent)

    def _device_power_w(self, idle_w: float, max_w: float, utilisation: float) -> float:
        u = _validated_utilisation(utilisation)
        return idle_w + (max_w - idle_w) * u ** self.utilisation_exponent

    def server_power_w(self, cpu_utilisation: float) -> float:
        """Power draw of one server at the given CPU utilisation (0-1)."""
        return self._device_power_w(self.server_idle_w, self.server_max_w, cpu_utilisation)

    def gpu_power_w(self, gpu_utilisation: float) -> float:
        """Power draw of one GPU node at the given GPU utilisation (0-1)."""
        return self._device_power_w(self.gpu_idle_w, self.gpu_max_w, gpu_utilisation)


@dataclass(frozen=True)
class RackPower:
    """IT power of one rack, split by device class. All values in kW."""

    server_kw: float
    gpu_kw: float

    @property
    def it_kw(self) -> float:
        return self.server_kw + self.gpu_kw
