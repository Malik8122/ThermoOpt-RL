"""Energy, carbon and cost accounting.

For a step of dt minutes with constant power P (kW):

    E  [kWh]  = P * dt / 60
    CO2 [g]   = E * carbon_intensity [gCO2/kWh]
    cost [INR]= E * price [INR/kWh]

IT and cooling energy are tracked separately. Carbon and cost are computed on
total facility energy (IT + cooling), because the grid supplies both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backend.simulation.enums import SimulationInvariantError
from backend.utils.units import kw_to_kwh


@dataclass(frozen=True)
class EnergyStep:
    it_kwh: float
    cooling_kwh: float
    carbon_g: float
    cost_inr: float

    @property
    def total_kwh(self) -> float:
        return self.it_kwh + self.cooling_kwh


@dataclass
class EnergyTotals:
    it_kwh: float = 0.0
    cooling_kwh: float = 0.0
    carbon_g: float = 0.0
    cost_inr: float = 0.0

    @property
    def total_kwh(self) -> float:
        return self.it_kwh + self.cooling_kwh

    def to_dict(self) -> dict:
        return {**asdict(self), "total_kwh": self.total_kwh}


class EnergyAccountant:
    def __init__(self) -> None:
        self.totals = EnergyTotals()

    def record(
        self,
        it_power_kw: float,
        cooling_power_kw: float,
        price_inr_per_kwh: float,
        carbon_g_per_kwh: float,
        minutes: float,
    ) -> EnergyStep:
        if min(it_power_kw, cooling_power_kw, price_inr_per_kwh, carbon_g_per_kwh, minutes) < 0:
            raise SimulationInvariantError("power, price, carbon intensity and duration must be non-negative")
        it_kwh = kw_to_kwh(it_power_kw, minutes)
        cooling_kwh = kw_to_kwh(cooling_power_kw, minutes)
        total_kwh = it_kwh + cooling_kwh
        step = EnergyStep(
            it_kwh=it_kwh,
            cooling_kwh=cooling_kwh,
            carbon_g=total_kwh * carbon_g_per_kwh,
            cost_inr=total_kwh * price_inr_per_kwh,
        )
        self.totals.it_kwh += step.it_kwh
        self.totals.cooling_kwh += step.cooling_kwh
        self.totals.carbon_g += step.carbon_g
        self.totals.cost_inr += step.cost_inr
        return step
