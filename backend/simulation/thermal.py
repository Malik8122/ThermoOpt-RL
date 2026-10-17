"""Rack thermal model, ambient model and temperature zones.

Per rack r, explicit Euler with timestep dt (minutes):

    dT_r/dt = a * L_r  -  b * eta_r * C  +  kappa * (T_amb(h) - T_r)

    T_r(t + dt) = clip( T_r + dt * dT_r/dt + sigma * sqrt(dt) * N(0, 1),  T_min, T_max )

    L_r   thermal load = w_cpu * cpu_r + w_gpu * gpu_r            fraction 0-1
    C     delivered shared cooling level                          fraction 0-1
    eta_r fraction of the shared cooling that reaches rack r      fraction 0-1
    a     = alpha * 100   °C/min for a fully loaded rack          (alpha = 0.035 °C/min per % point)
    b     = beta  * 100   °C/min at 100 % delivered cooling       (beta  = 0.045 °C/min per % point)
    kappa ambient exchange rate, 1/min (Newton's law of cooling)

The ambient term is proportional to the temperature *difference*, so it pulls a
rack towards room ambient: it heats a rack cooler than ambient and cools a rack
warmer than it, and vanishes at T = T_amb.

Steady state (dT/dt = 0):   T* = T_amb + (a * L - b * eta * C) / kappa
"""

from __future__ import annotations

import math

import numpy as np

from backend.config import ThermalConfig, ThermalZones
from backend.simulation.enums import TemperatureStatus
from backend.utils.units import HOURS_PER_DAY, PERCENT


def get_temperature_status(temperature_c: float, zones: ThermalZones) -> TemperatureStatus:
    """NORMAL below warning, WARNING in [warning, critical), CRITICAL at or above critical."""
    if temperature_c >= zones.critical_c:
        return TemperatureStatus.CRITICAL
    if temperature_c >= zones.warning_c:
        return TemperatureStatus.WARNING
    return TemperatureStatus.NORMAL


class AmbientModel:
    """Deterministic daily room-ambient profile: T_amb(h) = mean + A * cos(2*pi*(h - peak)/24)."""

    def __init__(self, mean_c: float, amplitude_c: float, peak_hour: float) -> None:
        self.mean_c = mean_c
        self.amplitude_c = amplitude_c
        self.peak_hour = peak_hour

    @classmethod
    def from_config(cls, cfg: ThermalConfig) -> "AmbientModel":
        return cls(cfg.ambient_temperature_c, cfg.ambient_daily_amplitude_c, cfg.ambient_peak_hour)

    def temperature_at(self, hour: float) -> float:
        phase = 2.0 * math.pi * (hour - self.peak_hour) / HOURS_PER_DAY
        return self.mean_c + self.amplitude_c * math.cos(phase)


class ThermalModel:
    def __init__(self, cfg: ThermalConfig, timestep_minutes: float) -> None:
        # Convert per-percentage-point coefficients to per-fraction, once.
        self.heating_rate = cfg.alpha * PERCENT      # °C/min at thermal load 1.0
        self.cooling_rate = cfg.beta * PERCENT       # °C/min at delivered cooling 1.0, eta 1.0
        self.kappa = cfg.ambient_coupling
        self.w_cpu = cfg.utilisation_weights.cpu
        self.w_gpu = cfg.utilisation_weights.gpu
        self.noise_std = cfg.noise_std_c
        self.t_min = cfg.min_temperature_c
        self.t_max = cfg.max_temperature_c
        self.zones = cfg.zones
        self.dt = timestep_minutes

    def thermal_load(self, cpu_utilisation: np.ndarray, gpu_utilisation: np.ndarray) -> np.ndarray:
        """Weighted CPU/GPU utilisation that drives heating, fraction 0-1."""
        return self.w_cpu * np.asarray(cpu_utilisation) + self.w_gpu * np.asarray(gpu_utilisation)

    def derivative(
        self, temperatures: np.ndarray, load: np.ndarray, effective_cooling: np.ndarray, ambient_c: float
    ) -> np.ndarray:
        """dT/dt in °C per minute for each rack."""
        heating = self.heating_rate * np.asarray(load)
        cooling = self.cooling_rate * np.asarray(effective_cooling)
        ambient = self.kappa * (ambient_c - np.asarray(temperatures))
        return heating - cooling + ambient

    def step(
        self,
        temperatures: np.ndarray,
        load: np.ndarray,
        effective_cooling: np.ndarray,
        ambient_c: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Advance all rack temperatures by one timestep. ``rng=None`` disables noise."""
        temps = np.asarray(temperatures, dtype=float)
        new = temps + self.dt * self.derivative(temps, load, effective_cooling, ambient_c)
        if rng is not None and self.noise_std > 0:
            new = new + self.noise_std * math.sqrt(self.dt) * rng.standard_normal(temps.shape)
        return np.clip(new, self.t_min, self.t_max)

    def equilibrium(self, load: np.ndarray, effective_cooling: np.ndarray, ambient_c: float) -> np.ndarray:
        """Noise-free steady-state temperature (before clipping)."""
        if self.kappa == 0:
            raise ValueError("no finite equilibrium without ambient coupling")
        balance = self.heating_rate * np.asarray(load) - self.cooling_rate * np.asarray(effective_cooling)
        return ambient_c + balance / self.kappa

    def status(self, temperature_c: float) -> TemperatureStatus:
        return get_temperature_status(temperature_c, self.zones)
