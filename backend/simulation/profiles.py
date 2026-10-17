"""Smooth 24-hour periodic profiles built from (hour, value) anchor points.

Used for the grid price, grid carbon intensity and workload-arrival intensity,
so the interpolation logic exists once.

``pchip`` (piecewise cubic Hermite) is smooth and shape-preserving: it passes
through every anchor and never overshoots between them, so a price can never
dip below the cheapest anchor. Periodicity is enforced by replicating the
anchors one day before and after, which makes 23:59 flow smoothly into 00:00.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from backend.utils.units import HOURS_PER_DAY, MINUTES_PER_HOUR

Interpolation = Literal["pchip", "linear"]


class PeriodicProfile:
    def __init__(self, hours: Sequence[float], values: Sequence[float], method: Interpolation = "pchip") -> None:
        hours_arr = np.asarray(hours, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        if hours_arr.shape != values_arr.shape or hours_arr.size < 2:
            raise ValueError("a profile needs at least two (hour, value) anchors")
        if np.any(np.diff(hours_arr) <= 0) or hours_arr[0] < 0 or hours_arr[-1] >= HOURS_PER_DAY:
            raise ValueError("profile hours must be strictly ascending within [0, 24)")
        self.hours = hours_arr
        self.values = values_arr
        self.method = method
        if method == "pchip":
            ext_hours = np.concatenate([hours_arr - HOURS_PER_DAY, hours_arr, hours_arr + HOURS_PER_DAY])
            self._pchip = PchipInterpolator(ext_hours, np.tile(values_arr, 3))
        elif method != "linear":
            raise ValueError(f"unknown interpolation method {method!r}")

    def __call__(self, hour: float | np.ndarray) -> float | np.ndarray:
        h = np.mod(np.asarray(hour, dtype=float), HOURS_PER_DAY)
        if self.method == "pchip":
            out = self._pchip(h)
        else:
            out = np.interp(h, self.hours, self.values, period=HOURS_PER_DAY)
        return float(out) if np.ndim(out) == 0 else out

    def daily_mean(self, resolution_minutes: int = 1) -> float:
        """Time-average of the profile over a full day."""
        samples = np.arange(0.0, HOURS_PER_DAY, resolution_minutes / MINUTES_PER_HOUR)
        return float(np.mean(self(samples)))
