"""Unit conversions. Every conversion factor in the project lives here, once."""

from __future__ import annotations

W_PER_KW = 1000.0
MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0
PERCENT = 100.0
G_PER_KG = 1000.0


def pct_to_fraction(value_pct: float) -> float:
    """0-100 percentage -> 0-1 fraction."""
    return value_pct / PERCENT


def fraction_to_pct(value: float) -> float:
    """0-1 fraction -> 0-100 percentage."""
    return value * PERCENT


def w_to_kw(watts: float) -> float:
    return watts / W_PER_KW


def kw_to_kwh(power_kw: float, minutes: float) -> float:
    """Energy of a constant power draw over ``minutes``: E = P * t / 60."""
    return power_kw * minutes / MINUTES_PER_HOUR


def g_to_kg(grams: float) -> float:
    return grams / G_PER_KG
