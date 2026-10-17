"""Grid price / carbon profile."""

import numpy as np
import pytest

from backend.simulation.grid import GridSignals


@pytest.fixture
def grid(cfg) -> GridSignals:
    return GridSignals(cfg.grid)


def test_profile_passes_through_brief_anchors(grid):
    for hour, price, carbon in [(10, 8.50, 420), (12, 9.20, 450), (14, 7.10, 310), (16, 6.50, 250)]:
        assert grid.price_at(hour) == pytest.approx(price)
        assert grid.carbon_at(hour) == pytest.approx(carbon)


def test_profile_is_periodic_and_does_not_overshoot(grid, cfg):
    assert grid.carbon_at(0.0) == pytest.approx(grid.carbon_at(24.0))
    assert grid.carbon_at(23.99) == pytest.approx(grid.carbon_at(0.0), abs=1.0)   # no jump at midnight
    hours = np.linspace(0, 24, 2000, endpoint=False)
    carbon = np.array([grid.carbon_at(h) for h in hours])
    prices = np.array([grid.price_at(h) for h in hours])
    anchors = np.array(cfg.grid.profile)
    assert carbon.min() >= anchors[:, 2].min() - 1e-9 and carbon.max() <= anchors[:, 2].max() + 1e-9
    assert prices.min() >= anchors[:, 1].min() - 1e-9 and prices.max() <= anchors[:, 1].max() + 1e-9


def test_high_carbon_flag(grid):
    assert grid.is_high_carbon(12.0) and not grid.is_high_carbon(16.0)
