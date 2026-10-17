"""Step 1 tests: the configuration loads, matches the brief, and rejects bad input."""

import numpy as np
import pytest
from pydantic import ValidationError

from backend.config import get_config, load_config
from backend.utils.seeding import derive_seed, make_rng


def test_default_config_matches_brief():
    cfg = get_config()
    dc = cfg.datacenter
    assert cfg.project.seed == 42
    assert dc.num_racks == 4 and dc.servers_per_rack == 10 and dc.gpu_nodes_per_rack == 2
    assert dc.total_cpu_capacity == 400 and dc.total_gpu_capacity == 400
    assert dc.initial_temperatures_c == [24.5, 23.8, 28.1, 23.5]
    assert dc.initial_cpu_utilisation_pct == [72, 51, 88, 43]
    assert dc.initial_gpu_utilisation_pct == [55, 40, 82, 30]
    assert cfg.cooling.initial_level_pct == 65
    assert cfg.cooling.levels_pct == [0, 20, 40, 60, 80, 100]
    assert (cfg.thermal.alpha, cfg.thermal.beta) == (0.035, 0.045)
    assert (cfg.thermal.zones.warning_c, cfg.thermal.zones.critical_c) == (27.0, 30.0)
    assert cfg.simulation.timestep_minutes == 1
    assert cfg.rl.value_iteration.gamma == 0.95
    assert cfg.cooling.max_cooling_power_kw == 16.0          # scaled-down simulation parameter
    assert cfg.cooling.response_time_minutes == 3.0
    assert cfg.thermal.rack_cooling_effectiveness == [1.0, 1.0, 0.75, 1.0]
    assert {k: t.probability for k, t in cfg.workloads.types.items()} == {
        "NORMAL_CPU": 0.50, "GPU_INFERENCE": 0.30, "AI_TRAINING": 0.20}
    assert dc.total_servers == 40 and dc.total_gpu_nodes == 8


def test_grid_profile_contains_brief_anchors():
    anchors = {hour: (price, carbon) for hour, price, carbon in get_config().grid.profile}
    assert anchors[10] == (8.50, 420)
    assert anchors[12] == (9.20, 450)
    assert anchors[14] == (7.10, 310)
    assert anchors[16] == (6.50, 250)


def test_derived_power_figures():
    cfg = get_config()
    # 4 racks x (10 x 500 W + 2 x 700 W) = 25.6 kW
    assert cfg.max_it_power_kw == pytest.approx(25.6)
    # 4 racks x (10 x 150 W + 2 x 100 W) = 6.8 kW
    assert cfg.idle_it_power_kw == pytest.approx(6.8)
    assert cfg.max_total_power_kw == pytest.approx(25.6 + cfg.cooling.max_cooling_power_kw)


def test_tabular_state_count():
    # temperature 4 x workload 3 x cooling 3 x carbon 2 x queue 2
    assert get_config().discretisation.num_states == 144


def test_overrides_are_applied():
    cfg = load_config(overrides={"project": {"seed": 7}, "cooling": {"max_cooling_power_kw": 400}})
    assert cfg.project.seed == 7
    assert cfg.cooling.max_cooling_power_kw == 400
    assert cfg.thermal.alpha == 0.035  # untouched sections keep file values


@pytest.mark.parametrize(
    "overrides",
    [
        {"datacenter": {"initial_temperatures_c": [24.0, 25.0]}},             # wrong rack count
        {"reward": {"weights": {"energy": 0.9}}},                            # weights do not sum to 1
        {"thermal": {"zones": {"warning_c": 31.0, "critical_c": 30.0}}},     # zones out of order
        {"workloads": {"types": {"NORMAL_CPU": {"probability": 0.9}}}},      # probabilities do not sum to 1
        {"workloads": {"types": {"NORMAL_CPU": {"cpu": {"distribution": "triangular", "min": 10, "max": 40, "mode": 50}}}}},
        {"thermal": {"ambient_coupling": 1.0}},                              # unstable explicit-Euler update
        {"cooling": {"levels_pct": [0, 50, 40]}},                            # not ascending
        {"thermal": {"alpah": 0.1}},                                          # typo -> unknown key
    ],
)
def test_invalid_configs_are_rejected(overrides):
    with pytest.raises(ValidationError):
        load_config(overrides=overrides)


def test_seeded_generators_are_reproducible():
    a = make_rng(42).normal(size=5)
    b = make_rng(42).normal(size=5)
    assert np.array_equal(a, b)
    assert derive_seed(42, "workloads") == derive_seed(42, "workloads")
    assert derive_seed(42, "workloads") != derive_seed(42, "thermal")
