"""Step 5: tabular state abstraction and action encodings."""

import itertools

import pytest

from backend.rl.actions import (
    N_TABULAR_ACTIONS,
    ActionCodec,
    CoolingCommand,
    PlacementKind,
    TabularPlacement,
    decode_tabular_action,
    encode_tabular_action,
)
from backend.rl.state_encoder import (
    TabularState,
    TabularStateEncoder,
    decode_tabular_state,
    encode_tabular_state,
)
from backend.simulation.datacentre import DataCentre


@pytest.fixture(scope="module")
def encoder(cfg) -> TabularStateEncoder:
    return TabularStateEncoder(cfg.discretisation)


# ------------------------------------------------------------ tabular states
def test_exactly_144_states(encoder):
    assert encoder.shape == (4, 3, 3, 2, 2)
    assert encoder.n_states == 144
    every_combination = itertools.product(*(range(n) for n in encoder.shape))
    indices = {encoder.encode(TabularState(*levels)) for levels in every_combination}
    assert indices == set(range(144))                      # bijection onto [0, 144)


def test_every_state_round_trips(encoder):
    for i in range(encoder.n_states):
        assert encoder.encode(encoder.decode(i)) == i
        assert encode_tabular_state(decode_tabular_state(i)) == i


def test_mixed_radix_order(encoder):
    assert encoder.encode(TabularState(0, 0, 0, 0, 1)) == 1
    assert encoder.encode(TabularState(0, 0, 0, 1, 0)) == 2
    assert encoder.encode(TabularState(0, 0, 1, 0, 0)) == 4
    assert encoder.encode(TabularState(1, 0, 0, 0, 0)) == 36
    assert encoder.encode(TabularState(3, 2, 2, 1, 1)) == 143


@pytest.mark.parametrize("temp, level", [(18.0, 0), (24.99, 0), (25.0, 1), (26.99, 1), (27.0, 2),
                                         (29.99, 2), (30.0, 3), (40.0, 3)])
def test_temperature_bin_boundaries(encoder, temp, level):
    assert encoder.discretise(temp, 50, 50, 300, 0).temperature == level


@pytest.mark.parametrize("util, level", [(0.0, 0), (39.99, 0), (40.0, 1), (69.99, 1), (70.0, 2), (100.0, 2)])
def test_workload_bins(encoder, util, level):
    assert encoder.discretise(24, util, 50, 300, 0).workload == level


@pytest.mark.parametrize("cooling, level", [(0, 0), (20, 0), (39.99, 0), (40, 1), (60, 1), (65, 1), (70, 2), (80, 2), (100, 2)])
def test_cooling_bins(encoder, cooling, level):
    assert encoder.discretise(24, 50, cooling, 300, 0).cooling == level


def test_carbon_and_queue_bins(encoder):
    assert encoder.discretise(24, 50, 50, 399.9, 0).carbon == 0
    assert encoder.discretise(24, 50, 50, 400.0, 0).carbon == 1
    assert [encoder.discretise(24, 50, 50, 300, q).queue for q in (0, 1, 7)] == [0, 1, 1]


def test_observe_reads_the_simulator(encoder, cfg):
    """Initial twin at 08:00: hottest 28.1 °C, 461/800 units used, cooling 65 %, carbon 440, queue empty."""
    state = encoder.observe(DataCentre(cfg))
    assert encoder.describe(state) == {"temperature": "HIGH", "workload": "MEDIUM", "cooling": "MEDIUM",
                                       "carbon": "HIGH", "queue": "EMPTY"}
    assert encoder.encode(state) == 90


def test_out_of_range_states_are_rejected(encoder):
    with pytest.raises(ValueError):
        encoder.encode(TabularState(4, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        encoder.decode(144)


# ------------------------------------------------------------------ actions
def test_exactly_9_tabular_actions_round_trip():
    assert N_TABULAR_ACTIONS == 9
    pairs = {decode_tabular_action(a) for a in range(9)}
    assert pairs == set(itertools.product(TabularPlacement, CoolingCommand))
    for a in range(9):
        assert encode_tabular_action(*decode_tabular_action(a)) == a
    with pytest.raises(ValueError):
        decode_tabular_action(9)


def test_exactly_18_primitive_actions_with_documented_layout(cfg):
    codec = ActionCodec(cfg.datacenter.num_racks)
    assert codec.n_actions == 18
    for a in range(18):
        act = codec.decode(a)
        assert codec.encode(act.placement, act.cooling) == a
        assert act.cooling == a % 3
        if a < 12:
            assert act.kind is PlacementKind.PLACE and act.rack_index == a // 3
        elif a < 15:
            assert act.kind is PlacementKind.DELAY
        else:
            assert act.kind is PlacementKind.MIGRATE
    assert len({(codec.decode(a).placement, codec.decode(a).cooling) for a in range(18)}) == 18
    with pytest.raises(ValueError):
        codec.decode(18)
