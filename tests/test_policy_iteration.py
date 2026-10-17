"""Step 7: Policy Iteration on toy MDPs with known solutions, and on the data-centre empirical model."""

import copy

import numpy as np
import pytest

from backend.config import load_config
from backend.evaluation.runner import METRICS, run_episode
from backend.rl.actions import N_TABULAR_ACTIONS
from backend.rl.environment import DataCenterEnv
from backend.rl.policy_iteration import PolicyIteration, PolicyIterationController, load, save
from backend.rl.state_encoder import TabularStateEncoder
from backend.rl.tabular_model import (
    ALWAYS_VALID_ACTION,
    TabularMDP,
    build_empirical_model,
    collect_snapshots,
    reseed_simulator,
)


# --------------------------------------------------------------------------- #
# Toy MDPs
# --------------------------------------------------------------------------- #
def deterministic_mdp(transitions: dict, n_states: int, n_actions: int, admissible=None) -> TabularMDP:
    """transitions[(s, a)] = (s_next, reward)."""
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros_like(P)
    for (s, a), (s2, r) in transitions.items():
        P[s, a, s2] = 1.0
        R[s, a, s2] = r
    if admissible is None:
        admissible = np.ones((n_states, n_actions), dtype=bool)
    return TabularMDP(P, R, admissible)


def solver(mdp, gamma=0.95, theta=1e-10, max_iterations=100, sweeps=10_000, tie=1e-9) -> PolicyIteration:
    return PolicyIteration(mdp, gamma, theta, max_iterations, sweeps, tie)


@pytest.fixture
def two_state():
    """s0: a0 stay (r 0), a1 -> s1 (r 1).  s1: a0 stay (r 2), a1 -> s0 (r 0).  Optimal: pi = [1, 0]."""
    return deterministic_mdp({(0, 0): (0, 0.0), (0, 1): (1, 1.0), (1, 0): (1, 2.0), (1, 1): (0, 0.0)}, 2, 2)


@pytest.fixture
def patience():
    """s0: a0 stay (r 1, greedy), a1 -> s1 (r 0).  s1: absorbing, r 2.  Far-sighted agents go to s1."""
    return deterministic_mdp({(0, 0): (0, 1.0), (0, 1): (1, 0.0), (1, 0): (1, 2.0), (1, 1): (1, 2.0)}, 2, 2)


def test_toy_two_state_optimal_policy_and_values(two_state):
    pi = solver(two_state)
    result = pi.train()
    assert result.converged
    assert list(pi.get_policy()) == [1, 0]
    np.testing.assert_allclose(pi.get_value_function(), [1 + 0.95 * 40.0, 40.0], atol=1e-6)


@pytest.mark.parametrize("gamma, expected_action", [(0.95, 1), (0.3, 0)])
def test_policy_respects_gamma(patience, gamma, expected_action):
    # gamma 0.95: stay = 1/0.05 = 20 < go = 0.95*2/0.05 = 38.  gamma 0.3: stay = 1.43 > go = 0.86.
    pi = solver(patience, gamma=gamma)
    pi.train()
    assert pi.predict(0) == expected_action


def test_evaluation_matches_exact_linear_solve():
    rng = np.random.default_rng(0)
    S, A = 5, 3
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    R = rng.normal(size=(S, A, S))
    mdp = TabularMDP(P, R, np.ones((S, A), dtype=bool))
    pi = solver(mdp, gamma=0.9)
    policy = np.array([0, 2, 1, 1, 0])
    V, deltas, converged = pi.policy_evaluation(policy)
    P_pi = P[np.arange(S), policy]
    R_pi = (P_pi * R[np.arange(S), policy]).sum(axis=1)
    exact = np.linalg.solve(np.eye(S) - 0.9 * P_pi, R_pi)
    assert converged and deltas[-1] < 1e-10
    np.testing.assert_allclose(V, exact, atol=1e-8)


def test_policy_iteration_matches_brute_force_optimum():
    rng = np.random.default_rng(1)
    S, A, gamma = 4, 3, 0.9
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    R = rng.normal(size=(S, A, S))
    mdp = TabularMDP(P, R, np.ones((S, A), dtype=bool))
    pi = solver(mdp, gamma=gamma)
    pi.train()
    best = None
    for code in range(A ** S):   # enumerate all 81 deterministic policies
        policy = np.array([(code // A ** i) % A for i in range(S)])
        P_pi = P[np.arange(S), policy]
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, (P_pi * R[np.arange(S), policy]).sum(axis=1))
        if best is None or V.sum() > best.sum() + 1e-12:
            best = V
    np.testing.assert_allclose(pi.get_value_function(), best, atol=1e-6)


def test_evaluation_stops_at_tolerance(two_state):
    loose, tight = solver(two_state, theta=1e-2), solver(two_state, theta=1e-8)
    policy = np.array([1, 0])
    _, d_loose, ok_loose = loose.policy_evaluation(policy)
    _, d_tight, ok_tight = tight.policy_evaluation(policy)
    assert ok_loose and ok_tight
    assert d_loose[-1] < 1e-2 <= d_loose[-2]
    assert len(d_loose) < len(d_tight)


def test_evaluation_reports_non_convergence(two_state):
    pi = solver(two_state, theta=1e-12, sweeps=5)
    V, deltas, converged = pi.policy_evaluation(np.array([1, 0]))
    assert not converged and len(deltas) == 5 and np.isfinite(V).all()


def test_improvement_selects_max_q(two_state):
    pi = solver(two_state)
    V = np.array([0.0, 40.0])
    new, changes = pi.policy_improvement(V, np.array([0, 1]))
    Q = pi.q_values(V)
    assert list(new) == list(np.argmax(Q, axis=1)) == [1, 0] and changes == 2


def test_inadmissible_action_never_selected():
    # Action 1 of s0 has a huge reward but is not admissible.
    admissible = np.array([[True, False], [True, True]])
    mdp = deterministic_mdp({(0, 0): (0, 0.0), (0, 1): (0, 1e6), (1, 0): (1, 0.0), (1, 1): (0, 0.0)}, 2, 2, admissible)
    pi = solver(mdp)
    pi.train()
    assert pi.predict(0) == 0
    assert np.isneginf(pi.q_values()[0, 1])


def test_initial_policy_is_lowest_admissible_action():
    admissible = np.array([[False, False, True], [False, True, True], [True, True, True]])
    mdp = deterministic_mdp({(s, a): (s, 0.0) for s in range(3) for a in range(3)}, 3, 3, admissible)
    assert list(solver(mdp).initialize_policy()) == [2, 1, 0]


def test_tie_breaking_is_deterministic():
    # All three actions of s0 are identical: lowest index wins, unless the current action is tied.
    mdp = deterministic_mdp({(0, a): (0, 1.0) for a in range(3)}, 1, 3)
    pi = solver(mdp)
    V = np.array([20.0])
    assert pi.policy_improvement(V, np.array([0]))[0][0] == 0
    assert pi.policy_improvement(V, np.array([2]))[0][0] == 2          # current action is among the ties
    worse = deterministic_mdp({(0, 0): (0, 1.0), (0, 1): (0, 1.0), (0, 2): (0, 0.5)}, 1, 3)
    assert solver(worse).policy_improvement(V, np.array([2]))[0][0] == 0   # not tied -> lowest best index


def test_mdp_rejects_malformed_models():
    P = np.full((2, 1, 2), 0.4)          # rows sum to 0.8
    with pytest.raises(ValueError):
        TabularMDP(P, np.zeros_like(P), np.ones((2, 1), dtype=bool))
    with pytest.raises(ValueError):
        TabularMDP(np.eye(2)[:, None, :], np.full((2, 1, 2), np.nan), np.ones((2, 1), dtype=bool))
    with pytest.raises(ValueError):
        TabularMDP(np.eye(2)[:, None, :], np.zeros((2, 1, 2)), np.zeros((2, 1), dtype=bool))


# --------------------------------------------------------------------------- #
# Data-centre model (small, fast configuration)
# --------------------------------------------------------------------------- #
SAMPLES = 5


@pytest.fixture(scope="module")
def small_cfg():
    return load_config(overrides={"rl": {"tabular_model": {"collection_episodes": 2, "collection_minutes": 60,
                                                           "snapshot_interval_minutes": 5,
                                                           "max_snapshots_per_state": 10}}})


@pytest.fixture(scope="module")
def model(small_cfg):
    return build_empirical_model(small_cfg, seed=42, samples_per_state_action=SAMPLES)


@pytest.fixture(scope="module")
def trained(model, small_cfg):
    pi = PolicyIteration.from_config(model.to_mdp(), small_cfg.rl.policy_iteration)
    pi.train()
    return pi


def test_model_uses_144_states_and_9_actions(model, trained):
    assert model.counts.shape == (144, N_TABULAR_ACTIONS, 144)
    assert trained.get_policy().shape == (144,) and trained.get_value_function().shape == (144,)


def test_model_probabilities_and_rewards(model):
    mdp = model.to_mdp()
    assert (mdp.P >= 0).all()
    np.testing.assert_allclose(mdp.P.sum(axis=2), 1.0, atol=1e-12)
    assert np.isfinite(mdp.R).all() and np.isfinite(mdp.expected_reward).all()
    assert (model.n_samples[model.modelled] == SAMPLES).all()


def test_model_is_closed_under_sampled_transitions(model):
    reached = np.flatnonzero(model.counts.sum(axis=(0, 1)) > 0)
    assert model.modelled[reached].all()
    unmodelled = np.flatnonzero(~model.modelled)
    assert model.counts[:, :, unmodelled].sum() == 0


def test_unmodelled_states_are_absorbing_with_only_the_safe_action(model):
    mdp = model.to_mdp()
    s = int(np.flatnonzero(~model.modelled)[0])
    assert mdp.P[s, :, s].tolist() == [1.0] * N_TABULAR_ACTIONS
    assert np.flatnonzero(mdp.admissible[s]).tolist() == [ALWAYS_VALID_ACTION]


def test_admissible_actions_were_valid_in_samples(model):
    mdp = model.to_mdp()
    m = model.modelled
    assert (model.valid_count[m][mdp.admissible[m]] > 0).all()
    assert mdp.admissible[m, ALWAYS_VALID_ACTION].all()
    assert (model.valid_fraction[m, ALWAYS_VALID_ACTION] == 1.0).all()


def test_same_seed_reproduces_model(model, small_cfg):
    again = build_empirical_model(small_cfg, seed=42, samples_per_state_action=SAMPLES)
    np.testing.assert_array_equal(again.counts, model.counts)
    np.testing.assert_array_equal(again.reward_sum, model.reward_sum)
    np.testing.assert_array_equal(again.valid_count, model.valid_count)


def test_different_seed_changes_model(model, small_cfg):
    other = build_empirical_model(small_cfg, seed=7, samples_per_state_action=SAMPLES)
    assert not np.array_equal(other.counts, model.counts)


def test_reseeding_gives_independent_samples(small_cfg):
    env = DataCenterEnv(representation="tabular", episode_minutes=60, config=small_cfg)
    env.reset(seed=42)
    base = copy.deepcopy(env.dc)

    def temps_after(seed):
        dc = copy.deepcopy(base)
        reseed_simulator(dc, seed)
        dc.step()
        return dc.temperatures()

    np.testing.assert_array_equal(temps_after(1), temps_after(1))
    assert not np.array_equal(temps_after(1), temps_after(2))


def test_snapshots_are_grouped_by_their_abstract_state(small_cfg):
    encoder = TabularStateEncoder(small_cfg.discretisation)
    pool = collect_snapshots(small_cfg, seed=42)
    for s, snaps in pool.items():
        assert 0 < len(snaps) <= small_cfg.rl.tabular_model.max_snapshots_per_state
        assert all(encoder.encode(encoder.observe(dc)) == s for dc in snaps)


def test_state_encoding_is_deterministic(small_cfg):
    env = DataCenterEnv(representation="tabular", episode_minutes=60, config=small_cfg)
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert a == b == 90   # documented initial state (RL_ENVIRONMENT.md §3)


def test_trained_policy_is_stable_admissible_and_finite(trained):
    r = trained.result
    assert r.converged and r.history[-1].policy_stable and r.history[-1].policy_changes == 0
    assert r.iterations <= trained.max_iterations
    assert all(h.evaluation_converged for h in r.history)
    policy = trained.get_policy()
    assert trained.mdp.admissible[np.arange(144), policy].all()
    assert np.isfinite(trained.get_value_function()).all()
    Q = trained.q_values()
    assert np.isfinite(Q[trained.mdp.admissible]).all()


def test_policy_is_greedy_with_respect_to_its_values(trained):
    Q = trained.q_values()
    chosen = Q[np.arange(144), trained.get_policy()]
    assert (chosen >= Q.max(axis=1) - trained.tie_tolerance).all()


def test_retraining_reproduces_policy_values_and_history(trained, model, small_cfg):
    again = PolicyIteration.from_config(model.to_mdp(), small_cfg.rl.policy_iteration)
    again.train()
    np.testing.assert_array_equal(again.get_policy(), trained.get_policy())
    np.testing.assert_array_equal(again.get_value_function(), trained.get_value_function())
    assert [h.policy_changes for h in again.result.history] == [h.policy_changes for h in trained.result.history]


def test_save_and_load_round_trip(tmp_path, trained, model, small_cfg):
    save(trained, model, small_cfg, tmp_path)
    loaded = load(tmp_path)
    np.testing.assert_array_equal(loaded.policy, trained.get_policy())
    np.testing.assert_array_equal(loaded.value_function, trained.get_value_function())
    assert all(loaded.predict(s) == trained.predict(s) for s in range(144))
    np.testing.assert_array_equal(loaded.model.counts, model.counts)
    p = loaded.payload
    assert p["gamma"] == 0.95 and p["state_space"]["n_states"] == 144 and p["action_space"]["n_actions"] == 9
    assert p["training"]["converged"] and p["hyperparameters"]["theta"] == 1e-6
    np.testing.assert_array_equal(loaded.model.to_mdp().admissible, np.asarray(p["admissible"], dtype=bool))


def test_policy_runs_full_episode_through_shared_runner(trained, small_cfg):
    env = DataCenterEnv(representation="tabular", episode_minutes=240, config=small_cfg)  # validates invariants
    controller = PolicyIterationController(trained.get_policy(), small_cfg)
    result = run_episode(env, controller, seed=42)
    assert set(METRICS) <= set(result)
    assert controller.state_visits.sum() == controller.action_counts.sum() == 240
    assert np.isfinite(result["total_reward"])
    visited = np.flatnonzero(controller.state_visits)
    assert all(controller.policy[s] == trained.predict(s) for s in visited)
