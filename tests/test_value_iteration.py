"""Step 8: Value Iteration on toy MDPs and on the saved Step 7 empirical model."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.config import get_config, load_config
from backend.evaluation.runner import METRICS, run_episode
from backend.rl.environment import DataCenterEnv
from backend.rl.policy_iteration import PolicyIteration
from backend.rl.policy_iteration import load as load_pi
from backend.rl.policy_iteration import save as save_pi
from backend.rl.tabular_model import (
    ALWAYS_VALID_ACTION,
    TabularMDP,
    build_empirical_model,
    model_hash,
    q_values,
)
from backend.rl.value_iteration import (
    ValueIteration,
    ValueIterationController,
    load,
    load_source_model,
    save,
    train_value_iteration,
)


# --------------------------------------------------------------------------- #
# Toy MDPs
# --------------------------------------------------------------------------- #
def mdp_from(transitions: dict, n_states: int, n_actions: int, admissible=None) -> TabularMDP:
    """transitions[(s, a)] = [(prob, s_next, reward), ...]."""
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros_like(P)
    for (s, a), outcomes in transitions.items():
        for prob, s2, r in outcomes:
            P[s, a, s2] += prob
            R[s, a, s2] = r
    if admissible is None:
        admissible = np.ones((n_states, n_actions), dtype=bool)
    return TabularMDP(P, R, admissible)


def vi_for(mdp, gamma=0.95, theta=1e-10, max_iterations=10_000, tie=1e-9) -> ValueIteration:
    vi = ValueIteration(mdp, gamma, theta, max_iterations, tie)
    vi.train()
    return vi


def brute_force(mdp: TabularMDP, gamma: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Optimal V* and every optimal deterministic policy, by enumerating all admissible policies."""
    S, A = mdp.n_states, mdp.n_actions
    options = [np.flatnonzero(mdp.admissible[s]) for s in range(S)]
    results = []
    for idx in np.ndindex(*[len(o) for o in options]):
        policy = np.array([options[s][i] for s, i in enumerate(idx)])
        P_pi = mdp.P[np.arange(S), policy]
        R_pi = (P_pi * mdp.R[np.arange(S), policy]).sum(axis=1)
        results.append((np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi), policy))
    best = max(results, key=lambda r: r[0].sum())[0]
    return best, [p for v, p in results if np.allclose(v, best, atol=1e-9)]


@pytest.fixture
def deterministic():
    """s0: a0 stay (r 0), a1 -> s1 (r 1).  s1: a0 stay (r 2), a1 -> s0 (r 0).  pi* = [1, 0]."""
    return mdp_from({(0, 0): [(1, 0, 0.0)], (0, 1): [(1, 1, 1.0)],
                     (1, 0): [(1, 1, 2.0)], (1, 1): [(1, 0, 0.0)]}, 2, 2)


@pytest.fixture
def stochastic():
    """A risky and a safe action from s0; s1 pays 3 per step, s2 costs 1 per step (both sticky)."""
    return mdp_from({
        (0, 0): [(0.5, 1, 0.0), (0.5, 2, 0.0)],     # gamble
        (0, 1): [(1.0, 0, 0.5)],                    # safe, small reward, stay
        (1, 0): [(0.9, 1, 3.0), (0.1, 0, 3.0)], (1, 1): [(1.0, 0, 0.0)],
        (2, 0): [(0.9, 2, -1.0), (0.1, 0, -1.0)], (2, 1): [(1.0, 0, -2.0)],
    }, 3, 2)


def test_deterministic_toy_known_solution(deterministic):
    vi = vi_for(deterministic)
    assert vi.result.converged
    assert list(vi.get_policy()) == [1, 0]
    np.testing.assert_allclose(vi.get_value_function(), [1 + 0.95 * 40.0, 40.0], atol=1e-7)


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.95])
def test_stochastic_toy_matches_brute_force(stochastic, gamma):
    vi = vi_for(stochastic, gamma=gamma)
    v_star, optimal_policies = brute_force(stochastic, gamma)
    np.testing.assert_allclose(vi.get_value_function(), v_star, atol=1e-7)
    assert any(np.array_equal(vi.get_policy(), p) for p in optimal_policies)


def test_random_mdps_match_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(5):
        S, A = 4, 3
        P = rng.random((S, A, S))
        P /= P.sum(axis=2, keepdims=True)
        mdp = TabularMDP(P, rng.normal(size=(S, A, S)), np.ones((S, A), dtype=bool))
        v_star, policies = brute_force(mdp, 0.9)
        vi = vi_for(mdp, gamma=0.9)
        np.testing.assert_allclose(vi.get_value_function(), v_star, atol=1e-7)
        assert any(np.array_equal(vi.get_policy(), p) for p in policies)


def test_terminal_state_example():
    """s2 is terminal: absorbing with reward 0. Going there (r 5) beats staying (r 0.2 forever) at gamma 0.9."""
    mdp = mdp_from({(0, 0): [(1, 0, 0.2)], (0, 1): [(1, 1, 0.0)],
                    (1, 0): [(1, 2, 5.0)], (1, 1): [(1, 0, 0.0)],
                    (2, 0): [(1, 2, 0.0)], (2, 1): [(1, 2, 0.0)]}, 3, 2)
    vi = vi_for(mdp, gamma=0.9)
    np.testing.assert_allclose(vi.get_value_function(), [0.9 * 5.0, 5.0, 0.0], atol=1e-8)
    assert vi.predict(0) == 1 and vi.predict(1) == 0


def test_tied_actions_break_to_lowest_index():
    mdp = mdp_from({(0, a): [(1, 0, 1.0)] for a in range(3)}, 1, 3)
    vi = vi_for(mdp)
    assert vi.predict(0) == 0
    tol = ValueIteration(mdp_from({(0, 0): [(1, 0, 1.0)], (0, 1): [(1, 0, 1.0 + 1e-6)], (0, 2): [(1, 0, 0.0)]}, 1, 3),
                         0.95, 1e-10, 10_000, tie_tolerance=1e-4)
    tol.train()
    assert tol.predict(0) == 0                      # within tie tolerance of the best -> lowest index


def test_inadmissible_action_never_selected():
    admissible = np.array([[True, False], [True, True]])
    mdp = mdp_from({(0, 0): [(1, 0, 0.0)], (0, 1): [(1, 0, 1e6)], (1, 0): [(1, 1, 0.0)], (1, 1): [(1, 0, 0.0)]},
                   2, 2, admissible)
    vi = vi_for(mdp)
    assert vi.predict(0) == 0 and vi.get_value_function()[0] == pytest.approx(0.0)
    assert np.isneginf(vi.q_values()[0, 1])


def test_bellman_optimality_update_single_step(stochastic):
    vi = ValueIteration(stochastic, 0.9, 1e-10, 10)
    V = np.array([1.0, -2.0, 0.5])
    expected = np.max(stochastic.expected_reward + 0.9 * stochastic.P @ V, axis=1)
    np.testing.assert_allclose(vi.value_update(V), expected)


def test_tolerance_and_iteration_limit(deterministic):
    loose, tight = vi_for(deterministic, theta=1e-2), vi_for(deterministic, theta=1e-8)
    assert loose.result.converged and loose.result.history[-1].max_delta < 1e-2 <= loose.result.history[-2].max_delta
    assert loose.result.iterations < tight.result.iterations
    capped = vi_for(deterministic, theta=1e-12, max_iterations=7)
    assert not capped.result.converged and capped.result.iterations == 7
    assert np.isfinite(capped.get_value_function()).all()


def test_pi_and_vi_agree_on_toy_mdps(deterministic, stochastic):
    for mdp in (deterministic, stochastic):
        vi = vi_for(mdp)
        pi = PolicyIteration(mdp, 0.95, 1e-10, 100, 10_000, 1e-9)
        pi.train()
        np.testing.assert_array_equal(vi.get_policy(), pi.get_policy())
        np.testing.assert_allclose(vi.get_value_function(), pi.get_value_function(), atol=1e-7)


# --------------------------------------------------------------------------- #
# Saved Step 7 model (the real artefact, if it has been trained)
# --------------------------------------------------------------------------- #
PI_DIR = get_config().rl.value_iteration.model_path
needs_artifact = pytest.mark.skipif(not (PI_DIR / "model.npz").exists(),
                                    reason="run `python -m backend.rl.policy_iteration` first")


@pytest.fixture(scope="module")
def source():
    return load_source_model(PI_DIR)


@pytest.fixture(scope="module")
def trained():
    return train_value_iteration(get_config(), PI_DIR)


@needs_artifact
def test_uses_saved_step7_model_with_matching_hash(source, trained):
    vi, src = trained
    payload = json.loads((PI_DIR / "policy.json").read_text(encoding="utf-8"))
    assert src.model_hash == model_hash(src.model) == payload["model"]["model_hash"]
    assert int(src.model.modelled.sum()) == 132 and int(src.model.n_samples.sum()) == 132 * 9 * 100
    assert vi.mdp.n_states == 144 and vi.mdp.n_actions == 9


@needs_artifact
def test_loading_never_builds_a_model(monkeypatch):
    import backend.rl.tabular_model as tm
    import backend.rl.value_iteration as vi_module

    assert not hasattr(vi_module, "build_empirical_model") and not hasattr(vi_module, "collect_snapshots")

    def forbidden(*args, **kwargs):
        raise AssertionError("Value Iteration must not rebuild the model")

    monkeypatch.setattr(tm, "build_empirical_model", forbidden)
    monkeypatch.setattr(tm, "collect_snapshots", forbidden)
    train_value_iteration(get_config(), PI_DIR)


@needs_artifact
def test_saved_model_is_well_formed(source):
    mdp = source.model.to_mdp()
    assert (mdp.P >= 0).all()
    np.testing.assert_allclose(mdp.P.sum(axis=2), 1.0, atol=1e-12)
    assert np.isfinite(mdp.R).all()


@needs_artifact
def test_empirical_vi_converges_finite_and_admissible(trained):
    vi, _ = trained
    r = vi.result
    assert r.converged and r.history[-1].max_delta < vi.theta and r.bellman_residual < vi.theta
    assert np.isfinite(vi.get_value_function()).all()
    assert np.isfinite(vi.q_values()[vi.mdp.admissible]).all()
    assert vi.mdp.admissible[np.arange(144), vi.get_policy()].all()


@needs_artifact
def test_empirical_vi_is_reproducible(trained):
    vi, _ = trained
    again, _ = train_value_iteration(get_config(), PI_DIR)
    np.testing.assert_array_equal(again.get_value_function(), vi.get_value_function())
    np.testing.assert_array_equal(again.get_policy(), vi.get_policy())


@needs_artifact
def test_empirical_vi_agrees_with_saved_policy_iteration(trained):
    vi, src = trained
    pi = load_pi(PI_DIR)
    m = src.model.modelled
    diff = np.abs(vi.get_value_function() - pi.value_function)[m]
    bound = 2 * vi.gamma * vi.theta / (1 - vi.gamma)          # both stopping errors, opposite signs
    assert diff.max() < bound
    assert np.array_equal(vi.get_policy()[m], pi.policy[m])


@needs_artifact
def test_vi_values_approach_from_above_pi_from_below(trained):
    vi, src = trained
    pi = load_pi(PI_DIR)
    m = src.model.modelled
    assert ((vi.get_value_function() - pi.value_function)[m] > 0).all()
    means = [h.v_mean for h in vi.result.history]
    assert all(b <= a for a, b in zip(means, means[1:]))       # rewards < 0, V_0 = 0: monotone decrease


@needs_artifact
def test_save_load_round_trip(tmp_path, trained, source):
    vi, src = trained
    save(vi, src, get_config(), tmp_path)
    loaded = load(tmp_path)
    np.testing.assert_array_equal(loaded.policy, vi.get_policy())
    np.testing.assert_array_equal(loaded.value_function, vi.get_value_function())
    assert all(loaded.predict(s) == vi.predict(s) for s in range(144))
    p = loaded.payload
    assert p["model"]["model_hash"] == source.model_hash and p["gamma"] == 0.95 and p["theta"] == 1e-6
    assert p["tie_tolerance"] == 1e-4 and p["state_space"]["n_states"] == 144 and p["action_space"]["n_actions"] == 9
    assert p["training"]["iterations"] == len(p["training"]["history"])


@needs_artifact
def test_load_rejects_a_different_model(tmp_path, trained):
    vi, src = trained
    save(vi, src, get_config(), tmp_path)
    payload = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    payload["model"]["model_hash"] = "0" * 64
    (tmp_path / "policy.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load(tmp_path)


def test_gamma_mismatch_with_source_model_is_rejected(tmp_path):
    small = load_config(overrides={"rl": {"tabular_model": {"collection_episodes": 1, "collection_minutes": 30}}})
    model = build_empirical_model(small, seed=1, samples_per_state_action=2)
    pi = PolicyIteration.from_config(model.to_mdp(), small.rl.policy_iteration)
    pi.train()
    save_pi(pi, model, small, tmp_path)
    other = load_config(overrides={"rl": {"value_iteration": {"gamma": 0.9}}})
    with pytest.raises(ValueError, match="gamma"):
        train_value_iteration(other, tmp_path)
    vi, src = train_value_iteration(small, tmp_path)     # same gamma: fine, and uses the saved model
    assert src.model_hash == model_hash(model)
    assert vi.mdp.admissible[~model.modelled, ALWAYS_VALID_ACTION].all()


@needs_artifact
def test_full_day_evaluation_through_shared_runner(trained):
    vi, _ = trained
    env = DataCenterEnv(representation="tabular", episode_minutes=1440)   # validates simulator invariants
    controller = ValueIterationController(vi.get_policy(), get_config())
    result = run_episode(env, controller, seed=42)
    assert set(METRICS) <= set(result) and np.isfinite(result["total_reward"])
    assert controller.state_visits.sum() == 1440


@needs_artifact
def test_q_values_helper_is_shared_with_policy_iteration(trained):
    vi, src = trained
    pi = PolicyIteration.from_config(src.model.to_mdp(), get_config().rl.policy_iteration)
    V = vi.get_value_function()
    np.testing.assert_array_equal(pi.q_values(V), q_values(vi.mdp, V, vi.gamma))
