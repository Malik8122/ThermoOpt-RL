"""Step 9: tabular Q-Learning on toy MDPs and on the tabular data-centre environment."""

import numpy as np
import pytest

from backend.config import get_config, load_config
from backend.evaluation.runner import METRICS, run_episode
from backend.rl.environment import DataCenterEnv
from backend.rl.q_learning import (
    CSV_FIELDS,
    FALLBACK_ACTION,
    QLearning,
    QLearningController,
    train_data_centre,
    write_training_csv,
)


# --------------------------------------------------------------------------- #
# Toy environment
# --------------------------------------------------------------------------- #
class ToyEnv:
    """Deterministic episodic MDP (gamma = 0.9 in the tests).

        s0: a0 -> s1 (r 0)          a1 -> END (r 1)
        s1: a0 -> END (r 10)        a1 -> s0 (r 0)
    Q*(s1,a0) = 10, Q*(s0,a0) = 9, Q*(s0,a1) = 1, Q*(s1,a1) = 8.1. Optimal: a0 everywhere.
    ``masked``: in s1 action 0 is unavailable, so the best s1 can do is go back to s0.
    """

    END = 2
    table = {(0, 0): (1, 0.0), (0, 1): (2, 1.0), (1, 0): (2, 10.0), (1, 1): (0, 0.0)}

    def __init__(self, masked: bool = False, max_steps: int = 50) -> None:
        self.masked, self.max_steps = masked, max_steps

    def action_masks(self):
        return np.array([not (self.masked and self.s == 1), True])

    def reset(self, seed=None, options=None):
        self.s, self.t = 0, 0
        return self.s, {"action_mask": self.action_masks()}

    def step(self, a):
        if not self.action_masks()[a]:
            raise AssertionError("agent chose a masked action")
        self.s, r = self.table[(self.s, a)]
        self.t += 1
        terminated = self.s == self.END
        return self.s, r, terminated, self.t >= self.max_steps and not terminated, {}


def toy_agent(**kw) -> QLearning:
    params = dict(learning_rate=0.5, gamma=0.9, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                  tie_tolerance=1e-9, seed=0)
    params.update(kw)
    return QLearning(3, 2, **params)


def train_toy(agent, env, episodes=400):
    for e in range(episodes):
        agent.learn_episode(env, agent.epsilon_at(e))
    return agent


# --------------------------------------------------------------------------- #
# Q-table and update
# --------------------------------------------------------------------------- #
def test_q_table_shape_init_and_finite(cfg):
    agent = QLearning.from_config(cfg.rl.q_learning)
    assert agent.get_q_table().shape == (144, 9)
    assert (agent.get_q_table() == 0).all() and np.isfinite(agent.get_q_table()).all()
    other = QLearning.from_config(cfg.rl.q_learning)
    np.testing.assert_array_equal(other.get_q_table(), agent.get_q_table())


def test_standard_update_is_correct():
    agent = toy_agent(learning_rate=0.25, gamma=0.8)
    agent.q_table[1] = [2.0, 6.0]
    agent.q_table[0, 0] = 1.0
    change = agent.update(0, 0, reward=0.5, next_state=1, next_mask=None, terminal=False)
    # target = 0.5 + 0.8 * 6 = 5.3; Q <- 1 + 0.25 * (5.3 - 1) = 2.075
    assert agent.q_table[0, 0] == pytest.approx(2.075) and change == pytest.approx(1.075)


def test_terminal_update_ignores_next_state():
    agent = toy_agent(learning_rate=0.5)
    agent.q_table[1] = [100.0, 100.0]
    agent.update(0, 1, reward=1.0, next_state=1, next_mask=None, terminal=True)
    assert agent.q_table[0, 1] == pytest.approx(0.5)            # 0 + 0.5 * (1 - 0)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.95])
def test_gamma_scales_bootstrap(gamma):
    agent = toy_agent(learning_rate=1.0, gamma=gamma)
    agent.q_table[1] = [4.0, 0.0]
    agent.update(0, 0, 1.0, 1, None, False)
    assert agent.q_table[0, 0] == pytest.approx(1.0 + gamma * 4.0)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_learning_rate_scales_step(alpha):
    agent = toy_agent(learning_rate=alpha)
    agent.update(0, 0, 2.0, 0, None, True)
    assert agent.q_table[0, 0] == pytest.approx(alpha * 2.0)


def test_masked_max_ignores_invalid_next_actions():
    agent = toy_agent(learning_rate=1.0, gamma=0.5)
    agent.q_table[1] = [1000.0, 2.0]
    agent.update(0, 0, 0.0, 1, next_mask=np.array([False, True]), terminal=False)
    assert agent.q_table[0, 0] == pytest.approx(1.0)            # 0.5 * 2, not 0.5 * 1000
    assert agent.masked_max(1, np.array([False, True])) == 2.0


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        toy_agent().select_action(0, np.array([False, False]), 0.0)


# --------------------------------------------------------------------------- #
# Exploration
# --------------------------------------------------------------------------- #
def test_epsilon_one_explores_uniformly_over_valid_actions():
    agent = QLearning(1, 4, 0.1, 0.9, 1.0, 0.0, 1.0, seed=3)
    agent.q_table[0] = [9.0, 0.0, 0.0, 0.0]
    mask = np.array([True, True, False, True])
    picks = [agent.select_action(0, mask, 1.0) for _ in range(3000)]
    assert all(explored for _, explored in picks)
    counts = np.bincount([a for a, _ in picks], minlength=4)
    assert counts[2] == 0 and all(900 < counts[a] < 1100 for a in (0, 1, 3))


def test_epsilon_zero_is_greedy_and_masked():
    agent = QLearning(1, 3, 0.1, 0.9, 1.0, 0.0, 1.0, seed=3)
    agent.q_table[0] = [1.0, 5.0, 3.0]
    assert agent.select_action(0, None, 0.0) == (1, False)
    assert agent.select_action(0, np.array([True, False, True]), 0.0) == (2, False)


def test_greedy_ties_break_to_lowest_valid_index():
    agent = QLearning(1, 4, 0.1, 0.9, 1.0, 0.0, 1.0, tie_tolerance=1e-4, seed=0)
    agent.q_table[0] = [0.0, 1.0, 1.0 + 5e-5, 1.0]
    assert agent.greedy_action(0) == 1
    assert agent.greedy_action(0, np.array([True, False, True, True])) == 2


def test_epsilon_schedule_decays_to_floor(cfg):
    agent = QLearning.from_config(cfg.rl.q_learning)
    assert agent.epsilon_at(0) == 1.0
    assert agent.epsilon_at(1) == pytest.approx(0.995)
    assert agent.epsilon_at(100) == pytest.approx(0.995 ** 100)
    assert agent.epsilon_at(10_000) == 0.05
    eps = [agent.epsilon_at(e) for e in range(1000)]
    assert all(b <= a for a, b in zip(eps, eps[1:])) and min(eps) == 0.05


def test_seeded_exploration_is_reproducible():
    def draws(seed):
        agent = QLearning(1, 9, 0.1, 0.9, 1.0, 0.0, 1.0, seed=seed)
        return [agent.select_action(0, None, 0.5) for _ in range(200)]

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)


# --------------------------------------------------------------------------- #
# Toy learning
# --------------------------------------------------------------------------- #
def test_toy_mdp_converges_to_known_solution():
    agent = train_toy(toy_agent(), ToyEnv())
    np.testing.assert_allclose(agent.q_table[:2], [[9.0, 1.0], [10.0, 8.1]], atol=0.05)
    assert agent.greedy_action(0) == 0 and agent.greedy_action(1) == 0
    assert (agent.q_table[ToyEnv.END] == 0).all()                # terminal state is never updated


def test_toy_mdp_with_mask_learns_masked_optimum():
    # s1 can only go back to s0, so s0 should take the immediate reward: Q*(s0,a1)=1 > Q*(s0,a0)=0.9*Q*(s1,a1)=0.81
    agent = train_toy(toy_agent(), ToyEnv(masked=True))
    assert agent.greedy_action(0) == 1
    assert agent.q_table[0, 1] == pytest.approx(1.0, abs=0.02)
    assert agent.q_table[0, 0] == pytest.approx(0.81, abs=0.05)
    assert agent.q_table[1, 0] == 0.0                            # the masked action was never taken or updated


def test_toy_discount_changes_learned_values():
    low = train_toy(toy_agent(gamma=0.05), ToyEnv())
    assert low.q_table[0, 0] == pytest.approx(0.5, abs=0.05)     # 0.05 * 10
    assert low.greedy_action(0) == 1                             # myopic: take the 1 now


# --------------------------------------------------------------------------- #
# Data-centre environment (short runs)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def short_run():
    cfg = get_config()
    agent = QLearning.from_config(cfg.rl.q_learning, seed=42)
    rows = train_data_centre(agent, cfg, episodes=3, episode_minutes=120, randomize_start_hour=True, log_every=0)
    return agent, rows


def test_trains_multiple_episodes_with_valid_records(short_run):
    agent, rows = short_run
    assert len(rows) == 3 and agent.episodes_trained == 3
    assert [r["episode"] for r in rows] == [0, 1, 2]
    assert all(set(CSV_FIELDS) == set(r) for r in rows)
    assert all(r["explore_steps"] + r["exploit_steps"] == 120 for r in rows)
    assert rows[0]["epsilon"] == 1.0 and rows[2]["epsilon"] == pytest.approx(0.995 ** 2)
    assert agent.state_counts.sum() == agent.visit_counts.sum() == 360


def test_masked_training_never_takes_invalid_actions(short_run):
    agent, rows = short_run
    assert sum(r["invalid_actions"] for r in rows) == 0
    taken = agent.visit_counts > 0
    assert (agent.valid_counts[taken] > 0).all()


def test_states_stay_within_144_and_q_finite(short_run):
    agent, _ = short_run
    assert agent.state_counts.shape == (144,) and np.isfinite(agent.get_q_table()).all()
    assert (agent.get_policy() >= 0).all() and (agent.get_policy() < 9).all()


def test_training_is_reproducible_and_seed_dependent():
    cfg = get_config()

    def run(seed):
        agent = QLearning.from_config(cfg.rl.q_learning, seed=seed)
        rows = train_data_centre(agent, cfg, 2, 60, True, log_every=0)
        return agent.get_q_table(), rows

    q1, r1 = run(5)
    q2, r2 = run(5)
    q3, _ = run(6)
    np.testing.assert_array_equal(q1, q2)
    assert r1 == r2
    assert not np.array_equal(q1, q3)


def test_learning_never_touches_the_pi_vi_model(monkeypatch):
    import backend.rl.q_learning as ql
    import backend.rl.tabular_model as tm

    for name in ("load_model", "build_empirical_model", "collect_snapshots", "load_source_model"):
        assert not hasattr(ql, name)

    def forbidden(*args, **kwargs):
        raise AssertionError("Q-Learning must learn from env.step only")

    for name in ("load_model", "build_empirical_model", "collect_snapshots"):
        monkeypatch.setattr(tm, name, forbidden)
    cfg = get_config()
    train_data_centre(QLearning.from_config(cfg.rl.q_learning), cfg, 1, 30, True, log_every=0)


def test_starts_from_zero_not_from_pi_vi(cfg):
    agent = QLearning.from_config(cfg.rl.q_learning)
    assert not agent.q_table.any() and not agent.visit_counts.any()
    # Untrained policy table: every state unvisited -> the fallback action, not the PI/VI policy.
    assert (agent.get_policy() == FALLBACK_ACTION).all()


def test_policy_table_uses_observed_valid_actions(short_run):
    agent, _ = short_run
    policy, adm = agent.get_policy(), agent.admissible()
    assert adm[np.arange(144), policy].all()
    unvisited = agent.state_counts == 0
    assert (policy[unvisited] == FALLBACK_ACTION).all()


def test_save_load_round_trip(tmp_path, short_run, cfg):
    agent, _ = short_run
    agent.save(tmp_path, cfg, metadata={"training": {"episodes": 3}})
    loaded = QLearning.load(tmp_path)
    np.testing.assert_array_equal(loaded.get_q_table(), agent.get_q_table())
    np.testing.assert_array_equal(loaded.get_policy(), agent.get_policy())
    rng = np.random.default_rng(0)
    for s in range(144):
        mask = rng.random(9) < 0.6
        mask[FALLBACK_ACTION] = True
        assert loaded.predict(s, mask) == agent.predict(s, mask)
    meta = loaded.metadata
    assert meta["q_table_shape"] == [144, 9] and meta["gamma"] == 0.95 and meta["learning_rate"] == 0.1
    assert meta["episodes_trained"] == 3 and meta["seed"] == 42 and meta["state_space"]["n_states"] == 144


def test_training_csv_is_written(tmp_path, short_run):
    _, rows = short_run
    path = tmp_path / "training.csv"
    write_training_csv(rows, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(CSV_FIELDS) and len(lines) == 4


def test_greedy_controller_uses_epsilon_zero_and_full_day_metrics(short_run, cfg):
    agent, _ = short_run
    rng_state = agent.rng.bit_generator.state
    controller = QLearningController(agent, cfg)
    env = DataCenterEnv(representation="tabular", episode_minutes=1440)   # validates simulator invariants
    result = run_episode(env, controller, seed=42)
    assert controller.epsilon == 0.0
    assert agent.rng.bit_generator.state == rng_state                    # no exploration draws at all
    assert set(METRICS) <= set(result) and result["invalid_action_rate"] == 0.0
    assert controller.state_visits.sum() == 1440


def test_config_rejects_min_above_start():
    with pytest.raises(ValueError):
        load_config(overrides={"rl": {"q_learning": {"epsilon_min": 0.5, "epsilon_start": 0.2}}})
