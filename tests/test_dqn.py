"""Step 10: DQN network, replay buffer, masking, Bellman targets, training, persistence, integration."""

import numpy as np
import pytest
import torch

from backend.config import get_config, load_config
from backend.evaluation.runner import METRICS, run_episode
from backend.rl.dqn import (
    CSV_FIELDS,
    DQNAgent,
    DQNController,
    QNetwork,
    configure_torch,
    train_data_centre,
)
from backend.rl.environment import DataCenterEnv
from backend.rl.replay_buffer import ReplayBuffer

configure_torch(1)


def agent_for(obs_dim=30, n_actions=18, **kw) -> DQNAgent:
    params = dict(hidden_sizes=[128, 128], learning_rate=1e-3, gamma=0.95, batch_size=8, replay_buffer_size=100,
                  target_update_frequency=5, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=100,
                  learning_starts=8, train_frequency=1, max_grad_norm=10.0, seed=0)
    params.update(kw)
    return DQNAgent(obs_dim, n_actions, **params)


def params_of(net) -> list[torch.Tensor]:
    return [p.detach().clone() for p in net.parameters()]


def fill(agent, n=20, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        mask = rng.random(agent.n_actions) < 0.7
        mask[0] = True
        agent.store_transition(rng.random(agent.obs_dim), int(rng.integers(agent.n_actions)), float(rng.normal()),
                               rng.random(agent.obs_dim), False, False, np.ones(agent.n_actions, bool), mask)


def set_constant_q(agent, values):
    """Make the online (and target) network output fixed Q-values for every input."""
    with torch.no_grad():
        last = agent.online.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.as_tensor(values, dtype=torch.float32))
    agent.target.load_state_dict(agent.online.state_dict())


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def test_network_shapes_and_architecture():
    net = QNetwork(30, 18, [128, 128])
    out = net(torch.zeros(5, 30))
    assert out.shape == (5, 18)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(30, 128), (128, 128), (128, 18)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in net.net) == 2


def test_network_rejects_wrong_input_width():
    with pytest.raises(ValueError):
        QNetwork(30, 18, [128, 128])(torch.zeros(1, 29))
    with pytest.raises(ValueError):
        agent_for().q_values(np.zeros(31, dtype=np.float32))


def test_parameters_finite_and_on_cpu():
    agent = agent_for()
    assert all(torch.isfinite(p).all() for p in agent.online.parameters())
    assert all(p.device.type == "cpu" for p in agent.online.parameters())
    assert agent.device.type == "cpu"


def test_config_agent_matches_project_baseline(cfg):
    agent = DQNAgent.from_config(cfg.rl.dqn)
    hp = agent.hyperparameters()
    assert (agent.obs_dim, agent.n_actions) == (30, 18) and hp["hidden_sizes"] == [128, 128]
    assert hp["learning_rate"] == 0.001 and hp["gamma"] == 0.95 and hp["batch_size"] == 64
    assert hp["replay_buffer_size"] == 10000 and hp["target_update_frequency"] == 100
    assert isinstance(agent.optimizer, torch.optim.Adam) and isinstance(agent.loss_fn, torch.nn.SmoothL1Loss)


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def test_buffer_stores_transitions_with_masks():
    buf = ReplayBuffer(10, 3, 4)
    m, m2 = np.array([1, 0, 1, 1], bool), np.array([0, 1, 1, 0], bool)
    buf.add([1, 2, 3], 2, 0.5, [4, 5, 6], True, False, m, m2)
    assert len(buf) == 1
    np.testing.assert_array_equal(buf.masks[0], m)
    np.testing.assert_array_equal(buf.next_masks[0], m2)
    assert buf.terminated[0] and not buf.truncated[0] and buf.actions[0] == 2 and buf.rewards[0] == 0.5


def test_buffer_masks_are_copies_not_references():
    buf = ReplayBuffer(4, 2, 3)
    mask = np.array([True, True, False])
    buf.add([0, 0], 0, 0.0, [0, 0], False, False, mask, mask)
    mask[:] = False
    assert buf.masks[0].tolist() == [True, True, False]


def test_buffer_sample_shapes():
    buf = ReplayBuffer(50, 30, 18)
    for i in range(20):
        buf.add(np.full(30, i), i % 18, float(i), np.full(30, i + 1), False, i == 19, np.ones(18, bool), np.ones(18, bool))
    b = buf.sample(8, np.random.default_rng(0))
    assert b.states.shape == (8, 30) and b.next_states.shape == (8, 30) and b.actions.shape == (8,)
    assert b.masks.shape == (8, 18) and b.next_masks.shape == (8, 18) and b.terminated.dtype == bool
    np.testing.assert_array_equal(b.next_states[:, 0], b.states[:, 0] + 1)   # rows stay aligned


def test_buffer_capacity_wraps_oldest_first():
    buf = ReplayBuffer(3, 1, 2)
    for i in range(5):
        buf.add([i], 0, float(i), [i], False, False, [1, 1], [1, 1])
    assert len(buf) == 3 and buf.total_added == 5
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_buffer_sampling_is_seeded_and_validated():
    buf = ReplayBuffer(20, 2, 2)
    for i in range(20):
        buf.add([i, i], 0, float(i), [i, i], False, False, [1, 1], [1, 1])
    a = buf.sample(10, np.random.default_rng(3)).rewards
    b = buf.sample(10, np.random.default_rng(3)).rewards
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        ReplayBuffer(5, 2, 2).sample(1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        buf.add([1, 2, 3], 0, 0.0, [1, 2], False, False, [1, 1], [1, 1])


# --------------------------------------------------------------------------- #
# Action selection
# --------------------------------------------------------------------------- #
def test_epsilon_one_selects_only_valid_actions():
    agent = agent_for()
    mask = np.zeros(18, bool)
    mask[[2, 7, 13]] = True
    picks = [agent.select_action(np.zeros(30), mask, 1.0) for _ in range(600)]
    assert all(explored for _, explored in picks)
    assert {a for a, _ in picks} == {2, 7, 13}


def test_epsilon_zero_selects_highest_valid_q():
    agent = agent_for()
    q = np.arange(18, dtype=float)
    set_constant_q(agent, q)
    assert agent.select_action(np.zeros(30), None, 0.0) == (17, False)
    mask = np.ones(18, bool)
    mask[15:] = False
    assert agent.select_action(np.zeros(30), mask, 0.0) == (14, False)


def test_ties_break_to_lowest_valid_index():
    agent = agent_for()
    set_constant_q(agent, np.zeros(18))
    assert agent.greedy_action(np.zeros(30)) == 0
    mask = np.ones(18, bool)
    mask[:4] = False
    assert agent.greedy_action(np.zeros(30), mask) == 4


def test_unmasked_mode_ignores_mask():
    agent = agent_for(use_action_mask=False)
    set_constant_q(agent, np.arange(18, dtype=float))
    mask = np.zeros(18, bool)
    mask[0] = True
    assert agent.greedy_action(np.zeros(30), mask) == 17               # masked-out action still chosen
    picks = {agent.select_action(np.zeros(30), mask, 1.0)[0] for _ in range(500)}
    assert len(picks) == 18


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        agent_for().greedy_action(np.zeros(30), np.zeros(18, bool))


# --------------------------------------------------------------------------- #
# Bellman target
# --------------------------------------------------------------------------- #
def target_for(agent, r, next_mask, terminated=False, truncated=False):
    return agent.compute_targets(torch.tensor([r]), torch.zeros(1, agent.obs_dim), torch.tensor([terminated]),
                                 torch.tensor([truncated]), torch.as_tensor(next_mask)[None]).item()


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.95])
def test_target_applies_gamma(gamma):
    agent = agent_for(gamma=gamma)
    set_constant_q(agent, np.r_[4.0, np.zeros(17)])
    assert target_for(agent, 1.0, np.ones(18, bool)) == pytest.approx(1.0 + gamma * 4.0)


def test_target_terminal_and_truncation_handling():
    agent = agent_for(truncation_is_terminal=True)
    set_constant_q(agent, np.full(18, 7.0))
    assert target_for(agent, 2.0, np.ones(18, bool), terminated=True) == pytest.approx(2.0)
    assert target_for(agent, 2.0, np.ones(18, bool), truncated=True) == pytest.approx(2.0)
    boot = agent_for(truncation_is_terminal=False)
    set_constant_q(boot, np.full(18, 7.0))
    assert target_for(boot, 2.0, np.ones(18, bool), truncated=True) == pytest.approx(2.0 + 0.95 * 7.0)
    assert target_for(boot, 2.0, np.ones(18, bool), terminated=True) == pytest.approx(2.0)


def test_target_uses_masked_next_max_only():
    agent = agent_for(gamma=0.5)
    q = np.zeros(18)
    q[3], q[9] = 1000.0, 6.0
    set_constant_q(agent, q)
    mask = np.zeros(18, bool)
    mask[[0, 9]] = True
    assert target_for(agent, 0.0, mask) == pytest.approx(3.0)            # 0.5 * 6, not 0.5 * 1000
    unmasked = agent_for(gamma=0.5, use_action_mask=False)
    set_constant_q(unmasked, q)
    assert target_for(unmasked, 0.0, mask) == pytest.approx(500.0)


def test_terminal_target_is_finite_even_with_masked_inf():
    agent = agent_for()
    y = agent.compute_targets(torch.tensor([1.0]), torch.zeros(1, 30), torch.tensor([True]), torch.tensor([False]),
                              torch.ones(1, 18, dtype=torch.bool))
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------- #
# Training mechanics
# --------------------------------------------------------------------------- #
def test_train_step_changes_parameters_and_is_finite():
    agent = agent_for()
    fill(agent)
    before = params_of(agent.online)
    loss, q = agent.train_step()
    assert np.isfinite(loss) and np.isfinite(q)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters()))
    assert all(torch.isfinite(p).all() for p in agent.online.parameters())


def test_target_network_syncs_every_n_updates():
    agent = agent_for(target_update_frequency=3)
    fill(agent)
    frozen = params_of(agent.target)
    agent.train_step(), agent.train_step()
    assert all(torch.equal(a, b) for a, b in zip(frozen, agent.target.parameters()))
    agent.train_step()
    assert agent.target_updates == 1
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))


def test_epsilon_linear_schedule(cfg):
    agent = DQNAgent.from_config(cfg.rl.dqn)
    assert agent.epsilon(0) == 1.0
    assert agent.epsilon(25_000) == pytest.approx(0.525)
    assert agent.epsilon(50_000) == pytest.approx(0.05)
    assert agent.epsilon(10**7) == pytest.approx(0.05)                   # floor (1 - 0.95 in floating point)
    eps = [agent.epsilon(t) for t in range(0, 60_000, 500)]
    assert all(b <= a for a, b in zip(eps, eps[1:])) and min(eps) >= 0.05 - 1e-12


class ToyEnv:
    """One-step contextual bandit: observation one-hot(context); action == context pays 1, else 0.
    With `masked`, action 0 is never valid, so context 0's best valid action is 1 (pays 0.5)."""

    def __init__(self, masked=False):
        self.masked, self.rng = masked, np.random.default_rng(0)

    def action_masks(self):
        return np.array([not self.masked, True, True])

    def reset(self, seed=None, options=None):
        self.c = int(self.rng.integers(3))
        return np.eye(3, dtype=np.float32)[self.c], {"action_mask": self.action_masks()}

    def step(self, a):
        assert self.action_masks()[a]
        r = 1.0 if a == self.c else (0.5 if self.masked and self.c == 0 and a == 1 else 0.0)
        return np.eye(3, dtype=np.float32)[self.c], r, True, False, {}


@pytest.mark.parametrize("masked", [False, True])
def test_toy_bandit_learns_preferred_action(masked):
    agent = agent_for(obs_dim=3, n_actions=3, hidden_sizes=[16], learning_rate=0.01, batch_size=16,
                      replay_buffer_size=500, learning_starts=32, epsilon_decay_steps=300, seed=1)
    env = ToyEnv(masked)
    for _ in range(800):
        agent.learn_episode(env)
    expected = [1, 1, 2] if masked else [0, 1, 2]
    for c in range(3):
        assert agent.greedy_action(np.eye(3)[c], env.action_masks()) == expected[c]
    assert agent.q_values(np.eye(3)[1])[1] == pytest.approx(1.0, abs=0.15)


def test_warm_up_delays_updates():
    agent = agent_for(obs_dim=3, n_actions=3, hidden_sizes=[8], learning_starts=20, batch_size=4)
    env = ToyEnv()
    for _ in range(19):
        agent.learn_episode(env)
    assert agent.training_updates == 0 and len(agent.buffer) == 19
    agent.learn_episode(env)
    assert agent.training_updates == 1


# --------------------------------------------------------------------------- #
# Data-centre training (short)
# --------------------------------------------------------------------------- #
def short_agent(seed=42, **kw):
    cfg = get_config()
    return DQNAgent.from_config(cfg.rl.dqn, seed=seed, learning_starts=64, **kw), cfg


@pytest.fixture(scope="module")
def short_run():
    agent, cfg = short_agent()
    rows = train_data_centre(agent, cfg, episodes=2, episode_minutes=60, randomize_start_hour=True, log_every=0)
    return agent, rows


def test_short_training_run_records_everything(short_run):
    agent, rows = short_run
    assert len(rows) == 2 and all(set(r) == set(CSV_FIELDS) for r in rows)
    assert agent.env_steps == 120 and len(agent.buffer) == 120 and agent.training_updates == 120 - 64 + 1
    assert rows[-1]["training_updates"] == agent.training_updates
    assert all(np.isfinite(r["mean_loss"]) for r in rows[1:])
    assert sum(r["invalid_actions"] for r in rows) == 0                    # masked: never invalid
    assert all(torch.isfinite(p).all() for p in agent.online.parameters())


def test_replay_masks_match_the_environment(short_run):
    agent, _ = short_run
    n = len(agent.buffer)
    assert agent.buffer.masks[:n].any(axis=1).all() and agent.buffer.next_masks[:n].any(axis=1).all()
    assert agent.buffer.masks[np.arange(n), agent.buffer.actions[:n]].all()    # taken actions were valid
    np.testing.assert_array_equal(agent.buffer.masks[1:60], agent.buffer.next_masks[:59])


def test_unmasked_training_can_take_invalid_actions():
    agent, cfg = short_agent(use_action_mask=False)
    rows = train_data_centre(agent, cfg, 1, 60, True, log_every=0)
    assert rows[0]["invalid_actions"] > 0      # environment semantics: no-op + penalty, counted


def test_fixed_seed_reproduces_init_and_training():
    a, cfg = short_agent(seed=7)
    b, _ = short_agent(seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))
    ra = train_data_centre(a, cfg, 1, 90, True, log_every=0)
    rb = train_data_centre(b, cfg, 1, 90, True, log_every=0)
    assert ra == rb
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))


def test_different_seed_gives_different_parameters():
    a, _ = short_agent(seed=1)
    b, _ = short_agent(seed=2)
    assert not all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))


def test_learning_never_touches_the_pi_vi_model(monkeypatch):
    import backend.rl.dqn as dqn_module
    import backend.rl.tabular_model as tm

    assert not any(hasattr(dqn_module, n) for n in ("load_model", "build_empirical_model", "TabularStateEncoder"))

    def forbidden(*a, **k):
        raise AssertionError("DQN must learn from env.step only")

    for name in ("load_model", "build_empirical_model", "collect_snapshots"):
        monkeypatch.setattr(tm, name, forbidden)
    agent, cfg = short_agent()
    train_data_centre(agent, cfg, 1, 70, True, log_every=0)


def test_save_load_round_trip(tmp_path, short_run):
    agent, _ = short_run
    agent.save(tmp_path, metadata={"training": {"episodes": 2}})
    loaded = DQNAgent.load(tmp_path)
    for net in ("online", "target"):
        assert all(torch.equal(x, y) for x, y in zip(getattr(agent, net).parameters(), getattr(loaded, net).parameters()))
    assert loaded.training_updates == agent.training_updates and loaded.env_steps == agent.env_steps
    assert loaded.optimizer.state_dict()["state"].keys() == agent.optimizer.state_dict()["state"].keys()
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, mask = rng.random(30).astype(np.float32), rng.random(18) < 0.5
        mask[13] = True
        assert loaded.predict(obs, mask) == agent.predict(obs, mask)
    meta = loaded.metadata
    assert meta["obs_dim"] == 30 and meta["n_actions"] == 18 and meta["hyperparameters"]["seed"] == 42


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def test_feature_environment_contract():
    env = DataCenterEnv(episode_minutes=30)
    obs, info = env.reset(seed=42)
    assert obs.shape == (30,) and env.observation_space.contains(obs) and env.action_space.n == 18
    assert info["action_mask"].shape == (18,)


def test_greedy_controller_full_day_with_metrics(short_run, cfg):
    agent, _ = short_run
    state = agent.explore_rng.bit_generator.state
    controller = DQNController(agent, cfg, record=True)
    env = DataCenterEnv(episode_minutes=1440)            # validates simulator invariants every step
    result = run_episode(env, controller, seed=42)
    assert controller.epsilon == 0.0 and agent.explore_rng.bit_generator.state == state
    assert set(METRICS) <= set(result) and result["invalid_action_rate"] == 0.0
    assert len(controller.observations) == 1440 and controller.action_counts.sum() == 1440
    assert all(m[a] for m, a in zip(controller.masks, controller.actions))


def test_q_values_on_real_observations_are_finite(short_run):
    agent, _ = short_run
    env = DataCenterEnv(episode_minutes=30)
    obs, _ = env.reset(seed=3)
    assert np.isfinite(agent.q_values(obs)).all() and agent.q_values(obs).shape == (18,)


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(overrides={"rl": {"dqn": {"epsilon_end": 0.9, "epsilon_start": 0.5}}})
    with pytest.raises(ValueError):
        load_config(overrides={"rl": {"dqn": {"batch_size": 20000}}})
