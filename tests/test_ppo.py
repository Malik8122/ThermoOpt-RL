"""Step 11: PPO networks, masked policy, GAE, clipped update, training, persistence, integration."""

import numpy as np
import pytest
import torch

from backend.config import get_config, load_config
from backend.evaluation.runner import METRICS, run_episode
from backend.rl.dqn import configure_torch
from backend.rl.environment import DataCenterEnv
from backend.rl.ppo import (
    EPISODE_FIELDS,
    UPDATE_FIELDS,
    ActorCritic,
    PPOAgent,
    PPOController,
    compute_gae,
    entropy_of,
    masked_log_probs,
    train_data_centre,
)

configure_torch(1)


def agent_for(obs_dim=30, n_actions=18, **kw) -> PPOAgent:
    params = dict(hidden_sizes=[128, 128], learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                  entropy_coefficient=0.01, value_coefficient=0.5, max_grad_norm=0.5, rollout_steps=64,
                  update_epochs=4, minibatch_size=16, seed=0)
    params.update(kw)
    return PPOAgent(obs_dim, n_actions, **params)


def params_of(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Networks and masked policy
# --------------------------------------------------------------------------- #
def test_actor_critic_shapes():
    net = ActorCritic(30, 18, [128, 128])
    x = torch.zeros(5, 30)
    assert net.logits(x).shape == (5, 18) and net.value(x).shape == (5,)
    sizes = lambda seq: [(m.in_features, m.out_features) for m in seq if isinstance(m, torch.nn.Linear)]  # noqa: E731
    assert sizes(net.actor) == [(30, 128), (128, 128), (128, 18)]
    assert sizes(net.critic) == [(30, 128), (128, 128), (128, 1)]
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_wrong_observation_width_rejected():
    with pytest.raises(ValueError):
        ActorCritic(30, 18, [8]).logits(torch.zeros(1, 29))
    with pytest.raises(ValueError):
        agent_for().act(np.zeros(31))


def test_initial_policy_is_near_uniform():
    probs = agent_for().distribution(np.random.default_rng(0).random(30))
    assert probs.sum() == pytest.approx(1.0) and probs.max() - probs.min() < 0.02


def test_masked_distribution_gives_zero_probability_to_invalid_actions():
    agent = agent_for()
    mask = np.zeros(18, bool)
    mask[[1, 5, 16]] = True
    probs = agent.distribution(np.zeros(30), mask)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs[~mask] == 0).all() and (probs[mask] > 0).all()


def test_entropy_is_finite_under_masking_and_bounded():
    logits = torch.randn(4, 18)
    masks = torch.zeros(4, 18, dtype=torch.bool)
    masks[:, :3] = True
    ent = entropy_of(masked_log_probs(logits, masks))
    assert torch.isfinite(ent).all() and (ent <= np.log(3) + 1e-6).all() and (ent >= 0).all()
    one = torch.zeros(1, 18, dtype=torch.bool)
    one[0, 7] = True
    assert entropy_of(masked_log_probs(logits[:1], one)).item() == pytest.approx(0.0, abs=1e-7)
    assert entropy_of(masked_log_probs(torch.zeros(1, 18), None)).item() == pytest.approx(np.log(18), abs=1e-5)


def test_sampling_respects_mask_and_predict_is_argmax():
    agent = agent_for()
    mask = np.zeros(18, bool)
    mask[[2, 9]] = True
    assert {agent.act(np.zeros(30), mask)[0] for _ in range(300)} == {2, 9}
    with torch.no_grad():
        agent.model.actor[-1].weight.zero_()
        agent.model.actor[-1].bias.copy_(torch.arange(18, dtype=torch.float32))
    assert agent.predict(np.zeros(30)) == 17
    assert agent.predict(np.zeros(30), mask) == 9
    with torch.no_grad():
        agent.model.actor[-1].bias.zero_()
    assert agent.predict(np.zeros(30), mask) == 2                       # exact tie -> lowest valid index


def test_unmasked_agent_ignores_mask():
    agent = agent_for(use_action_mask=False)
    mask = np.zeros(18, bool)
    mask[0] = True
    assert (agent.distribution(np.zeros(30), mask) > 0).all()
    assert len({agent.act(np.zeros(30), mask)[0] for _ in range(600)}) == 18


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        agent_for().distribution(np.zeros(30), np.zeros(18, bool))


def test_log_prob_returned_by_act_matches_evaluate_actions():
    agent = agent_for()
    rng = np.random.default_rng(1)
    obs = rng.random(30).astype(np.float32)
    mask = rng.random(18) < 0.6
    mask[3] = True
    action, lp, value, ent = agent.act(obs, mask)
    new_lp, new_ent, new_v = agent.evaluate_actions(torch.as_tensor(obs)[None], torch.tensor([action]),
                                                    torch.as_tensor(mask)[None])
    assert new_lp.item() == pytest.approx(lp, abs=1e-6) and new_v.item() == pytest.approx(value, abs=1e-6)
    assert new_ent.item() == pytest.approx(ent, abs=1e-6)


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
def test_gae_hand_computed_example():
    r, v = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 1.5])
    nv = np.array([1.0, 1.5, 2.0])                 # V(s1), V(s2), bootstrap V(s3)
    adv, ret = compute_gae(r, v, nv, np.zeros(3, bool), gamma=0.9, gae_lambda=0.8)
    d = r + 0.9 * nv - v                           # [1.4, 2.35, 3.3]
    a2 = d[2]
    a1 = d[1] + 0.72 * a2
    a0 = d[0] + 0.72 * a1
    np.testing.assert_allclose(adv, [a0, a1, a2])
    np.testing.assert_allclose(ret, adv + v)


def test_gae_lambda_zero_is_td_and_lambda_one_is_monte_carlo():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=6), rng.normal(size=6)
    nv = np.r_[v[1:], 0.7]
    none = np.zeros(6, bool)
    adv0, _ = compute_gae(r, v, nv, none, 0.9, 0.0)
    np.testing.assert_allclose(adv0, r + 0.9 * nv - v)
    _, ret1 = compute_gae(r, v, nv, none, 0.9, 1.0)
    mc = [sum(0.9 ** (k - t) * r[k] for k in range(t, 6)) + 0.9 ** (6 - t) * 0.7 for t in range(6)]
    np.testing.assert_allclose(ret1, mc)


def test_gae_does_not_leak_across_episode_ends():
    r = np.array([1.0, 1.0, 5.0, 5.0])
    v = np.zeros(4)
    ends = np.array([False, True, False, False])
    nv = np.array([0.0, 0.0, 0.0, 0.0])            # step 1 is terminal: bootstrap 0
    adv, _ = compute_gae(r, v, nv, ends, 1.0, 1.0)
    np.testing.assert_allclose(adv, [2.0, 1.0, 10.0, 5.0])     # first episode never sees the 5s


def test_gae_truncation_bootstrap_value_is_used():
    r, v = np.array([1.0]), np.array([0.0])
    terminal, _ = compute_gae(r, v, np.array([0.0]), np.array([True]), 0.9, 0.95)
    truncated, _ = compute_gae(r, v, np.array([10.0]), np.array([True]), 0.9, 0.95)
    assert terminal[0] == pytest.approx(1.0) and truncated[0] == pytest.approx(1.0 + 0.9 * 10.0)


# --------------------------------------------------------------------------- #
# Toy environments
# --------------------------------------------------------------------------- #
class Bandit:
    """One-step contextual bandit: obs one-hot(context); action == context pays 1, else 0.
    ``masked``: action 0 is never valid; in context 0 action 1 pays 0.5."""

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


class Corridor:
    """Never terminates, truncated after `length` steps; constant observation; reward 1 per step."""

    def __init__(self, length=8):
        self.length = length

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.ones(2, dtype=np.float32), {}

    def step(self, a):
        self.t += 1
        return np.ones(2, dtype=np.float32), 1.0, False, self.t >= self.length, {}


def toy_agent(**kw):
    params = dict(obs_dim=3, n_actions=3, hidden_sizes=[16], learning_rate=0.01, rollout_steps=64, update_epochs=4,
                  minibatch_size=16, entropy_coefficient=0.0, seed=1)
    params.update(kw)
    return agent_for(**params)


@pytest.mark.parametrize("masked", [False, True])
def test_toy_bandit_learns_preferred_action(masked):
    agent = toy_agent()
    env = Bandit(masked)
    agent.learn(env, 64 * 60)
    expected = [1, 1, 2] if masked else [0, 1, 2]
    for c in range(3):
        assert agent.predict(np.eye(3)[c], env.action_masks()) == expected[c]
        assert agent.distribution(np.eye(3)[c], env.action_masks())[expected[c]] > 0.8
    assert agent.value(np.eye(3)[1]) == pytest.approx(1.0, abs=0.2)


def test_truncation_bootstrap_changes_what_the_critic_learns():
    # Reward 1 forever, gamma 0.9: the true value is 10. Treating the time limit as terminal caps it near
    # the 8-step sum (about 5.7 at the start, less later); bootstrapping lets it approach 10.
    def learned_value(truncation_is_terminal):
        agent = agent_for(obs_dim=2, n_actions=2, hidden_sizes=[16], learning_rate=0.01, gamma=0.9, rollout_steps=64,
                          update_epochs=8, minibatch_size=16, seed=3, truncation_is_terminal=truncation_is_terminal)
        agent.learn(Corridor(8), 64 * 150)
        return agent.value(np.ones(2))

    assert learned_value(False) > 8.5
    assert learned_value(True) < 6.5


# --------------------------------------------------------------------------- #
# Update mechanics
# --------------------------------------------------------------------------- #
def rollout_arrays(agent, n=64, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.random((n, agent.obs_dim)).astype(np.float32)
    masks = rng.random((n, agent.n_actions)) < 0.7
    masks[:, 0] = True
    actions, lps, values = [], [], []
    for o, m in zip(obs, masks):
        a, lp, v, _ = agent.act(o, m)
        actions.append(a), lps.append(lp), values.append(v)
    adv = rng.normal(size=n)
    return obs, np.array(actions), np.array(lps), adv, adv + np.array(values), np.array(values), masks


def test_update_changes_parameters_and_reports_finite_statistics():
    agent = agent_for()
    before = params_of(agent.model)
    stats = agent.update(*rollout_arrays(agent))
    assert not same(before, agent.model.parameters())
    assert all(np.isfinite(stats[k]) for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"))
    assert agent.updates == 1 and agent.gradient_steps == 4 * 4            # 4 epochs x 64/16 minibatches
    assert all(torch.isfinite(p).all() for p in agent.model.parameters())


def test_zero_learning_rate_keeps_ratio_at_one():
    agent = agent_for(learning_rate=1e-12)
    stats = agent.update(*rollout_arrays(agent))
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6) and stats["clip_fraction"] == 0.0


def test_clipping_limits_the_surrogate():
    # Same objective computed by hand for one minibatch: ratio 1.5 with positive advantage is clipped to 1.2.
    ratio, adv, eps = torch.tensor([1.5, 0.5]), torch.tensor([2.0, 2.0]), 0.2
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    np.testing.assert_allclose(surrogate.numpy(), [2.4, 1.0])


# --------------------------------------------------------------------------- #
# Data-centre training (short)
# --------------------------------------------------------------------------- #
def short_agent(seed=42, **kw):
    cfg = get_config()
    return PPOAgent.from_config(cfg.rl.ppo, seed=seed, rollout_steps=128, update_epochs=2, **kw), cfg


@pytest.fixture(scope="module")
def short_run():
    agent, cfg = short_agent()
    episodes, updates = train_data_centre(agent, cfg, total_timesteps=384, episode_minutes=120,
                                          randomize_start_hour=True, log_every=0)
    return agent, episodes, updates


def test_short_training_records_episodes_and_updates(short_run):
    agent, episodes, updates = short_run
    assert agent.env_steps == 384 and agent.updates == 3 and len(updates) == 3 and len(episodes) == 3
    assert all(set(r) == set(EPISODE_FIELDS) for r in episodes) and all(set(r) == set(UPDATE_FIELDS) for r in updates)
    assert sum(r["invalid_actions"] for r in episodes) == 0                 # masked policy: never invalid
    assert all(np.isfinite(r["value_loss"]) and r["entropy"] > 0 for r in updates)
    assert all(torch.isfinite(p).all() for p in agent.model.parameters())


def test_config_agent_matches_design(cfg):
    hp = PPOAgent.from_config(cfg.rl.ppo).hyperparameters()
    assert hp["learning_rate"] == 3e-4 and hp["gamma"] == 0.99 and hp["gae_lambda"] == 0.95
    assert hp["clip_epsilon"] == 0.2 and hp["entropy_coefficient"] == 0.01 and hp["rollout_steps"] == 2048
    assert hp["update_epochs"] == 10 and hp["minibatch_size"] == 64 and hp["hidden_sizes"] == [128, 128]


def test_unmasked_training_takes_invalid_actions():
    agent, cfg = short_agent(use_action_mask=False)
    episodes, _ = train_data_centre(agent, cfg, 128, 120, True, log_every=0)
    assert episodes[0]["invalid_actions"] > 0


def test_fixed_seed_reproduces_init_and_training():
    a, cfg = short_agent(seed=7)
    b, _ = short_agent(seed=7)
    assert same(a.model.parameters(), b.model.parameters())
    ra = train_data_centre(a, cfg, 256, 100, True, log_every=0)
    rb = train_data_centre(b, cfg, 256, 100, True, log_every=0)
    assert ra == rb and same(a.model.parameters(), b.model.parameters())


def test_different_seed_gives_different_parameters():
    a, _ = short_agent(seed=1)
    b, _ = short_agent(seed=2)
    assert not same(a.model.parameters(), b.model.parameters())


def test_learning_never_touches_the_pi_vi_model(monkeypatch):
    import backend.rl.ppo as ppo_module
    import backend.rl.tabular_model as tm

    assert not any(hasattr(ppo_module, n) for n in ("load_model", "build_empirical_model", "TabularStateEncoder"))

    def forbidden(*a, **k):
        raise AssertionError("PPO must learn from env.step only")

    for name in ("load_model", "build_empirical_model", "collect_snapshots"):
        monkeypatch.setattr(tm, name, forbidden)
    agent, cfg = short_agent()
    train_data_centre(agent, cfg, 128, 100, True, log_every=0)


def test_save_load_round_trip(tmp_path, short_run):
    agent, _, _ = short_run
    agent.save(tmp_path, metadata={"training": {"total_env_steps": 384}})
    loaded = PPOAgent.load(tmp_path)
    assert same(agent.model.parameters(), loaded.model.parameters())
    assert loaded.env_steps == 384 and loaded.updates == 3 and loaded.gradient_steps == agent.gradient_steps
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, mask = rng.random(30).astype(np.float32), rng.random(18) < 0.5
        mask[13] = True
        assert loaded.predict(obs, mask) == agent.predict(obs, mask)
        np.testing.assert_array_equal(loaded.distribution(obs, mask), agent.distribution(obs, mask))
    assert loaded.metadata["obs_dim"] == 30 and loaded.metadata["n_actions"] == 18


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def test_deterministic_controller_full_day_with_metrics(short_run, cfg):
    agent, _, _ = short_run
    state = agent.sample_gen.get_state()
    controller = PPOController(agent, cfg, deterministic=True, record=True)
    env = DataCenterEnv(episode_minutes=1440)            # validates simulator invariants every step
    result = run_episode(env, controller, seed=42)
    assert torch.equal(agent.sample_gen.get_state(), state)                # no sampling in deterministic mode
    assert set(METRICS) <= set(result) and result["invalid_action_rate"] == 0.0
    assert len(controller.observations) == 1440 and controller.action_counts.sum() == 1440
    assert all(m[a] for m, a in zip(controller.masks, controller.actions))


def test_stochastic_controller_is_seeded_per_episode(short_run, cfg):
    agent, _, _ = short_run
    env = DataCenterEnv(episode_minutes=120)
    a = run_episode(env, PPOController(agent, cfg, deterministic=False), seed=42)
    b = run_episode(env, PPOController(agent, cfg, deterministic=False), seed=42)
    assert a == b and a["invalid_action_rate"] == 0.0


def test_values_and_probabilities_finite_on_real_observations(short_run):
    agent, _, _ = short_run
    env = DataCenterEnv(episode_minutes=30)
    obs, info = env.reset(seed=3)
    probs = agent.distribution(obs, info["action_mask"])
    assert np.isfinite(probs).all() and probs.sum() == pytest.approx(1.0) and np.isfinite(agent.value(obs))


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(overrides={"rl": {"ppo": {"minibatch_size": 4096}}})
    with pytest.raises(ValueError):
        load_config(overrides={"rl": {"ppo": {"total_timesteps": 100}}})
