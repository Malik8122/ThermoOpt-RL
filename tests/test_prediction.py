"""Step 12: temperature-forecasting features, dataset, model, baselines, provider."""

import numpy as np
import pytest
import torch

from backend.config import get_config, load_config
from backend.prediction.dataset import (
    EXTRA_FEATURES,
    FEATURE_READERS,
    History,
    Windows,
    generate_history,
    make_windows,
    rack_features,
    record_day,
    split_days,
)
from backend.prediction.evaluate import (
    baseline_predictions,
    crossing_detection,
    evaluate_windows,
    regression_metrics,
    subset,
)
from backend.prediction.model import ForecastProvider, RecurrentRegressor, TemperatureForecaster, train_forecaster
from backend.rl.baselines import make_baseline
from backend.rl.dqn import configure_torch
from backend.rl.environment import DataCenterEnv
from backend.utils.seeding import derive_seed

configure_torch(1)
BASELINE_POLICIES = ["random", "round_robin", "rule_based"]
FEATURES = get_config().prediction.features


def windows(history, days, features=None):
    return make_windows(history, days, 10, [5, 10, 15], features or FEATURES)


@pytest.fixture(scope="module")
def small_cfg():
    return load_config(overrides={"prediction": {"epochs": 3, "behaviour_policies": BASELINE_POLICIES,
                                                 "dataset_episodes": 6, "split": [0.5, 0.25, 0.25]}})


@pytest.fixture(scope="module")
def history(small_cfg):
    return generate_history(small_cfg, minutes=180)


@pytest.fixture(scope="module")
def trained(small_cfg, history):
    return train_forecaster(small_cfg, history)


# --------------------------------------------------------------------------- #
# Features and recording
# --------------------------------------------------------------------------- #
def test_rack_features_read_the_simulator(cfg):
    env = DataCenterEnv(episode_minutes=30)
    env.reset(seed=42)
    f = rack_features(env.dc, cfg.prediction.features)
    assert f.shape == (4, 6) and f.dtype == np.float32
    np.testing.assert_allclose(f[:, 0], env.dc.temperatures(), rtol=1e-6)
    np.testing.assert_allclose(f[:, 1], env.dc.cpu_utilisation(), rtol=1e-6)
    np.testing.assert_allclose(f[:, 3], [r.get_power_consumption().it_kw for r in env.dc.racks], rtol=1e-6)
    # received cooling = delivered level x effectiveness: Rack 3 gets 75 % of the others
    assert f[2, 4] == pytest.approx(0.75 * f[0, 4])
    assert (f[:, 5] == len(env.dc.queue)).all()


def test_unknown_feature_rejected(cfg):
    env = DataCenterEnv(episode_minutes=30)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        rack_features(env.dc, ["temperature", "humidity"])
    assert set(cfg.prediction.features) <= set(FEATURE_READERS)


def test_record_day_has_one_frame_per_minute_boundary(cfg):
    env = DataCenterEnv(episode_minutes=45)
    frames = record_day(env, make_baseline("round_robin", cfg), seed=3, names=cfg.prediction.features)
    assert frames.shape == (46, 4, 6) and np.isfinite(frames).all()
    np.testing.assert_allclose(frames[0, :, 0], cfg.datacenter.initial_temperatures_c, rtol=1e-6)
    np.testing.assert_allclose(frames[-1, :, 0], env.dc.temperatures(), rtol=1e-6)


def test_history_generation_rotates_policies_and_is_reproducible(small_cfg, history):
    assert history.frames.shape == (6, 181, 4, 9)                      # 6 configured + 3 ablation features
    assert history.feature_names == FEATURES + EXTRA_FEATURES
    assert history.policies == BASELINE_POLICIES * 2
    assert history.seeds == [derive_seed(42, "prediction", "day", i) for i in range(6)]
    again = generate_history(small_cfg, days=2, minutes=180)
    np.testing.assert_array_equal(again.frames, history.frames[:2])


def test_generalisation_days_use_different_seeds(small_cfg, history):
    extra = generate_history(small_cfg, days=1, policies=["random"], minutes=60, label="generalisation")
    assert not set(extra.seeds) & set(history.seeds)


def test_history_save_load_round_trip(tmp_path, history):
    history.save(tmp_path / "h.npz")
    loaded = History.load(tmp_path / "h.npz")
    np.testing.assert_array_equal(loaded.frames, history.frames)
    assert (loaded.policies, loaded.seeds, loaded.feature_names) == (history.policies, history.seeds, history.feature_names)


# --------------------------------------------------------------------------- #
# Splits and windows
# --------------------------------------------------------------------------- #
def test_split_is_chronological_by_whole_days():
    train, val, test = split_days(20, [0.7, 0.15, 0.15])
    assert (list(train), list(val), list(test)) == (list(range(14)), [14, 15, 16], [17, 18, 19])
    with pytest.raises(ValueError):
        split_days(2, [0.7, 0.15, 0.15])


def test_windows_are_aligned_with_the_history(history):
    w = windows(history, [1])
    n_per_rack = 181 - 9 - 15
    assert len(w) == 4 * n_per_rack and w.x.shape == (len(w), 10, 6) and w.y.shape == (len(w), 3)
    assert w.minute.min() == 9 and w.minute.max() == 181 - 15 - 1          # never leaves the day
    for i in (0, 57, len(w) - 1):
        d, r, t = w.day[i], w.rack[i], w.minute[i]
        np.testing.assert_array_equal(w.x[i], history.frames[d, t - 9:t + 1, r, :6])
        assert w.last_temperature[i] == history.frames[d, t, r, 0]
        for j, h in enumerate((5, 10, 15)):
            assert w.y[i, j] == pytest.approx(history.frames[d, t + h, r, 0] - history.frames[d, t, r, 0], abs=1e-5)
            assert w.target_temperature[i, j] == pytest.approx(history.frames[d, t + h, r, 0], abs=1e-4)


def test_train_and_test_windows_come_from_different_days(history):
    train = windows(history, range(0, 3))
    test = windows(history, range(5, 6))
    assert not set(train.day) & set(test.day)


# --------------------------------------------------------------------------- #
# Metrics and baselines
# --------------------------------------------------------------------------- #
def test_regression_metrics_known_values():
    m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["mae"] == pytest.approx(2 / 3) and m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert m["r2"] == pytest.approx(1 - 4 / 2) and m["bias"] == pytest.approx(2 / 3)
    assert regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["r2"] == 1.0


def hand_windows(temps_now, temps_prev, changes):
    n = len(temps_now)
    x = np.zeros((n, 10, 6), dtype=np.float32)
    x[:, -1, 0], x[:, -2, 0] = temps_now, temps_prev
    y = np.asarray(changes, dtype=np.float32)
    return Windows(x, y, np.asarray(temps_now, dtype=np.float32), np.zeros(n, int), np.zeros(n, int), np.zeros(n, int))


def test_baselines_persistence_and_linear_trend():
    w = hand_windows([25.0, 26.0], [24.5, 26.2], [[0, 0, 0], [0, 0, 0]])
    b = baseline_predictions(w, 0, [5, 10, 15])
    np.testing.assert_allclose(b["persistence"], [[25, 25, 25], [26, 26, 26]])
    np.testing.assert_allclose(b["linear_trend"], [[27.5, 30.0, 32.5], [25.0, 24.0, 23.0]], atol=1e-4)


def test_zero_output_forecaster_is_exactly_persistence(cfg):
    f = TemperatureForecaster.from_config(cfg)
    with torch.no_grad():
        f.net.head.weight.zero_(), f.net.head.bias.zero_()
    w = hand_windows([25.0, 28.0], [25.0, 28.0], [[1, 2, 3], [-1, -2, -3]])
    np.testing.assert_allclose(f.predict(w.x), [[25, 25, 25], [28, 28, 28]])
    r = evaluate_windows(f, w)
    assert r["+15"]["skill_vs_persistence"] == pytest.approx(0.0) and r["+5"]["model"]["mae"] == pytest.approx(1.0)


def test_crossing_detection_counts(cfg):
    f = TemperatureForecaster.from_config(cfg)
    with torch.no_grad():
        f.net.head.weight.zero_()
        f.net.head.bias.copy_(torch.tensor([0.0, 0.0, 2.0]))     # always predicts +2 °C at +15
    w = hand_windows([26.0, 26.0, 24.0, 28.0], [26.0] * 4, [[0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    c = crossing_detection(f, w, threshold=27.0)["+15"]
    assert c["windows_starting_below"] == 3 and c["actual_crossings"] == 1       # the 28 °C window is excluded
    assert c["model"] == {"precision": 0.5, "recall": 1.0, "flagged": 2}        # flags both 26 °C windows
    assert c["persistence"]["flagged"] == 0 and c["persistence"]["recall"] == 0.0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("architecture", ["lstm", "gru"])
def test_network_shapes(architecture):
    net = RecurrentRegressor(6, 3, architecture, 64, 2, 0.1)
    assert net(torch.zeros(5, 10, 6)).shape == (5, 3)
    assert isinstance(net.rnn, {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[architecture])
    assert net.rnn.num_layers == 2 and net.rnn.hidden_size == 64
    with pytest.raises(ValueError):
        net(torch.zeros(5, 10, 5))


def test_predict_validates_window_shape(cfg):
    f = TemperatureForecaster.from_config(cfg)
    with pytest.raises(ValueError):
        f.predict(np.zeros((2, 9, 6), dtype=np.float32))
    assert f.sequence_length == 10 and f.horizons == [5, 10, 15] and f.architecture == "lstm"


def test_learns_a_simple_rule_and_beats_persistence():
    # Synthetic: the temperature rises by 2 x (feature 1 at the last step) per horizon unit. Persistence cannot know.
    rng = np.random.default_rng(0)

    def synth(n):
        x = rng.normal(size=(n, 10, 3)).astype(np.float32)
        x[:, :, 0] += 25
        drive = x[:, -1, 1]
        y = np.stack([2 * drive, 4 * drive, 6 * drive], axis=1).astype(np.float32)
        return Windows(x, y, x[:, -1, 0].copy(), np.zeros(n, int), np.zeros(n, int), np.zeros(n, int))

    f = TemperatureForecaster(["temperature", "a", "b"], [5, 10, 15], 10, "lstm", 16, 1, 0.0, seed=0)
    rows = f.fit(synth(2000), synth(400), epochs=15, batch_size=64, learning_rate=0.01)
    r = evaluate_windows(f, synth(400))
    assert rows[-1]["train_loss"] < 0.1 * rows[0]["train_loss"]
    assert r["+15"]["skill_vs_persistence"] > 0.9 and r["+15"]["model"]["r2"] > 0.9


def test_training_records_history_and_keeps_best_epoch(trained, small_cfg, history):
    f, rows, info = trained
    assert len(rows) == 3 and sum(r["best"] for r in rows) >= 1
    assert info["best_epoch"] == min(rows, key=lambda r: r["val_loss"])["epoch"]
    assert info["train_days"] == [0, 1, 2] and info["val_days"] == [3, 4] and info["test_days"] == [5]
    val = windows(history, info["val_days"])
    pred = (f.predict_change(val.x) / f.y_std)
    assert float(np.mean((pred - val.y / f.y_std) ** 2)) == pytest.approx(info["best_val_loss"], rel=1e-4)
    assert all(torch.isfinite(p).all() for p in f.net.parameters())


def test_scalers_come_from_training_data_only(trained, history):
    f, _, info = trained
    train = windows(history, info["train_days"])
    np.testing.assert_allclose(f.x_mean, train.x.reshape(-1, 6).mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(f.y_std, train.y.std(axis=0), rtol=1e-5)


def test_training_is_reproducible(trained, small_cfg, history):
    f, rows, _ = trained
    g, rows2, _ = train_forecaster(small_cfg, history)
    assert rows == rows2
    assert all(torch.equal(a, b) for a, b in zip(f.net.parameters(), g.net.parameters()))


def test_save_load_round_trip(tmp_path, trained, history):
    f, _, info = trained
    f.save(tmp_path, {"training": info})
    g = TemperatureForecaster.load(tmp_path)
    w = windows(history, info["test_days"])
    np.testing.assert_array_equal(g.predict(w.x), f.predict(w.x))
    assert g.metadata["architecture"] == "lstm" and g.metadata["training"]["best_epoch"] == info["best_epoch"]
    assert np.isfinite(g.predict(w.x)).all()


def test_evaluation_report_structure(trained, history):
    f, _, info = trained
    w = windows(history, info["test_days"])
    r = evaluate_windows(f, w)
    assert set(r) == {"+5", "+10", "+15"}
    assert {"model", "persistence", "linear_trend", "skill_vs_persistence"} <= set(r["+5"])
    assert all(np.isfinite(r[h][m]["mae"]) for h in r for m in ("model", "persistence", "linear_trend"))
    assert len(subset(w, w.rack == 2)) == len(w) // 4


# --------------------------------------------------------------------------- #
# Live provider
# --------------------------------------------------------------------------- #
def test_provider_matches_offline_forecast_and_is_idempotent(trained, small_cfg):
    f, _, _ = trained
    provider = ForecastProvider(f)
    env = DataCenterEnv(episode_minutes=60, config=small_cfg)
    env.reset(seed=11)
    frames = [rack_features(env.dc, f.feature_names)]
    first = provider(env.dc)
    np.testing.assert_allclose(first, f.predict(np.stack([frames[0]] * 10, axis=1)), atol=1e-5)   # padded history
    for _ in range(14):
        env.step(env.codec.encode(env.codec.delay_index, 1))
        frames.append(rack_features(env.dc, f.feature_names))
        live = provider(env.dc)
    assert live.shape == (4, 3) and len(provider.history) == 10
    np.testing.assert_allclose(live, f.predict(np.stack(frames[-10:], axis=1)), atol=1e-5)
    np.testing.assert_array_equal(provider(env.dc), live)            # same minute: no extra frame
    assert len(provider.history) == 10


def test_provider_resets_on_a_new_episode(trained, small_cfg):
    f, _, _ = trained
    provider = ForecastProvider(f)
    env = DataCenterEnv(episode_minutes=60, config=small_cfg)
    env.reset(seed=1)
    provider(env.dc)
    for _ in range(5):
        env.step(env.codec.encode(env.codec.delay_index, 1))
        provider(env.dc)
    assert len(provider.history) == 6
    env.reset(seed=2)
    provider(env.dc)
    assert len(provider.history) == 1


def test_environment_with_provider_gives_42_features(trained, small_cfg):
    f, _, _ = trained
    env = DataCenterEnv(episode_minutes=30, config=small_cfg, prediction_provider=ForecastProvider(f))
    obs, _ = env.reset(seed=42)
    assert obs.shape == (42,) and env.observation_space.contains(obs)
    names = env.observation_builder.feature_names
    assert names[30] == "rack1_forecast_t+5" and names[-1] == "rack4_forecast_t+15"
    for _ in range(30):
        obs, _, _, truncated, _ = env.step(env.codec.encode(env.codec.delay_index, 1))
        assert obs.shape == (42,) and env.observation_space.contains(obs) and np.isfinite(obs).all()
    assert truncated
    plain = DataCenterEnv(episode_minutes=30, config=small_cfg)
    assert plain.reset(seed=42)[0].shape == (30,)                    # the 30-feature observation is unchanged


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(overrides={"prediction": {"features": ["cpu_utilisation"]}})
    with pytest.raises(ValueError):
        load_config(overrides={"prediction": {"behaviour_policies": []}})
    assert get_config().prediction.behaviour_policies == ["random", "round_robin", "rule_based", "ppo"]


# --------------------------------------------------------------------------- #
# Feature ablation
# --------------------------------------------------------------------------- #
def test_extra_features_read_the_simulator(cfg):
    env = DataCenterEnv(episode_minutes=30)
    env.reset(seed=42)
    f = rack_features(env.dc, EXTRA_FEATURES)
    racks = env.dc.racks
    np.testing.assert_allclose(f[:, 0], [env.dc.cooling.target_level * r.cooling_effectiveness for r in racks], rtol=1e-6)
    assert f[2, 0] == pytest.approx(0.75 * f[0, 0])
    for i, rack in enumerate(racks):
        soon = [w for w in rack.workloads if w.remaining_time <= 15]
        expected = sum(0.6 * w.cpu_requirement / 100 + 0.4 * w.gpu_requirement / 100 for w in soon)
        assert f[i, 1] == pytest.approx(expected, abs=1e-6)
        assert 0.0 <= f[i, 1] <= 1.0
    assert (f[:, 2] == f[0, 2]).all() and 20.0 < f[0, 2] < 28.0            # one room ambient, plausible range


def test_windows_select_feature_columns_by_name(history):
    base = windows(history, [0])
    extended = windows(history, [0], FEATURES + EXTRA_FEATURES)
    assert base.x.shape[-1] == 6 and extended.x.shape[-1] == 9
    np.testing.assert_array_equal(extended.x[..., :6], base.x)
    np.testing.assert_array_equal(extended.y, base.y)                      # same targets either way
    reordered = windows(history, [0], ["cooling_target", "temperature"])
    np.testing.assert_array_equal(reordered.x[..., 1], base.x[..., 0])
    np.testing.assert_array_equal(reordered.last_temperature, base.last_temperature)


def test_extended_forecaster_trains_and_serves(small_cfg, history):
    f, rows, _ = train_forecaster(small_cfg, history, features=FEATURES + EXTRA_FEATURES)
    assert f.feature_names == FEATURES + EXTRA_FEATURES and f.net.n_features == 9 and len(rows) == 3
    env = DataCenterEnv(episode_minutes=20, config=small_cfg, prediction_provider=ForecastProvider(f))
    obs, _ = env.reset(seed=5)
    assert obs.shape == (42,) and env.observation_space.contains(obs)
