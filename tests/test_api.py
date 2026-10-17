"""Dashboard API: live twin control, every controller, saved results, WebSocket."""

import json
import math

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.service import CONTROLLERS, SimulationService, controller_available
from backend.config import get_config


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(autostart_loop=False)) as c:
        yield c


def reset(client, **body):
    r = client.post("/api/simulation/reset", json={"seed": 42, "controller": "rule_based", **body})
    assert r.status_code == 200
    return r.json()


def all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return True


def test_status_describes_the_initial_twin(client):
    s = reset(client)
    assert s["minute"] == 0 and s["clock"] == "08:00" and s["controller"] == "rule_based" and s["seed"] == 42
    assert [r["name"] for r in s["racks"]] == ["Rack 1", "Rack 2", "Rack 3", "Rack 4"]
    assert [round(r["temperature"], 1) for r in s["racks"]] == [24.5, 23.8, 28.1, 23.5]
    assert s["racks"][2]["status"] == "WARNING" and s["racks"][2]["cooling_effectiveness"] == 0.75
    assert s["zones"] == {"warning": 27.0, "critical": 30.0} and s["horizons"] == [5, 10, 15]
    assert s["power"]["total_kw"] == pytest.approx(s["power"]["it_kw"] + s["power"]["cooling_kw"])
    assert client.get("/api/status").json()["minute"] == 0
    assert len(client.get("/api/racks").json()) == 4


def test_step_advances_time_and_records_a_decision(client):
    reset(client)
    s = client.post("/api/simulation/step", json={"minutes": 1}).json()
    assert s["minute"] == 1 and s["counters"]["steps"] == 1
    d = s["last_decision"]
    assert d["controller"] == "rule_based" and d["explanation"] and isinstance(d["valid"], bool)
    assert set(d["reward_parts"]) == {"energy", "carbon", "thermal", "sla", "action", "completion"}
    assert d["reward"] == pytest.approx(sum(d["reward_parts"].values()), abs=0.11)   # + invalid penalty, if any
    s = client.post("/api/simulation/step", json={"minutes": 30}).json()
    assert s["minute"] == 31 and s["totals"]["energy_kwh"] > 0


def test_reset_is_reproducible(client):
    reset(client)
    a = client.post("/api/simulation/step", json={"minutes": 45}).json()
    reset(client)
    b = client.post("/api/simulation/step", json={"minutes": 45}).json()
    assert a["racks"] == b["racks"] and a["totals"] == b["totals"] and a["counters"] == b["counters"]


def test_history_and_workloads(client):
    reset(client)
    client.post("/api/simulation/step", json={"minutes": 20})
    h = client.get("/api/history").json()
    assert len(h["history"]) == 21 and len(h["decisions"]) == 20
    assert h["history"][-1]["minute"] == 20 and len(h["history"][-1]["temps"]) == 4
    w = client.get("/api/workloads").json()
    assert set(w) == {"queue", "running", "finished", "decisions", "counters"} and len(w["running"]) > 0
    assert {"id", "type", "cpu", "gpu", "slack", "status"} <= set(w["running"][0])


@pytest.mark.parametrize("spec", CONTROLLERS, ids=lambda s: s.key)
def test_every_controller_can_drive_the_twin(client, spec):
    if not controller_available(spec, get_config()):
        pytest.skip(f"models/{spec.model_dir} not trained")
    s = reset(client, controller=spec.key)
    assert s["controller"] == spec.key and s["controller_label"] == spec.label
    s = client.post("/api/simulation/step", json={"minutes": 60}).json()
    assert s["minute"] == 60 and all_finite(s)
    assert all(18.0 <= r["temperature"] <= 40.0 for r in s["racks"])


def test_controller_can_be_switched_mid_episode(client):
    reset(client)
    client.post("/api/simulation/step", json={"minutes": 10})
    r = client.post("/api/simulation/controller", json={"controller": "round_robin"})
    assert r.status_code == 200 and r.json()["controller"] == "round_robin" and r.json()["minute"] == 10
    s = client.post("/api/simulation/step", json={"minutes": 5}).json()
    assert s["minute"] == 15 and s["last_decision"]["controller"] == "round_robin"
    active = [c for c in client.get("/api/controllers").json() if c["active"]]
    assert [c["key"] for c in active] == ["round_robin"]


def test_unknown_controller_is_rejected(client):
    assert client.post("/api/simulation/controller", json={"controller": "oracle"}).status_code == 400
    assert client.post("/api/simulation/reset", json={"controller": "oracle"}).status_code == 400


def test_start_pause_and_speed(client):
    reset(client)
    assert client.post("/api/simulation/start").json()["running"] is True
    assert client.post("/api/simulation/speed", json={"speed": 12}).json()["speed"] == 12
    assert client.post("/api/simulation/pause").json()["running"] is False
    assert client.post("/api/simulation/speed", json={"speed": 0}).status_code == 422


def test_inject_workload(client):
    s = reset(client)
    before = len(s["queue"])
    r = client.post("/api/workload", json={"workload_type": "AI_TRAINING", "cpu": 30, "gpu": 80, "duration": 40,
                                           "slack": 120, "priority": 3}).json()
    assert r["accepted"] and r["job"]["id"] == "USER-001" and r["job"]["flexible"] and r["job"]["gpu"] == 80
    queue = client.get("/api/status").json()["queue"]
    assert len(queue) == before + 1 and queue[-1]["id"] == "USER-001"
    assert client.post("/api/workload", json={"cpu": 0, "gpu": 0}).status_code == 400
    assert client.post("/api/workload", json={"cpu": 500}).status_code == 422


def test_live_predictions(client):
    reset(client)
    p = client.get("/api/predictions").json()
    assert p["horizons"] == [5, 10, 15] and len(p["racks"]) == 4
    if p["available"]:
        assert all(len(r["forecast"]) == 3 for r in p["racks"])


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["racks"] == 4 and m["servers"] == 40 and m["gpu_nodes"] == 8
    assert len(m["grid_profile"]["hour"]) == len(m["grid_profile"]["carbon"]) == 97
    assert sum(m["reward_weights"].values()) == pytest.approx(1.0)


def test_comparison_serves_saved_results(client):
    c = client.get("/api/comparison").json()
    by_key = {e["key"]: e for e in c["controllers"]}
    assert {"random", "round_robin", "rule_based"} <= set(by_key)
    assert c["protocol"] == {"seeds": [42, 43, 44, 45, 46], "episode_minutes": 1440}
    saved = json.loads((get_config().paths.experiments / "baselines_5x1440min.json").read_text(encoding="utf-8"))
    assert by_key["round_robin"]["metrics"]["total_reward"] == saved["controllers"]["round_robin"]["summary"]["total_reward"]
    assert len(by_key["round_robin"]["per_seed"]) == 5 and all_finite(c)
    assert {m["key"] for m in c["metrics"]} >= {"total_reward", "energy_kwh", "sla_failures"}
    assert all(e["primary"] or e["note"] for e in c["controllers"])


def test_training_metrics(client):
    assert client.get("/api/training/sarsa/metrics").status_code == 404
    for algo in ("q_learning", "dqn", "ppo"):
        r = client.get(f"/api/training/{algo}/metrics")
        assert r.status_code == 200
        data = r.json()
        for run in data["runs"]:
            ep = run["episodes"]
            assert len(ep["total_reward"]) == len(ep["episode"]) <= 400 and all_finite(ep)


def test_prediction_summary(client):
    r = client.get("/api/predictions/summary")
    if r.status_code == 404:
        pytest.skip("forecaster not evaluated")
    p = r.json()
    assert set(p["test"]) == {"+5", "+10", "+15"} and p["model"]["architecture"] == "lstm"
    assert p["test"]["+5"]["model"]["mae"] < p["test"]["+5"]["persistence"]["mae"]


def test_websocket_sends_init_then_ticks(client):
    reset(client)
    with client.websocket_connect("/ws/simulation") as ws:
        init = ws.receive_json()
        assert init["type"] == "init" and init["state"]["minute"] == 0 and len(init["controllers"]) == len(CONTROLLERS)
        client.post("/api/simulation/step", json={"minutes": 1})
        tick = ws.receive_json()
        assert tick["type"] == "tick" and tick["state"]["minute"] == 1 and tick["point"]["minute"] == 1
        assert tick["decision"]["explanation"]


def test_episode_rolls_over_to_the_next_seed():
    service = SimulationService(get_config(), controller="round_robin", seed=7)
    service.env.max_steps = 3            # shorten the day for the test
    for _ in range(3):
        service.step()
    assert service.env.stats.steps == 3 and service.seed == 7
    service.step()
    assert service.seed == 8 and service.episode == 1 and service.env.dc.minute == 1


def test_built_frontend_is_served_when_present():
    dist = get_config().paths.experiments.parent / "frontend" / "dist"
    if not (dist / "index.html").exists():
        pytest.skip("frontend not built (npm run build)")
    with TestClient(create_app(autostart_loop=False)) as c:
        for path in ("/", "/twin", "/compare"):          # client-side routes fall back to index.html
            r = c.get(path)
            assert r.status_code == 200 and '<div id="root">' in r.text
        assert c.get("/api/status").json()["minute"] == 0   # API routes still win over the fallback
