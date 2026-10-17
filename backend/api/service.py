"""Live simulation service for the dashboard (one twin, one active controller).

The service owns a feature-representation ``DataCenterEnv`` and steps it one simulated minute at a time. Any
controller can drive it:

  * feature controllers (random, round-robin, rule-based, DQN, PPO) receive the 30-feature observation;
  * tabular controllers (Policy / Value Iteration, Q-Learning) receive the 144-state id computed from the same
    simulator, and their macro-action is resolved to a primitive action exactly as the tabular environment does.

Everything the frontend shows is read from the simulator or from a controller's actual decision. The
"explanation" of a decision is assembled from those facts; it never guesses a controller's motives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.config import Config, get_config
from backend.rl.actions import (
    N_TABULAR_ACTIONS,
    CoolingCommand,
    PlacementKind,
    decode_tabular_action,
    resolve_tabular_action,
    validate_action,
)
from backend.rl.baselines import Controller, make_baseline
from backend.rl.environment import DataCenterEnv
from backend.simulation.enums import WorkloadStatus, WorkloadType
from backend.simulation.workload import Workload
from backend.utils.units import fraction_to_pct

HISTORY_MINUTES = 360
DECISIONS_KEPT = 80
FINISHED_KEPT = 40


@dataclass(frozen=True)
class ControllerSpec:
    key: str
    label: str
    family: str            # baseline | planning | tabular-rl | deep-rl
    representation: str    # features | tabular
    description: str
    model_dir: str | None  # under models/; None = needs no trained model
    step: int              # project step that produced it


CONTROLLERS: tuple[ControllerSpec, ...] = (
    ControllerSpec("rule_based", "Rule-based", "baseline", "features",
                   "Hand-written rules: carbon-aware delay, coolest-rack placement, threshold cooling.", None, 6),
    ControllerSpec("round_robin", "Round-robin", "baseline", "features",
                   "Cycles through the racks; cooling held at 60 %.", None, 6),
    ControllerSpec("random", "Random", "baseline", "features",
                   "Uniform over the currently valid actions.", None, 6),
    ControllerSpec("policy_iteration", "Policy Iteration", "planning", "tabular",
                   "Exact dynamic programming on an empirical 144-state model.", "policy_iteration", 7),
    ControllerSpec("value_iteration", "Value Iteration", "planning", "tabular",
                   "Bellman-optimality sweeps on the same 144-state model.", "value_iteration", 8),
    ControllerSpec("q_learning", "Q-Learning", "tabular-rl", "tabular",
                   "Model-free tabular learning from 1000 simulated days.", "q_learning", 9),
    ControllerSpec("dqn", "DQN", "deep-rl", "features",
                   "Neural Q-network on 30 features with action masking.", "dqn", 10),
    ControllerSpec("ppo", "PPO", "deep-rl", "features",
                   "Actor-critic policy on 30 features, sampled from its action distribution.", "ppo", 11),
)
SPEC = {c.key: c for c in CONTROLLERS}


def controller_available(spec: ControllerSpec, config: Config) -> bool:
    return spec.model_dir is None or (config.paths.models / spec.model_dir).exists()


def build_controller(spec: ControllerSpec, config: Config) -> Controller:
    models = config.paths.models
    if spec.key in ("random", "round_robin", "rule_based"):
        return make_baseline(spec.key, config)
    if spec.key == "policy_iteration":
        from backend.rl.policy_iteration import PolicyIterationController, load
        return PolicyIterationController(load(models / "policy_iteration").policy, config)
    if spec.key == "value_iteration":
        from backend.rl.value_iteration import ValueIterationController, load
        return ValueIterationController(load(models / "value_iteration").policy, config)
    if spec.key == "q_learning":
        from backend.rl.q_learning import QLearning, QLearningController
        return QLearningController(QLearning.load(models / "q_learning"), config)
    if spec.key == "dqn":
        from backend.rl.dqn import DQNAgent, DQNController
        return DQNController(DQNAgent.load(models / "dqn"), config)
    if spec.key == "ppo":
        from backend.rl.ppo import PPOAgent, PPOController
        return PPOController(PPOAgent.load(models / "ppo"), config, deterministic=False)
    raise ValueError(f"unknown controller {spec.key!r}")


class SimulationService:
    def __init__(self, config: Config | None = None, controller: str = "rule_based", seed: int | None = None) -> None:
        self.config = config or get_config()
        self.env = DataCenterEnv(representation="features", config=self.config)
        self.forecaster = self._load_forecaster()
        self._controllers: dict[str, Controller] = {}
        self.running = False
        self.speed = 4.0                      # simulated minutes per real second
        self.episode = 0
        self.user_jobs = 0
        self.listeners: list[Callable[[dict], None]] = []
        self.controller_key = controller
        self.reset(seed=self.config.project.seed if seed is None else seed, controller=controller)

    # ------------------------------------------------------------------ setup
    def _load_forecaster(self):
        directory = self.config.paths.models / "temperature_predictor"
        if not directory.exists():
            return None
        from backend.prediction.model import ForecastProvider, TemperatureForecaster
        return ForecastProvider(TemperatureForecaster.load(directory))

    def controllers(self) -> list[dict]:
        return [{**spec.__dict__, "available": controller_available(spec, self.config),
                 "active": spec.key == self.controller_key} for spec in CONTROLLERS]

    def _controller(self, key: str) -> Controller:
        if key not in SPEC:
            raise ValueError(f"unknown controller {key!r}")
        if not controller_available(SPEC[key], self.config):
            raise ValueError(f"controller {key!r} has no trained model in models/{SPEC[key].model_dir}")
        if key not in self._controllers:
            self._controllers[key] = build_controller(SPEC[key], self.config)
        return self._controllers[key]

    def set_controller(self, key: str) -> None:
        controller = self._controller(key)
        controller.reset(self.seed)
        self.controller_key = key

    def reset(self, seed: int | None = None, controller: str | None = None, start_hour: float | None = None) -> dict:
        if seed is not None:
            self.seed = int(seed)
        options = {"start_hour": start_hour} if start_hour is not None else None
        self.observation, _ = self.env.reset(seed=self.seed, options=options)
        self.set_controller(controller or self.controller_key)
        if self.forecaster is not None:
            self.forecaster.reset()
        self.history: deque[dict] = deque(maxlen=HISTORY_MINUTES)
        self.decisions: deque[dict] = deque(maxlen=DECISIONS_KEPT)
        self.finished: deque[dict] = deque(maxlen=FINISHED_KEPT)
        self.last_info: dict = {}
        self.last_decision: dict | None = None
        self.forecast = self._forecast()
        self.history.append(self._point(reward=None))
        return self.state()

    # --------------------------------------------------------------- stepping
    def _tabular_mask(self) -> np.ndarray:
        dc, codec = self.env.dc, self.env.codec
        mask = np.zeros(N_TABULAR_ACTIONS, dtype=bool)
        for a in range(N_TABULAR_ACTIONS):
            primitive, _ = resolve_tabular_action(dc, codec, a)
            mask[a] = primitive is not None and validate_action(dc, primitive).valid
        return mask

    def _choose_action(self) -> int:
        """Primitive action index chosen by the active controller for the current state."""
        env, controller = self.env, self._controller(self.controller_key)
        if SPEC[self.controller_key].representation == "features":
            return int(controller.act(self.observation, env.action_masks()))
        state = env.state_encoder.encode(env.state_encoder.observe(env.dc))
        macro = int(controller.act(state, self._tabular_mask()))
        primitive, _ = resolve_tabular_action(env.dc, env.codec, macro)
        if primitive is None:
            # A placement macro with no feasible rack: issue the placement on rack 1, which the environment
            # rejects as invalid (a no-op plus the invalid-action penalty), the same outcome as the tabular env.
            _, cooling = decode_tabular_action(macro)
            return env.codec.encode(0, cooling)
        return env.codec.encode(primitive.placement, primitive.cooling)

    def step(self) -> dict:
        """Advance one simulated minute. Returns the tick message sent to WebSocket clients."""
        if self.env.stats.steps >= self.env.max_steps:
            self.episode += 1
            self.seed += 1
            self.reset(seed=self.seed)
        before = self._facts_before()
        action = self._choose_action()
        self.observation, reward, _, truncated, info = self.env.step(action)
        self.last_info = info
        self.forecast = self._forecast()
        decision = self._decision(action, info, before, reward)
        self.last_decision = decision
        self.decisions.appendleft(decision)
        for job_id in info["failed_job_ids"]:
            self._record_finished(job_id)
        for job in self.env.dc.workloads.values():
            if job.status is WorkloadStatus.COMPLETED and job.completed_at == self.env.dc.minute:
                self._record_finished(job.job_id)
        point = self._point(reward)
        self.history.append(point)
        return {"type": "tick", "state": self.state(), "point": point, "decision": decision,
                "episode_done": bool(truncated)}

    def step_many(self, minutes: int) -> dict:
        message = {}
        for _ in range(max(1, min(int(minutes), 240))):
            message = self.step()
        return message

    # ------------------------------------------------------------- injection
    def inject_workload(self, workload_type: str, cpu: int, gpu: int, duration: int, slack: int, priority: int) -> dict:
        dc = self.env.dc
        self.user_jobs += 1
        job = Workload(job_id=f"USER-{self.user_jobs:03d}", workload_type=WorkloadType(workload_type),
                       cpu_requirement=int(cpu), gpu_requirement=int(gpu), arrival_time=dc.minute,
                       duration=int(duration), deadline=dc.minute + int(duration) + int(slack),
                       priority=int(priority), flexible=int(slack) >= 60, created_at=dc.minute)
        accepted = dc.submit_workload(job)
        self.observation = self.env.observation_builder.build(dc)
        return {"accepted": bool(accepted), "job": self._job(job)}

    # ----------------------------------------------------------------- views
    def _forecast(self) -> list[list[float]] | None:
        if self.forecaster is None:
            return None
        return np.round(self.forecaster(self.env.dc), 3).tolist()

    def _job(self, job: Workload) -> dict:
        now = self.env.dc.minute
        return {"id": job.job_id, "type": job.workload_type.value, "cpu": job.cpu_requirement,
                "gpu": job.gpu_requirement, "priority": job.priority, "flexible": job.flexible,
                "status": job.status.value, "duration": job.duration, "remaining": job.remaining_time,
                "arrival": job.arrival_time, "deadline": job.deadline, "slack": job.slack_at(now),
                "rack": job.assigned_rack, "delays": job.delay_count, "migrations": job.migration_count,
                "background": job.is_background, "started_at": job.started_at, "completed_at": job.completed_at,
                "failure_reason": job.failure_reason.value if job.failure_reason else None}

    def _record_finished(self, job_id: str) -> None:
        job = self.env.dc.workloads[job_id]
        if not any(f["id"] == job_id for f in self.finished):
            self.finished.appendleft(self._job(job))

    def _facts_before(self) -> dict:
        dc = self.env.dc
        head = dc.queue[0] if dc.queue else None
        return {"head": self._job(head) if head else None, "temps": dc.temperatures().tolist(),
                "cooling_target": dc.cooling.target_level, "carbon": dc.grid.carbon_at(dc.hour_of_day),
                "forecast": self.forecast}

    def _decision(self, action: int, info: dict, before: dict, reward: float) -> dict:
        """A decision record with an explanation built only from simulator facts."""
        primitive = self.env.codec.decode(action)
        zones = self.config.thermal.zones
        temps = before["temps"]
        hottest = int(np.argmax(temps))
        parts: list[str] = []
        head = before["head"]
        effect = info["placement_effect"]
        if not info["action_valid"]:
            parts.append(f"Rejected: {str(info['invalid_action_reason']).replace('_', ' ').lower()}.")
        elif effect == "place" and head:
            r = primitive.rack_index
            order = sorted(range(len(temps)), key=lambda i: temps[i])
            rank = order.index(r)
            note = "the coolest rack" if rank == 0 else ("the hottest rack" if rank == len(temps) - 1 else
                                                         f"{rank + 1}{'nd' if rank == 1 else 'rd'} coolest rack")
            parts.append(f"Placed {head['id']} ({head['cpu']} CPU / {head['gpu']} GPU) on Rack {r + 1}, "
                         f"{note} at {temps[r]:.1f} °C.")
        elif effect == "delay" and head:
            kind = "flexible" if head["flexible"] else "strict"
            parts.append(f"Delayed {head['id']} ({kind}, {head['slack']} min of slack left) while grid carbon is "
                         f"{before['carbon']:.0f} g/kWh.")
        elif effect == "migrate" and info["migration"]:
            m = info["migration"]
            parts.append(f"Migrated {m['job_id']} from Rack {m['source_rack'] + 1} ({temps[m['source_rack']]:.1f} °C) "
                         f"to Rack {m['target_rack'] + 1} ({temps[m['target_rack']]:.1f} °C).")
        elif effect == "no_job":
            parts.append("Queue empty: no placement to make.")
        if info["action_valid"] and primitive.cooling is not CoolingCommand.MAINTAIN:
            verb = "Raised" if primitive.cooling is CoolingCommand.INCREASE else "Lowered"
            status = ("critical" if temps[hottest] >= zones.critical_c else
                      "warning" if temps[hottest] >= zones.warning_c else "normal")
            text = (f"{verb} cooling {fraction_to_pct(before['cooling_target']):.0f} % → "
                    f"{info['cooling_target_pct']:.0f} % with Rack {hottest + 1} at {temps[hottest]:.1f} °C ({status})")
            if before["forecast"]:
                text += f"; its +5 min forecast was {before['forecast'][hottest][0]:.1f} °C"
            parts.append(text + ".")
        return {
            "minute": self.env.dc.minute, "clock": info["interval_start"], "action": action,
            "description": info["action_description"], "kind": primitive.kind.value,
            "cooling": primitive.cooling.name, "rack": primitive.rack_index if primitive.kind is PlacementKind.PLACE else None,
            "valid": info["action_valid"], "invalid_reason": info["invalid_action_reason"],
            "explanation": " ".join(parts) or "Held everything as it was.",
            "reward": reward,
            "reward_parts": {k: info[f"reward_{k}"] for k in ("energy", "carbon", "thermal", "sla", "action", "completion")},
            "controller": self.controller_key,
        }

    def _point(self, reward: float | None) -> dict:
        dc, info = self.env.dc, self.last_info
        return {"minute": dc.minute, "clock": dc.clock, "temps": np.round(dc.temperatures(), 3).tolist(),
                "cooling_target": dc.cooling.target_level, "cooling_delivered": dc.cooling.current_level,
                "it_kw": info.get("it_power_kw", sum(r.get_power_consumption().it_kw for r in dc.racks)),
                "cooling_kw": info.get("cooling_power_kw", dc.cooling.power_kw()),
                "carbon": dc.grid.carbon_at(dc.hour_of_day), "price": dc.grid.price_at(dc.hour_of_day),
                "ambient": dc.ambient.temperature_at(dc.hour_of_day), "queue": len(dc.queue),
                "reward": reward, "forecast": self.forecast}

    def state(self) -> dict:
        dc = self.env.dc
        snap = dc.snapshot()
        zones = self.config.thermal.zones
        racks = []
        for r, raw in zip(dc.racks, snap["racks"]):
            racks.append({"id": r.rack_id, "name": r.name, "temperature": raw["temperature_c"],
                          "status": str(raw["temperature_status"]), "cpu": raw["cpu_utilisation"],
                          "gpu": raw["gpu_utilisation"], "it_kw": raw["it_power_kw"],
                          "cooling_received": raw["effective_cooling"],
                          "cooling_effectiveness": raw["cooling_effectiveness"],
                          "jobs": [self._job(w) for w in r.workloads],
                          "forecast": self.forecast[r.rack_id] if self.forecast else None})
        it_kw = sum(r["it_kw"] for r in racks)
        cooling_kw = snap["cooling"]["power_kw"]
        totals = snap["totals"]
        stats = self.env.stats
        return {
            "minute": dc.minute, "clock": dc.clock, "hour": dc.hour_of_day, "episode": self.episode,
            "seed": self.seed, "episode_minutes": self.env.episode_minutes, "running": self.running,
            "speed": self.speed, "controller": self.controller_key,
            "controller_label": SPEC[self.controller_key].label,
            "racks": racks,
            "cooling": {"target": snap["cooling"]["target_level"], "delivered": snap["cooling"]["current_level"],
                        "progress": snap["cooling"]["progress"], "power_kw": cooling_kw},
            "grid": {"carbon": snap["carbon_g_per_kwh"], "price": snap["price_inr_per_kwh"]},
            "ambient": snap["ambient_c"],
            "power": {"it_kw": it_kw, "cooling_kw": cooling_kw, "total_kw": it_kw + cooling_kw,
                      "pue": (it_kw + cooling_kw) / it_kw if it_kw > 0 else None},
            "totals": {"energy_kwh": totals["total_kwh"], "it_kwh": totals["it_kwh"],
                       "cooling_kwh": totals["cooling_kwh"], "carbon_kg": totals["carbon_g"] / 1000.0,
                       "cost_inr": totals["cost_inr"]},
            "queue": [self._job(w) for w in dc.queue],
            "finished": list(self.finished),
            "counters": {"arrived": stats.jobs_arrived, "completed": stats.jobs_completed, "failed": stats.jobs_failed,
                         "migrations": stats.migrations, "cooling_changes": stats.cooling_changes,
                         "invalid_actions": stats.invalid_actions, "steps": stats.steps,
                         "critical_rack_minutes": stats.critical_rack_minutes,
                         "warning_rack_minutes": stats.warning_or_worse_rack_minutes,
                         "total_reward": stats.total_reward,
                         "peak_temperature": stats.peak_temperature_c if stats.steps else None},
            "zones": {"warning": zones.warning_c, "critical": zones.critical_c},
            "horizons": list(self.config.prediction.horizons_minutes),
            "last_decision": self.last_decision,
        }

    def init_message(self) -> dict:
        return {"type": "init", "state": self.state(), "history": list(self.history),
                "decisions": list(self.decisions), "controllers": self.controllers()}
