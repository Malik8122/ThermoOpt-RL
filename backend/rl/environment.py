"""Gymnasium interface to the ThermoOpt digital twin.

``DataCenterEnv`` contains no physics. Each ``step()``:

  1. decodes the action (tabular macro-actions are first resolved to a primitive action),
  2. validates it against the current simulator state (read-only, atomic),
  3. issues the commands to ``DataCentre`` if valid (place / delay / migrate, cooling target),
  4. advances the simulator one timestep with ``DataCentre.step()``,
  5. reads the simulator's outcome (energy, carbon, temperatures, job events) and scores it with
     ``calculate_reward``.

Two representations share this class:
    representation="features"  Box(30,) or Box(42,) with forecasts    Discrete(18)   (DQN, PPO)
    representation="tabular"   Discrete(144)                          Discrete(9)    (PI, VI, Q-Learning)

Episodes never terminate (the data centre keeps running); they are *truncated*
after ``episode_minutes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.config import Config, get_config
from backend.rl.actions import (
    N_TABULAR_ACTIONS,
    ActionCheck,
    ActionCodec,
    AppliedAction,
    PrimitiveAction,
    apply_action,
    resolve_tabular_action,
    validate_action,
)
from backend.rl.observations import ObservationBuilder, PredictionProvider
from backend.rl.reward import RewardBreakdown, RewardInputs, calculate_reward
from backend.rl.state_encoder import TabularStateEncoder
from backend.simulation.datacentre import DataCentre, StepRecord
from backend.simulation.enums import TemperatureStatus, WorkloadStatus
from backend.utils.units import MINUTES_PER_HOUR, fraction_to_pct, g_to_kg

Representation = Literal["features", "tabular"]


@dataclass
class EpisodeStats:
    steps: int = 0
    total_reward: float = 0.0
    invalid_actions: int = 0
    warning_or_worse_rack_minutes: int = 0
    critical_rack_minutes: int = 0
    peak_temperature_c: float = float("-inf")
    jobs_arrived: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    migrations: int = 0
    cooling_changes: int = 0
    reward_by_component: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**self.__dict__, "reward_by_component": dict(self.reward_by_component)}


class DataCenterEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        representation: Representation = "features",
        episode_minutes: int | None = None,
        start_hour: float | None = None,
        randomize_start_hour: bool = False,
        prediction_provider: PredictionProvider | None = None,
        render_mode: str | None = None,
        config: Config | None = None,
        validate_simulator: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or get_config()
        if representation not in ("features", "tabular"):
            raise ValueError(f"representation must be 'features' or 'tabular', got {representation!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if representation == "tabular" and prediction_provider is not None:
            raise ValueError("forecasts are only used by the feature representation")
        self.representation = representation
        self.render_mode = render_mode
        self.dt = self.config.simulation.timestep_minutes
        self.episode_minutes = episode_minutes or self.config.simulation.episode_minutes
        if self.episode_minutes <= 0 or self.episode_minutes % self.dt:
            raise ValueError("episode_minutes must be a positive multiple of the timestep")
        self.max_steps = self.episode_minutes // self.dt
        self.start_hour = start_hour
        self.randomize_start_hour = randomize_start_hour
        self.validate_simulator = validate_simulator

        self.codec = ActionCodec(self.config.datacenter.num_racks)
        self.state_encoder = TabularStateEncoder(self.config.discretisation)
        self.observation_builder = ObservationBuilder(self.config, prediction_provider)

        if representation == "tabular":
            self.action_space = spaces.Discrete(N_TABULAR_ACTIONS)
            self.observation_space = spaces.Discrete(self.state_encoder.n_states)
        else:
            self.action_space = spaces.Discrete(self.codec.n_actions)
            self.observation_space = spaces.Box(
                low=self.observation_builder.low, high=self.observation_builder.high, dtype=np.float32
            )

        self.dc: DataCentre | None = None
        self.stats = EpisodeStats()
        self.last_reward: RewardBreakdown | None = None
        self.last_record: StepRecord | None = None
        self.last_info: dict[str, Any] = {}
        self.episode_seed: int | None = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        # With an explicit seed the twin uses it directly, so env.reset(seed=42) reproduces the
        # seed-42 simulator run. Without one, a seed is drawn from the env's seeded RNG.
        self.episode_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.dc = DataCentre(self.config, seed=self.episode_seed, start_hour=self._choose_start_hour(options),
                             validate=self.validate_simulator)
        self.stats = EpisodeStats()
        self.last_reward, self.last_record = None, None
        info = {
            "timestamp": self.dc.clock,
            "step": 0,
            "episode_seed": self.episode_seed,
            "start_hour": self.dc.start_hour,
            "episode_minutes": self.episode_minutes,
            "representation": self.representation,
            "action_mask": self.action_masks(),
        }
        self.last_info = info
        return self._observe(), info

    def step(self, action):
        if self.dc is None:
            raise RuntimeError("call reset() before step()")
        if self.stats.steps >= self.max_steps:
            raise RuntimeError("episode is over; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is not in {self.action_space}")
        action = int(action)

        check, tabular_reason = self._check(action)
        applied = apply_action(self.dc, check) if check is not None and check.valid else AppliedAction()
        valid = check is not None and check.valid
        reason = tabular_reason.value if tabular_reason else (check.reason_text if check else None)

        # Snapshot job sets so events of this step can be attributed without touching the simulator.
        running_before = {jid for rack in self.dc.racks for jid in rack.active_workload_ids}
        pending_before = {w.job_id for w in self.dc.queue}
        known_before = set(self.dc.workloads)

        record = self.dc.step()

        completed = [self.dc.workloads[j] for j in running_before
                     if self.dc.workloads[j].status is WorkloadStatus.COMPLETED]
        candidates = pending_before | (set(self.dc.workloads) - known_before)
        failed = [self.dc.workloads[j] for j in candidates if self.dc.workloads[j].status is WorkloadStatus.FAILED]

        reward = calculate_reward(RewardInputs(
            total_energy_kwh=record.total_energy_kwh,
            carbon_g=record.carbon_g,
            timestep_minutes=self.dt,
            rack_temperatures_c=record.rack_temperature_c,
            jobs_failed=len(failed),
            started_wait_fractions=(applied.started_wait_fraction,) if applied.started_job else (),
            completed_priorities=tuple(w.priority for w in completed if not w.is_background),
            cooling_changed=applied.cooling_changed,
            migrations=int(applied.migrated),
            invalid_action=not valid,
        ), self.config)

        self._update_stats(record, reward, valid, applied, len(completed), len(failed))
        self.last_reward, self.last_record = reward, record
        truncated = self.stats.steps >= self.max_steps
        info = self._step_info(action, check, valid, reason, applied, record, reward, completed, failed)
        self.last_info = info
        if self.render_mode == "human":
            print(self.render_text())
        return self._observe(), float(reward.total), False, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self.render_text()
        if self.render_mode == "human":
            print(self.render_text())
        return None

    def close(self) -> None:
        self.dc = None

    # ----------------------------------------------------------------- helpers
    def _choose_start_hour(self, options: dict) -> float:
        if options.get("start_hour") is not None:
            return float(options["start_hour"])
        if self.randomize_start_hour:
            # whole minutes, drawn from the env's seeded RNG
            return float(self.np_random.integers(0, 24 * int(MINUTES_PER_HOUR))) / MINUTES_PER_HOUR
        return self.start_hour if self.start_hour is not None else self.config.simulation.start_hour

    def _check(self, action: int):
        """Return (ActionCheck | None, tabular failure reason | None) for an action index."""
        if self.representation == "tabular":
            primitive, reason = resolve_tabular_action(self.dc, self.codec, action)
            if primitive is None:
                return None, reason
            return validate_action(self.dc, primitive), None
        return validate_action(self.dc, self.codec.decode(action)), None

    def action_masks(self) -> np.ndarray:
        """Boolean mask of the actions that are valid in the current state (read-only)."""
        if self.dc is None:
            raise RuntimeError("call reset() first")
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            check, _ = self._check(a)
            mask[a] = check is not None and check.valid
        return mask

    def decode_action(self, action: int) -> PrimitiveAction:
        """Primitive action for an index in the *feature* action space."""
        return self.codec.decode(action)

    def _observe(self):
        if self.representation == "tabular":
            return self.state_encoder.encode(self.state_encoder.observe(self.dc))
        return self.observation_builder.build(self.dc)

    def _update_stats(self, record: StepRecord, reward: RewardBreakdown, valid: bool, applied: AppliedAction,
                      n_completed: int, n_failed: int) -> None:
        s = self.stats
        s.steps += 1
        s.total_reward += reward.total
        s.invalid_actions += int(not valid)
        s.warning_or_worse_rack_minutes += sum(st is not TemperatureStatus.NORMAL for st in record.rack_status)
        s.critical_rack_minutes += sum(st is TemperatureStatus.CRITICAL for st in record.rack_status)
        s.peak_temperature_c = max(s.peak_temperature_c, max(record.rack_temperature_c))
        s.jobs_arrived += record.arrivals
        s.jobs_completed += n_completed
        s.jobs_failed += n_failed
        s.migrations += int(applied.migrated)
        s.cooling_changes += int(applied.cooling_changed)
        for c in reward.components:
            s.reward_by_component[c.name] = s.reward_by_component.get(c.name, 0.0) + c.contribution

    def _step_info(self, action: int, check: ActionCheck | None, valid: bool, reason: str | None,
                   applied: AppliedAction, record: StepRecord, reward: RewardBreakdown,
                   completed: list, failed: list) -> dict[str, Any]:
        temps = record.rack_temperature_c
        return {
            "timestamp": self.dc.clock,                 # simulated clock after the step
            "interval_start": record.clock,
            "step": self.stats.steps,
            "action": action,
            "primitive_action": self.codec.encode(check.action.placement, check.action.cooling) if check else None,
            "action_description": check.action.describe() if check else "no feasible rack for tabular placement",
            "action_valid": valid,
            "invalid_action_reason": None if valid else reason,
            "placement_effect": check.placement_effect if check else "none",
            "job_id": check.job_id if check and valid else None,
            "migration": check.migration.__dict__ if check and check.migration else None,
            "cooling_target_pct": fraction_to_pct(record.cooling_target),
            "cooling_delivered_pct": fraction_to_pct(record.cooling_level),

            "reward_total": reward.total,
            "reward_energy": reward.energy.contribution,
            "reward_carbon": reward.carbon.contribution,
            "reward_thermal": reward.thermal.contribution,
            "reward_sla": reward.sla.contribution,
            "reward_action": reward.action.contribution,
            "reward_completion": reward.completion.contribution,

            "energy_kwh": record.total_energy_kwh,
            "it_energy_kwh": record.it_energy_kwh,
            "cooling_energy_kwh": record.cooling_energy_kwh,
            "carbon_kg": g_to_kg(record.carbon_g),
            "electricity_cost": record.cost_inr,
            "carbon_intensity": record.carbon_g_per_kwh,
            "electricity_price": record.price_inr_per_kwh,
            "it_power_kw": record.it_power_kw,
            "cooling_power_kw": record.cooling_power_kw,
            "total_power_kw": record.total_power_kw,

            "rack_temperatures": list(temps),
            "rack_status": [str(s) for s in record.rack_status],
            "avg_temperature": float(np.mean(temps)),
            "peak_temperature": float(max(temps)),
            "thermal_violations": sum(s is not TemperatureStatus.NORMAL for s in record.rack_status),
            "critical_violations": sum(s is TemperatureStatus.CRITICAL for s in record.rack_status),

            "jobs_arrived": record.arrivals,
            "jobs_completed": len(completed),
            "jobs_failed": len(failed),
            "active_jobs": record.running,
            "queue_length": record.queue_length,
            "failed_job_ids": sorted(w.job_id for w in failed),

            "episode": self.stats.to_dict(),
        }

    # ------------------------------------------------------------- explanation
    def explain_reward(self) -> str:
        if self.last_reward is None:
            return "No step taken yet."
        return self.last_reward.explain()

    def render_text(self) -> str:
        if self.dc is None:
            return "Environment not reset."
        dc, rec, rw = self.dc, self.last_record, self.last_reward
        lines = [f"Step: {self.stats.steps}/{self.max_steps}", f"Time: {dc.clock}", "", "Rack temperatures:"]
        lines += [f"  R{i + 1}: {r.temperature_c:5.2f} C  {r.get_temperature_status()}" for i, r in enumerate(dc.racks)]
        lines += ["CPU:"] + [f"  R{i + 1}: {fraction_to_pct(r.get_cpu_utilisation()):5.1f}%" for i, r in enumerate(dc.racks)]
        lines += ["GPU:"] + [f"  R{i + 1}: {fraction_to_pct(r.get_gpu_utilisation()):5.1f}%" for i, r in enumerate(dc.racks)]
        lines += ["", f"Cooling: target {fraction_to_pct(dc.cooling.target_level):.0f}%, "
                      f"delivered {fraction_to_pct(dc.cooling.current_level):.1f}%",
                  f"Queue: {len(dc.queue)} waiting, {sum(len(r.active_workload_ids) for r in dc.racks)} running"]
        if rec is not None and rw is not None:
            t = dc.accountant.totals
            lines += [
                f"Power: IT {rec.it_power_kw:.2f} kW, cooling {rec.cooling_power_kw:.2f} kW",
                f"Energy: {t.total_kwh:.3f} kWh   Carbon: {g_to_kg(t.carbon_g):.3f} kg   Cost: INR {t.cost_inr:.2f}",
                f"Last action: {self.last_info.get('action_description')} "
                f"({'valid' if self.last_info.get('action_valid') else 'INVALID: ' + str(self.last_info.get('invalid_action_reason'))})",
                "", "Reward:",
            ]
            lines += [f"  {c.name.capitalize():<11}{c.contribution:+.4f}" for c in rw.components]
            lines += [f"  {'Total':<11}{rw.total:+.4f}", f"Episode return: {self.stats.total_reward:+.3f}"]
        return "\n".join(lines)
