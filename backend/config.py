"""Typed, validated access to config.yaml.

Every module in the project reads its parameters through ``get_config()``
rather than hard-coding numbers, so an experiment is fully described by one
YAML file plus a seed.

Usage::

    from backend.config import get_config
    cfg = get_config()
    cfg.thermal.alpha                  # 0.035
    cfg.datacenter.total_cpu_capacity  # 400

Run ``python -m backend.config`` to print a summary of the active settings.
Set the ``THERMOOPT_CONFIG`` environment variable to load a different file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class _Strict(BaseModel):
    """Base model: unknown keys are an error, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascending(values: list[float], name: str) -> list[float]:
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly ascending, got {values}")
    return values


def _range_pair(pair: list[float], name: str) -> list[float]:
    if len(pair) != 2 or pair[0] > pair[1]:
        raise ValueError(f"{name} must be [low, high] with low <= high, got {pair}")
    return pair


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class ProjectConfig(_Strict):
    name: str
    phase: int
    seed: int


class PathsConfig(_Strict):
    data_dir: str
    models_dir: str
    experiments_dir: str

    @property
    def data(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def models(self) -> Path:
        return PROJECT_ROOT / self.models_dir

    @property
    def experiments(self) -> Path:
        return PROJECT_ROOT / self.experiments_dir


class SimulationConfig(_Strict):
    timestep_minutes: int = Field(gt=0)
    episode_minutes: int = Field(gt=0)
    train_episode_minutes: int = Field(gt=0)
    start_hour: float = Field(ge=0, lt=24)
    randomize_start_hour_in_training: bool

    @property
    def steps_per_episode(self) -> int:
        return self.episode_minutes // self.timestep_minutes

    @property
    def steps_per_train_episode(self) -> int:
        return self.train_episode_minutes // self.timestep_minutes


class DataCentreConfig(_Strict):
    num_racks: int = Field(gt=0)
    rack_names: list[str]
    servers_per_rack: int = Field(gt=0)
    gpu_nodes_per_rack: int = Field(ge=0)
    cpu_capacity_per_rack: int = Field(gt=0)
    gpu_capacity_per_rack: int = Field(gt=0)
    initial_temperatures_c: list[float]
    initial_cpu_utilisation_pct: list[float]
    initial_gpu_utilisation_pct: list[float]

    @model_validator(mode="after")
    def _per_rack_lengths(self) -> "DataCentreConfig":
        for name in ("rack_names", "initial_temperatures_c",
                     "initial_cpu_utilisation_pct", "initial_gpu_utilisation_pct"):
            if len(getattr(self, name)) != self.num_racks:
                raise ValueError(f"datacenter.{name} needs {self.num_racks} entries")
        for name in ("initial_cpu_utilisation_pct", "initial_gpu_utilisation_pct"):
            if any(not 0 <= u <= 100 for u in getattr(self, name)):
                raise ValueError(f"datacenter.{name} values must be within 0-100")
        return self

    @property
    def total_cpu_capacity(self) -> int:
        return self.num_racks * self.cpu_capacity_per_rack

    @property
    def total_gpu_capacity(self) -> int:
        return self.num_racks * self.gpu_capacity_per_rack

    @property
    def total_servers(self) -> int:
        return self.num_racks * self.servers_per_rack

    @property
    def total_gpu_nodes(self) -> int:
        return self.num_racks * self.gpu_nodes_per_rack


class UtilisationWeights(_Strict):
    cpu: float = Field(ge=0)
    gpu: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "UtilisationWeights":
        if abs(self.cpu + self.gpu - 1.0) > 1e-6:
            raise ValueError("thermal.utilisation_weights must sum to 1")
        return self


class ThermalZones(_Strict):
    warning_c: float
    critical_c: float

    @model_validator(mode="after")
    def _ordered(self) -> "ThermalZones":
        if self.warning_c >= self.critical_c:
            raise ValueError("thermal.zones.warning_c must be below critical_c")
        return self


class ThermalConfig(_Strict):
    """Thermal coefficients. ``alpha`` and ``beta`` are per *percentage point* per minute."""

    alpha: float = Field(ge=0)
    beta: float = Field(ge=0)
    ambient_coupling: float = Field(ge=0, le=1)
    ambient_temperature_c: float
    ambient_daily_amplitude_c: float = Field(ge=0)
    ambient_peak_hour: float = Field(ge=0, lt=24)
    utilisation_weights: UtilisationWeights
    rack_cooling_effectiveness: list[float]
    noise_std_c: float = Field(ge=0)
    min_temperature_c: float
    max_temperature_c: float
    zones: ThermalZones

    @model_validator(mode="after")
    def _bounds(self) -> "ThermalConfig":
        if self.min_temperature_c >= self.max_temperature_c:
            raise ValueError("thermal.min_temperature_c must be below max_temperature_c")
        if not self.min_temperature_c < self.zones.warning_c < self.zones.critical_c < self.max_temperature_c:
            raise ValueError("thermal zones must lie inside [min_temperature_c, max_temperature_c]")
        if any(not 0 < e <= 1 for e in self.rack_cooling_effectiveness):
            raise ValueError("thermal.rack_cooling_effectiveness values must be in (0, 1]")
        return self


class CoolingConfig(_Strict):
    levels_pct: list[float]
    initial_level_pct: float = Field(ge=0, le=100)
    max_cooling_power_kw: float = Field(gt=0)
    power_exponent: float = Field(gt=0)
    response_time_minutes: float = Field(ge=0)

    @field_validator("levels_pct")
    @classmethod
    def _levels(cls, v: list[float]) -> list[float]:
        _ascending(v, "cooling.levels_pct")
        if v[0] < 0 or v[-1] > 100:
            raise ValueError("cooling.levels_pct must lie within 0-100")
        return v


class PowerConfig(_Strict):
    server_idle_w: float = Field(ge=0)
    server_max_w: float = Field(gt=0)
    gpu_idle_w: float = Field(ge=0)
    gpu_max_w: float = Field(gt=0)
    utilisation_exponent: float = Field(gt=0)

    @model_validator(mode="after")
    def _idle_below_max(self) -> "PowerConfig":
        if self.server_idle_w > self.server_max_w or self.gpu_idle_w > self.gpu_max_w:
            raise ValueError("power: idle power must not exceed max power")
        return self


class ResourceDistributionConfig(_Strict):
    """Integer resource requirement drawn from a bounded distribution on [min, max]."""

    distribution: Literal["uniform", "triangular", "beta"]
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    mode: float | None = None    # triangular only
    a: float | None = Field(default=None, gt=0)  # beta only
    b: float | None = Field(default=None, gt=0)  # beta only

    @model_validator(mode="after")
    def _parameters(self) -> "ResourceDistributionConfig":
        if self.min > self.max:
            raise ValueError("resource distribution needs min <= max")
        if self.distribution == "triangular" and (self.mode is None or not self.min <= self.mode <= self.max):
            raise ValueError("triangular distribution needs min <= mode <= max")
        if self.distribution == "beta" and (self.a is None or self.b is None):
            raise ValueError("beta distribution needs a and b")
        return self


class DurationConfig(_Strict):
    distribution: Literal["lognormal", "uniform"]
    mean: float = Field(gt=0)
    std: float = Field(ge=0)
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "DurationConfig":
        if not self.min <= self.mean <= self.max:
            raise ValueError("duration_minutes must satisfy min <= mean <= max")
        return self


class WorkloadTypeConfig(_Strict):
    probability: float = Field(ge=0, le=1)
    cpu: ResourceDistributionConfig
    gpu: ResourceDistributionConfig
    duration_minutes: DurationConfig
    flexible_probability: float = Field(ge=0, le=1)


class ArrivalConfig(_Strict):
    pattern: Literal["constant", "time_varying"]
    mean_rate_per_minute: float = Field(ge=0)
    interpolation: Literal["pchip", "linear"]
    profile: list[tuple[float, float]]   # (hour, relative intensity)

    @field_validator("profile")
    @classmethod
    def _profile(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        hours = [p[0] for p in v]
        _ascending(hours, "workloads.arrival.profile hours")
        if len(v) < 2 or hours[0] < 0 or hours[-1] >= 24:
            raise ValueError("workloads.arrival.profile needs >= 2 anchors with hours in [0, 24)")
        if any(p[1] < 0 for p in v) or not any(p[1] > 0 for p in v):
            raise ValueError("workloads.arrival.profile intensities must be >= 0 and not all zero")
        return v


class WorkloadConfig(_Strict):
    arrival: ArrivalConfig
    max_queue_length: int = Field(gt=0)
    strict_slack_minutes: list[float]
    flexible_slack_minutes: list[float]
    priority_probabilities: list[float]
    background_remaining_minutes: list[float]
    types: dict[Literal["NORMAL_CPU", "GPU_INFERENCE", "AI_TRAINING"], WorkloadTypeConfig]

    @field_validator("strict_slack_minutes", "flexible_slack_minutes", "background_remaining_minutes")
    @classmethod
    def _pairs(cls, v: list[float]) -> list[float]:
        return _range_pair(v, "minutes range")

    @model_validator(mode="after")
    def _probabilities(self) -> "WorkloadConfig":
        if abs(sum(t.probability for t in self.types.values()) - 1.0) > 1e-6:
            raise ValueError("workloads.types probabilities must sum to 1")
        if abs(sum(self.priority_probabilities) - 1.0) > 1e-6:
            raise ValueError("workloads.priority_probabilities must sum to 1")
        if self.background_remaining_minutes[0] <= 0:
            raise ValueError("workloads.background_remaining_minutes must be positive")
        return self


class GridConfig(_Strict):
    interpolation: Literal["pchip", "linear"]
    profile: list[tuple[float, float, float]]  # (hour, price INR/kWh, carbon gCO2/kWh)
    high_carbon_threshold: float = Field(gt=0)

    @field_validator("profile")
    @classmethod
    def _profile(cls, v: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        if len(v) < 2:
            raise ValueError("grid.profile needs at least two anchor points")
        hours = [p[0] for p in v]
        _ascending(hours, "grid.profile hours")
        if hours[0] < 0 or hours[-1] >= 24:
            raise ValueError("grid.profile hours must lie within [0, 24)")
        if any(p[1] < 0 or p[2] < 0 for p in v):
            raise ValueError("grid.profile prices and carbon intensities must be non-negative")
        return v

    @property
    def max_price(self) -> float:
        return max(p[1] for p in self.profile)

    @property
    def max_carbon(self) -> float:
        return max(p[2] for p in self.profile)


class EnvironmentConfig(_Strict):
    invalid_action_penalty: float = Field(ge=0)
    queue_normalisation: int = Field(gt=0)
    carbon_lookahead_minutes: int = Field(gt=0)


class RewardWeights(_Strict):
    energy: float = Field(ge=0)
    carbon: float = Field(ge=0)
    thermal: float = Field(ge=0)
    sla: float = Field(ge=0)
    switching: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "RewardWeights":
        total = self.energy + self.carbon + self.thermal + self.sla + self.switching
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reward.weights must sum to 1, got {total:.4f}")
        return self


class ThermalRewardConfig(_Strict):
    warning_scale: float = Field(ge=0)
    critical_base: float = Field(ge=0)
    critical_slope: float = Field(ge=0)


class RewardConfig(_Strict):
    weights: RewardWeights
    completion_bonus: float = Field(ge=0)
    thermal: ThermalRewardConfig
    sla_violation_penalty: float = Field(ge=0)
    sla_delay_penalty: float = Field(ge=0)
    sla_normalisation: float = Field(gt=0)
    switching_normalisation: float = Field(gt=0)

    @model_validator(mode="after")
    def _delay_cheaper_than_violation(self) -> "RewardConfig":
        # A job that waits (within its deadline) must never cost as much as one that misses it.
        if self.sla_delay_penalty >= self.sla_violation_penalty:
            raise ValueError("reward.sla_delay_penalty must be smaller than reward.sla_violation_penalty")
        return self


class DiscretisationConfig(_Strict):
    temperature_edges_c: list[float]
    workload_edges_pct: list[float]
    cooling_edges_pct: list[float]
    carbon_edges: list[float]
    queue_edges: list[float]

    @field_validator("*")
    @classmethod
    def _edges(cls, v: list[float], info) -> list[float]:
        return _ascending(v, f"discretisation.{info.field_name}")

    @property
    def num_states(self) -> int:
        n = 1
        for edges in (self.temperature_edges_c, self.workload_edges_pct, self.cooling_edges_pct,
                      self.carbon_edges, self.queue_edges):
            n *= len(edges) + 1
        return n


class RoundRobinConfig(_Strict):
    cooling_pct: float = Field(ge=0, le=100)


class RuleBasedConfig(_Strict):
    avoid_rack_at_c: float
    cooling_increase_above_c: float
    cooling_decrease_below_c: float
    min_slack_to_delay_minutes: int = Field(ge=0)
    max_queue_to_delay: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "RuleBasedConfig":
        if self.cooling_decrease_below_c >= self.cooling_increase_above_c:
            raise ValueError("baselines.rule_based: cooling_decrease_below_c must be < cooling_increase_above_c")
        return self


class BaselinesConfig(_Strict):
    round_robin: RoundRobinConfig
    rule_based: RuleBasedConfig


class ValueIterationConfig(_Strict):
    gamma: float = Field(gt=0, lt=1)
    theta: float = Field(gt=0)
    max_iterations: int = Field(gt=0)
    tie_tolerance: float = Field(ge=0)
    model_dir: str                 # relative to the repository root

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / self.model_dir


class PlanningConfig(_Strict):
    """Model-based planning settings (Policy Iteration)."""

    gamma: float = Field(gt=0, lt=1)
    theta: float = Field(gt=0)
    max_iterations: int = Field(gt=0)
    model_samples_per_state_action: int = Field(gt=0)


class PolicyIterationConfig(PlanningConfig):
    max_evaluation_sweeps: int = Field(gt=0)
    tie_tolerance: float = Field(ge=0)


class TabularModelConfig(_Strict):
    """How the empirical model of the aggregated MDP is collected (shared by Policy and Value Iteration)."""

    collection_episodes: int = Field(gt=0)
    collection_minutes: int = Field(gt=0)
    snapshot_interval_minutes: int = Field(gt=0)
    max_snapshots_per_state: int = Field(gt=0)


class QLearningConfig(_Strict):
    learning_rate: float = Field(gt=0, le=1)
    gamma: float = Field(gt=0, lt=1)
    epsilon_start: float = Field(ge=0, le=1)
    epsilon_min: float = Field(ge=0, le=1)
    epsilon_decay: float = Field(gt=0, le=1)
    episodes: int = Field(gt=0)
    episode_minutes: int = Field(gt=0)
    randomize_start_hour: bool
    seed: int
    tie_tolerance: float = Field(ge=0)

    @model_validator(mode="after")
    def _epsilon_order(self) -> "QLearningConfig":
        if self.epsilon_min > self.epsilon_start:
            raise ValueError("rl.q_learning.epsilon_min must be <= epsilon_start")
        return self


class DQNConfig(_Strict):
    learning_rate: float = Field(gt=0)
    gamma: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=0)
    replay_buffer_size: int = Field(gt=0)
    target_update_frequency: int = Field(gt=0)
    epsilon_start: float = Field(ge=0, le=1)
    epsilon_end: float = Field(ge=0, le=1)
    epsilon_decay_steps: int = Field(gt=0)
    hidden_sizes: list[int]
    learning_starts: int = Field(ge=0)
    train_frequency: int = Field(gt=0)
    max_grad_norm: float = Field(gt=0)
    loss: Literal["huber", "mse"]
    episodes: int = Field(gt=0)
    episode_minutes: int = Field(gt=0)
    randomize_start_hour: bool
    use_action_mask: bool
    truncation_is_terminal: bool
    seed: int
    device: Literal["cpu", "cuda"]
    torch_threads: int = Field(gt=0)

    @model_validator(mode="after")
    def _checks(self) -> "DQNConfig":
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("rl.dqn.epsilon_end must be <= epsilon_start")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError("rl.dqn.batch_size must not exceed replay_buffer_size")
        return self


class PPOConfig(_Strict):
    learning_rate: float = Field(gt=0)
    gamma: float = Field(gt=0, lt=1)
    gae_lambda: float = Field(gt=0, le=1)
    clip_epsilon: float = Field(gt=0, lt=1)
    entropy_coefficient: float = Field(ge=0)
    value_coefficient: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    rollout_steps: int = Field(gt=0)
    update_epochs: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    hidden_sizes: list[int]
    total_timesteps: int = Field(gt=0)
    episode_minutes: int = Field(gt=0)
    randomize_start_hour: bool
    use_action_mask: bool
    normalize_advantages: bool
    truncation_is_terminal: bool
    seed: int
    device: Literal["cpu", "cuda"]
    torch_threads: int = Field(gt=0)

    @model_validator(mode="after")
    def _checks(self) -> "PPOConfig":
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("rl.ppo.minibatch_size must not exceed rollout_steps")
        if self.total_timesteps < self.rollout_steps:
            raise ValueError("rl.ppo.total_timesteps must cover at least one rollout")
        return self


class PredictivePPOConfig(_Strict):
    total_timesteps: int = Field(gt=0)


class RLConfig(_Strict):
    tabular_model: TabularModelConfig
    policy_iteration: PolicyIterationConfig
    value_iteration: ValueIterationConfig
    q_learning: QLearningConfig
    dqn: DQNConfig
    ppo: PPOConfig
    predictive_ppo: PredictivePPOConfig


class PredictionConfig(_Strict):
    architecture: Literal["lstm", "gru"]
    sequence_length: int = Field(gt=0)
    horizons_minutes: list[int]
    features: list[str]
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    dataset_episodes: int = Field(gt=0)
    split: list[float]
    behaviour_policies: list[Literal["random", "round_robin", "rule_based", "ppo", "dqn"]] = Field(min_length=1)
    seed: int
    torch_threads: int = Field(gt=0)

    @model_validator(mode="after")
    def _checks(self) -> "PredictionConfig":
        _ascending(self.horizons_minutes, "prediction.horizons_minutes")
        if "temperature" not in self.features:
            raise ValueError("prediction.features must include 'temperature'")
        if len(self.split) != 3 or abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError("prediction.split must be three fractions summing to 1")
        return self


class EvaluationConfig(_Strict):
    seeds: list[int]
    episode_minutes: int = Field(gt=0)


class APIConfig(_Strict):
    host: str
    port: int = Field(gt=0, lt=65536)
    cors_origins: list[str]
    websocket_tick_ms: int = Field(gt=0)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class Config(_Strict):
    project: ProjectConfig
    paths: PathsConfig
    simulation: SimulationConfig
    datacenter: DataCentreConfig
    thermal: ThermalConfig
    cooling: CoolingConfig
    power: PowerConfig
    workloads: WorkloadConfig
    grid: GridConfig
    environment: EnvironmentConfig
    reward: RewardConfig
    discretisation: DiscretisationConfig
    baselines: BaselinesConfig
    rl: RLConfig
    prediction: PredictionConfig
    evaluation: EvaluationConfig
    api: APIConfig

    @model_validator(mode="after")
    def _cross_section(self) -> "Config":
        if len(self.thermal.rack_cooling_effectiveness) != self.datacenter.num_racks:
            raise ValueError("thermal.rack_cooling_effectiveness needs one entry per rack")
        if self.thermal.ambient_coupling * self.simulation.timestep_minutes >= 1:
            # explicit Euler on the ambient term would overshoot / oscillate
            raise ValueError("thermal.ambient_coupling * timestep_minutes must be < 1 for a stable update")
        levels = self.cooling.levels_pct
        if not levels[0] <= self.cooling.initial_level_pct <= levels[-1]:
            raise ValueError("cooling.initial_level_pct must lie within cooling.levels_pct")
        if self.baselines.round_robin.cooling_pct not in levels:
            raise ValueError("baselines.round_robin.cooling_pct must be one of cooling.levels_pct")
        for name, sim_minutes in (("episode_minutes", self.simulation.episode_minutes),
                                  ("train_episode_minutes", self.simulation.train_episode_minutes)):
            if sim_minutes % self.simulation.timestep_minutes:
                raise ValueError(f"simulation.{name} must be a multiple of timestep_minutes")
        if any(h % self.simulation.timestep_minutes for h in self.prediction.horizons_minutes):
            raise ValueError("prediction.horizons_minutes must be multiples of timestep_minutes")
        return self

    # Derived quantities used across the project -------------------------- #
    @property
    def max_it_power_kw(self) -> float:
        dc, p = self.datacenter, self.power
        per_rack_w = dc.servers_per_rack * p.server_max_w + dc.gpu_nodes_per_rack * p.gpu_max_w
        return dc.num_racks * per_rack_w / 1000.0

    @property
    def idle_it_power_kw(self) -> float:
        dc, p = self.datacenter, self.power
        per_rack_w = dc.servers_per_rack * p.server_idle_w + dc.gpu_nodes_per_rack * p.gpu_idle_w
        return dc.num_racks * per_rack_w / 1000.0

    @property
    def max_total_power_kw(self) -> float:
        return self.max_it_power_kw + self.cooling.max_cooling_power_kw

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load and validate a config file.

    ``overrides`` is a nested dict merged on top of the file, which is handy in
    tests and experiments, e.g. ``{"project": {"seed": 7}}``.
    """
    path = Path(path or os.environ.get("THERMOOPT_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached config. Call ``get_config.cache_clear()`` to reload."""
    return load_config()


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _summary(cfg: Config) -> str:
    dc, th, co = cfg.datacenter, cfg.thermal, cfg.cooling
    lines = [
        f"{cfg.project.name} - Phase {cfg.project.phase} configuration (seed={cfg.project.seed})",
        "",
        f"Racks                : {dc.num_racks} x ({dc.servers_per_rack} servers + {dc.gpu_nodes_per_rack} GPU nodes)",
        f"Capacity             : CPU {dc.total_cpu_capacity:.0f} units, GPU {dc.total_gpu_capacity:.0f} units",
        f"IT power range       : {cfg.idle_it_power_kw:.1f} - {cfg.max_it_power_kw:.1f} kW",
        f"Cooling              : max {co.max_cooling_power_kw:.1f} kW (scaled-down), levels {co.levels_pct}, "
        f"start {co.initial_level_pct:.0f}%, response {co.response_time_minutes:g} min",
        f"Thermal              : alpha={th.alpha}, beta={th.beta}, kappa={th.ambient_coupling}, "
        f"ambient={th.ambient_temperature_c} +/- {th.ambient_daily_amplitude_c} C",
        f"Zones                : NORMAL < {th.zones.warning_c} <= WARNING < {th.zones.critical_c} <= CRITICAL",
        f"Episode              : {cfg.simulation.steps_per_episode} steps x {cfg.simulation.timestep_minutes} min "
        f"(training: {cfg.simulation.steps_per_train_episode} steps)",
        f"Workload types       : " + ", ".join(f"{k} ({t.probability:.0%})" for k, t in cfg.workloads.types.items()),
        f"Arrivals             : {cfg.workloads.arrival.pattern}, "
        f"mean {cfg.workloads.arrival.mean_rate_per_minute} jobs/min",
        f"Reward weights       : {cfg.reward.weights.model_dump()}",
        f"Tabular state space  : {cfg.discretisation.num_states} discrete states",
        f"Paths                : data={cfg.paths.data}, models={cfg.paths.models}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary(get_config()))
