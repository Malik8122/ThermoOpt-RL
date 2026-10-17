"""Baseline controllers (Step 6): Random, Round-robin and Rule-based.

Every controller, baseline or learned, exposes the same interface (PHASE1_DESIGN §2):

    controller.reset(seed)                          once per episode
    action = controller.act(observation, mask)      once per step

``observation`` is the feature observation of ``DataCenterEnv`` (30 features, or 42 with
forecasts; only the first 30 are read here) and ``action`` is an index in the 18-action
primitive space. ``mask`` is ``env.action_masks()``; only the random controller needs it.
The baselines read nothing but the observation, so the evaluation harness and the live
dashboard drive them exactly like an RL agent.

    random       uniform over the valid actions (all actions if no mask is given)
    round-robin  cycle through the racks from where the last placement left off, skip racks the
                 head job doesn't fit on, delay it if it fits nowhere; hold cooling at a fixed level
    rule-based   1. carbon-aware: delay a flexible head job while carbon >= grid.high_carbon_threshold
                    and the job can wait (at least ``min_slack_to_delay_minutes`` of slack left and
                    no more than ``max_queue_to_delay`` jobs waiting, so the delay starves no one)
                 2. thermal-aware: place on the least-utilised rack below ``avoid_rack_at_c``;
                    if only hot racks fit, delay while the job can wait, else use the coolest one
                 3. cooling: +1 level if any rack > ``cooling_increase_above_c``,
                    -1 level if every rack < ``cooling_decrease_below_c``, else maintain

Every threshold is in ``config.yaml → baselines``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from backend.config import Config, get_config
from backend.rl.actions import ActionCodec, CoolingCommand
from backend.rl.observations import ObservationBuilder
from backend.utils.seeding import derive_seed, make_rng
from backend.utils.units import pct_to_fraction

_LEVEL_TOLERANCE = 1e-4   # observations are float32


class Controller(ABC):
    """A policy over the primitive (18-action) space."""

    name: str = "controller"

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.codec = ActionCodec(self.config.datacenter.num_racks)

    def reset(self, seed: int | None = None) -> None:
        """Clear per-episode state. Called by the runner before every episode."""

    @abstractmethod
    def act(self, observation: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        ...


# --------------------------------------------------------------------------- #
# Reading the observation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObservedState:
    """The parts of the feature observation a baseline needs, in physical units."""

    rack_temperature_c: np.ndarray
    rack_utilisation: np.ndarray       # capacity-weighted CPU + GPU, 0-1
    cooling_setpoint: float            # 0-1
    job_waiting: bool
    head_flexible: bool
    head_slack_minutes: float          # lower bound: exact below the urgency horizon, else the horizon
    head_fits: np.ndarray              # bool per rack
    queue_length: float                # jobs waiting (saturates at environment.queue_normalisation)
    carbon_g_per_kwh: float


class ObservationReader:
    """Inverts the normalisation of ``ObservationBuilder`` for the base features."""

    def __init__(self, config: Config) -> None:
        builder = ObservationBuilder(config)
        self.index = {name: i for i, name in enumerate(builder.feature_names)}
        self.n = config.datacenter.num_racks
        self.t_min, self.t_span = builder.t_min, builder.t_span
        self.max_carbon = builder.max_carbon
        self.urgency_horizon = builder.urgency_horizon
        self.queue_norm = builder.queue_norm
        dc = config.datacenter
        self.cpu_share = dc.cpu_capacity_per_rack / (dc.cpu_capacity_per_rack + dc.gpu_capacity_per_rack)

    def _racks(self, obs: np.ndarray, feature: str) -> np.ndarray:
        start = self.index[f"rack1_{feature}"]
        return np.asarray(obs[start:start + self.n], dtype=float)

    def read(self, obs: np.ndarray) -> ObservedState:
        f = lambda name: float(obs[self.index[name]])  # noqa: E731
        start = self.index["head_fits_rack1"]
        cpu, gpu = self._racks(obs, "cpu"), self._racks(obs, "gpu")
        return ObservedState(
            rack_temperature_c=self.t_min + self._racks(obs, "temperature") * self.t_span,
            rack_utilisation=self.cpu_share * cpu + (1.0 - self.cpu_share) * gpu,
            cooling_setpoint=f("cooling_setpoint"),
            job_waiting=f("job_waiting") > 0.5,
            head_flexible=f("head_flexible") > 0.5,
            head_slack_minutes=(1.0 - f("head_urgency")) * self.urgency_horizon,
            head_fits=np.asarray(obs[start:start + self.n]) > 0.5,
            carbon_g_per_kwh=f("carbon_intensity") * self.max_carbon,
            queue_length=f("queue_length") * self.queue_norm,
        )


def _cooling_levels(config: Config) -> list[float]:
    return sorted(pct_to_fraction(p) for p in config.cooling.levels_pct)


def _can_increase(levels: list[float], setpoint: float) -> bool:
    return setpoint < levels[-1] - _LEVEL_TOLERANCE


def _can_decrease(levels: list[float], setpoint: float) -> bool:
    return setpoint > levels[0] + _LEVEL_TOLERANCE


# --------------------------------------------------------------------------- #
# Controllers
# --------------------------------------------------------------------------- #
class RandomController(Controller):
    """Uniform over the valid actions; uniform over all actions if no mask is given."""

    name = "random"

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        self.rng = make_rng(derive_seed(self.config.project.seed, "random-controller"))

    def reset(self, seed: int | None = None) -> None:
        base = self.config.project.seed if seed is None else seed
        self.rng = make_rng(derive_seed(base, "random-controller"))

    def act(self, observation: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        if action_mask is None or not action_mask.any():
            return int(self.rng.integers(self.codec.n_actions))
        return int(self.rng.choice(np.flatnonzero(action_mask)))


class RoundRobinController(Controller):
    """Cycle through the racks, skipping full ones; hold cooling at a fixed level."""

    name = "round_robin"

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        self.reader = ObservationReader(self.config)
        self.levels = _cooling_levels(self.config)
        self.cooling_target = pct_to_fraction(self.config.baselines.round_robin.cooling_pct)
        self.next_rack = 0

    def reset(self, seed: int | None = None) -> None:
        self.next_rack = 0

    def act(self, observation: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        s = self.reader.read(observation)
        placement = self.codec.delay_index   # also the no-op when the queue is empty
        if s.job_waiting:
            n = self.codec.num_racks
            for offset in range(n):
                rack = (self.next_rack + offset) % n
                if s.head_fits[rack]:
                    placement, self.next_rack = rack, (rack + 1) % n
                    break
        return self.codec.encode(placement, self._cooling(s.cooling_setpoint))

    def _cooling(self, setpoint: float) -> CoolingCommand:
        """Step the set-point towards the fixed target (one level per minute), then hold it."""
        if setpoint > self.cooling_target + _LEVEL_TOLERANCE:
            return CoolingCommand.DECREASE
        if setpoint < self.cooling_target - _LEVEL_TOLERANCE:
            return CoolingCommand.INCREASE
        return CoolingCommand.MAINTAIN


class RuleBasedController(Controller):
    """Hand-written heuristic: carbon-aware delay, thermal-aware placement, threshold cooling."""

    name = "rule_based"

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config)
        self.reader = ObservationReader(self.config)
        self.levels = _cooling_levels(self.config)
        self.rules = self.config.baselines.rule_based
        self.high_carbon = self.config.grid.high_carbon_threshold

    def act(self, observation: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        s = self.reader.read(observation)
        return self.codec.encode(self._placement(s), self._cooling(s))

    def _can_wait(self, s: ObservedState) -> bool:
        """A deliberate delay must leave the head job enough slack and must not hold up other jobs.

        Delaying uses the step's only placement action and sends the head job to the back of the
        queue, so with a long queue it starves the jobs behind it (strict jobs have 0-10 min of slack).
        """
        return (s.head_slack_minutes >= self.rules.min_slack_to_delay_minutes - _LEVEL_TOLERANCE
                and s.queue_length <= self.rules.max_queue_to_delay + _LEVEL_TOLERANCE)

    def _placement(self, s: ObservedState) -> int:
        delay = self.codec.delay_index
        if not s.job_waiting or not s.head_fits.any():
            return delay
        if s.head_flexible and s.carbon_g_per_kwh >= self.high_carbon and self._can_wait(s):
            return delay
        fits = np.flatnonzero(s.head_fits)
        cool = [r for r in fits if s.rack_temperature_c[r] < self.rules.avoid_rack_at_c]
        if cool:
            return int(min(cool, key=lambda r: (s.rack_utilisation[r], r)))
        if self._can_wait(s):
            return delay
        return int(min(fits, key=lambda r: (s.rack_temperature_c[r], r)))

    def _cooling(self, s: ObservedState) -> CoolingCommand:
        temps = s.rack_temperature_c
        if temps.max() > self.rules.cooling_increase_above_c and _can_increase(self.levels, s.cooling_setpoint):
            return CoolingCommand.INCREASE
        if temps.max() < self.rules.cooling_decrease_below_c and _can_decrease(self.levels, s.cooling_setpoint):
            return CoolingCommand.DECREASE
        return CoolingCommand.MAINTAIN


BASELINES: dict[str, type[Controller]] = {
    RandomController.name: RandomController,
    RoundRobinController.name: RoundRobinController,
    RuleBasedController.name: RuleBasedController,
}


def make_baseline(name: str, config: Config | None = None) -> Controller:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    return BASELINES[name](config)
