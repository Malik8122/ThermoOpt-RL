"""Synthetic workload generation: arrival processes, job sampling, background jobs.

All distributions are SIMULATION ASSUMPTIONS configured in config.yaml ->
workloads. They are not fitted to measured traces.

Arrivals in a step of dt minutes:  N ~ Poisson(lambda(h) * dt)
    constant:      lambda(h) = mean_rate
    time_varying:  lambda(h) = mean_rate * m(h) / mean_day(m)
The normalisation makes both patterns produce the same expected jobs per day,
so they can be compared directly.

Jobs arriving during [t, t + dt) are released to the queue at the step
boundary t + dt; their deadline is computed from that release time, so even a
zero-slack job can still be started on time at the next decision.

Every random draw uses the generator's own ``numpy.random.Generator``, so the
job stream depends only on the seed and not on anything else in the program.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from backend.config import (
    ArrivalConfig,
    DurationConfig,
    ResourceDistributionConfig,
    WorkloadConfig,
    WorkloadTypeConfig,
)
from backend.simulation.enums import WorkloadType
from backend.simulation.profiles import PeriodicProfile
from backend.simulation.workload import MIN_PRIORITY, Workload


# --------------------------------------------------------------------------- #
# Arrival processes
# --------------------------------------------------------------------------- #
class ArrivalProcess(Protocol):
    def rate_at(self, hour: float) -> float:
        """Expected arrivals per minute at clock time ``hour``."""


class ConstantArrival:
    def __init__(self, rate_per_minute: float) -> None:
        self.rate_per_minute = rate_per_minute

    def rate_at(self, hour: float) -> float:
        return self.rate_per_minute


class TimeVaryingArrival:
    """Daily pattern, e.g. LOW early morning, MEDIUM daytime, HIGH afternoon peak."""

    def __init__(self, mean_rate_per_minute: float, profile: PeriodicProfile) -> None:
        self.mean_rate_per_minute = mean_rate_per_minute
        self.profile = profile
        self._profile_mean = profile.daily_mean()

    def rate_at(self, hour: float) -> float:
        return self.mean_rate_per_minute * self.profile(hour) / self._profile_mean


def build_arrival_process(cfg: ArrivalConfig) -> ArrivalProcess:
    if cfg.pattern == "constant":
        return ConstantArrival(cfg.mean_rate_per_minute)
    profile = PeriodicProfile([p[0] for p in cfg.profile], [p[1] for p in cfg.profile], cfg.interpolation)
    return TimeVaryingArrival(cfg.mean_rate_per_minute, profile)


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def sample_resource(dist: ResourceDistributionConfig, rng: np.random.Generator) -> int:
    """Integer resource units from a bounded distribution on [min, max]."""
    if dist.min == dist.max:
        return dist.min
    if dist.distribution == "uniform":
        value = rng.uniform(dist.min, dist.max)
    elif dist.distribution == "triangular":
        value = rng.triangular(dist.min, dist.mode, dist.max)
    else:  # beta, scaled to [min, max]
        value = dist.min + (dist.max - dist.min) * rng.beta(dist.a, dist.b)
    return int(np.clip(round(value), dist.min, dist.max))


def lognormal_parameters(mean: float, std: float) -> tuple[float, float]:
    """(mu, sigma) of the underlying normal so the log-normal has the given mean and std."""
    sigma_sq = math.log(1.0 + (std / mean) ** 2)
    return math.log(mean) - sigma_sq / 2.0, math.sqrt(sigma_sq)


def sample_duration(dist: DurationConfig, rng: np.random.Generator, timestep_minutes: int) -> int:
    """Duration in minutes: a whole number of timesteps, clipped to [min, max]."""
    if dist.distribution == "lognormal":
        mu, sigma = lognormal_parameters(dist.mean, dist.std)
        raw = rng.lognormal(mu, sigma)
    else:
        raw = rng.uniform(dist.min, dist.max)
    clipped = min(max(raw, dist.min), dist.max)
    steps = max(1, round(clipped / timestep_minutes))
    return steps * timestep_minutes


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class WorkloadGenerator:
    def __init__(self, cfg: WorkloadConfig, rng: np.random.Generator, timestep_minutes: int) -> None:
        self.cfg = cfg
        self.rng = rng
        self.dt = timestep_minutes
        self.arrivals = build_arrival_process(cfg.arrival)
        self._types = [WorkloadType(name) for name in cfg.types]
        self._type_probs = np.array([cfg.types[t].probability for t in self._types])
        self._priority_probs = np.asarray(cfg.priority_probabilities)
        self._counter = 0
        self._background_counter = 0

    def generate(self, now: int, hour: float) -> list[Workload]:
        """Jobs arriving during the step [now, now + dt), released at now + dt."""
        expected = self.arrivals.rate_at(hour) * self.dt
        count = int(self.rng.poisson(expected))
        return [self.sample_workload(release_time=now + self.dt) for _ in range(count)]

    def sample_workload(self, release_time: int, workload_type: WorkloadType | None = None) -> Workload:
        wtype = workload_type or self._types[self.rng.choice(len(self._types), p=self._type_probs)]
        tcfg: WorkloadTypeConfig = self.cfg.types[wtype.value]
        cpu = sample_resource(tcfg.cpu, self.rng)
        gpu = sample_resource(tcfg.gpu, self.rng)
        duration = sample_duration(tcfg.duration_minutes, self.rng, self.dt)
        flexible = bool(self.rng.random() < tcfg.flexible_probability)
        slack_low, slack_high = self.cfg.flexible_slack_minutes if flexible else self.cfg.strict_slack_minutes
        slack = int(round(self.rng.uniform(slack_low, slack_high)))
        priority = MIN_PRIORITY + int(self.rng.choice(len(self._priority_probs), p=self._priority_probs))
        self._counter += 1
        return Workload(
            job_id=f"J-{self._counter:05d}",
            workload_type=wtype,
            cpu_requirement=cpu,
            gpu_requirement=gpu,
            arrival_time=release_time,
            duration=duration,
            deadline=release_time + duration + slack,
            priority=priority,
            flexible=flexible,
            created_at=release_time,
        )

    def create_background_workloads(
        self, rack_index: int, cpu_units: int, gpu_units: int, rng: np.random.Generator, now: int = 0
    ) -> list[Workload]:
        """Jobs that are already running at reset and add up to exactly ``cpu_units`` / ``gpu_units``.

        GPU demand is filled first with GPU job types (chosen in proportion to
        their configured probabilities), then leftover CPU with NORMAL_CPU jobs.
        The last job of each kind is trimmed so the totals match exactly, which
        is why background jobs may fall below a type's minimum size. Each gets a
        remaining run time ~ U(background_remaining_minutes).

        Uses a separate ``rng`` so the arrival stream does not depend on the
        initial rack state.
        """
        gpu_types = [t for t in self._types if self.cfg.types[t.value].gpu.max > 0]
        gpu_probs = np.array([self.cfg.types[t.value].probability for t in gpu_types])
        gpu_probs = gpu_probs / gpu_probs.sum()
        jobs: list[Workload] = []
        cpu_left, gpu_left = cpu_units, gpu_units
        while gpu_left > 0:
            wtype = gpu_types[rng.choice(len(gpu_types), p=gpu_probs)]
            tcfg = self.cfg.types[wtype.value]
            gpu = min(gpu_left, sample_resource(tcfg.gpu, rng))
            cpu = min(cpu_left, sample_resource(tcfg.cpu, rng))
            jobs.append(self._background_job(rack_index, wtype, cpu, gpu, now, rng))
            cpu_left -= cpu
            gpu_left -= gpu
        cpu_type = WorkloadType.NORMAL_CPU
        while cpu_left > 0:
            cpu = min(cpu_left, sample_resource(self.cfg.types[cpu_type.value].cpu, rng))
            jobs.append(self._background_job(rack_index, cpu_type, cpu, 0, now, rng))
            cpu_left -= cpu
        return jobs

    def _background_job(
        self, rack_index: int, wtype: WorkloadType, cpu: int, gpu: int, now: int, rng: np.random.Generator
    ) -> Workload:
        low, high = self.cfg.background_remaining_minutes
        steps = max(1, round(rng.uniform(low, high) / self.dt))
        remaining = steps * self.dt
        self._background_counter += 1
        return Workload(
            job_id=f"BG-R{rack_index + 1}-{self._background_counter:03d}",
            workload_type=wtype,
            cpu_requirement=cpu,
            gpu_requirement=gpu,
            arrival_time=now,
            duration=remaining,
            deadline=now + remaining,
            priority=MIN_PRIORITY,
            flexible=False,
            created_at=now,
            is_background=True,
        )
