"""Step 3: workload model, state machine and generator statistics."""

import numpy as np
import pytest

from backend.config import load_config
from backend.simulation.enums import FailureReason, WorkloadStateError, WorkloadStatus, WorkloadType
from backend.simulation.workload_generator import (
    ConstantArrival,
    WorkloadGenerator,
    build_arrival_process,
    lognormal_parameters,
)
from backend.utils.seeding import make_rng
from tests.conftest import make_job

N_SAMPLES = 20_000


@pytest.fixture(scope="module")
def samples(cfg):
    gen = WorkloadGenerator(cfg.workloads, make_rng(42), timestep_minutes=1)
    return [gen.sample_workload(release_time=0) for _ in range(N_SAMPLES)]


def test_generator_is_reproducible_for_a_seed(cfg):
    def stream(seed):
        gen = WorkloadGenerator(cfg.workloads, make_rng(seed), 1)
        return [(w.workload_type, w.cpu_requirement, w.gpu_requirement, w.duration, w.deadline)
                for minute in range(300) for w in gen.generate(minute, 8 + minute / 60)]
    assert stream(42) == stream(42)
    assert stream(42) != stream(43)


def test_type_mix_matches_configured_probabilities(samples):
    counts = {t: sum(w.workload_type is t for w in samples) / N_SAMPLES for t in WorkloadType}
    # binomial std at n=20k is <= 0.0036; allow ~4 sigma
    assert counts[WorkloadType.NORMAL_CPU] == pytest.approx(0.50, abs=0.015)
    assert counts[WorkloadType.GPU_INFERENCE] == pytest.approx(0.30, abs=0.015)
    assert counts[WorkloadType.AI_TRAINING] == pytest.approx(0.20, abs=0.015)


def test_resource_requirements_respect_type_ranges(cfg, samples):
    for w in samples:
        tcfg = cfg.workloads.types[w.workload_type.value]
        assert tcfg.cpu.min <= w.cpu_requirement <= tcfg.cpu.max
        assert tcfg.gpu.min <= w.gpu_requirement <= tcfg.gpu.max
        assert isinstance(w.cpu_requirement, int) and isinstance(w.gpu_requirement, int)
    assert all(w.gpu_requirement == 0 for w in samples if w.workload_type is WorkloadType.NORMAL_CPU)


def test_resource_distributions_are_not_uniform(samples):
    """NORMAL_CPU CPU is triangular(10, 20, 40): mean (10+20+40)/3 = 23.3, well below the uniform mean 25."""
    cpu = np.array([w.cpu_requirement for w in samples if w.workload_type is WorkloadType.NORMAL_CPU])
    assert cpu.mean() == pytest.approx(70 / 3, abs=0.4)
    assert np.mean(cpu < 25) > 0.55


def test_durations_are_whole_minutes_within_bounds(cfg, samples):
    for w in samples:
        d = cfg.workloads.types[w.workload_type.value].duration_minutes
        assert d.min <= w.duration <= d.max and isinstance(w.duration, int)
    training = np.array([w.duration for w in samples if w.workload_type is WorkloadType.AI_TRAINING])
    assert training.mean() == pytest.approx(45, rel=0.05)    # clipping barely moves the mean


def test_lognormal_parameterisation_reproduces_mean_and_std():
    mu, sigma = lognormal_parameters(30.0, 12.0)
    draws = np.random.default_rng(1).lognormal(mu, sigma, 200_000)
    assert draws.mean() == pytest.approx(30.0, rel=0.01)
    assert draws.std() == pytest.approx(12.0, rel=0.02)


def test_deadline_slack_depends_on_flexibility(cfg, samples):
    for w in samples:
        slack = w.deadline - w.arrival_time - w.duration
        low, high = cfg.workloads.flexible_slack_minutes if w.flexible else cfg.workloads.strict_slack_minutes
        assert low <= slack <= high
    flex_share = np.mean([w.flexible for w in samples if w.workload_type is WorkloadType.AI_TRAINING])
    assert flex_share == pytest.approx(0.7, abs=0.03)


def test_priorities_follow_configured_distribution(samples):
    shares = [np.mean([w.priority == p for w in samples]) for p in (1, 2, 3)]
    assert shares == pytest.approx([0.5, 0.35, 0.15], abs=0.015)


def test_time_varying_arrivals(cfg):
    process = build_arrival_process(cfg.workloads.arrival)
    hours = np.arange(0, 24, 1 / 60)
    rates = np.array([process.rate_at(h) for h in hours])
    assert rates.mean() == pytest.approx(0.30, rel=1e-3)          # same daily volume as constant
    assert process.rate_at(4.0) < process.rate_at(10.0) < process.rate_at(14.0)   # LOW < MEDIUM < HIGH
    gen = WorkloadGenerator(cfg.workloads, make_rng(7), 1)
    early = sum(len(gen.generate(0, 4.0)) for _ in range(5000))
    peak = sum(len(gen.generate(0, 14.0)) for _ in range(5000))
    assert peak > 2.5 * early


def test_constant_arrival_ignores_time_of_day(cfg):
    constant = load_config(overrides={"workloads": {"arrival": {"pattern": "constant"}}})
    process = build_arrival_process(constant.workloads.arrival)
    assert isinstance(process, ConstantArrival)
    assert {process.rate_at(h) for h in (0, 4, 14, 22)} == {0.30}


def test_arrivals_are_released_at_the_step_boundary(cfg):
    gen = WorkloadGenerator(cfg.workloads, make_rng(3), 1)
    jobs = [w for _ in range(200) for w in gen.generate(100, 14.0)]
    assert jobs and all(w.arrival_time == 101 and w.can_meet_deadline(101) for w in jobs)


def test_legal_status_transitions():
    job = make_job(duration=2)
    job.mark_delayed()
    assert job.status is WorkloadStatus.DELAYED and job.delay_count == 1
    job.mark_running(rack_index=0, now=5)
    job.advance(1)
    with pytest.raises(WorkloadStateError):
        job.mark_completed(6)                      # one minute of work still remaining
    job.advance(1)
    job.mark_completed(7)
    assert (job.status, job.started_at, job.completed_at) == (WorkloadStatus.COMPLETED, 5, 7)


@pytest.mark.parametrize("action", ["start", "delay", "fail", "advance"])
def test_completed_job_cannot_be_reused(action):
    job = make_job(duration=1)
    job.mark_running(0, 0)
    job.advance(1)
    job.mark_completed(1)
    with pytest.raises(WorkloadStateError):
        {"start": lambda: job.mark_running(1, 2), "delay": job.mark_delayed,
         "fail": lambda: job.mark_failed(FailureReason.DEADLINE_MISSED, 2), "advance": lambda: job.advance(1)}[action]()


def test_invalid_workloads_are_rejected():
    with pytest.raises(ValueError):
        make_job(cpu=0, gpu=0)
    with pytest.raises(ValueError):
        make_job(duration=10, slack=-1)            # deadline before earliest finish
