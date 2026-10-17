"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import pytest

from backend.config import Config, get_config, load_config
from backend.simulation.enums import WorkloadType
from backend.simulation.workload import Workload


@pytest.fixture(scope="session")
def cfg() -> Config:
    return get_config()


@pytest.fixture
def noiseless_cfg() -> Config:
    """Default config with thermal noise off and a constant 24 °C ambient, for exact equation checks."""
    return load_config(overrides={"thermal": {"noise_std_c": 0.0, "ambient_daily_amplitude_c": 0.0}})


def make_job(
    job_id: str = "T-1",
    cpu: int = 10,
    gpu: int = 0,
    duration: int = 5,
    arrival: int = 0,
    slack: int = 100,
    workload_type: WorkloadType = WorkloadType.NORMAL_CPU,
) -> Workload:
    return Workload(
        job_id=job_id,
        workload_type=workload_type,
        cpu_requirement=cpu,
        gpu_requirement=gpu,
        arrival_time=arrival,
        duration=duration,
        deadline=arrival + duration + slack,
        priority=2,
        flexible=slack > 10,
        created_at=arrival,
    )
