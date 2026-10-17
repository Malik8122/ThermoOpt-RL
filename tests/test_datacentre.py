"""Steps 2-4 integrated: the DataCentre step loop, accounting and reproducibility."""

import numpy as np
import pytest

from backend.config import load_config
from backend.simulation.datacentre import DataCentre
from backend.simulation.enums import FailureReason, WorkloadStatus
from backend.simulation.grid import GridSignals
from backend.simulation.run_baseline import first_fit_dispatch, run, write_csv
from tests.conftest import make_job


@pytest.fixture
def quiet_cfg():
    """No random arrivals, no noise: only the jobs a test submits."""
    return load_config(overrides={
        "workloads": {"arrival": {"mean_rate_per_minute": 0.0}},
        "thermal": {"noise_std_c": 0.0},
    })


def test_job_occupies_resources_for_exactly_its_duration(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    rack = dc.racks[3]
    base_cpu = rack.get_cpu_utilisation()
    job = make_job("X", cpu=30, duration=5)
    dc.submit_workload(job)
    assert rack.get_available_cpu() == 57                 # QUEUED: holds nothing
    assert dc.place_workload("X", 3).success
    loaded_steps = 0
    for _ in range(8):
        record = dc.step()
        loaded_steps += record.rack_cpu_utilisation[3] >= base_cpu + 0.30 - 1e-9
    assert loaded_steps == 5
    assert job.status is WorkloadStatus.COMPLETED and (job.started_at, job.completed_at) == (0, 5)
    assert "X" not in rack.active_workload_ids


def test_completed_jobs_release_resources(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    total = lambda: sum(r.get_available_cpu() + r.get_available_gpu() for r in dc.racks)  # noqa: E731
    for _ in range(200):                                   # every background job finishes within 120 min
        dc.step()
    assert total() == 800
    assert all(w.status is WorkloadStatus.COMPLETED for w in dc.workloads.values())


def test_energy_carbon_and_cost_equations(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    grid = GridSignals(quiet_cfg.grid)
    records = [dc.step() for _ in range(30)]
    for r in records:
        hour = 8.0 + r.minute / 60
        assert r.it_energy_kwh == pytest.approx(r.it_power_kw / 60)
        assert r.cooling_energy_kwh == pytest.approx(r.cooling_power_kw / 60)
        assert r.total_energy_kwh == pytest.approx(r.total_power_kw / 60)
        assert r.carbon_g == pytest.approx(r.total_energy_kwh * grid.carbon_at(hour))
        assert r.cost_inr == pytest.approx(r.total_energy_kwh * grid.price_at(hour))
        assert r.total_power_kw == pytest.approx(r.it_power_kw + r.cooling_power_kw)
        assert r.it_power_kw == pytest.approx(r.server_power_kw + r.gpu_power_kw)
    last = records[-1]
    assert last.cumulative_total_energy_kwh == pytest.approx(sum(r.total_energy_kwh for r in records))
    assert last.cumulative_carbon_g == pytest.approx(sum(r.carbon_g for r in records))
    assert last.cumulative_cost_inr == pytest.approx(sum(r.cost_inr for r in records))


def test_known_energy_value_for_fixed_cooling(quiet_cfg):
    """65 % cooling for 60 min = 10.4 kW * 1 h = 10.4 kWh."""
    dc = DataCentre(quiet_cfg)
    records = [dc.step() for _ in range(60)]
    assert records[-1].cumulative_cooling_energy_kwh == pytest.approx(10.4)


def test_cooling_power_is_not_reduced_by_rack_effectiveness(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    dc.set_cooling_target(0.8)
    for _ in range(3):
        record = dc.step()
    assert record.cooling_power_kw == pytest.approx(0.8 * 16.0)
    assert record.rack_effective_cooling == pytest.approx((0.8, 0.8, 0.6, 0.8))
    assert [r.cooling_power_kw for r in dc.racks] == pytest.approx([0.8 * 16.0 / 4] * 4)


def test_waiting_job_fails_when_deadline_can_no_longer_be_met(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    job = make_job("S", cpu=10, duration=10, slack=3)     # latest start = minute 3
    dc.submit_workload(job)
    for _ in range(3):
        dc.step()
    assert job.is_pending
    dc.step()
    assert job.status is WorkloadStatus.FAILED and job.failure_reason is FailureReason.DEADLINE_MISSED
    assert job not in dc.queue and job.started_at is None


def test_queue_overflow_is_rejected(quiet_cfg):
    dc = DataCentre(load_config(overrides={"workloads": {"arrival": {"mean_rate_per_minute": 0.0},
                                                         "max_queue_length": 2}}))
    results = [dc.submit_workload(make_job(f"Q{i}")) for i in range(3)]
    assert results == [True, True, False]
    assert dc.workloads["Q2"].failure_reason is FailureReason.QUEUE_FULL


def test_delay_moves_job_to_back_of_queue(quiet_cfg):
    dc = DataCentre(quiet_cfg)
    for i in range(3):
        dc.submit_workload(make_job(f"D{i}"))
    dc.delay_workload("D0")
    assert [w.job_id for w in dc.queue] == ["D1", "D2", "D0"]
    assert dc.workloads["D0"].status is WorkloadStatus.DELAYED


def test_full_day_keeps_invariants_and_physical_bounds(cfg):
    """24 h with random arrivals: check_invariants() runs every step; all values stay physical."""
    records = run(cfg, 1440, seed=42)
    temps = np.array([r.rack_temperature_c for r in records])
    cpu = np.array([r.rack_cpu_utilisation for r in records])
    gpu = np.array([r.rack_gpu_utilisation for r in records])
    assert temps.min() >= 18.0 and temps.max() <= 40.0
    assert cpu.min() >= 0 and cpu.max() <= 1 and gpu.min() >= 0 and gpu.max() <= 1
    assert min(r.total_power_kw for r in records) > 0
    assert all(0 <= r.cooling_level <= 1 for r in records)


def test_seed_42_run_is_reproducible(cfg, tmp_path):
    digest_a = write_csv(run(cfg, 60, seed=42), tmp_path / "a.csv")
    digest_b = write_csv(run(cfg, 60, seed=42), tmp_path / "b.csv")
    digest_c = write_csv(run(cfg, 60, seed=43), tmp_path / "c.csv")
    assert digest_a == digest_b
    assert digest_a != digest_c


def test_first_fit_dispatch_never_overallocates(cfg):
    dc = DataCentre(cfg)
    for _ in range(50):
        dc.submit_workload(dc.generator.sample_workload(release_time=dc.minute))
    first_fit_dispatch(dc)
    for rack in dc.racks:
        assert 0 <= rack.get_available_cpu() <= rack.cpu_capacity
        assert 0 <= rack.get_available_gpu() <= rack.gpu_capacity
    dc.check_invariants()
