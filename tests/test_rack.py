"""Step 2: rack resource accounting, power aggregation and validation."""

import pytest

from backend.simulation.datacentre import DataCentre
from backend.simulation.enums import AssignmentFailure, SimulationInvariantError, WorkloadStatus
from tests.conftest import make_job


@pytest.fixture
def dc(cfg) -> DataCentre:
    return DataCentre(cfg)


@pytest.fixture
def empty_rack(dc):
    rack = dc.racks[3]
    for jid in rack.active_workload_ids:
        rack.remove_workload(jid)
    return rack


def test_background_jobs_reproduce_initial_state_exactly(dc, cfg):
    dcfg = cfg.datacenter
    for i, rack in enumerate(dc.racks):
        assert rack.get_cpu_utilisation() == pytest.approx(dcfg.initial_cpu_utilisation_pct[i] / 100)
        assert rack.get_gpu_utilisation() == pytest.approx(dcfg.initial_gpu_utilisation_pct[i] / 100)
        assert rack.temperature_c == dcfg.initial_temperatures_c[i]
        assert all(dc.workloads[j].status is WorkloadStatus.RUNNING for j in rack.active_workload_ids)
    assert [r.cooling_effectiveness for r in dc.racks] == [1.0, 1.0, 0.75, 1.0]
    assert sum(len(r.servers) for r in dc.racks) == 40 and sum(len(r.gpu_nodes) for r in dc.racks) == 8


def test_assignment_respects_cpu_capacity(empty_rack):
    assert empty_rack.assign_workload(make_job("a", cpu=60), 0).success
    result = empty_rack.assign_workload(make_job("b", cpu=41), 0)
    assert not result.success and result.reason is AssignmentFailure.INSUFFICIENT_CPU
    assert empty_rack.get_available_cpu() == 40           # nothing over-allocated
    assert empty_rack.assign_workload(make_job("c", cpu=40), 0).success   # exact fit is allowed
    assert empty_rack.get_available_cpu() == 0


def test_assignment_respects_gpu_capacity(empty_rack):
    assert empty_rack.assign_workload(make_job("a", cpu=10, gpu=70), 0).success
    job = make_job("b", cpu=10, gpu=31)
    result = empty_rack.assign_workload(job, 0)
    assert not result.success and result.reason is AssignmentFailure.INSUFFICIENT_GPU
    assert job.status is WorkloadStatus.QUEUED and job.assigned_rack is None
    assert empty_rack.get_available_gpu() == 30 and empty_rack.get_available_cpu() == 90


def test_remove_releases_resources(empty_rack):
    empty_rack.assign_workload(make_job("a", cpu=25, gpu=40), 0)
    empty_rack.assign_workload(make_job("b", cpu=15, gpu=0), 0)
    empty_rack.remove_workload("a")
    assert (empty_rack.get_available_cpu(), empty_rack.get_available_gpu()) == (85, 100)
    assert empty_rack.active_workload_ids == ["b"]


def test_only_waiting_jobs_can_be_assigned(empty_rack):
    running = make_job("a")
    empty_rack.assign_workload(running, 0)
    result = empty_rack.assign_workload(running, 0)          # already RUNNING
    assert not result.success and result.reason is AssignmentFailure.INVALID_STATUS
    assert empty_rack.get_available_cpu() == 90


def test_rack_power_is_sum_of_devices(dc):
    rack1 = dc.racks[0].get_power_consumption()   # CPU 72 %, GPU 55 %
    assert rack1.server_kw == pytest.approx(10 * (150 + 350 * 0.72) / 1000)
    assert rack1.gpu_kw == pytest.approx(2 * (100 + 600 * 0.55) / 1000)
    assert rack1.it_kw == pytest.approx(4.02 + 0.86)


def test_idle_rack_draws_idle_power(empty_rack):
    assert empty_rack.get_power_consumption().it_kw == pytest.approx((10 * 150 + 2 * 100) / 1000)


def test_rack_rejects_physically_impossible_values(dc):
    rack = dc.racks[0]
    with pytest.raises(SimulationInvariantError):
        rack.update_temperature(45.0)
    with pytest.raises(SimulationInvariantError):
        rack.update_cooling(1.2, 5.0)
    with pytest.raises(SimulationInvariantError):
        rack.update_cooling(0.5, -1.0)
