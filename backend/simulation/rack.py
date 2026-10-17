"""A rack: resource accounting, IT power, received cooling and temperature state.

The rack does *not* compute temperature dynamics or cooling response itself.
Those equations live in ``thermal.py`` and ``cooling.py``; the ``DataCentre``
evaluates them and pushes results in via ``update_temperature`` and
``update_cooling``. The rack validates every value it is given.

Resources are integer units, so accounting is exact (no floating-point drift).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import ThermalZones
from backend.simulation.enums import AssignmentFailure, SimulationInvariantError, TemperatureStatus, WorkloadStatus
from backend.simulation.hardware import GPUNode, Server
from backend.simulation.power import RackPower
from backend.simulation.thermal import get_temperature_status
from backend.simulation.workload import Workload
from backend.utils.units import w_to_kw

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AssignmentResult:
    success: bool
    reason: AssignmentFailure | None = None
    message: str = ""

    @classmethod
    def ok(cls) -> "AssignmentResult":
        return cls(True)

    @classmethod
    def fail(cls, reason: AssignmentFailure, message: str) -> "AssignmentResult":
        return cls(False, reason, message)


class Rack:
    def __init__(
        self,
        rack_id: int,
        name: str,
        cpu_capacity: int,
        gpu_capacity: int,
        servers: list[Server],
        gpu_nodes: list[GPUNode],
        temperature_c: float,
        cooling_effectiveness: float,
        temperature_bounds: tuple[float, float],
        zones: ThermalZones,
    ) -> None:
        if not 0 < cooling_effectiveness <= 1:
            raise ValueError("cooling_effectiveness must be in (0, 1]")
        self.rack_id = rack_id
        self.name = name
        self.cpu_capacity = cpu_capacity
        self.gpu_capacity = gpu_capacity
        self.servers = servers
        self.gpu_nodes = gpu_nodes
        self.cooling_effectiveness = cooling_effectiveness
        self.temperature_bounds = temperature_bounds
        self.zones = zones
        self._used_cpu = 0
        self._used_gpu = 0
        self._workloads: dict[str, Workload] = {}
        self.cooling_allocation = 0.0   # shared delivered cooling level reaching this rack's zone, 0-1
        self.cooling_power_kw = 0.0     # this rack's share of the plant's electrical cooling power
        self.temperature_c = temperature_c
        self.update_temperature(temperature_c)

    # ---------------------------------------------------------------- resources
    def get_cpu_utilisation(self) -> float:
        return self._used_cpu / self.cpu_capacity

    def get_gpu_utilisation(self) -> float:
        return self._used_gpu / self.gpu_capacity

    def get_available_cpu(self) -> int:
        return self.cpu_capacity - self._used_cpu

    def get_available_gpu(self) -> int:
        return self.gpu_capacity - self._used_gpu

    @property
    def active_workload_ids(self) -> list[str]:
        return list(self._workloads)

    @property
    def workloads(self) -> list[Workload]:
        return list(self._workloads.values())

    def can_accept_workload(self, workload: Workload) -> AssignmentResult:
        if not workload.is_pending:
            return AssignmentResult.fail(
                AssignmentFailure.INVALID_STATUS, f"{workload.job_id} is {workload.status}, not waiting"
            )
        return self.check_capacity(workload)

    def check_capacity(self, workload: Workload) -> AssignmentResult:
        """Whether the workload's CPU and GPU requirements fit in the free capacity (status not checked)."""
        if workload.cpu_requirement > self.get_available_cpu():
            return AssignmentResult.fail(
                AssignmentFailure.INSUFFICIENT_CPU,
                f"{self.name}: needs {workload.cpu_requirement} CPU, {self.get_available_cpu()} available",
            )
        if workload.gpu_requirement > self.get_available_gpu():
            return AssignmentResult.fail(
                AssignmentFailure.INSUFFICIENT_GPU,
                f"{self.name}: needs {workload.gpu_requirement} GPU, {self.get_available_gpu()} available",
            )
        return AssignmentResult.ok()

    def assign_workload(self, workload: Workload, now: int) -> AssignmentResult:
        """Start ``workload`` on this rack if it fits. Never over-allocates."""
        result = self.can_accept_workload(workload)
        if not result.success:
            return result
        workload.mark_running(self.rack_id, now)
        self._workloads[workload.job_id] = workload
        self._used_cpu += workload.cpu_requirement
        self._used_gpu += workload.gpu_requirement
        self._sync_devices()
        self.check_invariants()
        return result

    def receive_migrated_workload(self, workload: Workload) -> AssignmentResult:
        """Host a RUNNING job migrated from another rack (the caller has already released it there)."""
        if workload.status is not WorkloadStatus.RUNNING:
            return AssignmentResult.fail(
                AssignmentFailure.INVALID_STATUS, f"{workload.job_id} is {workload.status}, not running"
            )
        result = self.check_capacity(workload)
        if not result.success:
            return result
        workload.mark_migrated(self.rack_id)
        self._workloads[workload.job_id] = workload
        self._used_cpu += workload.cpu_requirement
        self._used_gpu += workload.gpu_requirement
        self._sync_devices()
        self.check_invariants()
        return result

    def remove_workload(self, job_id: str) -> Workload:
        """Release a workload's resources and return it. The caller sets its final status."""
        if job_id not in self._workloads:
            raise KeyError(f"{job_id} is not running on {self.name}")
        workload = self._workloads.pop(job_id)
        self._used_cpu -= workload.cpu_requirement
        self._used_gpu -= workload.gpu_requirement
        self._sync_devices()
        return workload

    def _sync_devices(self) -> None:
        """Spread rack load evenly over devices (intra-rack load-balancing assumption)."""
        cpu, gpu = self.get_cpu_utilisation(), self.get_gpu_utilisation()
        for server in self.servers:
            server.utilisation = cpu
        for node in self.gpu_nodes:
            node.utilisation = gpu

    # -------------------------------------------------------------------- power
    def get_power_consumption(self) -> RackPower:
        """IT power of the rack in kW, summed over its devices."""
        return RackPower(
            server_kw=w_to_kw(sum(s.power_w() for s in self.servers)),
            gpu_kw=w_to_kw(sum(g.power_w() for g in self.gpu_nodes)),
        )

    # ------------------------------------------------------- cooling / thermal
    @property
    def effective_cooling(self) -> float:
        """Cooling this rack actually receives: allocation * effectiveness (0-1)."""
        return self.cooling_allocation * self.cooling_effectiveness

    def update_cooling(self, allocation: float, cooling_power_kw: float) -> None:
        """Record the delivered shared cooling level and this rack's share of plant power."""
        if not -_TOLERANCE <= allocation <= 1 + _TOLERANCE:
            raise SimulationInvariantError(f"{self.name}: cooling allocation {allocation} outside [0, 1]")
        if cooling_power_kw < 0:
            raise SimulationInvariantError(f"{self.name}: negative cooling power {cooling_power_kw}")
        self.cooling_allocation = allocation
        self.cooling_power_kw = cooling_power_kw

    def update_temperature(self, temperature_c: float) -> None:
        low, high = self.temperature_bounds
        if not low - _TOLERANCE <= temperature_c <= high + _TOLERANCE:
            raise SimulationInvariantError(f"{self.name}: temperature {temperature_c} outside [{low}, {high}]")
        self.temperature_c = float(temperature_c)

    def get_temperature_status(self) -> TemperatureStatus:
        return get_temperature_status(self.temperature_c, self.zones)

    # --------------------------------------------------------------- integrity
    def check_invariants(self) -> None:
        used_cpu = sum(w.cpu_requirement for w in self._workloads.values())
        used_gpu = sum(w.gpu_requirement for w in self._workloads.values())
        problems = []
        if used_cpu != self._used_cpu or used_gpu != self._used_gpu:
            problems.append(f"usage {self._used_cpu}/{self._used_gpu} != job sum {used_cpu}/{used_gpu}")
        if not 0 <= self._used_cpu <= self.cpu_capacity:
            problems.append(f"CPU usage {self._used_cpu} outside [0, {self.cpu_capacity}]")
        if not 0 <= self._used_gpu <= self.gpu_capacity:
            problems.append(f"GPU usage {self._used_gpu} outside [0, {self.gpu_capacity}]")
        for w in self._workloads.values():
            if w.status is not WorkloadStatus.RUNNING or w.assigned_rack != self.rack_id:
                problems.append(f"{w.job_id} holds resources while {w.status} on rack {w.assigned_rack}")
        if problems:
            raise SimulationInvariantError(f"{self.name}: " + "; ".join(problems))

    def to_dict(self) -> dict:
        power = self.get_power_consumption()
        return {
            "rack_id": self.rack_id,
            "name": self.name,
            "cpu_capacity": self.cpu_capacity,
            "available_cpu": self.get_available_cpu(),
            "cpu_utilisation": self.get_cpu_utilisation(),
            "gpu_capacity": self.gpu_capacity,
            "available_gpu": self.get_available_gpu(),
            "gpu_utilisation": self.get_gpu_utilisation(),
            "temperature_c": self.temperature_c,
            "temperature_status": self.get_temperature_status(),
            "cooling_allocation": self.cooling_allocation,
            "cooling_effectiveness": self.cooling_effectiveness,
            "effective_cooling": self.effective_cooling,
            "server_power_kw": power.server_kw,
            "gpu_power_kw": power.gpu_kw,
            "it_power_kw": power.it_kw,
            "cooling_power_kw": self.cooling_power_kw,
            "active_workload_ids": self.active_workload_ids,
        }
