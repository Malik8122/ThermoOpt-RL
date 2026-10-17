"""The data-centre digital twin: composes racks, cooling, thermal, grid,
workload generation and accounting into one simulated facility.

``DataCentre`` holds state and advances time. It does not decide anything:
placement, delay and cooling set-points are *commands* issued between steps
by a controller (a baseline, an RL agent, the dashboard, or a test).

Order of operations in ``step()`` for the interval [t, t + dt):

  1. Arrivals        jobs arriving during the interval join the queue (released at t + dt)
  2. Utilisation     read from the jobs RUNNING during the interval
  3. IT power        from utilisation, per device
  4. Cooling         delivered level ramps towards the target; cooling power
  5. Temperatures    thermal model with the load and delivered cooling of the interval
  6. Grid signals    price and carbon intensity at time t
  7. Accounting      energy, carbon and cost of the interval (per step + cumulative)
  8. Execution       running jobs progress by dt; finished jobs complete at t + dt
                     and release their resources
  9. Deadlines       waiting jobs that can no longer finish on time -> FAILED

Utilisation, power and heat are evaluated *before* completions are processed,
so a job with duration d contributes load to exactly d minutes of simulation.
This is the reason execution/completion comes after the physics, not before.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields

import numpy as np

from backend.config import Config
from backend.simulation.accounting import EnergyAccountant, EnergyStep
from backend.simulation.cooling import CoolingSystem
from backend.simulation.enums import (
    AssignmentFailure,
    FailureReason,
    SimulationInvariantError,
    TemperatureStatus,
    WorkloadStatus,
)
from backend.simulation.grid import GridSignals
from backend.simulation.hardware import GPUNode, Server
from backend.simulation.power import PowerModel
from backend.simulation.rack import AssignmentResult, Rack
from backend.simulation.thermal import AmbientModel, ThermalModel
from backend.simulation.workload import Workload
from backend.simulation.workload_generator import WorkloadGenerator
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng
from backend.utils.units import HOURS_PER_DAY, MINUTES_PER_HOUR, fraction_to_pct, pct_to_fraction

logger = get_logger(__name__)


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened in one step. Per-rack values are tuples ordered by rack."""

    minute: int                     # start of the interval, minutes since episode start
    clock: str                      # HH:MM at the start of the interval
    ambient_c: float
    price_inr_per_kwh: float
    carbon_g_per_kwh: float
    rack_cpu_utilisation: tuple[float, ...]     # during the interval, 0-1
    rack_gpu_utilisation: tuple[float, ...]
    rack_it_power_kw: tuple[float, ...]
    rack_effective_cooling: tuple[float, ...]   # 0-1
    rack_temperature_c: tuple[float, ...]       # at the end of the interval
    rack_status: tuple[TemperatureStatus, ...]
    cooling_target: float           # 0-1
    cooling_level: float            # delivered, 0-1
    server_power_kw: float
    gpu_power_kw: float
    it_power_kw: float
    cooling_power_kw: float
    total_power_kw: float
    it_energy_kwh: float
    cooling_energy_kwh: float
    total_energy_kwh: float
    carbon_g: float
    cost_inr: float
    cumulative_it_energy_kwh: float
    cumulative_cooling_energy_kwh: float
    cumulative_total_energy_kwh: float
    cumulative_carbon_g: float
    cumulative_cost_inr: float
    arrivals: int
    completed: int
    failed: int                     # deadline misses + queue-full rejections
    queue_length: int
    running: int

    def to_row(self, precision: int = 6) -> dict[str, object]:
        """Flat dict for CSV output. Fractions are written as percentages."""
        row: dict[str, object] = {}
        per_rack = {
            "rack_cpu_utilisation": ("cpu_pct", fraction_to_pct),
            "rack_gpu_utilisation": ("gpu_pct", fraction_to_pct),
            "rack_it_power_kw": ("it_power_kw", None),
            "rack_effective_cooling": ("effective_cooling_pct", fraction_to_pct),
            "rack_temperature_c": ("temperature_c", None),
            "rack_status": ("status", None),
        }
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in per_rack:
                suffix, convert = per_rack[f.name]
                for i, v in enumerate(value):
                    v = convert(v) if convert else v
                    row[f"rack{i + 1}_{suffix}"] = round(v, precision) if isinstance(v, float) else str(v)
            elif f.name in ("cooling_target", "cooling_level"):
                row[f"{f.name}_pct"] = round(fraction_to_pct(value), precision)
            else:
                row[f.name] = round(value, precision) if isinstance(value, float) else value
        return row


class DataCentre:
    def __init__(self, config: Config, seed: int | None = None, start_hour: float | None = None,
                 validate: bool = True) -> None:
        self.config = config
        self.seed = config.project.seed if seed is None else seed
        self.start_hour = config.simulation.start_hour if start_hour is None else start_hour
        self.dt = config.simulation.timestep_minutes
        self.validate = validate
        self.minute = 0

        self.power_model = PowerModel.from_config(config.power)
        self.cooling = CoolingSystem.from_config(config.cooling)
        self.thermal = ThermalModel(config.thermal, self.dt)
        self.ambient = AmbientModel.from_config(config.thermal)
        self.grid = GridSignals(config.grid)
        self.accountant = EnergyAccountant()
        # Independent random streams: changing one component never shifts another.
        self._thermal_rng = make_rng(derive_seed(self.seed, "thermal"))
        self.generator = WorkloadGenerator(
            config.workloads, make_rng(derive_seed(self.seed, "workloads")), self.dt
        )

        self.racks = self._build_racks()
        self.queue: list[Workload] = []
        self.workloads: dict[str, Workload] = {}
        self.counters: Counter[str] = Counter()
        self._seed_background_jobs(make_rng(derive_seed(self.seed, "background")))
        self._apply_cooling()
        if self.validate:
            self.check_invariants()

    # ------------------------------------------------------------------ build
    def _build_racks(self) -> list[Rack]:
        dc, th = self.config.datacenter, self.config.thermal
        racks = []
        for i in range(dc.num_racks):
            racks.append(Rack(
                rack_id=i,
                name=dc.rack_names[i],
                cpu_capacity=dc.cpu_capacity_per_rack,
                gpu_capacity=dc.gpu_capacity_per_rack,
                servers=[Server(f"R{i + 1}-S{j + 1:02d}", self.power_model) for j in range(dc.servers_per_rack)],
                gpu_nodes=[GPUNode(f"R{i + 1}-G{j + 1}", self.power_model) for j in range(dc.gpu_nodes_per_rack)],
                temperature_c=dc.initial_temperatures_c[i],
                cooling_effectiveness=th.rack_cooling_effectiveness[i],
                temperature_bounds=(th.min_temperature_c, th.max_temperature_c),
                zones=th.zones,
            ))
        return racks

    def _seed_background_jobs(self, rng: np.random.Generator) -> None:
        """Create running jobs that reproduce the configured initial utilisation exactly."""
        dc = self.config.datacenter
        for rack, cpu_pct, gpu_pct in zip(self.racks, dc.initial_cpu_utilisation_pct, dc.initial_gpu_utilisation_pct):
            cpu_units = round(pct_to_fraction(cpu_pct) * rack.cpu_capacity)
            gpu_units = round(pct_to_fraction(gpu_pct) * rack.gpu_capacity)
            for job in self.generator.create_background_workloads(rack.rack_id, cpu_units, gpu_units, rng, self.minute):
                result = rack.assign_workload(job, self.minute)
                if not result.success:
                    raise SimulationInvariantError(f"background job does not fit: {result.message}")
                self.workloads[job.job_id] = job

    # ------------------------------------------------------------------ clock
    @property
    def hour_of_day(self) -> float:
        return (self.start_hour + self.minute / MINUTES_PER_HOUR) % HOURS_PER_DAY

    @property
    def clock(self) -> str:
        total_minutes = int(round(self.hour_of_day * MINUTES_PER_HOUR))
        return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

    # ------------------------------------------------------------ rack arrays
    def temperatures(self) -> np.ndarray:
        return np.array([r.temperature_c for r in self.racks])

    def cpu_utilisation(self) -> np.ndarray:
        return np.array([r.get_cpu_utilisation() for r in self.racks])

    def gpu_utilisation(self) -> np.ndarray:
        return np.array([r.get_gpu_utilisation() for r in self.racks])

    # --------------------------------------------------------------- commands
    def submit_workload(self, workload: Workload) -> bool:
        """Add a job to the queue. Rejected (FAILED, QUEUE_FULL) if the queue is at capacity."""
        self.workloads[workload.job_id] = workload
        self.counters["arrived"] += 1
        if len(self.queue) >= self.config.workloads.max_queue_length:
            workload.mark_failed(FailureReason.QUEUE_FULL, self.minute)
            self.counters["failed"] += 1
            self.counters["rejected"] += 1
            logger.debug("%s rejected: queue full", workload.job_id)
            return False
        self.queue.append(workload)
        return True

    def place_workload(self, job_id: str, rack_index: int) -> AssignmentResult:
        """Start a waiting job on a rack now. Fails with a reason if it does not fit."""
        if not 0 <= rack_index < len(self.racks):
            return AssignmentResult.fail(AssignmentFailure.UNKNOWN_RACK, f"no rack with index {rack_index}")
        job = next((w for w in self.queue if w.job_id == job_id), None)
        if job is None:
            return AssignmentResult.fail(AssignmentFailure.UNKNOWN_WORKLOAD, f"{job_id} is not waiting in the queue")
        result = self.racks[rack_index].assign_workload(job, self.minute)
        if result.success:
            self.queue.remove(job)
            self.counters["started"] += 1
        return result

    def migrate_workload(self, job_id: str, target_rack_index: int) -> AssignmentResult:
        """Move a RUNNING job to another rack, keeping its progress.

        Capacity on the target is checked *before* anything is released on the
        source, so a failed migration leaves the twin untouched. Migration is
        modelled as instantaneous with no lost work (a simplifying assumption);
        its cost is expressed through the reward's switching penalty.
        """
        if not 0 <= target_rack_index < len(self.racks):
            return AssignmentResult.fail(AssignmentFailure.UNKNOWN_RACK, f"no rack with index {target_rack_index}")
        job = self.workloads.get(job_id)
        if job is None or job.status is not WorkloadStatus.RUNNING:
            return AssignmentResult.fail(AssignmentFailure.UNKNOWN_WORKLOAD, f"{job_id} is not running")
        if job.assigned_rack == target_rack_index:
            return AssignmentResult.fail(AssignmentFailure.SAME_RACK, f"{job_id} already runs on that rack")
        source, target = self.racks[job.assigned_rack], self.racks[target_rack_index]
        check = target.check_capacity(job)
        if not check.success:
            return check
        source.remove_workload(job_id)
        result = target.receive_migrated_workload(job)
        if not result.success:  # unreachable after the capacity check; fail loudly rather than lose the job
            raise SimulationInvariantError(f"migration of {job_id} failed after capacity check: {result.message}")
        self.counters["migrations"] += 1
        return result

    def delay_workload(self, job_id: str) -> None:
        """Defer a waiting job: mark it DELAYED and move it to the back of the queue."""
        job = next((w for w in self.queue if w.job_id == job_id), None)
        if job is None:
            raise KeyError(f"{job_id} is not waiting in the queue")
        job.mark_delayed()
        self.queue.remove(job)
        self.queue.append(job)
        self.counters["delay_decisions"] += 1

    def set_cooling_target(self, level: float) -> None:
        """Request a cooling level (fraction 0-1); delivered level follows with a delay."""
        self.cooling.set_target(level)

    # ----------------------------------------------------------------- physics
    def _apply_cooling(self) -> None:
        share_kw = self.cooling.power_kw() / len(self.racks)
        for rack in self.racks:
            rack.update_cooling(self.cooling.current_level, share_kw)

    def step(self) -> StepRecord:
        """Advance the twin by one timestep. See the module docstring for the order."""
        now, hour, dt = self.minute, self.hour_of_day, self.dt
        clock = self.clock

        # 1. arrivals
        arrivals = self.generator.generate(now, hour)
        rejected = sum(not self.submit_workload(job) for job in arrivals)

        # 2-3. utilisation and IT power during the interval
        cpu, gpu = self.cpu_utilisation(), self.gpu_utilisation()
        rack_power = [r.get_power_consumption() for r in self.racks]
        server_kw = sum(p.server_kw for p in rack_power)
        gpu_kw = sum(p.gpu_kw for p in rack_power)
        it_kw = server_kw + gpu_kw

        # 4. cooling response
        self.cooling.advance(dt)
        self._apply_cooling()
        cooling_kw = self.cooling.power_kw()

        # 5. temperatures
        ambient = self.ambient.temperature_at(hour)
        effective = np.array([r.effective_cooling for r in self.racks])
        new_temps = self.thermal.step(
            self.temperatures(), self.thermal.thermal_load(cpu, gpu), effective, ambient, self._thermal_rng
        )
        for rack, temp in zip(self.racks, new_temps):
            rack.update_temperature(temp)

        # 6-7. grid signals and accounting
        price, carbon = self.grid.price_at(hour), self.grid.carbon_at(hour)
        energy: EnergyStep = self.accountant.record(it_kw, cooling_kw, price, carbon, dt)

        # 8-9. execution, completion, deadline expiry at t + dt
        self.minute = now + dt
        completed = self._advance_running(dt)
        failed = self._expire_deadlines() + rejected

        if self.validate:
            self.check_invariants()

        totals = self.accountant.totals
        return StepRecord(
            minute=now,
            clock=clock,
            ambient_c=ambient,
            price_inr_per_kwh=price,
            carbon_g_per_kwh=carbon,
            rack_cpu_utilisation=tuple(cpu.tolist()),
            rack_gpu_utilisation=tuple(gpu.tolist()),
            rack_it_power_kw=tuple(p.it_kw for p in rack_power),
            rack_effective_cooling=tuple(effective.tolist()),
            rack_temperature_c=tuple(new_temps.tolist()),
            rack_status=tuple(r.get_temperature_status() for r in self.racks),
            cooling_target=self.cooling.target_level,
            cooling_level=self.cooling.current_level,
            server_power_kw=server_kw,
            gpu_power_kw=gpu_kw,
            it_power_kw=it_kw,
            cooling_power_kw=cooling_kw,
            total_power_kw=it_kw + cooling_kw,
            it_energy_kwh=energy.it_kwh,
            cooling_energy_kwh=energy.cooling_kwh,
            total_energy_kwh=energy.total_kwh,
            carbon_g=energy.carbon_g,
            cost_inr=energy.cost_inr,
            cumulative_it_energy_kwh=totals.it_kwh,
            cumulative_cooling_energy_kwh=totals.cooling_kwh,
            cumulative_total_energy_kwh=totals.total_kwh,
            cumulative_carbon_g=totals.carbon_g,
            cumulative_cost_inr=totals.cost_inr,
            arrivals=len(arrivals),
            completed=completed,
            failed=failed,
            queue_length=len(self.queue),
            running=sum(len(r.active_workload_ids) for r in self.racks),
        )

    def _advance_running(self, minutes: int) -> int:
        completed = 0
        for rack in self.racks:
            for job in rack.workloads:
                job.advance(minutes)
                if job.remaining_time == 0:
                    rack.remove_workload(job.job_id)
                    job.mark_completed(self.minute)
                    completed += 1
        self.counters["completed"] += completed
        return completed

    def _expire_deadlines(self) -> int:
        expired = [w for w in self.queue if not w.can_meet_deadline(self.minute)]
        for job in expired:
            job.mark_failed(FailureReason.DEADLINE_MISSED, self.minute)
            self.queue.remove(job)
            logger.debug("%s failed: deadline %d missed at minute %d", job.job_id, job.deadline, self.minute)
        self.counters["failed"] += len(expired)
        return len(expired)

    # --------------------------------------------------------------- integrity
    def check_invariants(self) -> None:
        """Cross-check racks, queue and job states. Raises SimulationInvariantError on any mismatch."""
        for rack in self.racks:
            rack.check_invariants()
        on_racks = {jid for r in self.racks for jid in r.active_workload_ids}
        queued = {w.job_id for w in self.queue}
        for job in self.workloads.values():
            if job.status is WorkloadStatus.RUNNING and job.job_id not in on_racks:
                raise SimulationInvariantError(f"{job.job_id} is RUNNING but on no rack")
            if job.is_pending and job.job_id not in queued:
                raise SimulationInvariantError(f"{job.job_id} is {job.status} but not in the queue")
            if not job.is_pending and job.job_id in queued:
                raise SimulationInvariantError(f"{job.job_id} is {job.status} but still queued")
        if len(queued) != len(self.queue):
            raise SimulationInvariantError("a job appears in the queue more than once")

    # ---------------------------------------------------------------- reporting
    def snapshot(self) -> dict:
        """JSON-friendly view of the current state (used by the API in Step 15)."""
        return {
            "minute": self.minute,
            "clock": self.clock,
            "hour_of_day": self.hour_of_day,
            "ambient_c": self.ambient.temperature_at(self.hour_of_day),
            "price_inr_per_kwh": self.grid.price_at(self.hour_of_day),
            "carbon_g_per_kwh": self.grid.carbon_at(self.hour_of_day),
            "cooling": {
                "target_level": self.cooling.target_level,
                "current_level": self.cooling.current_level,
                "progress": self.cooling.progress,
                "power_kw": self.cooling.power_kw(),
            },
            "racks": [r.to_dict() for r in self.racks],
            "queue": [w.job_id for w in self.queue],
            "totals": self.accountant.totals.to_dict(),
            "counters": dict(self.counters),
        }
