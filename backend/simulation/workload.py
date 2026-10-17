"""The workload (job) record and its legal status transitions.

All times are integer simulated minutes since the start of the episode.

    QUEUED ──delay──► DELAYED ──┐
       │                        │
       ├────────start───────────┴──► RUNNING ──finish──► COMPLETED
       │                        │     ▲    │
       │                        │     └────┘ migrate (stays RUNNING, changes rack)
       └──────fail──────────────┴──► FAILED

Only RUNNING workloads hold rack resources. Every transition checks its
precondition and raises ``WorkloadStateError`` otherwise, so bookkeeping bugs
surface immediately instead of corrupting resource accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from backend.simulation.enums import (
    PENDING_STATUSES,
    FailureReason,
    WorkloadStateError,
    WorkloadStatus,
    WorkloadType,
)

MIN_PRIORITY = 1
MAX_PRIORITY = 3


@dataclass
class Workload:
    job_id: str
    workload_type: WorkloadType
    cpu_requirement: int
    gpu_requirement: int
    arrival_time: int            # minute the job becomes visible in the queue
    duration: int                # minutes of execution needed
    deadline: int                # minute by which it must have finished
    priority: int                # 1 (low) .. 3 (high)
    flexible: bool               # True if the deadline leaves room to defer it
    created_at: int              # minute the record was created
    status: WorkloadStatus = WorkloadStatus.QUEUED
    assigned_rack: int | None = None
    started_at: int | None = None
    completed_at: int | None = None
    remaining_time: int = field(init=False)
    delay_count: int = 0
    migration_count: int = 0
    failure_reason: FailureReason | None = None
    is_background: bool = False  # already running when the episode started

    def __post_init__(self) -> None:
        if self.cpu_requirement < 0 or self.gpu_requirement < 0:
            raise ValueError(f"{self.job_id}: resource requirements must be non-negative")
        if self.cpu_requirement == 0 and self.gpu_requirement == 0:
            raise ValueError(f"{self.job_id}: a workload must require some CPU or GPU")
        if self.duration <= 0:
            raise ValueError(f"{self.job_id}: duration must be positive")
        if self.deadline < self.arrival_time + self.duration:
            raise ValueError(f"{self.job_id}: deadline is earlier than the earliest possible finish")
        if not MIN_PRIORITY <= self.priority <= MAX_PRIORITY:
            raise ValueError(f"{self.job_id}: priority must be in [{MIN_PRIORITY}, {MAX_PRIORITY}]")
        self.remaining_time = self.duration

    # ------------------------------------------------------------------ queries
    @property
    def is_pending(self) -> bool:
        return self.status in PENDING_STATUSES

    @property
    def latest_start(self) -> int:
        """Last minute at which starting still meets the deadline."""
        return self.deadline - self.duration

    def slack_at(self, now: int) -> int:
        """Minutes the job can still wait at time ``now`` (negative = already too late)."""
        return self.latest_start - now

    def can_meet_deadline(self, now: int) -> bool:
        return now <= self.latest_start

    # -------------------------------------------------------------- transitions
    def mark_running(self, rack_index: int, now: int) -> None:
        self._require(self.is_pending, "start")
        self.status = WorkloadStatus.RUNNING
        self.assigned_rack = rack_index
        self.started_at = now

    def mark_migrated(self, rack_index: int) -> None:
        """Record a move of a RUNNING job to another rack. Progress (remaining_time) is kept."""
        self._require(self.status is WorkloadStatus.RUNNING and rack_index != self.assigned_rack, "migrate")
        self.assigned_rack = rack_index
        self.migration_count += 1

    def advance(self, minutes: int) -> None:
        self._require(self.status is WorkloadStatus.RUNNING, "advance")
        self.remaining_time = max(0, self.remaining_time - minutes)

    def mark_completed(self, now: int) -> None:
        self._require(self.status is WorkloadStatus.RUNNING and self.remaining_time == 0, "complete")
        self.status = WorkloadStatus.COMPLETED
        self.completed_at = now

    def mark_delayed(self) -> None:
        self._require(self.is_pending, "delay")
        self.status = WorkloadStatus.DELAYED
        self.delay_count += 1

    def mark_failed(self, reason: FailureReason, now: int) -> None:
        self._require(self.is_pending, "fail")
        self.status = WorkloadStatus.FAILED
        self.failure_reason = reason
        self.completed_at = now

    def _require(self, condition: bool, action: str) -> None:
        if not condition:
            raise WorkloadStateError(
                f"cannot {action} {self.job_id}: status={self.status}, remaining={self.remaining_time}"
            )

    def to_dict(self) -> dict:
        return asdict(self)
