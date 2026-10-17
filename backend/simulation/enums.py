"""Enumerations and exceptions shared by the simulation. Use these, never bare strings."""

from __future__ import annotations

from enum import StrEnum


class WorkloadType(StrEnum):
    NORMAL_CPU = "NORMAL_CPU"
    GPU_INFERENCE = "GPU_INFERENCE"
    AI_TRAINING = "AI_TRAINING"


class WorkloadStatus(StrEnum):
    QUEUED = "QUEUED"        # waiting, not holding resources
    DELAYED = "DELAYED"      # deliberately deferred by a controller; still waiting, not holding resources
    RUNNING = "RUNNING"      # holding resources on exactly one rack
    COMPLETED = "COMPLETED"  # finished, resources released
    FAILED = "FAILED"        # could not meet its deadline, or was rejected; never held resources after failing


PENDING_STATUSES = frozenset({WorkloadStatus.QUEUED, WorkloadStatus.DELAYED})


class FailureReason(StrEnum):
    DEADLINE_MISSED = "DEADLINE_MISSED"  # could no longer finish before its deadline even if started now
    QUEUE_FULL = "QUEUE_FULL"            # rejected on arrival because the queue was at capacity


class TemperatureStatus(StrEnum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AssignmentFailure(StrEnum):
    INSUFFICIENT_CPU = "INSUFFICIENT_CPU"
    INSUFFICIENT_GPU = "INSUFFICIENT_GPU"
    INVALID_STATUS = "INVALID_STATUS"
    UNKNOWN_RACK = "UNKNOWN_RACK"
    UNKNOWN_WORKLOAD = "UNKNOWN_WORKLOAD"
    SAME_RACK = "SAME_RACK"


class WorkloadStateError(RuntimeError):
    """An illegal workload status transition was attempted."""


class SimulationInvariantError(AssertionError):
    """A physical or accounting invariant of the twin was violated (a bug, not a user error)."""
