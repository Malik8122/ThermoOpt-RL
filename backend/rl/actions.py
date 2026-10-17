"""Action spaces, validation and application.

Deep-RL (primitive) action space, ``Discrete((num_racks + 2) * 3)`` = 18:

    action = placement * 3 + cooling
    placement: 0 .. num_racks-1  place the head-of-queue job on that rack
               num_racks         delay the head-of-queue job (move it to the back, status DELAYED)
               num_racks + 1     migrate one job from the hottest rack to the coolest cooler rack that fits it
    cooling:   0 decrease one level, 1 maintain, 2 increase one level

Tabular (macro) action space, ``Discrete(9)``:

    action = tabular_placement * 3 + cooling
    tabular_placement: 0 place head job on the coolest rack that fits it
                       1 place head job on the least-utilised rack that fits it
                       2 delay the head job
    A macro-action is resolved to exactly one primitive action for the current state.

Validation is side-effect free and *atomic*: a joint action is valid only if its
placement part and its cooling part are both valid. An invalid action changes
nothing in the simulator (time still advances when the environment steps) and is
never replaced by a different valid action.

With an empty queue, "place" and "delay" have nothing to act on. They are valid
no-ops (``placement_effect = "no_job"``), as specified in PHASE1_DESIGN §4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from backend.simulation.datacentre import DataCentre
from backend.simulation.workload import Workload


class CoolingCommand(IntEnum):
    DECREASE = 0
    MAINTAIN = 1
    INCREASE = 2


class PlacementKind(StrEnum):
    PLACE = "PLACE"
    DELAY = "DELAY"
    MIGRATE = "MIGRATE"


class TabularPlacement(IntEnum):
    PLACE_COOLEST = 0
    PLACE_LEAST_UTILISED = 1
    DELAY = 2


class InvalidActionReason(StrEnum):
    INSUFFICIENT_CPU = "INSUFFICIENT_CPU"
    INSUFFICIENT_GPU = "INSUFFICIENT_GPU"
    NO_FEASIBLE_RACK = "NO_FEASIBLE_RACK"               # tabular placement: no rack can host the head job
    NO_MIGRATION_CANDIDATE = "NO_MIGRATION_CANDIDATE"   # nothing on the hottest rack fits on a cooler rack
    COOLING_AT_MAXIMUM = "COOLING_AT_MAXIMUM"
    COOLING_AT_MINIMUM = "COOLING_AT_MINIMUM"


N_COOLING_COMMANDS = len(CoolingCommand)
N_TABULAR_ACTIONS = len(TabularPlacement) * N_COOLING_COMMANDS   # 9


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrimitiveAction:
    placement: int
    cooling: CoolingCommand
    num_racks: int

    @property
    def kind(self) -> PlacementKind:
        if self.placement < self.num_racks:
            return PlacementKind.PLACE
        return PlacementKind.DELAY if self.placement == self.num_racks else PlacementKind.MIGRATE

    @property
    def rack_index(self) -> int | None:
        return self.placement if self.kind is PlacementKind.PLACE else None

    def describe(self) -> str:
        placement = {PlacementKind.PLACE: f"place head job on Rack {self.placement + 1}",
                     PlacementKind.DELAY: "delay head job",
                     PlacementKind.MIGRATE: "migrate from hottest rack"}[self.kind]
        return f"{placement} + {self.cooling.name.lower()} cooling"


class ActionCodec:
    """Deterministic mapping between integers and primitive actions."""

    def __init__(self, num_racks: int) -> None:
        self.num_racks = num_racks
        self.delay_index = num_racks
        self.migrate_index = num_racks + 1
        self.n_placements = num_racks + 2
        self.n_actions = self.n_placements * N_COOLING_COMMANDS

    def encode(self, placement: int, cooling: CoolingCommand) -> int:
        if not 0 <= placement < self.n_placements:
            raise ValueError(f"placement must be in [0, {self.n_placements}), got {placement}")
        return int(placement) * N_COOLING_COMMANDS + int(CoolingCommand(cooling))

    def decode(self, action: int) -> PrimitiveAction:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        placement, cooling = divmod(int(action), N_COOLING_COMMANDS)
        return PrimitiveAction(placement, CoolingCommand(cooling), self.num_racks)


def encode_tabular_action(placement: TabularPlacement, cooling: CoolingCommand) -> int:
    return int(TabularPlacement(placement)) * N_COOLING_COMMANDS + int(CoolingCommand(cooling))


def decode_tabular_action(action: int) -> tuple[TabularPlacement, CoolingCommand]:
    if not 0 <= action < N_TABULAR_ACTIONS:
        raise ValueError(f"tabular action must be in [0, {N_TABULAR_ACTIONS}), got {action}")
    placement, cooling = divmod(int(action), N_COOLING_COMMANDS)
    return TabularPlacement(placement), CoolingCommand(cooling)


# --------------------------------------------------------------------------- #
# Validation (read-only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MigrationPlan:
    job_id: str
    source_rack: int
    target_rack: int


@dataclass(frozen=True)
class ActionCheck:
    action: PrimitiveAction
    valid: bool
    reasons: tuple[InvalidActionReason, ...]
    placement_effect: str            # "place" | "delay" | "migrate" | "no_job" | "none" (if invalid)
    job_id: str | None               # head job acted on (place/delay), or migrated job
    migration: MigrationPlan | None
    new_cooling_target: float        # target after the cooling part (unchanged for MAINTAIN)

    @property
    def reason_text(self) -> str | None:
        return "; ".join(self.reasons) if self.reasons else None


def job_thermal_load(dc: DataCentre, job: Workload, rack_index: int) -> float:
    """Share of a rack's thermal load (0-1) contributed by one job, using the thermal model's weights."""
    rack = dc.racks[rack_index]
    return (dc.thermal.w_cpu * job.cpu_requirement / rack.cpu_capacity
            + dc.thermal.w_gpu * job.gpu_requirement / rack.gpu_capacity)


def plan_migration(dc: DataCentre) -> MigrationPlan | None:
    """Deterministic migration rule for the MIGRATE action.

    Source: the hottest rack (ties -> lowest index). Jobs on it are tried in
    order of decreasing thermal-load contribution (ties -> job_id). Each goes to
    the coolest rack that is strictly cooler than the source and has capacity
    (ties -> lowest index). Returns None if no job can move.
    """
    source = max(dc.racks, key=lambda r: (r.temperature_c, -r.rack_id))
    cooler = sorted((r for r in dc.racks if r.temperature_c < source.temperature_c),
                    key=lambda r: (r.temperature_c, r.rack_id))
    jobs = sorted(source.workloads, key=lambda w: (-job_thermal_load(dc, w, source.rack_id), w.job_id))
    for job in jobs:
        for target in cooler:
            if target.check_capacity(job).success:
                return MigrationPlan(job.job_id, source.rack_id, target.rack_id)
    return None


def validate_action(dc: DataCentre, action: PrimitiveAction) -> ActionCheck:
    reasons: list[InvalidActionReason] = []
    head = dc.queue[0] if dc.queue else None
    job_id: str | None = head.job_id if head else None
    migration: MigrationPlan | None = None

    if action.kind is PlacementKind.MIGRATE:
        migration = plan_migration(dc)
        effect = "migrate"
        job_id = migration.job_id if migration else None
        if migration is None:
            reasons.append(InvalidActionReason.NO_MIGRATION_CANDIDATE)
    elif head is None:
        effect = "no_job"
    elif action.kind is PlacementKind.DELAY:
        effect = "delay"
    else:
        effect = "place"
        result = dc.racks[action.rack_index].can_accept_workload(head)
        if not result.success:
            reasons.append(InvalidActionReason(result.reason.value))

    target = dc.cooling.target_level
    if action.cooling is CoolingCommand.INCREASE:
        above = dc.cooling.level_above()
        if above is None:
            reasons.append(InvalidActionReason.COOLING_AT_MAXIMUM)
        else:
            target = above
    elif action.cooling is CoolingCommand.DECREASE:
        below = dc.cooling.level_below()
        if below is None:
            reasons.append(InvalidActionReason.COOLING_AT_MINIMUM)
        else:
            target = below

    valid = not reasons
    return ActionCheck(
        action=action,
        valid=valid,
        reasons=tuple(reasons),
        placement_effect=effect if valid else "none",
        job_id=job_id,
        migration=migration if valid else None,
        new_cooling_target=target if valid else dc.cooling.target_level,
    )


# --------------------------------------------------------------------------- #
# Tabular macro-action resolution
# --------------------------------------------------------------------------- #
def rack_utilisation(dc: DataCentre, rack_index: int) -> float:
    """Combined CPU+GPU utilisation of a rack (0-1), capacity weighted."""
    r = dc.racks[rack_index]
    used = (r.cpu_capacity - r.get_available_cpu()) + (r.gpu_capacity - r.get_available_gpu())
    return used / (r.cpu_capacity + r.gpu_capacity)


def resolve_tabular_action(
    dc: DataCentre, codec: ActionCodec, action: int
) -> tuple[PrimitiveAction | None, InvalidActionReason | None]:
    """Map a macro-action to the primitive action it means in the current state.

    Returns ``(None, NO_FEASIBLE_RACK)`` if a placement macro has a job but no rack can host it.
    """
    placement, cooling = decode_tabular_action(action)
    head = dc.queue[0] if dc.queue else None
    if placement is TabularPlacement.DELAY or head is None:
        return codec.decode(codec.encode(codec.delay_index, cooling)), None
    feasible = [r for r in dc.racks if r.can_accept_workload(head).success]
    if not feasible:
        return None, InvalidActionReason.NO_FEASIBLE_RACK
    if placement is TabularPlacement.PLACE_COOLEST:
        rack = min(feasible, key=lambda r: (r.temperature_c, r.rack_id))
    else:
        rack = min(feasible, key=lambda r: (rack_utilisation(dc, r.rack_id), r.rack_id))
    return codec.decode(codec.encode(rack.rack_id, cooling)), None


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppliedAction:
    started_job: Workload | None = None
    started_wait_fraction: float = 0.0   # share of the job's slack spent waiting before it started, 0-1
    migrated: bool = False
    cooling_changed: bool = False


def slack_used_fraction(job: Workload, now: int) -> float:
    """Waiting time as a fraction of the slack the job was given (0 if it had no slack)."""
    slack = job.deadline - job.arrival_time - job.duration
    if slack <= 0:
        return 0.0
    return min(1.0, max(0.0, (now - job.arrival_time) / slack))


def apply_action(dc: DataCentre, check: ActionCheck) -> AppliedAction:
    """Issue a validated action to the simulator. Invalid actions are a no-op."""
    if not check.valid:
        return AppliedAction()
    action = check.action
    started, wait_fraction, migrated = None, 0.0, False

    if check.placement_effect == "place":
        job = dc.queue[0]
        result = dc.place_workload(job.job_id, action.rack_index)
        if not result.success:  # validated just before; failing here is a bug
            raise RuntimeError(f"validated placement failed: {result.message}")
        started, wait_fraction = job, slack_used_fraction(job, dc.minute)
    elif check.placement_effect == "delay":
        dc.delay_workload(dc.queue[0].job_id)
    elif check.placement_effect == "migrate":
        plan = check.migration
        result = dc.migrate_workload(plan.job_id, plan.target_rack)
        if not result.success:
            raise RuntimeError(f"validated migration failed: {result.message}")
        migrated = True

    cooling_changed = False
    if action.cooling is not CoolingCommand.MAINTAIN:
        dc.set_cooling_target(check.new_cooling_target)
        cooling_changed = True

    return AppliedAction(started, wait_fraction, migrated, cooling_changed)
