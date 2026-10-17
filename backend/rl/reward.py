"""Decomposed reward function (PHASE1_DESIGN §4.5, docs/RL_ENVIRONMENT.md §8-13).

    r_t = - w_E * E_t - w_C * C_t - w_T * TH_t - w_S * S_t - w_M * M_t - p_inv * 1[invalid]
          + b * sum(priority_j for agent-scheduled jobs completed in step t)

Every input comes from the simulator after the step (``RewardInputs``):
energy and carbon are the accountant's per-step values, temperatures are the
rack temperatures at the end of the step. ``calculate_reward`` is a pure
function: the same inputs and config always give the same breakdown.

Normalisation (per step of dt minutes, P_max = max IT power + max cooling power):

    E_t  = E_total_kWh / (P_max * dt / 60)                                    in [0, 1]
    C_t  = CO2_g / (P_max * dt / 60 * I_max)                                  in [0, 1]
    TH_t = mean_r theta(T_r)
           theta(T) = 0                                     T <  T_warn
                    = warning_scale * (T - T_warn) / (T_crit - T_warn)       T_warn <= T < T_crit
                    = critical_base + critical_slope * (T - T_crit)         T >= T_crit
    S_t  = (p_viol * N_failed + p_delay * sum_j waited_j / slack_j) / N_sla
    M_t  = (1[cooling target changed] + N_migrations) / N_switch               in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import Config, ThermalRewardConfig, ThermalZones
from backend.utils.units import MINUTES_PER_HOUR


@dataclass(frozen=True)
class RewardInputs:
    """Simulator outcome of one step, plus what the agent's action did."""

    total_energy_kwh: float
    carbon_g: float
    timestep_minutes: float
    rack_temperatures_c: tuple[float, ...]
    jobs_failed: int                                   # deadline missed or rejected this step
    started_wait_fractions: tuple[float, ...] = ()     # per job started this step: waited / slack, 0-1
    completed_priorities: tuple[int, ...] = ()         # agent-scheduled jobs completed this step
    cooling_changed: bool = False
    migrations: int = 0
    invalid_action: bool = False


@dataclass(frozen=True)
class RewardComponent:
    name: str
    value: float          # normalised magnitude (>= 0)
    weight: float
    contribution: float   # signed contribution to the total reward
    formula: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "weight": self.weight,
                "contribution": self.contribution, "formula": self.formula, "note": self.note}


@dataclass(frozen=True)
class RewardBreakdown:
    energy: RewardComponent
    carbon: RewardComponent
    thermal: RewardComponent
    sla: RewardComponent
    action: RewardComponent
    completion: RewardComponent

    @property
    def components(self) -> tuple[RewardComponent, ...]:
        return self.energy, self.carbon, self.thermal, self.sla, self.action, self.completion

    @property
    def total(self) -> float:
        return sum(c.contribution for c in self.components)

    def to_dict(self) -> dict:
        return {"total": self.total, **{c.name: c.to_dict() for c in self.components}}

    def explain(self) -> str:
        lines = []
        for c in self.components:
            lines.append(f"{c.name.capitalize()}:")
            lines.append(f"  formula            = {c.formula}")
            lines.append(f"  normalised value   = {c.value:.4f}")
            lines.append(f"  weight             = {c.weight:.4f}")
            lines.append(f"  contribution       = {c.contribution:+.4f}")
            if c.note:
                lines.append(f"  note               = {c.note}")
        lines.append(f"Total reward         = {self.total:+.4f}")
        return "\n".join(lines)


def thermal_penalty(temperature_c: float, zones: ThermalZones, cfg: ThermalRewardConfig) -> float:
    """Per-rack thermal severity: 0 when NORMAL, ramps to warning_scale across WARNING, jumps when CRITICAL."""
    if temperature_c < zones.warning_c:
        return 0.0
    if temperature_c < zones.critical_c:
        return cfg.warning_scale * (temperature_c - zones.warning_c) / (zones.critical_c - zones.warning_c)
    return cfg.critical_base + cfg.critical_slope * (temperature_c - zones.critical_c)


def max_job_run_cost(config: Config) -> float:
    """Upper bound on the energy + carbon reward penalty caused by running any single job.

    Largest configured job (max CPU, max GPU, max duration of each type) alone on a rack,
    at the highest carbon intensity. Used to check that a deadline violation is never
    cheaper than doing the work (see config.yaml -> reward.sla_violation_penalty).
    """
    dc, p, w = config.datacenter, config.power, config.reward.weights
    step_hours = config.simulation.timestep_minutes / MINUTES_PER_HOUR
    worst = 0.0
    for t in config.workloads.types.values():
        cpu_frac = t.cpu.max / dc.cpu_capacity_per_rack
        gpu_frac = t.gpu.max / dc.gpu_capacity_per_rack
        extra_kw = (dc.servers_per_rack * (p.server_max_w - p.server_idle_w) * cpu_frac ** p.utilisation_exponent
                    + dc.gpu_nodes_per_rack * (p.gpu_max_w - p.gpu_idle_w) * gpu_frac ** p.utilisation_exponent) / 1000.0
        steps = t.duration_minutes.max / config.simulation.timestep_minutes
        per_step = (w.energy + w.carbon) * extra_kw * step_hours / (config.max_total_power_kw * step_hours)
        worst = max(worst, per_step * steps)
    return worst


def calculate_reward(inputs: RewardInputs, config: Config) -> RewardBreakdown:
    rc, w = config.reward, config.reward.weights
    step_hours = inputs.timestep_minutes / MINUTES_PER_HOUR
    max_energy_kwh = config.max_total_power_kw * step_hours
    max_carbon_g = max_energy_kwh * config.grid.max_carbon

    energy = inputs.total_energy_kwh / max_energy_kwh
    carbon = inputs.carbon_g / max_carbon_g
    temps = inputs.rack_temperatures_c
    thermal = sum(thermal_penalty(t, config.thermal.zones, rc.thermal) for t in temps) / len(temps)
    sla_points = (rc.sla_violation_penalty * inputs.jobs_failed
                  + rc.sla_delay_penalty * sum(inputs.started_wait_fractions))
    sla = sla_points / rc.sla_normalisation
    switching = (int(inputs.cooling_changed) + inputs.migrations) / rc.switching_normalisation
    invalid_penalty = config.environment.invalid_action_penalty if inputs.invalid_action else 0.0
    completed = float(sum(inputs.completed_priorities))

    return RewardBreakdown(
        energy=RewardComponent("energy", energy, w.energy, 0.0 - w.energy * energy,
                               "E_total_kWh / (P_max * dt/60)"),
        carbon=RewardComponent("carbon", carbon, w.carbon, 0.0 - w.carbon * carbon,
                               "CO2_g / (P_max * dt/60 * I_max)"),
        thermal=RewardComponent("thermal", thermal, w.thermal, 0.0 - w.thermal * thermal,
                                "mean over racks of theta(T)",
                                f"{sum(t >= config.thermal.zones.critical_c for t in temps)} rack(s) critical"),
        sla=RewardComponent("sla", sla, w.sla, 0.0 - w.sla * sla,
                            "(p_viol * N_failed + p_delay * sum(waited/slack)) / N_sla",
                            f"{inputs.jobs_failed} failed, {len(inputs.started_wait_fractions)} started"),
        action=RewardComponent("action", switching, w.switching, 0.0 - w.switching * switching - invalid_penalty,
                               "(1[cooling changed] + N_migrations) / N_switch",
                               f"invalid-action penalty {invalid_penalty:.2f}" if inputs.invalid_action else ""),
        completion=RewardComponent("completion", completed, rc.completion_bonus, rc.completion_bonus * completed,
                                   "sum of priorities of completed agent-scheduled jobs"),
    )
