"""Open-loop reference run of the digital twin (no RL).

Controller used for this run (deliberately simple, *not* one of the Step 6 baselines):
  * placement: FIFO first-fit. Each waiting job, in arrival order, starts on the
    lowest-numbered rack with enough free CPU and GPU; otherwise it keeps waiting.
  * cooling: target held at the initial level (65 %) for the whole run.

Usage:
    python -m backend.simulation.run_baseline                      # 60 min, seed 42 -> data/baseline_60min.csv
    python -m backend.simulation.run_baseline --minutes 1440 --out data/baseline_24h.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from backend.config import Config, get_config
from backend.simulation.datacentre import DataCentre, StepRecord
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import set_global_seed
from backend.utils.units import g_to_kg

logger = get_logger("run_baseline")


def first_fit_dispatch(dc: DataCentre) -> int:
    """Start every waiting job that fits somewhere, lowest rack index first. Returns jobs started."""
    started = 0
    for job in list(dc.queue):
        for rack in dc.racks:
            if rack.can_accept_workload(job).success:
                dc.place_workload(job.job_id, rack.rack_id)
                started += 1
                break
    return started


def run(config: Config, minutes: int, seed: int, start_hour: float | None = None) -> list[StepRecord]:
    set_global_seed(seed)
    dc = DataCentre(config, seed=seed, start_hour=start_hour)
    records = []
    for _ in range(minutes // dc.dt):
        first_fit_dispatch(dc)
        records.append(dc.step())
    return records


def write_csv(records: list[StepRecord], path: Path) -> str:
    """Write records to CSV and return the file's SHA-256 (for reproducibility checks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in records]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise(records: list[StepRecord], initial_temperatures: list[float]) -> str:
    last = records[-1]
    lines = [f"Simulated {len(records)} steps ({records[0].clock} -> end of {last.clock})", ""]
    lines.append("Rack   T start  T end   T min   T max   mean CPU  mean GPU  final status")
    for i, t0 in enumerate(initial_temperatures):
        temps = [t0] + [r.rack_temperature_c[i] for r in records]
        cpu = sum(r.rack_cpu_utilisation[i] for r in records) / len(records)
        gpu = sum(r.rack_gpu_utilisation[i] for r in records) / len(records)
        lines.append(
            f"R{i + 1}     {temps[0]:6.2f}  {temps[-1]:6.2f}  {min(temps):6.2f}  {max(temps):6.2f}"
            f"   {cpu:6.1%}    {gpu:6.1%}    {last.rack_status[i]}"
        )
    mean = lambda attr: sum(getattr(r, attr) for r in records) / len(records)  # noqa: E731
    lines += [
        "",
        f"Mean power  : IT {mean('it_power_kw'):.2f} kW | cooling {mean('cooling_power_kw'):.2f} kW | "
        f"total {mean('total_power_kw'):.2f} kW | PUE {mean('total_power_kw') / mean('it_power_kw'):.3f}",
        f"Energy      : IT {last.cumulative_it_energy_kwh:.3f} kWh | cooling {last.cumulative_cooling_energy_kwh:.3f} kWh"
        f" | total {last.cumulative_total_energy_kwh:.3f} kWh",
        f"Carbon      : {g_to_kg(last.cumulative_carbon_g):.3f} kg CO2",
        f"Cost        : INR {last.cumulative_cost_inr:.2f}",
        f"Jobs        : {sum(r.arrivals for r in records)} arrived, {sum(r.completed for r in records)} completed, "
        f"{sum(r.failed for r in records)} failed, {last.running} running, {last.queue_length} queued at end",
    ]
    return "\n".join(lines)


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=cfg.project.seed)
    parser.add_argument("--start-hour", type=float, default=None, help="clock time at reset (default from config)")
    parser.add_argument("--out", type=Path, default=cfg.paths.data / "baseline_60min.csv")
    args = parser.parse_args()

    records = run(cfg, args.minutes, args.seed, args.start_hour)
    digest = write_csv(records, args.out)
    print(summarise(records, cfg.datacenter.initial_temperatures_c))
    logger.info("wrote %d rows to %s", len(records), args.out)
    logger.info("sha256 %s", digest)


if __name__ == "__main__":
    main()
