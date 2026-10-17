import { motion } from "motion/react";
import { Flame, Leaf, ListChecks, Snowflake, Thermometer, Zap } from "lucide-react";
import { Suspense } from "react";
import { CoolingChart, DecisionFeed, PowerChart, RackCard, TemperatureChart, Transport } from "../components/Live";
import { Scene } from "../components/Scene";
import { AnimatedNumber, Legend, Loading, PageHeader, Panel, PanelHead } from "../components/ui";
import { RACK_COLORS, tempColor } from "../lib/colors";
import { useSim } from "../lib/store";

function Hud() {
  const sim = useSim((s) => s.sim)!;
  const hottest = sim.racks.reduce((a, b) => (a.temperature > b.temperature ? a : b));
  const items = [
    { icon: <Zap size={14} />, label: "Total power", value: sim.power.total_kw, digits: 1, unit: "kW", color: "#4a94ee" },
    { icon: <Snowflake size={14} />, label: "PUE", value: sim.power.pue, digits: 3, unit: "", color: "#38bdf8" },
    { icon: <Thermometer size={14} />, label: "Hottest rack", value: hottest.temperature, digits: 1, unit: `°C · ${hottest.name}`, color: tempColor(hottest.temperature) },
    { icon: <Leaf size={14} />, label: "Grid carbon", value: sim.grid.carbon, digits: 0, unit: "g/kWh", color: "#23b88a" },
    { icon: <ListChecks size={14} />, label: "Queue", value: sim.queue.length, digits: 0, unit: "jobs", color: "#e0a100" },
    { icon: <Flame size={14} />, label: "SLA failures", value: sim.counters.failed ?? 0, digits: 0, unit: "", color: "#f43f5e" },
  ];
  return (
    <div className="pointer-events-none absolute inset-x-3 bottom-3 grid grid-cols-3 gap-2 lg:grid-cols-6">
      {items.map((it, i) => (
        <motion.div key={it.label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 + i * 0.05 }}
          className="rounded-xl border border-white/10 bg-[#070a10]/92 px-3 py-2">
          <div className="flex min-w-0 items-center gap-1.5 text-[10px] text-ink-3" style={{ color: it.color }}>{it.icon}<span className="eyebrow truncate !text-[9px] !tracking-[0.1em] !text-ink-3">{it.label}</span></div>
          <div className="mt-0.5"><AnimatedNumber value={it.value} digits={it.digits} className="text-lg font-medium text-ink" /><span className="ml-1 text-[10px] text-ink-3">{it.unit}</span></div>
        </motion.div>
      ))}
    </div>
  );
}

export function Twin() {
  const sim = useSim((s) => s.sim);
  if (!sim) return <Loading rows={6} />;
  return (
    <div>
      <PageHeader eyebrow="Live simulation" title="Digital twin"
        lead="Four racks, one shared cooling plant. Every glow, fan turn and particle is driven by the simulator's state, minute by minute. Pick a controller and watch what it does." />
      <div className="mb-4"><Transport /></div>

      <div className="grid grid-cols-12 gap-4">
        <Panel pad={false} className="relative col-span-12 h-[560px] overflow-hidden xl:col-span-8">
          <Suspense fallback={null}><Scene /></Suspense>
          <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2">
            <span className="rounded-lg border border-white/10 bg-[#070a10]/92 px-2.5 py-1 text-[10.5px] text-ink-2">drag to orbit · scroll to zoom</span>
          </div>
          <Hud />
        </Panel>

        <Panel className="col-span-12 flex h-[560px] flex-col xl:col-span-4" delay={0.08}>
          <PanelHead eyebrow="Explainable control" title="Decision feed" />
          <div className="-mr-2 min-h-0 flex-1 overflow-y-auto pr-2"><DecisionFeed limit={10} /></div>
        </Panel>

        {sim.racks.map((r, i) => (
          <div key={r.id} className="col-span-12 sm:col-span-6 xl:col-span-3"><RackCard rack={r} zones={sim.zones} horizons={sim.horizons} delay={0.1 + i * 0.05} /></div>
        ))}

        <Panel className="col-span-12 xl:col-span-6" delay={0.2}>
          <PanelHead eyebrow="Last 6 hours" title="Rack temperature" right={<Legend items={sim.racks.map((r, i) => ({ label: r.name, color: RACK_COLORS[i] }))} />} />
          <TemperatureChart />
        </Panel>
        <Panel className="col-span-12 md:col-span-6 xl:col-span-3" delay={0.25}>
          <PanelHead eyebrow="kW, stacked" title="Power draw" right={<Legend items={[{ label: "IT", color: "#4a94ee" }, { label: "Cooling", color: "#38bdf8" }]} />} />
          <PowerChart height={230} />
        </Panel>
        <Panel className="col-span-12 md:col-span-6 xl:col-span-3" delay={0.3}>
          <PanelHead eyebrow="3-minute ramp" title="Cooling" right={<Legend items={[{ label: "Set-point", color: "#a3aebb", dashed: true }, { label: "Delivered", color: "#38bdf8" }]} />} />
          <CoolingChart height={230} />
        </Panel>
      </div>
    </div>
  );
}
