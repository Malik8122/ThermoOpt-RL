import { AnimatePresence, motion } from "motion/react";
import { Award, FlaskConical } from "lucide-react";
import { useMemo, useState } from "react";
import { CartesianGrid, Cell, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { AXIS, EASE, GRID, Legend, Loading, PageHeader, Panel, PanelHead, Pill } from "../components/ui";
import { FAMILY } from "../lib/colors";
import { fmt, useResource } from "../lib/format";

type Entry = { key: string; label: string; family: string; step: number; primary: boolean; note: string; metrics: Record<string, { mean: number; std: number }>; per_seed: Record<string, number>[] };
type Metric = { key: string; label: string; unit: string; better: "higher" | "lower" };

const HEADLINE = ["total_reward", "sla_failures", "energy_kwh", "carbon_kg", "cost_inr", "critical_rack_minutes", "warning_or_worse_rack_minutes", "completed_jobs", "cooling_changes", "invalid_action_rate"];
const digitsFor = (m: string) => (m === "invalid_action_rate" ? 3 : m.includes("temperature") ? 1 : 0);

function Ranking({ entries, metric }: { entries: Entry[]; metric: Metric }) {
  const rows = useMemo(() => entries.filter((e) => e.metrics[metric.key]).map((e) => ({ ...e, value: e.metrics[metric.key].mean, std: e.metrics[metric.key].std }))
    .sort((a, b) => (metric.better === "higher" ? b.value - a.value : a.value - b.value)), [entries, metric]);
  const scale = Math.max(...rows.map((r) => Math.abs(r.value) + r.std), 1e-9);
  const seedMax = Math.max(scale, ...rows.flatMap((r) => r.per_seed.map((s) => Math.abs(s[metric.key] ?? 0))));
  const d = digitsFor(metric.key);
  return (
    <div className="space-y-1.5">
      <AnimatePresence initial={false}>
        {rows.map((r, i) => {
          const color = FAMILY[r.family].color;
          const width = (Math.abs(r.value) / seedMax) * 100;
          return (
            <motion.div key={r.key} layout transition={{ type: "spring", stiffness: 260, damping: 30 }} initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
              className={`group grid grid-cols-[26px_190px_1fr_120px] items-center gap-3 rounded-xl px-2 py-1.5 transition-colors hover:bg-white/[0.04] ${r.primary ? "" : "opacity-80"}`}>
              <span className={`num grid h-6 w-6 place-items-center rounded-lg text-[11px] ${i === 0 ? "bg-warm/20 text-warm" : "text-ink-3"}`}>{i === 0 ? <Award size={13} /> : i + 1}</span>
              <span className="min-w-0">
                <span className="flex items-center gap-2 truncate text-[13px] font-semibold text-ink"><span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />{r.label}</span>
                {!r.primary && <span className="block truncate pl-4 text-[10.5px] text-ink-3" title={r.note}>{r.note}</span>}
              </span>
              <div className="relative h-7">
                <div className="absolute inset-y-1 left-0 right-0 rounded-md bg-white/[0.035]" />
                <motion.div className="absolute inset-y-1 left-0 rounded-md" animate={{ width: `${width}%` }} transition={{ duration: 0.7, ease: EASE }}
                  style={{ background: r.primary ? `linear-gradient(90deg, ${color}44, ${color})` : `repeating-linear-gradient(135deg, ${color}66 0 6px, ${color}33 6px 12px)`, boxShadow: r.primary ? `0 0 16px ${color}44` : undefined }} />
                {/* mean ± std whisker */}
                <motion.div className="absolute top-1/2 h-px bg-ink/70" animate={{ left: `${((Math.abs(r.value) - r.std) / seedMax) * 100}%`, width: `${((2 * r.std) / seedMax) * 100}%` }} transition={{ duration: 0.7, ease: EASE }} />
                {/* one dot per evaluation day */}
                {r.per_seed.map((s) => s[metric.key] !== undefined && (
                  <motion.span key={s.seed} title={`seed ${s.seed}: ${fmt(s[metric.key], d)}`} className="absolute top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-base bg-ink"
                    animate={{ left: `${(Math.abs(s[metric.key]) / seedMax) * 100}%` }} transition={{ duration: 0.7, ease: EASE }} />
                ))}
              </div>
              <span className="text-right"><span className="num text-[13px] text-ink">{fmt(r.value, d)}</span><span className="num ml-1 text-[10.5px] text-ink-3">± {fmt(r.std, d)}</span></span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

function TradeOff({ entries }: { entries: Entry[] }) {
  // Controllers with identical results (Policy and Value Iteration) share one point and one label.
  const merged: { label: string; family: string; x: number; y: number; z: number; up: boolean }[] = [];
  for (const e of entries.filter((c) => c.primary)) {
    const x = e.metrics.energy_kwh.mean, y = e.metrics.sla_failures.mean;
    const same = merged.find((m) => Math.abs(m.x - x) < 0.5 && Math.abs(m.y - y) < 0.5);
    if (same) same.label = `${same.label.replace(" Iteration", "")} / ${e.label}`;
    else merged.push({ label: e.label, family: e.family, x, y, z: Math.max(4, e.metrics.critical_rack_minutes.mean), up: true });
  }
  // neighbours along the x-axis alternate label side so they do not collide
  const data = merged.sort((a, b) => a.x - b.x).map((d, i) => ({ ...d, up: i % 2 === 0 }));
  return (
    <ResponsiveContainer width="100%" height={330}>
      <ScatterChart margin={{ top: 16, right: 30, bottom: 26, left: 4 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.055)" />
        <XAxis dataKey="x" type="number" name="Energy" domain={[(min: number) => Math.floor((min - 15) / 10) * 10, (max: number) => Math.ceil((max + 15) / 10) * 10]} {...AXIS} tickCount={6} tickFormatter={(v) => String(Math.round(v))} label={{ value: "energy per day (kWh)  →  worse", position: "bottom", fill: "#66717f", fontSize: 11, offset: 6 }} />
        <YAxis dataKey="y" type="number" name="SLA failures" domain={[0, (max: number) => Math.ceil((max + 25) / 20) * 20]} {...AXIS} tickCount={6} tickFormatter={(v) => String(Math.round(v))} label={{ value: "SLA failures per day", angle: -90, position: "insideLeft", fill: "#66717f", fontSize: 11, dy: 50 }} />
        <ZAxis dataKey="z" range={[90, 900]} name="Critical rack-minutes" />
        <Tooltip cursor={{ stroke: "rgba(255,255,255,0.15)" }} content={({ active, payload }: any) => active && payload?.length ? (
          <div className="rounded-xl border border-line-strong bg-[#0b1017]/95 px-3 py-2 text-xs shadow-2xl">
            <div className="mb-1 font-semibold text-ink">{payload[0].payload.label}</div>
            <div className="text-ink-2">energy <span className="num text-ink">{fmt(payload[0].payload.x, 0)}</span> kWh</div>
            <div className="text-ink-2">SLA failures <span className="num text-ink">{fmt(payload[0].payload.y, 0)}</span></div>
            <div className="text-ink-2">critical <span className="num text-ink">{fmt(payload[0].payload.z, 1)}</span> rack-min</div>
          </div>) : null} />
        <Scatter data={data} isAnimationActive animationDuration={900}>
          {data.map((d) => <Cell key={d.label} fill={FAMILY[d.family].color} fillOpacity={0.55} stroke={FAMILY[d.family].color} strokeWidth={1.5} />)}
          <LabelList dataKey="label" content={({ x, y, width, height, value, index }: any) => {
            const d = data[index];
            const cx = Number(x) + Number(width) / 2, r = Number(height) / 2;
            return <text x={cx} y={d.up ? Number(y) - 8 : Number(y) + 2 * r + 15} textAnchor="middle" fill="#a3aebb" fontSize={11} fontFamily="Manrope Variable">{value}</text>;
          }} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function Compare() {
  const [data] = useResource<{ protocol: any; controllers: Entry[]; metrics: Metric[] }>("/api/comparison");
  const [metricKey, setMetricKey] = useState("total_reward");
  const [ablations, setAblations] = useState(false);
  if (!data) return <Loading rows={8} />;
  const metric = data.metrics.find((m) => m.key === metricKey)!;
  const entries = data.controllers.filter((c) => ablations || c.primary);
  const primary = data.controllers.filter((c) => c.primary);
  const learned = primary.filter((c) => c.family !== "baseline");
  const best = (list: Entry[], key: string, better: "higher" | "lower") => list.reduce((a, b) => ((better === "higher" ? b.metrics[key].mean > a.metrics[key].mean : b.metrics[key].mean < a.metrics[key].mean) ? b : a));
  const cards = [
    { title: "Best daily reward", e: best(primary, "total_reward", "higher"), key: "total_reward", unit: "" },
    { title: "Best learned controller", e: best(learned, "total_reward", "higher"), key: "total_reward", unit: "" },
    { title: "Fewest SLA failures", e: best(primary, "sla_failures", "lower"), key: "sla_failures", unit: "jobs / day" },
    { title: "Lowest energy", e: best(primary, "energy_kwh", "lower"), key: "energy_kwh", unit: "kWh / day" },
  ];
  return (
    <div>
      <PageHeader eyebrow={`Saved evaluation · seeds ${data.protocol?.seeds.join(", ")} · ${data.protocol?.episode_minutes / 60} h each`} title="Algorithm comparison"
        lead="Every controller ran the same five simulated days. Bars show the mean, the whisker ± one standard deviation, and each dot is one day." />

      <div className="mb-4 grid grid-cols-2 gap-4 xl:grid-cols-4">
        {cards.map((c, i) => (
          <Panel key={c.title} delay={i * 0.05} className="overflow-hidden">
            <div className="pointer-events-none absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-30 blur-2xl" style={{ background: FAMILY[c.e.family].color }} />
            <div className="eyebrow">{c.title}</div>
            <div className="mt-2 font-display text-xl font-semibold tracking-tight">{c.e.label}</div>
            <div className="mt-1 flex items-baseline gap-1.5"><span className="num text-sm text-ink-2">{fmt(c.e.metrics[c.key].mean, 0)}</span><span className="text-[11px] text-ink-3">{c.unit || "reward"}</span></div>
            <div className="mt-2"><Pill color={FAMILY[c.e.family].color} dot>{FAMILY[c.e.family].label}</Pill></div>
          </Panel>
        ))}
      </div>

      <Panel delay={0.15}>
        <PanelHead eyebrow={`${metric.better === "higher" ? "higher" : "lower"} is better${metric.unit ? ` · ${metric.unit}` : ""}`} title={`Ranked by ${metric.label.toLowerCase()}`}
          right={
            <button onClick={() => setAblations((a) => !a)} className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition ${ablations ? "border-ice/60 bg-ice/10 text-ice" : "border-line text-ink-2 hover:text-ink"}`}>
              <FlaskConical size={14} /> {ablations ? "Hide" : "Show"} ablations
            </button>} />
        <div className="mb-4 flex flex-wrap gap-1.5">
          {HEADLINE.map((k) => {
            const m = data.metrics.find((x) => x.key === k)!;
            const on = k === metricKey;
            return (
              <button key={k} onClick={() => setMetricKey(k)} className={`relative rounded-full px-3 py-1.5 text-[11.5px] font-medium transition ${on ? "text-base" : "text-ink-2 hover:text-ink"}`}>
                {on && <motion.span layoutId="metric-chip" className="absolute inset-0 rounded-full bg-ink" transition={{ type: "spring", stiffness: 400, damping: 34 }} />}
                {!on && <span className="absolute inset-0 rounded-full border border-line" />}
                <span className="relative">{m.label}</span>
              </button>
            );
          })}
        </div>
        <Ranking entries={entries} metric={metric} />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
          <Legend items={Object.values(FAMILY).map((f) => ({ label: f.label, color: f.color }))} />
          <span className="text-[11px] text-ink-3">Bar length is the magnitude of the value. Hatched bars are ablations, not headline results.</span>
        </div>
      </Panel>

      <div className="mt-4 grid grid-cols-12 gap-4">
        <Panel className="col-span-12 xl:col-span-6" delay={0.2}>
          <PanelHead eyebrow="Bottom-left is best · bubble size = critical rack-minutes" title="Energy vs reliability" />
          <TradeOff entries={data.controllers} />
          <p className="text-xs leading-relaxed text-ink-3">Controllers that use the least energy do so by finishing fewer jobs. The useful ones sit low on the chart without drifting far right.</p>
        </Panel>

        <Panel className="col-span-12 xl:col-span-6" delay={0.25}>
          <PanelHead eyebrow="Mean of 5 days" title="Scorecard" />
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead><tr className="border-b border-line">
                <th className="eyebrow py-2 pr-3 font-normal">Controller</th>
                {["total_reward", "energy_kwh", "carbon_kg", "cost_inr", "critical_rack_minutes", "sla_failures"].map((k) => <th key={k} className="eyebrow px-2 py-2 text-right font-normal">{data.metrics.find((m) => m.key === k)!.label}</th>)}
              </tr></thead>
              <tbody>
                {primary.map((c, i) => (
                  <motion.tr key={c.key} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 + i * 0.04 }} className="border-b border-line/60 transition-colors hover:bg-white/[0.03]">
                    <td className="py-2.5 pr-3"><span className="flex items-center gap-2 font-semibold text-ink"><span className="h-2 w-2 rounded-full" style={{ background: FAMILY[c.family].color }} />{c.label}</span></td>
                    {["total_reward", "energy_kwh", "carbon_kg", "cost_inr", "critical_rack_minutes", "sla_failures"].map((k) => {
                      const m = data.metrics.find((x) => x.key === k)!;
                      const isBest = best(primary, k, m.better).key === c.key;
                      return <td key={k} className={`num px-2 py-2.5 text-right ${isBest ? "font-semibold text-safe" : "text-ink-2"}`}>{fmt(c.metrics[k].mean, k === "critical_rack_minutes" ? 1 : 0)}</td>;
                    })}
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-3">Green marks the best value in each column. Differences of a few tens of reward points between the top controllers are within day-to-day noise.</p>
        </Panel>
      </div>
    </div>
  );
}
