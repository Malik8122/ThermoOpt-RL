import { AnimatePresence, motion } from "motion/react";
import { Check, ChevronDown, CircleSlash, FastForward, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { FAMILY, RACK_COLORS, STATUS, tempColor } from "../lib/colors";
import { fmt } from "../lib/format";
import { api, type Decision, type Rack, useChartHistory, useSim } from "../lib/store";
import { AXIS, AnimatedNumber, ChartTip, EASE, GRID, Legend, Meter, Pill, Segmented, TempGauge } from "./ui";

// ------------------------------------------------------------------------------------------------ transport
const SPEEDS = [{ value: "1", label: "1×" }, { value: "4", label: "4×" }, { value: "12", label: "12×" }, { value: "30", label: "30×" }, { value: "60", label: "60×" }];

export function ControllerPicker() {
  const controllers = useSim((s) => s.controllers);
  const active = useSim((s) => s.sim?.controller);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, []);
  const current = controllers.find((c) => c.key === active);
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-3 rounded-xl border border-line-strong bg-white/[0.06] py-2 pl-3 pr-2.5 text-left transition hover:bg-white/[0.1]">
        <span className="h-2 w-2 rounded-full" style={{ background: FAMILY[current?.family ?? "baseline"].color, boxShadow: `0 0 10px ${FAMILY[current?.family ?? "baseline"].color}` }} />
        <span>
          <span className="eyebrow block !text-[9px]">Controller</span>
          <span className="block text-[13px] font-semibold leading-tight text-ink">{current?.label ?? "…"}</span>
        </span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} className="text-ink-3"><ChevronDown size={16} /></motion.span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -6, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -6, scale: 0.97 }} transition={{ duration: 0.18, ease: EASE }}
            className="absolute left-0 top-full z-40 mt-2 w-[340px] origin-top-left rounded-2xl border border-line-strong bg-[#0a0f16] p-2 shadow-2xl">
            {controllers.map((c) => (
              <button key={c.key} disabled={!c.available} onClick={() => { api.controller(c.key); setOpen(false); }}
                className={`flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition ${c.available ? "hover:bg-white/[0.06]" : "cursor-not-allowed opacity-40"} ${c.key === active ? "bg-white/[0.05]" : ""}`}>
                <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: FAMILY[c.family].color }} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2 text-[13px] font-semibold text-ink">{c.label}<span className="eyebrow !text-[9px]">step {c.step}</span></span>
                  <span className="block text-[11.5px] leading-snug text-ink-3">{c.available ? c.description : "No trained model found."}</span>
                </span>
                {c.key === active && <Check size={15} className="mt-1 text-safe" />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Transport({ compact = false }: { compact?: boolean }) {
  const sim = useSim((s) => s.sim);
  if (!sim) return null;
  const progress = sim.minute / sim.episode_minutes;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <ControllerPicker />
      <motion.button whileTap={{ scale: 0.94 }} onClick={() => (sim.running ? api.pause() : api.start())}
        className={`flex h-[46px] items-center gap-2 rounded-xl px-5 text-sm font-semibold transition ${sim.running ? "border border-line-strong bg-white/[0.07] text-ink hover:bg-white/[0.12]" : "bg-ink text-base hover:bg-white"}`}
        style={sim.running ? undefined : { boxShadow: "0 0 30px rgba(94,234,212,0.35)" }}>
        {sim.running ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
        {sim.running ? "Pause" : "Run"}
      </motion.button>
      <Segmented id="speed" options={SPEEDS} value={String(sim.speed) as any} onChange={(v) => api.speed(Number(v))} />
      {!compact && (
        <div className="flex items-center gap-1.5">
          {[{ m: 1, icon: <SkipForward size={14} />, label: "+1 min" }, { m: 15, icon: <FastForward size={14} />, label: "+15" }, { m: 60, icon: <FastForward size={14} />, label: "+60" }].map((b) => (
            <button key={b.m} onClick={() => api.step(b.m)} className="flex h-9 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[11px] font-medium text-ink-2 transition hover:border-line-strong hover:text-ink">{b.icon}{b.label}</button>
          ))}
          <button onClick={() => api.reset({ seed: sim.seed })} title="Restart this day" className="flex h-9 items-center gap-1.5 rounded-lg border border-line px-2.5 text-[11px] font-medium text-ink-2 transition hover:border-line-strong hover:text-ink"><RotateCcw size={14} />Restart</button>
        </div>
      )}
      <div className="ml-auto hidden min-w-[180px] flex-1 items-center gap-3 xl:flex" style={{ maxWidth: 300 }}>
        <span className="num text-xs text-ink-2">{sim.clock}</span>
        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.07]">
          <motion.div className="thermal-bar absolute inset-y-0 left-0 rounded-full" animate={{ width: `${progress * 100}%` }} transition={{ ease: "linear", duration: 0.25 }} />
        </div>
        <span className="num text-[11px] text-ink-3">{fmt(progress * 100, 0)}%</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ rack card
export function RackCard({ rack, zones, horizons, delay = 0 }: { rack: Rack; zones: { warning: number; critical: number }; horizons: number[]; delay?: number }) {
  const c = tempColor(rack.temperature);
  const status = STATUS[rack.status];
  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: EASE, delay }}
      className="panel relative overflow-hidden p-4">
      <motion.div className="pointer-events-none absolute -right-12 -top-14 h-36 w-36 rounded-full blur-3xl" animate={{ background: c, opacity: rack.status === "NORMAL" ? 0.14 : 0.32 }} transition={{ duration: 1.2 }} />
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: RACK_COLORS[rack.id] }} />
            <span className="font-display text-sm font-semibold">{rack.name}</span>
          </div>
          <div className="mt-1.5 flex items-center gap-1.5">
            <Pill color={status} dot>{rack.status.toLowerCase()}</Pill>
            {rack.cooling_effectiveness < 1 && <Pill>cooling ×{rack.cooling_effectiveness}</Pill>}
          </div>
        </div>
        <div className="relative -mr-1 -mt-1">
          <TempGauge value={rack.temperature} warning={zones.warning} critical={zones.critical} size={112} />
          <div className="absolute inset-x-0 bottom-1 text-center">
            <AnimatedNumber value={rack.temperature} digits={1} className="text-xl font-medium" />
            <span className="text-[10px] text-ink-3"> °C</span>
          </div>
        </div>
      </div>
      <div className="mt-3 space-y-2">
        {[{ k: "CPU", v: rack.cpu, color: "#69b7ff" }, { k: "GPU", v: rack.gpu, color: "#34f5c5" }].map((u) => (
          <div key={u.k} className="flex items-center gap-2.5">
            <span className="eyebrow w-7 !text-[9px]">{u.k}</span>
            <Meter value={u.v} color={u.color} height={5} />
            <span className="num w-9 text-right text-[11px] text-ink-2">{fmt(u.v * 100, 0)}%</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-between border-t border-line pt-3">
        <span className="text-[11px] text-ink-3"><span className="num text-ink-2">{fmt(rack.it_kw, 2)}</span> kW · {rack.jobs.length} jobs</span>
        {rack.forecast && (
          <div className="flex items-center gap-1">
            {rack.forecast.map((f, i) => (
              <span key={i} title={`forecast in ${horizons[i]} min`} className="num rounded-md border px-1.5 py-0.5 text-[10px]" style={{ borderColor: `${tempColor(f)}55`, color: tempColor(f), background: `${tempColor(f)}14` }}>
                +{horizons[i]}′ {f.toFixed(1)}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ------------------------------------------------------------------------------------------------ decisions
const PART_COLOR: Record<string, string> = { energy: "#4a94ee", carbon: "#23b88a", thermal: "#f43f5e", sla: "#e0a100", action: "#9c8ff0", completion: "#5eead4" };

export function RewardBar({ parts }: { parts: Record<string, number> }) {
  const penalties = Object.entries(parts).filter(([, v]) => v < 0);
  const total = penalties.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]" title={Object.entries(parts).map(([k, v]) => `${k} ${v.toFixed(3)}`).join("  ·  ")}>
      {penalties.map(([k, v]) => <div key={k} style={{ width: `${(Math.abs(v) / total) * 100}%`, background: PART_COLOR[k] }} />)}
    </div>
  );
}

const DecisionRow = memo(function DecisionRow({ d }: { d: Decision }) {
  const tone = !d.valid ? "#f43f5e" : d.kind === "PLACE" ? "#5eead4" : d.kind === "MIGRATE" ? "#fbbf24" : "#8a96a5";
  return (
    <motion.li initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.25, ease: EASE }}>
      <div className="mb-2 rounded-xl border border-line bg-white/[0.025] p-3">
        <div className="mb-1.5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="num text-[11px] text-ink-3">{d.clock}</span>
            <Pill color={tone}>{d.valid ? d.kind.toLowerCase() : "rejected"}</Pill>
            {d.cooling !== "MAINTAIN" && <Pill color="#38bdf8">cooling {d.cooling === "INCREASE" ? "↑" : "↓"}</Pill>}
          </div>
          <span className="num text-[11px]" style={{ color: d.reward < -1 ? "#f43f5e" : "#a3aebb" }}>{d.reward.toFixed(3)}</span>
        </div>
        <p className="text-[12px] leading-snug text-ink-2">{d.explanation}</p>
        <div className="mt-2"><RewardBar parts={d.reward_parts} /></div>
      </div>
    </motion.li>
  );
});

export function DecisionFeed({ limit = 14 }: { limit?: number }) {
  const decisions = useSim((s) => s.decisions);
  const shown = useMemo(() => decisions.slice(0, limit), [decisions, limit]);
  return (
    <div>
      <div className="mb-3"><Legend items={Object.entries(PART_COLOR).filter(([k]) => k !== "completion").map(([k, color]) => ({ label: k, color }))} /></div>
      {shown.length === 0 ? (
        <div className="flex items-center gap-2 rounded-xl border border-dashed border-line px-3 py-6 text-xs text-ink-3"><CircleSlash size={14} /> No decisions yet. Press Run.</div>
      ) : (
        <ul className="m-0 list-none p-0">{shown.map((d) => <DecisionRow key={`${d.minute}-${d.controller}`} d={d} />)}</ul>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ live charts
export const TemperatureChart = memo(function TemperatureChart({ height = 230 }: { height?: number }) {
  const history = useChartHistory();
  const zones = useSim((s) => s.sim?.zones);
  const racks = useSim((s) => s.sim?.racks);
  const names = useMemo(() => racks?.map((r) => r.name) ?? [], [racks]);
  const data = useMemo(() => history.map((p) => ({ clock: p.clock, r0: p.temps[0], r1: p.temps[1], r2: p.temps[2], r3: p.temps[3] })), [history]);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="clock" {...AXIS} minTickGap={48} />
        <YAxis {...AXIS} domain={[17, (max: number) => Math.max(31, Math.ceil(max + 1))]} tickCount={6} unit="°" />
        {zones && <ReferenceLine y={zones.warning} stroke="#fbbf24" strokeDasharray="4 5" strokeOpacity={0.7} label={{ value: "warning", position: "insideTopRight", fill: "#fbbf24", fontSize: 10 }} />}
        {zones && <ReferenceLine y={zones.critical} stroke="#f43f5e" strokeDasharray="4 5" strokeOpacity={0.8} label={{ value: "critical", position: "insideTopRight", fill: "#f43f5e", fontSize: 10 }} />}
        <Tooltip content={<ChartTip unit="°C" digits={1} />} cursor={{ stroke: "rgba(255,255,255,0.18)" }} />
        {[0, 1, 2, 3].map((i) => <Line key={i} type="monotone" dataKey={`r${i}`} name={names[i] ?? `Rack ${i + 1}`} stroke={RACK_COLORS[i]} strokeWidth={2} dot={false} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  );
});

export const PowerChart = memo(function PowerChart({ height = 200 }: { height?: number }) {
  const history = useChartHistory();
  const data = useMemo(() => history.map((p) => ({ clock: p.clock, it: p.it_kw, cooling: p.cooling_kw })), [history]);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <defs>
          <linearGradient id="gIt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#4a94ee" stopOpacity={0.55} /><stop offset="1" stopColor="#4a94ee" stopOpacity={0.04} /></linearGradient>
          <linearGradient id="gCool" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#38bdf8" stopOpacity={0.5} /><stop offset="1" stopColor="#38bdf8" stopOpacity={0.04} /></linearGradient>
        </defs>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="clock" {...AXIS} minTickGap={48} />
        <YAxis {...AXIS} tickCount={5} />
        <Tooltip content={<ChartTip unit="kW" digits={1} />} cursor={{ stroke: "rgba(255,255,255,0.18)" }} />
        <Area type="monotone" dataKey="it" name="IT" stackId="p" stroke="#4a94ee" strokeWidth={2} fill="url(#gIt)" isAnimationActive={false} />
        <Area type="monotone" dataKey="cooling" name="Cooling" stackId="p" stroke="#38bdf8" strokeWidth={2} fill="url(#gCool)" isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
});

export const CoolingChart = memo(function CoolingChart({ height = 200 }: { height?: number }) {
  const history = useChartHistory();
  const data = useMemo(() => history.map((p) => ({ clock: p.clock, target: p.cooling_target * 100, delivered: p.cooling_delivered * 100 })), [history]);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="clock" {...AXIS} minTickGap={48} />
        <YAxis {...AXIS} domain={[0, 100]} ticks={[0, 20, 40, 60, 80, 100]} unit="%" />
        <Tooltip content={<ChartTip unit="%" digits={0} />} cursor={{ stroke: "rgba(255,255,255,0.18)" }} />
        <Line type="stepAfter" dataKey="target" name="Set-point" stroke="#a3aebb" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="delivered" name="Delivered" stroke="#38bdf8" strokeWidth={2.2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
});
