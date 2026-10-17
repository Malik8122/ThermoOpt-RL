import { motion } from "motion/react";
import { ArrowUpRight, Coins, Cpu, Leaf, ShieldAlert, Snowflake, Zap } from "lucide-react";
import { Suspense, memo, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Transport } from "../components/Live";
import { Scene } from "../components/Scene";
import { AXIS, AnimatedNumber, ChartTip, EASE, GRID, Legend, Loading, Panel, PanelHead, Pill, Stat } from "../components/ui";
import { FAMILY } from "../lib/colors";
import { fmt, useResource } from "../lib/format";
import { api, useSim } from "../lib/store";

const Leaderboard = memo(function Leaderboard() {
  const [data] = useResource("/api/comparison");
  const rows = useMemo(() => {
    if (!data) return [];
    return data.controllers.filter((c: any) => c.primary).map((c: any) => ({ ...c, reward: c.metrics.total_reward.mean, sla: c.metrics.sla_failures.mean }))
      .sort((a: any, b: any) => b.reward - a.reward);
  }, [data]);
  if (!data) return <Loading rows={6} />;
  const worst = Math.min(...rows.map((r: any) => r.reward));
  return (
    <div className="space-y-2.5">
      {rows.map((r: any, i: number) => (
        <motion.div key={r.key} initial={{ opacity: 0, x: -18 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 + i * 0.06, ease: EASE, duration: 0.5 }} className="flex items-center gap-3">
          <span className="num w-4 text-[11px] text-ink-3">{i + 1}</span>
          <span className="w-[118px] truncate text-[12.5px] font-medium text-ink">{r.label}</span>
          <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-white/[0.04]">
            <motion.div className="absolute inset-y-0 left-0 rounded-md" initial={{ width: 0 }} animate={{ width: `${(r.reward / worst) * 100}%` }} transition={{ delay: 0.25 + i * 0.06, duration: 0.9, ease: EASE }}
              style={{ background: `linear-gradient(90deg, ${FAMILY[r.family].color}55, ${FAMILY[r.family].color})` }} />
          </div>
          <span className="num w-14 text-right text-[12px] text-ink">{fmt(r.reward, 0)}</span>
          <span className="num w-16 text-right text-[11px] text-ink-3">{fmt(r.sla, 0)} fails</span>
        </motion.div>
      ))}
      <div className="flex items-center justify-between pt-2">
        <Legend items={Object.values(FAMILY).map((f) => ({ label: f.label, color: f.color }))} />
        <span className="text-[10.5px] text-ink-3">daily reward, 5 simulated days · shorter bar is better</span>
      </div>
    </div>
  );
});

const GridProfile = memo(function GridProfile() {
  const [meta] = useResource("/api/meta");
  const hour = useSim((s) => Math.round((s.sim?.hour ?? 8) * 4) / 4);
  const data = useMemo(() => (meta ? meta.grid_profile.hour.map((h: number, i: number) => ({ h, carbon: meta.grid_profile.carbon[i], price: meta.grid_profile.price[i] })) : []), [meta]);
  if (!meta) return <Loading />;
  return (
    <ResponsiveContainer width="100%" height={190}>
      <AreaChart data={data} margin={{ top: 10, right: 6, bottom: 0, left: -18 }}>
        <defs><linearGradient id="gCarbon" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#23b88a" stopOpacity={0.5} /><stop offset="1" stopColor="#23b88a" stopOpacity={0.02} /></linearGradient></defs>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="h" type="number" domain={[0, 24]} ticks={[0, 4, 8, 12, 16, 20, 24]} tickFormatter={(h) => `${String(h).padStart(2, "0")}:00`} {...AXIS} />
        <YAxis {...AXIS} domain={[200, 560]} tickCount={5} />
        <Tooltip content={<ChartTip unit="g/kWh" digits={0} labelFormat={(h: number) => `${String(Math.floor(h)).padStart(2, "0")}:${String(Math.round((h % 1) * 60)).padStart(2, "0")}`} />} />
        <ReferenceLine y={meta.high_carbon_threshold} stroke="#fbbf24" strokeDasharray="4 5" strokeOpacity={0.6} label={{ value: "high-carbon threshold", position: "insideBottomRight", fill: "#fbbf24", fontSize: 10 }} />
        <Area type="monotone" dataKey="carbon" name="Carbon intensity" stroke="#23b88a" strokeWidth={2} fill="url(#gCarbon)" isAnimationActive={false} />
        <ReferenceLine x={hour} stroke="#eef2f7" strokeWidth={1.5} label={{ value: "now", position: "top", fill: "#eef2f7", fontSize: 10 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
});

/** Every controller in the project. Clicking one loads it into the live twin. */
const LineUp = memo(function LineUp() {
  const controllers = useSim((s) => s.controllers);
  const active = useSim((s) => s.sim?.controller);
  const [data] = useResource("/api/comparison");
  const navigate = useNavigate();
  const score = (key: string) => data?.controllers.find((c: any) => c.key === key)?.metrics;
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {controllers.map((c, i) => {
        const color = FAMILY[c.family].color;
        const m = score(c.key);
        return (
          <motion.button key={c.key} disabled={!c.available} whileHover={c.available ? { y: -4 } : undefined} whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: c.available ? 1 : 0.45, y: 0 }} transition={{ delay: 0.3 + i * 0.04, duration: 0.45, ease: EASE }}
            onClick={async () => { await api.controller(c.key); navigate("/twin"); }}
            className="group relative overflow-hidden rounded-2xl border bg-white/[0.025] p-4 text-left transition-colors hover:bg-white/[0.05]"
            style={{ borderColor: c.key === active ? `${color}99` : "rgba(255,255,255,0.075)" }}>
            <div className="pointer-events-none absolute -right-10 -top-10 h-24 w-24 rounded-full opacity-20 blur-2xl transition-opacity group-hover:opacity-50" style={{ background: color }} />
            <div className="flex items-center justify-between">
              <span className="eyebrow !text-[9px]">step {c.step}</span>
              {c.key === active ? <Pill color={color} dot>driving now</Pill> : <ArrowUpRight size={14} className="text-ink-3 transition group-hover:text-ink" />}
            </div>
            <div className="mt-2 font-display text-[15px] font-semibold tracking-tight">{c.label}</div>
            <div className="mt-1 min-h-[34px] text-[11.5px] leading-snug text-ink-3">{c.description}</div>
            <div className="mt-3 flex items-center justify-between border-t border-line pt-2.5 text-[11px]">
              <span style={{ color }}>{FAMILY[c.family].label}</span>
              {m && <span className="text-ink-3">reward <span className="num text-ink">{fmt(m.total_reward.mean, 0)}</span></span>}
            </div>
          </motion.button>
        );
      })}
    </div>
  );
});

export function Overview() {
  const sim = useSim((s) => s.sim);
  const [meta] = useResource("/api/meta");
  return (
    <div>
      {/* hero */}
      <div className="grid grid-cols-12 gap-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE }} className="col-span-12 flex flex-col justify-center py-4 xl:col-span-5">
          <div className="mb-4 flex flex-wrap gap-2">
            <Pill color="#38bdf8" dot>Reinforcement learning</Pill>
            <Pill color="#2dd4bf" dot>Digital twin</Pill>
            <Pill color="#fbbf24" dot>Carbon-aware</Pill>
          </div>
          <h1 className="font-display text-[46px] font-semibold leading-[1.02] tracking-[-0.035em]">
            Teaching a data centre<br />to <span className="thermal-text">keep its cool</span>.
          </h1>
          <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-ink-2">
            ThermoOpt trains controllers that decide, every minute, where each job runs and how hard to cool. The goal: use less
            energy and carbon without overheating a rack or missing a deadline.
            {meta && <> The twin has <span className="text-ink">{meta.racks} racks</span>, <span className="text-ink">{meta.servers} servers</span> and <span className="text-ink">{meta.gpu_nodes} GPU nodes</span>.</>}
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link to="/twin" className="group flex items-center gap-2 rounded-xl bg-ink px-5 py-3 text-sm font-semibold text-base transition hover:bg-white" style={{ boxShadow: "0 0 34px rgba(94,234,212,0.3)" }}>
              Open the live twin <ArrowUpRight size={16} className="transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <Link to="/compare" className="rounded-xl border border-line-strong px-5 py-3 text-sm font-semibold text-ink transition hover:bg-white/[0.06]">Compare the algorithms</Link>
          </div>
        </motion.div>

        <Panel pad={false} className="relative col-span-12 h-[430px] overflow-hidden xl:col-span-7" delay={0.1}>
          <Suspense fallback={null}><Scene compact /></Suspense>
          <div className="absolute inset-x-4 bottom-4 rounded-2xl border border-white/10 bg-[#070a10]/92 p-3"><Transport compact /></div>
        </Panel>
      </div>

      {/* live KPIs */}
      {sim ? (
        <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
          <Stat delay={0.05} label="Power now" value={sim.power.total_kw} unit="kW" accent="#4a94ee" icon={<Zap size={15} />} sub={<>IT <span className="num text-ink">{fmt(sim.power.it_kw)}</span> · cooling <span className="num text-ink">{fmt(sim.power.cooling_kw)}</span></>} />
          <Stat delay={0.1} label="PUE" value={sim.power.pue} digits={3} accent="#38bdf8" icon={<Snowflake size={15} />} sub={<>cooling at <span className="num text-ink">{fmt(sim.cooling.delivered * 100, 0)}%</span></>} />
          <Stat delay={0.15} label="Energy today" value={sim.totals.energy_kwh} unit="kWh" accent="#9c8ff0" icon={<Cpu size={15} />} sub={<>since 08:00, minute {sim.minute}</>} />
          <Stat delay={0.2} label="Carbon today" value={sim.totals.carbon_kg} unit="kg CO₂" accent="#23b88a" icon={<Leaf size={15} />} sub={<>grid at <span className="num text-ink">{fmt(sim.grid.carbon, 0)}</span> g/kWh</>} />
          <Stat delay={0.25} label="Cost today" value={sim.totals.cost_inr} digits={0} unit="₹" accent="#e0a100" icon={<Coins size={15} />} sub={<>tariff <span className="num text-ink">₹{fmt(sim.grid.price, 2)}</span>/kWh</>} />
          <Stat delay={0.3} label="SLA failures" value={sim.counters.failed ?? 0} digits={0} accent="#f43f5e" icon={<ShieldAlert size={15} />} sub={<><span className="num text-ink">{sim.counters.completed ?? 0}</span> jobs completed</>} />
        </div>
      ) : <div className="mt-4"><Loading rows={2} /></div>}

      <div className="mt-4 grid grid-cols-12 gap-4">
        <Panel className="col-span-12 xl:col-span-7" delay={0.2}>
          <PanelHead eyebrow="Saved evaluation · same 5 days for everyone" title="Controller leaderboard" right={<Link to="/compare" className="flex items-center gap-1 text-xs font-medium text-ice hover:underline">Details <ArrowUpRight size={13} /></Link>} />
          <Leaderboard />
        </Panel>
        <Panel className="col-span-12 xl:col-span-5" delay={0.25}>
          <PanelHead eyebrow="Synthetic 24-hour profile" title="Grid carbon intensity" />
          <GridProfile />
          <p className="mt-3 text-xs leading-relaxed text-ink-3">Carbon dips in the afternoon when solar peaks. A carbon-aware controller tries to run flexible jobs in that window.</p>
        </Panel>
      </div>

      <div className="mb-3 mt-8 flex items-end justify-between">
        <div>
          <div className="eyebrow mb-1">Eight controllers, from hand-written rules to deep RL</div>
          <h2 className="font-display text-xl font-semibold tracking-tight">Put one in charge</h2>
        </div>
        <span className="text-xs text-ink-3">click a card to hand it the live twin</span>
      </div>
      <LineUp />
    </div>
  );
}
