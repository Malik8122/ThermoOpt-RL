import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartTip, EASE, Empty, GRID, Legend, Loading, PageHeader, Panel, PanelHead, Segmented } from "../components/ui";
import { SERIES } from "../lib/colors";
import { fmt, movingAverage, tick, useResource } from "../lib/format";

type Run = { key: string; label: string; episodes: Record<string, any>; updates: Record<string, any> | null };
type ChartSpec = { key: string; title: string; unit?: string; source?: "episodes" | "updates"; log?: boolean; smooth?: boolean; digits?: number };

const ALGOS = [
  { value: "q_learning", label: "Q-Learning" },
  { value: "dqn", label: "DQN" },
  { value: "ppo", label: "PPO" },
] as const;

const BLURB: Record<string, string> = {
  q_learning: "A 144 × 9 table learned from 1,000 simulated days. Exploration decays from 100 % to 5 % of actions.",
  dqn: "A neural Q-network on 30 features with replay and a target network. The masked run is compared with one trained without action masking.",
  ppo: "An actor-critic policy improved in small clipped steps. Curves are plotted against environment steps so runs of different length line up.",
};

const SHARED: ChartSpec[] = [
  { key: "total_reward", title: "Daily reward while exploring", digits: 0 },
  { key: "sla_failures", title: "SLA failures per day", unit: "jobs", digits: 0 },
  { key: "critical_rack_minutes", title: "Critical rack-minutes per day", unit: "min", digits: 0 },
  { key: "energy_kwh", title: "Energy per day", unit: "kWh", digits: 0 },
];
const SPECIFIC: Record<string, ChartSpec[]> = {
  q_learning: [
    { key: "epsilon", title: "Exploration rate ε", smooth: false, digits: 3 },
    { key: "mean_abs_q_update", title: "Mean |ΔQ| per update", digits: 3 },
    { key: "states_visited_so_far", title: "States visited (of 144)", smooth: false, digits: 0 },
  ],
  dqn: [
    { key: "epsilon", title: "Exploration rate ε", smooth: false, digits: 3 },
    { key: "mean_loss", title: "Huber loss", log: true, digits: 3 },
    { key: "mean_q", title: "Mean predicted Q", digits: 2 },
    { key: "invalid_action_rate", title: "Invalid-action rate", digits: 3 },
  ],
  ppo: [
    { key: "entropy", title: "Policy entropy", unit: "nats", source: "updates", smooth: false, digits: 2 },
    { key: "value_loss", title: "Value loss", source: "updates", log: true, smooth: false, digits: 2 },
    { key: "approx_kl", title: "Approximate KL per update", source: "updates", smooth: false, digits: 4 },
    { key: "explained_variance", title: "Critic explained variance", source: "updates", smooth: false, digits: 2 },
  ],
};

/** Rows keyed by x, with `<run>` (raw) and `<run>_ma` (moving average) columns for every visible run. */
function buildRows(runs: Run[], spec: ChartSpec, xKey: string, visible: Set<string>) {
  const rows = new Map<number, any>();
  for (const run of runs) {
    if (!visible.has(run.key)) continue;
    const src = (spec.source ?? "episodes") === "updates" ? run.updates : run.episodes;
    const ys: (number | null)[] | undefined = src?.[spec.key];
    const xs: number[] | undefined = src?.[xKey] ?? src?.episode ?? src?.update;
    if (!ys || !xs) continue;
    const window = Math.max(5, Math.round(ys.length / 22));
    const ma = spec.smooth === false ? null : movingAverage(ys, window);
    ys.forEach((y, i) => {
      const x = xKey === "env_steps" ? Math.round(xs[i] / 100) / 10 : xs[i];
      const row = rows.get(x) ?? { x };
      row[run.key] = spec.log && y !== null && y <= 0 ? null : y;
      if (ma) row[`${run.key}_ma`] = ma[i];
      rows.set(x, row);
    });
  }
  return [...rows.values()].sort((a, b) => a.x - b.x);
}

function Curve({ runs, spec, xKey, visible, colors, delay }: { runs: Run[]; spec: ChartSpec; xKey: string; visible: Set<string>; colors: Record<string, string>; delay: number }) {
  const rows = useMemo(() => buildRows(runs, spec, xKey, visible), [runs, spec, xKey, visible]);
  const smooth = spec.smooth !== false;
  const shown = runs.filter((r) => visible.has(r.key) && rows.some((row) => row[r.key] !== undefined && row[r.key] !== null));
  return (
    <Panel delay={delay}>
      <PanelHead eyebrow={smooth ? "raw (faint) + moving average" : spec.unit ?? " "} title={spec.title} />
      {shown.length === 0 ? <Empty>Not recorded for this algorithm</Empty> : (
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={rows} margin={{ top: 6, right: 18, bottom: 0, left: -4 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} {...AXIS} tickCount={6} tickFormatter={(v) => (xKey === "env_steps" ? `${Math.round(v)}k` : String(Math.round(v)))} />
            <YAxis {...AXIS} scale={spec.log ? "log" : "auto"} domain={["auto", "auto"]} tickCount={5} tickFormatter={tick} width={46} />
            <Tooltip content={<ChartTip digits={spec.digits ?? 2} unit={spec.unit} labelFormat={(v: number) => (xKey === "env_steps" ? `${v}k steps` : `episode ${v}`)} />} cursor={{ stroke: "rgba(255,255,255,0.18)" }} />
            {shown.map((r) => <Line key={r.key} dataKey={r.key} name={smooth ? `${r.label} (raw)` : r.label} stroke={colors[r.key]} strokeWidth={smooth ? 1 : 2} strokeOpacity={smooth ? 0.22 : 1} dot={false} connectNulls isAnimationActive animationDuration={900} />)}
            {smooth && shown.map((r) => <Line key={`${r.key}_ma`} dataKey={`${r.key}_ma`} name={r.label} stroke={colors[r.key]} strokeWidth={2.2} dot={false} connectNulls isAnimationActive animationDuration={1200} />)}
          </LineChart>
        </ResponsiveContainer>
      )}
    </Panel>
  );
}

export function Training() {
  const [algo, setAlgo] = useState<(typeof ALGOS)[number]["value"]>("ppo");
  const [data, error] = useResource<{ label: string; runs: Run[] }>(`/api/training/${algo}/metrics`);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const runs = data?.runs ?? [];
  const colors = useMemo(() => Object.fromEntries(runs.map((r, i) => [r.key, SERIES[i]])), [runs]);
  const visible = useMemo(() => new Set(runs.filter((r) => !hidden[`${algo}:${r.key}`]).map((r) => r.key)), [runs, hidden, algo]);
  const xKey = algo === "ppo" ? "env_steps" : "episode";
  const main = runs[0]?.episodes;
  const last = (key: string, n = 20) => { const v: number[] = (main?.[key] ?? []).filter((x: any) => x !== null); const tail = v.slice(-n); return tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : null; };

  return (
    <div>
      <PageHeader eyebrow="Saved training runs" title="RL training"
        lead={BLURB[algo]}
        right={<Segmented id="algo" options={ALGOS as any} value={algo} onChange={setAlgo} />} />

      {error && <Empty>No saved runs for this algorithm yet ({error}).</Empty>}
      {!data && !error && <Loading rows={5} />}
      {data && (
        <motion.div key={algo} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, ease: EASE }}>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {runs.map((r) => {
                const off = hidden[`${algo}:${r.key}`];
                return (
                  <button key={r.key} onClick={() => setHidden((h) => ({ ...h, [`${algo}:${r.key}`]: !off }))}
                    className="flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition" style={{ borderColor: off ? "rgba(255,255,255,0.08)" : `${colors[r.key]}77`, background: off ? "transparent" : `${colors[r.key]}14`, color: off ? "#66717f" : "#eef2f7" }}>
                    <span className="h-2 w-2 rounded-full" style={{ background: off ? "#66717f" : colors[r.key] }} />{r.label}
                    <span className="num text-[10px] text-ink-3">{r.episodes.rows} days</span>
                  </button>
                );
              })}
            </div>
            <div className="flex gap-5 text-xs text-ink-3">
              <span>last-20-day reward <span className="num ml-1 text-ink">{fmt(last("total_reward"), 0)}</span></span>
              <span>SLA failures <span className="num ml-1 text-ink">{fmt(last("sla_failures"), 0)}</span></span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {SHARED.map((s, i) => <Curve key={s.key} runs={runs} spec={s} xKey={xKey} visible={visible} colors={colors} delay={i * 0.05} />)}
          </div>
          <div className="mb-3 mt-7 flex items-center justify-between">
            <h2 className="font-display text-lg font-semibold tracking-tight">Inside the learner</h2>
            <Legend items={runs.filter((r) => visible.has(r.key)).map((r) => ({ label: r.label, color: colors[r.key] }))} />
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {SPECIFIC[algo].map((s, i) => <Curve key={s.key} runs={runs} spec={s} xKey={s.source === "updates" ? "env_steps" : xKey} visible={visible} colors={colors} delay={0.2 + i * 0.05} />)}
          </div>
        </motion.div>
      )}
    </div>
  );
}
