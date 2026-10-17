import { motion } from "motion/react";
import { memo, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Transport } from "../components/Live";
import { AXIS, AnimatedNumber, ChartTip, EASE, Empty, GRID, Legend, Loading, Meter, PageHeader, Panel, PanelHead, Pill, Segmented } from "../components/ui";
import { RACK_COLORS, tempColor } from "../lib/colors";
import { fmt, useResource } from "../lib/format";
import { useChartHistory, useSim } from "../lib/store";

const MODEL = "#4a94ee", PERSIST = "#8a96a5", TREND = "#ee7a45";

function LiveRack({ index }: { index: number }) {
  const rack = useSim((s) => s.sim!.racks[index]);
  const horizons = useSim((s) => s.sim!.horizons);
  const zones = useSim((s) => s.sim!.zones);
  if (!rack.forecast) return null;
  const pts = [rack.temperature, ...rack.forecast];
  const lo = Math.min(...pts, zones.warning - 4) - 0.5, hi = Math.max(...pts, zones.warning + 1) + 0.5;
  const W = 200, H = 74;
  const x = (i: number) => 10 + (i / 3) * (W - 20), y = (v: number) => H - 8 - ((v - lo) / (hi - lo)) * (H - 16);
  const path = pts.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  const delta = rack.forecast[0] - rack.temperature;
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05, ease: EASE }} className="panel p-4">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm font-semibold"><span className="h-2 w-2 rounded-full" style={{ background: RACK_COLORS[index] }} />{rack.name}</span>
        <Pill color={delta > 0.3 ? "#fb7a3c" : delta < -0.3 ? "#38bdf8" : undefined}>{delta > 0.3 ? "warming" : delta < -0.3 ? "cooling" : "steady"}</Pill>
      </div>
      <div className="mt-2 flex items-end justify-between gap-2">
        <div><AnimatedNumber value={rack.temperature} className="text-2xl font-medium" /><span className="text-xs text-ink-3"> °C now</span></div>
        <svg width={W} height={H} className="overflow-visible">
          <line x1={0} x2={W} y1={y(zones.warning)} y2={y(zones.warning)} stroke="#fbbf24" strokeDasharray="3 4" strokeOpacity={0.6} />
          <motion.path d={path} fill="none" stroke={tempColor(rack.forecast[2])} strokeWidth={2} strokeLinecap="round" animate={{ d: path }} transition={{ duration: 0.4 }} />
          {pts.map((v, i) => <motion.circle key={i} r={i === 0 ? 3.5 : 3} fill={i === 0 ? "#eef2f7" : tempColor(v)} animate={{ cx: x(i), cy: y(v) }} transition={{ duration: 0.4 }} />)}
        </svg>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-1.5">
        {rack.forecast.map((f, i) => (
          <div key={i} className="rounded-lg border border-line bg-black/20 px-2 py-1.5 text-center">
            <div className="eyebrow !text-[9px]">+{horizons[i]} min</div>
            <div className="num text-sm" style={{ color: tempColor(f) }}>{f.toFixed(1)}°</div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/** What the forecaster said vs what happened, from the live history. */
const Hindsight = memo(function Hindsight() {
  const history = useChartHistory(2000, 400);
  const sim = useSim((s) => s.sim)!;
  const [rack, setRack] = useState("2");
  const [h, setH] = useState("0");
  const r = Number(rack), hi = Number(h), horizon = sim.horizons[hi];
  const { rows, mae, n } = useMemo(() => {
    const made = new Map<number, number>();
    history.forEach((p) => { if (p.forecast) made.set(p.minute + horizon, p.forecast[r][hi]); });
    const rows = history.map((p) => ({ clock: p.clock, actual: p.temps[r], forecast: made.get(p.minute) ?? null, persistence: null as number | null }));
    const byMinute = new Map(history.map((p) => [p.minute, p.temps[r]]));
    history.forEach((p, i) => { rows[i].persistence = byMinute.get(p.minute - horizon) ?? null; });
    const pairs = rows.filter((x) => x.forecast !== null);
    const mae = pairs.length ? pairs.reduce((s, x) => s + Math.abs(x.forecast! - x.actual), 0) / pairs.length : null;
    return { rows, mae, n: pairs.length };
  }, [history, r, hi, horizon]);
  return (
    <Panel delay={0.15}>
      <PanelHead eyebrow={`Live · forecast made ${horizon} min earlier vs what happened`} title="Forecast in hindsight"
        right={<div className="flex flex-wrap gap-2">
          <Segmented id="hr" size="sm" options={sim.racks.map((x) => ({ value: String(x.id), label: x.name }))} value={rack} onChange={setRack} />
          <Segmented id="hh" size="sm" options={sim.horizons.map((x, i) => ({ value: String(i), label: `+${x} min` }))} value={h} onChange={setH} />
        </div>} />
      {n < 3 ? <Empty>Run the twin for a few minutes to compare forecasts with outcomes.</Empty> : (
        <>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="clock" {...AXIS} minTickGap={48} />
              <YAxis {...AXIS} domain={["auto", "auto"]} unit="°" tickCount={6} />
              <ReferenceLine y={sim.zones.warning} stroke="#fbbf24" strokeDasharray="4 5" strokeOpacity={0.6} />
              <Tooltip content={<ChartTip unit="°C" digits={1} />} cursor={{ stroke: "rgba(255,255,255,0.18)" }} />
              <Line dataKey="persistence" name={`Temperature ${horizon} min earlier`} stroke={PERSIST} strokeWidth={1.2} strokeDasharray="4 4" dot={false} isAnimationActive={false} connectNulls />
              <Line dataKey="actual" name="Actual" stroke="#eef2f7" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line dataKey="forecast" name="Forecast" stroke={MODEL} strokeWidth={2.2} dot={false} isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
            <Legend items={[{ label: "Actual", color: "#eef2f7" }, { label: "Forecast", color: MODEL }, { label: "No-change baseline", color: PERSIST, dashed: true }]} />
            <span className="text-xs text-ink-3">live error over {n} minutes <span className="num ml-1 text-ink">{fmt(mae, 2)} °C</span></span>
          </div>
        </>
      )}
    </Panel>
  );
});

/** Saved evaluation. Memoised and not subscribed to the live twin, so its chart animations run once. */
const Evaluation = memo(function Evaluation() {
  const [ev, error] = useResource<any>("/api/predictions/summary");
  const horizons = ev ? Object.keys(ev.test) : [];
  const bars = useMemo(() => (ev ? Object.keys(ev.test).map((h) => ({ h: `${h} min`, model: ev.test[h].model.mae, persistence: ev.test[h].persistence.mae, trend: ev.test[h].linear_trend.mae })) : []), [ev]);
  const curve = useMemo(() => (ev?.training.curve ? ev.training.curve.epoch.map((e: number, i: number) => ({ epoch: e, train: ev.training.curve.train_loss[i], val: ev.training.curve.val_loss[i] })) : []), [ev]);
  return (
    <>
      <div className="mb-3 mt-8 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="eyebrow mb-1">Saved evaluation · held-out days</div>
          <h2 className="font-display text-xl font-semibold tracking-tight">How good is the forecast?</h2>
        </div>
        {ev && <div className="flex gap-2"><Pill>{ev.model.architecture.toUpperCase()} · {ev.model.num_layers} × {ev.model.hidden_size}</Pill><Pill>{fmt(ev.model.parameters, 0)} parameters</Pill><Pill>best epoch {ev.training.best_epoch}</Pill></div>}
      </div>

      {error && <Empty>The forecaster has not been evaluated yet.</Empty>}
      {!ev && !error && <Loading rows={4} />}
      {ev && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {horizons.map((h, i) => {
              const t = ev.test[h], cut = t.mae_reduction_vs_persistence_pct, cross = ev.warning_crossing_detection[h];
              return (
                <Panel key={h} delay={i * 0.06} className="overflow-hidden">
                  <div className="pointer-events-none absolute -right-8 -top-10 h-28 w-28 rounded-full blur-3xl" style={{ background: MODEL, opacity: 0.1 + cut / 120 }} />
                  <div className="flex items-center justify-between"><span className="font-display text-lg font-semibold">{h} minutes</span><Pill color={cut > 25 ? "#2dd4bf" : cut > 12 ? "#fbbf24" : "#fb7a3c"} dot>{fmt(cut, 0)}% better than no-change</Pill></div>
                  <div className="mt-3 flex items-baseline gap-2"><span className="num text-[34px] font-medium leading-none">{fmt(t.model.mae, 2)}</span><span className="text-xs text-ink-3">°C mean error</span></div>
                  <div className="mt-4 space-y-2.5">
                    {[{ k: "Forecast accuracy (R²)", v: t.model.r2, c: MODEL }, { k: "Warning crossings caught", v: cross.model.recall, c: "#fbbf24" }, { k: "Alerts that were right", v: cross.model.precision ?? 0, c: "#2dd4bf" }].map((m) => (
                      <div key={m.k}>
                        <div className="mb-1 flex justify-between text-[11px]"><span className="text-ink-3">{m.k}</span><span className="num text-ink-2">{fmt(m.v * 100, m.v < 0.1 ? 1 : 0)}%</span></div>
                        <Meter value={m.v} color={m.c} height={4} />
                      </div>
                    ))}
                  </div>
                </Panel>
              );
            })}
          </div>

          <div className="mt-4 grid grid-cols-12 gap-4">
            <Panel className="col-span-12 xl:col-span-5" delay={0.2}>
              <PanelHead eyebrow="Mean error in °C · lower is better" title="Against simple baselines" right={<Legend items={[{ label: "LSTM", color: MODEL }, { label: "No change", color: PERSIST }, { label: "Linear trend", color: TREND }]} />} />
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={bars} margin={{ top: 8, right: 8, bottom: 0, left: -18 }} barGap={3} barCategoryGap="24%">
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="h" {...AXIS} />
                  <YAxis {...AXIS} tickCount={5} />
                  <Tooltip content={<ChartTip unit="°C" digits={2} />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="model" name="LSTM" fill={MODEL} radius={[4, 4, 0, 0]} animationDuration={900} />
                  <Bar dataKey="persistence" name="No change" fill={PERSIST} radius={[4, 4, 0, 0]} animationDuration={900} />
                  <Bar dataKey="trend" name="Linear trend" fill={TREND} radius={[4, 4, 0, 0]} animationDuration={900} />
                </BarChart>
              </ResponsiveContainer>
            </Panel>

            <Panel className="col-span-12 md:col-span-6 xl:col-span-4" delay={0.25}>
              <PanelHead eyebrow="+15 min error by who is in control · °C" title="Depends on the controller" />
              <div className="space-y-3">
                {Object.entries<any>(ev.generalisation_days).map(([policy, v]) => {
                  const last = horizons[horizons.length - 1], m = v[last].model.mae, p = v[last].persistence.mae, top = 3.5;
                  return (
                    <div key={policy}>
                      <div className="mb-1 flex justify-between text-[11.5px]"><span className="font-medium text-ink">{policy.replace("_", "-")}{policy === "dqn" && <span className="ml-1.5 text-ink-3">· unseen in training</span>}</span><span className="num text-ink-2">{fmt(m, 2)} <span className="text-ink-3">vs {fmt(p, 2)}</span></span></div>
                      <div className="relative h-2 rounded-full bg-white/[0.06]">
                        <motion.div className="absolute inset-y-0 left-0 rounded-full" style={{ background: PERSIST, opacity: 0.45 }} initial={{ width: 0 }} animate={{ width: `${(p / top) * 100}%` }} transition={{ duration: 0.8, ease: EASE }} />
                        <motion.div className="absolute inset-y-0 left-0 rounded-full" style={{ background: MODEL }} initial={{ width: 0 }} animate={{ width: `${(m / top) * 100}%` }} transition={{ duration: 0.8, ease: EASE, delay: 0.1 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3"><Legend items={[{ label: "LSTM", color: MODEL }, { label: "No change", color: PERSIST }]} /></div>
            </Panel>

            <Panel className="col-span-12 md:col-span-6 xl:col-span-3" delay={0.3}>
              <PanelHead eyebrow="Loss per epoch" title="Training" right={<Legend items={[{ label: "train", color: MODEL }, { label: "validation", color: TREND }]} />} />
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={curve} margin={{ top: 14, right: 8, bottom: 0, left: -18 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="epoch" type="number" domain={[1, "dataMax"]} {...AXIS} tickCount={6} />
                  <YAxis {...AXIS} domain={["auto", "auto"]} tickCount={5} />
                  <ReferenceLine x={ev.training.best_epoch} stroke="#eef2f7" strokeOpacity={0.5} label={{ value: "kept", position: "top", fill: "#a3aebb", fontSize: 10 }} />
                  <Tooltip content={<ChartTip digits={3} labelFormat={(e: number) => `epoch ${e}`} />} />
                  <Line dataKey="train" name="train" stroke={MODEL} strokeWidth={2} dot={false} animationDuration={1000} />
                  <Line dataKey="val" name="validation" stroke={TREND} strokeWidth={2} dot={false} animationDuration={1000} />
                </LineChart>
              </ResponsiveContainer>
              <p className="mt-2 text-[11.5px] leading-relaxed text-ink-3">Validation loss bottoms out early, then rises as the model overfits. The best epoch's weights are the ones kept.</p>
            </Panel>
          </div>
        </>
      )}
    </>
  );
});

export function Forecast() {
  const sim = useSim((s) => s.sim);
  if (!sim) return <Loading rows={6} />;
  return (
    <div>
      <PageHeader eyebrow="LSTM temperature forecaster" title="Predictive cooling"
        lead="A recurrent network reads the last 10 minutes of each rack and forecasts its temperature 5, 10 and 15 minutes ahead. The cooling plant needs 3 minutes to respond, so an early warning is only useful if it is right." />
      <div className="mb-4"><Transport /></div>

      {sim.racks[0].forecast ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">{sim.racks.map((r) => <LiveRack key={r.id} index={r.id} />)}</div>
      ) : <Empty>No trained forecaster found. Train it with <code className="num">python -m backend.prediction.model</code>.</Empty>}

      <div className="mt-4"><Hindsight /></div>
      <Evaluation />
    </div>
  );
}
