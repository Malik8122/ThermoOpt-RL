import { animate, motion } from "motion/react";
import { clsx } from "clsx";
import { type ReactNode, useEffect, useRef } from "react";
import { tempColor } from "../lib/colors";
import { clamp } from "../lib/format";

export const EASE = [0.22, 1, 0.36, 1] as const;

/** Glass panel that rises into place. `delay` staggers a grid of panels. */
export function Panel({ children, className, delay = 0, pad = true }: { children: ReactNode; className?: string; delay?: number; pad?: boolean }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: EASE, delay }}
      className={clsx("panel", pad && "p-5", className)}
    >
      {children}
    </motion.section>
  );
}

export function PanelHead({ eyebrow, title, right, icon }: { eyebrow?: string; title: string; right?: ReactNode; icon?: ReactNode }) {
  return (
    <header className="mb-4 flex items-start justify-between gap-3">
      <div className="flex items-center gap-3">
        {icon && <div className="grid h-9 w-9 place-items-center rounded-xl border border-line bg-white/[0.04] text-ink-2">{icon}</div>}
        <div>
          {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
          <h2 className="font-display text-[15px] font-semibold tracking-tight text-ink">{title}</h2>
        </div>
      </div>
      {right}
    </header>
  );
}

// One Intl.NumberFormat per precision, created once. (toLocaleString builds a new formatter on every call.)
const FORMATTERS = new Map<number, Intl.NumberFormat>();
const formatter = (digits: number) => {
  let f = FORMATTERS.get(digits);
  if (!f) { f = new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }); FORMATTERS.set(digits, f); }
  return f;
};

/** A number that tweens to its new value without re-rendering React. Tiny changes are written directly. */
export function AnimatedNumber({ value, digits = 1, className, duration = 0.4 }: { value: number | null | undefined; digits?: number; className?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const prev = useRef<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (value === null || value === undefined || Number.isNaN(value)) { el.textContent = "—"; prev.current = null; return; }
    const f = formatter(digits);
    const from = prev.current;
    prev.current = value;
    const step = 10 ** -digits;
    if (from === null || Math.abs(value - from) < 3 * step) { el.textContent = f.format(value); return; }   // nothing worth animating
    let last = "";
    const controls = animate(from, value, { duration, ease: "easeOut", onUpdate: (v) => { const text = f.format(v); if (text !== last) { el.textContent = text; last = text; } } });
    return () => controls.stop();
  }, [value, digits, duration]);
  return <span ref={ref} className={clsx("num", className)} />;
}

export function Stat({ label, value, unit, digits = 1, accent, sub, icon, delay = 0 }: {
  label: string; value: number | null | undefined; unit?: string; digits?: number; accent?: string; sub?: ReactNode; icon?: ReactNode; delay?: number;
}) {
  return (
    <Panel delay={delay} className="group overflow-hidden">
      <div className="pointer-events-none absolute -right-10 -top-10 h-28 w-28 rounded-full opacity-25 blur-2xl transition-opacity duration-500 group-hover:opacity-50" style={{ background: accent ?? "#38bdf8" }} />
      <div className="flex items-center justify-between">
        <span className="eyebrow">{label}</span>
        {icon && <span className="text-ink-3" style={{ color: accent }}>{icon}</span>}
      </div>
      <div className="mt-3 flex items-baseline gap-1.5">
        <AnimatedNumber value={value} digits={digits} className="text-[30px] font-medium leading-none text-ink" />
        {unit && <span className="text-xs text-ink-3">{unit}</span>}
      </div>
      {sub && <div className="mt-2 text-xs text-ink-2">{sub}</div>}
    </Panel>
  );
}

export function Pill({ children, color, className, dot = false }: { children: ReactNode; color?: string; className?: string; dot?: boolean }) {
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium leading-none", className)}
      style={{ borderColor: color ? `${color}55` : "rgba(255,255,255,0.1)", background: color ? `${color}1a` : "rgba(255,255,255,0.04)", color: color ?? "#a3aebb" }}>
      {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />}
      {children}
    </span>
  );
}

/** Segmented control with a sliding highlight. */
export function Segmented<T extends string>({ options, value, onChange, id, size = "md" }: {
  options: { value: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; id: string; size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-black/30 p-1">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={clsx("relative rounded-lg font-medium transition-colors", size === "sm" ? "px-2.5 py-1 text-[11px]" : "px-3.5 py-1.5 text-xs",
            value === o.value ? "text-ink" : "text-ink-3 hover:text-ink-2")}>
          {value === o.value && (
            <motion.span layoutId={`seg-${id}`} className="absolute inset-0 rounded-lg border border-line-strong bg-white/[0.08]" transition={{ type: "spring", stiffness: 420, damping: 34 }} />
          )}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

/** Horizontal meter that springs to its value. */
export function Meter({ value, color, height = 6, track = "rgba(255,255,255,0.07)" }: { value: number; color: string; height?: number; track?: string }) {
  return (
    <div className="w-full overflow-hidden rounded-full" style={{ height, background: track }}>
      <motion.div className="h-full rounded-full" style={{ background: color, boxShadow: `0 0 12px ${color}88` }}
        animate={{ width: `${clamp(value, 0, 1) * 100}%` }} transition={{ duration: 0.28, ease: "easeOut" }} />
    </div>
  );
}

/** Semi-circular temperature gauge with warning / critical ticks. */
export function TempGauge({ value, min = 18, max = 36, warning, critical, size = 132 }: { value: number; min?: number; max?: number; warning: number; critical: number; size?: number }) {
  const r = size / 2 - 10;
  const cx = size / 2, cy = size / 2;
  const frac = (v: number) => clamp((v - min) / (max - min), 0, 1);
  const point = (f: number) => { const a = Math.PI * (1 - f); return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; };
  const arc = (f0: number, f1: number) => { const [x0, y0] = point(f0); const [x1, y1] = point(f1); return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`; };
  const length = Math.PI * r;
  const color = tempColor(value);
  const tick = (v: number, c: string) => { const f = frac(v); const a = Math.PI * (1 - f); return <line x1={cx + (r - 9) * Math.cos(a)} y1={cy - (r - 9) * Math.sin(a)} x2={cx + (r + 7) * Math.cos(a)} y2={cy - (r + 7) * Math.sin(a)} stroke={c} strokeWidth={2} strokeLinecap="round" />; };
  return (
    <svg width={size} height={size / 2 + 14} viewBox={`0 0 ${size} ${size / 2 + 14}`}>
      <path d={arc(0, 1)} stroke="rgba(255,255,255,0.08)" strokeWidth={8} fill="none" strokeLinecap="round" />
      <motion.path d={arc(0, 1)} stroke={color} strokeWidth={8} fill="none" strokeLinecap="round" strokeDasharray={length}
        animate={{ strokeDashoffset: length * (1 - frac(value)), stroke: color }} transition={{ type: "spring", stiffness: 90, damping: 20 }}
        style={{ filter: `drop-shadow(0 0 8px ${color})` }} />
      {tick(warning, "#fbbf24")}
      {tick(critical, "#f43f5e")}
    </svg>
  );
}

/** Tiny inline trend line. */
export function Spark({ values, color = "#38bdf8", width = 120, height = 34, fill = true }: { values: number[]; color?: string; width?: number; height?: number; fill?: boolean }) {
  if (values.length < 2) return <div style={{ width, height }} />;
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * width, height - 3 - ((v - lo) / span) * (height - 6)]);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const id = `sp-${color.replace("#", "")}`;
  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs><linearGradient id={id} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={color} stopOpacity={0.35} /><stop offset="1" stopColor={color} stopOpacity={0} /></linearGradient></defs>
      {fill && <path d={`${d} L${width},${height} L0,${height} Z`} fill={`url(#${id})`} />}
      <path d={d} fill="none" stroke={color} strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={2.6} fill={color} />
    </svg>
  );
}

export function PageHeader({ eyebrow, title, lead, right }: { eyebrow: string; title: ReactNode; lead?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE }}>
        <div className="eyebrow mb-2">{eyebrow}</div>
        <h1 className="font-display text-[30px] font-semibold leading-[1.1] tracking-tight text-ink">{title}</h1>
        {lead && <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-2">{lead}</p>}
      </motion.div>
      {right}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="grid place-items-center rounded-xl border border-dashed border-line px-4 py-8 text-center text-xs text-ink-3">{children}</div>;
}

export function Loading({ rows = 3 }: { rows?: number }) {
  return <div className="space-y-3">{Array.from({ length: rows }, (_, i) => <div key={i} className="skeleton h-10 w-full" />)}</div>;
}

// ---- chart helpers (Recharts) ---------------------------------------------------------------------------------
export const AXIS = {
  stroke: "rgba(255,255,255,0.08)", tickLine: false, axisLine: false,
  tick: { fontSize: 10.5, fill: "#66717f", fontFamily: "JetBrains Mono Variable, monospace" },
} as const;
export const GRID = { stroke: "rgba(255,255,255,0.055)", vertical: false } as const;

export function ChartTip({ active, payload, label, unit, labelFormat, digits = 2 }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line-strong bg-[#0b1017] px-3 py-2 shadow-2xl">
      <div className="eyebrow mb-1.5">{labelFormat ? labelFormat(label) : label}</div>
      {payload.filter((p: any) => p.value !== null && p.value !== undefined && !p.hide).map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-5 text-xs">
          <span className="flex items-center gap-2 text-ink-2"><span className="h-2 w-2 rounded-full" style={{ background: p.color ?? p.stroke ?? p.fill }} />{p.name}</span>
          <span className="num text-ink">{Number(p.value).toFixed(digits)}{unit ? ` ${unit}` : ""}</span>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5 text-[11px] text-ink-2">
          <span className="h-[3px] w-4 rounded-full" style={i.dashed ? { background: `repeating-linear-gradient(90deg, ${i.color} 0 4px, transparent 4px 7px)` } : { background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
