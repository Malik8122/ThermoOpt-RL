import { AnimatePresence, motion } from "motion/react";
import { Activity, Boxes, BrainCircuit, Gauge, LayoutDashboard, Radar, Trophy, WifiOff } from "lucide-react";
import { memo, useEffect } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { tempColor } from "../lib/colors";
import { useSim } from "../lib/store";
import { AnimatedNumber, EASE } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/twin", label: "Digital Twin", icon: Boxes },
  { to: "/workloads", label: "Workloads", icon: Activity },
  { to: "/training", label: "RL Training", icon: BrainCircuit },
  { to: "/compare", label: "Algorithms", icon: Trophy },
  { to: "/forecast", label: "Predictive Cooling", icon: Radar },
];

const Backdrop = memo(function Backdrop() {
  // Two soft thermal glows + a faint grid. Plain gradients: painted once, never re-blurred or animated.
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-base">
      <div className="absolute inset-0" style={{ background: "radial-gradient(60vh 60vh at 8% 0%, rgba(56,189,248,0.13), transparent 70%), radial-gradient(70vh 70vh at 96% 100%, rgba(45,212,191,0.10), transparent 70%)" }} />
      <div className="absolute inset-0 opacity-[0.5]" style={{ backgroundImage: "linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px)", backgroundSize: "56px 56px", maskImage: "radial-gradient(ellipse at 50% 30%, #000 20%, transparent 75%)" }} />
    </div>
  );
});

function Sidebar() {
  const sim = useSim((s) => s.sim);
  return (
    <aside className="flex w-[232px] shrink-0 flex-col border-r border-line bg-[#06090e]/90 px-4 py-5">
      <div className="mb-8 flex items-center gap-3 px-2">
        <div className="relative grid h-10 w-10 place-items-center rounded-xl border border-line-strong bg-white/[0.05]">
          <Gauge size={19} className="text-ink" />
          <span className="thermal-bar absolute -bottom-px left-2 right-2 h-[2px] rounded-full" />
        </div>
        <div>
          <div className="font-display text-[15px] font-semibold tracking-tight">ThermoOpt</div>
          <div className="eyebrow !text-[9px]">RL cooling lab</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === "/"} className="group relative">
            {({ isActive }) => (
              <div className={`relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13px] font-medium transition-colors ${isActive ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                {isActive && (
                  <motion.div layoutId="nav-active" className="absolute inset-0 rounded-xl border border-line-strong bg-white/[0.07]" transition={{ type: "spring", stiffness: 380, damping: 32 }}>
                    <span className="thermal-bar absolute -left-px top-2.5 bottom-2.5 w-[3px] rounded-full" />
                  </motion.div>
                )}
                <Icon size={17} className="relative" strokeWidth={isActive ? 2.2 : 1.8} />
                <span className="relative">{label}</span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-3">
        {sim && (
          <div className="rounded-2xl border border-line bg-white/[0.03] p-3.5">
            <div className="eyebrow mb-2">Rack temperatures</div>
            <div className="flex items-end gap-1.5">
              {sim.racks.map((r) => (
                <div key={r.id} className="flex flex-1 flex-col items-center gap-1.5">
                  <div className="relative h-14 w-full overflow-hidden rounded-md bg-white/[0.05]">
                    <motion.div className="absolute inset-x-0 bottom-0 rounded-md" animate={{ height: `${Math.min(100, Math.max(6, ((r.temperature - 18) / 16) * 100))}%`, background: tempColor(r.temperature) }}
                      transition={{ type: "spring", stiffness: 90, damping: 18 }} style={{ boxShadow: `0 0 14px ${tempColor(r.temperature)}` }} />
                  </div>
                  <span className="num text-[10px] text-ink-2">{r.temperature.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="px-2 text-[10.5px] leading-relaxed text-ink-3">Simulated digital twin. Grid prices and carbon are synthetic.</div>
      </div>
    </aside>
  );
}

function TopBar() {
  const sim = useSim((s) => s.sim);
  const connected = useSim((s) => s.connected);
  return (
    <div className="flex h-14 shrink-0 items-center justify-between border-b border-line bg-[#06090e]/90 px-7">
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-2.5 w-2.5">
            {sim?.running && <span className="absolute inline-flex h-full w-full animate-pulse-ring rounded-full bg-safe" />}
            <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${!connected ? "bg-crit" : sim?.running ? "bg-safe" : "bg-ink-3"}`} />
          </span>
          <span className="text-xs font-medium text-ink-2">{!connected ? "Backend offline" : sim?.running ? "Live" : "Paused"}</span>
        </div>
        {sim && (
          <>
            <span className="h-4 w-px bg-line" />
            <span className="num text-sm text-ink">{sim.clock}</span>
            <span className="text-xs text-ink-3">day {sim.episode + 1} · seed {sim.seed} · minute {sim.minute}/{sim.episode_minutes}</span>
          </>
        )}
      </div>
      {sim ? (
        <div className="flex items-center gap-6 text-xs text-ink-3">
          <span>Controller <span className="ml-1.5 font-medium text-ink">{sim.controller_label}</span></span>
          <span>Power <AnimatedNumber value={sim.power.total_kw} digits={1} className="ml-1.5 text-ink" /> kW</span>
          <span>Carbon <AnimatedNumber value={sim.grid.carbon} digits={0} className="ml-1.5 text-ink" /> g/kWh</span>
          <span>Queue <AnimatedNumber value={sim.queue.length} digits={0} className="ml-1.5 text-ink" /></span>
        </div>
      ) : (
        !connected && <span className="flex items-center gap-2 text-xs text-crit"><WifiOff size={14} /> start it with <code className="num rounded bg-white/5 px-1.5 py-0.5 text-ink-2">python -m backend.api.main</code></span>
      )}
    </div>
  );
}

export function Shell() {
  const connect = useSim((s) => s.connect);
  const location = useLocation();
  useEffect(() => { connect(); }, [connect]);
  return (
    <div className="flex h-full">
      <Backdrop />
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
          <AnimatePresence mode="wait">
            <motion.div key={location.pathname} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.28, ease: EASE }} className="mx-auto max-w-[1500px] px-7 py-7">
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
