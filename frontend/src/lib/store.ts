// Live data layer: one WebSocket to the backend twin, plus thin REST helpers.
// The dashboard computes nothing about the simulation itself; it renders what the backend sends.
import { useEffect, useMemo, useState } from "react";
import { create } from "zustand";

export type Job = {
  id: string; type: string; cpu: number; gpu: number; priority: number; flexible: boolean; status: string;
  duration: number; remaining: number; arrival: number; deadline: number; slack: number; rack: number | null;
  delays: number; migrations: number; background: boolean; started_at: number | null; completed_at: number | null;
  failure_reason: string | null;
};
export type Rack = {
  id: number; name: string; temperature: number; status: "NORMAL" | "WARNING" | "CRITICAL"; cpu: number; gpu: number;
  it_kw: number; cooling_received: number; cooling_effectiveness: number; jobs: Job[]; forecast: number[] | null;
};
export type Decision = {
  minute: number; clock: string; action: number; description: string; kind: "PLACE" | "DELAY" | "MIGRATE";
  cooling: "DECREASE" | "MAINTAIN" | "INCREASE"; rack: number | null; valid: boolean; invalid_reason: string | null;
  explanation: string; reward: number; reward_parts: Record<string, number>; controller: string;
};
export type Point = {
  minute: number; clock: string; temps: number[]; cooling_target: number; cooling_delivered: number; it_kw: number;
  cooling_kw: number; carbon: number; price: number; ambient: number; queue: number; reward: number | null;
  forecast: number[][] | null;
};
export type SimState = {
  minute: number; clock: string; hour: number; episode: number; seed: number; episode_minutes: number;
  running: boolean; speed: number; controller: string; controller_label: string; racks: Rack[];
  cooling: { target: number; delivered: number; progress: number; power_kw: number };
  grid: { carbon: number; price: number }; ambient: number;
  power: { it_kw: number; cooling_kw: number; total_kw: number; pue: number | null };
  totals: { energy_kwh: number; it_kwh: number; cooling_kwh: number; carbon_kg: number; cost_inr: number };
  queue: Job[]; finished: Job[];
  counters: Record<string, number | null>;
  zones: { warning: number; critical: number }; horizons: number[]; last_decision: Decision | null;
};
export type ControllerInfo = {
  key: string; label: string; family: string; representation: string; description: string; step: number;
  available: boolean; active: boolean;
};

type Store = {
  connected: boolean;
  sim: SimState | null;
  history: Point[];
  decisions: Decision[];
  controllers: ControllerInfo[];
  pulse: number; // increments on every decision, for one-shot animations
  connect: () => void;
};

const HISTORY = 360;
const DECISIONS = 80;
const FLUSH_MS = 250;
let socket: WebSocket | null = null;
let retry = 0;
let pending: { state: SimState; point: Point; decision: Decision }[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;

export const useSim = create<Store>((set, get) => ({
  connected: false,
  sim: null,
  history: [],
  decisions: [],
  controllers: [],
  pulse: 0,
  connect: () => {
    if (socket && socket.readyState <= 1) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws/simulation`);
    socket.onopen = () => { retry = 0; set({ connected: true }); };
    socket.onclose = () => {
      set({ connected: false });
      retry = Math.min(retry + 1, 6);
      setTimeout(() => get().connect(), 400 * 2 ** retry);
    };
    socket.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "init") {
        pending = [];
        set({ sim: m.state, history: m.history, decisions: m.decisions, controllers: m.controllers });
      } else if (m.type === "tick") {
        pending.push(m);
        if (!flushTimer) {
          flushTimer = setTimeout(() => {
            flushTimer = null;
            const batch = pending;
            pending = [];
            if (!batch.length) return;
            const s = get();
            const history = [...s.history, ...batch.map((b) => b.point)].slice(-HISTORY);
            const decisions = [...batch.map((b) => b.decision).reverse(), ...s.decisions].slice(0, DECISIONS);
            set({ sim: batch[batch.length - 1].state, history, decisions, pulse: s.pulse + batch.length });
          }, FLUSH_MS);
        }
      } else if (m.type === "control") {
        const sim = get().sim;
        if (sim) set({ sim: { ...sim, running: m.running, speed: m.speed } });
      } else if (m.type === "controller") {
        set({ sim: m.state, controllers: m.controllers });
      } else if (m.type === "state") {
        set({ sim: m.state });
      }
    };
  },
}));

/**
 * The live history for charts: refreshed at most once per `intervalMs` and thinned to about `maxPoints`.
 * Charts are the most expensive thing to redraw, and they do not need every tick.
 */
export function useChartHistory(intervalMs = 2000, maxPoints = 120): Point[] {
  const [snapshot, setSnapshot] = useState<Point[]>(() => useSim.getState().history);
  useEffect(() => {
    const read = () => setSnapshot((prev) => { const cur = useSim.getState().history; return prev === cur ? prev : cur; });
    read();
    const id = setInterval(read, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return useMemo(() => {
    const stride = Math.ceil(snapshot.length / maxPoints);
    if (stride <= 1) return snapshot;
    const last = snapshot.length - 1;
    return snapshot.filter((_, i) => (last - i) % stride === 0);      // always keeps the newest point
  }, [snapshot, maxPoints]);
}

async function post(path: string, body?: unknown) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail ?? `${path} failed (${r.status})`);
  return r.json();
}

export const api = {
  start: () => post("/api/simulation/start"),
  pause: () => post("/api/simulation/pause"),
  speed: (speed: number) => post("/api/simulation/speed", { speed }),
  step: (minutes: number) => post("/api/simulation/step", { minutes }),
  reset: (body: { seed?: number; controller?: string; start_hour?: number } = {}) => post("/api/simulation/reset", body),
  controller: (controller: string) => post("/api/simulation/controller", { controller }),
  inject: (job: Record<string, unknown>) => post("/api/workload", job),
  get: async <T = any>(path: string): Promise<T> => {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path} failed (${r.status})`);
    return r.json();
  },
};

// tiny cache for the saved-results endpoints (they do not change while the page is open)
const cache = new Map<string, Promise<any>>();
export function cached<T = any>(path: string): Promise<T> {
  if (!cache.has(path)) cache.set(path, api.get<T>(path).catch((e) => { cache.delete(path); throw e; }));
  return cache.get(path)!;
}
