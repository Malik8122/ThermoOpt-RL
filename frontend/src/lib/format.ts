import { useEffect, useState } from "react";
import { cached } from "./store";

export const fmt = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const pct = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? "—" : `${fmt(v * 100, digits)}%`);
export const clockOf = (minute: number, startHour = 8) => {
  const total = Math.round(startHour * 60 + minute) % 1440;
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
};
/** Compact axis tick: 1200 -> 1.2k, 85 -> 85, 2.5 -> 2.5, 0.0105 -> 0.011. */
export const tick = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  if (a >= 10 || a === 0) return v.toFixed(0);
  if (a >= 1) return String(Number(v.toFixed(1)));
  return String(Number(v.toPrecision(2)));
};
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Fetch a saved-results endpoint once. Returns [data, error]. */
export function useResource<T = any>(path: string): [T | null, string | null] {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null); setError(null);
    cached<T>(path).then((d) => live && setData(d)).catch((e) => live && setError(String(e.message ?? e)));
    return () => { live = false; };
  }, [path]);
  return [data, error];
}

/** Centred moving average that ignores nulls (visual aid only; raw data is always drawn too). */
export function movingAverage(values: (number | null)[], window: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0, n = 0;
  const q: number[] = [];
  for (const v of values) {
    if (v !== null && v !== undefined) { q.push(v); sum += v; n++; }
    if (q.length > window) { sum -= q.shift()!; n--; }
    out.push(q.length >= Math.min(window, 3) ? sum / n : null);
  }
  return out;
}
