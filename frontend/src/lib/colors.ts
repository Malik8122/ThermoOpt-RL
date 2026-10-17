// Colour helpers. Temperature uses one continuous thermal ramp everywhere (3D scene, gauges, chips).

export const STATUS = { NORMAL: "#2dd4bf", WARNING: "#fbbf24", CRITICAL: "#f43f5e" } as const;
export const SERIES = ["#4a94ee", "#ee7a45", "#23b88a", "#e0a100", "#e06694", "#9c8ff0"];
export const RACK_COLORS = ["#4a94ee", "#23b88a", "#ee7a45", "#9c8ff0"]; // identity colours for Rack 1-4

export const FAMILY: Record<string, { label: string; color: string }> = {
  baseline: { label: "Hand-written baseline", color: "#8a96a5" },
  planning: { label: "Model-based planning", color: "#9c8ff0" },
  "tabular-rl": { label: "Tabular RL", color: "#23b88a" },
  "deep-rl": { label: "Deep RL", color: "#4a94ee" },
};

const STOPS: [number, [number, number, number]][] = [
  [18, [56, 189, 248]],
  [23, [45, 212, 191]],
  [27, [251, 191, 36]],
  [30, [244, 63, 94]],
  [35, [255, 200, 220]],
];

export function tempRgb(t: number): [number, number, number] {
  if (t <= STOPS[0][0]) return STOPS[0][1];
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0);
      return [0, 1, 2].map((j) => Math.round(c0[j] + (c1[j] - c0[j]) * k)) as [number, number, number];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

export const tempColor = (t: number) => `rgb(${tempRgb(t).join(",")})`;
export const tempGlow = (t: number, a = 0.35) => `rgba(${tempRgb(t).join(",")},${a})`;

export const JOB_TYPE: Record<string, { label: string; color: string; short: string }> = {
  NORMAL_CPU: { label: "CPU job", color: "#4a94ee", short: "CPU" },
  GPU_INFERENCE: { label: "GPU inference", color: "#23b88a", short: "INF" },
  AI_TRAINING: { label: "AI training", color: "#e06694", short: "TRN" },
};
