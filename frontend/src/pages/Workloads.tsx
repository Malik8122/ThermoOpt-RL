import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, Clock3, Hourglass, Plus, Send, XCircle } from "lucide-react";
import { useState } from "react";
import { Transport } from "../components/Live";
import { EASE, Empty, Loading, Meter, PageHeader, Panel, PanelHead, Pill, Stat } from "../components/ui";
import { JOB_TYPE, RACK_COLORS } from "../lib/colors";
import { clamp } from "../lib/format";
import { api, type Job, useSim } from "../lib/store";

function JobChip({ job, mode }: { job: Job; mode: "queue" | "running" | "done" }) {
  const t = JOB_TYPE[job.type];
  const urgent = mode === "queue" && job.slack <= 5;
  const progress = 1 - job.remaining / job.duration;
  return (
    <motion.div initial={{ opacity: 0, scale: 0.92, y: 6 }} animate={{ opacity: 1, scale: 1, y: 0 }} transition={{ duration: 0.28, ease: EASE }}
      className="rounded-xl border bg-white/[0.03] p-2.5" style={{ borderColor: urgent ? "#f43f5e88" : "rgba(255,255,255,0.075)", boxShadow: urgent ? "0 0 18px rgba(244,63,94,0.25)" : undefined }}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <span className="rounded px-1 py-px text-[9px] font-bold" style={{ background: `${t.color}26`, color: t.color }}>{t.short}</span>
          <span className="num text-[11px] text-ink">{job.id}</span>
        </span>
        <span className="flex items-center gap-1">
          {Array.from({ length: job.priority }, (_, i) => <span key={i} className="h-1 w-1 rounded-full bg-warm" />)}
        </span>
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[10.5px] text-ink-3">
        <span className="whitespace-nowrap"><span className="num text-ink-2">{job.cpu}</span> CPU · <span className="num text-ink-2">{job.gpu}</span> GPU</span>
        {mode === "queue" && <span className="num" style={{ color: urgent ? "#f43f5e" : job.flexible ? "#2dd4bf" : "#a3aebb" }}>{job.slack}′ slack</span>}
        {mode === "running" && <span className="num whitespace-nowrap text-ink-2" title="minutes left">{job.remaining}′</span>}
        {mode === "done" && <span className="num" style={{ color: job.status === "FAILED" ? "#f43f5e" : "#2dd4bf" }}>{job.status === "FAILED" ? "missed deadline" : "completed"}</span>}
      </div>
      {mode === "running" && <div className="mt-2"><Meter value={progress} color={t.color} height={3} /></div>}
      {mode === "queue" && <div className="mt-2"><Meter value={clamp(job.slack / 120, 0, 1)} color={urgent ? "#f43f5e" : job.flexible ? "#2dd4bf" : "#a3aebb"} height={3} /></div>}
    </motion.div>
  );
}

function InjectForm() {
  const [type, setType] = useState("AI_TRAINING");
  const [cpu, setCpu] = useState(30);
  const [gpu, setGpu] = useState(70);
  const [duration, setDuration] = useState(40);
  const [slack, setSlack] = useState(60);
  const [priority, setPriority] = useState(2);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  const submit = async () => {
    try {
      const r = await api.inject({ workload_type: type, cpu, gpu, duration, slack, priority });
      setNote({ ok: r.accepted, text: r.accepted ? `${r.job.id} joined the queue` : "Queue full: job rejected" });
    } catch (e: any) { setNote({ ok: false, text: e.message }); }
    setTimeout(() => setNote(null), 3200);
  };
  const slider = (label: string, value: number, set: (v: number) => void, min: number, max: number, unit: string) => (
    <label className="block">
      <div className="mb-1 flex justify-between text-[11px]"><span className="text-ink-3">{label}</span><span className="num text-ink">{value}{unit}</span></div>
      <input type="range" min={min} max={max} value={value} onChange={(e) => set(Number(e.target.value))} className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#38bdf8]" />
    </label>
  );
  return (
    <div className="space-y-3.5">
      <div className="grid grid-cols-3 gap-1.5">
        {Object.entries(JOB_TYPE).map(([k, t]) => (
          <button key={k} onClick={() => setType(k)} className="rounded-lg border px-2 py-2 text-[11px] font-medium transition"
            style={{ borderColor: type === k ? t.color : "rgba(255,255,255,0.08)", background: type === k ? `${t.color}1f` : "transparent", color: type === k ? t.color : "#a3aebb" }}>{t.label}</button>
        ))}
      </div>
      {slider("CPU units", cpu, setCpu, 0, 100, "")}
      {slider("GPU units", gpu, setGpu, 0, 100, "")}
      {slider("Duration", duration, setDuration, 1, 180, " min")}
      {slider("Deadline slack", slack, setSlack, 0, 240, " min")}
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-ink-3">Priority</span>
        <div className="flex gap-1.5">{[1, 2, 3].map((p) => <button key={p} onClick={() => setPriority(p)} className={`num h-7 w-7 rounded-lg border text-xs transition ${priority === p ? "border-warm bg-warm/15 text-warm" : "border-line text-ink-3"}`}>{p}</button>)}</div>
      </div>
      <motion.button whileTap={{ scale: 0.97 }} onClick={submit} disabled={cpu === 0 && gpu === 0}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-ink py-2.5 text-sm font-semibold text-base transition hover:bg-white disabled:opacity-40"><Send size={15} /> Submit job</motion.button>
      <AnimatePresence>
        {note && <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="text-center text-xs" style={{ color: note.ok ? "#2dd4bf" : "#f43f5e" }}>{note.text}</motion.div>}
      </AnimatePresence>
    </div>
  );
}

export function Workloads() {
  const sim = useSim((s) => s.sim);
  if (!sim) return <Loading rows={6} />;
  const running = sim.racks.reduce((n, r) => n + r.jobs.length, 0);
  return (
    <div>
      <PageHeader eyebrow="Jobs" title="Workloads"
        lead="Jobs arrive at random, wait in the queue, run on a rack and finish, or miss their deadline. The controller may place only one job per minute, so a long queue is costly." />
      <div className="mb-4"><Transport /></div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat delay={0.02} label="Waiting" value={sim.queue.length} digits={0} unit="jobs" accent="#e0a100" icon={<Hourglass size={15} />} />
        <Stat delay={0.06} label="Running" value={running} digits={0} unit="jobs" accent="#4a94ee" icon={<Clock3 size={15} />} />
        <Stat delay={0.1} label="Completed today" value={sim.counters.completed ?? 0} digits={0} accent="#2dd4bf" icon={<CheckCircle2 size={15} />} />
        <Stat delay={0.14} label="Missed deadline" value={sim.counters.failed ?? 0} digits={0} accent="#f43f5e" icon={<XCircle size={15} />} />
      </div>

      <div className="mt-4 grid grid-cols-12 items-start gap-4">
        <Panel className="col-span-12 lg:col-span-2" delay={0.1}>
          <PanelHead eyebrow="Head of queue first" title={`Queue · ${sim.queue.length}`} />
          <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1">
            {sim.queue.map((j) => <JobChip key={j.id} job={j} mode="queue" />)}
            {sim.queue.length === 0 && <Empty>Queue is empty</Empty>}
          </div>
        </Panel>

        <Panel className="col-span-12 lg:col-span-7" delay={0.15}>
          <PanelHead eyebrow="Jobs holding CPU / GPU capacity" title="Running on the racks" right={
            <div className="flex gap-2">{Object.values(JOB_TYPE).map((t) => <Pill key={t.label} color={t.color} dot>{t.label}</Pill>)}</div>} />
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            {sim.racks.map((r) => (
              <div key={r.id} className="rounded-2xl border border-line bg-black/20 p-2.5">
                <div className="mb-2 flex items-center justify-between px-1">
                  <span className="flex items-center gap-1.5 text-xs font-semibold"><span className="h-2 w-2 rounded-full" style={{ background: RACK_COLORS[r.id] }} />{r.name}</span>
                  <span className="num text-[10px] text-ink-3">{r.jobs.length}</span>
                </div>
                <div className="mb-2 space-y-1 px-1">
                  <Meter value={r.cpu} color="#69b7ff" height={3} />
                  <Meter value={r.gpu} color="#34f5c5" height={3} />
                </div>
                <div className="max-h-[400px] space-y-2 overflow-y-auto pr-0.5">
                  {r.jobs.map((j) => <JobChip key={j.id} job={j} mode="running" />)}
                  {r.jobs.length === 0 && <Empty>idle</Empty>}
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <div className="col-span-12 space-y-4 lg:col-span-3">
          <Panel delay={0.2}>
            <PanelHead eyebrow="Try it" title="Inject a job" icon={<Plus size={16} />} />
            <InjectForm />
          </Panel>
          <Panel delay={0.25}>
            <PanelHead eyebrow="Most recent first" title="Finished" />
            <div className="max-h-[230px] space-y-2 overflow-y-auto pr-1">
              {sim.finished.slice(0, 12).map((j) => <JobChip key={j.id} job={j} mode="done" />)}
              {sim.finished.length === 0 && <Empty>Nothing finished yet</Empty>}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
