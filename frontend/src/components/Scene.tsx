// Real-time 3D view of the digital twin. Everything animated here is driven by the backend state:
// rack colour = temperature, lit server slots = CPU / GPU utilisation, fan speed and air particles = delivered
// cooling, rising heat = temperature, floor pulses = the controller's last placement / migration.
import { Grid, OrbitControls, PerformanceMonitor, RoundedBox } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import { type RefObject, memo, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { tempColor, tempRgb } from "../lib/colors";
import { useSim } from "../lib/store";

const RACK_X = [-4.5, -1.5, 1.5, 4.5];
const RACK_H = 3.4;
const SERVERS = 10;
const GPUS = 2;
const sim = () => useSim.getState().sim;
const tmp = new THREE.Color();

function radialTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function Rack({ index, glow }: { index: number; glow: THREE.Texture }) {
  const top = useRef<THREE.MeshStandardMaterial>(null);
  const edges = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const halo = useRef<THREE.MeshBasicMaterial>(null);
  const haloMesh = useRef<THREE.Mesh>(null);
  const light = useRef<THREE.PointLight>(null);
  const cpuLeds = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const gpuLeds = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const color = useRef(new THREE.Color("#2dd4bf"));
  const slotH = 0.2;

  useFrame(({ clock }, dt) => {
    const rack = sim()?.racks[index];
    if (!rack) return;
    const [r, g, b] = tempRgb(rack.temperature);
    tmp.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    color.current.lerp(tmp, 1 - Math.exp(-dt * 3));
    const heat = THREE.MathUtils.clamp((rack.temperature - 20) / 12, 0, 1);
    const throb = rack.status === "CRITICAL" ? 0.55 + 0.45 * Math.sin(clock.elapsedTime * 7) : 1;
    if (top.current) { top.current.emissive.copy(color.current); top.current.emissiveIntensity = (1.4 + heat * 2.6) * throb; }
    edges.current.forEach((m) => { if (m) { m.emissive.copy(color.current); m.emissiveIntensity = 0.5 + heat * 1.6; } });
    if (halo.current) { halo.current.color.copy(color.current); halo.current.opacity = (0.16 + heat * 0.5) * throb; }
    if (haloMesh.current) haloMesh.current.scale.setScalar(3.2 + heat * 2.6);
    if (light.current) { light.current.color.copy(color.current); light.current.intensity = 4 + heat * 22; }
    const litCpu = rack.cpu * SERVERS, litGpu = rack.gpu * GPUS;
    cpuLeds.current.forEach((m, i) => {
      if (!m) return;
      const on = THREE.MathUtils.clamp(litCpu - i, 0, 1);
      const flicker = on > 0 ? 0.82 + 0.18 * Math.sin(clock.elapsedTime * (5 + i * 1.7) + index * 3 + i) : 0;
      m.emissiveIntensity = THREE.MathUtils.lerp(m.emissiveIntensity, 0.06 + on * 2.4 * flicker, 0.2);
    });
    gpuLeds.current.forEach((m, i) => {
      if (!m) return;
      const on = THREE.MathUtils.clamp(litGpu - i, 0, 1);
      m.emissiveIntensity = THREE.MathUtils.lerp(m.emissiveIntensity, 0.06 + on * 3.2 * (0.85 + 0.15 * Math.sin(clock.elapsedTime * 3 + i)), 0.2);
    });
  });

  return (
    <group position={[RACK_X[index], 0, 0]}>
      {/* cabinet */}
      <RoundedBox args={[1.56, RACK_H, 1.56]} radius={0.06} smoothness={3} position={[0, RACK_H / 2, 0]}>
        <meshStandardMaterial color="#0b1118" metalness={0.75} roughness={0.32} />
      </RoundedBox>
      {/* glowing cap + side edges, coloured by temperature */}
      <mesh position={[0, RACK_H + 0.035, 0]}>
        <boxGeometry args={[1.5, 0.05, 1.5]} />
        <meshStandardMaterial ref={top} color="#05070b" emissive="#2dd4bf" emissiveIntensity={1.5} toneMapped={false} />
      </mesh>
      {[-0.79, 0.79].map((x, k) => (
        <mesh key={x} position={[x, RACK_H / 2, 0.79]}>
          <boxGeometry args={[0.025, RACK_H - 0.12, 0.025]} />
          <meshStandardMaterial ref={(m) => { edges.current[k] = m; }} color="#05070b" emissive="#2dd4bf" emissiveIntensity={0.6} toneMapped={false} />
        </mesh>
      ))}
      {/* GPU nodes (bottom, two tall slots) */}
      {Array.from({ length: GPUS }, (_, i) => (
        <group key={`g${i}`} position={[0, 0.24 + i * 0.36, 0.785]}>
          <mesh><boxGeometry args={[1.36, 0.3, 0.02]} /><meshStandardMaterial color="#111a24" metalness={0.6} roughness={0.5} /></mesh>
          <mesh position={[0, 0, 0.012]}>
            <boxGeometry args={[1.18, 0.09, 0.01]} />
            <meshStandardMaterial ref={(m) => { gpuLeds.current[i] = m; }} color="#061512" emissive="#34f5c5" emissiveIntensity={0.06} toneMapped={false} />
          </mesh>
        </group>
      ))}
      {/* servers (ten thin slots above) */}
      {Array.from({ length: SERVERS }, (_, i) => (
        <group key={`s${i}`} position={[0, 0.98 + i * (slotH + 0.035), 0.785]}>
          <mesh><boxGeometry args={[1.36, slotH, 0.02]} /><meshStandardMaterial color="#0f1721" metalness={0.6} roughness={0.5} /></mesh>
          <mesh position={[-0.4, 0, 0.012]}>
            <boxGeometry args={[0.42, 0.045, 0.01]} />
            <meshStandardMaterial ref={(m) => { cpuLeds.current[i] = m; }} color="#06101b" emissive="#69b7ff" emissiveIntensity={0.06} toneMapped={false} />
          </mesh>
        </group>
      ))}
      {/* floor halo + light */}
      <mesh ref={haloMesh} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial ref={halo} map={glow} transparent depthWrite={false} blending={THREE.AdditiveBlending} opacity={0.2} />
      </mesh>
      <pointLight ref={light} position={[0, 1.8, 1.9]} distance={7} decay={2} intensity={6} />
    </group>
  );
}

// ---- labels: plain DOM elements positioned every frame by projecting 3D anchors to the screen ------------------
const ANCHORS: [number, number, number][] = [...RACK_X.map((x) => [x, RACK_H + 0.42, 0] as [number, number, number]), [-7.9, 3.35, 0]];

function LabelProjector({ targets }: { targets: RefObject<(HTMLDivElement | null)[]> }) {
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    ANCHORS.forEach((a, i) => {
      const el = targets.current?.[i];
      if (!el) return;
      v.set(a[0], a[1], a[2]);
      const distance = camera.position.distanceTo(v);
      v.project(camera);
      const visible = v.z < 1;
      const scale = THREE.MathUtils.clamp(15 / distance, 0.62, 1.2);
      el.style.opacity = visible ? "1" : "0";
      el.style.transform = `translate(${(v.x * 0.5 + 0.5) * size.width}px, ${(-v.y * 0.5 + 0.5) * size.height}px) translate(-50%, -100%) scale(${scale})`;
    });
  });
  return null;
}

function RackLabel({ index }: { index: number }) {
  const rack = useSim((s) => s.sim?.racks[index]);
  if (!rack) return null;
  const c = tempColor(rack.temperature);
  return (
    <div className="whitespace-nowrap rounded-lg border border-white/10 bg-[#070a10]/92 px-2.5 py-1 text-center" style={{ boxShadow: `0 0 22px ${c}55` }}>
      <div className="eyebrow !text-[8.5px]">{rack.name}</div>
      <div className="num text-[15px] font-medium leading-tight" style={{ color: c }}>{rack.temperature.toFixed(1)}°</div>
      {rack.cooling_effectiveness < 1 && <div className="text-[8px] leading-tight text-ink-3">weak cooling</div>}
    </div>
  );
}

function CoolingLabel() {
  const delivered = useSim((s) => s.sim?.cooling.delivered ?? 0);
  const target = useSim((s) => s.sim?.cooling.target ?? 0);
  return (
    <div className="whitespace-nowrap rounded-lg border border-white/10 bg-[#070a10]/92 px-2.5 py-1 text-center" style={{ boxShadow: "0 0 22px #38bdf855" }}>
      <div className="eyebrow !text-[8.5px]">Cooling plant</div>
      <div className="num text-[15px] font-medium leading-tight text-ice">{(delivered * 100).toFixed(0)}%
        {Math.abs(target - delivered) > 0.005 && <span className="ml-1 text-[10px] text-ink-3">→ {(target * 100).toFixed(0)}%</span>}
      </div>
    </div>
  );
}

function CoolingUnit() {
  const fan = useRef<THREE.Group>(null);
  const ring = useRef<THREE.MeshStandardMaterial>(null);
  useFrame((_, dt) => {
    const d = sim()?.cooling.delivered ?? 0;
    if (fan.current) fan.current.rotation.x -= dt * (0.6 + d * 16);
    if (ring.current) ring.current.emissiveIntensity = 0.5 + d * 3.2;
  });
  return (
    <group position={[-7.9, 0, 0]}>
      <RoundedBox args={[1.5, 2.9, 2.6]} radius={0.08} smoothness={3} position={[0, 1.45, 0]}>
        <meshStandardMaterial color="#0c131c" metalness={0.7} roughness={0.35} />
      </RoundedBox>
      <group position={[0.77, 1.55, 0]}>
        <mesh rotation={[0, Math.PI / 2, 0]}>
          <torusGeometry args={[0.86, 0.045, 12, 60]} />
          <meshStandardMaterial ref={ring} color="#05070b" emissive="#38bdf8" emissiveIntensity={1.2} toneMapped={false} />
        </mesh>
        <group ref={fan}>
          {[0, 1, 2, 3, 4].map((i) => (
            <mesh key={i} rotation={[(i * Math.PI * 2) / 5, 0, 0]} position={[0.01, 0, 0]}>
              <boxGeometry args={[0.03, 1.5, 0.2]} />
              <meshStandardMaterial color="#1b2a3a" metalness={0.8} roughness={0.3} emissive="#38bdf8" emissiveIntensity={0.12} />
            </mesh>
          ))}
        </group>
      </group>
    </group>
  );
}

/** Cold air streaming from the plant down the front aisle; speed follows delivered cooling. */
function AirFlow({ count = 240 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null);
  const { positions, speeds } = useMemo(() => {
    const positions = new Float32Array(count * 3), speeds = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = -7 + Math.random() * 13.5;
      positions[i * 3 + 1] = 0.15 + Math.random() * 3.3;
      positions[i * 3 + 2] = 0.95 + Math.random() * 1.9;
      speeds[i] = 0.55 + Math.random() * 0.9;
    }
    return { positions, speeds };
  }, [count]);
  useFrame(({ clock }, dt) => {
    const p = ref.current;
    if (!p) return;
    const d = sim()?.cooling.delivered ?? 0;
    const a = p.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < count; i++) {
      let x = a.getX(i) + dt * speeds[i] * (0.25 + d * 4.2);
      if (x > 6.5) x = -7;
      a.setX(i, x);
      a.setY(i, a.getY(i) + Math.sin(clock.elapsedTime * 1.3 + i) * 0.0016);
    }
    a.needsUpdate = true;
    (p.material as THREE.PointsMaterial).opacity = 0.12 + d * 0.62;
  });
  return (
    <points ref={ref}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
      <pointsMaterial color="#7dd3fc" size={0.07} transparent opacity={0.5} depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
    </points>
  );
}

/** Heat rising behind each rack; amount and colour follow the rack temperature. */
function HeatPlumes({ perRack = 55 }: { perRack?: number }) {
  const ref = useRef<THREE.Points>(null);
  const count = perRack * RACK_X.length;
  const { positions, colors, seeds } = useMemo(() => {
    const positions = new Float32Array(count * 3), colors = new Float32Array(count * 3), seeds = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const rack = Math.floor(i / perRack);
      positions[i * 3] = RACK_X[rack] + (Math.random() - 0.5) * 1.3;
      positions[i * 3 + 1] = RACK_H + Math.random() * 3;
      positions[i * 3 + 2] = -0.9 + (Math.random() - 0.5) * 0.9;
      seeds[i] = Math.random();
    }
    return { positions, colors, seeds };
  }, [count, perRack]);
  useFrame((_, dt) => {
    const p = ref.current, s = sim();
    if (!p || !s) return;
    const pos = p.geometry.attributes.position as THREE.BufferAttribute;
    const col = p.geometry.attributes.color as THREE.BufferAttribute;
    for (let i = 0; i < count; i++) {
      const rack = Math.floor(i / perRack);
      const t = s.racks[rack].temperature;
      const heat = THREE.MathUtils.clamp((t - 21) / 10, 0, 1);
      let y = pos.getY(i) + dt * (0.25 + heat * 1.5) * (0.5 + seeds[i]);
      if (y > RACK_H + 3.2) { y = RACK_H; pos.setX(i, RACK_X[rack] + (Math.random() - 0.5) * 1.3); }
      pos.setY(i, y);
      const fade = (1 - (y - RACK_H) / 3.2) * heat * (seeds[i] < heat + 0.15 ? 1 : 0);
      const [r, g, b] = tempRgb(t);
      col.setXYZ(i, (r / 255) * fade, (g / 255) * fade, (b / 255) * fade);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
  });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial vertexColors size={0.11} transparent depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
    </points>
  );
}

/** Expanding floor rings when the controller places (teal), migrates (amber → teal) or is rejected (red). */
function Pulses() {
  const POOL = 8;
  const meshes = useRef<(THREE.Mesh | null)[]>([]);
  const live = useRef<{ x: number; color: string; t: number }[]>([]);
  useEffect(() => useSim.subscribe((state, prev) => {
    if (state.pulse === prev.pulse) return;
    const d = state.decisions[0];
    if (!d) return;
    const push = (rack: number, color: string) => { live.current.push({ x: RACK_X[rack], color, t: 0 }); if (live.current.length > POOL) live.current.shift(); };
    if (!d.valid && d.rack !== null) push(d.rack, "#f43f5e");
    else if (d.kind === "PLACE" && d.rack !== null) push(d.rack, "#5eead4");
    else if (d.kind === "MIGRATE" && d.valid) {
      const m = /Rack (\d).*Rack (\d)/.exec(d.explanation);
      if (m) { push(+m[1] - 1, "#fbbf24"); push(+m[2] - 1, "#5eead4"); }
    }
  }), []);
  useFrame((_, dt) => {
    live.current.forEach((p) => { p.t += dt; });
    live.current = live.current.filter((p) => p.t < 1.4);
    meshes.current.forEach((m, i) => {
      if (!m) return;
      const p = live.current[i];
      m.visible = !!p;
      if (!p) return;
      const k = p.t / 1.4;
      m.position.set(p.x, 0.04, 0.3);
      m.scale.setScalar(1 + k * 2.6);
      const mat = m.material as THREE.MeshBasicMaterial;
      mat.color.set(p.color);
      mat.opacity = (1 - k) * 0.9;
    });
  });
  return (
    <>
      {Array.from({ length: POOL }, (_, i) => (
        <mesh key={i} ref={(m) => { meshes.current[i] = m; }} rotation={[-Math.PI / 2, 0, 0]} visible={false}>
          <ringGeometry args={[0.92, 1.0, 64]} />
          <meshBasicMaterial transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
        </mesh>
      ))}
    </>
  );
}

export const Scene = memo(function Scene({ compact = false }: { compact?: boolean }) {
  const glow = useMemo(radialTexture, []);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  // Quality steps down automatically if the frame rate drops: first a lower pixel ratio, then no bloom.
  const [dpr, setDpr] = useState(Math.min(window.devicePixelRatio, 1.25));
  const [bloom, setBloom] = useState(true);
  return (
    <div className="relative h-full w-full">
      <Canvas dpr={dpr} camera={{ position: compact ? [3.2, 4.6, 15.2] : [4.6, 5.4, 14.6], fov: compact ? 36 : 40 }} gl={{ antialias: false, powerPreference: "high-performance", stencil: false }}>
        <PerformanceMonitor bounds={() => [32, 62]} flipflops={2} onDecline={() => (dpr > 0.8 ? setDpr(0.75) : setBloom(false))} onFallback={() => { setDpr(0.75); setBloom(false); }} />
        <color attach="background" args={["#05070b"]} />
        <fog attach="fog" args={["#05070b", 17, 44]} />
        <ambientLight intensity={0.5} />
        <hemisphereLight args={["#9cc7ff", "#06080c", 0.55]} />
        <directionalLight position={[6, 11, 9]} intensity={1.25} color="#dbeafe" />
        <Grid position={[0, 0, 0]} infiniteGrid cellSize={1} sectionSize={4} cellThickness={0.6} sectionThickness={1} cellColor="#152130" sectionColor="#22405c" fadeDistance={36} fadeStrength={1.6} />
        {RACK_X.map((_, i) => <Rack key={i} index={i} glow={glow} />)}
        <CoolingUnit />
        <AirFlow />
        <HeatPlumes />
        <Pulses />
        <LabelProjector targets={labels} />
        <OrbitControls target={[-1.7, 1.55, 0]} enablePan={false} enableZoom={!compact} minDistance={9} maxDistance={30} minPolarAngle={0.5} maxPolarAngle={1.46}
          autoRotate autoRotateSpeed={compact ? 0.5 : 0.32} enableDamping dampingFactor={0.06} />
        {bloom && (
          <EffectComposer multisampling={0} resolutionScale={0.5}>
            <Bloom intensity={1.15} luminanceThreshold={0.22} luminanceSmoothing={0.35} mipmapBlur radius={0.7} levels={5} />
            <Vignette eskil={false} offset={0.22} darkness={0.82} />
          </EffectComposer>
        )}
      </Canvas>
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {ANCHORS.map((_, i) => (
          <div key={i} ref={(el) => { labels.current[i] = el; }} className="absolute left-0 top-0 origin-bottom opacity-0 transition-opacity duration-300" style={{ willChange: "transform" }}>
            {i < RACK_X.length ? <RackLabel index={i} /> : <CoolingLabel />}
          </div>
        ))}
      </div>
    </div>
  );
});
