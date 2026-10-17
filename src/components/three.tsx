/** 3D views (React Three Fiber). Loaded lazily; each one is fed by real model tensors. */
import { Canvas } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { Component, useMemo, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import { useColors } from '../sim/appState'
import { pca3, norm, row, type Mat } from '../engine/tensor'
import { showToken, type ForwardTrace } from '../engine/model'

class GLBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed
      ? <div className="grid h-full place-items-center p-6 text-center text-muted">3D view needs WebGL, which this browser has disabled. The 2D panels show the same data.</div>
      : this.props.children
  }
}

const reduced = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

function Scene({ children, camera = [6, 4, 7] as [number, number, number], autoRotate = true }: { children: ReactNode; camera?: [number, number, number]; autoRotate?: boolean }) {
  return (
    <GLBoundary>
      <Canvas camera={{ position: camera, fov: 45 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true, antialias: true }}>
        <ambientLight intensity={0.7} />
        <pointLight position={[10, 10, 10]} intensity={60} />
        {children}
        <OrbitControls makeDefault enableDamping autoRotate={autoRotate && !reduced()} autoRotateSpeed={0.5} />
      </Canvas>
    </GLBoundary>
  )
}

/** All vocabulary embeddings projected to 3 principal components. */
export function EmbeddingCloud({ table, vocab, highlight, selected }: { table: Mat; vocab: string[]; highlight: number[]; selected?: number }) {
  const c = useColors()
  const { coords, explained } = useMemo(() => pca3(table), [table])
  const scale = useMemo(() => {
    let ss = 0
    for (const v of coords) ss += v * v
    return 1.3 / Math.sqrt(ss / coords.length) // ~1.3 units per standard deviation
  }, [coords])
  const [hover, setHover] = useState<number | null>(null)
  const hl = new Set(highlight)
  const kind = (t: string) => (t === '<bos>' ? c.rose : t.length === 1 ? c.faint : t.startsWith('▁') ? c.cyan : c.violet)
  return (
    <div className="relative h-full min-h-[300px]">
      <Scene>
        <gridHelper args={[8, 16, c.lineStrong, c.line]} position={[0, -3.2, 0]} />
        {vocab.map((t, i) => {
          const p: [number, number, number] = [coords[i * 3] * scale, coords[i * 3 + 1] * scale, coords[i * 3 + 2] * scale]
          const on = hl.has(i)
          return (
            <group key={i} position={p}>
              <mesh onPointerOver={(e) => { e.stopPropagation(); setHover(i) }} onPointerOut={() => setHover(null)}>
                <sphereGeometry args={[on ? 0.13 : 0.06, 16, 16]} />
                <meshStandardMaterial color={on ? c.amber : kind(t)} emissive={on ? c.amber : kind(t)} emissiveIntensity={on ? 0.8 : 0.35} />
              </mesh>
              {(on || hover === i) && (
                <Html center distanceFactor={9} style={{ pointerEvents: 'none' }}>
                  <div className={`whitespace-pre rounded px-1.5 py-0.5 font-mono text-[11px] ${i === selected ? 'bg-amber text-bg' : 'bg-panel/90 text-fg border border-line'}`}
                    style={{ transform: 'translateY(-16px)' }}>{showToken(t)}</div>
                </Html>
              )}
            </group>
          )
        })}
      </Scene>
      <div className="pointer-events-none absolute bottom-2 left-2 flex flex-wrap gap-3 font-mono text-[10.5px] text-muted">
        <span><i className="inline-block h-2 w-2 rounded-full" style={{ background: c.cyan }} /> word-start</span>
        <span><i className="inline-block h-2 w-2 rounded-full" style={{ background: c.violet }} /> sub-word</span>
        <span><i className="inline-block h-2 w-2 rounded-full" style={{ background: c.faint }} /> character</span>
        <span><i className="inline-block h-2 w-2 rounded-full" style={{ background: c.amber }} /> your prompt</span>
        <span>PCA explains {(explained.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of variance</span>
      </div>
    </div>
  )
}

/** The residual stream as a 3D lattice: tokens × depth, box height = vector norm. */
export function ResidualTower({ trace, depthIndex, layer, head, selected }: {
  trace: ForwardTrace; depthIndex: number; layer: number; head: number; selected: number
}) {
  const c = useColors()
  const stages = useMemo(() => {
    const s: { name: string; m: Mat }[] = [{ name: 'x⁽⁰⁾', m: trace.x0 }]
    trace.layers.forEach((L, l) => { s.push({ name: `+attn ${l + 1}`, m: L.xMid }, { name: `+mlp ${l + 1}`, m: L.xOut }) })
    return s
  }, [trace])
  const n = trace.ids.length
  const maxN = useMemo(() => Math.max(...stages.flatMap((s) => Array.from({ length: n }, (_, i) => norm(row(s.m, i))))), [stages, n])
  const dx = 1.1, dz = 1.4
  const x0 = (-(n - 1) * dx) / 2, zF = ((stages.length - 1) * dz) / 2 // x⁽⁰⁾ nearest the viewer, deeper layers recede
  const probs = trace.layers[layer].heads[head].probs
  const zAttn = zF - (layer * 2 + 0.5) * dz
  return (
    <div className="relative h-full min-h-[300px]">
      <Scene camera={[-n * 0.35 - 2, 5.5, zF + n * 0.55 + 5]} autoRotate={false}>
        {stages.map((s, si) => (
          <group key={si} position={[0, 0, zF - si * dz]}>
            {Array.from({ length: n }, (_, i) => {
              const h = (norm(row(s.m, i)) / maxN) * 2.6 + 0.05
              const active = si === depthIndex
              const col = new THREE.Color(si === 0 ? c.teal : si % 2 === 1 ? c.cyan : c.violet)
              return (
                <mesh key={i} position={[x0 + i * dx, h / 2, 0]}>
                  <boxGeometry args={[0.7, h, 0.7]} />
                  <meshStandardMaterial color={col} transparent opacity={active ? 0.95 : 0.28} emissive={i === selected && active ? c.amber : col}
                    emissiveIntensity={active ? 0.5 : 0.1} />
                </mesh>
              )
            })}
            <Html position={[x0 - 1.1, 0.2, 0]} center style={{ pointerEvents: 'none' }}>
              <div className={`whitespace-nowrap font-mono text-[10.5px] ${si === depthIndex ? 'text-amber' : 'text-faint'}`}>{s.name}</div>
            </Html>
          </group>
        ))}
        {/* attention flows for the chosen layer/head: from key j (earlier) to query i */}
        {Array.from({ length: n }, (_, i) => Array.from({ length: i + 1 }, (_, j) => {
          const w = probs.data[i * n + j]
          if (w < 0.08) return null
          const a = new THREE.Vector3(x0 + j * dx, 0.05, zAttn + 0.5 * dz)
          const b = new THREE.Vector3(x0 + i * dx, 0.05, zAttn - 0.5 * dz)
          const mid = a.clone().add(b).multiplyScalar(0.5).add(new THREE.Vector3(0, 0.4 + Math.abs(i - j) * 0.25, 0))
          const pts = new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(24)
          const geo = new THREE.BufferGeometry().setFromPoints(pts)
          return (
            <primitive key={`${i}-${j}`} object={new THREE.Line(geo, new THREE.LineBasicMaterial({ color: i === selected ? c.amber : c.cyan, transparent: true, opacity: Math.min(1, w * 1.4) }))} />
          )
        }))}
        {trace.tokens.map((t, i) => (
          <Html key={i} position={[x0 + i * dx, -0.25, zF + 0.9]} center style={{ pointerEvents: 'none' }}>
            <div className={`whitespace-pre font-mono text-[11px] ${i === selected ? 'text-amber' : 'text-fg'}`}>{showToken(t)}</div>
          </Html>
        ))}
        <gridHelper args={[Math.max(n * dx + 3, 8), 16, c.lineStrong, c.line]} />
      </Scene>
      <div className="pointer-events-none absolute bottom-2 left-2 font-mono text-[10.5px] text-muted">
        Box height = ‖x‖ of each token at each depth · curves = layer {layer + 1}, head {head + 1} attention (weights ≥ 0.08) · drag to orbit
      </div>
    </div>
  )
}
