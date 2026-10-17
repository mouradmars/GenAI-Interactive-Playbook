/** 3D view of a 3-component projection (React Three Fiber). Loaded lazily; falls back when WebGL is unavailable. */
import { Canvas } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { Component, useMemo, useState, type ReactNode } from 'react'
import { useColors } from '../../sim/appState'
import type { Vec } from '../../engine/linalg'
import type { Space } from '../../engine/embeddings'
import { groupColor } from './Scatter'

class GLBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <div className="grid h-full place-items-center p-6 text-center text-muted">3D needs WebGL, which is disabled here. Switch the projection to 2D.</div> : this.props.children }
}

export default function Scatter3D({ space, coords, focus, selected, onFocus }: { space: Space; coords: Vec[]; focus: number; selected: number[]; onFocus: (i: number) => void }) {
  const c = useColors()
  const [hover, setHover] = useState<number | null>(null)
  const { color, legend } = groupColor(c, space.groups)
  const s = useMemo(() => {
    let ss = 0
    for (const p of coords) ss += p[0] ** 2 + p[1] ** 2 + (p[2] ?? 0) ** 2
    return 2.2 / Math.sqrt(ss / coords.length || 1)
  }, [coords])
  const sel = new Set(selected)
  const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  return (
    <div className="relative h-[460px]">
      <GLBoundary>
        <Canvas camera={{ position: [5, 4, 6], fov: 45 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }}>
          <ambientLight intensity={0.8} />
          <pointLight position={[8, 8, 8]} intensity={50} />
          <axesHelper args={[2.5]} />
          <gridHelper args={[8, 16, c.lineStrong, c.line]} position={[0, -2.6, 0]} />
          {coords.map((p, i) => (
            <group key={i} position={[p[0] * s, p[1] * s, (p[2] ?? 0) * s]}>
              <mesh onClick={(e) => { e.stopPropagation(); onFocus(i) }} onPointerOver={(e) => { e.stopPropagation(); setHover(i) }} onPointerOut={() => setHover(null)}>
                <sphereGeometry args={[i === focus ? 0.13 : sel.has(i) ? 0.1 : 0.07, 16, 16]} />
                <meshStandardMaterial color={color(space.groups[i])} emissive={color(space.groups[i])} emissiveIntensity={i === focus ? 0.9 : 0.35} />
              </mesh>
              {(i === focus || sel.has(i) || hover === i || space.words.length <= 30) && (
                <Html center distanceFactor={9} style={{ pointerEvents: 'none' }}>
                  <div className={`whitespace-nowrap rounded px-1 font-mono text-[11px] ${i === focus ? 'bg-amber text-bg' : 'text-fg'}`} style={{ transform: 'translateY(-14px)' }}>{space.words[i]}</div>
                </Html>
              )}
            </group>
          ))}
          <OrbitControls makeDefault enableDamping autoRotate={!reduce} autoRotateSpeed={0.4} />
        </Canvas>
      </GLBoundary>
      <div className="pointer-events-none absolute left-2 top-2 flex max-w-[70%] flex-wrap gap-x-3 gap-y-1 font-mono text-[10.5px] text-muted">
        {legend.map(({ g, col }) => <span key={g} className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-full" style={{ background: col }} />{g}</span>)}
      </div>
      <div className="pointer-events-none absolute bottom-2 left-2 font-mono text-[10.5px] text-muted">red / green / blue axes = components 1 / 2 / 3 · drag to orbit, scroll to zoom</div>
    </div>
  )
}
