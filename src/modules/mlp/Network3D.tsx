/** Optional 3-D view of the same network (React Three Fiber). Same data as the 2-D view; click a neuron to select it. */
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Component, useMemo, type ReactNode } from 'react'
import * as THREE from 'three'
import { useColors } from '../../sim/appState'
import { selKey } from './model'
import type { NetView } from './derive'
import { useMlp } from './store'

class GLBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <div className="grid h-full place-items-center p-6 text-center text-muted">3D needs WebGL, which is unavailable here. The 2D view shows the same data.</div> : this.props.children }
}

export default function Network3D({ net }: { net: NetView }) {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), selection = useMlp((s) => s.selection), select = useMlp((s) => s.select)
  const { trace: t, reveal, bwd: g } = net
  const L = cfg.sizes.length - 1
  const pos = useMemo(() => cfg.sizes.map((n, l) => {
    const cols = n > 8 ? Math.ceil(Math.sqrt(n)) : 1, rows = Math.ceil(n / cols), gap = n > 20 ? 0.42 : 0.8
    return Array.from({ length: n }, (_, k) => new THREE.Vector3((l - L / 2) * 2.6, ((rows - 1) / 2 - Math.floor(k / cols)) * gap, ((k % cols) - (cols - 1) / 2) * gap))
  }), [cfg.sizes, L])
  const edges = useMemo(() => {
    const maxW = Math.max(1e-9, ...params.W.flat(2).map(Math.abs))
    const pts: number[] = [], cols: number[] = []
    const ca = new THREE.Color(c.amber), cv = new THREE.Color(c.violet), cr = new THREE.Color(c.rose), ct = new THREE.Color(c.teal), bg = new THREE.Color(c.bg)
    for (let l = 1; l <= L; l++) {
      const grads = reveal.gradFrom <= l, mg = Math.max(1e-12, ...g.dW[l - 1].flat().map(Math.abs))
      for (let j = 0; j < cfg.sizes[l]; j++) for (let i = 0; i < cfg.sizes[l - 1]; i++) {
        const v = grads ? g.dW[l - 1][j][i] / mg : params.W[l - 1][j][i] / maxW
        const col = (grads ? (v >= 0 ? cr : ct) : v >= 0 ? ca : cv).clone().lerp(bg, 1 - Math.min(1, 0.15 + Math.abs(v)))
        const a = pos[l - 1][i], b = pos[l][j]
        pts.push(a.x, a.y, a.z, b.x, b.y, b.z); cols.push(col.r, col.g, col.b, col.r, col.g, col.b)
      }
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)); geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3))
    return geo
  }, [params, pos, cfg.sizes, L, reveal.gradFrom, g, c])
  const sk = selKey(selection)
  return (
    <div className="h-[380px] w-full overflow-hidden rounded-lg border border-line" data-testid="network-3d">
      <GLBoundary>
        <Canvas camera={{ position: [0, 2.5, 9], fov: 45 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true, antialias: true }}>
          <ambientLight intensity={0.8} /><pointLight position={[8, 8, 8]} intensity={50} />
          <lineSegments geometry={edges}><lineBasicMaterial vertexColors transparent opacity={0.85} /></lineSegments>
          {pos.map((layer, l) => layer.map((p, j) => {
            const known = l === 0 || reveal.aUpTo >= l, v = t.a[l][j], on = sk === `n${l}.${j}`
            const f = l === 0 ? 'input' : cfg.acts[l - 1]
            const tt = f === 'sigmoid' || f === 'softmax' ? Math.max(0, Math.min(1, v)) : Math.min(1, Math.abs(v))
            const col = known ? new THREE.Color(c.panel2).lerp(new THREE.Color(v < 0 && f !== 'sigmoid' && f !== 'softmax' ? c.violet : c.cyan), Math.sqrt(tt)) : new THREE.Color(c.panel2)
            return (
              <mesh key={`${l}.${j}`} position={p} onClick={(e) => { e.stopPropagation(); select(on ? null : { kind: 'neuron', layer: l, j }) }}>
                <sphereGeometry args={[cfg.sizes[l] > 20 ? 0.13 : 0.22, 20, 20]} />
                <meshStandardMaterial color={col} emissive={on ? c.cyan : '#000000'} emissiveIntensity={on ? 0.8 : 0} />
              </mesh>
            )
          }))}
          <OrbitControls makeDefault enableDamping />
        </Canvas>
      </GLBoundary>
      <div className="pointer-events-none -mt-7 px-3 pb-2 text-right font-mono text-[10.5px] text-faint">input → {cfg.sizes.slice(1, -1).map((_, l) => `hidden ${l + 1} → `).join('')}output · drag to orbit · click a neuron</div>
    </div>
  )
}
