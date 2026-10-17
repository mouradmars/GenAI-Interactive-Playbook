/**
 * Computational graph of one chain-rule path from the selected parameter (or neuron) to the loss, with the
 * chain-rule animation and the path explorer. The sum over all paths is checked against backprop and
 * against central finite differences, live.
 */
import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, Pause, Play, RotateCcw, XCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { chainPaths, gradientCheck, gradOf, paramName, type ParamRef } from '../../engine/nn'
import { Tex } from '../../components/viz'
import { neuronName, pathChain, selKey, type ChainNode, type Selection } from './model'
import type { NetView } from './derive'
import { n4, tn } from './narrate'
import { useMlp } from './store'

export const startOf = (s: Selection): ParamRef | { kind: 'a'; layer: number; j: number } => (s.kind === 'neuron' ? { kind: 'a', layer: s.layer, j: s.j } : s)
export const selectionName = (s: Selection, L: number) => (s.kind === 'neuron' ? neuronName(s.layer, s.j, L) : paramName(s))

/** is the forward value / gradient of a chain node known at the current micro-step? */
function knownNode(n: ChainNode, r: NetView['reveal']) {
  const fwd = n.kind === 'w' || n.kind === 'b' ? true : n.kind === 'L' ? r.loss : n.kind === 'z' ? r.zUpTo >= n.layer || n.layer === 0 : r.aUpTo >= n.layer || n.layer === 0
  const bwd = n.kind === 'L' ? r.deltaFrom <= n.layer - 1 : n.kind === 'w' || n.kind === 'b' ? r.gradFrom <= n.layer : n.layer === 0 ? r.gradFrom <= 1 : r.deltaFrom <= n.layer
  return { fwd, bwd }
}

export function GraphView({ net, compact = false }: { net: NetView; compact?: boolean }) {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), selection = useMlp((s) => s.selection), pathIdx = useMlp((s) => s.pathIdx), setPathIdx = useMlp((s) => s.setPathIdx)
  const { trace: t, bwd: g, sample, loss, L, reveal } = net
  const sel: Selection = selection ?? { kind: 'w', layer: 1, j: 0, i: 0 }
  const cp = useMemo(() => chainPaths(cfg, params, t, sample.y, startOf(sel), 600), [cfg, params, t, sample.y, sel])
  const order = useMemo(() => cp.paths.map((p, k) => ({ k, p })).sort((a, b) => Math.abs(b.p.product) - Math.abs(a.p.product)), [cp])
  const pi = Math.min(pathIdx, Math.max(0, cp.paths.length - 1))
  const chosen = cp.paths.length ? (pathIdx === 0 ? order[0].k : pi) : -1
  const chain = useMemo(() => (chosen >= 0 ? pathChain(cfg, params, t, g, loss, sel, cp.paths[chosen]) : null), [cfg, params, t, g, loss, sel, cp, chosen])
  const truth = sel.kind === 'neuron' ? g.dA[sel.layer][sel.j] : gradOf(g, sel)
  const numeric = useMemo(() => (sel.kind === 'neuron' ? null : gradientCheck(cfg, params, [sample], 1e-5, [sel])[0]), [cfg, params, sample, sel])

  // chain-rule animation: k = number of edges revealed from the loss end
  const E = chain?.edges.length ?? 0
  const [k, setK] = useState(E)
  const [playing, setPlaying] = useState(false)
  const [animated, setAnimated] = useState(false)
  useEffect(() => { setK(E); setPlaying(false); setAnimated(false) }, [E, selKey(sel), chosen])
  useEffect(() => {
    if (!playing) return
    if (k >= E) { setPlaying(false); return }
    const id = setTimeout(() => setK((x) => x + 1), 900)
    return () => clearTimeout(id)
  }, [playing, k, E])

  if (!chain) return <div className="text-muted">No path from this element to the loss.</div>
  const { nodes, edges } = chain
  const NW = 100, GAP = compact ? 170 : 205, Hh = 150, W = nodes.length * NW + (nodes.length - 1) * (GAP - NW) + 20
  const x = (i: number) => 10 + i * GAP
  const revealedEdge = (e: number) => e >= E - k
  // without the animation, a backward value appears once the timeline's backward pass has computed it
  const showTail = (i: number) => (animated ? revealedEdge(i) : knownNode(nodes[i], reveal).bwd)
  const showGrad = (i: number) => knownNode(nodes[i], reveal).bwd || (animated && (i >= E - k))

  const nodeColor = (n: ChainNode) => (n.kind === 'w' || n.kind === 'b' ? c.amber : n.kind === 'L' ? c.rose : n.kind === 'z' ? c.violet : c.cyan)
  const productTex = edges.map((e) => `\\underbrace{${tn(e.value)}}_{${e.tex}}`).join('\\cdot ')

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px]">∂L/∂<b className="font-mono text-amber">{selectionName(sel, L)}</b> along path {(order.findIndex((o) => o.k === chosen) + 1)} of {cp.paths.length}{cp.truncated ? '+' : ''}</span>
        <div className="ml-auto flex items-center gap-1">
          <button type="button" className="btn" onClick={() => { setK(0); setAnimated(true); setPlaying(true) }} aria-label="Animate chain rule"><Play size={14} />Animate chain rule</button>
          {playing ? <button type="button" className="btn icon-btn" onClick={() => setPlaying(false)} aria-label="Pause chain animation"><Pause size={14} /></button>
            : k < E && <button type="button" className="btn icon-btn" onClick={() => setPlaying(true)} aria-label="Resume chain animation"><Play size={14} /></button>}
          <button type="button" className="btn icon-btn" onClick={() => { setPlaying(false); setK(E); setAnimated(true) }} aria-label="Show whole chain"><RotateCcw size={14} /></button>
        </div>
      </div>
      <div className="overflow-x-auto scroll-thin rounded-lg border border-line bg-panel-2/40">
        <svg width={W} height={Hh} viewBox={`0 0 ${W} ${Hh}`} role="img" aria-label="Computational graph" style={{ minWidth: W }}>
          <defs>
            <marker id="arF" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c.lineStrong} /></marker>
            <marker id="arB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c.rose} /></marker>
          </defs>
          {edges.map((e, i) => {
            const x1 = x(i) + NW, x2 = x(i + 1), on = showTail(i), cur = animated && revealedEdge(i) && i === E - k
            return (
              <g key={i}>
                <line x1={x1 + 2} y1={52} x2={x2 - 4} y2={52} stroke={c.lineStrong} markerEnd="url(#arF)" />
                <text x={(x1 + x2) / 2} y={40} textAnchor="middle" fontSize={9.5} fill={c.muted} fontFamily="var(--font-mono)">{e.label.replace(/^∂z\S*\/∂(w|b) = /, '∂z/∂$1 = ').replace(' (softmax)', '')}</text>
                <text x={(x1 + x2) / 2} y={70} textAnchor="middle" fontSize={11.5} fill={c.fg} fontFamily="var(--font-mono)">× {n4(e.value)}</text>
                {on && (
                  <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <line x1={x2 - 4} y1={100} x2={x1 + 4} y2={100} stroke={c.rose} strokeWidth={cur ? 2.4 : 1.4} markerEnd="url(#arB)" />
                    <text x={(x1 + x2) / 2} y={118} textAnchor="middle" fontSize={10.5} fill={c.rose} fontFamily="var(--font-mono)">∏ = {n4(e.tail)}</text>
                  </motion.g>
                )}
              </g>
            )
          })}
          {nodes.map((n, i) => {
            const kn = knownNode(n, reveal), col = nodeColor(n)
            return (
              <g key={n.id + i}>
                <rect x={x(i)} y={22} width={NW} height={60} rx={10} fill={c.panel} stroke={col} strokeWidth={1.4} />
                <text x={x(i) + NW / 2} y={42} textAnchor="middle" fontSize={13} fill={col} fontFamily="var(--font-mono)">{n.label}</text>
                <text x={x(i) + NW / 2} y={62} textAnchor="middle" fontSize={11.5} fill={kn.fwd ? c.fg : c.faint} fontFamily="var(--font-mono)">{kn.fwd ? n4(n.value) : '?'}</text>
                <text x={x(i) + NW / 2} y={140} textAnchor="middle" fontSize={10} fill={c.muted} fontFamily="var(--font-mono)">∂L/∂· {showGrad(i) ? n4(n.grad) : '?'}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={`${selKey(sel)}-${chosen}`} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="overflow-x-auto scroll-thin text-[13px]">
          <Tex block tex={`\\frac{\\partial L}{\\partial ${nodes[0].tex}}\\Big|_{\\text{this path}} = ${productTex} = ${n4(cp.paths[chosen].product)}`} />
        </motion.div>
      </AnimatePresence>
      <VerifyStrip total={cp.total} truth={truth} numeric={numeric?.numeric ?? null} paths={cp.paths.length} truncated={cp.truncated} />
      {!compact && cp.paths.length > 1 && (
        <div className="grid gap-1">
          <div className="eyebrow">path explorer · every route from {selectionName(sel, L)} to the loss (largest first)</div>
          <div className="max-h-[220px] overflow-y-auto scroll-thin rounded-lg border border-line">
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-panel text-left text-faint"><tr><th className="px-2 py-1">#</th><th className="px-2 py-1">route</th><th className="px-2 py-1 text-right">product</th><th className="px-2 py-1 text-right">share</th></tr></thead>
              <tbody>
                {order.slice(0, 60).map(({ k: idx, p }, r) => (
                  <tr key={idx} onClick={() => setPathIdx(idx === order[0].k ? 0 : idx)} className={`cursor-pointer border-t border-line ${idx === chosen ? 'bg-raise text-cyan' : 'hover:bg-panel-2'}`}>
                    <td className="px-2 py-1 font-mono text-faint">{r + 1}</td>
                    <td className="px-2 py-1 font-mono">{p.nodes.map((nd) => neuronName(nd.layer, nd.j, L)).join(' → ')} → L</td>
                    <td className="px-2 py-1 text-right font-mono">{n4(p.product)}</td>
                    <td className="px-2 py-1 text-right font-mono text-muted">{cp.total !== 0 ? `${((p.product / cp.total) * 100).toFixed(0)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {order.length > 60 && <div className="text-[11.5px] text-faint">Showing 60 of {order.length} paths; the sum uses all of them.</div>}
        </div>
      )}
    </div>
  )
}

export function VerifyStrip({ total, truth, numeric, paths, truncated }: { total: number; truth: number; numeric: number | null; paths: number; truncated: boolean }) {
  const okPaths = !truncated && Math.abs(total - truth) <= 1e-9 * Math.max(1, Math.abs(truth))
  const okNum = numeric === null ? null : Math.abs(numeric - truth) <= 1e-6 * Math.max(1, Math.abs(truth)) + 1e-9
  const Ok = ({ ok }: { ok: boolean | null }) => (ok === null ? null : ok ? <CheckCircle2 size={14} className="text-teal" /> : <XCircle size={14} className="text-rose" />)
  return (
    <div className="grid grid-cols-1 gap-2 rounded-lg border border-line bg-panel-2/50 p-2 text-[12.5px] sm:grid-cols-3" data-testid="verify-strip">
      <div><div className="eyebrow">sum over {paths}{truncated ? '+' : ''} path{paths === 1 ? '' : 's'}</div><div className="flex items-center gap-1.5 font-mono">{truncated ? 'too many to list' : n4(total)} <Ok ok={truncated ? null : okPaths} /></div></div>
      <div><div className="eyebrow">backpropagation</div><div className="font-mono text-cyan" data-testid="bp-value">{n4(truth)}</div></div>
      <div><div className="eyebrow">finite difference (ε = 10⁻⁵)</div><div className="flex items-center gap-1.5 font-mono">{numeric === null ? <span className="text-faint">for parameters</span> : n4(numeric)} <Ok ok={okNum} /></div></div>
    </div>
  )
}
