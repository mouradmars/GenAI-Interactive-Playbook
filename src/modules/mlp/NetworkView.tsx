/**
 * The clickable 2-D network. Neurons, biases and connections are selectable; values appear only once the
 * micro-step that computes them has run. Large layers are drawn compactly (64 inputs as an 8×8 grid).
 */
import { motion, useReducedMotion } from 'framer-motion'
import { memo, useMemo } from 'react'
import { interpolateRgb } from 'd3'
import { useApp, useColors, type Colors } from '../../sim/appState'
import { ACT_LABEL, paramName, type Act } from '../../engine/nn'
import { neuronName, selKey, type Selection } from './model'
import type { NetView } from './derive'
import { useMlp } from './store'

export const viewWidth = (L: number) => Math.min(920, 230 * L + 340)
interface Pos { x: number; y: number; r: number }

export function layoutNet(sizes: number[], H: number): Pos[][] {
  const L = sizes.length - 1, VW = viewWidth(L), x0 = 80, x1 = VW - 150
  return sizes.map((n, l) => {
    const x = L === 0 ? x0 : x0 + ((x1 - x0) * l) / L
    if (n > 20) {
      const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), cell = Math.min(22, (H - 70) / rows)
      return Array.from({ length: n }, (_, k) => ({ x: x + ((k % cols) - (cols - 1) / 2) * cell, y: H / 2 + 8 + (Math.floor(k / cols) - (rows - 1) / 2) * cell, r: cell * 0.42 }))
    }
    const gap = Math.min(84, (H - 70) / n), r = Math.max(5, Math.min(25, gap * 0.36))
    return Array.from({ length: n }, (_, k) => ({ x, y: H / 2 + 8 + (k - (n - 1) / 2) * gap, r }))
  })
}

const actColor = (c: Colors, f: Act | 'input', maxAbs: number) => {
  const pos = interpolateRgb(c.panel2, c.cyan), neg = interpolateRgb(c.panel2, c.violet)
  if (f === 'sigmoid' || f === 'softmax') return (v: number) => pos(Math.sqrt(Math.max(0, Math.min(1, v))))
  return (v: number) => { const t = Math.min(1, Math.abs(v) / (maxAbs || 1)); return v >= 0 ? pos(Math.sqrt(t)) : neg(Math.sqrt(t)) }
}
const f2 = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2)).replace('-', '−')

export const NetworkView = memo(function NetworkView({ net, height = 400 }: { net: NetView; height?: number }) {
  const c = useColors()
  const { depth } = useApp()
  const reduce = useReducedMotion()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), selection = useMlp((s) => s.selection), select = useMlp((s) => s.select)
  const edgeMode = useMlp((s) => s.edgeMode), micro = useMlp((s) => s.micro), compare = useMlp((s) => s.compare)
  const { trace: t, bwd: g, reveal, step, L, sample, loss, update } = net
  const H = Math.max(height, cfg.sizes.some((n) => n > 20) ? 280 : 0), VW = viewWidth(cfg.sizes.length - 1)
  const pos = useMemo(() => layoutNet(cfg.sizes, H), [cfg.sizes, H])
  const nEdges = cfg.sizes.slice(1).reduce((s, n, l) => s + n * cfg.sizes[l], 0)
  const dense = nEdges > 260
  const sk = selKey(selection)
  const isDigits = useMlp((s) => s.dataset === 'digits' || !!s.probe) && cfg.sizes[0] === 64 && cfg.sizes[L] === 10
  const selNeuron = selection?.kind === 'neuron' ? selection : null
  const updates = useMemo(() => {
    const m = new Map<string, number>()
    update.records.forEach((r) => m.set(selKey(r.ref), r.step))
    return m
  }, [update.records])

  // what the edges encode at this moment
  const edgeShow = (l: number): 'weights' | 'grads' | 'updates' =>
    edgeMode === 'weights' ? 'weights' : edgeMode === 'grads' ? 'grads' : reveal.update ? 'updates' : reveal.gradFrom <= l ? 'grads' : 'weights'
  const maxW = Math.max(1e-9, ...params.W.flat(2).map(Math.abs))
  const maxG = (l: number) => Math.max(1e-12, ...g.dW[l - 1].flat().map(Math.abs))
  const maxU = Math.max(1e-12, ...update.records.map((r) => Math.abs(r.step)))

  const activeLayer = step.kind === 'z' || step.kind === 'act' || step.kind === 'grad' || step.kind === 'delta' ? step.layer : step.kind === 'input' ? 0 : step.kind === 'dout' || step.kind === 'loss' ? L : -1
  const lossX = VW - 58, lossY = H / 2 + 8

  const pick = (s: Selection) => select(selKey(s) === sk ? null : s)

  return (
    <svg viewBox={`0 0 ${VW} ${H}`} className="block w-full select-none" role="img" aria-label={`Network ${cfg.sizes.join('-')}`} style={{ maxHeight: H }}>
      <defs>
        <filter id="nnglow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>
      {/* active layer band */}
      {activeLayer >= 0 && <rect x={pos[activeLayer][0].x - 46} y={22} width={92} height={H - 30} rx={14} fill={c.cyan} opacity={0.06} />}
      {/* layer headers */}
      {cfg.sizes.map((n, l) => (
        <text key={l} x={pos[l][0].x} y={14} textAnchor="middle" fontSize={13} fill={l === activeLayer ? c.cyan : c.muted} fontFamily="var(--font-mono)">
          {l === 0 ? `input · ${n}` : `${l === L ? 'output' : `hidden ${l}`} · ${n} · ${ACT_LABEL[cfg.acts[l - 1]]}`}
        </text>
      ))}
      {/* edges */}
      {cfg.sizes.slice(1).map((n, li) => {
        const l = li + 1, mode = edgeShow(l), mg = maxG(l)
        return (
          <g key={`e${l}`}>
            {Array.from({ length: n }, (_, j) => Array.from({ length: cfg.sizes[l - 1] }, (_, i) => {
              const a = pos[l - 1][i], b = pos[l][j]
              const w = params.W[l - 1][j][i], key = `w${l}.${j}.${i}`
              const v = mode === 'weights' ? w / maxW : mode === 'grads' ? g.dW[l - 1][j][i] / mg : (updates.get(key) ?? 0) / maxU
              const col = mode === 'weights' ? (v >= 0 ? c.amber : c.violet) : v >= 0 ? c.rose : c.teal
              const isSel = key === sk
              const touches = selNeuron && ((selNeuron.layer === l && selNeuron.j === j) || (selNeuron.layer === l - 1 && selNeuron.j === i))
              const op = isSel ? 1 : dense ? (touches ? 0.8 : 0.05 + 0.5 * Math.abs(v) ** 1.5) : touches ? 1 : 0.25 + 0.65 * Math.abs(v)
              const width = isSel ? 4 : dense ? 0.6 + 1.2 * Math.abs(v) : 0.8 + 4.2 * Math.abs(v)
              return (
                <g key={key}>
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={isSel ? c.cyan : col} strokeWidth={width} opacity={op} strokeLinecap="round" />
                  {!dense && (
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={10} style={{ cursor: 'pointer' }} onClick={() => pick({ kind: 'w', layer: l, j, i })}
                      data-testid={`edge-${l}-${j}-${i}`}>
                      <title>{`${paramName({ kind: 'w', layer: l, j, i })} = ${w.toFixed(4)}${reveal.gradFrom <= l ? ` · ∂L/∂w = ${g.dW[l - 1][j][i].toFixed(4)}` : ''}`}</title>
                    </line>
                  )}
                </g>
              )
            }))}
          </g>
        )
      })}
      {/* signal pulses for the current micro-step (purely decorative: values come from the engine) */}
      {!reduce && <Pulses key={micro} pos={pos} step={step} c={c} L={L} lossX={lossX} lossY={lossY} sizes={cfg.sizes} />}
      {/* loss box */}
      <g>
        {cfg.sizes[L] <= 20 && pos[L].map((p, k) => <line key={k} x1={p.x + p.r} y1={p.y} x2={lossX - 34} y2={lossY} stroke={c.line} strokeDasharray="3 3" />)}
        <rect x={lossX - 34} y={lossY - 36} width={80} height={72} rx={12} fill={c.panel} stroke={reveal.loss ? c.amber : c.line} strokeWidth={reveal.loss ? 1.6 : 1} />
        <text x={lossX + 6} y={lossY - 16} textAnchor="middle" fontSize={11} fill={c.muted} fontFamily="var(--font-mono)">loss</text>
        <text x={lossX + 6} y={lossY + 6} textAnchor="middle" fontSize={17} fill={reveal.loss ? c.amber : c.faint} fontFamily="var(--font-mono)" data-testid="net-loss">{reveal.loss ? loss.toFixed(4) : '?'}</text>
        <text x={lossX + 6} y={lossY + 25} textAnchor="middle" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">
          {cfg.sizes[L] === 1 ? `y = ${sample.y[0]}` : isDigits ? `digit ${sample.label ?? sample.y.indexOf(1)}` : `class ${sample.y.indexOf(1) + 1}`}
        </text>
      </g>
      {/* neurons */}
      {cfg.sizes.map((n, l) => {
        const f: Act | 'input' = l === 0 ? 'input' : cfg.acts[l - 1]
        const vals = t.a[l], known = l === 0 || reveal.aUpTo >= l, zKnown = l > 0 && reveal.zUpTo >= l, dKnown = l > 0 && reveal.deltaFrom <= l
        const col = actColor(c, f, Math.max(1e-9, ...vals.map(Math.abs)))
        const maxD = l > 0 ? Math.max(1e-12, ...g.delta[l].map(Math.abs)) : 1
        return (
          <g key={`n${l}`}>
            {pos[l].map((p, j) => {
              const key = `n${l}.${j}`, isSel = key === sk, showText = p.r >= 14
              const d = l > 0 ? g.delta[l][j] : 0
              return (
                <g key={key} style={{ cursor: 'pointer' }} onClick={() => pick({ kind: 'neuron', layer: l, j })} data-testid={`neuron-${l}-${j}`}>
                  <title>{`${neuronName(l, j, L)}${known ? ` · a = ${vals[j].toFixed(4)}` : ''}${zKnown ? ` · z = ${t.z[l][j].toFixed(4)}` : ''}${dKnown ? ` · δ = ${d.toFixed(4)}` : ''}`}</title>
                  {dKnown && <circle cx={p.x} cy={p.y} r={p.r + 3 + 4 * Math.abs(d) / maxD} fill="none" stroke={d >= 0 ? c.rose : c.teal} strokeWidth={1.5 + 2.5 * Math.abs(d) / maxD} opacity={0.9} />}
                  {n > 20
                    ? <rect x={p.x - p.r} y={p.y - p.r} width={p.r * 2} height={p.r * 2} rx={2} fill={known ? col(vals[j]) : c.panel2} stroke={isSel ? c.cyan : c.lineStrong} strokeWidth={isSel ? 2.5 : 0.6} />
                    : <circle cx={p.x} cy={p.y} r={p.r} fill={known ? col(vals[j]) : c.panel2} stroke={isSel ? c.cyan : l === activeLayer ? c.cyan : c.lineStrong} strokeWidth={isSel ? 3 : 1.2}
                        strokeDasharray={known ? undefined : '3 3'} filter={isSel ? 'url(#nnglow)' : undefined} />}
                  {showText && (
                    <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={p.r >= 20 ? 12.5 : 11} fontFamily="var(--font-mono)" fill={known && (f === 'sigmoid' || f === 'softmax' ? vals[j] > 0.55 : Math.abs(vals[j]) > 0.6) ? c.bg : c.fg} pointerEvents="none">
                      {known ? f2(vals[j]) : zKnown ? 'f(z)' : '?'}
                    </text>
                  )}
                  {showText && n <= 6 && l > 0 && depth !== 'beginner' && (zKnown || dKnown) && (
                    <text x={p.x} y={p.y + p.r + 13} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill={c.muted} pointerEvents="none">
                      {zKnown ? `z ${f2(t.z[l][j])}` : ''}{dKnown ? ` δ ${f2(d)}` : ''}
                    </text>
                  )}
                  {showText && l === L && compare && net.original && (
                    <text x={p.x} y={p.y - p.r - 6} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill={c.violet} pointerEvents="none">was {f2(net.original.trace.a[l][j])}</text>
                  )}
                  {l === L && isDigits && <text x={p.x + p.r + 6} y={p.y + 4} fontSize={11} fontFamily="var(--font-mono)" fill={j === sample.label ? c.amber : c.muted} pointerEvents="none">{j}</text>}
                  {showText && l === 0 && <text x={p.x - p.r - 6} y={p.y + 4} textAnchor="end" fontSize={10.5} fontFamily="var(--font-mono)" fill={c.muted} pointerEvents="none">{neuronName(0, j, L)}</text>}
                </g>
              )
            })}
            {/* biases: small diamonds above-left of each neuron (only where there is room) */}
            {l > 0 && pos[l][0].r >= 12 && n <= 6 && pos[l].map((p, j) => {
              const key = `b${l}.${j}`, isSel = key === sk, bx = p.x - p.r - 9, by = p.y - p.r - 4
              return (
                <g key={key} style={{ cursor: 'pointer' }} onClick={(e) => { e.stopPropagation(); pick({ kind: 'b', layer: l, j }) }} data-testid={`bias-${l}-${j}`}>
                  <title>{`${paramName({ kind: 'b', layer: l, j })} = ${params.b[l - 1][j].toFixed(4)}${reveal.gradFrom <= l ? ` · ∂L/∂b = ${g.db[l - 1][j].toFixed(4)}` : ''}`}</title>
                  <rect x={bx - 5} y={by - 5} width={10} height={10} transform={`rotate(45 ${bx} ${by})`} fill={isSel ? c.cyan : c.panel} stroke={isSel ? c.cyan : c.amber} strokeWidth={1.2} />
                </g>
              )
            })}
          </g>
        )
      })}
    </svg>
  )
})

function Pulses({ pos, step, c, L, lossX, lossY, sizes }: { pos: Pos[][]; step: NetView['step']; c: Colors; L: number; lossX: number; lossY: number; sizes: number[] }) {
  const segs: { x1: number; y1: number; x2: number; y2: number; col: string }[] = []
  const pairs = (l: number, backward: boolean, col: string) => {
    if (sizes[l] * sizes[l - 1] > 64) return
    for (let j = 0; j < sizes[l]; j++) for (let i = 0; i < sizes[l - 1]; i++) {
      const a = pos[l - 1][i], b = pos[l][j]
      segs.push(backward ? { x1: b.x, y1: b.y, x2: a.x, y2: a.y, col } : { x1: a.x, y1: a.y, x2: b.x, y2: b.y, col })
    }
  }
  if (step.kind === 'z') pairs(step.layer, false, c.cyan)
  if (step.kind === 'delta') pairs(step.layer + 1, true, c.rose)
  if (step.kind === 'loss' && sizes[L] <= 20) pos[L].forEach((p) => segs.push({ x1: p.x, y1: p.y, x2: lossX - 34, y2: lossY, col: c.amber }))
  if (step.kind === 'dout' && sizes[L] <= 20) pos[L].forEach((p) => segs.push({ x1: lossX - 34, y1: lossY, x2: p.x, y2: p.y, col: c.rose }))
  return (
    <g pointerEvents="none">
      {segs.map((s, k) => (
        <motion.circle key={k} r={3.2} fill={s.col} initial={{ cx: s.x1, cy: s.y1, opacity: 0 }} animate={{ cx: [s.x1, s.x2], cy: [s.y1, s.y2], opacity: [0, 1, 1, 0] }}
          transition={{ duration: 1.1, repeat: Infinity, repeatDelay: 0.25, ease: 'easeInOut', delay: (k % 7) * 0.04 }} />
      ))}
    </g>
  )
}

export function EdgeLegend() {
  const c = useColors()
  const edgeMode = useMlp((s) => s.edgeMode), setEdgeMode = useMlp((s) => s.setEdgeMode)
  const sw = (col: string) => <span className="inline-block h-[3px] w-5 rounded" style={{ background: col }} />
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11.5px] text-muted">
      <span className="flex items-center gap-1.5">{sw(c.amber)} w &gt; 0 {sw(c.violet)} w &lt; 0</span>
      <span className="flex items-center gap-1.5">{sw(c.rose)} gradient / step &gt; 0 {sw(c.teal)} &lt; 0</span>
      <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-3 rounded-full border-2" style={{ borderColor: c.rose }} /> ring = δ (error)</span>
      <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rotate-45 border" style={{ borderColor: c.amber }} /> bias</span>
      <span className="flex items-center gap-1.5">width ∝ magnitude</span>
      <div className="ml-auto flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Edges show">
        {(['auto', 'weights', 'grads'] as const).map((m) => (
          <button key={m} type="button" role="radio" aria-checked={edgeMode === m} onClick={() => setEdgeMode(m)} className={`rounded px-2 py-0.5 text-[11px] ${edgeMode === m ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>
            {m === 'auto' ? 'follow step' : m === 'weights' ? 'weights' : 'gradients'}
          </button>
        ))}
      </div>
    </div>
  )
}
