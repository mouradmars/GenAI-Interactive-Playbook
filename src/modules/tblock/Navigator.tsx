/** Architecture navigator (clickable, depth-shaded block diagram) and the token & attention bar. */
import { motion, useReducedMotion } from 'framer-motion'
import { Pencil } from 'lucide-react'
import { useState } from 'react'
import { useColors } from '../../sim/appState'
import type { BlockTrace } from '../../engine/block'
import { useRoles } from './common'
import { ARCH_SHORT, firstStageOf, type NodeId, type Stage } from './stages'
import { useTb } from './store'

interface NodeSpec { id: NodeId; label: string; sub?: string; color: string; w: number }

export function ArchNavigator({ stage, onSeek }: { stage: Stage; onSeek: (i: number) => void }) {
  const c = useColors(), r = useRoles()
  const arch = useTb((s) => s.arch), H = useTb((s) => s.H), maskOff = useTb((s) => s.maskOff), act = useTb((s) => s.act)
  const nodes: NodeSpec[] = arch === 'gpt'
    ? [
      { id: 'in', label: 'X', sub: 'input', color: r.tok, w: 62 }, { id: 'ln1', label: 'LayerNorm', sub: 'LN₁', color: r.ln, w: 84 },
      { id: 'mhsa', label: 'Masked MHSA', sub: `${H} head${H > 1 ? 's' : ''} · ${maskOff ? 'mask OFF' : 'causal'}`, color: r.score, w: 128 },
      { id: 'add1', label: '⊕', sub: 'residual', color: r.res, w: 50 }, { id: 'ln2', label: 'LayerNorm', sub: 'LN₂', color: r.ln, w: 84 },
      { id: 'ffn', label: 'MLP', sub: act === 'swiglu' ? 'SwiGLU' : act.toUpperCase(), color: r.ffn, w: 84 }, { id: 'add2', label: '⊕', sub: 'residual', color: r.res, w: 50 },
      { id: 'out', label: 'Y', sub: 'output', color: r.tok, w: 62 },
    ]
    : [
      { id: 'in', label: 'X', sub: 'input', color: r.tok, w: 62 },
      { id: 'mhsa', label: 'MHSA', sub: `${H} head${H > 1 ? 's' : ''} · bidirectional`, color: r.score, w: 140 },
      { id: 'add1', label: '⊕ + LN₁', sub: 'add, then norm', color: r.res, w: 92 },
      { id: 'ffn', label: 'MLP', sub: act === 'swiglu' ? 'SwiGLU' : act.toUpperCase(), color: r.ffn, w: 84 },
      { id: 'add2', label: '⊕ + LN₂', sub: 'add, then norm', color: r.res, w: 92 },
      { id: 'out', label: 'Y', sub: 'output', color: r.tok, w: 62 },
    ]
  const gap = 26, depth = 7, Hh = 116, y0 = 44, bh = 46
  const xs: number[] = []
  let x = 12
  for (const n of nodes) { xs.push(x); x += n.w + gap }
  const W = x - gap + 12
  const idx = (id: NodeId) => nodes.findIndex((n) => n.id === id)
  // residual skip arcs: from the stream before each sublayer to its ⊕
  const skips: [number, number][] = arch === 'gpt' ? [[0, idx('add1')], [idx('add1'), idx('add2')]] : [[0, idx('add1')], [idx('add1'), idx('add2')]]
  const cx = (k: number) => xs[k] + nodes[k].w / 2
  return (
    <div className="overflow-x-auto scroll-thin rounded-xl border border-line bg-panel/60 p-2" aria-label="Architecture navigator">
      <div className="mb-1 flex items-center justify-between gap-2 px-1">
        <span className="eyebrow">architecture · click a component</span>
        <span className="chip text-cyan" data-testid="arch-label">{ARCH_SHORT[arch]}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${Hh}`} className="mx-auto block w-full" style={{ maxWidth: W * 1.35, minWidth: W * 0.8 }} role="img" aria-label="Transformer block diagram">
        {skips.map(([a, b], k) => {
          const x1 = cx(a), x2 = cx(b), yy = 16
          return <path key={k} d={`M${x1},${y0} C${x1},${yy} ${x2},${yy} ${x2},${y0}`} fill="none" stroke={r.res} strokeWidth={2} strokeDasharray="5 4" opacity={0.85} />
        })}
        <text x={(cx(0) + cx(idx('add1'))) / 2} y={14} textAnchor="middle" fontSize={10} fill={r.res} fontFamily="var(--font-mono)">residual stream</text>
        {nodes.slice(0, -1).map((n, k) => <line key={k} x1={xs[k] + n.w} y1={y0 + bh / 2} x2={xs[k + 1] - 3} y2={y0 + bh / 2} stroke={c.lineStrong} strokeWidth={1.5} markerEnd="url(#navArrow)" />)}
        <defs><marker id="navArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c.lineStrong} /></marker></defs>
        {nodes.map((n, k) => {
          const on = stage.node === n.id, X0 = xs[k]
          return (
            <g key={n.id} style={{ cursor: 'pointer' }} onClick={() => onSeek(firstStageOf(arch, n.id))} data-testid={`nav-${n.id}`} role="button" aria-label={`Go to ${n.label}`}>
              {/* depth faces */}
              <path d={`M${X0},${y0} L${X0 + depth},${y0 - depth} L${X0 + n.w + depth},${y0 - depth} L${X0 + n.w},${y0} Z`} fill={n.color} opacity={on ? 0.55 : 0.18} />
              <path d={`M${X0 + n.w},${y0} L${X0 + n.w + depth},${y0 - depth} L${X0 + n.w + depth},${y0 + bh - depth} L${X0 + n.w},${y0 + bh} Z`} fill={n.color} opacity={on ? 0.4 : 0.12} />
              <rect x={X0} y={y0} width={n.w} height={bh} rx={6} fill={c.panel} stroke={n.color} strokeWidth={on ? 2.6 : 1.2} />
              {on && <motion.rect x={X0 - 3} y={y0 - 3} width={n.w + 6} height={bh + 6} rx={8} fill="none" stroke={n.color} strokeWidth={1} initial={{ opacity: 0.2 }} animate={{ opacity: [0.2, 0.8, 0.2] }} transition={{ duration: 1.8, repeat: Infinity }} />}
              <text x={X0 + n.w / 2} y={y0 + 20} textAnchor="middle" fontSize={13} fill={on ? n.color : c.fg} fontFamily="var(--font-mono)">{n.label}</text>
              <text x={X0 + n.w / 2} y={y0 + 36} textAnchor="middle" fontSize={9.5} fill={c.muted} fontFamily="var(--font-mono)">{n.sub}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** token chips with editable labels; during attention stages, arcs show the selected head's weights from the query token */
export function TokenBar({ trace, showArcs }: { trace: BlockTrace; showArcs: boolean }) {
  const c = useColors(), r = useRoles()
  const reduce = useReducedMotion()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), setToken = useTb((s) => s.setToken)
  const [editing, setEditing] = useState<number | null>(null)
  const T = tokens.length, cw = 92, W = T * cw, top = showArcs ? 56 : 4
  const A = trace.heads[sel.h]?.A
  return (
    <div className="overflow-x-auto scroll-thin rounded-xl border border-line bg-panel/60 p-2" aria-label="Token and attention explorer">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2 px-1 text-[11.5px] text-muted">
        <span className="eyebrow">tokens · click to select the query (and shift-click a key)</span>
        <span><span style={{ color: r.tok }}>●</span> query token {sel.i + 1} <span className="ml-2" style={{ color: r.k }}>●</span> key token {sel.j + 1}{showArcs ? ` · arcs: head ${sel.h + 1} weights from “${tokens[sel.i]}”` : ''}</span>
      </div>
      <svg width={W} height={top + 40} viewBox={`0 0 ${W} ${top + 40}`} className="mx-auto block" role="img" aria-label="Tokens">
        {showArcs && A && A[sel.i].map((w, j) => {
          if (w <= 0) return null
          const x1 = sel.i * cw + cw / 2, x2 = j * cw + cw / 2
          const h = j === sel.i ? 0 : Math.min(48, 14 + Math.abs(x2 - x1) * 0.18)
          return j === sel.i
            ? <circle key={j} cx={x1} cy={top - 12} r={4 + 8 * w} fill="none" stroke={r.score} strokeWidth={1 + 5 * w} opacity={0.35 + 0.65 * w} />
            : <motion.path key={`${sel.i}-${j}-${sel.h}`} d={`M${x1},${top - 2} Q${(x1 + x2) / 2},${top - 2 - h * 1.6} ${x2},${top - 2}`} fill="none" stroke={r.score} strokeWidth={1 + 7 * w} opacity={0.3 + 0.7 * w} strokeLinecap="round"
              initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.6 }} />
        })}
        {tokens.map((t, i) => {
          const q = i === sel.i, k = i === sel.j
          const masked = showArcs && A && A[sel.i][i] === 0 && i !== sel.i
          return (
            <g key={i} transform={`translate(${i * cw + 4}, ${top})`} style={{ cursor: 'pointer' }}
              onClick={(e) => (e.shiftKey ? select({ j: i, cell: null }) : select({ i, cell: null, j: Math.min(sel.j, trace.cfg.causal ? i : trace.cfg.T - 1) }))} data-testid={`token-${i}`}>
              <rect width={cw - 8} height={32} rx={8} fill={c.panel2} stroke={q ? r.tok : k ? r.k : c.line} strokeWidth={q || k ? 2.2 : 1} opacity={masked ? 0.4 : 1} />
              <text x={(cw - 8) / 2} y={20} textAnchor="middle" fontSize={12.5} fill={masked ? c.faint : c.fg} fontFamily="var(--font-mono)">{t.length > 10 ? t.slice(0, 9) + '…' : t}</text>
              <text x={4} y={-3} fontSize={9} fill={c.faint} fontFamily="var(--font-mono)">{i + 1}</text>
            </g>
          )
        })}
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-1 px-1">
        {editing === null
          ? <button type="button" className="btn !py-0.5 text-[11.5px]" onClick={() => setEditing(sel.i)}><Pencil size={12} />rename token {sel.i + 1}</button>
          : <form className="flex items-center gap-1" onSubmit={(e) => { e.preventDefault(); setEditing(null) }}>
              <input autoFocus defaultValue={tokens[editing]} onChange={(e) => setToken(editing, e.target.value)} className="w-32 rounded border border-line bg-panel-2 px-2 py-0.5 font-mono text-[12px]" aria-label={`Label of token ${editing + 1}`} />
              <button type="submit" className="btn !py-0.5 text-[11.5px]">done</button>
            </form>}
        <span className="text-[11px] text-faint">Renaming changes the (toy) embedding lookup upstream, so X changes too.</span>
      </div>
    </div>
  )
}
