/**
 * Microscope A · token relationship view. Queries on the left, keys on the right, one connection per pair:
 * width = raw score magnitude (raw mode) or attention probability (normalized mode). Masked pairs are dashed
 * and marked ⊘. Clicking a connection or a token updates the shared selection.
 */
import { motion, useReducedMotion } from 'framer-motion'
import type { ReactElement } from 'react'
import { useColors } from '../../sim/appState'
import type { AttnTrace } from '../../engine/attention'
import { GLYPH, fx, pct, useRoles } from './common'
import { useAttn } from './store'

export function TokenGraph({ trace, tokens, mode, allRows = false, compact = false, head }: {
  trace: AttnTrace; tokens: string[]; mode: 'raw' | 'prob'; allRows?: boolean; compact?: boolean; head?: number
}) {
  const c = useColors(), r = useRoles()
  const reduce = useReducedMotion()
  const sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const h = head ?? sel.h
  const hd = trace.heads[Math.min(h, trace.heads.length - 1)]
  const n = trace.Tq, m = trace.Tk
  const rowH = compact ? 34 : 44, top = 26, W = compact ? 330 : 440, nodeW = compact ? 92 : 112
  const H = top + Math.max(n, m) * rowH + 8
  const xL = 6 + nodeW, xR = W - 6 - nodeW
  const y = (k: number) => top + k * rowH + rowH / 2
  const maxRaw = Math.max(1e-9, ...hd.scaled.flat().filter(Number.isFinite).map(Math.abs))
  const keyTok = trace.self ? tokens : tokens.map((_, k) => `src ${k + 1}`)
  const edges: ReactElement[] = []
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const focus = allRows || i === sel.i
    const masked = trace.M[i][j] === -Infinity
    const raw = hd.scaled[i][j], p = hd.A[i][j]
    const isSel = i === sel.i && j === sel.j
    const t = mode === 'prob' ? p : Math.abs(raw) / maxRaw
    const width = masked ? 1.2 : 1 + (compact ? 7 : 10) * t
    const col = masked ? r.mask : mode === 'prob' ? r.prob : r.score
    const dash = masked ? '4 4' : mode === 'raw' && raw < 0 ? '1.5 3.5' : undefined
    const d = `M${xL},${y(i)} C${(xL + xR) / 2},${y(i)} ${(xL + xR) / 2},${y(j)} ${xR},${y(j)}`
    edges.push(
      <g key={`${i}-${j}`} onClick={() => select({ i, j, cell: null })} style={{ cursor: 'pointer' }} data-testid={`edge-${i}-${j}`}>
        <title>{`query ${i + 1} → key ${j + 1}: ${masked ? 'masked' : `score ${fx(raw, 3)}, weight ${pct(p)}`}`}</title>
        <path d={d} fill="none" stroke="transparent" strokeWidth={12} />
        {isSel && !masked && <path d={d} fill="none" stroke={c.fg} strokeWidth={width + 3} opacity={0.5} strokeLinecap="round" />}
        <motion.path key={`${mode}-${h}-${sel.i}`} d={d} fill="none" stroke={col} strokeWidth={width} strokeDasharray={dash} strokeLinecap="round"
          opacity={focus ? (masked ? 0.7 : 0.25 + 0.75 * Math.min(1, t + 0.15)) : 0.07}
          initial={reduce || !focus ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.55, delay: focus ? j * 0.06 : 0 }} />
        {focus && !allRows && (
          <text x={xR - 8} y={y(j) - 6} textAnchor="end" fontSize={10.5} fontFamily="var(--font-mono)" fill={masked ? r.mask : col}>
            {masked ? '⊘ masked' : mode === 'prob' ? pct(p) : fx(raw, 2)}
          </text>
        )}
      </g>,
    )
  }
  const node = (k: number, side: 'q' | 'k') => {
    const isQ = side === 'q'
    const on = isQ ? k === sel.i : k === sel.j
    const x0 = isQ ? 6 : xR
    const label = isQ ? tokens[k] : keyTok[k]
    const roleCol = isQ ? r.q : r.k
    return (
      <g key={`${side}${k}`} transform={`translate(${x0}, ${y(k) - 14})`} style={{ cursor: 'pointer' }} onClick={() => select(isQ ? { i: k, cell: null } : { j: k, cell: null })} data-testid={`gnode-${side}-${k}`}
        role="button" tabIndex={0} aria-label={`${isQ ? 'Query' : 'Key'} token ${k + 1} ${label}`} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(isQ ? { i: k } : { j: k }) } }}>
        <rect width={nodeW} height={28} rx={7} fill={c.panel2} stroke={on ? roleCol : c.line} strokeWidth={on ? 2.2 : 1} />
        <text x={8} y={18} fontSize={11} fill={roleCol} fontFamily="var(--font-mono)">{GLYPH[side]}</text>
        <text x={22} y={18} fontSize={compact ? 11 : 12} fill={c.fg} fontFamily="var(--font-mono)">{k + 1}·{label.length > (compact ? 7 : 9) ? label.slice(0, compact ? 6 : 8) + '…' : label}</text>
      </g>
    )
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" style={{ maxWidth: W * 1.25 }} role="img" aria-label={`Token relationship graph, head ${h + 1}, ${mode === 'prob' ? 'attention probabilities' : 'raw scores'}`} data-testid="token-graph">
      <text x={6} y={14} fontSize={10.5} fill={r.q} fontFamily="var(--font-mono)">{GLYPH.q} queries</text>
      <text x={W - 6} y={14} textAnchor="end" fontSize={10.5} fill={r.k} fontFamily="var(--font-mono)">keys {GLYPH.k}</text>
      <text x={W / 2} y={14} textAnchor="middle" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">{mode === 'prob' ? 'width = attention weight' : 'width = |scaled score| · dotted = negative'}</text>
      {edges}
      {Array.from({ length: n }, (_, k) => node(k, 'q'))}
      {Array.from({ length: m }, (_, k) => node(k, 'k'))}
    </svg>
  )
}
