/** Shared pieces for module 02: role colours, the derived block trace, an SVG matrix view and small helpers. */
import { interpolateRgb } from 'd3'
import { memo, useMemo, useState, type ReactNode } from 'react'
import { useApp, useColors } from '../../sim/appState'
import { runBlock, type BlockTrace, type Matrix } from '../../engine/block'
import { cfgOf, useTb } from './store'

/** role colours from the brief: tokens cyan, Q purple, K blue, V orange, scores yellow, residual green, FFN pink */
export function useRoles() {
  const { theme } = useApp()
  return theme === 'dark'
    ? { tok: '#45e1f2', q: '#b48cff', k: '#5b9dff', v: '#ff9a3c', score: '#ffd84a', res: '#46d97e', ffn: '#ff6fb3', ln: '#9fb2c8' }
    : { tok: '#0a8aa0', q: '#6d3fd6', k: '#1f5fd1', v: '#c45f00', score: '#9a7a00', res: '#178a45', ffn: '#c22d79', ln: '#4b5d73' }
}
export type Roles = ReturnType<typeof useRoles>

export function useTrace(): BlockTrace {
  const arch = useTb((s) => s.arch), T = useTb((s) => s.T), d = useTb((s) => s.d), H = useTb((s) => s.H), dff = useTb((s) => s.dff), act = useTb((s) => s.act), eps = useTb((s) => s.eps), maskOff = useTb((s) => s.maskOff)
  const params = useTb((s) => s.params), X = useTb((s) => s.X), iv = useTb((s) => s.iv)
  return useMemo(() => runBlock(cfgOf({ arch, T, d, H, dff, act, eps, maskOff }), params, X, iv), [arch, T, d, H, dff, act, eps, maskOff, params, X, iv])
}

export const f3 = (v: number) => (v === -Infinity ? '−∞' : v === Infinity ? '∞' : Number.isNaN(v) ? 'NaN' : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(3).replace('-', '−'))
export const f4 = (v: number) => (v === -Infinity ? '−∞' : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(4).replace('-', '−'))
/** a number for TeX (parenthesized when negative) */
export const tn = (v: number, d = 3) => (v === -Infinity ? '-\\infty' : v < 0 && Math.abs(v) >= 5e-13 ? `(${v.toFixed(d)})` : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d))

export interface MatrixProps {
  m: Matrix
  title?: ReactNode
  rowLabels?: string[]
  colLabels?: string[]
  /** single-hue (prob / positive) or diverging */
  mode?: 'diverging' | 'prob'
  color?: string
  selected?: [number, number] | null
  hlRows?: number[]
  hlCols?: number[]
  onCell?: (r: number, c: number) => void
  digits?: number
  cell?: number
  colGroup?: number
  groupColors?: string[]
  showValues?: boolean
  ariaLabel: string
  testid?: string
  maxAbs?: number
  dimCols?: number[]
}

/** crisp SVG matrix with values, row/column highlighting and clickable cells (hover shows the exact value) */
export const MatrixView = memo(function MatrixView({ m, title, rowLabels, colLabels, mode = 'diverging', color, selected, hlRows = [], hlCols = [], onCell, digits = 2, cell, colGroup, groupColors, showValues = true, ariaLabel, testid, maxAbs, dimCols = [] }: MatrixProps) {
  const c = useColors()
  const [hover, setHover] = useState<[number, number] | null>(null)
  const R = m.length, C = m[0]?.length ?? 0
  const cw = cell ?? (C > 12 ? 34 : 46), ch = Math.min(30, cw * 0.7 + 6)
  const lw = rowLabels ? 62 : 6, th = colLabels ? 20 : 4
  const W = lw + C * cw + 4, H = th + R * ch + 4
  const finite = m.flat().filter(Number.isFinite)
  const ma = maxAbs ?? Math.max(1e-9, ...finite.map(Math.abs))
  const base = color ?? c.cyan
  const pos = interpolateRgb(c.panel2, base), neg = interpolateRgb(c.panel2, c.muted)
  const fill = (v: number) => {
    if (!Number.isFinite(v)) return c.bg
    if (mode === 'prob') return pos(Math.sqrt(Math.max(0, Math.min(1, v / (maxAbs ?? 1)))))
    const t = Math.min(1, Math.abs(v) / ma)
    return v >= 0 ? pos(Math.sqrt(t)) : neg(Math.sqrt(t))
  }
  const hv = hover ?? selected
  return (
    <div className="min-w-0">
      {title && <div className="mb-1 text-[12px] text-muted">{title}</div>}
      <div className="overflow-x-auto scroll-thin">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} data-testid={testid} className="block">
          {colLabels?.map((l, j) => <text key={j} x={lw + j * cw + cw / 2} y={13} textAnchor="middle" fontSize={10.5} fill={hlCols.includes(j) ? c.fg : c.muted} fontFamily="var(--font-mono)">{l}</text>)}
          {rowLabels?.map((l, i) => <text key={i} x={lw - 6} y={th + i * ch + ch / 2 + 4} textAnchor="end" fontSize={11} fill={hlRows.includes(i) ? c.fg : c.muted} fontFamily="var(--font-mono)">{l.length > 8 ? l.slice(0, 7) + '…' : l}</text>)}
          {colGroup && groupColors && Array.from({ length: Math.ceil(C / colGroup) }, (_, g) => (
            <rect key={g} x={lw + g * colGroup * cw} y={th - 3} width={colGroup * cw} height={2.5} fill={groupColors[g % groupColors.length]} />
          ))}
          {m.map((row, i) => row.map((v, j) => {
            const isSel = selected && selected[0] === i && selected[1] === j
            const hl = hlRows.includes(i) || hlCols.includes(j)
            const dim = dimCols.includes(j)
            return (
              <g key={`${i}.${j}`} onClick={onCell ? () => onCell(i, j) : undefined} onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)} style={onCell ? { cursor: 'pointer' } : undefined}>
                <rect x={lw + j * cw + 1} y={th + i * ch + 1} width={cw - 2} height={ch - 2} rx={4} fill={fill(v)} opacity={dim ? 0.3 : 1}
                  stroke={isSel ? c.fg : hl ? base : 'transparent'} strokeWidth={isSel ? 2.2 : 1.2} data-testid={testid ? `${testid}-${i}-${j}` : undefined} />
                {showValues && (
                  <text x={lw + j * cw + cw / 2} y={th + i * ch + ch / 2 + 4} textAnchor="middle" fontSize={cw < 40 ? 9.5 : 11} fontFamily="var(--font-mono)" pointerEvents="none"
                    fill={!Number.isFinite(v) ? c.faint : mode === 'prob' ? (v / (maxAbs ?? 1) > 0.45 ? c.bg : c.fg) : Math.abs(v) / ma > 0.6 ? c.bg : c.fg}>
                    {Number.isFinite(v) ? (Math.abs(v) < 5e-13 ? 0 : v).toFixed(digits).replace('-', '−') : '−∞'}
                  </text>
                )}
              </g>
            )
          }))}
        </svg>
      </div>
      {hv && Number.isFinite(m[hv[0]]?.[hv[1]]) && <div className="mt-0.5 font-mono text-[11px] text-faint">[{rowLabels?.[hv[0]] ?? hv[0]}, {colLabels?.[hv[1]] ?? hv[1]}] = {m[hv[0]][hv[1]].toPrecision(8)}</div>}
    </div>
  )
})

export function RoleLegend() {
  const r = useRoles()
  const items: [string, string][] = [[r.tok, 'token / hidden state'], [r.q, 'query'], [r.k, 'key'], [r.v, 'value'], [r.score, 'attention score'], [r.res, 'residual'], [r.ffn, 'feedforward']]
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-muted" aria-label="Colour legend">
      {items.map(([col, l]) => <span key={l} className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: col }} />{l}</span>)}
    </div>
  )
}

/** a signed horizontal bar strip for one vector */
export function VecBars({ v, color, labels, height = 90, maxAbs, highlight, testid }: { v: number[]; color: string; labels?: string[]; height?: number; maxAbs?: number; highlight?: number; testid?: string }) {
  const c = useColors()
  const W = Math.max(160, v.length * 44), mid = height / 2, ma = maxAbs ?? Math.max(1e-9, ...v.map(Math.abs)), bw = W / v.length
  return (
    <svg viewBox={`0 0 ${W} ${height + 16}`} className="block w-full max-w-[300px]" role="img" aria-label="vector" data-testid={testid}>
      <line x1={0} x2={W} y1={mid} y2={mid} stroke={c.lineStrong} />
      {v.map((x, k) => {
        const h = (Math.abs(x) / ma) * (mid - 12)
        return (
          <g key={k}>
            <rect x={k * bw + 6} y={x >= 0 ? mid - h : mid} width={bw - 12} height={Math.max(1, h)} rx={3} fill={color} opacity={highlight === undefined || highlight === k ? 1 : 0.45} />
            <text x={k * bw + bw / 2} y={x >= 0 ? mid - h - 3 : mid + h + 11} textAnchor="middle" fontSize={10} fill={c.fg} fontFamily="var(--font-mono)">{x.toFixed(2).replace('-', '−')}</text>
            <text x={k * bw + bw / 2} y={height + 13} textAnchor="middle" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">{labels?.[k] ?? k + 1}</text>
          </g>
        )
      })}
    </svg>
  )
}
