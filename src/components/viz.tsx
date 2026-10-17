/** Reusable SVG visualizations: heatmaps, vector bars, probability bars, token chips, KaTeX. */
import katex from 'katex'
import { interpolateRgb } from 'd3'
import { memo, useMemo, useState, type ReactNode } from 'react'
import { useColors, type Colors } from '../sim/appState'
import { showToken } from '../engine/model'
import type { Mat } from '../engine/tensor'

export function Tex({ tex, block = false, className = '' }: { tex: string; block?: boolean; className?: string }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: block, throwOnError: false, output: 'html' }), [tex, block])
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}

export const fmt = (v: number, d = 3) =>
  v === -Infinity ? '−∞' : v === Infinity ? '∞' : Number.isNaN(v) ? 'NaN' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(d)

export function divergingColor(c: Colors, maxAbs: number) {
  const neg = interpolateRgb(c.panel2, c.violet), pos = interpolateRgb(c.panel2, c.amber)
  return (v: number) => {
    const t = Math.min(1, Math.abs(v) / (maxAbs || 1))
    return v < 0 ? neg(Math.sqrt(t)) : pos(Math.sqrt(t))
  }
}
export function seqColor(c: Colors, max = 1) {
  const f = interpolateRgb(c.panel2, c.cyan)
  return (v: number) => f(Math.min(1, Math.sqrt(Math.max(0, v) / (max || 1))))
}

interface HeatmapProps {
  m: Mat
  rowLabels?: string[]
  colLabels?: string[]
  mode?: 'diverging' | 'prob'
  maxAbs?: number
  selectedRow?: number
  selectedCell?: [number, number] | null
  onCell?: (i: number, j: number) => void
  colGroup?: number
  mask?: boolean
  showValues?: boolean
  cell?: number
  title?: string
  ariaLabel?: string
  /** decimals for printed values (diverging mode); default 1 */
  digits?: number
}

/** Matrix heatmap. Rows are tokens (usually). Hover shows the exact value. */
export const Heatmap = memo(function Heatmap({
  m, rowLabels, colLabels, mode = 'diverging', maxAbs, selectedRow, selectedCell, onCell, colGroup, mask, showValues, cell, title, ariaLabel, digits,
}: HeatmapProps) {
  const c = useColors()
  const [hover, setHover] = useState<[number, number] | null>(null)
  const cw = cell ?? (m.cols > 40 ? 11 : m.cols > 16 ? 18 : 34)
  const ch = cell ?? (m.cols > 40 ? 18 : 30)
  const lw = rowLabels ? 76 : 0
  const th = colLabels ? 64 : title ? 18 : 4
  const W = lw + m.cols * cw + 2
  const H = th + m.rows * ch + 2
  const ma = maxAbs ?? Math.max(1e-6, ...Array.from(m.data).filter(Number.isFinite).map(Math.abs))
  const color = mode === 'prob' ? seqColor(c, 1) : divergingColor(c, ma)
  const hv = hover ? m.data[hover[0] * m.cols + hover[1]] : null
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W * 1.6, display: 'block' }} role="img" aria-label={ariaLabel ?? title ?? 'matrix heatmap'}
        onMouseLeave={() => setHover(null)}>
        {title && <text x={lw} y={12} fontSize={11} fill={c.muted} fontFamily="JetBrains Mono, monospace">{title}</text>}
        {colLabels?.map((l, j) => (
          <text key={j} x={lw + j * cw + cw / 2} y={th - 6} fontSize={11} fill={c.muted} textAnchor="start"
            transform={`rotate(-50 ${lw + j * cw + cw / 2} ${th - 6})`} fontFamily="JetBrains Mono, monospace">{l}</text>
        ))}
        {rowLabels?.map((l, i) => (
          <text key={i} x={lw - 6} y={th + i * ch + ch / 2 + 4} fontSize={11.5} textAnchor="end"
            fill={i === selectedRow ? c.amber : c.fg} fontFamily="JetBrains Mono, monospace">{l}</text>
        ))}
        {Array.from({ length: m.rows }, (_, i) =>
          Array.from({ length: m.cols }, (_, j) => {
            const v = m.data[i * m.cols + j]
            const masked = mask && j > i
            return (
              <rect key={`${i}-${j}`} x={lw + j * cw} y={th + i * ch} width={cw - 1} height={ch - 1} rx={1.5}
                fill={masked ? c.bg : color(v)} stroke={masked ? c.line : 'none'} strokeDasharray={masked ? '2 2' : undefined}
                style={{ cursor: onCell ? 'pointer' : 'default' }}
                onMouseEnter={() => setHover([i, j])} onClick={() => onCell?.(i, j)} />
            )
          }),
        )}
        {showValues && Array.from({ length: m.rows }, (_, i) =>
          Array.from({ length: m.cols }, (_, j) => {
            const v = m.data[i * m.cols + j]
            if (mask && j > i) return null
            return <text key={`t${i}-${j}`} x={lw + j * cw + cw / 2} y={th + i * ch + ch / 2 + 4} fontSize={10} textAnchor="middle"
              fill={c.fg} pointerEvents="none" fontFamily="JetBrains Mono, monospace">{mode === 'prob' ? v.toFixed(2) : fmt(v, digits ?? 1)}</text>
          }),
        )}
        {colGroup && Array.from({ length: Math.floor(m.cols / colGroup) - 1 }, (_, g) => (
          <line key={g} x1={lw + (g + 1) * colGroup * cw - 0.5} x2={lw + (g + 1) * colGroup * cw - 0.5} y1={th - 2} y2={H} stroke={c.cyan} strokeOpacity={0.6} strokeDasharray="3 3" />
        ))}
        {selectedRow !== undefined && selectedRow >= 0 && (
          <rect x={lw - 1} y={th + selectedRow * ch - 1} width={m.cols * cw + 1} height={ch + 1} fill="none" stroke={c.amber} strokeWidth={1.5} rx={2} pointerEvents="none" />
        )}
        {selectedCell && (
          <rect x={lw + selectedCell[1] * cw - 1} y={th + selectedCell[0] * ch - 1} width={cw + 1} height={ch + 1} fill="none" stroke={c.cyan} strokeWidth={2} rx={2} pointerEvents="none" />
        )}
      </svg>
      {hover && hv !== null && (
        <div className="pointer-events-none absolute right-1 top-0 rounded-md border border-line bg-panel px-2 py-1 font-mono text-[11px] text-fg shadow-lg font-num">
          [{hover[0]}, {hover[1]}]{rowLabels ? ` ${rowLabels[hover[0]]}` : ''} = <span className="text-amber">{fmt(hv, 4)}</span>
        </div>
      )}
    </div>
  )
})

/** One vector drawn as a bar sparkline (positive up, negative down). */
export function VectorBars({ v, height = 70, color, label, maxAbs, highlight }: {
  v: ArrayLike<number>; height?: number; color?: string; label?: ReactNode; maxAbs?: number; highlight?: number
}) {
  const c = useColors()
  const n = v.length
  const ma = maxAbs ?? Math.max(1e-6, ...Array.from(v, Math.abs))
  const w = 6
  const mid = height / 2
  return (
    <div className="min-w-0">
      {label && <div className="mb-1 font-mono text-[11px] text-muted">{label}</div>}
      <svg viewBox={`0 0 ${n * w} ${height}`} width="100%" height={height} preserveAspectRatio="none" role="img" aria-label="vector values">
        <line x1={0} x2={n * w} y1={mid} y2={mid} stroke={c.line} />
        {Array.from(v, (x, i) => {
          const h = (Math.abs(x) / ma) * (mid - 2)
          return <rect key={i} x={i * w + 0.5} width={w - 1} y={x >= 0 ? mid - h : mid} height={Math.max(0.5, h)}
            fill={i === highlight ? c.cyan : color ?? (x >= 0 ? c.amber : c.violet)} />
        })}
      </svg>
    </div>
  )
}

/** Labelled horizontal probability / logit bars. */
export function ProbBars({ items, max, unit = 'prob', highlight, accent }: {
  items: { label: string; value: number; sub?: string; dim?: boolean }[]; max?: number; unit?: 'prob' | 'logit'; highlight?: number; accent?: string
}) {
  const c = useColors()
  const mx = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-9)
  return (
    <div className="grid gap-[3px]">
      {items.map((it, k) => (
        <div key={k} className="grid grid-cols-[96px_1fr_64px] items-center gap-2 font-mono text-[12px]" style={{ opacity: it.dim ? 0.35 : 1 }}>
          <span className={`truncate text-right ${k === highlight ? 'text-amber' : 'text-fg'}`}>{it.label}</span>
          <div className="h-[14px] rounded-sm" style={{ background: c.panel2 }}>
            <div className="h-full rounded-sm transition-[width] duration-500"
              style={{ width: `${Math.max(0, (Math.abs(it.value) / mx) * 100)}%`, background: k === highlight ? c.amber : accent ?? (unit === 'logit' && it.value < 0 ? c.violet : c.cyan) }} />
          </div>
          <span className="text-right text-muted font-num">{unit === 'prob' ? `${(it.value * 100).toFixed(it.value < 0.001 ? 3 : 1)}%` : fmt(it.value, 2)}</span>
        </div>
      ))}
    </div>
  )
}

export function TokenStrip({ tokens, ids, selected, onSelect, extra }: {
  tokens: string[]; ids?: number[]; selected?: number; onSelect?: (i: number) => void; extra?: (i: number) => ReactNode
}) {
  return (
    <div className="flex flex-wrap gap-1.5" role="listbox" aria-label="tokens">
      {tokens.map((t, i) => (
        <button key={i} type="button" className="chip" data-selected={i === selected} onClick={() => onSelect?.(i)} role="option" aria-selected={i === selected}>
          <span>{showToken(t)}</span>
          {ids && <span className="id">{ids[i]}</span>}
          {extra?.(i)}
        </button>
      ))}
    </div>
  )
}

export function Card({ title, children, className = '', right }: { title?: ReactNode; children: ReactNode; className?: string; right?: ReactNode }) {
  return (
    <section className={`min-w-0 rounded-xl border border-line bg-panel/80 p-3 ${className}`}>
      {(title || right) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="eyebrow">{title}</div>
          {right}
        </div>
      )}
      {children}
    </section>
  )
}
