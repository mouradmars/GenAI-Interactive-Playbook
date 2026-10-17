/**
 * Shared pieces for module 03: role colours with non-colour glyphs, the derived attention traces, an SVG matrix
 * grid (clickable, keyboard-accessible, with hatched masked cells) and an editable matrix with validation.
 */
import { interpolateRgb } from 'd3'
import { memo, useId, useMemo, useState, type ReactNode } from 'react'
import { useApp, useColors } from '../../sim/appState'
import { runAttention, type AttnTrace, type Matrix } from '../../engine/attention'
import { cfgOf, useAttn, windowTokens } from './store'

export type Role = 'x' | 'q' | 'k' | 'v' | 'score' | 'prob' | 'mask' | 'out' | 'w'
/** colours from the brief (inputs cyan, Q violet, K blue, V orange, scores amber, probabilities teal, masked muted red, outputs green) */
export function useRoles(): Record<Role, string> {
  const { theme } = useApp()
  return theme === 'dark'
    ? { x: '#45e1f2', q: '#b48cff', k: '#5b9dff', v: '#ff9a3c', score: '#f6c445', prob: '#34d6bd', mask: '#c96a72', out: '#46d97e', w: '#9fb2c8' }
    : { x: '#0a8aa0', q: '#6d3fd6', k: '#1f5fd1', v: '#c45f00', score: '#946800', prob: '#0d8a78', mask: '#b4545e', out: '#178a45', w: '#4b5d73' }
}
/** a glyph per role so meaning never relies on colour alone */
export const GLYPH: Record<Role, string> = { x: '●', q: '◆', k: '■', v: '▲', score: '✶', prob: '%', mask: '⊘', out: '★', w: '▦' }
export const ROLE_NAME: Record<Role, string> = { x: 'input X', q: 'query Q', k: 'key K', v: 'value V', score: 'score', prob: 'probability', mask: 'masked', out: 'output', w: 'weights' }

export function RoleLegend({ compact = false }: { compact?: boolean }) {
  const r = useRoles()
  const items: Role[] = ['x', 'q', 'k', 'v', 'score', 'prob', 'mask', 'out']
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-muted" aria-label="Colour legend" data-testid="attn-legend">
      {!compact && <span className="eyebrow">legend</span>}
      {items.map((k) => (
        <span key={k} className="flex items-center gap-1">
          <span aria-hidden="true" style={{ color: r[k] }} className="font-mono">{GLYPH[k]}</span>{ROLE_NAME[k]}
        </span>
      ))}
    </div>
  )
}
export function RoleTag({ role, children }: { role: Role; children: ReactNode }) {
  const r = useRoles()
  return <span className="font-mono" style={{ color: r[role] }}><span aria-hidden="true">{GLYPH[role]} </span>{children}</span>
}

/* ------------------------------------------------------------------ derived traces */
export function useTrace(): AttnTrace {
  const X = useAttn((s) => s.X), params = useAttn((s) => s.params), H = useAttn((s) => s.H), causal = useAttn((s) => s.causal), scale = useAttn((s) => s.scale)
  const pad = useAttn((s) => s.pad), blocked = useAttn((s) => s.blocked), headsOn = useAttn((s) => s.headsOn)
  return useMemo(() => runAttention(X, params, cfgOf({ H, causal, scale, pad, blocked, headsOn })), [X, params, H, causal, scale, pad, blocked, headsOn])
}
export function useBaselineTrace(): AttnTrace | null {
  const b = useAttn((s) => s.baseline)
  return useMemo(() => (b ? runAttention(b.X, b.params, b.cfg) : null), [b])
}
export function useTokens(): string[] {
  const sentence = useAttn((s) => s.sentence), start = useAttn((s) => s.windowStart), T = useAttn((s) => s.T)
  return useMemo(() => windowTokens(sentence, start, T), [sentence, start, T])
}
/** short display label "3·the" */
export const tokLabel = (tokens: string[], i: number) => `${i + 1}·${tokens[i].length > 7 ? tokens[i].slice(0, 6) + '…' : tokens[i]}`

/* ------------------------------------------------------------------ number formatting */
const clean = (v: number) => (Math.abs(v) < 5e-13 ? 0 : v)
export const fx = (v: number, d = 3) => {
  if (v === -Infinity) return '−∞'
  if (v === Infinity) return '∞'
  if (Number.isNaN(v)) return 'NaN'
  const s = clean(v).toFixed(d)
  return Number(s) === 0 ? (0).toFixed(d) : s.replace('-', '−')
}
/** number for TeX, parenthesized when negative */
export const tx = (v: number, d = 3) => (v === -Infinity ? '-\\infty' : clean(v) < 0 ? `(${clean(v).toFixed(d)})` : clean(v).toFixed(d))
export const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/* ------------------------------------------------------------------ the matrix grid */
export interface GridProps {
  m: Matrix
  role: Role
  title?: ReactNode
  rowLabels?: string[]
  colLabels?: string[]
  /** probability mode: fixed 0…1 scale; signed mode: magnitude scale, negatives in grey with a minus sign */
  mode?: 'signed' | 'prob'
  selected?: [number, number] | null
  hlRows?: number[]
  hlCols?: number[]
  hlCells?: [number, number][]
  /** cells drawn as masked (hatched) */
  masked?: boolean[][]
  dimRows?: number[]
  onCell?: (r: number, c: number) => void
  digits?: number
  cell?: number
  maxAbs?: number
  ariaLabel: string
  testid?: string
  /** draw thin separators every `group` columns (head blocks) */
  group?: number
  groupColors?: string[]
}

export const Grid = memo(function Grid({ m, role, title, rowLabels, colLabels, mode = 'signed', selected, hlRows = [], hlCols = [], hlCells = [], masked, dimRows = [], onCell, digits = 2, cell, maxAbs, ariaLabel, testid, group, groupColors }: GridProps) {
  const c = useColors(), roles = useRoles()
  const uid = useId().replace(/:/g, '')
  const [hover, setHover] = useState<[number, number] | null>(null)
  const R = m.length, C = m[0]?.length ?? 0
  const cw = cell ?? 50, ch = Math.round(Math.min(32, cw * 0.62 + 4))
  const lw = rowLabels ? Math.max(...rowLabels.map((l) => l.length)) * 6.6 + 10 : 4, th = colLabels ? 20 : 4
  const W = lw + C * cw + 4, H = th + R * ch + 4
  const base = roles[role]
  const finite = m.flat().filter(Number.isFinite)
  const ma = maxAbs ?? Math.max(1e-9, ...finite.map(Math.abs))
  const pos = interpolateRgb(c.panel2, base), neg = interpolateRgb(c.panel2, c.muted)
  const fill = (v: number) => {
    if (!Number.isFinite(v)) return c.bg
    const t = mode === 'prob' ? Math.max(0, Math.min(1, v)) : Math.min(1, Math.abs(v) / ma)
    return v >= 0 || mode === 'prob' ? pos(Math.sqrt(t)) : neg(Math.sqrt(t))
  }
  const ink = (v: number) => {
    if (!Number.isFinite(v)) return roles.mask
    const t = mode === 'prob' ? v : Math.abs(v) / ma
    return t > 0.55 && v >= 0 ? c.bg : c.fg
  }
  const isHl = (i: number, j: number) => hlCells.some(([a, b]) => a === i && b === j)
  const hv = hover ?? selected
  return (
    <div className="min-w-0">
      {title && <div className="mb-1 text-[12px] text-muted">{title}</div>}
      <div className="overflow-x-auto scroll-thin">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} data-testid={testid} className="block">
          <defs>
            <pattern id={`hatch-${uid}`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill={c.bg} /><line x1="0" y1="0" x2="0" y2="6" stroke={roles.mask} strokeWidth="1.6" opacity="0.55" />
            </pattern>
          </defs>
          {colLabels?.map((l, j) => <text key={j} x={lw + j * cw + cw / 2} y={13} textAnchor="middle" fontSize={10.5} fill={hlCols.includes(j) ? c.fg : c.muted} fontWeight={hlCols.includes(j) ? 700 : 400} fontFamily="var(--font-mono)">{l}</text>)}
          {rowLabels?.map((l, i) => <text key={i} x={lw - 6} y={th + i * ch + ch / 2 + 4} textAnchor="end" fontSize={11} fill={hlRows.includes(i) ? c.fg : c.muted} fontWeight={hlRows.includes(i) ? 700 : 400} fontFamily="var(--font-mono)">{l}</text>)}
          {group && groupColors && Array.from({ length: Math.ceil(C / group) }, (_, g) => (
            <rect key={g} x={lw + g * group * cw + 2} y={th - 3} width={group * cw - 4} height={2.5} rx={1} fill={groupColors[g % groupColors.length]} />
          ))}
          {m.map((row, i) => row.map((v, j) => {
            const isSel = !!selected && selected[0] === i && selected[1] === j
            const hl = hlRows.includes(i) || hlCols.includes(j) || isHl(i, j)
            const mk = masked?.[i]?.[j] || !Number.isFinite(v)
            const dim = dimRows.includes(i)
            const label = `${ariaLabel}: row ${rowLabels?.[i] ?? i + 1}, column ${colLabels?.[j] ?? j + 1}, ${mk ? 'masked' : fx(v, 4)}`
            return (
              <g key={`${i}.${j}`} onClick={onCell ? () => onCell(i, j) : undefined} onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)}
                style={onCell ? { cursor: 'pointer' } : undefined} tabIndex={onCell ? 0 : undefined} role={onCell ? 'button' : undefined} aria-label={onCell ? label : undefined}
                onKeyDown={onCell ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onCell(i, j) } } : undefined}
                data-testid={testid ? `${testid}-${i}-${j}` : undefined}>
                <title>{label}</title>
                <rect x={lw + j * cw + 1} y={th + i * ch + 1} width={cw - 2} height={ch - 2} rx={4} fill={mk ? `url(#hatch-${uid})` : fill(v)} opacity={dim ? 0.3 : 1}
                  stroke={isSel ? c.fg : hl ? base : mk ? roles.mask : 'transparent'} strokeWidth={isSel ? 2.4 : hl ? 1.6 : 0.8} style={{ transition: 'fill .35s' }} />
                <text x={lw + j * cw + cw / 2} y={th + i * ch + ch / 2 + 4} textAnchor="middle" fontSize={cw < 44 ? 9.5 : 11} fontFamily="var(--font-mono)" pointerEvents="none" fill={mk ? roles.mask : ink(v)} opacity={dim ? 0.45 : 1}>
                  {mk && Number.isFinite(v) ? (mode === 'prob' ? '0' : '⊘') : fx(v, digits)}
                </text>
              </g>
            )
          }))}
        </svg>
      </div>
      <div className="mt-0.5 h-4 font-mono text-[11px] text-faint">
        {hv && m[hv[0]]?.[hv[1]] !== undefined && <>[{rowLabels?.[hv[0]] ?? hv[0] + 1}, {colLabels?.[hv[1]] ?? hv[1] + 1}] = {Number.isFinite(m[hv[0]][hv[1]]) ? m[hv[0]][hv[1]].toPrecision(8) : '−∞'}{masked?.[hv[0]]?.[hv[1]] ? ' · masked' : ''}</>}
      </div>
    </div>
  )
})

/* ------------------------------------------------------------------ editable matrix */
export function EditableMatrix({ m, onSet, rowLabels, colLabels, role, testid, label, hlCols = [], disabled = false }: {
  m: Matrix; onSet: (r: number, c: number, v: number) => string | null; rowLabels?: string[]; colLabels?: string[]; role: Role; testid?: string; label: string; hlCols?: number[]; disabled?: boolean
}) {
  const roles = useRoles()
  const [err, setErr] = useState<string | null>(null)
  const commit = (r: number, c: number, raw: string, el: HTMLInputElement) => {
    const v = Number(raw.replace(',', '.').replace('−', '-'))
    const e = raw.trim() === '' ? 'Enter a number.' : onSet(r, c, v)
    setErr(e ? `${label} [${r + 1}, ${c + 1}]: ${e}` : null)
    if (e) el.value = String(m[r][c])
  }
  return (
    <div className="min-w-0">
      <div className="overflow-x-auto scroll-thin">
        <table className="border-separate border-spacing-[3px] font-mono text-[12px]" aria-label={`${label} (editable)`} data-testid={testid}>
          {colLabels && <thead><tr><th />{colLabels.map((l, j) => <th key={j} className={`px-1 text-[10.5px] font-normal ${hlCols.includes(j) ? 'text-fg' : 'text-faint'}`}>{l}</th>)}</tr></thead>}
          <tbody>
            {m.map((row, r) => (
              <tr key={r}>
                {rowLabels && <th className="pr-1 text-right text-[10.5px] font-normal text-faint">{rowLabels[r]}</th>}
                {row.map((v, c) => (
                  <td key={c}>
                    <input key={`${r}.${c}.${v}`} type="text" inputMode="decimal" defaultValue={String(v)} disabled={disabled} aria-label={`${label} row ${r + 1} column ${c + 1}`}
                      className="w-[54px] rounded border px-1 py-0.5 text-right" style={{ borderColor: hlCols.includes(c) ? roles[role] : undefined }}
                      onBlur={(e) => { if (e.target.value !== String(v)) commit(r, c, e.target.value, e.target) }}
                      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
                      data-testid={testid ? `${testid}-${r}-${c}` : undefined} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {err && <p className="mt-1 text-[11.5px] text-rose" role="alert">{err}</p>}
    </div>
  )
}

/** a small segmented control */
export function Seg<T extends string | number>({ value, options, onChange, label }: { value: T; options: { v: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} type="button" role="radio" aria-checked={value === o.v} title={o.title} onClick={() => onChange(o.v)}
          className={`rounded-md px-2 py-1 text-[12px] ${value === o.v ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{o.label}</button>
      ))}
    </div>
  )
}

/** head picker shared by the stage views */
export function HeadPicker() {
  const r = useRoles()
  const H = useAttn((s) => s.H), h = useAttn((s) => s.sel.h), select = useAttn((s) => s.select), headsOn = useAttn((s) => s.headsOn)
  if (H === 1) return <span className="chip text-[11.5px] text-muted">1 head · d_k = 4</span>
  return (
    <div className="flex flex-wrap items-center gap-1" role="radiogroup" aria-label="Attention head">
      {Array.from({ length: H }, (_, k) => (
        <button key={k} type="button" role="radio" aria-checked={h === k} onClick={() => select({ h: k, cell: null })} className={`btn !h-7 !py-0 text-[12px] ${h === k ? 'active' : ''}`} style={h === k ? { color: headColors(r)[k % 4], borderColor: headColors(r)[k % 4] } : undefined}>
          head {k + 1}{headsOn[k] ? '' : ' (off)'}
        </button>
      ))}
    </div>
  )
}
/** head colours are deliberately outside the role palette (pink, lime, sand, brown) so they never suggest Q/K/V meanings */
export function headColors(r: Record<Role, string>): string[] {
  return r.x === '#45e1f2' ? ['#ff6fb3', '#c8e35a', '#e9dcc0', '#c79a6b'] : ['#c22d79', '#5f7f00', '#6b6455', '#8a5a2b']
}
