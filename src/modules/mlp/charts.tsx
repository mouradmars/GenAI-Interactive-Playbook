/** Small SVG charts used by the loss lab and the training panel. Data always comes from the engine. */
import { useId } from 'react'
import { useColors } from '../../sim/appState'

export interface Series { points: number[][]; color: string; label?: string; dashed?: boolean; width?: number }
export interface Marker { x: number; y: number; color: string; label?: string; r?: number }

export function LineChart({ series, markers = [], lines = [], xDomain, yDomain, xLabel, yLabel, height = 200, logY = false, ariaLabel, onClickX, legendLeft = false }: {
  series: Series[]; markers?: Marker[]; lines?: { x1: number; y1: number; x2: number; y2: number; color: string; dashed?: boolean; arrow?: boolean }[]
  xDomain?: [number, number]; yDomain?: [number, number]; xLabel?: string; yLabel?: string; height?: number; logY?: boolean; ariaLabel: string; onClickX?: (x: number) => void; legendLeft?: boolean
}) {
  const c = useColors()
  const uid = useId().replace(/:/g, '')
  const W = 460, H = height, ml = 46, mr = 12, mt = yLabel ? 20 : 10, mb = 30
  const ty = (v: number) => (logY ? Math.log10(Math.max(1e-12, v)) : v)
  const all = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p[1]))
  const xd = xDomain ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))]
  let yd = yDomain ?? [Math.min(...all.map((p) => ty(p[1]))), Math.max(...all.map((p) => ty(p[1])))]
  if (yDomain && logY) yd = [ty(yDomain[0]), ty(yDomain[1])]
  if (!Number.isFinite(yd[0]) || !Number.isFinite(yd[1])) yd = [0, 1]
  if (yd[1] - yd[0] < 1e-9) yd = [yd[0] - 0.5, yd[1] + 0.5]
  const sx = (x: number) => ml + ((x - xd[0]) / (xd[1] - xd[0] || 1)) * (W - ml - mr)
  const sy = (v: number) => mt + (1 - (ty(v) - yd[0]) / (yd[1] - yd[0])) * (H - mt - mb)
  const clampY = (y: number) => Math.max(mt - 4, Math.min(H - mb + 4, y))
  const ticks = (a: number, b: number, n = 4) => Array.from({ length: n + 1 }, (_, k) => a + ((b - a) * k) / n)
  const fmt = (v: number) => (Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={ariaLabel}
      onClick={onClickX ? (e) => { const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W; onClickX(xd[0] + ((px - ml) / (W - ml - mr)) * (xd[1] - xd[0])) } : undefined}
      style={onClickX ? { cursor: 'crosshair' } : undefined}>
      <defs><marker id={`lcA${uid}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c.teal} /></marker></defs>
      {ticks(xd[0], xd[1]).map((x, k) => <g key={`x${k}`}><line x1={sx(x)} x2={sx(x)} y1={mt} y2={H - mb} stroke={c.line} strokeDasharray="2 3" /><text x={sx(x)} y={H - mb + 13} fontSize={10} textAnchor="middle" fill={c.faint} fontFamily="var(--font-mono)">{fmt(x)}</text></g>)}
      {ticks(yd[0], yd[1]).map((y, k) => <g key={`y${k}`}><line x1={ml} x2={W - mr} y1={mt + (1 - k / 4) * (H - mt - mb)} y2={mt + (1 - k / 4) * (H - mt - mb)} stroke={c.line} strokeDasharray="2 3" /><text x={ml - 4} y={mt + (1 - k / 4) * (H - mt - mb) + 3} fontSize={10} textAnchor="end" fill={c.faint} fontFamily="var(--font-mono)">{logY ? `1e${y.toFixed(1)}` : fmt(y)}</text></g>)}
      {xLabel && <text x={W - mr} y={H - 3} fontSize={10.5} textAnchor="end" fill={c.muted}>{xLabel}</text>}
      {yLabel && <text x={4} y={2} fontSize={10.5} fill={c.muted} dominantBaseline="hanging">{yLabel}</text>}
      <clipPath id={`lcC${uid}`}><rect x={ml} y={mt - 2} width={W - ml - mr} height={H - mt - mb + 4} /></clipPath>
      <g clipPath={`url(#lcC${uid})`}>
        {series.map((s, k) => {
          const pts = s.points.filter((p) => Number.isFinite(p[1]))
          return <path key={k} d={pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${clampY(sy(p[1])).toFixed(1)}`).join('')} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? '5 4' : undefined} />
        })}
        {lines.map((l, k) => <line key={k} x1={sx(l.x1)} y1={clampY(sy(l.y1))} x2={sx(l.x2)} y2={clampY(sy(l.y2))} stroke={l.color} strokeWidth={1.6} strokeDasharray={l.dashed ? '4 3' : undefined} markerEnd={l.arrow ? `url(#lcA${uid})` : undefined} />)}
      </g>
      {markers.map((m, k) => <g key={k}><circle cx={sx(m.x)} cy={clampY(sy(m.y))} r={m.r ?? 4.5} fill={m.color} stroke={c.bg} strokeWidth={1.5} />{m.label && <text x={sx(m.x) > W - 80 ? sx(m.x) - 7 : sx(m.x) + 7} textAnchor={sx(m.x) > W - 80 ? 'end' : 'start'} y={clampY(sy(m.y)) - 7} fontSize={10.5} fill={m.color} fontFamily="var(--font-mono)">{m.label}</text>}</g>)}
      {series.some((s) => s.label) && (
        <g transform={`translate(${legendLeft ? ml + 8 : W - mr - 170}, ${mt + 6})`}>
          {series.filter((s) => s.label).map((s, k) => <g key={k} transform={`translate(0, ${k * 14})`}><line x1={0} x2={16} y1={0} y2={0} stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} /><text x={21} y={3.5} fontSize={10.5} fill={c.muted}>{s.label}</text></g>)}
        </g>
      )}
    </svg>
  )
}
