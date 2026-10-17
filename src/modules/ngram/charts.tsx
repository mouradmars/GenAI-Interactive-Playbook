/** Small SVG charts for the n-gram lab, drawn to scale from computed values. */
import { motion } from 'framer-motion'
import { useColors } from '../../sim/appState'
import { STAGES, type StageId } from './lesson'

/** A line chart of one or more series over a shared x grid, with a marker at the current x. */
export function CurveChart({ xs, series, current, xLabel, logX = false, height = 190 }: {
  xs: number[]; series: { name: string; ys: number[]; color: string }[]; current: number; xLabel: string; logX?: boolean; height?: number
}) {
  const c = useColors()
  const W = 420, H = height, L = 46, R = 12, T = 22, B = 34
  const fx = (x: number) => (logX ? Math.log10(x) : x)
  const x0 = fx(xs[0]), x1 = fx(xs[xs.length - 1])
  const finite = series.flatMap((s) => s.ys).filter(Number.isFinite)
  const yMax = Math.max(...finite) * 1.05, yMin = Math.max(0, Math.min(...finite) * 0.95)
  const px = (x: number) => L + ((fx(x) - x0) / (x1 - x0 || 1)) * (W - L - R)
  const py = (y: number) => T + (1 - (y - yMin) / (yMax - yMin || 1)) * (H - T - B)
  const ticksY = Array.from({ length: 4 }, (_, i) => yMin + ((yMax - yMin) * i) / 3)
  const ticksX = logX ? xs.filter((x) => [0.01, 0.1, 1, 10].some((t) => Math.abs(x - t) < 1e-9)) : [xs[0], xs[Math.floor(xs.length / 2)], xs[xs.length - 1]]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`perplexity versus ${xLabel}`}>
      {ticksY.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={py(t)} y2={py(t)} stroke={c.line} />
          <text x={L - 6} y={py(t) + 3} fontSize={10} textAnchor="end" fill={c.muted} fontFamily="JetBrains Mono, monospace">{t.toFixed(1)}</text>
        </g>
      ))}
      {ticksX.map((t) => <text key={t} x={px(t)} y={H - B + 14} fontSize={10} textAnchor="middle" fill={c.muted} fontFamily="JetBrains Mono, monospace">{t}</text>)}
      <text x={(L + W - R) / 2} y={H - 4} fontSize={10.5} textAnchor="middle" fill={c.faint} fontFamily="JetBrains Mono, monospace">{xLabel}</text>
      <text x={L} y={11} fontSize={10.5} fill={c.faint} fontFamily="JetBrains Mono, monospace">perplexity</text>
      {series.map((s) => {
        let d = '', on = false
        s.ys.forEach((y, i) => {
          if (!Number.isFinite(y)) { on = false; return }
          d += `${on ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${py(y).toFixed(1)} `
          on = true
        })
        return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={2} />
      })}
      <line x1={px(current)} x2={px(current)} y1={T} y2={H - B} stroke={c.amber} strokeDasharray="3 3" />
      {series.map((s) => {
        const i = xs.reduce((b, x, k) => (Math.abs(x - current) < Math.abs(xs[b] - current) ? k : b), 0)
        return Number.isFinite(s.ys[i]) ? <circle key={s.name} cx={px(xs[i])} cy={py(s.ys[i])} r={4} fill={s.color} stroke={c.bg} strokeWidth={1.5} /> : null
      })}
      <g transform={`translate(${W - R - 150}, ${T + 4})`}>
        {series.map((s, i) => (
          <g key={s.name} transform={`translate(0, ${i * 14})`}>
            <rect width={10} height={3} y={4} fill={s.color} />
            <text x={14} y={9} fontSize={10} fill={c.muted} fontFamily="JetBrains Mono, monospace">{s.name}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

/** Animated pipeline of the guided stages. Click a node to jump. */
export function Pipeline({ index, onSeek }: { index: number; onSeek: (i: number) => void }) {
  const c = useColors()
  const n = STAGES.length
  const W = 1100, step = (W - 80) / (n - 1), y = 44
  const x = (i: number) => 40 + i * step
  const groups: { name: string; a: number; b: number }[] = []
  STAGES.forEach((s, i) => { const g = groups[groups.length - 1]; if (g?.name === s.group) g.b = i; else groups.push({ name: s.group, a: i, b: i }) })
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg viewBox={`0 0 ${W} 92`} width="100%" style={{ minWidth: 640 }} role="img" aria-label="n-gram pipeline" data-snapshot="skip">
        {groups.map((g) => (
          <g key={g.name}>
            <line x1={x(g.a) - 14} x2={x(g.b) + 14} y1={12} y2={12} stroke={c.lineStrong} />
            <text x={(x(g.a) + x(g.b)) / 2} y={9} fontSize={10} textAnchor="middle" fill={c.faint} fontFamily="JetBrains Mono, monospace" letterSpacing="1.2">{g.name.toUpperCase()}</text>
          </g>
        ))}
        <line x1={x(0)} x2={x(n - 1)} y1={y} y2={y} stroke={c.line} strokeWidth={3} />
        <motion.line x1={x(0)} y1={y} y2={y} stroke={c.cyan} strokeWidth={3} animate={{ x2: x(index) }} transition={{ duration: 0.5 }} />
        {/* data particles flowing into the current stage */}
        {index > 0 && [0, 1, 2].map((k) => (
          <motion.circle key={`${index}-${k}`} r={3} cy={y} fill={c.amber}
            initial={{ cx: x(Math.max(0, index - 1)), opacity: 0 }} animate={{ cx: x(index), opacity: [0, 1, 0] }}
            transition={{ duration: 1.2, delay: k * 0.35, repeat: Infinity, repeatDelay: 0.6 }} />
        ))}
        {STAGES.map((s, i) => (
          <g key={s.id} onClick={() => onSeek(i)} style={{ cursor: 'pointer' }} role="button" aria-label={`Stage ${i + 1}: ${s.title}`}>
            <circle cx={x(i)} cy={y} r={i === index ? 12 : 8} fill={i <= index ? (i === index ? c.cyan : c.teal) : c.panel2} stroke={i === index ? c.cyan : c.lineStrong} strokeWidth={1.5} />
            {i === index && <circle cx={x(i)} cy={y} r={18} fill="none" stroke={c.cyan} strokeOpacity={0.35} />}
            <text x={x(i)} y={y + 4} fontSize={10} textAnchor="middle" fill={i <= index ? c.bg : c.muted} fontFamily="JetBrains Mono, monospace">{i + 1}</text>
            <text x={x(i)} y={y + 34} fontSize={11} textAnchor="middle" fill={i === index ? c.cyan : c.muted} fontFamily="JetBrains Mono, monospace">{s.short}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

export const stageIndex = (id: StageId) => STAGES.findIndex((s) => s.id === id)
