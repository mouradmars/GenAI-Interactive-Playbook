/** Controls, diagrams and one view per guided stage for module 09. Every number comes from the engine. */
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw, Shuffle } from 'lucide-react'
import { Fragment, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Card, Heatmap } from '../../components/viz'
import { useApp, useColors } from '../../sim/appState'
import { mat } from '../../engine/tensor'
import {
  ARCH_LABEL, SYMBOLS, TASK, accuracyAt, cost, hiddenOf, occlusion, profileAgreement, randomSeq, schedule, seqText, trained, vnorm,
  type Arch, type Regime,
} from '../../engine/seq'
import { f3, f4, pct, sci, useSeq } from './context'

/* ================================================================ controls */

export function SourceBadge() {
  const { source, regime } = useSeq()
  return source === 'trained'
    ? <span className="sim-badge" title="Weights trained by training/train_seq.py">trained models · {regime === 'mixed' ? 'mixed lengths 2–20' : 'long-only 15–25'}</span>
    : <span className="sim-badge toy" title="Seeded random weights, not trained">educational simulation · seeded random weights</span>
}

export function Slider({ id, label, value, min, max, step, onChange }: { id: string; label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="grid gap-1 text-[12px] text-muted font-num" htmlFor={id}>{label}
      <input id={id} type="range" className="w-full min-w-0" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}

function Radio<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label={label}>
      {options.map(([v, l]) => <button key={v} type="button" role="radio" aria-checked={value === v} onClick={() => onChange(v)} className={`rounded-md px-2.5 py-1 text-[12px] ${value === v ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{l}</button>)}
    </div>
  )
}

export function SourceBar({ showSim = true }: { showSim?: boolean }) {
  const { source, setSource, regime, setRegime, sim, setSim, seq, setSeq, ids, seed } = useSeq()
  const [draft, setDraft] = useState(seq)
  const [clicks, setClicks] = useState(0)
  useEffect(() => setDraft(seq), [seq])
  return (
    <Card title="models and input" right={<SourceBadge />}>
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Radio label="model source" value={source} onChange={setSource} options={[['trained', 'Trained models'], ['sim', 'Random weights (simulation)']]} />
          {source === 'trained' && <Radio label="training regime" value={regime} onChange={setRegime} options={[['mixed', 'mixed lengths 2–20'], ['long', 'long-only 15–25']]} />}
        </div>
        {source === 'sim' && showSim && (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            <Slider id="sim-scale" label={`recurrent / attention scale = ${sim.scale.toFixed(2)}`} value={sim.scale} min={0.1} max={3} step={0.05} onChange={(scale) => setSim({ scale })} />
            <Slider id="sim-fb" label={`LSTM forget-gate bias = ${sim.forgetBias.toFixed(1)}`} value={sim.forgetBias} min={-3} max={5} step={0.1} onChange={(forgetBias) => setSim({ forgetBias })} />
            <Slider id="sim-h" label={`hidden width = ${sim.hidden}`} value={sim.hidden} min={2} max={16} step={2} onChange={(hidden) => setSim({ hidden })} />
            <label className="grid gap-1 text-[12px] text-muted" htmlFor="sim-seed">weight seed
              <div className="flex gap-1"><input id="sim-seed" type="number" className="h-8 w-full min-w-0 px-2 font-mono" value={sim.seed} onChange={(e) => setSim({ seed: +e.target.value || 0 })} />
                <button type="button" className="btn h-8" onClick={() => setSim({ seed: sim.seed + 1 })}>Next</button></div>
            </label>
          </div>
        )}
        <form className="flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); setSeq(draft) }}>
          <label htmlFor="seq-input" className="grid min-w-0 flex-1 basis-full gap-1 text-[12px] text-muted sm:basis-[260px]">input sequence (letters a–d, 1–30 symbols)
            <input id="seq-input" className="h-9 w-full px-2 font-mono text-[15px] tracking-[0.2em]" value={draft} spellCheck={false} autoComplete="off"
              onChange={(e) => setDraft(e.target.value.toLowerCase().replace(/[^abcd]/g, '').slice(0, 30))} onBlur={() => draft && setSeq(draft)} />
          </label>
          <button type="submit" className="btn btn-primary h-9">Run</button>
          <button type="button" className="btn h-9" onClick={() => { setSeq(seqText(randomSeq(ids.length, seed * 7919 + ids.length * 101 + clicks + 1))); setClicks(clicks + 1) }} title="Random sequence of the same length (seeded: the n-th click always gives the same sequence)"><Shuffle size={14} />Random</button>
          {[4, 10, 20].map((T) => <button key={T} type="button" className="btn h-9" onClick={() => setSeq(seqText(randomSeq(T, 1000 + T)))}>T = {T}</button>)}
        </form>
      </div>
    </Card>
  )
}

export function CursorBar() {
  const { cursor, ids } = useSeq()
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-panel/70 px-3 py-2">
      <span className="eyebrow">time step</span>
      <div className="flex items-center gap-1" role="group" aria-label="time step controls">
        <button type="button" className="btn icon-btn h-8" onClick={cursor.reset} aria-label="First time step"><RotateCcw size={14} /></button>
        <button type="button" className="btn icon-btn h-8" onClick={cursor.prev} aria-label="Previous time step"><ChevronLeft size={16} /></button>
        <button type="button" className="btn h-8 w-[76px] justify-center" onClick={cursor.toggle} aria-label={cursor.playing ? 'Pause time' : 'Play time'}>{cursor.playing ? <><Pause size={14} />Pause</> : <><Play size={14} />Play</>}</button>
        <button type="button" className="btn icon-btn h-8" onClick={cursor.next} aria-label="Next time step"><ChevronRight size={16} /></button>
      </div>
      <span className="font-mono text-[13px] font-num" aria-live="polite">t = <span className="text-amber">{cursor.t + 1}</span> / {cursor.T}</span>
      <div className="flex flex-wrap gap-1">
        {ids.map((id, t) => (
          <button key={t} type="button" onClick={() => cursor.set(t)} aria-label={`time ${t + 1}: ${SYMBOLS[id]}`}
            className={`h-7 w-7 rounded-md border font-mono text-[13px] ${t === cursor.t ? 'border-amber bg-amber/20 text-amber' : t < cursor.t ? 'border-line-strong text-fg' : 'border-line text-faint'} ${t === 0 ? 'ring-1 ring-teal/60' : ''}`}>{SYMBOLS[id]}</button>
        ))}
      </div>
      <span className="font-mono text-[11px] text-faint">teal ring = the symbol to recall</span>
    </div>
  )
}

export function Predictions({ compact = false }: { compact?: boolean }) {
  const { runs, target } = useSeq()
  const c = useColors()
  return (
    <div className={`grid gap-3 ${compact ? '' : 'md:grid-cols-3'}`}>
      {(['rnn', 'lstm', 'attn'] as Arch[]).map((a) => {
        const r = runs[a]
        return (
          <div key={a} className="rounded-lg border border-line bg-panel-2 p-2">
            <div className="flex items-baseline justify-between"><span className="text-[13px]">{ARCH_LABEL[a]}</span>
              <span className={`font-mono text-[12px] ${r.pred === target ? 'text-ok' : 'text-rose'}`}>{r.pred === target ? '✓' : '✗'} says “{SYMBOLS[r.pred]}”</span></div>
            <div className="mt-1 grid gap-[2px]">
              {r.trace.probs.map((p, k) => (
                <div key={k} className="grid grid-cols-[18px_1fr_52px] items-center gap-1 font-mono text-[11.5px]">
                  <span className={k === target ? 'text-teal' : 'text-muted'}>{SYMBOLS[k]}</span>
                  <div className="h-2.5 rounded-sm" style={{ background: c.panel }}><div className="h-full rounded-sm transition-[width] duration-500" style={{ width: `${p * 100}%`, background: k === target ? c.teal : c.violet }} /></div>
                  <span className="text-right text-muted font-num">{pct(p)}</span>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ================================================================ charts */

/** Units × time heatmap with columns after the cursor greyed out (not yet computed at time t). */
export function StateHeatmap({ rows, t, label, maxAbs = 1, mode = 'diverging', unit, onUnit }: { rows: number[][]; t: number; label: string; maxAbs?: number; mode?: 'diverging' | 'gate'; unit?: number; onUnit?: (u: number) => void }) {
  const c = useColors()
  const T = rows.length, H = rows[0]?.length ?? 0
  const cw = Math.max(10, Math.min(28, 560 / T)), ch = Math.max(9, Math.min(16, 150 / H))
  const col = (v: number) => {
    if (mode === 'gate') return `color-mix(in oklab, ${c.cyan} ${Math.round(v * 100)}%, ${c.panel2})`
    const a = Math.min(1, Math.abs(v) / maxAbs)
    return `color-mix(in oklab, ${v >= 0 ? c.amber : c.violet} ${Math.round(Math.sqrt(a) * 100)}%, ${c.panel2})`
  }
  return (
    <div className="min-w-0 overflow-x-auto scroll-thin">
      <div className="mb-1 font-mono text-[11px] text-muted">{label}</div>
      <svg viewBox={`0 0 ${T * cw + 34} ${H * ch + 18}`} width="100%" style={{ maxWidth: T * cw + 34, display: 'block' }} role="img" aria-label={label}>
        {Array.from({ length: H }, (_, j) => <text key={j} x={28} y={j * ch + ch * 0.75} fontSize={8.5} textAnchor="end" fill={j === unit ? c.amber : c.faint} style={{ cursor: onUnit ? 'pointer' : 'default' }} onClick={() => onUnit?.(j)}>u{j + 1}</text>)}
        {rows.map((r, s) => r.map((v, j) => (
          <rect key={`${s}-${j}`} x={32 + s * cw} y={j * ch} width={cw - 1} height={ch - 1} rx={1.5} fill={s > t ? c.bg : col(v)} stroke={s > t ? c.line : 'none'} strokeDasharray={s > t ? '2 2' : undefined}
            onClick={() => onUnit?.(j)} style={{ cursor: onUnit ? 'pointer' : 'default' }}><title>{`t=${s + 1}, unit ${j + 1}: ${v.toFixed(4)}`}</title></rect>
        )))}
        {unit !== undefined && <rect x={32} y={unit * ch - 0.5} width={T * cw} height={ch} fill="none" stroke={c.amber} strokeWidth={1} rx={2} />}
        <rect x={32 + t * cw - 0.5} y={-0.5} width={cw} height={H * ch} fill="none" stroke={c.fg} strokeWidth={1.2} rx={2} />
        {rows.map((_, s) => (T <= 20 || s % 2 === 0) && <text key={s} x={32 + s * cw + cw / 2} y={H * ch + 12} fontSize={9} textAnchor="middle" fill={s === t ? c.amber : c.faint}>{s + 1}</text>)}
      </svg>
    </div>
  )
}

export function LogChart({ series, xLabel, height = 220, yLabel, marker }: { series: { name: string; ys: number[]; color: string; dashed?: boolean }[]; xLabel: string; height?: number; yLabel: string; marker?: number }) {
  const c = useColors()
  const W = 560, H = height, L = 52, Rm = 14, Tm = 14, B = 34
  const all = series.flatMap((s) => s.ys).filter((v) => v > 0 && Number.isFinite(v))
  const lo = Math.floor(Math.log10(Math.min(...all, 1))), hi = Math.ceil(Math.log10(Math.max(...all, 1e-9)))
  const n = Math.max(...series.map((s) => s.ys.length))
  const x = (i: number) => L + (i / Math.max(1, n - 1)) * (W - L - Rm)
  const y = (v: number) => Tm + (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / Math.max(1, hi - lo)) * (H - Tm - B)
  const ticks = Array.from({ length: hi - lo + 1 }, (_, k) => lo + k).filter((_, k, a) => a.length <= 9 || k % Math.ceil(a.length / 8) === 0)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 680, display: 'block' }} role="img" aria-label={`${yLabel} by ${xLabel}, log scale`}>
      {ticks.map((e) => <g key={e}><line x1={L} x2={W - Rm} y1={y(10 ** e)} y2={y(10 ** e)} stroke={c.line} /><text x={L - 6} y={y(10 ** e) + 3} fontSize={9.5} textAnchor="end" fill={c.faint}>1e{e}</text></g>)}
      {Array.from({ length: n }, (_, i) => (n <= 20 || i % 2 === 0) && <text key={i} x={x(i)} y={H - 18} fontSize={9.5} textAnchor="middle" fill={i === marker ? c.amber : c.faint}>{i + 1}</text>)}
      <text x={(L + W - Rm) / 2} y={H - 3} fontSize={10.5} textAnchor="middle" fill={c.muted}>{xLabel}</text>
      <text x={12} y={(Tm + H - B) / 2} fontSize={10.5} textAnchor="middle" fill={c.muted} transform={`rotate(-90 12 ${(Tm + H - B) / 2})`}>{yLabel}</text>
      {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={Tm} y2={H - B} stroke={c.amber} strokeDasharray="3 3" />}
      {series.map((s) => (
        <g key={s.name}>
          <path d={s.ys.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dashed ? '5 4' : undefined} />
          {s.ys.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.8} fill={s.color}><title>{`${s.name}, position ${i + 1}: ${v.toExponential(3)}`}</title></circle>)}
        </g>
      ))}
      {series.map((s, k) => <text key={s.name} x={L + 8 + (k % 3) * 165} y={Tm + 10 + Math.floor(k / 3) * 14} fontSize={11} fill={s.color}>● {s.name}</text>)}
    </svg>
  )
}

export function LinChart({ xs, series, xLabel, yMax = 1, height = 200, yFmt = (v: number) => v.toFixed(2), hline, marker }: { xs: number[]; series: { name: string; ys: number[]; color: string; dashed?: boolean }[]; xLabel: string; yMax?: number; height?: number; yFmt?: (v: number) => string; hline?: { y: number; label: string }; marker?: number }) {
  const c = useColors()
  const W = 560, H = height, L = 44, Rm = 14, Tm = 14, B = 32
  const all = series.flatMap((s) => s.ys)
  const ymin = Math.min(0, ...all), ymax = Math.max(yMax, ...all)
  const x = (v: number) => L + ((v - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0])) * (W - L - Rm)
  const y = (v: number) => Tm + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - Tm - B)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 680, display: 'block' }} role="img" aria-label={`chart by ${xLabel}`}>
      {[ymin, (ymin + ymax) / 2, ymax].map((t) => <g key={t}><line x1={L} x2={W - Rm} y1={y(t)} y2={y(t)} stroke={c.line} /><text x={L - 5} y={y(t) + 3} fontSize={9.5} textAnchor="end" fill={c.faint}>{yFmt(t)}</text></g>)}
      {xs.filter((_, i) => xs.length <= 16 || i % Math.ceil(xs.length / 12) === 0).map((v) => <text key={v} x={x(v)} y={H - 16} fontSize={9.5} textAnchor="middle" fill={c.faint}>{v}</text>)}
      <text x={(L + W - Rm) / 2} y={H - 2} fontSize={10.5} textAnchor="middle" fill={c.muted}>{xLabel}</text>
      {hline && <g><line x1={L} x2={W - Rm} y1={y(hline.y)} y2={y(hline.y)} stroke={c.rose} strokeDasharray="4 4" /><text x={W - Rm - 4} y={y(hline.y) - 4} fontSize={10} textAnchor="end" fill={c.rose}>{hline.label}</text></g>}
      {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={Tm} y2={H - B} stroke={c.amber} strokeDasharray="3 3" />}
      {series.map((s, k) => <path key={k} d={s.ys.map((v, i) => `${i ? 'L' : 'M'}${x(xs[i])},${y(v)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={s.name ? 2.2 : 1.2} strokeDasharray={s.dashed ? '5 4' : undefined} />)}
      {series.filter((s) => s.name).map((s, k) => <text key={s.name} x={L + 8 + k * 140} y={Tm + 10} fontSize={11} fill={s.color}>● {s.name}</text>)}
    </svg>
  )
}

const archColor = (c: ReturnType<typeof useColors>, a: Arch) => (a === 'rnn' ? c.rose : a === 'lstm' ? c.cyan : c.amber)

/* ================================================================ stages */

export function GuidedStage() {
  const { stage } = useSeq()
  const V: Record<string, () => React.JSX.Element> = {
    task: TaskView, rnn: RNNView, lstm: LSTMView, cell: CellView, attention: AttentionView, parallel: ParallelView,
    gradients: GradientView, caveat: CaveatView, lengths: LengthsView, compare: CompareView,
  }
  const View = V[stage.id]
  return (
    <AnimatePresence mode="wait">
      <motion.div key={stage.id} initial={{ opacity: 0.4, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0.4, y: -6 }} transition={{ duration: 0.2 }} className="grid min-w-0 gap-3 [&>*]:min-w-0">
        <View />
      </motion.div>
    </AnimatePresence>
  )
}

/* ---- 1. task */
function TaskView() {
  const { ids, target } = useSeq()
  const c = useColors()
  const T = ids.length, w = 44
  return (
    <>
      <SourceBar />
      <Card title={`the task · recall the first symbol after ${T - 1} more`}>
        <div className="overflow-x-auto scroll-thin">
          <svg viewBox={`0 0 ${T * w + 120} 110`} width="100%" style={{ maxWidth: T * w + 120, display: 'block' }} role="img" aria-label="task diagram">
            <path d={`M ${w / 2 + 10} 34 C ${w / 2 + 10} 4, ${T * w + 50} 4, ${T * w + 60} 44`} fill="none" stroke={c.teal} strokeWidth={2} strokeDasharray="5 4" />
            <text x={(T * w) / 2 + 30} y={14} fontSize={11} textAnchor="middle" fill={c.teal}>dependency distance {T - 1}</text>
            {ids.map((id, t) => (
              <g key={t}>
                <rect x={10 + t * w} y={44} width={w - 8} height={34} rx={7} fill={t === 0 ? c.teal : c.panel2} fillOpacity={t === 0 ? 0.25 : 1} stroke={t === 0 ? c.teal : c.lineStrong} />
                <text x={10 + t * w + (w - 8) / 2} y={66} fontSize={16} textAnchor="middle" fill={c.fg} fontFamily="JetBrains Mono, monospace">{SYMBOLS[id]}</text>
                <text x={10 + t * w + (w - 8) / 2} y={94} fontSize={9.5} textAnchor="middle" fill={c.faint}>{t + 1}</text>
              </g>
            ))}
            <text x={T * w + 24} y={66} fontSize={16} fill={c.muted}>→</text>
            <rect x={T * w + 44} y={44} width={50} height={34} rx={7} fill="none" stroke={c.teal} strokeWidth={2} />
            <text x={T * w + 69} y={66} fontSize={16} textAnchor="middle" fill={c.teal} fontFamily="JetBrains Mono, monospace">{SYMBOLS[target]}?</text>
          </svg>
        </div>
      </Card>
      <Card title="what each model predicts for this sequence"><Predictions /></Card>
      <Card title="the three models (same width, same training budget)">
        <table className="w-full font-mono text-[12px] font-num">
          <thead><tr className="text-left text-faint"><th className="font-normal">model</th><th className="font-normal">connects time by</th><th className="font-normal">parameters</th><th className="font-normal">sequential steps for T = {T}</th><th className="font-normal">path x₁ → output</th></tr></thead>
          <tbody>{(['rnn', 'lstm', 'attn'] as Arch[]).map((a) => { const k = cost(a, T, 8); return (
            <tr key={a} className="border-b border-line/50"><td>{ARCH_LABEL[a]}</td><td className="font-sans text-muted">{a === 'rnn' ? 'one squashed state' : a === 'lstm' ? 'gated cell + state' : 'direct weighted lookup'}</td><td>{trained('mixed', a).nParams}</td><td>{k.sequentialSteps}</td><td>{k.pathLength}</td></tr>) })}</tbody>
        </table>
        <p className="mt-2 text-[12px] text-muted">Trained with {TASK.optimizer}, {TASK.steps.toLocaleString()} steps of {TASK.batch} sequences, learning rate {TASK.lr}, identical data stream for every model.</p>
      </Card>
    </>
  )
}

/* ---- 2. RNN */
function RNNView() {
  const { runs, cursor, unit, setUnit, ids } = useSeq()
  const c = useColors()
  const tr = runs.rnn.trace
  if (tr.arch !== 'rnn') return <></>
  const T = ids.length, t = cursor.t, w = 62
  return (
    <>
      <SourceBar showSim />
      <CursorBar />
      <Card title="unrolled through time · the same cell (same weights) at every step">
        <div className="overflow-x-auto scroll-thin">
          <svg viewBox={`0 0 ${T * w + 60} 150`} width="100%" style={{ minWidth: Math.min(T * w + 60, 900), display: 'block' }} role="img" aria-label="unrolled RNN">
            <defs><marker id="ar-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill={c.muted} /></marker></defs>
            <text x={6} y={66} fontSize={10} fill={c.faint}>h₀=0</text>
            {ids.map((id, s) => {
              const x = 40 + s * w, on = s <= t
              const hv = tr.h[s]
              return (
                <g key={s} opacity={on ? 1 : 0.28}>
                  <line x1={x - 22} y1={62} x2={x - 2} y2={62} stroke={s === t ? c.amber : c.muted} strokeWidth={s === t ? 2 : 1.2} markerEnd="url(#ar-r)" />
                  <rect x={x} y={40} width={40} height={44} rx={8} fill={c.panel2} stroke={s === t ? c.amber : c.lineStrong} strokeWidth={s === t ? 2 : 1} />
                  {hv.map((v, j) => <rect key={j} x={x + 4 + (j * 32) / hv.length} y={62 - Math.max(0, v) * 18} width={32 / hv.length - 1} height={Math.abs(v) * 18} fill={v >= 0 ? c.amber : c.violet} opacity={0.9} />)}
                  <line x1={x + 20} y1={118} x2={x + 20} y2={88} stroke={c.muted} markerEnd="url(#ar-r)" />
                  <text x={x + 20} y={134} fontSize={14} textAnchor="middle" fill={s === 0 ? c.teal : c.fg} fontFamily="JetBrains Mono, monospace">{SYMBOLS[id]}</text>
                  <text x={x + 20} y={32} fontSize={10} textAnchor="middle" fill={s === t ? c.amber : c.faint}>h{s + 1}</text>
                </g>
              )
            })}
            <line x1={40 + T * w - 20} y1={62} x2={40 + T * w} y2={62} stroke={c.muted} markerEnd="url(#ar-r)" opacity={t === T - 1 ? 1 : 0.28} />
            <text x={44 + T * w} y={66} fontSize={11} fill={t === T - 1 ? c.teal : c.faint}>ŷ</text>
          </svg>
        </div>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-[1.4fr_1fr]">
        <Card title="hidden state h_t · units × time · click a unit"><StateHeatmap rows={tr.h} t={t} label="h_t = tanh(…) ∈ (−1, 1)" unit={unit} onUnit={setUnit} /></Card>
        <Card title={`unit ${unit + 1} at t = ${t + 1}`}>
          <dl className="grid grid-cols-[1fr_auto] gap-y-1 font-mono text-[12.5px] font-num">
            <dt className="text-muted">W_x x_t (column of symbol “{SYMBOLS[ids[t]]}”)</dt><dd>{f4(runs.rnn.params.arch === 'rnn' ? runs.rnn.params.Wx[unit][ids[t]] : 0)}</dd>
            <dt className="text-muted">W_h h_(t−1)</dt><dd>{f4(runs.rnn.params.arch === 'rnn' ? runs.rnn.params.Wh[unit].reduce((s, wv, k) => s + wv * (t ? tr.h[t - 1][k] : 0), 0) : 0)}</dd>
            <dt className="text-muted">+ b</dt><dd>{f4(runs.rnn.params.arch === 'rnn' ? runs.rnn.params.b[unit] : 0)}</dd>
            <dt className="text-muted">= pre-activation a</dt><dd>{f4(tr.a[t][unit])}</dd>
            <dt className="text-muted">h = tanh(a)</dt><dd className="text-amber">{f4(tr.h[t][unit])}</dd>
          </dl>
          <p className="mt-2 text-[12px] text-muted">Everything the network knows about earlier symbols must fit in these {tr.h[0].length} numbers.</p>
        </Card>
      </div>
    </>
  )
}

/* ---- 3. LSTM gates */
export function LSTMCellDiagram({ t, unit }: { t: number; unit: number }) {
  const { runs } = useSeq()
  const c = useColors()
  const tr = runs.lstm.trace
  if (tr.arch !== 'lstm') return null
  const H = tr.h[0].length
  const u = Math.min(unit, H - 1)
  const cp = t ? tr.c[t - 1][u] : 0, hp = t ? tr.h[t - 1][u] : 0
  const v = { i: tr.i[t][u], f: tr.f[t][u], g: tr.g[t][u], o: tr.o[t][u], c: tr.c[t][u], tc: tr.tc[t][u], h: tr.h[t][u] }
  const gate = (x: number, y: number, label: string, val: number, kind: 'σ' | 'tanh') => (
    <g>
      <rect x={x - 26} y={y - 18} width={52} height={36} rx={7} fill={`color-mix(in oklab, ${kind === 'σ' ? c.cyan : c.violet} ${Math.round((kind === 'σ' ? val : Math.abs(val)) * 60)}%, ${c.panel2})`} stroke={c.lineStrong} />
      <text x={x} y={y - 3} fontSize={10.5} textAnchor="middle" fill={c.fg}>{label}</text>
      <text x={x} y={y + 11} fontSize={11} textAnchor="middle" fill={c.fg} fontFamily="JetBrains Mono, monospace">{val.toFixed(2)}</text>
    </g>
  )
  const op = (x: number, y: number, s: string) => <g><circle cx={x} cy={y} r={11} fill={c.panel} stroke={c.amber} strokeWidth={1.5} /><text x={x} y={y + 4} fontSize={12} textAnchor="middle" fill={c.amber}>{s}</text></g>
  return (
    <svg viewBox="0 0 560 300" width="100%" style={{ maxWidth: 640, display: 'block' }} role="img" aria-label={`LSTM cell, unit ${u + 1}, time ${t + 1}`}>
      <defs><marker id="ar-l" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill={c.muted} /></marker></defs>
      <rect x={70} y={22} width={420} height={250} rx={18} fill="none" stroke={c.line} />
      {/* cell-state highway */}
      <line x1={20} y1={50} x2={540} y2={50} stroke={c.teal} strokeWidth={4} strokeOpacity={0.55} />
      <text x={20} y={40} fontSize={11} fill={c.teal}>c(t−1) = {cp.toFixed(3)}</text>
      <text x={540} y={40} fontSize={11} textAnchor="end" fill={c.teal}>c(t) = {v.c.toFixed(3)}</text>
      {op(170, 50, '×')}{op(300, 50, '+')}
      {/* hidden path */}
      <line x1={20} y1={250} x2={110} y2={250} stroke={c.muted} markerEnd="url(#ar-l)" />
      <text x={20} y={268} fontSize={11} fill={c.muted}>h(t−1) = {hp.toFixed(3)}</text>
      <line x1={110} y1={250} x2={410} y2={250} stroke={c.line} strokeWidth={1.5} />
      {[170, 240, 300, 380].map((x) => <line key={x} x1={x} y1={250} x2={x} y2={186} stroke={c.line} strokeWidth={1.5} />)}
      {gate(170, 168, 'forget f', v.f, 'σ')}{gate(240, 168, 'input i', v.i, 'σ')}{gate(300, 128, 'cand. g', v.g, 'tanh')}{gate(380, 168, 'output o', v.o, 'σ')}
      <line x1={170} y1={150} x2={170} y2={61} stroke={c.cyan} markerEnd="url(#ar-l)" />
      <line x1={240} y1={150} x2={240} y2={94} stroke={c.cyan} /><line x1={300} y1={110} x2={300} y2={94} stroke={c.violet} />
      {op(270, 94, '×')}
      <line x1={270} y1={83} x2={295} y2={58} stroke={c.amber} markerEnd="url(#ar-l)" />
      {/* output */}
      <line x1={430} y1={50} x2={430} y2={100} stroke={c.teal} markerEnd="url(#ar-l)" />
      <rect x={404} y={100} width={52} height={28} rx={7} fill={c.panel2} stroke={c.lineStrong} /><text x={430} y={118} fontSize={10.5} textAnchor="middle" fill={c.fg}>tanh {v.tc.toFixed(2)}</text>
      {op(430, 160, '×')}
      <line x1={380} y1={150} x2={418} y2={160} stroke={c.cyan} />
      <line x1={430} y1={128} x2={430} y2={148} stroke={c.muted} />
      <line x1={441} y1={160} x2={540} y2={160} stroke={c.amber} strokeWidth={2} markerEnd="url(#ar-l)" />
      <text x={540} y={178} fontSize={11} textAnchor="end" fill={c.amber}>h(t) = {v.h.toFixed(3)}</text>
      <text x={110} y={290} fontSize={10.5} fill={c.muted}>x(t) joins h(t−1) as input to all four gates</text>
      <text x={300} y={78} fontSize={10} textAnchor="middle" fill={c.amber}>i·g = {(v.i * v.g).toFixed(3)}</text>
      <text x={170} y={80} fontSize={10} textAnchor="middle" fill={c.cyan}>f·c = {(v.f * cp).toFixed(3)}</text>
    </svg>
  )
}

function LSTMView() {
  const { runs, cursor, unit, setUnit } = useSeq()
  const tr = runs.lstm.trace
  if (tr.arch !== 'lstm') return <></>
  const t = cursor.t, u = Math.min(unit, tr.h[0].length - 1)
  return (
    <>
      <SourceBar />
      <CursorBar />
      <div className="grid gap-3 @5xl:grid-cols-[1.1fr_1fr]">
        <Card title={`one LSTM cell · unit ${u + 1} at t = ${t + 1} · values are exact`}><LSTMCellDiagram t={t} unit={u} /></Card>
        <Card title="every unit, every step · click a unit">
          <div className="grid gap-2">
            <StateHeatmap rows={tr.f} t={t} label="forget gate f (0 = erase, 1 = keep)" mode="gate" unit={u} onUnit={setUnit} />
            <StateHeatmap rows={tr.i} t={t} label="input gate i (0 = ignore, 1 = write)" mode="gate" unit={u} onUnit={setUnit} />
            <StateHeatmap rows={tr.o} t={t} label="output gate o" mode="gate" unit={u} onUnit={setUnit} />
          </div>
        </Card>
      </div>
      <Card title="the update for this unit, term by term">
        <p className="font-mono text-[13px] leading-relaxed font-num">
          c<sub>{t + 1}</sub> = f·c<sub>{t}</sub> + i·g = {f3(tr.f[t][u])}·{f3(t ? tr.c[t - 1][u] : 0)} + {f3(tr.i[t][u])}·{f3(tr.g[t][u])} = <span className="text-teal">{f4(tr.c[t][u])}</span>
          <br />h<sub>{t + 1}</sub> = o·tanh(c) = {f3(tr.o[t][u])}·{f3(tr.tc[t][u])} = <span className="text-amber">{f4(tr.h[t][u])}</span>
        </p>
      </Card>
    </>
  )
}

/* ---- 4. cell highway */
function CellView() {
  const { runs, cursor, unit, setUnit, ids } = useSeq()
  const c = useColors()
  const tr = runs.lstm.trace
  if (tr.arch !== 'lstm') return <></>
  const H = tr.c[0].length, u = Math.min(unit, H - 1), T = ids.length
  const survive = (j: number) => tr.f.slice(1).reduce((p, f) => p * f[j], 1)
  const ranked = Array.from({ length: H }, (_, j) => j).sort((a, b) => survive(b) - survive(a))
  return (
    <>
      <SourceBar />
      <CursorBar />
      <div className="grid gap-3 @5xl:grid-cols-[1.3fr_1fr]">
        <Card title={`cell state c_t over time · unit ${u + 1} highlighted`}>
          <LinChart xs={ids.map((_, s) => s + 1)} xLabel="time step" yMax={Math.max(1, ...tr.c.flat().map(Math.abs))} yFmt={(v) => v.toFixed(1)} marker={cursor.t + 1}
            series={[...Array.from({ length: H }, (_, j) => j).filter((j) => j !== u).map((j) => ({ name: '', ys: tr.c.map((r) => r[j]), color: c.lineStrong })), { name: `unit ${u + 1}`, ys: tr.c.map((r) => r[u]), color: c.teal }]} />
          <StateHeatmap rows={tr.c} t={cursor.t} label="cell state c_t (unbounded)" maxAbs={Math.max(1, ...tr.c.flat().map(Math.abs))} unit={u} onUnit={setUnit} />
        </Card>
        <Card title="how much of c₁ survives the direct path to c_T?">
          <table className="w-full font-mono text-[12px] font-num">
            <thead><tr className="text-left text-faint"><th className="font-normal">unit</th><th className="font-normal">∏ f_t (t = 2…{T})</th><th className="font-normal">c₁</th><th className="font-normal">c_T</th></tr></thead>
            <tbody>{ranked.map((j) => (
              <tr key={j} className={`cursor-pointer border-b border-line/50 ${j === u ? 'text-amber' : ''}`} onClick={() => setUnit(j)}>
                <td>u{j + 1}</td><td>{sci(survive(j))}</td><td>{f3(tr.c[0][j])}</td><td>{f3(tr.c[T - 1][j])}</td>
              </tr>
            ))}</tbody>
          </table>
          <p className="mt-2 text-[12px] text-muted">A product near 1 means that unit can carry the first symbol to the end untouched. In an RNN the corresponding factor includes W_h and tanh′ at every step.</p>
        </Card>
      </div>
    </>
  )
}

/* ---- 5. attention */
function AttentionView() {
  const { runs, cursor, ids } = useSeq()
  const c = useColors()
  const tr = runs.attn.trace
  const [cell, setCell] = useState<[number, number] | null>(null)
  if (tr.arch !== 'attn') return <></>
  const T = ids.length, t = cursor.t, step = 46, base = 120
  const [qi, kj] = cell && cell[1] <= cell[0] ? cell : [t, 0]
  const H = tr.q[0].length
  return (
    <>
      <SourceBar />
      <CursorBar />
      <Card title={`where does position ${t + 1} look? · arc width = attention weight${t < T - 1 ? ' · this row is computed but not used by the prediction' : ' · this row feeds the prediction'}`}>
        <div className="overflow-x-auto scroll-thin">
          <svg viewBox={`0 0 ${T * step + 20} ${base + 50}`} width="100%" style={{ minWidth: Math.min(T * step + 20, 900), display: 'block' }} role="img" aria-label="attention arcs">
            {Array.from({ length: t + 1 }, (_, s) => {
              const wgt = tr.A[t][s], x1 = 10 + s * step + step / 2, x2 = 10 + t * step + step / 2
              const h = s === t ? 28 : Math.min(base - 12, 26 + Math.abs(x2 - x1) * 0.3)
              const d = s === t ? `M ${x2 - 10} ${base} C ${x2 - 18} ${base - 40}, ${x2 + 18} ${base - 40}, ${x2 + 10} ${base}` : `M ${x1} ${base} Q ${(x1 + x2) / 2} ${base - h * 2} ${x2} ${base}`
              return <path key={s} d={d} fill="none" stroke={c.amber} strokeWidth={0.8 + wgt * 12} strokeOpacity={0.12 + wgt * 0.88} strokeLinecap="round" />
            })}
            {ids.map((id, s) => (
              <g key={s} onClick={() => cursor.set(s)} style={{ cursor: 'pointer' }}>
                <rect x={10 + s * step + 4} y={base + 4} width={step - 8} height={26} rx={6} fill={s === t ? c.amber : c.panel2} stroke={s > t ? c.line : c.lineStrong} strokeDasharray={s > t ? '3 3' : undefined} />
                <text x={10 + s * step + step / 2} y={base + 21} fontSize={13} textAnchor="middle" fill={s === t ? c.bg : s > t ? c.faint : s === 0 ? c.teal : c.fg} fontFamily="JetBrains Mono, monospace">{SYMBOLS[id]}</text>
                {s <= t && <text x={10 + s * step + step / 2} y={base + 44} fontSize={10} textAnchor="middle" fill={tr.A[t][s] > 0.2 ? c.amber : c.faint}>{(tr.A[t][s] * 100).toFixed(0)}%</text>}
              </g>
            ))}
          </svg>
        </div>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-[1.2fr_1fr]">
        <Card title="causal attention matrix A · rows = query position, columns = read position · click a cell">
          <div className="overflow-x-auto scroll-thin">
            <Heatmap m={mat(T, T, tr.A.flat())} mode="prob" mask rowLabels={ids.map((id, s) => `${s + 1}:${SYMBOLS[id]}`)} colLabels={ids.map((id, s) => `${s + 1}:${SYMBOLS[id]}`)}
              selectedRow={t} selectedCell={[qi, kj]} onCell={(i, j) => j <= i && setCell([i, j])} showValues={T <= 12} cell={T <= 12 ? 34 : 20} ariaLabel="attention matrix" />
          </div>
        </Card>
        <Card title={`score for query ${qi + 1}, key ${kj + 1}`}>
          <p className="break-words font-mono text-[12px] leading-relaxed text-muted font-num">
            q·k = {tr.q[qi].slice(0, 4).map((x, j) => `(${f3(x)})(${f3(tr.k[kj][j])})`).join(' + ')} + … ({H} terms) = {f4(tr.q[qi].reduce((s, x, j) => s + x * tr.k[kj][j], 0))}
            <br />÷ √{H} = <span className="text-fg">{f4(tr.scores[qi][kj])}</span>
            <br />softmax over positions 1…{qi + 1} → <span className="text-amber">{f4(tr.A[qi][kj])}</span>
          </p>
          <p className="mt-2 text-[12px] text-muted">Position enters only through the sinusoidal vectors added to the embeddings: e_t = W_e x_t + p_t.</p>
        </Card>
      </div>
    </>
  )
}

/* ---- 6. parallelism */
function ParallelView() {
  const { ids } = useSeq()
  const c = useColors()
  const [T, setT] = useState(Math.max(4, Math.min(16, ids.length)))
  const [H, setH] = useState(8)
  const [stageNow, setStageNow] = useState(0)
  const [play, setPlay] = useState(true)
  const maxStage = T + 1
  useEffect(() => { if (!play) return; const id = setTimeout(() => setStageNow((s) => (s + 1) % (maxStage + 1)), 520); return () => clearTimeout(id) }, [play, stageNow, maxStage])
  const archs: Arch[] = ['rnn', 'lstm', 'attn']
  const Ts = Array.from({ length: 31 }, (_, k) => k * 4 + 4)
  const cw = 26
  return (
    <>
      <div className="flex flex-wrap items-end gap-3">
        <div className="w-56"><Slider id="par-T" label={`sequence length T = ${T}`} value={T} min={2} max={32} step={1} onChange={setT} /></div>
        <div className="w-48"><Slider id="par-H" label={`width d = ${H}`} value={H} min={4} max={64} step={4} onChange={setH} /></div>
        <button type="button" className="btn" onClick={() => setPlay(!play)}>{play ? <><Pause size={14} />Pause</> : <><Play size={14} />Play</>}</button>
        <button type="button" className="btn" onClick={() => { setPlay(false); setStageNow((s) => Math.min(maxStage, s + 1)) }} aria-label="Next schedule step"><ChevronRight size={14} />Step</button>
        <button type="button" className="btn" onClick={() => { setPlay(false); setStageNow(0) }}><RotateCcw size={14} />Reset</button>
        <span className="sim-badge toy">exact operation counts · no timing measurement</span>
      </div>
      <Card title={`schedule · each column is one round of work that can run at the same time · round ${stageNow + 1}`}>
        <div className="grid gap-3 overflow-x-auto scroll-thin">
          {archs.map((a) => {
            const ops = schedule(a, T)
            const stages = Math.max(...ops.map((o) => o.stage)) + 1
            const rows = a === 'attn' ? T : 1
            return (
              <div key={a}>
                <div className="mb-1 flex items-baseline gap-3 font-mono text-[11.5px]"><span style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</span><span className="text-faint">{stages} rounds · up to {cost(a, T, H).parallelWidth} operations per round</span></div>
                <svg viewBox={`0 0 ${(maxStage + 1) * cw + 4} ${Math.min(rows, 12) * 12 + 6}`} width={(maxStage + 1) * cw + 4} height={Math.min(rows, 12) * 12 + 6} role="img" aria-label={`${ARCH_LABEL[a]} schedule`}>
                  {ops.map((o, k) => {
                    const row = a === 'attn' && o.position !== null ? Math.min(o.position, 11) : 0
                    if (a === 'attn' && o.position !== null && o.position > 11) return null
                    const done = o.stage < stageNow, now = o.stage === stageNow
                    return <rect key={k} x={2 + o.stage * cw} y={3 + row * 12} width={cw - 3} height={10} rx={2} fill={now ? archColor(c, a) : done ? `color-mix(in oklab, ${archColor(c, a)} 45%, ${c.panel2})` : c.panel2} stroke={c.line}><title>{o.label}</title></rect>
                  })}
                </svg>
              </div>
            )
          })}
          {T > 12 && <p className="text-[11.5px] text-faint">Attention rows beyond 12 are omitted from the drawing; they run in the same rounds.</p>}
        </div>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-2">
        <Card title={`cost for T = ${T}, d = ${H} (forward pass)`}>
          <table className="w-full font-mono text-[12px] font-num">
            <thead><tr className="text-left text-faint"><th className="font-normal">quantity</th>{archs.map((a) => <th key={a} className="font-normal" style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</th>)}</tr></thead>
            <tbody>
              {([['multiply-adds', 'macs'], ['dependent rounds', 'sequentialSteps'], ['max parallel width', 'parallelWidth'], ['path x₁ → output', 'pathLength'], ['activations stored', 'activationMemory']] as const).map(([l, k]) => (
                <tr key={k} className="border-b border-line/50"><td className="text-muted">{l}</td>{archs.map((a) => <td key={a}>{cost(a, T, H)[k].toLocaleString()}</td>)}</tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[12px] text-muted">Attention does more total work as T grows (the T² score matrix) but needs only 5 dependent rounds, so on parallel hardware it finishes sooner.</p>
        </Card>
        <Card title="how the costs grow with T">
          <LogChart xLabel="sequence length T (×4)" yLabel="count" height={200} series={[
            { name: 'RNN rounds', ys: Ts.map((x) => cost('rnn', x, H).sequentialSteps), color: c.rose },
            { name: 'attention rounds', ys: Ts.map((x) => cost('attn', x, H).sequentialSteps), color: c.amber },
            { name: 'attention MACs', ys: Ts.map((x) => cost('attn', x, H).macs), color: c.amber, dashed: true },
            { name: 'RNN MACs', ys: Ts.map((x) => cost('rnn', x, H).macs), color: c.rose, dashed: true },
          ]} />
        </Card>
      </div>
    </>
  )
}

/* ---- 7. gradients */
function GradientView() {
  const { runs, objective, setObjective, ids, source } = useSeq()
  const c = useColors()
  const archs: Arch[] = ['rnn', 'lstm', 'attn']
  const norms = (a: Arch) => runs[a].grads.dx.map((d) => vnorm(d))
  const ratio = (a: Arch) => { const n = norms(a); return n[n.length - 1] ? n[0] / n[n.length - 1] : 0 }
  return (
    <>
      <SourceBar />
      <div className="flex flex-wrap items-center gap-2">
        <Radio label="gradient objective" value={objective} onChange={setObjective} options={[['logit', 'sensitivity of the target logit'], ['loss', 'training loss −log p']]} />
        <span className="font-mono text-[11.5px] text-faint">exact backpropagation through time · checked against finite differences in the tests</span>
      </div>
      <div className="grid gap-3 @5xl:grid-cols-[1.4fr_1fr]">
        <Card title={`‖∂L/∂x_t‖ for each input position · L = ${objective === 'loss' ? '−log p(target)' : 'z_target'}`}>
          <LogChart xLabel="input position t (1 = the symbol to recall)" yLabel="gradient norm" series={archs.map((a) => ({ name: ARCH_LABEL[a], ys: norms(a), color: archColor(c, a) }))} />
        </Card>
        <Card title="signal reaching x₁ relative to x_T">
          <dl className="grid grid-cols-[1fr_auto] gap-y-1.5 font-mono text-[12.5px] font-num">
            {archs.map((a) => <Fragment key={a}><dt style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</dt><dd>{sci(ratio(a))}</dd></Fragment>)}
          </dl>
          <p className="mt-3 text-[12px] leading-relaxed text-muted">
            {source === 'sim' ? 'Drag the recurrent scale: below about 1 the RNN ratio collapses exponentially (vanishing), above it grows (exploding). The LSTM forget bias sets how slowly its cell path decays.'
              : objective === 'loss' ? 'Trained models that are already confident have a loss gradient near 0 everywhere: there is nothing left to learn. Switch to the logit objective to see sensitivity.'
              : 'Even in trained models the RNN’s sensitivity to x₁ is tiny: it has latched the first symbol into a saturated state. The Caveat stage shows that replacing x₁ still changes its answer.'}
          </p>
          <p className="mt-2 font-mono text-[11px] text-faint">T = {ids.length}; loss / logit value: {archs.map((a) => `${ARCH_LABEL[a].split(' ')[0]} ${f3(runs[a].grads.loss)}`).join(' · ')}</p>
        </Card>
      </div>
      <Card title="the same gradients along the unrolled chain (brightness = log gradient norm)">
        <div className="grid gap-1 overflow-x-auto scroll-thin">
          {archs.map((a) => {
            const n = norms(a), lg = n.map((v) => Math.log10(Math.max(v, 1e-30)))
            const hi = Math.max(...lg), lo = Math.min(...lg, hi - 1)
            return (
              <div key={a} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-mono text-[11.5px]" style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</span>
                {ids.map((id, s) => (
                  <span key={s} title={`position ${s + 1}: ${n[s].toExponential(2)}`} className="grid h-7 w-7 shrink-0 place-items-center rounded font-mono text-[12px]"
                    style={{ background: `color-mix(in oklab, ${archColor(c, a)} ${Math.round(((lg[s] - lo) / (hi - lo || 1)) * 85) + 5}%, ${c.panel2})`, color: c.fg }}>{SYMBOLS[id]}</span>
                ))}
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )
}

/* ---- 8. caveat */
export function CaveatView() {
  const { runs, ids, target, source } = useSeq()
  const c = useColors()
  const T = ids.length
  const occ = useMemo(() => Object.fromEntries((['rnn', 'lstm', 'attn'] as Arch[]).map((a) => [a, occlusion(runs[a].params, ids, target)])) as Record<Arch, number[]>, [runs, ids, target])
  const tr = runs.attn.trace
  if (tr.arch !== 'attn') return <></>
  const att = tr.A[T - 1]
  const grad = runs.attn.grads.dx.map((d) => vnorm(d))
  const contrib = tr.v.map((v, s) => vnorm(v.map((x) => x * att[s])))
  const agree = profileAgreement(att, occ.attn)
  const norm = (xs: number[]) => { const m = Math.max(...xs.map(Math.abs), 1e-12); return xs.map((x) => x / m) }
  const rows: { name: string; vals: number[]; color: string; raw: number[] }[] = [
    { name: 'attention weight A_{T,s}', vals: norm(att), color: c.amber, raw: att },
    { name: '‖A_{T,s} v_s‖ (weight × value)', vals: norm(contrib), color: c.violet, raw: contrib },
    { name: 'occlusion Δp (replace symbol s)', vals: norm(occ.attn), color: c.teal, raw: occ.attn },
    { name: 'gradient ‖∂z/∂x_s‖', vals: norm(grad), color: c.cyan, raw: grad },
  ]
  return (
    <>
      <SourceBar />
      <Card title={`self-attention model · four ways to ask “which input mattered?” · each row scaled to its own maximum`}
        right={<span className={`font-mono text-[12px] ${agree.sameTop && agree.pearson > 0.7 ? 'text-ok' : agree.pearson > 0.3 ? 'text-amber' : 'text-rose'}`}>attention ↔ occlusion: {agree.sameTop ? 'same top position' : `top differs (${agree.topA + 1} vs ${agree.topB + 1})`} · correlation {f3(agree.pearson)}</span>}>
        <div className="overflow-x-auto scroll-thin">
          <table className="font-mono text-[12px] font-num">
            <thead><tr><th className="pr-3 text-left font-normal text-faint">measure</th>{ids.map((id, s) => <th key={s} className={`w-12 text-center font-normal ${s === 0 ? 'text-teal' : 'text-faint'}`}>{s + 1}:{SYMBOLS[id]}</th>)}</tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td className="whitespace-nowrap py-1 pr-3 text-muted">{r.name}</td>
                  {r.vals.map((v, s) => (
                    <td key={s} className="p-0.5"><div className="grid h-8 place-items-center rounded text-[10.5px]" title={r.raw[s].toExponential(3)}
                      style={{ background: `color-mix(in oklab, ${v >= 0 ? r.color : c.rose} ${Math.round(Math.abs(v) * 85)}%, ${c.panel2})` }}>{Math.abs(r.raw[s]) < 0.005 ? '·' : r.raw[s].toFixed(2)}</div></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ul className="mt-3 grid list-disc gap-1 pl-5 text-[12.5px] leading-relaxed text-muted">
          <li>The last symbol reaches the output through the residual path (o_T + e_T) without any attention weight: compare its attention cell with its occlusion and gradient cells.</li>
          <li>{source === 'trained' ? 'In this trained model attention, occlusion and the prediction line up on the first symbol. That agreement is an empirical finding for this model and task, not a property of attention.' : 'With random weights the attention pattern and the measured influence can disagree completely. Try other seeds.'}</li>
          <li>Only the last row of the attention matrix is used; the other rows look just as “meaningful” but affect nothing.</li>
        </ul>
      </Card>
      <Card title="recurrent models: gradient versus occlusion for the same prediction">
        <div className="grid gap-3 @3xl:grid-cols-2">
          {(['rnn', 'lstm'] as Arch[]).map((a) => {
            const g = runs[a].grads.dx.map((d) => vnorm(d))
            return (
              <div key={a}>
                <div className="mb-1 font-mono text-[12px]" style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</div>
                <dl className="grid grid-cols-[1fr_auto] gap-y-1 font-mono text-[12px] font-num">
                  <dt className="text-muted">occlusion of x₁ (Δp)</dt><dd className="text-teal">{f3(occ[a][0])}</dd>
                  <dt className="text-muted">largest occlusion elsewhere</dt><dd>{f3(Math.max(...occ[a].slice(1)))}</dd>
                  <dt className="text-muted">gradient ‖∂z/∂x₁‖</dt><dd>{sci(g[0])}</dd>
                  <dt className="text-muted">gradient ‖∂z/∂x_T‖</dt><dd>{sci(g[T - 1])}</dd>
                </dl>
              </div>
            )
          })}
        </div>
        <p className="mt-2 text-[12px] text-muted">A gradient is a local slope. A model can be saturated (tiny gradient) with respect to an input whose replacement changes the answer completely.</p>
      </Card>
    </>
  )
}

/* ---- 9. lengths (real training results) */
function LengthsView() {
  const c = useColors()
  const archs: Arch[] = ['rnn', 'lstm', 'attn']
  const [evalT, setEvalT] = useState(12)
  const [live, setLive] = useState<Record<string, number> | null>(null)
  const { depth } = useApp()
  const xs = trained('mixed', 'rnn').accuracy.map((a) => a.T)
  const Regimes: [Regime, string][] = [['mixed', 'trained on lengths 2–20'], ['long', 'trained on lengths 15–25 only']]
  return (
    <>
      <div className="flex flex-wrap items-center gap-2"><span className="sim-badge">real training results · training/train_seq.py</span><span className="font-mono text-[11.5px] text-faint">accuracy on 400 fixed random sequences per length; chance = 25%</span></div>
      <div className="grid gap-3 @4xl:grid-cols-2">
        {Regimes.map(([r, l]) => (
          <Card key={r} title={`accuracy vs sequence length · ${l}`}>
            <LinChart xs={xs} xLabel="sequence length T (dependency distance T − 1)" yMax={1} yFmt={(v) => `${Math.round(v * 100)}%`} hline={{ y: 0.25, label: 'chance' }} marker={r === 'mixed' ? 20 : 25}
              series={archs.map((a) => ({ name: ARCH_LABEL[a], ys: trained(r, a).accuracy.map((x) => x.acc), color: archColor(c, a), dashed: a === 'attn' }))} />
            <p className="text-[11.5px] text-faint">Dashed amber line marks the longest training length; to its right the models extrapolate.{r === 'mixed' ? ' All three curves sit on top of each other at 100%.' : ''}</p>
          </Card>
        ))}
      </div>
      <div className="grid gap-3 @4xl:grid-cols-2">
        {Regimes.map(([r, l]) => (
          <Card key={r} title={`training loss · ${l}`}>
            <LinChart xs={trained(r, 'rnn').history.map((h) => h.step)} xLabel="training step" yMax={1.5} yFmt={(v) => v.toFixed(2)} hline={{ y: Math.log(4), label: 'ln 4 = guessing' }}
              series={archs.map((a) => ({ name: ARCH_LABEL[a], ys: trained(r, a).history.map((h) => h.loss), color: archColor(c, a), dashed: a === 'attn' }))} />
          </Card>
        ))}
      </div>
      <Card title="check it yourself: re-evaluate the trained models in your browser">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-56"><Slider id="eval-T" label={`length T = ${evalT}`} value={evalT} min={2} max={30} step={1} onChange={(v) => { setEvalT(v); setLive(null) }} /></div>
          <button type="button" className="btn btn-primary" onClick={() => setLive(Object.fromEntries(Regimes.flatMap(([r]) => archs.map((a) => [`${r}-${a}`, accuracyAt(trained(r, a).params, evalT, 200, 7)]))))}>Evaluate 200 new sequences</button>
        </div>
        {live && (
          <table className="mt-3 w-full font-mono text-[12px] font-num">
            <thead><tr className="text-left text-faint"><th className="font-normal">regime</th>{archs.map((a) => <th key={a} className="font-normal" style={{ color: archColor(c, a) }}>{ARCH_LABEL[a]}</th>)}</tr></thead>
            <tbody>{Regimes.map(([r, l]) => <tr key={r} className="border-b border-line/50"><td className="text-muted">{l}</td>{archs.map((a) => <td key={a}>{pct(live[`${r}-${a}`])} <span className="text-faint">(Python: {pct(trained(r, a).accuracy.find((x) => x.T === evalT)!.acc)})</span></td>)}</tr>)}</tbody>
          </table>
        )}
        {depth !== 'beginner' && <p className="mt-2 text-[12px] text-muted">These sequences are generated by the browser’s seeded generator, independent of the Python test set, so small differences are sampling noise.</p>}
      </Card>
    </>
  )
}

/* ---- 10. compare */
export function CompareView() {
  const { runs, cursor, unit, target, ids } = useSeq()
  const c = useColors()
  const t = cursor.t, T = ids.length
  const r = runs.rnn.trace, l = runs.lstm.trace, a = runs.attn.trace
  const g = (x: Arch) => vnorm(runs[x].grads.dx[0])
  return (
    <>
      <SourceBar />
      <CursorBar />
      <div className="grid gap-3 @5xl:grid-cols-3">
        <Card title={`RNN · h_${t + 1}`}>{r.arch === 'rnn' && <StateHeatmap rows={r.h} t={t} label="hidden state" />}</Card>
        <Card title={`LSTM · c_${t + 1} and gates`}>{l.arch === 'lstm' && <><StateHeatmap rows={l.c} t={t} label="cell state" maxAbs={Math.max(1, ...l.c.flat().map(Math.abs))} /><StateHeatmap rows={l.f} t={t} label="forget gate" mode="gate" /></>}</Card>
        <Card title={`attention · row ${t + 1}`}>{a.arch === 'attn' && (
          <div className="grid gap-[3px]">{a.A[t].slice(0, t + 1).map((w, s) => (
            <div key={s} className="grid grid-cols-[34px_1fr_44px] items-center gap-1 font-mono text-[11.5px]"><span className={s === 0 ? 'text-teal' : 'text-muted'}>{s + 1}:{SYMBOLS[ids[s]]}</span>
              <div className="h-2.5 rounded-sm" style={{ background: c.panel2 }}><div className="h-full rounded-sm" style={{ width: `${w * 100}%`, background: c.amber }} /></div><span className="text-right text-muted">{pct(w)}</span></div>
          ))}</div>
        )}</Card>
      </div>
      <Card title="summary for this sequence">
        <table className="w-full font-mono text-[12px] font-num">
          <thead><tr className="text-left text-faint"><th className="font-normal">model</th><th className="font-normal">prediction</th><th className="font-normal">p(correct)</th><th className="font-normal">dependent steps</th><th className="font-normal">path x₁→out</th><th className="font-normal">‖∂z/∂x₁‖</th><th className="font-normal">state size</th></tr></thead>
          <tbody>{(['rnn', 'lstm', 'attn'] as Arch[]).map((x) => (
            <tr key={x} className="border-b border-line/50"><td style={{ color: archColor(c, x) }}>{ARCH_LABEL[x]}</td><td className={runs[x].pred === target ? 'text-ok' : 'text-rose'}>{SYMBOLS[runs[x].pred]}</td><td>{pct(runs[x].trace.probs[target])}</td>
              <td>{cost(x, T, hiddenOf(runs[x].params)).sequentialSteps}</td><td>{cost(x, T, hiddenOf(runs[x].params)).pathLength}</td><td>{sci(g(x))}</td>
              <td>{x === 'attn' ? `${T} × ${hiddenOf(runs[x].params)} (all positions)` : x === 'lstm' ? `2 × ${hiddenOf(runs[x].params)}` : `${hiddenOf(runs[x].params)}`}</td></tr>
          ))}</tbody>
        </table>
        <p className="mt-2 font-mono text-[11px] text-faint">unit {unit + 1} selected · gradient objective follows the Gradients stage setting</p>
      </Card>
    </>
  )
}
