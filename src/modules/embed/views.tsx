/** Controls, charts and one view per guided stage for module 06. Every number is computed from the current space. */
import { AnimatePresence, motion } from 'framer-motion'
import { lazy, Suspense, useMemo, useRef, useState, type ReactNode } from 'react'
import { Card, Heatmap } from '../../components/viz'
import { useApp, useColors } from '../../sim/appState'
import { mat, softmax, topk } from '../../engine/tensor'
import { angleDeg, cosine, dot, euclid, norm, pca, sub, add, type Vec } from '../../engine/linalg'
import {
  KIND_LABEL, METRIC_LABEL, analogy, contextualEmbeddings, misleadingPairs, nearest, preservationByDims, projectPoint, similarityMatrix,
  type Metric, type Space,
} from '../../engine/embeddings'
import { Scatter2D, type Arrow } from './Scatter'
import { STAGES, type SpaceId } from './lesson'
import { f3, f4, pct, useEmb } from './context'

const Scatter3D = lazy(() => import('./Scatter3D'))

/* =================================================================== controls */

export function SourceBadge({ space }: { space: Space }) {
  return <span className={`sim-badge ${space.kind === 'simulation' ? 'toy' : ''}`} title={space.provenance}>{KIND_LABEL[space.kind]}</span>
}

export function SpacePicker({ value, onChange, idp = '' }: { value: SpaceId; onChange: (s: SpaceId) => void; idp?: string }) {
  const opts: { id: SpaceId; label: string; sub: string }[] = [
    { id: 'toy', label: 'Hand-built', sub: 'simulation' }, { id: 'ppmi', label: 'PPMI-SVD', sub: 'trained here' }, { id: 'nano', label: 'Transformer', sub: 'Observatory-Nano' },
  ]
  return (
    <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="embedding source" id={`${idp}space`}>
      {opts.map((o) => (
        <button key={o.id} type="button" role="radio" aria-checked={value === o.id} className="btn h-auto flex-col items-start gap-0 py-1" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          <span>{o.label}</span><span className={`font-mono text-[10px] ${o.id === 'toy' ? 'text-amber' : 'text-teal'}`}>{o.sub}</span>
        </button>
      ))}
    </div>
  )
}

export function Slider({ id, label, value, min, max, step, onChange }: { id: string; label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="grid gap-1 text-[12px] text-muted font-num" htmlFor={id}>{label}
      <input id={id} type="range" className="w-full min-w-0" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}

export function PPMIControls({ idp = '' }: { idp?: string }) {
  const { ppmiCfg, setPPMI, spaces } = useEmb()
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <Slider id={`${idp}win`} label={`context window ±${ppmiCfg.window}`} value={ppmiCfg.window} min={1} max={8} step={1} onChange={(window) => setPPMI({ window })} />
      <Slider id={`${idp}dim`} label={`dimensions d = ${ppmiCfg.dim}`} value={ppmiCfg.dim} min={2} max={64} step={1} onChange={(dim) => setPPMI({ dim })} />
      <Slider id={`${idp}alpha`} label={`context smoothing α = ${ppmiCfg.alpha.toFixed(2)}`} value={ppmiCfg.alpha} min={0.25} max={1} step={0.05} onChange={(alpha) => setPPMI({ alpha })} />
      <Slider id={`${idp}minc`} label={`min count = ${ppmiCfg.minCount}`} value={ppmiCfg.minCount} min={1} max={50} step={1} onChange={(minCount) => setPPMI({ minCount })} />
      <p className="col-span-full font-mono text-[11px] text-faint">{spaces.ppmi.words.length} words × {spaces.ppmi.dim} dims · {String(spaces.ppmi.stats?.['SVD energy kept'] ?? '')} of PPMI energy kept</p>
    </div>
  )
}

export function ProjControls({ idp = '', showDims = true }: { idp?: string; showDims?: boolean }) {
  const { proj, setProj, stale } = useEmb()
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="projection method">
          {(['pca', 'umap'] as const).map((m) => <button key={m} type="button" role="radio" aria-checked={proj.method === m} onClick={() => setProj({ method: m })} className={`rounded-md px-2.5 py-1 font-mono text-[12px] uppercase ${proj.method === m ? 'bg-raise text-cyan' : 'text-muted'}`}>{m}</button>)}
        </div>
        {showDims && <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="projection dimensions">
          {([2, 3] as const).map((d) => <button key={d} type="button" role="radio" aria-checked={proj.dims === d} onClick={() => setProj({ dims: d })} className={`rounded-md px-2.5 py-1 font-mono text-[12px] ${proj.dims === d ? 'bg-raise text-cyan' : 'text-muted'}`}>{d}D</button>)}
        </div>}
        <label className="flex items-center gap-1.5 text-[12px] text-muted" title="Project unit-length vectors, so the picture shows direction (what cosine measures) rather than length">
          <input type="checkbox" id={`${idp}norm`} checked={proj.normalize} onChange={(e) => setProj({ normalize: e.target.checked })} />normalize to unit length
        </label>
        {stale && <span className="font-mono text-[11px] text-amber" role="status">computing…</span>}
      </div>
      {proj.method === 'umap' && (
        <div className="grid gap-2 sm:grid-cols-2">
          <Slider id={`${idp}nn`} label={`n_neighbors = ${proj.nNeighbors}`} value={proj.nNeighbors} min={2} max={30} step={1} onChange={(nNeighbors) => setProj({ nNeighbors })} />
          <Slider id={`${idp}md`} label={`min_dist = ${proj.minDist.toFixed(2)}`} value={proj.minDist} min={0} max={0.99} step={0.01} onChange={(minDist) => setProj({ minDist })} />
          <Slider id={`${idp}ep`} label={`epochs = ${proj.nEpochs}`} value={proj.nEpochs} min={50} max={600} step={50} onChange={(nEpochs) => setProj({ nEpochs })} />
          <label className="grid gap-1 text-[12px] text-muted" htmlFor={`${idp}pseed`}>random seed
            <div className="flex gap-1"><input id={`${idp}pseed`} type="number" className="h-8 w-full min-w-0 px-2 font-mono" value={proj.seed} onChange={(e) => setProj({ seed: +e.target.value || 0 })} />
              <button type="button" className="btn h-8" onClick={() => setProj({ seed: proj.seed + 1 })}>Next</button></div>
          </label>
        </div>
      )}
    </div>
  )
}

export function MetricPicker() {
  const { metric, setMetric } = useEmb()
  return (
    <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="similarity measure">
      {(['cosine', 'dot', 'euclidean'] as Metric[]).map((m) => <button key={m} type="button" role="radio" aria-checked={metric === m} onClick={() => setMetric(m)} className={`rounded-md px-2.5 py-1 text-[12px] ${metric === m ? 'bg-raise text-cyan' : 'text-muted'}`}>{METRIC_LABEL[m]}</button>)}
    </div>
  )
}

export function WordSelect({ id, label, value, onChange, allowNone }: { id: string; label: string; value: string; onChange: (w: string) => void; allowNone?: boolean }) {
  const { space } = useEmb()
  return (
    <label className="grid gap-1 text-[12px] text-muted" htmlFor={id}>{label}
      <select id={id} className="h-8 w-full min-w-0 px-2 font-mono text-[13px]" value={value} onChange={(e) => onChange(e.target.value)}>
        {allowNone && <option value="">(none)</option>}
        {space.words.map((w) => <option key={w} value={w}>{w}</option>)}
      </select>
    </label>
  )
}

export function SelectedChips() {
  const { space, picks, setPicks } = useEmb()
  const [add, setAdd] = useState('')
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {picks.selected.filter((w) => space.words.includes(w)).map((w) => (
        <button key={w} type="button" className="chip" onClick={() => setPicks({ selected: picks.selected.filter((x) => x !== w) })} aria-label={`Remove ${w}`}>{w} <span className="id">✕</span></button>
      ))}
      <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); if (space.words.includes(add) && !picks.selected.includes(add)) setPicks({ selected: [...picks.selected, add] }); setAdd('') }}>
        <label htmlFor="add-word" className="sr-only">Add a word</label>
        <input id="add-word" list="emb-words" className="h-8 w-32 px-2 font-mono text-[12px]" placeholder="add word…" value={add} onChange={(e) => setAdd(e.target.value)} />
        <datalist id="emb-words">{space.words.map((w) => <option key={w} value={w} />)}</datalist>
        <button type="submit" className="btn h-8">Add</button>
      </form>
    </div>
  )
}

/** The map, in 2D (SVG) or 3D (WebGL), wired to the lab's focus and selection. */
export function MapPanel({ title, neighbourK = 0, arrows, extraPoints, height, force2D, fitTo }: { title?: ReactNode; neighbourK?: number; arrows?: Arrow[]; extraPoints?: { at: Vec; label: string; color: string }[]; height?: number; force2D?: boolean; fitTo?: number[] }) {
  const { space, projection, focusIdx, selIdx, picks, setPicks, proj, highlightPair } = useEmb()
  const onFocus = (i: number) => setPicks({ focus: space.words[i] })
  const onToggle = (i: number) => { const w = space.words[i]; setPicks({ selected: picks.selected.includes(w) ? picks.selected.filter((x) => x !== w) : [...picks.selected, w] }) }
  const is3D = proj.dims === 3 && !force2D
  const coords = force2D && proj.dims === 3 ? projection.coords.map((c) => c.slice(0, 2)) : projection.coords
  return (
    <Card title={title ?? `${space.title} · ${proj.method.toUpperCase()} ${is3D ? '3D' : '2D'} · ${space.words.length} words, ${space.dim} dims → ${is3D ? 3 : 2}`}
      right={<span className="font-mono text-[10.5px] text-faint">click = focus · shift-click = select · scroll = zoom</span>}>
      {is3D
        ? <Suspense fallback={<div className="grid h-[460px] place-items-center text-muted">Loading 3D view…</div>}><Scatter3D space={space} coords={coords} focus={focusIdx} selected={selIdx} onFocus={onFocus} /></Suspense>
        : <Scatter2D space={space} coords={coords} focus={focusIdx} selected={selIdx} onFocus={onFocus} onToggle={onToggle} neighbourK={neighbourK} arrows={arrows} extraPoints={extraPoints} highlightPair={highlightPair} height={height} fitTo={fitTo} />}
    </Card>
  )
}

/* ===================================================================== charts */

export function LineChart({ xs, series, xLabel, yLabel, yMax, current, height = 180, xFmt = (x: number) => String(x) }: {
  xs: number[]; series: { name: string; ys: number[]; color: string }[]; xLabel: string; yLabel?: string; yMax?: number; current?: number; height?: number; xFmt?: (x: number) => string
}) {
  const c = useColors()
  const W = 420, H = height, L = 40, R = 12, T = 12, B = 30
  const all = series.flatMap((s) => s.ys).filter(Number.isFinite)
  const y1 = yMax ?? Math.max(1e-9, ...all), y0 = Math.min(0, ...all)
  const x = (i: number) => L + (i / Math.max(1, xs.length - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B)
  const ticks = [y0, (y0 + y1) / 2, y1]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 560, display: 'block' }} role="img" aria-label={`${yLabel ?? 'value'} by ${xLabel}`}>
      {ticks.map((t) => <g key={t}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke={c.line} /><text x={L - 5} y={y(t) + 3} fontSize={9.5} textAnchor="end" fill={c.faint}>{t.toFixed(2)}</text></g>)}
      {xs.map((v, i) => (xs.length <= 12 || i % Math.ceil(xs.length / 10) === 0) && <text key={i} x={x(i)} y={H - 14} fontSize={9.5} textAnchor="middle" fill={c.faint}>{xFmt(v)}</text>)}
      <text x={(L + W - R) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill={c.muted}>{xLabel}</text>
      {current !== undefined && xs.indexOf(current) >= 0 && <line x1={x(xs.indexOf(current))} x2={x(xs.indexOf(current))} y1={T} y2={H - B} stroke={c.amber} strokeDasharray="3 3" />}
      {series.map((s) => (
        <g key={s.name}>
          <path d={s.ys.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} />
          {s.ys.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.6} fill={s.color}><title>{`${s.name} at ${xFmt(xs[i])}: ${v.toFixed(4)}`}</title></circle>)}
        </g>
      ))}
      {series.length > 1 && series.map((s, k) => <text key={s.name} x={L + 6 + k * 120} y={T + 10} fontSize={10} fill={s.color}>● {s.name}</text>)}
    </svg>
  )
}

function Bars({ items, max, fmt = f3 }: { items: { label: string; value: number; color?: string; dim?: boolean }[]; max?: number; fmt?: (x: number) => string }) {
  const c = useColors()
  const m = max ?? Math.max(1e-9, ...items.map((i) => Math.abs(i.value)))
  return (
    <div className="grid gap-[3px]">
      {items.map((it, k) => (
        <div key={k} className="grid grid-cols-[minmax(70px,120px)_1fr_64px] items-center gap-2 font-mono text-[12px]" style={{ opacity: it.dim ? 0.45 : 1 }}>
          <span className="truncate text-right">{it.label}</span>
          <div className="relative h-[14px] rounded-sm" style={{ background: c.panel2 }}>
            <div className="absolute inset-y-0 rounded-sm transition-[width] duration-500" style={{ left: it.value < 0 ? `${50 - (Math.abs(it.value) / m) * 50}%` : max === undefined || items.every((x) => x.value >= 0) ? 0 : '50%', width: `${(Math.abs(it.value) / m) * (items.some((x) => x.value < 0) ? 50 : 100)}%`, background: it.color ?? (it.value < 0 ? c.violet : c.cyan) }} />
          </div>
          <span className="text-right text-muted font-num">{fmt(it.value)}</span>
        </div>
      ))}
    </div>
  )
}

/* ================================================================== stages */

export function GuidedStage() {
  const { stage } = useEmb()
  const V: Record<string, () => React.JSX.Element> = {
    lookup: LookupView, dot: DotView, cosine: CosineView, matrix: MatrixView, neighbours: NeighboursView, arithmetic: ArithmeticView,
    pca: PCAView, umap: UMAPView, distortion: DistortionView, context: ContextView, compare: CompareStage,
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

function SpaceBar({ ppmi = true }: { ppmi?: boolean }) {
  const { spaceId, setSpaceId, space } = useEmb()
  return (
    <Card title="embedding source" right={<SourceBadge space={space} />}>
      <div className="grid gap-3 lg:grid-cols-[auto_1fr] lg:items-start">
        <SpacePicker value={spaceId} onChange={setSpaceId} />
        <p className="max-w-[70ch] text-[12.5px] leading-relaxed text-muted">{space.provenance}</p>
      </div>
      {ppmi && spaceId === 'ppmi' && <div className="mt-3 border-t border-line pt-3"><PPMIControls /></div>}
    </Card>
  )
}

/* ---- 1. lookup */
function LookupView() {
  const { space, focusIdx, picks, setPicks } = useEmb()
  const { depth } = useApp()
  const rows = space.words.length
  const showRows = Math.min(rows, 24)
  const start = Math.max(0, Math.min(focusIdx - Math.floor(showRows / 2), rows - showRows))
  const sub = mat(showRows, space.dim, space.vecs.slice(start, start + showRows).flat())
  const v = space.vecs[focusIdx]
  return (
    <>
      <SpaceBar />
      <div className="grid gap-3 @5xl:grid-cols-[1.2fr_1fr]">
        <Card title={`embedding table E · [${rows} × ${space.dim}]${rows > showRows ? ` · rows ${start + 1}–${start + showRows}` : ''}`}>
          <div className="overflow-x-auto scroll-thin">
            <Heatmap m={sub} rowLabels={space.words.slice(start, start + showRows).map((w) => w.slice(0, 10))} colLabels={space.featureNames ?? undefined}
              selectedRow={focusIdx - start} onCell={(i) => setPicks({ focus: space.words[start + i] })} cell={space.dim > 30 ? 9 : space.dim > 14 ? 16 : 28} ariaLabel="embedding table" />
          </div>
        </Card>
        <Card title={`lookup “${picks.focus}”: one-hot × E = row ${focusIdx}`}>
          <WordSelect id="lk-focus" label="word" value={picks.focus} onChange={(focus) => setPicks({ focus })} />
          <div className="mt-3 overflow-x-auto font-mono text-[11.5px] scroll-thin">
            <div className="text-faint">1<sub>w</sub> (length {rows}): one 1 at position {focusIdx}, zeros elsewhere</div>
            <div className="mt-1 flex flex-wrap gap-[2px]">{space.words.slice(0, Math.min(rows, 60)).map((w, i) => <span key={w} title={w} className={`inline-block h-3 w-3 rounded-sm ${i === focusIdx ? 'bg-amber' : 'bg-panel-2 border border-line'}`} />)}{rows > 60 && <span className="text-faint">…</span>}</div>
          </div>
          <div className="mt-3 grid gap-[3px]">
            {(space.featureNames ? v.map((x, i) => ({ label: space.featureNames![i], value: x })) : v.slice(0, depth === 'advanced' ? 48 : 12).map((x, i) => ({ label: `dim ${i}`, value: x }))).map((it) => (
              <div key={it.label} className="grid grid-cols-[80px_1fr_60px] items-center gap-2 font-mono text-[12px]">
                <span className="text-right text-muted">{it.label}</span>
                <div className="relative h-3 rounded-sm bg-panel-2"><div className="absolute inset-y-0 rounded-sm" style={{ left: it.value < 0 ? `${50 - Math.min(50, (Math.abs(it.value) / (Math.max(...v.map(Math.abs)) || 1)) * 50)}%` : '50%', width: `${Math.min(50, (Math.abs(it.value) / (Math.max(...v.map(Math.abs)) || 1)) * 50)}%`, background: it.value < 0 ? 'var(--violet)' : 'var(--amber)' }} /></div>
                <span className="text-right font-num">{f3(it.value)}</span>
              </div>
            ))}
            {!space.featureNames && v.length > (depth === 'advanced' ? 48 : 12) && <p className="font-mono text-[11px] text-faint">… {v.length - 12} more dimensions (advanced depth shows all). These have no names: the model chose the basis.</p>}
          </div>
          <p className="mt-2 font-mono text-[12px] text-muted font-num">‖e‖ = {f4(norm(v))}</p>
        </Card>
      </div>
    </>
  )
}

/* ---- 2/3. dot and cosine playground */
function VectorPlane({ showProjection = true, showAngle = true }: { showProjection?: boolean; showAngle?: boolean }) {
  const { vecA, vecB, setVecA, setVecB, scaleA } = useEmb()
  const c = useColors()
  const svgRef = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<'a' | 'b' | null>(null)
  const R = 5, S = 400, u = S / (2 * R)
  const P = (v: Vec): [number, number] => [S / 2 + v[0] * u, S / 2 - v[1] * u]
  const a: Vec = [vecA[0] * scaleA, vecA[1] * scaleA], b: Vec = vecB
  const toVec = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * S, y = ((e.clientY - r.top) / r.height) * S
    const snap = (t: number) => Math.max(-R, Math.min(R, Math.round(t * 4) / 4))
    return [snap((x - S / 2) / u), snap((S / 2 - y) / u)] as [number, number]
  }
  const projLen = norm(a) ? dot(a, b) / norm(a) : 0
  const projV: Vec = norm(a) ? a.map((x) => (x * dot(a, b)) / dot(a, a)) : [0, 0]
  const th = (Math.atan2(a[1], a[0]) * 180) / Math.PI, tb = (Math.atan2(b[1], b[0]) * 180) / Math.PI
  const arc = (() => {
    const r = 34, s = (Math.min(th, tb) * Math.PI) / 180
    let d = Math.abs(th - tb); let s0 = Math.min(th, tb)
    if (d > 180) { d = 360 - d; s0 = Math.max(th, tb) }
    const e = ((s0 + d) * Math.PI) / 180, s1 = (s0 * Math.PI) / 180
    void s
    return `M${S / 2 + r * Math.cos(s1)},${S / 2 - r * Math.sin(s1)} A${r},${r} 0 0 0 ${S / 2 + r * Math.cos(e)},${S / 2 - r * Math.sin(e)}`
  })()
  return (
    <svg ref={svgRef} viewBox={`0 0 ${S} ${S}`} width="100%" style={{ maxWidth: 460, touchAction: 'none' }} role="img" aria-label="two draggable vectors"
      onPointerMove={(e) => { if (!drag) return; const v = toVec(e); if (drag === 'a') setVecA([v[0] / scaleA, v[1] / scaleA]); else setVecB(v) }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
      <defs>
        <marker id="va" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill={c.cyan} /></marker>
        <marker id="vb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill={c.amber} /></marker>
      </defs>
      {Array.from({ length: 2 * R + 1 }, (_, i) => i - R).map((t) => (
        <g key={t}>
          <line x1={P([t, -R])[0]} x2={P([t, R])[0]} y1={0} y2={S} stroke={t === 0 ? c.lineStrong : c.line} strokeWidth={t === 0 ? 1.4 : 0.6} />
          <line y1={P([-R, t])[1]} y2={P([R, t])[1]} x1={0} x2={S} stroke={t === 0 ? c.lineStrong : c.line} strokeWidth={t === 0 ? 1.4 : 0.6} />
          {t !== 0 && <text x={P([t, 0])[0]} y={S / 2 + 12} fontSize={9} textAnchor="middle" fill={c.faint}>{t}</text>}
        </g>
      ))}
      {showProjection && norm(a) > 0 && <>
        <line x1={P(b)[0]} y1={P(b)[1]} x2={P(projV)[0]} y2={P(projV)[1]} stroke={c.muted} strokeDasharray="4 3" />
        <line x1={S / 2} y1={S / 2} x2={P(projV)[0]} y2={P(projV)[1]} stroke={c.teal} strokeWidth={5} strokeOpacity={0.55} />
        <text x={P(projV)[0] + 6} y={P(projV)[1] + 16} fontSize={11} fill={c.teal} fontFamily="JetBrains Mono, monospace">shadow {projLen.toFixed(2)}</text>
      </>}
      {showAngle && norm(a) > 0 && norm(b) > 0 && <path d={arc} fill="none" stroke={c.violet} strokeWidth={2} />}
      <line x1={S / 2} y1={S / 2} x2={P(a)[0]} y2={P(a)[1]} stroke={c.cyan} strokeWidth={3} markerEnd="url(#va)" />
      <line x1={S / 2} y1={S / 2} x2={P(b)[0]} y2={P(b)[1]} stroke={c.amber} strokeWidth={3} markerEnd="url(#vb)" />
      <circle cx={P(a)[0]} cy={P(a)[1]} r={12} fill={c.cyan} fillOpacity={0.18} stroke={c.cyan} style={{ cursor: 'grab' }} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag('a') }} aria-label="drag vector a" />
      <circle cx={P(b)[0]} cy={P(b)[1]} r={12} fill={c.amber} fillOpacity={0.18} stroke={c.amber} style={{ cursor: 'grab' }} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag('b') }} aria-label="drag vector b" />
      <text x={P(a)[0] + 12} y={P(a)[1] - 10} fontSize={14} fill={c.cyan} fontWeight={600}>a</text>
      <text x={P(b)[0] + 12} y={P(b)[1] - 10} fontSize={14} fill={c.amber} fontWeight={600}>b</text>
    </svg>
  )
}

function VecInputs() {
  const { vecA, vecB, setVecA, setVecB } = useEmb()
  const inp = (id: string, val: number, on: (v: number) => void) => <input id={id} type="number" step={0.25} min={-5} max={5} className="h-8 w-20 px-2 font-mono" value={val} onChange={(e) => on(Math.max(-5, Math.min(5, +e.target.value || 0)))} />
  return (
    <div className="flex flex-wrap items-end gap-3 font-mono text-[12px]">
      <span className="text-cyan">a =</span>
      <label htmlFor="a0" className="grid gap-0.5 text-faint">a₁{inp('a0', vecA[0], (x) => setVecA([x, vecA[1]]))}</label>
      <label htmlFor="a1" className="grid gap-0.5 text-faint">a₂{inp('a1', vecA[1], (x) => setVecA([vecA[0], x]))}</label>
      <span className="ml-3 text-amber">b =</span>
      <label htmlFor="b0" className="grid gap-0.5 text-faint">b₁{inp('b0', vecB[0], (x) => setVecB([x, vecB[1]]))}</label>
      <label htmlFor="b1" className="grid gap-0.5 text-faint">b₂{inp('b1', vecB[1], (x) => setVecB([vecB[0], x]))}</label>
    </div>
  )
}

function Presets() {
  const { setVecA, setVecB, setScaleA } = useEmb()
  const p: [string, [number, number], [number, number]][] = [['same direction', [2, 1], [4, 2]], ['perpendicular', [3, 1], [-1, 3]], ['opposite', [2, 2], [-3, -3]], ['acute, different lengths', [4, 0.5], [1.5, 2]]]
  return <div className="flex flex-wrap gap-1.5">{p.map(([l, a, b]) => <button key={l} type="button" className="btn h-7 text-[12px]" onClick={() => { setVecA(a); setVecB(b); setScaleA(1) }}>{l}</button>)}</div>
}

function DotView() {
  const { vecA, vecB, scaleA } = useEmb()
  const a = [vecA[0] * scaleA, vecA[1] * scaleA], b = vecB
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><Presets /><span className="sim-badge toy">exact geometry · vectors you draw</span></div>
      <div className="grid gap-3 @4xl:grid-cols-[minmax(0,460px)_1fr]">
        <Card title="drag the arrow tips (snaps to 0.25)"><VectorPlane showAngle={false} /><div className="mt-2"><VecInputs /></div></Card>
        <Card title="a · b, two ways">
          <div className="grid gap-3 font-mono text-[13px] font-num">
            <div><div className="eyebrow mb-1">algebra: sum of products</div>
              <p>a · b = ({f3(a[0])})({f3(b[0])}) + ({f3(a[1])})({f3(b[1])}) = {f3(a[0] * b[0])} + {f3(a[1] * b[1])} = <span className="text-amber text-[16px]">{f3(dot(a, b))}</span></p></div>
            <div><div className="eyebrow mb-1">geometry: length × shadow</div>
              <p>‖a‖ = √({f3(a[0])}² + {f3(a[1])}²) = {f3(norm(a))}</p>
              <p>shadow of b on a = ‖b‖ cos θ = {f3(norm(a) ? dot(a, b) / norm(a) : 0)}</p>
              <p>‖a‖ × shadow = <span className="text-amber">{f3(norm(a) * (norm(a) ? dot(a, b) / norm(a) : 0))}</span> {Math.abs(dot(a, b) - norm(a) * (norm(a) ? dot(a, b) / norm(a) : 0)) < 1e-9 ? '✓ same' : ''}</p></div>
            <p className="font-sans text-[13px] text-muted">{dot(a, b) > 1e-9 ? 'Positive: the vectors point broadly the same way.' : dot(a, b) < -1e-9 ? 'Negative: they point broadly in opposite directions.' : 'Zero: the vectors are perpendicular (orthogonal).'}</p>
          </div>
        </Card>
      </div>
    </>
  )
}

function CosineView() {
  const { vecA, vecB, scaleA, setScaleA } = useEmb()
  const a = [vecA[0] * scaleA, vecA[1] * scaleA], b = vecB
  const ks = [0.25, 0.5, 1, 1.5, 2]
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><Presets /><span className="sim-badge toy">exact geometry · vectors you draw</span></div>
      <div className="grid gap-3 @4xl:grid-cols-[minmax(0,460px)_1fr]">
        <Card title="drag the tips · stretch a with the slider">
          <VectorPlane showProjection={false} />
          <div className="mt-2 grid gap-2"><Slider id="scale-a" label={`scale a by ${scaleA.toFixed(2)}`} value={scaleA} min={0.25} max={2} step={0.05} onChange={setScaleA} /><VecInputs /></div>
        </Card>
        <Card title="cosine = dot product ÷ both lengths">
          <div className="grid gap-2 font-mono text-[13px] font-num">
            <p>a · b = {f3(dot(a, b))}</p>
            <p>‖a‖ ‖b‖ = {f3(norm(a))} × {f3(norm(b))} = {f3(norm(a) * norm(b))}</p>
            <p>cos θ = {f3(dot(a, b))} / {f3(norm(a) * norm(b))} = <span className="text-[18px] text-amber">{f4(cosine(a, b))}</span></p>
            <p>θ = arccos(cos θ) = <span className="text-violet">{(norm(a) && norm(b) ? angleDeg(a, b) : 0).toFixed(1)}°</span></p>
          </div>
          <div className="mt-3">
            <div className="eyebrow mb-1">if a were scaled by…</div>
            <table className="w-full font-mono text-[12px] font-num">
              <thead><tr className="text-left text-faint"><th className="font-normal">scale</th><th className="font-normal">a · b</th><th className="font-normal">cos</th></tr></thead>
              <tbody>{ks.map((s) => { const as = [vecA[0] * s, vecA[1] * s]; return <tr key={s} className={Math.abs(s - scaleA) < 1e-9 ? 'text-amber' : ''}><td>×{s}</td><td>{f3(dot(as, b))}</td><td>{f4(cosine(as, b))}</td></tr> })}</tbody>
            </table>
            <p className="mt-2 text-[12.5px] text-muted">The dot product grows with length; the cosine column never changes.</p>
          </div>
        </Card>
      </div>
    </>
  )
}

/* ---- 4. similarity matrix */
function MatrixView() {
  const { space, selIdx, metric, cell, setCell } = useEmb()
  const M = useMemo(() => similarityMatrix(space, selIdx, metric), [space, selIdx, metric])
  const m = mat(selIdx.length, selIdx.length, M.flat())
  const labels = selIdx.map((i) => space.words[i])
  const ma = metric === 'cosine' ? 1 : Math.max(1e-9, ...M.flat().map(Math.abs))
  return (
    <>
      <SpaceBar />
      <div className="flex flex-wrap items-center gap-2"><MetricPicker /></div>
      <Card title="selected words · click ✕ to remove, add any word, or shift-click on a map"><SelectedChips /></Card>
      <Card title={`${METRIC_LABEL[metric]} matrix · ${selIdx.length} × ${selIdx.length}${metric === 'euclidean' ? ' (distance: small = similar)' : ''}`}>
        <div className="overflow-x-auto scroll-thin">
          <Heatmap m={m} rowLabels={labels} colLabels={labels} maxAbs={ma} showValues={selIdx.length <= 12} digits={2} cell={selIdx.length <= 12 ? 46 : 26}
            selectedCell={cell} onCell={(i, j) => setCell([i, j])} ariaLabel="similarity matrix" />
        </div>
      </Card>
    </>
  )
}

/* ---- 5. neighbours */
function NeighboursView() {
  const { space, focusIdx, picks, setPicks, metric, k, setK } = useEmb()
  const nn = useMemo(() => nearest(space, space.vecs[focusIdx], k, metric, [focusIdx]), [space, focusIdx, k, metric])
  const vecLen = (i: number) => norm(space.vecs[i])
  return (
    <>
      <SpaceBar />
      <div className="flex flex-wrap items-end gap-3">
        <div className="w-48"><WordSelect id="nn-focus" label="query word" value={picks.focus} onChange={(focus) => setPicks({ focus })} /></div>
        <MetricPicker />
        <div className="w-40"><Slider id="nn-k" label={`k = ${k}`} value={k} min={1} max={15} step={1} onChange={setK} /></div>
      </div>
      <div className="grid gap-3 @5xl:grid-cols-[1.3fr_1fr]">
        <MapPanel neighbourK={k} force2D />
        <Card title={`${k} nearest to “${picks.focus}” by ${METRIC_LABEL[metric]} · full ${space.dim}-D`}>
          <Bars items={nn.map((x) => ({ label: x.word, value: x.score }))} fmt={f4} max={metric === 'cosine' ? 1 : undefined} />
          <table className="mt-3 w-full font-mono text-[11.5px] font-num">
            <thead><tr className="text-left text-faint"><th className="font-normal">word</th><th className="font-normal">cos</th><th className="font-normal">dot</th><th className="font-normal">‖e‖</th></tr></thead>
            <tbody>{nn.map((x) => <tr key={x.i} className="border-b border-line/50"><td>{x.word}</td><td>{f4(cosine(space.vecs[focusIdx], space.vecs[x.i]))}</td><td>{f3(dot(space.vecs[focusIdx], space.vecs[x.i]))}</td><td>{f3(vecLen(x.i))}</td></tr>)}</tbody>
          </table>
          {metric === 'dot' && <p className="mt-2 text-[12px] text-muted">The dot product rewards long vectors: compare the ‖e‖ column. Switch back to cosine to rank by direction only.</p>}
        </Card>
      </div>
    </>
  )
}

/* ---- 6. arithmetic */
export function AnalogyBox({ compact = false }: { compact?: boolean }) {
  const { space, picks, setPicks } = useEmb()
  const W = (w: string) => space.words.indexOf(w)
  const [a, b, c, e] = picks.analogy
  const ok = [a, b, c].every((w) => W(w) >= 0) && new Set([a, b, c]).size === 3
  const res = useMemo(() => (ok ? analogy(space, W(a), W(b), W(c), 8, space.kind !== 'simulation', e && W(e) >= 0 ? W(e) : undefined) : null), [space, a, b, c, e, ok])
  const set = (i: number, w: string) => { const n = [...picks.analogy] as typeof picks.analogy; n[i] = w; setPicks({ analogy: n }) }
  return (
    <Card title={`analogy · ${a} is to ${b} as ${c} is to ?`}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <WordSelect id="an-a" label="a" value={a} onChange={(w) => set(0, w)} />
        <WordSelect id="an-b" label="b" value={b} onChange={(w) => set(1, w)} />
        <WordSelect id="an-c" label="c" value={c} onChange={(w) => set(2, w)} />
        <WordSelect id="an-e" label="expected (optional)" value={e} onChange={(w) => set(3, w)} allowNone />
      </div>
      {!ok && <p className="mt-2 text-[13px] text-rose">Pick three different words.</p>}
      {res && (
        <div className="mt-3 grid gap-3">
          <p className="font-mono text-[13px]">x* = argmax cos(e<sub>x</sub>, e<sub>{b}</sub> − e<sub>{a}</sub> + e<sub>{c}</sub>){space.kind !== 'simulation' && <span className="text-faint"> (inputs unit-normalized)</span>}, x ∉ {'{'}{a}, {b}, {c}{'}'}</p>
          <Bars items={res.ranked.map((x, k) => ({ label: `${k + 1}. ${x.word}`, value: x.score, color: x.word === e ? 'var(--ok)' : k === 0 ? 'var(--amber)' : undefined }))} max={1} fmt={f4} />
          {e && res.expectedRank !== undefined && (
            <p role="status" className={`text-[13px] ${res.expectedRank === 1 ? 'text-ok' : 'text-rose'}`}>
              “{e}” is ranked {res.expectedRank} of {space.words.length - 3} (cos {f4(res.expectedScore!)}). {res.expectedRank === 1 ? 'The analogy works in this space.' : 'The analogy does not land on the expected word here.'}
            </p>
          )}
          {!compact && space.featureNames && (
            <div className="overflow-x-auto scroll-thin">
              <table className="font-mono text-[11.5px] font-num">
                <thead><tr className="text-faint"><th className="pr-2 text-left font-normal">feature</th>{space.featureNames.map((f) => <th key={f} className="px-1 font-normal">{f}</th>)}</tr></thead>
                <tbody>
                  {[[b, space.vecs[W(b)]], [`− ${a}`, space.vecs[W(a)].map((x) => -x)], [`+ ${c}`, space.vecs[W(c)]], ['= target', res.target], [res.ranked[0].word, space.vecs[res.ranked[0].i]]].map(([l, v]) => (
                    <tr key={String(l)} className={l === '= target' ? 'text-amber' : ''}><td className="pr-2">{l as string}</td>{(v as number[]).map((x, i) => <td key={i} className="px-1 text-center">{Number.isInteger(x) ? x : x.toFixed(2)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

function ArithmeticView() {
  const { space, picks, setPicks, projection, focusIdx, selIdx } = useEmb()
  const c = useColors()
  const [view, setView] = useState<'plane' | 'pca'>('plane')
  const W = (w: string) => space.words.indexOf(w)
  const [a, b, cc] = picks.analogy
  const ok = [a, b, cc].every((w) => W(w) >= 0) && new Set([a, b, cc]).size === 3
  // vectors as the analogy uses them: raw for the hand-built space, unit length for learned spaces (3CosAdd)
  const unitv = (v: Vec) => (space.kind === 'simulation' ? v : v.map((x) => x / (norm(v) || 1)))
  const plane = useMemo(() => {
    if (!ok) return null
    const va = unitv(space.vecs[W(a)]), vb = unitv(space.vecs[W(b)]), vc = unitv(space.vecs[W(cc)])
    const target = add(sub(vb, va), vc)
    // orthonormal basis of the plane through a spanned by (b − a) and (c − a), by Gram-Schmidt
    const d1 = sub(vb, va), d2 = sub(vc, va)
    const u1 = d1.map((x) => x / (norm(d1) || 1))
    const r2 = sub(d2, u1.map((x) => x * dot(d2, u1)))
    const u2 = r2.map((x) => x / (norm(r2) || 1))
    const to2 = (v: Vec) => { const w = sub(v, va); return [dot(w, u1), dot(w, u2)] }
    const coords = space.vecs.map((v) => to2(unitv(v)))
    const residual = norm(sub(target, add(va, add(u1.map((x) => x * dot(sub(target, va), u1)), u2.map((x) => x * dot(sub(target, va), u2))))))
    return { coords, target: to2(target), residual, d2norm: norm(d2) }
  }, [space, a, b, cc, ok])
  let arrows: Arrow[] = [], extra: { at: Vec; label: string; color: string }[] = [], coords: Vec[] = projection.coords.map((p) => p.slice(0, 2))
  let shrink: number | null = null
  if (ok && plane) {
    let pt: Vec
    if (view === 'plane') { coords = plane.coords; pt = plane.target }
    else {
      const target = add(sub(unitv(space.vecs[W(b)]), unitv(space.vecs[W(a)])), unitv(space.vecs[W(cc)]))
      const meanLen = space.vecs.reduce((s2, v) => s2 + norm(v), 0) / space.vecs.length
      const tPlot = projection.normalized || space.kind === 'simulation' ? target : target.map((x) => (x * meanLen) / (norm(target) || 1))
      pt = projectPoint(projection, tPlot).slice(0, 2)
      const full = projection.normalized ? norm(sub(unitv(space.vecs[W(cc)]).map((x) => x / (norm(unitv(space.vecs[W(cc)])) || 1)), unitv(space.vecs[W(a)]).map((x) => x / (norm(unitv(space.vecs[W(a)])) || 1)))) : norm(sub(space.vecs[W(cc)], space.vecs[W(a)]))
      shrink = full ? Math.hypot(coords[W(cc)][0] - coords[W(a)][0], coords[W(cc)][1] - coords[W(a)][1]) / full : null
    }
    arrows = [{ from: coords[W(a)], to: coords[W(b)], label: `${b} − ${a}`, color: c.cyan }, { from: coords[W(cc)], to: pt, label: 'same offset', color: c.amber, dashed: true }]
    extra = [{ at: pt, label: `${b} − ${a} + ${cc}`, color: c.amber }]
  }
  const canPCA = !!projection.pcaModel
  return (
    <>
      <SpaceBar />
      <div className="grid gap-3 @5xl:grid-cols-[1fr_1.1fr]">
        <AnalogyBox />
        <Card title={view === 'plane' ? `the analogy plane through ${a}, spanned by ${b} − ${a} and ${cc} − ${a}` : 'the same offsets in the 2D PCA shadow'}
          right={<div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="analogy view">
            {([['plane', 'analogy plane'], ['pca', 'PCA shadow']] as const).map(([v, l]) => <button key={v} type="button" role="radio" aria-checked={view === v} disabled={v === 'pca' && !canPCA} onClick={() => setView(v)} className={`rounded-md px-2 py-1 text-[11.5px] ${view === v ? 'bg-raise text-cyan' : 'text-muted'}`}>{l}</button>)}
          </div>}>
          {ok ? <Scatter2D space={space} coords={coords} focus={focusIdx} selected={selIdx} onFocus={(i) => setPicks({ focus: space.words[i] })} arrows={arrows} extraPoints={extra} height={400} fitTo={[W(a), W(b), W(cc)]} ariaLabel="analogy map" />
            : <p className="text-muted">Pick three different words.</p>}
          {ok && plane && view === 'plane' && (
            <p className="mt-2 text-[12px] leading-relaxed text-muted">An exact linear projection: a, b and c sit at their true relative positions, and the target b − a + c lies in this plane by construction (off-plane residual {plane.residual.toExponential(1)}). If the relation holds, {cc} → target is parallel to {a} → {b} and lands on the expected word. Other words are projected onto the same plane and may look closer than they are.</p>
          )}
          {ok && view === 'pca' && shrink !== null && (
            <p className={`mt-2 text-[12px] leading-relaxed ${shrink < 0.5 ? 'text-amber' : 'text-muted'}`}>On screen, {a} and {cc} are drawn at {(shrink * 100).toFixed(0)}% of their true distance. {shrink < 0.3 ? 'The direction that separates them is almost invisible in this shadow, so the two arrows collapse onto each other.' : ''}</p>
          )}
          {!canPCA && view === 'pca' && <p className="text-muted">The PCA shadow needs the PCA projection (switch away from UMAP).</p>}
        </Card>
      </div>
    </>
  )
}

/* ---- 7. PCA */
function PCAView() {
  const { space, proj } = useEmb()
  const c = useColors()
  const full = useMemo(() => pca(space.vecs, Math.min(space.dim, 12)), [space])
  const cum = full.explained.reduce<number[]>((acc, x) => [...acc, (acc.at(-1) ?? 0) + x], [])
  return (
    <>
      <SpaceBar />
      <ProjControls />
      <div className="grid gap-3 @5xl:grid-cols-[1.4fr_1fr]">
        <MapPanel neighbourK={0} />
        <div className="grid gap-3">
          <Card title="explained variance per principal component">
            <svg viewBox="0 0 360 170" width="100%" style={{ maxWidth: 520, display: 'block' }} role="img" aria-label="scree plot">
              {full.explained.map((x, i) => {
                const bw = 300 / full.explained.length
                return (
                  <g key={i}>
                    <rect x={40 + i * bw + 2} y={140 - x * 120 / Math.max(...full.explained)} width={bw - 4} height={x * 120 / Math.max(...full.explained)} fill={i < proj.dims ? c.cyan : c.lineStrong} rx={2}><title>{`PC${i + 1}: ${pct(x)}`}</title></rect>
                    <text x={40 + i * bw + bw / 2} y={154} fontSize={9} textAnchor="middle" fill={c.faint}>{i + 1}</text>
                    <circle cx={40 + i * bw + bw / 2} cy={140 - cum[i] * 120} r={2.5} fill={c.amber} />
                  </g>
                )
              })}
              <path d={cum.map((v, i) => `${i ? 'L' : 'M'}${40 + (i + 0.5) * (300 / cum.length)},${140 - v * 120}`).join(' ')} fill="none" stroke={c.amber} strokeWidth={1.5} />
              <text x={40} y={12} fontSize={10} fill={c.amber}>● cumulative</text>
              <text x={200} y={168} fontSize={10} textAnchor="middle" fill={c.muted}>component</text>
            </svg>
            <p className="font-mono text-[12.5px] font-num">shown: {pct(cum[proj.dims - 1] ?? 0)} of the variance · hidden: <span className="text-rose">{pct(1 - (cum[proj.dims - 1] ?? 0))}</span></p>
          </Card>
          {space.featureNames && (
            <Card title="what the axes are made of (loadings)">
              <table className="w-full font-mono text-[11.5px] font-num">
                <thead><tr className="text-left text-faint"><th className="font-normal">feature</th><th className="font-normal">PC1</th><th className="font-normal">PC2</th></tr></thead>
                <tbody>{space.featureNames.map((f, i) => <tr key={f}><td>{f}</td>{[0, 1].map((k) => <td key={k} style={{ color: Math.abs(full.components[k][i]) > 0.3 ? 'var(--amber)' : undefined }}>{f3(full.components[k][i])}</td>)}</tr>)}</tbody>
              </table>
              <p className="mt-1 text-[12px] text-muted">Only the hand-built space has named features, so only here can the axes be read.</p>
            </Card>
          )}
          <Faithfulness />
        </div>
      </div>
    </>
  )
}

export function Faithfulness() {
  const { projection } = useEmb()
  const p = projection.preservation
  return (
    <Card title="how faithful is this picture?">
      <dl className="grid grid-cols-[1fr_auto] gap-y-1 font-mono text-[12.5px] font-num">
        <dt className="text-muted">neighbourhood preservation (k = {p.k})</dt><dd className={p.mean > 0.7 ? 'text-ok' : p.mean > 0.4 ? 'text-amber' : 'text-rose'}>{pct(p.mean)}</dd>
        <dt className="text-muted">random layout would score</dt><dd>{pct(p.chance)}</dd>
        <dt className="text-muted">Shepard ρ (global distances)</dt><dd className={projection.shepard.rho > 0.7 ? 'text-ok' : projection.shepard.rho > 0.4 ? 'text-amber' : 'text-rose'}>{f3(projection.shepard.rho)}</dd>
        {projection.explained && <><dt className="text-muted">variance shown</dt><dd>{pct(projection.explained.reduce((a, b) => a + b, 0))}</dd></>}
      </dl>
    </Card>
  )
}

/* ---- 8. UMAP */
function UMAPView() {
  return (
    <>
      <SpaceBar />
      <ProjControls />
      <div className="grid gap-3 @5xl:grid-cols-[1.4fr_1fr]">
        <MapPanel neighbourK={5} force2D />
        <div className="grid gap-3">
          <Faithfulness />
          <Card title="read a UMAP map carefully">
            <ul className="grid list-disc gap-1 pl-5 text-[13px] leading-relaxed text-muted">
              <li>Neighbours inside an island are usually real neighbours.</li>
              <li>The distance between islands, and their sizes, carry little meaning.</li>
              <li>Change the seed: the layout moves, the islands stay. That is what a seeded, stochastic method looks like.</li>
              <li>Change n_neighbors from 3 to 25 and watch local detail trade off against global shape.</li>
            </ul>
          </Card>
        </div>
      </div>
    </>
  )
}

/* ---- 9. distortion */
function DistortionView() {
  const { space, projection, setHighlightPair, highlightPair, setPicks } = useEmb()
  const c = useColors()
  const mis = useMemo(() => misleadingPairs(projection, 6), [projection])
  const sweep = useMemo(() => preservationByDims(space, Math.min(space.dim, 16)), [space])
  const { high, low } = projection.shepard
  const maxH = Math.max(...high, 1e-9), maxL = Math.max(...low, 1e-9)
  const stride = Math.max(1, Math.floor(high.length / 2500))
  const PairList = ({ items, title, tone }: { items: typeof mis.falseFriends; title: string; tone: string }) => (
    <Card title={title}>
      <div className="grid gap-1">
        {items.map((m) => {
          const [i, j] = m.pair
          const on = highlightPair?.[0] === i && highlightPair?.[1] === j
          return (
            <button key={`${i}-${j}`} type="button" className={`grid grid-cols-[1fr_auto_auto] items-baseline gap-2 rounded px-2 py-1 text-left font-mono text-[12px] font-num ${on ? 'bg-raise' : 'hover:bg-panel-2'}`}
              onClick={() => { setHighlightPair(on ? null : [i, j]); setPicks({ focus: space.words[i] }) }}>
              <span className={tone}>{space.words[i]} · {space.words[j]}</span>
              <span className="text-muted" title="cosine in full dimension">cos {f3(1 - m.high)}</span>
              <span className="text-faint" title="distance on screen">screen {m.low.toFixed(2)}</span>
            </button>
          )
        })}
      </div>
    </Card>
  )
  return (
    <>
      <SpaceBar ppmi={false} />
      <ProjControls showDims={false} />
      <div className="grid gap-3 @5xl:grid-cols-[1.3fr_1fr]">
        <MapPanel neighbourK={0} force2D height={400} title="click a pair on the right to draw it in red" />
        <Card title={`Shepard plot · ${high.length.toLocaleString()} word pairs · ρ = ${f3(projection.shepard.rho)}`}>
          <svg viewBox="0 0 360 300" width="100%" style={{ maxWidth: 560, display: 'block' }} role="img" aria-label="Shepard plot of true versus projected distance">
            <line x1={40} y1={260} x2={350} y2={260} stroke={c.lineStrong} /><line x1={40} y1={10} x2={40} y2={260} stroke={c.lineStrong} />
            {high.map((h, i) => i % stride === 0 && <circle key={i} cx={40 + (h / maxH) * 305} cy={260 - (low[i] / maxL) * 245} r={1.3} fill={c.cyan} fillOpacity={0.35} />)}
            {highlightPair && (() => { const k = projection.shepard.pairs.findIndex(([a, b]) => a === highlightPair[0] && b === highlightPair[1]); return k >= 0 ? <circle cx={40 + (high[k] / maxH) * 305} cy={260 - (low[k] / maxL) * 245} r={6} fill="none" stroke={c.rose} strokeWidth={2} /> : null })()}
            <text x={195} y={290} fontSize={10.5} textAnchor="middle" fill={c.muted}>true distance (1 − cos, full {space.dim}-D)</text>
            <text x={12} y={135} fontSize={10.5} textAnchor="middle" fill={c.muted} transform="rotate(-90 12 135)">distance on screen</text>
          </svg>
          <p className="text-[12px] text-muted">A faithful map would put every dot on one rising curve. Vertical spread at a given x means pairs equally far apart in reality are drawn at very different distances.</p>
        </Card>
      </div>
      <div className="grid gap-3 @5xl:grid-cols-3">
        <PairList items={mis.falseFriends} title="false friends · close on screen, far in reality" tone="text-rose" />
        <PairList items={mis.hiddenTwins} title="hidden twins · far on screen, close in reality" tone="text-teal" />
        <Card title="how many dimensions does it take?">
          <LineChart xs={sweep.map((s) => s.dims)} xLabel="PCA dimensions kept" series={[{ name: 'neighbourhoods kept', ys: sweep.map((s) => s.preservation), color: c.cyan }, { name: 'variance kept', ys: sweep.map((s) => s.kept), color: c.amber }]} yMax={1} current={2} />
          <p className="text-[12px] text-muted">Both curves only approach 1 with many more than 2 dimensions. The dashed line marks the 2D picture.</p>
        </Card>
      </div>
    </>
  )
}

/* ---- 10. contextual */
export function ContextView() {
  const { model, ctxSentences, setCtxSentences, ctxToken, setCtxToken } = useEmb()
  const c = useColors()
  const sentences = ctxSentences.split('\n').map((s) => s.trim()).filter(Boolean)
  const tokenCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const s of sentences) for (const id of new Set(model.encodePrompt(s).ids.slice(1))) m.set(model.vocab[id], (m.get(model.vocab[id]) ?? 0) + 1)
    return [...m].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
  }, [ctxSentences, model])
  const token = tokenCounts.some(([t]) => t === ctxToken) ? ctxToken : tokenCounts[0]?.[0]
  const ce = useMemo(() => (token ? contextualEmbeddings(model, sentences, token) : null), [model, ctxSentences, token])
  const preds = useMemo(() => ce?.occ.map((o) => {
    const ids = model.encodePrompt(o.text).ids
    const tr = model.forward(ids)
    const V = tr.logits.cols
    const p = softmax(tr.logits.data.subarray(o.position * V, (o.position + 1) * V))
    const t = topk(p, 1)[0]
    return { tok: model.vocab[t].replace('▁', '·'), p: p[t] }
  }) ?? [], [ce, model])
  const meanOff = ce?.layers.map((l) => {
    const n = l.vecs.length
    let s = 0, k = 0
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { s += cosine(l.vecs[i], l.vecs[j]); k++ }
    return k ? s / k : 1
  }) ?? []
  return (
    <>
      <Card title="sentences (one per line)" right={<span className="sim-badge">trained model · Observatory-Nano activations</span>}>
        <div className="grid gap-3 @3xl:grid-cols-[1fr_minmax(0,320px)]">
          <div>
            <label htmlFor="ctx-sents" className="sr-only">Sentences</label>
            <textarea id="ctx-sents" rows={6} className="w-full px-3 py-2 font-mono text-[13px]" value={ctxSentences} onChange={(e) => setCtxSentences(e.target.value)} spellCheck={false} />
            <p className="mt-1 text-[11.5px] text-faint">The model knows only its toy world (animals, foods, capitals, colours, counting). Letters outside a–z are dropped.</p>
          </div>
          <div>
            <div className="eyebrow mb-1">token that occurs in ≥ 2 sentences</div>
            <div className="flex flex-wrap gap-1">{tokenCounts.slice(0, 18).map(([t, n]) => <button key={t} type="button" className="chip" data-selected={t === token} onClick={() => setCtxToken(t)}>{t.replace('▁', '·')}<span className="id">{n}×</span></button>)}</div>
            {!tokenCounts.length && <p className="text-[13px] text-rose">No token appears in two sentences yet.</p>}
          </div>
        </div>
      </Card>
      {ce && ce.occ.length >= 2 && (
        <>
          <div className="grid gap-3 @5xl:grid-cols-5 @3xl:grid-cols-3">
            {ce.layers.map((l, k) => {
              const n = l.vecs.length
              const M = mat(n, n, l.vecs.flatMap((a) => l.vecs.map((b) => cosine(a, b))))
              return (
                <Card key={l.name} title={`${k + 1} · ${l.name}`}>
                  <Heatmap m={M} rowLabels={ce.occ.map((o) => o.label)} maxAbs={1} showValues={n <= 6} digits={2} cell={n <= 6 ? 38 : 16} ariaLabel={`cosine between occurrences, ${l.name}`} />
                  <p className="mt-1 font-mono text-[11px] text-muted font-num">mean off-diagonal cos {f3(meanOff[k])}</p>
                </Card>
              )
            })}
          </div>
          <div className="grid gap-3 @4xl:grid-cols-[1fr_1.2fr]">
            <Card title="mean similarity between occurrences, by depth">
              <LineChart xs={ce.layers.map((_, i) => i)} xFmt={(i) => ['W_E', '+pos', 'blk 1', 'blk 2', 'LN_f'][i]} xLabel="depth in the network" series={[{ name: 'mean cos', ys: meanOff, color: c.cyan }]} yMax={1} />
            </Card>
            <Card title={`the ${ce.occ.length} occurrences of “${token!.replace('▁', '·')}” and what the model predicts next`}>
              <table className="w-full font-mono text-[12px] font-num">
                <thead><tr className="text-left text-faint"><th className="font-normal">id</th><th className="font-normal">sentence</th><th className="font-normal">pos</th><th className="font-normal">next-token guess</th></tr></thead>
                <tbody>{ce.occ.map((o, i) => <tr key={o.label} className="border-b border-line/50"><td>{o.label}</td><td className="max-w-[260px] truncate">{o.text}</td><td>{o.position}</td><td><span className="text-amber">{preds[i]?.tok}</span> {pct(preds[i]?.p ?? 0)}</td></tr>)}</tbody>
              </table>
              <p className="mt-2 text-[12px] text-muted">Occurrences that must predict different next words end up with different vectors after block 2.</p>
            </Card>
          </div>
        </>
      )}
      {ce && ce.occ.length < 2 && <Card title="need two occurrences"><p className="text-muted">Add another sentence containing this token.</p></Card>}
    </>
  )
}

/* ---- 11. compare */
function CompareStage() {
  return <ComparePanelLazy />
}
const ComparePanelLazy = lazy(() => import('./panels').then((m) => ({ default: m.ComparePanel })))

export { STAGES, euclid }
