/**
 * Bottom learning timeline: transport for the micro-steps of one iteration (forward → loss → backward →
 * update), the timeline mode, and the recorded history of iterations, which can be rewound to any state.
 */
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Pause, Play, RotateCcw } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { useColors } from '../../sim/appState'
import { MODE_LABEL, PHASE_LABEL, microSteps, phaseOf, stepLabel, type Phase, type TimelineMode } from './model'
import { useMlp } from './store'

export function MicroTimeline() {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), micro = useMlp((s) => s.micro), mode = useMlp((s) => s.mode), playing = useMlp((s) => s.playing), speed = useMlp((s) => s.speed)
  const { next, prev, resetMicro, togglePlay, setMicro, setMode, setSpeed, setPlaying } = useMlp((s) => s)
  const history = useMlp((s) => s.history), hIndex = useMlp((s) => s.hIndex)
  const L = cfg.sizes.length - 1
  const steps = useMemo(() => microSteps(L, mode).map((s) => ({ phase: phaseOf(s), ...stepLabel(s, L) })), [L, mode])
  const atEnd = micro >= steps.length - 1

  // playback: advance one micro-step per tick; in full-iteration mode the last step commits and continues
  useEffect(() => {
    if (!playing) return
    if (atEnd && mode !== 'iteration') { setPlaying(false); return }
    const id = setTimeout(() => { next(); setPlaying(true) }, 1000 / speed)
    return () => clearTimeout(id)
  }, [playing, micro, speed, atEnd, mode, next, setPlaying, hIndex])

  const groups: { phase: Phase; start: number; len: number }[] = []
  steps.forEach((s, i) => { const g = groups[groups.length - 1]; if (g?.phase === s.phase) g.len++; else groups.push({ phase: s.phase, start: i, len: 1 }) })
  const phaseCol: Record<Phase, string> = { forward: c.cyan, loss: c.amber, backward: c.rose, update: c.teal }
  const cur = history[hIndex]

  return (
    <div className="flex flex-col gap-2 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1" role="group" aria-label="micro-step playback">
          <button type="button" className="btn icon-btn" onClick={resetMicro} aria-label="Back to the first micro-step (Home)" title="First step (Home)"><RotateCcw size={15} /></button>
          <button type="button" className="btn icon-btn" onClick={prev} disabled={micro === 0} aria-label="Previous micro-step (←)" title="Previous (←)"><ChevronLeft size={17} /></button>
          <button type="button" className="btn btn-primary w-[86px] justify-center" onClick={togglePlay} aria-label={playing ? 'Pause (Space)' : 'Play (Space)'}>{playing ? <><Pause size={15} /> Pause</> : <><Play size={15} /> Play</>}</button>
          <button type="button" className="btn icon-btn" onClick={next} disabled={atEnd && mode !== 'iteration'} aria-label={atEnd && mode === 'iteration' ? 'Apply update and start the next iteration (→)' : 'Next micro-step (→)'} title="Next (→)"><ChevronRight size={17} /></button>
        </div>
        <span className="font-mono text-[12px] text-muted font-num max-sm:text-[11px]" aria-live="polite" data-testid="micro-status">
          step <span className="text-fg">{micro + 1}</span> / {steps.length} · {steps[micro]?.title}
        </span>
        {atEnd && mode === 'iteration' && <button type="button" className="btn text-teal" onClick={next} data-testid="apply-update">Apply update → iteration {cur.iter + 1}</button>}
        <div className="ml-auto flex flex-wrap items-center gap-3">
          <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Timeline mode">
            {(['forward', 'backprop', 'iteration'] as TimelineMode[]).map((m) => (
              <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)} className={`rounded px-2 py-0.5 text-[11.5px] ${mode === m ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{MODE_LABEL[m]}</button>
            ))}
          </div>
          <label className="flex items-center gap-2 font-mono text-[11.5px] text-muted font-num max-sm:hidden">speed {speed.toFixed(1)}/s
            <input type="range" min={0.2} max={6} step={0.1} value={speed} onChange={(e) => setSpeed(+e.target.value)} className="w-20" aria-label="Playback speed" />
          </label>
        </div>
      </div>
      <div className="flex gap-[3px] overflow-x-auto pb-1 scroll-thin">
        {groups.map((g) => (
          <div key={g.start} className="flex min-w-fit flex-col gap-1" style={{ flex: g.len }}>
            <span className="eyebrow truncate pl-0.5" style={{ color: phaseCol[g.phase] }}>{PHASE_LABEL[g.phase]}</span>
            <div className="flex gap-[3px]">
              {steps.slice(g.start, g.start + g.len).map((s, k) => {
                const i = g.start + k, state = i === micro ? 'now' : i < micro ? 'past' : 'next'
                return (
                  <button key={i} type="button" onClick={() => setMicro(i)} title={s.title} aria-label={`Go to micro-step ${i + 1}: ${s.title}`} aria-current={state === 'now' ? 'step' : undefined}
                    className={`h-7 min-w-[42px] flex-1 rounded-md border px-1.5 font-mono text-[11px] transition-colors ${state === 'now' ? 'shadow-[var(--glow)]' : state === 'past' ? 'border-line-strong bg-raise text-fg' : 'border-line bg-panel-2 text-faint'}`}
                    style={state === 'now' ? { borderColor: phaseCol[s.phase], color: phaseCol[s.phase], background: `color-mix(in oklab, ${phaseCol[s.phase]} 14%, transparent)` } : undefined}>
                    {s.short}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
      <HistoryStrip />
    </div>
  )
}

/** Training history: a sparkline of the dataset loss at every recorded state; click or scrub to rewind. */
export function HistoryStrip() {
  const c = useColors()
  const history = useMlp((s) => s.history), hIndex = useMlp((s) => s.hIndex), rewind = useMlp((s) => s.rewind)
  const n = history.length, cur = history[hIndex]
  const W = 600, H = 34
  const losses = history.map((h) => h.loss), max = Math.max(1e-9, ...losses), min = Math.min(...losses)
  const xs = (i: number) => (n === 1 ? W / 2 : (i / (n - 1)) * W), ys = (v: number) => H - 3 - ((v - min) / (max - min || 1)) * (H - 6)
  const d = losses.map((v, i) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join('')
  const epochs = useMemo(() => { const out: number[] = []; history.forEach((h, i) => { if (i && h.epoch !== history[i - 1].epoch) out.push(i) }); return out }, [history])
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-line pt-2">
      <span className="eyebrow">history</span>
      <button type="button" className="btn icon-btn !h-7 !w-7 max-sm:!hidden" onClick={() => rewind(0)} disabled={hIndex === 0} aria-label="Rewind to the initial state"><ChevronsLeft size={14} /></button>
      <button type="button" className="btn icon-btn !h-7 !w-7" onClick={() => rewind(hIndex - 1)} disabled={hIndex === 0} aria-label="Previous recorded state"><ChevronLeft size={14} /></button>
      <div className="relative min-w-[160px] flex-1">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="block h-[34px] w-full" aria-hidden="true">
          {epochs.map((i) => <line key={i} x1={xs(i)} x2={xs(i)} y1={0} y2={H} stroke={c.line} />)}
          <path d={d} fill="none" stroke={c.amber} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
          <line x1={xs(hIndex)} x2={xs(hIndex)} y1={0} y2={H} stroke={c.cyan} strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </svg>
        <input type="range" min={0} max={n - 1} value={hIndex} onChange={(e) => rewind(+e.target.value)} className="absolute inset-0 w-full cursor-pointer opacity-0" aria-label="Rewind to a recorded state" disabled={n === 1} />
      </div>
      <button type="button" className="btn icon-btn !h-7 !w-7" onClick={() => rewind(hIndex + 1)} disabled={hIndex >= n - 1} aria-label="Next recorded state"><ChevronRight size={14} /></button>
      <button type="button" className="btn icon-btn !h-7 !w-7 max-sm:!hidden" onClick={() => rewind(n - 1)} disabled={hIndex >= n - 1} aria-label="Jump to the latest state"><ChevronsRight size={14} /></button>
      <span className="font-mono text-[11.5px] text-muted font-num max-sm:basis-full max-sm:text-[10.5px]" data-testid="history-status">
        state {hIndex + 1}/{n} · iteration {cur.iter} · epoch {cur.epoch} · loss <span className="text-amber">{cur.loss.toFixed(4)}</span> · {cur.note}
      </span>
      {hIndex < n - 1 && <span className="text-[11px] text-violet">viewing the past: a new step starts a new branch</span>}
    </div>
  )
}
