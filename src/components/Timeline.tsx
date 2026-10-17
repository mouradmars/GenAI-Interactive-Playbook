/** Bottom transport: step/play controls, speed, and a scrub timeline grouped by stage. */
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import type { ReactNode } from 'react'
import type { StepEngine } from '../sim/stepEngine'

export function Timeline({ engine, steps, extra }: { engine: StepEngine; steps: { short: string; group: string; title: string }[]; extra?: ReactNode }) {
  const groups: { name: string; start: number; len: number }[] = []
  steps.forEach((s, i) => {
    const g = groups[groups.length - 1]
    if (g?.name === s.group) g.len++
    else groups.push({ name: s.group, start: i, len: 1 })
  })
  return (
    <div className="flex flex-col gap-2 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1" role="group" aria-label="playback">
          <button type="button" className="btn icon-btn" onClick={engine.reset} aria-label="Reset (Home)" title="Reset (Home)"><RotateCcw size={15} /></button>
          <button type="button" className="btn icon-btn" onClick={engine.prev} disabled={engine.index === 0} aria-label="Step backward (←)" title="Step backward (←)"><ChevronLeft size={17} /></button>
          <button type="button" className="btn btn-primary w-[86px] justify-center" onClick={engine.toggle} aria-label={engine.playing ? 'Pause (Space)' : 'Play (Space)'} title="Play / pause (Space)">
            {engine.playing ? <><Pause size={15} /> Pause</> : <><Play size={15} /> Play</>}
          </button>
          <button type="button" className="btn icon-btn" onClick={engine.next} disabled={engine.index >= engine.count - 1} aria-label="Step forward (→)" title="Step forward (→)"><ChevronRight size={17} /></button>
        </div>
        <span className="font-mono text-[12px] text-muted font-num" aria-live="polite">
          step <span className="text-fg">{engine.index + 1}</span> / {engine.count} · {steps[engine.index]?.title}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-3">
          {extra}
          <label className="flex items-center gap-2 font-mono text-[11.5px] text-muted font-num" htmlFor="speed">
            speed {engine.speed.toFixed(1)}/s
            <input id="speed" type="range" min={0.2} max={4} step={0.1} value={engine.speed} onChange={(e) => engine.setSpeed(+e.target.value)} className="w-24" />
          </label>
        </div>
      </div>
      <div className="relative">
        <div className="flex gap-[3px] overflow-x-auto pb-1 scroll-thin">
          {groups.map((g) => (
            <div key={g.start} className="flex min-w-fit flex-col gap-1" style={{ flex: g.len }}>
              <span className="eyebrow truncate pl-0.5">{g.name}</span>
              <div className="flex gap-[3px]">
                {steps.slice(g.start, g.start + g.len).map((s, k) => {
                  const i = g.start + k
                  const state = i === engine.index ? 'now' : i < engine.index ? 'past' : 'next'
                  return (
                    <button key={i} type="button" onClick={() => engine.seek(i)} title={s.title} aria-label={`Go to step ${i + 1}: ${s.title}`} aria-current={state === 'now' ? 'step' : undefined}
                      className={`h-7 min-w-[46px] flex-1 rounded-md border px-1.5 font-mono text-[10.5px] transition-colors ${
                        state === 'now' ? 'border-cyan bg-cyan/15 text-cyan shadow-[var(--glow)]' : state === 'past' ? 'border-line-strong bg-raise text-fg' : 'border-line bg-panel-2 text-faint'}`}>
                      {s.short}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
        <label className="sr-only" htmlFor="scrub">Scrub timeline</label>
        <input id="scrub" type="range" min={0} max={engine.count - 1} value={engine.index} onChange={(e) => engine.seek(+e.target.value)} className="mt-1 w-full" />
      </div>
    </div>
  )
}
