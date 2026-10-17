/** Guided mode "Follow one token through attention": twelve paused, inspectable steps with rewind and replay. */
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, Pause, Play, Repeat, RotateCcw } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { Card, Tex } from '../../components/viz'
import { useStepEngine } from '../../sim/stepEngine'
import type { AttnTrace, Matrix } from '../../engine/attention'
import { Grid, HeadPicker, tokLabel, useTokens, type Role } from './common'
import { TokenCards } from './viewsInput'
import { FOLLOW_STAGE, followSteps, type FollowMatrix } from './follow'
import { PipelineView } from './PipelineView'
import { stageById } from './stages'
import { attnStore, useAttn } from './store'

export function FollowTab({ trace }: { trace: AttnTrace }) {
  const tokens = useTokens(), sel = useAttn((s) => s.sel), params = useAttn((s) => s.params), followStep = useAttn((s) => s.followStep)
  const steps = useMemo(() => followSteps(trace, sel.i, sel.h, params.Wo), [trace, sel.i, sel.h, params.Wo])
  const eng = useStepEngine(steps.length, followStep)
  useEffect(() => { attnStore.getState().setFollowStep(eng.index) }, [eng.index])
  useEffect(() => { if (followStep !== eng.index) eng.seek(followStep) }, [followStep]) // eslint-disable-line react-hooks/exhaustive-deps
  const s = steps[eng.index]
  const hd = trace.heads[sel.h]
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const kl = tokens.map((_, k) => `k${k + 1}`)
  const dl = (n: number, p = 'c') => Array.from({ length: n }, (_, k) => `${p}${k + 1}`)
  const mats: Record<FollowMatrix, { m: Matrix; role: Role; title: string; cols: string[]; mode?: 'prob'; masked?: boolean }> = {
    X: { m: trace.X, role: 'x', title: 'X', cols: dl(trace.dModel, 'f') },
    Q: { m: hd.Q, role: 'q', title: `Q${sel.h + 1}`, cols: dl(trace.dk) },
    K: { m: hd.K, role: 'k', title: `K${sel.h + 1}`, cols: dl(trace.dk) },
    S: { m: hd.S, role: 'score', title: 'S = QKᵀ', cols: kl },
    Ss: { m: hd.scaled, role: 'score', title: trace.cfg.scale ? 'S / √dₖ' : 'S (unscaled)', cols: kl },
    M: { m: hd.logits, role: 'score', title: 'Z = S̃ + M', cols: kl, masked: true },
    A: { m: hd.A, role: 'prob', title: 'A', cols: kl, mode: 'prob', masked: true },
    V: { m: hd.V, role: 'v', title: `V${sel.h + 1}`, cols: dl(trace.dk) },
    O: { m: hd.O, role: 'out', title: `head ${sel.h + 1} output`, cols: dl(trace.dk) },
    H: { m: trace.concat, role: 'out', title: 'Concat', cols: dl(trace.cfg.H * trace.dk, 'h') },
    out: { m: trace.out, role: 'out', title: 'output = H W_O', cols: dl(trace.dModel, 'f') },
  }
  const mm = mats[s.show]
  return (
    <div className="grid gap-3">
      <Card title="choose the query token and head, then press play (each step pauses for inspection)">
        <div className="flex flex-wrap items-center gap-2"><TokenCards showKey={false} /><HeadPicker /></div>
      </Card>
      <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Follow-the-token controls">
        <button type="button" className="btn icon-btn" onClick={eng.reset} aria-label="Rewind to the start"><RotateCcw size={15} /></button>
        <button type="button" className="btn icon-btn" onClick={eng.prev} disabled={eng.index === 0} aria-label="Previous step"><ChevronLeft size={16} /></button>
        <button type="button" className="btn btn-primary w-[86px] justify-center" onClick={eng.toggle} data-testid="follow-play">{eng.playing ? <><Pause size={14} />Pause</> : <><Play size={14} />Play</>}</button>
        <button type="button" className="btn icon-btn" onClick={eng.next} disabled={eng.index >= steps.length - 1} aria-label="Next step" data-testid="follow-next"><ChevronRight size={16} /></button>
        <button type="button" className="btn" onClick={() => { eng.reset(); setTimeout(eng.toggle, 0) }}><Repeat size={14} />Replay all</button>
        <label className="ml-2 flex items-center gap-1 font-mono text-[11.5px] text-muted">speed {eng.speed.toFixed(1)}/s<input type="range" min={0.2} max={3} step={0.1} value={eng.speed} onChange={(e) => eng.setSpeed(+e.target.value)} className="w-24" aria-label="Follow speed" /></label>
        <span className="font-mono text-[12px] text-muted" aria-live="polite" data-testid="follow-status">step {eng.index + 1} / {steps.length} · {s.title}</span>
      </div>
      <ol className="flex gap-1 overflow-x-auto pb-1 scroll-thin" aria-label="Journey">
        {steps.map((x, k) => (
          <li key={x.id}><button type="button" onClick={() => eng.seek(k)} aria-current={k === eng.index ? 'step' : undefined}
            className={`h-8 whitespace-nowrap rounded-md border px-2 font-mono text-[11px] ${k === eng.index ? 'border-cyan bg-cyan/15 text-cyan' : k < eng.index ? 'border-line-strong text-fg' : 'border-line text-faint'}`}>{k + 1}. {x.title}</button></li>
        ))}
      </ol>
      <AnimatePresence mode="wait">
        <motion.div key={`${s.id}-${sel.i}-${sel.h}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Card title={`${eng.index + 1} · ${s.title} · token ${sel.i + 1} “${tokens[sel.i]}”`}>
            <div className="rounded-lg border border-cyan/40 bg-cyan/5 px-2 py-1 overflow-x-auto scroll-thin"><Tex tex={s.tex} block /></div>
            <div className="mt-2 grid gap-1 overflow-x-auto scroll-thin" data-testid="follow-numbers">{s.numbers.map((l, k) => <Tex key={k} tex={l} block />)}</div>
            <p className="mt-2 text-[12.5px] text-muted"><b className="text-fg">Purpose:</b> {s.purpose}</p>
          </Card>
          <Card title={`${mm.title} · highlighted cells are used in this step`}>
            <Grid m={mm.m} role={mm.role} mode={mm.mode} rowLabels={rl} colLabels={mm.cols} hlCells={s.cells} hlRows={s.rows} masked={mm.masked ? trace.M.map((r) => r.map((v) => v === -Infinity)) : undefined}
              onCell={(i) => attnStore.getState().select({ i })} ariaLabel={`${mm.title} for this step`} testid="follow-grid" />
          </Card>
        </motion.div>
      </AnimatePresence>
      <Card title="microscope C · where this step sits in the computation">
        <PipelineView trace={trace} stage={stageById(FOLLOW_STAGE[s.id])} />
      </Card>
    </div>
  )
}
