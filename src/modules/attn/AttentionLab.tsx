/** Module 03 · Attention — scaled dot-product and multi-head attention from scratch. */
import { AnimatePresence, motion } from 'framer-motion'
import { BookOpen, Download, Footprints, FlaskConical, Microscope, Bookmark, RefreshCw, Repeat, Route, Target, Telescope } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useHotkeys, Workspace } from '../../components/Shell'
import { Timeline } from '../../components/Timeline'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import { RoleLegend, Seg, useTokens, useTrace } from './common'
import { Navigator } from './Navigator'
import { Inspector } from './Inspector'
import { PipelineView } from './PipelineView'
import { STAGES, stageById, stageIndex, type StageId } from './stages'
import { FOLLOW_IDS, FOLLOW_STAGE } from './follow'
import { TokensView, EmbedView } from './viewsInput'
import { QKVView, DotView, ScaleView, MaskView } from './viewsScore'
import { SoftmaxView, HeatmapView, AggregateView } from './viewsMix'
import { HeadsView, ConcatView, ProjView, OutputView } from './viewsHeads'
import { MicroscopesTab } from './MicroscopesTab'
import { FollowTab } from './FollowTab'
import { ExperimentsTab } from './ExperimentsTab'
import { ChallengeTab } from './ChallengeTab'
import { ResearchTab } from './ResearchTab'
import { LessonsTab, NotesBar } from './Lessons'
import { exportPayload } from './exportSheet'
import { seekRef } from './seek'
import { attnStore, useAttn, type AttnTab } from './store'
import type { AttnTrace } from '../../engine/attention'

function StageView({ id, trace }: { id: StageId; trace: AttnTrace }) {
  switch (id) {
    case 'tokens': return <TokensView trace={trace} />
    case 'embed': return <EmbedView trace={trace} />
    case 'qkv': return <QKVView trace={trace} />
    case 'dot': return <DotView trace={trace} />
    case 'scale': return <ScaleView trace={trace} />
    case 'mask': return <MaskView trace={trace} />
    case 'softmax': return <SoftmaxView trace={trace} />
    case 'heatmap': return <HeatmapView trace={trace} />
    case 'aggregate': return <AggregateView trace={trace} />
    case 'heads': return <HeadsView trace={trace} />
    case 'concat': return <ConcatView trace={trace} />
    case 'proj': return <ProjView trace={trace} />
    case 'output': return <OutputView trace={trace} />
  }
}

const TABS: { id: AttnTab; label: string; icon: React.ReactNode }[] = [
  { id: 'guided', label: 'Guided pipeline', icon: <Route size={15} /> },
  { id: 'microscopes', label: 'Three microscopes', icon: <Microscope size={15} /> },
  { id: 'follow', label: 'Follow one token', icon: <Footprints size={15} /> },
  { id: 'experiments', label: 'Experiments', icon: <FlaskConical size={15} /> },
  { id: 'challenge', label: 'Challenges', icon: <Target size={15} /> },
  { id: 'research', label: 'Research mode', icon: <Telescope size={15} /> },
  { id: 'lessons', label: 'Lessons', icon: <BookOpen size={15} /> },
]
const HEADINGS: Record<Exclude<AttnTab, 'guided'>, string> = {
  microscopes: 'Three synchronized microscopes', follow: 'Follow one token through attention', experiments: 'Interactive experiments',
  challenge: 'Student challenges', research: 'Advanced research mode', lessons: 'Classroom lessons',
}

export default function AttentionLab() {
  const engine = useStepEngine(STAGES.length)
  const stage = STAGES[engine.index]
  const tab = useAttn((s) => s.tab), setTab = useAttn((s) => s.setTab), followStep = useAttn((s) => s.followStep)
  const trace = useTrace()
  const tokens = useTokens()
  const { presenting } = useApp()
  const [exportOpen, setExportOpen] = useState(false)
  const [replay, setReplay] = useState(0)
  const stageRef = useRef<HTMLDivElement>(null)
  const toGuided = <A extends unknown[]>(f: (...a: A) => void) => (...a: A) => { setTab('guided'); f(...a) }
  useEffect(() => { seekRef.current = (id: StageId) => { setTab('guided'); engine.seek(stageIndex(id)) } }, [engine, setTab])

  useHotkeys({
    ArrowRight: toGuided(engine.next), ArrowLeft: toGuided(engine.prev), Space: engine.toggle, Home: engine.reset,
    e: () => setExportOpen(true), r: () => setReplay((x) => x + 1),
    h: () => { const s = attnStore.getState(); s.select({ h: (s.sel.h + 1) % s.H }) },
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const heading = tab === 'guided' ? stage.title : HEADINGS[tab]
  return (
    <>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 03 · attention</span>
                  {tab === 'research'
                    ? <span className="sim-badge toy">toy model and a labelled real-model section</span>
                    : <span className="sim-badge toy" data-testid="attn-sim-badge">educational simulation · hand-chosen toy weights, not a trained model</span>}
                  <span className="sim-badge">exact float64 · checked against PyTorch</span>
                </div>
                <h1 className="stage-title mt-1 font-display text-[25px] leading-tight" data-testid="attn-heading">{heading}</h1>
                {tab === 'guided' && <p className="text-[12.5px] text-muted">Scaled dot-product and multi-head attention from scratch · step {engine.index + 1} of {STAGES.length}</p>}
              </div>
              <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
            </div>
            <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Lab modes">
              {TABS.map((t) => (
                <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} data-testid={`tab-${t.id}`}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>
              ))}
            </div>
            {presenting && tab === 'guided' && <NotesBar stageId={stage.id} />}
            {!presenting && (tab === 'guided' || tab === 'microscopes') && <ConfigBar />}
            {tab === 'guided' && (
              <div className="grid gap-3 @5xl:grid-cols-[200px_minmax(0,1fr)]">
                <Navigator index={engine.index} onSeek={engine.seek} trace={trace} />
                <div className="@container grid min-w-0 content-start gap-3">
                  <AnimatePresence mode="wait">
                    <motion.div key={`${stage.id}-${replay}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} data-testid={`stage-${stage.id}`} className={presenting ? '[zoom:1.12]' : undefined}>
                      <StageView id={stage.id} trace={trace} />
                    </motion.div>
                  </AnimatePresence>
                  {!presenting && stage.id !== 'output' && (
                    <details className="rounded-xl border border-line bg-panel/60 p-2" open>
                      <summary className="cursor-pointer px-1 text-[12px] text-muted">microscope C · where this operation sits in the computation</summary>
                      <PipelineView trace={trace} stage={stage} onSeek={engine.seek} compact />
                    </details>
                  )}
                </div>
              </div>
            )}
            {tab === 'microscopes' && <MicroscopesTab trace={trace} />}
            {tab === 'follow' && <FollowTab trace={trace} />}
            {tab === 'experiments' && <ExperimentsTab />}
            {tab === 'challenge' && <ChallengeTab trace={trace} />}
            {tab === 'research' && <ResearchTab trace={trace} />}
            {tab === 'lessons' && <LessonsTab />}
            <RoleLegend />
          </div>
        }
        inspector={<Inspector trace={trace} stage={tab === 'follow' ? stageById(FOLLOW_STAGE[FOLLOW_IDS[followStep] ?? 'input']) : tab === 'microscopes' ? stageById('heatmap') : stage} />}
        timeline={<Timeline engine={{ ...engine, seek: toGuided(engine.seek), next: toGuided(engine.next), prev: toGuided(engine.prev) }} steps={STAGES}
          extra={<>
            <label className="flex items-center gap-1 text-[12px] text-muted">jump to
              <select className="rounded border border-line px-1 text-[12px]" value={stage.id} onChange={(e) => { setTab('guided'); engine.seek(stageIndex(e.target.value as StageId)) }} aria-label="Jump to operation">
                {STAGES.map((s, k) => <option key={s.id} value={s.id}>{k + 1}. {s.title}</option>)}
              </select>
            </label>
            <button type="button" className="btn" onClick={() => { setTab('guided'); setReplay((x) => x + 1) }} title="Replay this operation (R)"><Repeat size={14} />Replay</button>
          </>} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={(): ExportPayload => exportPayload(attnStore.getState(), trace, tokens, stage.id)} stageEl={stageRef.current} />
    </>
  )
}

function ConfigBar() {
  const H = useAttn((s) => s.H), causal = useAttn((s) => s.causal), scale = useAttn((s) => s.scale), edited = useAttn((s) => s.paramsEdited || s.xEdited)
  const st = attnStore.getState()
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-line bg-panel/60 px-3 py-2" aria-label="Configuration" data-testid="attn-config">
      <span className="font-mono text-[12px] text-muted">d_model = 4 · batch = 1</span>
      <span className="flex items-center gap-1 text-[12px] text-muted">heads <Seg label="Heads" value={H} onChange={(v) => st.setH(v)} options={[1, 2, 4].map((v) => ({ v, label: String(v), title: `d_k = ${4 / v}` }))} /> <span className="font-mono">d_k = {4 / H}</span></span>
      <span className="flex items-center gap-1 text-[12px] text-muted">mask <Seg label="Causal mask" value={causal ? 'causal' : 'bi'} onChange={(v) => st.setCausal(v === 'causal')} options={[{ v: 'causal', label: 'causal' }, { v: 'bi', label: 'bidirectional' }]} /></span>
      <span className="flex items-center gap-1 text-[12px] text-muted">scaling <Seg label="Scaling" value={scale ? 'on' : 'off'} onChange={(v) => st.setScale(v === 'on')} options={[{ v: 'on', label: '÷√dₖ' }, { v: 'off', label: 'off' }]} /></span>
      <span className="ml-auto flex flex-wrap items-center gap-1">
        <button type="button" className="btn !h-7 text-[12px]" onClick={() => st.saveBaseline('saved state')} title="Save the current state to compare with later"><Bookmark size={12} />save state</button>
        <button type="button" className="btn !h-7 text-[12px]" onClick={st.resetAll} data-testid="reset-all"><RefreshCw size={12} />reset to preset{edited ? ' (edited)' : ''}</button>
      </span>
    </div>
  )
}
