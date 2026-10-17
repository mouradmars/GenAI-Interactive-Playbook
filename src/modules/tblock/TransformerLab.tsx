/** Module 02 · Inside the Transformer Block — an interactive mathematical microscope. */
import { AnimatePresence, motion } from 'framer-motion'
import { BookOpen, Boxes, Columns2, Cpu, Download, FlaskConical, Layers, Microscope, RefreshCw, Repeat, Target } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useHotkeys, Workspace } from '../../components/Shell'
import { Timeline } from '../../components/Timeline'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import { shapeTable, type MlpAct } from '../../engine/block'
import { RoleLegend, useTrace } from './common'
import { ArchNavigator, TokenBar } from './Navigator'
import { StageView } from './StageViews'
import { ExplainPanel, Inspector } from './Inspector'
import { ARCH_LABEL, stagesFor, type ArchPreset } from './stages'
import { CompareArch } from './CompareArch'
import { HeadsLab } from './HeadsLab'
import { StackLab } from './StackLab'
import { RealModel } from './RealModel'
import { AdvancedLab } from './AdvancedLab'
import { ChallengePanel } from './ChallengePanel'
import { LessonsPanel, NotesBar } from './Lessons'
import { SENTENCES, tbStore, useTb, type TbTab } from './store'
import { calcSheet } from './exportSheet'
import { seekRef } from './seek'

const ATT_ARC_STAGES = new Set(['dot', 'scores', 'scale', 'mask', 'softmax', 'weights', 'av'])

export default function TransformerLab() {
  const arch = useTb((s) => s.arch), tab = useTb((s) => s.tab), setTab = useTb((s) => s.setTab)
  const stages = useMemo(() => stagesFor(arch), [arch])
  const engine = useStepEngine(stages.length)
  const stage = stages[Math.min(engine.index, stages.length - 1)]
  const trace = useTrace()
  const { presenting } = useApp()
  const [exportOpen, setExportOpen] = useState(false)
  const [replay, setReplay] = useState(0)
  const stageRef = useRef<HTMLDivElement>(null)
  // keep the same stage when switching architecture, where it exists
  const prevArch = useRef(arch)
  useEffect(() => {
    if (prevArch.current === arch) return
    const old = stagesFor(prevArch.current)[engine.index]?.id
    prevArch.current = arch
    const k = stages.findIndex((s) => s.id === old)
    engine.seek(k >= 0 ? k : Math.min(engine.index, stages.length - 1))
  }, [arch]) // eslint-disable-line react-hooks/exhaustive-deps
  // registered with the lesson runner so guided demos can move the timeline
  useEffect(() => { seekRef.current = (id: string) => { setTab('block'); const k = stagesFor(tbStore.getState().arch).findIndex((s) => s.id === id); if (k >= 0) engine.seek(k) } }, [engine, setTab])

  useHotkeys({
    ArrowRight: () => { setTab('block'); engine.next() }, ArrowLeft: () => { setTab('block'); engine.prev() }, Space: engine.toggle, Home: engine.reset,
    e: () => setExportOpen(true), r: () => setReplay((x) => x + 1),
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const payload = (): ExportPayload => {
    const s = tbStore.getState()
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'Inside the Transformer Block', exportedAt: new Date().toISOString(),
      provenance: 'educational simulation: seeded one-decimal toy weights (not trained), computed exactly in float64 by src/engine/block.ts (tested against PyTorch)',
      architecture: ARCH_LABEL[s.arch], config: trace.cfg, tokens: s.tokens, seed: s.seed, interventions: s.iv, stage: stage.id,
      tensors: { X: trace.X, Q: trace.Q, K: trace.K, V: trace.V, heads: trace.heads.map((h) => ({ S: h.S, scaled: h.scaled, mask: h.mask.map((r) => r.map((v) => (v === 0 ? 0 : '-inf'))), A: h.A, O: h.O })), concat: trace.concat, attn: trace.attn, H: trace.H, Z1: trace.Z1, act: trace.act, mlp: trace.mlp, Y: trace.Y },
      shapes: shapeTable(trace).map((x) => ({ name: x.name, shape: x.shape })), params: s.params,
    }
    return { json, markdown: calcSheet(trace, s.tokens, s.arch, s.sel), filenameBase: `transformer-block-${s.arch}-${stage.id}` }
  }

  const tabs: { id: TbTab; label: string; icon: React.ReactNode }[] = [
    { id: 'block', label: 'Block microscope', icon: <Microscope size={15} /> }, { id: 'compare', label: 'Compare A / B', icon: <Columns2 size={15} /> },
    { id: 'heads', label: 'Heads', icon: <Boxes size={15} /> }, { id: 'stack', label: 'Stack', icon: <Layers size={15} /> }, { id: 'real', label: 'Real model', icon: <Cpu size={15} /> },
    { id: 'advanced', label: 'Advanced', icon: <FlaskConical size={15} /> }, { id: 'challenge', label: 'Challenges', icon: <Target size={15} /> }, { id: 'lessons', label: 'Lessons', icon: <BookOpen size={15} /> },
  ]
  const heading = tab === 'block' ? stage.title : { compare: 'Decoder vs encoder blocks', heads: 'Multi-head attention explorer', stack: 'Stacking blocks', real: 'Inside a real trained model', advanced: 'Advanced experiments', challenge: 'Student challenges', lessons: 'Guided lessons' }[tab as Exclude<TbTab, 'block'>]

  return (
    <>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 02 · inside the transformer block</span>
                  {tab === 'real' ? <span className="sim-badge">real trained model · Observatory-Nano</span> : <span className="sim-badge toy" data-testid="toy-badge">educational simulation · seeded toy weights, not trained</span>}
                </div>
                <h1 className="stage-title mt-1 font-display text-[25px] leading-tight" data-testid="tb-heading">{heading}</h1>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="Architecture preset">
                  {(['gpt', 'bert'] as ArchPreset[]).map((a) => (
                    <button key={a} type="button" role="radio" aria-checked={arch === a} onClick={() => tbStore.getState().setArch(a)} className={`rounded-md px-2.5 py-1.5 text-[12px] ${arch === a ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`} title={ARCH_LABEL[a]}>
                      {a === 'gpt' ? 'A · GPT pre-LN' : 'B · BERT post-LN'}
                    </button>
                  ))}
                </div>
                <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
              </div>
            </div>
            <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Views">
              {tabs.map((t) => (
                <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>
              ))}
            </div>
            {presenting && <NotesBar stageId={stage.id} />}
            {tab === 'block' && (
              <>
                {!presenting && <ConfigBar />}
                <ArchNavigator stage={stage} onSeek={engine.seek} />
                <TokenBar trace={trace} showArcs={ATT_ARC_STAGES.has(stage.id)} />
                <AnimatePresence mode="wait">
                  <motion.div key={`${stage.id}-${arch}-${replay}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                    <StageView id={stage.id} trace={trace} />
                  </motion.div>
                </AnimatePresence>
                <ExplainPanel stage={stage} />
              </>
            )}
            {tab === 'compare' && <CompareArch />}
            {tab === 'heads' && <HeadsLab />}
            {tab === 'stack' && <StackLab />}
            {tab === 'real' && <RealModel />}
            {tab === 'advanced' && <AdvancedLab />}
            {tab === 'challenge' && <ChallengePanel />}
            {tab === 'lessons' && <LessonsPanel />}
            <RoleLegend />
          </div>
        }
        inspector={<Inspector trace={trace} stage={stage} />}
        timeline={<Timeline engine={{ ...engine, seek: (i: number) => { setTab('block'); engine.seek(i) }, next: () => { setTab('block'); engine.next() }, prev: () => { setTab('block'); engine.prev() } }} steps={stages}
          extra={<button type="button" className="btn" onClick={() => { setTab('block'); setReplay((x) => x + 1) }} title="Replay this operation (R)"><Repeat size={14} />Replay step</button>} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </>
  )
}


function ConfigBar() {
  const s = useTb((x) => x)
  const step = (label: string, v: number, on: (n: number) => void, min: number, max: number, aria: string) => (
    <label className="flex items-center gap-1 text-[12px] text-muted">{label}
      <button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => on(v - 1)} disabled={v <= min} aria-label={`Decrease ${aria}`}>−</button>
      <span className="w-6 text-center font-mono text-fg" aria-label={aria}>{v}</span>
      <button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => on(v + 1)} disabled={v >= max} aria-label={`Increase ${aria}`}>+</button>
    </label>
  )
  const divisors = [1, 2, 3, 4, 6, 8].filter((h) => h <= s.d && s.d % h === 0)
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-line bg-panel/60 px-3 py-2" aria-label="Configuration">
      <label className="flex items-center gap-1 text-[12px] text-muted">sentence
        <select className="max-w-[200px] rounded border border-line bg-panel-2 px-1" value={s.sentence} onChange={(e) => s.setSentence(+e.target.value)} aria-label="Sentence">{SENTENCES.map((x, k) => <option key={k} value={k}>{x.label}</option>)}</select>
      </label>
      {step('T', s.T, (n) => s.setDims({ T: n }), 2, 10, 'sequence length')}
      {step('d', s.d, (n) => s.setDims({ d: n }), 2, 8, 'model dimension')}
      <label className="flex items-center gap-1 text-[12px] text-muted">heads
        <select className="rounded border border-line bg-panel-2 px-1" value={s.H} onChange={(e) => s.setDims({ H: +e.target.value })} aria-label="Number of heads">{divisors.map((h) => <option key={h} value={h}>{h}</option>)}</select>
        <span className="font-mono text-faint">dₖ = {s.d / s.H}</span>
      </label>
      {step('d_ff', s.dff, (n) => s.setDims({ dff: n }), 2, 32, 'feedforward dimension')}
      <label className="flex items-center gap-1 text-[12px] text-muted">φ
        <select className="rounded border border-line bg-panel-2 px-1" value={s.act} onChange={(e) => s.setAct(e.target.value as MlpAct)} aria-label="Feedforward activation"><option value="gelu">GELU</option><option value="relu">ReLU</option><option value="silu">SiLU</option><option value="swiglu">SwiGLU</option></select>
      </label>
      <label className="flex items-center gap-1 text-[12px] text-muted">seed<input type="number" min={0} value={s.seed} onChange={(e) => s.setSeed(+e.target.value || 0)} className="w-14 rounded border border-line bg-panel-2 px-1 font-mono" aria-label="Weight seed" /></label>
      <button type="button" className="btn !py-0.5 text-[12px]" onClick={s.resetValues}><RefreshCw size={12} />reset values</button>
    </div>
  )
}
