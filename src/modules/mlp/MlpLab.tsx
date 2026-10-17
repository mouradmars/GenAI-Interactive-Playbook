/** Module 04 · Neural Networks & Backpropagation — Interactive Learning Laboratory. */
import { Activity, Box, Download, Grid3x3, Target, Waves, PanelLeftClose, Settings2, FlaskConical, GraduationCap, LineChart as LineIcon, Microscope, Square, TrendingDown } from 'lucide-react'
import { lazy, Suspense, useRef, useState } from 'react'
import { useHotkeys, Workspace } from '../../components/Shell'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { Card } from '../../components/viz'
import { useApp } from '../../sim/appState'
import { gradientCheck, paramCount } from '../../engine/nn'
import { ConfigPanel } from './ConfigPanel'
import { useNet } from './derive'
import { GraphView, selectionName } from './GraphView'
import { GradCheck } from './GradCheck'
import { Inspector } from './Inspector'
import { LessonPanel } from './LessonPanel'
import { LossLab } from './LossLab'
import { MicroTimeline } from './MicroTimeline'
import { EdgeLegend, NetworkView } from './NetworkView'
import { TrainPanel } from './TrainPanel'
import { DigitsLab } from './DigitsLab'
import { ActivationLab } from './ActivationLab'
import { VanishLab } from './VanishLab'
import { ChallengePanel } from './ChallengePanel'
import { isCorrect, makeChallenges } from './challenges'
import { DIGITS, digitTest, isTrainedDigits } from '../../engine/digits'
import { accuracy } from '../../engine/nn'
import { stepLabel } from './model'
import { mlpStore, useMlp, type LabTab } from './store'

const Network3D = lazy(() => import('./Network3D'))

export default function MlpLab() {
  const net = useNet()
  const { depth, presenting } = useApp()
  const tab = useMlp((s) => s.tab), setTab = useMlp((s) => s.setTab), cfg = useMlp((s) => s.cfg), view3d = useMlp((s) => s.view3d), setView3d = useMlp((s) => s.setView3d)
  const selection = useMlp((s) => s.selection)
  const [exportOpen, setExportOpen] = useState(false)
  const [configOpen, setConfigOpen] = useState(() => typeof window === 'undefined' || window.innerWidth > 900)
  const stageRef = useRef<HTMLDivElement>(null)
  // presentation mode: the network gets the room; the configuration stays one click away
  const showConfig = configOpen && !presenting
  const st = () => mlpStore.getState()

  useHotkeys({
    ArrowRight: () => st().next(), ArrowLeft: () => st().prev(), Space: () => st().togglePlay(), Home: () => st().resetMicro(),
    e: () => setExportOpen(true), '[': () => st().rewind(st().hIndex - 1), ']': () => st().rewind(st().hIndex + 1), g: () => st().setView3d(!st().view3d),
  }, [])

  const payload = (): ExportPayload => {
    const s = st(), sample = net.sample
    const check = gradientCheck(s.cfg, s.params, [sample])
    const qs = makeChallenges(s.challenge.seed)
    const digitsInfo = s.cfg.sizes[0] === 64 && s.cfg.sizes[s.cfg.sizes.length - 1] === 10 ? { trainedWeights: isTrainedDigits(s.cfg, s.params), testAccuracy: accuracy(s.cfg, s.params, digitTest()), exportedTestAccuracy: DIGITS.metrics.testAcc } : null
    const maxRel = Math.max(0, ...check.map((r) => (Math.abs(r.analytic) + Math.abs(r.numeric) > 1e-9 ? r.relErr : 0)))
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'Neural Networks & Backpropagation', exportedAt: new Date().toISOString(),
      provenance: 'exact computation in the browser (engine/nn.ts); no trained model involved', preset: s.preset, config: s.cfg, init: s.init, optimizer: s.opt, batchSize: s.batch,
      dataset: s.dataset, sampleIndex: s.sampleIdx, sample, params: s.params, forward: { z: net.trace.z, a: net.trace.a }, loss: net.loss,
      backward: { dA: net.bwd.dA, delta: net.bwd.delta, dW: net.bwd.dW, db: net.bwd.db, outputRule: net.bwd.outputRule },
      gradientCheck: { epsilon: 1e-5, parameters: check.length, maxRelativeError: maxRel }, microStep: stepLabel(net.step, net.L).title, depth,
      history: s.history.map((h) => ({ iteration: h.iter, epoch: h.epoch, loss: h.loss, kind: h.kind })),
      digits: digitsInfo, challenge: { seed: s.challenge.seed, score: qs.filter((q) => s.challenge.answers[q.id]?.ok).length, questions: qs.map((q) => ({ id: q.id, question: q.prompt, answer: q.answer, student: s.challenge.answers[q.id]?.given ?? null, correct: s.challenge.answers[q.id] ? isCorrect(q, s.challenge.answers[q.id].given) : false })) },
    }
    const md = [
      '# Neural network & backpropagation · experiment record', '',
      `- Architecture: ${s.cfg.sizes.join(' → ')} (${s.cfg.acts.join(', ')}), loss ${s.cfg.loss}, ${paramCount(s.cfg.sizes)} parameters`,
      `- Initialization: ${s.init.scheme}, seed ${s.init.seed}, scale ${s.init.scale}`,
      `- Sample: x = (${sample.x.slice(0, 8).map((v) => v.toFixed(3)).join(', ')}${sample.x.length > 8 ? ', …' : ''}), y = (${sample.y.join(', ')})`,
      `- Prediction ŷ = (${net.yhat.map((v) => v.toFixed(4)).join(', ')}), loss ${net.loss.toFixed(6)}`,
      `- Gradient check (ε = 1e-5): ${check.length} parameters, max relative error ${maxRel.toExponential(2)}`,
      `- History: ${s.history.length} recorded states; current iteration ${s.history[s.hIndex].iter}, epoch ${s.history[s.hIndex].epoch}, dataset loss ${s.history[s.hIndex].loss.toFixed(6)}`, '',
      ...(digitsInfo ? [`- Digits: ${digitsInfo.trainedWeights ? 'exported trained weights' : 'modified / retrained weights'}, test accuracy ${(digitsInfo.testAccuracy * 100).toFixed(1)} % on 450 held-out images`] : []),
      `- Challenge (seed ${s.challenge.seed}): ${qs.filter((q) => s.challenge.answers[q.id]?.ok).length} / ${qs.length} correct`, '',
      '## Output-layer error δ', `(${net.bwd.delta[net.L].map((v) => v.toFixed(6)).join(', ')}) via ${net.bwd.outputRule}`, '',
      'All values computed exactly by the engine; the visualization never computes values of its own.',
    ].join('\n')
    return { json, markdown: md, filenameBase: `mlp-${s.cfg.sizes.join('-')}-iter${s.history[s.hIndex].iter}` }
  }

  const tabs: { id: LabTab; label: string; icon: React.ReactNode }[] = [
    { id: 'microscope', label: 'Microscope', icon: <Microscope size={15} /> }, { id: 'check', label: 'Gradient check', icon: <FlaskConical size={15} /> },
    { id: 'loss', label: 'Loss lab', icon: <TrendingDown size={15} /> }, { id: 'train', label: 'Training', icon: <LineIcon size={15} /> },
    { id: 'digits', label: 'Digits', icon: <Grid3x3 size={15} /> }, { id: 'activations', label: 'Activations', icon: <Activity size={15} /> }, { id: 'vanishing', label: 'Vanishing', icon: <Waves size={15} /> },
    { id: 'challenge', label: 'Challenge', icon: <Target size={15} /> }, { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]
  const heading = tab === 'microscope' ? stepLabel(net.step, net.L).heading : tab === 'check' ? 'Gradient checker' : tab === 'loss' ? 'Loss lab' : tab === 'train' ? 'Gradient descent, step by step' : tab === 'digits' ? 'A trained digit classifier' : tab === 'activations' ? 'Activation functions' : tab === 'vanishing' ? 'Vanishing and exploding gradients' : tab === 'challenge' ? 'Student challenge' : 'Lesson guide'

  return (
    <>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 04 · neural networks & backpropagation</span>
                  <span className="sim-badge">exact computation · every value live</span>
                </div>
                <h1 className="stage-title mt-1 font-display text-[24px] leading-tight" data-testid="mlp-heading">{heading}</h1>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Lab views">
                  {tabs.map((t) => (
                    <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>
                  ))}
                </div>
                <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
              </div>
            </div>
            <div className={`grid gap-3 ${showConfig ? '@3xl:grid-cols-[236px_minmax(0,1fr)]' : ''}`}>
              {showConfig ? (
                <aside className="relative rounded-xl border border-line bg-panel/60 p-3 @max-3xl:max-h-[340px] @max-3xl:overflow-y-auto @3xl:max-h-[calc(100vh-230px)] @3xl:overflow-y-auto scroll-thin" aria-label="Configuration panel">
                  <button type="button" className="btn icon-btn absolute right-2 top-2 !h-6 !w-6 !p-0" onClick={() => setConfigOpen(false)} aria-label="Hide configuration panel" title="Hide configuration"><PanelLeftClose size={13} /></button>
                  <ConfigPanel />
                </aside>
              ) : (
                <button type="button" className="btn justify-self-start" onClick={() => setConfigOpen(true)} aria-label="Show configuration panel"><Settings2 size={14} />Configure · {cfg.sizes.join(' → ')} · {cfg.loss.toUpperCase()}</button>
              )}
              <div className="@container grid min-w-0 content-start gap-3">
                {tab === 'microscope' && (
                  <>
                    <Card title={`network · ${cfg.sizes.join(' → ')}`} right={
                      <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Network view">
                        <button type="button" role="radio" aria-checked={!view3d} onClick={() => setView3d(false)} className={`flex items-center gap-1 rounded px-2 py-0.5 text-[11.5px] ${!view3d ? 'bg-raise text-cyan' : 'text-muted'}`}><Square size={12} />2D</button>
                        <button type="button" role="radio" aria-checked={view3d} onClick={() => setView3d(true)} className={`flex items-center gap-1 rounded px-2 py-0.5 text-[11.5px] ${view3d ? 'bg-raise text-cyan' : 'text-muted'}`}><Box size={12} />3D</button>
                      </div>}>
                      {view3d ? <Suspense fallback={<div className="h-[380px] text-muted">Loading 3D…</div>}><Network3D net={net} /></Suspense> : <NetworkView net={net} height={presenting ? 520 : cfg.sizes.some((n) => n > 8) ? 440 : 360} />}
                      <div className="mt-2"><EdgeLegend /></div>
                    </Card>
                    <Card title={`computational graph · ${selection ? selectionName(selection, net.L) : 'select an element'}`}>
                      <GraphView net={net} />
                    </Card>
                  </>
                )}
                {tab === 'check' && <GradCheck />}
                {tab === 'loss' && <LossLab />}
                {tab === 'train' && <TrainPanel />}
                {tab === 'digits' && <DigitsLab />}
                {tab === 'activations' && <ActivationLab />}
                {tab === 'vanishing' && <VanishLab />}
                {tab === 'challenge' && <ChallengePanel />}
                {tab === 'lesson' && <LessonPanel />}
              </div>
            </div>
          </div>
        }
        inspector={<Inspector net={net} />}
        timeline={<MicroTimeline />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </>
  )
}
