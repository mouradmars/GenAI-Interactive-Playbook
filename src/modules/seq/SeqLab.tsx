/** Module 09 · RNN, LSTM & Transformer — Interactive Laboratory. */
import { motion } from 'framer-motion'
import { Compass, Download, GraduationCap, Route, SplitSquareHorizontal, Target } from 'lucide-react'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Timeline } from '../../components/Timeline'
import { useHotkeys, Workspace } from '../../components/Shell'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp, useColors } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import {
  ARCH_LABEL, SYMBOLS, TASK, backward, cost, defaultSim, forward, hiddenOf, makeSeqQuestions, occlusion, parseSeq, profileAgreement, simParams, trained, vnorm,
  type Arch, type Objective, type Regime, type SimConfig,
} from '../../engine/seq'
import { STAGES } from './lesson'
import { DEFAULT_SEQ, SEQ_EXAMPLES } from './examples'
import { SeqCtx, f4, pct, sci, useTimeCursor, type ModelRun, type SeqLab, type SeqTab, type Source } from './context'
import { GuidedStage } from './views'
import { ChallengePanel, ComparePanel, ExplorePanel, LessonPanel, SeqInspector, type Results } from './panels'

export default function SeqLabModule() {
  const engine = useStepEngine(STAGES.length)
  const stage = STAGES[engine.index]
  const { depth } = useApp()
  const [tab, setTab] = useState<SeqTab>('guided')
  const [source, setSource] = useState<Source>('trained')
  const [regime, setRegime] = useState<Regime>('mixed')
  const [sim, setSimRaw] = useState<SimConfig>(defaultSim)
  const [seq, setSeqRaw] = useState(DEFAULT_SEQ)
  const [objective, setObjective] = useState<Objective>('logit')
  const [unit, setUnit] = useState(0)
  const [seed, setSeed] = useState(1)
  const [results, setResults] = useState<Results>({})
  const [exportOpen, setExportOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)

  const ids = useMemo(() => { const p = parseSeq(seq); return p.length ? p.slice(0, 30) : parseSeq(DEFAULT_SEQ) }, [seq])
  const target = ids[0]
  const setSeq = (s: string) => setSeqRaw(s.toLowerCase().replace(/[^abcd]/g, '').slice(0, 30) || DEFAULT_SEQ)
  const setSim = (p: Partial<SimConfig>) => setSimRaw((s) => ({ ...s, ...p }))
  const cursor = useTimeCursor(ids.length)

  const runs = useMemo(() => {
    const out = {} as Record<Arch, ModelRun>
    for (const a of ['rnn', 'lstm', 'attn'] as Arch[]) {
      const params = source === 'trained' ? trained(regime, a).params : simParams(a, sim)
      const trace = forward(params, ids)
      out[a] = { arch: a, params, trace, grads: backward(params, trace, target, objective), pred: trace.probs.indexOf(Math.max(...trace.probs)) }
    }
    return out
  }, [source, regime, sim, ids, target, objective])
  useEffect(() => { if (unit >= hiddenOf(runs.rnn.params)) setUnit(0) }, [runs, unit])

  const openExample = (id: string) => {
    const ex = SEQ_EXAMPLES.find((e) => e.id === id)!
    setSource(ex.source); if (ex.regime) setRegime(ex.regime); if (ex.sim) setSimRaw({ ...defaultSim, ...ex.sim }); setSeq(ex.seq)
    setTab('guided'); engine.seek(STAGES.findIndex((s) => s.id === ex.stage))
  }
  const goTo = (id: string) => { setTab('guided'); engine.seek(STAGES.findIndex((s) => s.id === id)) }

  const lab: SeqLab = { source, setSource, regime, setRegime, sim, setSim, seq: seqTextOf(ids), setSeq, ids, target, objective, setObjective, runs, cursor, unit, setUnit, stage, tab, seed, setSeed }

  useHotkeys({
    ArrowRight: () => { setTab('guided'); engine.next() }, ArrowLeft: () => { setTab('guided'); engine.prev() }, Space: engine.toggle, Home: engine.reset,
    e: () => setExportOpen(true), ']': cursor.next, '[': cursor.prev,
  }, [engine.next, engine.prev, engine.toggle, engine.reset, cursor.t, cursor.T])

  const payload = (): ExportPayload => {
    const archs: Arch[] = ['rnn', 'lstm', 'attn']
    const per = archs.map((a) => {
      const r = runs[a]; const occ = occlusion(r.params, ids, target)
      return {
        model: ARCH_LABEL[a], prediction: SYMBOLS[r.pred], pCorrect: r.trace.probs[target], probs: r.trace.probs,
        gradientNorms: r.grads.dx.map((d) => vnorm(d)), occlusion: occ, cost: cost(a, ids.length, hiddenOf(r.params)),
        attentionLastRow: r.trace.arch === 'attn' ? r.trace.A[ids.length - 1] : undefined,
        attentionVsOcclusion: r.trace.arch === 'attn' ? profileAgreement(r.trace.A[ids.length - 1], occ) : undefined,
        trainedAccuracy: source === 'trained' ? trained(regime, a).accuracy : undefined,
      }
    })
    const qs = makeSeqQuestions(seed).map((q) => { const r = results[`${seed}-${q.id}`]; return { question: q.prompt, answer: q.options ? q.options[q.answer] : q.answer, student: r ? (q.options ? q.options[+r.given] : r.given) : null, correct: r?.ok ?? false } })
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'RNN, LSTM & Transformer — Interactive Laboratory', exportedAt: new Date().toISOString(),
      provenance: source === 'trained' ? `trained models (regime ${regime}) from training/train_seq.py; ${TASK.optimizer}, ${TASK.steps} steps` : 'educational simulation: seeded random weights, not trained',
      task: 'recall the first symbol', sequence: seqTextOf(ids), target: SYMBOLS[target], source, regime, simulation: source === 'sim' ? sim : null, gradientObjective: objective,
      stage: stage.id, mode: tab, depth, timeCursor: cursor.t + 1, models: per, challenge: { seed, questions: qs },
    }
    const md = [
      '# RNN, LSTM & Transformer · experiment record', '',
      `- Source: ${json.provenance}`,
      `- Sequence: ${seqTextOf(ids)} (T = ${ids.length}); target = first symbol “${SYMBOLS[target]}”`,
      `- Gradient objective: ${objective === 'loss' ? 'training loss −log p(target)' : 'target logit z_target'}`, '',
      '| model | prediction | p(correct) | ‖∂/∂x₁‖ | ‖∂/∂x_T‖ | dependent steps | path x₁→out | MACs |', '|---|---|---|---|---|---|---|---|',
      ...per.map((p) => `| ${p.model} | ${p.prediction} | ${pct(p.pCorrect)} | ${sci(p.gradientNorms[0])} | ${sci(p.gradientNorms[ids.length - 1])} | ${p.cost.sequentialSteps} | ${p.cost.pathLength} | ${p.cost.macs.toLocaleString()} |`), '',
      `Attention (last row): ${per[2].attentionLastRow?.map((w) => w.toFixed(3)).join(', ')}`,
      `Occlusion (attention model): ${per[2].occlusion.map((w) => w.toFixed(3)).join(', ')}`,
      `Attention ↔ occlusion: ${per[2].attentionVsOcclusion?.sameTop ? 'same top position' : 'different top position'}, correlation ${f4(per[2].attentionVsOcclusion?.pearson ?? 0)}`, '',
      ...(source === 'trained' ? ['## Trained accuracy (Python evaluation, 400 sequences per length)', '| T | RNN | LSTM | attention |', '|---|---|---|---|',
        ...[2, 5, 10, 15, 20, 25, 30].map((T) => `| ${T} | ${archs.map((a) => pct(trained(regime, a).accuracy.find((x) => x.T === T)!.acc)).join(' | ')} |`), ''] : []),
      `## Challenge (seed ${seed}): ${qs.filter((q) => q.correct).length} / ${qs.length} correct`,
      ...qs.map((q, i) => `${i + 1}. ${q.question} · answer ${typeof q.answer === 'number' ? q.answer.toFixed(4) : q.answer}${q.student !== null ? ` · student: ${q.student} ${q.correct ? '✓' : '✗'}` : ''}`), '',
      source === 'trained' ? 'All values computed from the trained weights.' : 'Note: random-weight simulation, not trained models.',
    ].join('\n')
    return { json, markdown: md, filenameBase: `seq-${source}-${regime}-${stage.id}` }
  }

  const tabs: { id: SeqTab; label: string; icon: React.ReactNode }[] = [
    { id: 'guided', label: 'Guided', icon: <Route size={15} /> }, { id: 'explore', label: 'Explore', icon: <Compass size={15} /> },
    { id: 'compare', label: 'Compare', icon: <SplitSquareHorizontal size={15} /> }, { id: 'challenge', label: 'Challenge', icon: <Target size={15} /> },
    { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]
  const heading = tab === 'guided' ? stage.title : tab === 'explore' ? 'Exploration mode' : tab === 'compare' ? 'Compare configurations' : tab === 'challenge' ? 'Student challenge mode' : 'Lesson guide'

  return (
    <SeqCtx.Provider value={lab}>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 09 · rnn, lstm & transformer</span>
                  {source === 'trained' ? <span className="sim-badge">trained models · {regime}</span> : <span className="sim-badge toy">educational simulation · random weights</span>}
                </div>
                <h1 className="stage-title mt-1 font-display text-[26px] leading-tight">{heading}</h1>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Lab modes">
                  {tabs.map((t) => (
                    <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>
                  ))}
                </div>
                <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
              </div>
            </div>
            {tab === 'guided' && <Pipeline index={engine.index} onSeek={engine.seek} />}
            <Suspense fallback={<div className="text-muted">Loading…</div>}>
              {tab === 'guided' && <GuidedStage />}
              {tab === 'explore' && <ExplorePanel />}
              {tab === 'compare' && <ComparePanel />}
              {tab === 'challenge' && <ChallengePanel results={results} setResults={setResults} />}
              {tab === 'lesson' && <LessonPanel goTo={goTo} openExample={openExample} />}
            </Suspense>
          </div>
        }
        inspector={<SeqInspector />}
        timeline={<Timeline engine={{ ...engine, seek: (i: number) => { setTab('guided'); engine.seek(i) } }} steps={STAGES} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </SeqCtx.Provider>
  )
}

const seqTextOf = (ids: number[]) => ids.map((i) => SYMBOLS[i]).join('')

/** Animated pipeline of the guided stages. */
function Pipeline({ index, onSeek }: { index: number; onSeek: (i: number) => void }) {
  const c = useColors()
  const n = STAGES.length
  const W = 1100, step = (W - 80) / (n - 1), y = 44
  const x = (i: number) => 40 + i * step
  const groups: { name: string; a: number; b: number }[] = []
  STAGES.forEach((s, i) => { const g = groups[groups.length - 1]; if (g?.name === s.group) g.b = i; else groups.push({ name: s.group, a: i, b: i }) })
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg viewBox={`0 0 ${W} 92`} width="100%" style={{ minWidth: 640 }} role="img" aria-label="sequence lab pipeline" data-snapshot="skip">
        {groups.map((g) => (
          <g key={g.name}>
            <line x1={x(g.a) - 14} x2={x(g.b) + 14} y1={12} y2={12} stroke={c.lineStrong} />
            <text x={(x(g.a) + x(g.b)) / 2} y={9} fontSize={10} textAnchor="middle" fill={c.faint} fontFamily="JetBrains Mono, monospace" letterSpacing="1.2">{g.name.toUpperCase()}</text>
          </g>
        ))}
        <line x1={x(0)} x2={x(n - 1)} y1={y} y2={y} stroke={c.line} strokeWidth={3} />
        <motion.line x1={x(0)} y1={y} y2={y} stroke={c.rose} strokeWidth={3} animate={{ x2: x(index) }} transition={{ duration: 0.5 }} />
        {index > 0 && [0, 1, 2].map((k) => (
          <motion.circle key={`${index}-${k}`} r={3} cy={y} fill={c.amber} initial={{ cx: x(Math.max(0, index - 1)), opacity: 0 }} animate={{ cx: x(index), opacity: [0, 1, 0] }}
            transition={{ duration: 1.2, delay: k * 0.35, repeat: Infinity, repeatDelay: 0.6 }} />
        ))}
        {STAGES.map((s, i) => (
          <g key={s.id} onClick={() => onSeek(i)} style={{ cursor: 'pointer' }} role="button" aria-label={`Stage ${i + 1}: ${s.title}`}>
            <circle cx={x(i)} cy={y} r={i === index ? 12 : 8} fill={i <= index ? (i === index ? c.rose : c.teal) : c.panel2} stroke={i === index ? c.rose : c.lineStrong} strokeWidth={1.5} />
            {i === index && <circle cx={x(i)} cy={y} r={18} fill="none" stroke={c.rose} strokeOpacity={0.35} />}
            <text x={x(i)} y={y + 4} fontSize={10} textAnchor="middle" fill={i <= index ? c.bg : c.muted} fontFamily="JetBrains Mono, monospace">{i + 1}</text>
            <text x={x(i)} y={y + 34} fontSize={11} textAnchor="middle" fill={i === index ? c.rose : c.muted} fontFamily="JetBrains Mono, monospace">{s.short}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}
