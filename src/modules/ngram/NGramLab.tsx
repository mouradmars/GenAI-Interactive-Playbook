/** Module 08 · N-gram language models. */
import { Compass, Download, GraduationCap, SplitSquareHorizontal, Target, Route } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Timeline } from '../../components/Timeline'
import { useHotkeys, Workspace } from '../../components/Shell'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import {
  BOS, METHOD_LABEL, NGramModel, defaultConfig, distSum, makeScorer, makeChallenges, scoreSentences, type Method, type ModelConfig, type Order,
} from '../../engine/ngram'
import { NGRAM_EXAMPLES } from './examples'
import { STAGES } from './lesson'
import { NGCtx, fmtPP, type NGLab, type NGTab } from './context'
import { GuidedStage } from './views'
import { Pipeline, stageIndex } from './charts'
import { ABPanel, ChallengePanel, ExplorePanel, LessonPanel, NGInspector } from './panels'

export default function NGramLab() {
  const first = NGRAM_EXAMPLES[0]
  const [corpus, setCorpus] = useState(first.corpus)
  const [test, setTest] = useState(first.test)
  const [heldOut, setHeldOut] = useState(first.heldOut)
  const [exampleId, setExampleId] = useState<string | null>(first.id)
  const [cfg, setCfgRaw] = useState<ModelConfig>({ ...defaultConfig, method: 'kn' })
  const [tab, setTab] = useState<NGTab>('guided')
  const [focus, setFocus] = useState(0)
  const [ctx, setCtx] = useState<string[]>([BOS, 'i'])
  const [seed, setSeed] = useState(1)
  const [results, setResults] = useState<Record<string, { given: string; ok: boolean; revealed: boolean }>>({})
  const [exportOpen, setExportOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)
  const engine = useStepEngine(STAGES.length)
  const { depth } = useApp()
  const stage = STAGES[engine.index]

  const model = useMemo(() => new NGramModel(corpus), [corpus])
  const eff: ModelConfig = useMemo(() => ({ ...cfg, method: (tab === 'guided' && stage.method) || cfg.method }), [cfg, tab, stage])
  const scorer = useMemo(() => makeScorer(model, eff), [model, eff])
  const score = useMemo(() => scoreSentences(scorer, test), [scorer, test])
  const heldScore = useMemo(() => scoreSentences(scorer, heldOut), [scorer, heldOut])
  useEffect(() => { if (focus >= score.tokens.length) setFocus(0) }, [score.tokens.length, focus])

  const setCfg = (p: Partial<ModelConfig>) => setCfgRaw((c) => ({ ...c, ...p }))
  const loadExample = (id: string) => {
    const ex = NGRAM_EXAMPLES.find((e) => e.id === id)!
    setCorpus(ex.corpus); setTest(ex.test); setHeldOut(ex.heldOut); setExampleId(id); setFocus(0)
    const m = new NGramModel(ex.corpus)
    const firstWord = m.sents[0]?.[0] ?? BOS
    setCtx([BOS, firstWord])
  }
  const editCorpus = (s: string) => { setCorpus(s); setExampleId(null) }
  // the guided stage dictates the method; in other modes keep the user's choice in sync
  useEffect(() => { if (tab === 'guided' && stage.method && stage.id !== 'score' && stage.id !== 'zeros') setCfg({ method: stage.method }) }, [stage, tab])

  const goTo = (stageId: string, exId?: string) => { if (exId) loadExample(exId); setTab('guided'); engine.seek(stageIndex(stageId as never)) }

  const lab: NGLab = {
    model, corpus, setCorpus: editCorpus, test, setTest, heldOut, setHeldOut, exampleId, loadExample, cfg, setCfg, eff, scorer, score, heldScore,
    stage, tab, focus, setFocus, ctx, setCtx, seed, setSeed,
  }

  useHotkeys({
    ArrowRight: () => { setTab('guided'); engine.next() }, ArrowLeft: () => { setTab('guided'); engine.prev() }, Space: engine.toggle, Home: engine.reset,
    e: () => setExportOpen(true),
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const payload = (): ExportPayload => {
    const methods = Object.keys(METHOD_LABEL) as Method[]
    const comparison = methods.map((m) => ({
      method: METHOD_LABEL[m],
      orders: ([1, 2, 3] as Order[]).map((o) => {
        const s = makeScorer(model, { ...cfg, method: m, order: o })
        return { order: o, testPP: scoreSentences(s, test).perplexity, heldOutPP: scoreSentences(s, heldOut).perplexity }
      }),
    }))
    const ch = makeChallenges(model, seed).map((q) => ({ question: q.prompt, answer: q.answer, student: results[`${seed}-${q.id}`]?.given ?? null, correct: results[`${seed}-${q.id}`]?.ok ?? false }))
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'N-gram language models', exportedAt: new Date().toISOString(),
      provenance: 'Exact counts and probabilities computed from the corpus below (not a simulation).',
      corpus, test, heldOut, example: exampleId, config: eff, stage: stage.id, mode: tab, depth,
      corpusStats: { sentences: model.sents.length, N: model.N, V: model.V, unigramTypes: model.uni.size, bigramTypes: model.bi.size, trigramTypes: model.tri.size },
      contextExplorer: { history: ctx, sum: distSum(scorer, ctx) },
      testScore: { log2prob: score.log2prob, crossEntropy: score.crossEntropy, perplexity: score.perplexity, tokens: score.tokens.map((t) => ({ history: t.history, token: t.token, word: t.word, p: t.p, log2p: t.log2p, seen: t.seenNgram, parts: t.breakdown.parts })) },
      heldOutPerplexity: heldScore.perplexity, comparison, challenge: { seed, questions: ch },
    }
    const md = [
      '# N-gram language models · experiment record', '',
      `- Corpus: ${model.sents.length} sentences, N = ${model.N} predicted tokens, |V| = ${model.V}${exampleId ? ` (example: ${NGRAM_EXAMPLES.find((e) => e.id === exampleId)!.title})` : ' (custom)'}`,
      `- Model: ${METHOD_LABEL[eff.method]}, order ${eff.order}, k = ${eff.k}, λ = [${eff.lambdas.map((x) => x.toFixed(2)).join(', ')}], d = ${eff.dBackoff}, D = ${eff.dKN}`,
      `- Test text: "${test}"`, '',
      '## Test scoring', '| t | history → token | P | −log₂ P |', '|---|---|---|---|',
      ...score.tokens.map((t) => `| ${t.i + 1} | ${t.history.join(' ')} → ${t.token} | ${t.p.toPrecision(5)} | ${Number.isFinite(t.log2p) ? (-t.log2p).toFixed(4) : '∞'} |`), '',
      `Cross-entropy ${Number.isFinite(score.crossEntropy) ? score.crossEntropy.toFixed(4) : '∞'} bits/token · perplexity ${fmtPP(score.perplexity)} · held-out perplexity ${fmtPP(heldScore.perplexity)}`, '',
      '## Method comparison (test / held-out perplexity)', '| method | unigram | bigram | trigram |', '|---|---|---|---|',
      ...comparison.map((r) => `| ${r.method} | ${r.orders.map((o) => `${fmtPP(o.testPP)} / ${fmtPP(o.heldOutPP)}`).join(' | ')} |`), '',
      `## Challenge (seed ${seed}): ${ch.filter((q) => q.correct).length} / ${ch.length} correct`,
      ...ch.map((q, i) => `${i + 1}. ${q.question} · answer ${Number.isInteger(q.answer) ? q.answer : q.answer.toFixed(5)}${q.student !== null ? ` · student: ${q.student} ${q.correct ? '✓' : '✗'}` : ''}`), '',
      'All values are exact computations on the corpus above.',
    ].join('\n')
    return { json, markdown: md, filenameBase: `ngram-${eff.method}-n${eff.order}-${stage.id}` }
  }

  const tabs: { id: NGTab; label: string; icon: React.ReactNode }[] = [
    { id: 'guided', label: 'Guided', icon: <Route size={15} /> },
    { id: 'explore', label: 'Explore', icon: <Compass size={15} /> },
    { id: 'compare', label: 'Compare', icon: <SplitSquareHorizontal size={15} /> },
    { id: 'challenge', label: 'Challenge', icon: <Target size={15} /> },
    { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]
  const heading = tab === 'guided' ? stage.title : tab === 'explore' ? 'Exploration mode' : tab === 'compare' ? 'Compare two models' : tab === 'challenge' ? 'Student challenge mode' : 'Lesson guide'

  return (
    <NGCtx.Provider value={lab}>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 08 · n-gram language models</span>
                  <span className="sim-badge" title="Counts and probabilities are computed exactly from the corpus">exact computation · your corpus</span>
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
            {tab === 'guided' && <GuidedStage />}
            {tab === 'explore' && <ExplorePanel />}
            {tab === 'compare' && <ABPanel />}
            {tab === 'challenge' && <ChallengePanel results={results} setResults={setResults} />}
            {tab === 'lesson' && <LessonPanel goTo={goTo} />}
          </div>
        }
        inspector={<NGInspector />}
        timeline={<Timeline engine={{ ...engine, seek: (i: number) => { setTab('guided'); engine.seek(i) } }} steps={STAGES} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </NGCtx.Provider>
  )
}
