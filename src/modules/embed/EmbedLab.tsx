/** Module 06 · Embedding Space — Interactive Laboratory. */
import { motion } from 'framer-motion'
import { Compass, Download, GraduationCap, Route, SplitSquareHorizontal, Target } from 'lucide-react'
import { Suspense, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { Timeline } from '../../components/Timeline'
import { useHotkeys, Workspace } from '../../components/Shell'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp, useColors } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import { getModel } from '../../engine/model'
import { cosine, norm } from '../../engine/linalg'
import {
  KIND_LABEL, METRIC_LABEL, analogy, defaultPPMI, defaultProj, makeEmbChallenges, nanoSpace, nearest, project, toySpace, trainPPMISpace,
  type Metric, type PPMIConfig, type ProjConfig, type Space,
} from '../../engine/embeddings'
import wordCorpus from '../../data/embed-corpus.txt?raw'
import { STAGES, type SpaceId } from './lesson'
import { DEFAULT_CONTEXT_SENTENCES, EMB_EXAMPLES } from './examples'
import { EmbCtx, f3, f4, pct, type EmbLab, type EmbTab, type Picks } from './context'
import { GuidedStage } from './views'
import { ChallengePanel, ComparePanel, EmbInspector, ExplorePanel, LessonPanel, type ChallengeResults } from './panels'

const DEFAULT_PICKS: Record<SpaceId, Picks> = {
  toy: { focus: 'king', selected: ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'cat', 'dog', 'apple', 'lemon'], analogy: ['man', 'king', 'woman', 'queen'] },
  ppmi: { focus: 'paris', selected: ['paris', 'rome', 'france', 'italy', 'cat', 'dog', 'fish', 'bones', 'three', 'four'], analogy: ['paris', 'france', 'rome', 'italy'] },
  nano: { focus: '·cat', selected: ['·cat', '·dog', '·bird', '·fish', '·bones', '·cheese', '·three', '·four', '·park', '·beach'], analogy: ['·cat', '·fish', '·dog', '·bones'] },
}

function fitPicks(p: Picks, s: Space): Picks {
  const has = (w: string) => s.words.includes(w)
  return {
    focus: has(p.focus) ? p.focus : s.words[0],
    selected: p.selected.filter(has),
    analogy: p.analogy.map((w, i) => (has(w) || (i === 3 && w === '') ? w : s.words[i] ?? '')) as Picks['analogy'],
  }
}

export default function EmbedLab() {
  const model = getModel()
  const engine = useStepEngine(STAGES.length)
  const stage = STAGES[engine.index]
  const { depth } = useApp()
  const [tab, setTab] = useState<EmbTab>('guided')
  const [spaceId, setSpaceId] = useState<SpaceId>(stage.space ?? 'toy')
  const [ppmiCfg, setPPMIRaw] = useState<PPMIConfig>(defaultPPMI)
  const [proj, setProjRaw] = useState<ProjConfig>(defaultProj)
  const [metric, setMetric] = useState<Metric>('cosine')
  const [k, setK] = useState(5)
  const [picksBy, setPicksBy] = useState<Record<SpaceId, Picks>>(DEFAULT_PICKS)
  const [cell, setCell] = useState<[number, number] | null>(null)
  const [highlightPair, setHighlightPair] = useState<[number, number] | null>(null)
  const [ctxSentences, setCtxSentences] = useState(DEFAULT_CONTEXT_SENTENCES)
  const [ctxToken, setCtxToken] = useState('▁is')
  const [seed, setSeed] = useState(1)
  const [vecA, setVecA] = useState<[number, number]>([3, 1])
  const [vecB, setVecB] = useState<[number, number]>([1, 2])
  const [scaleA, setScaleA] = useState(1)
  const [results, setResults] = useState<ChallengeResults>({})
  const [exportOpen, setExportOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)

  const toy = useMemo(() => toySpace(), [])
  const nano = useMemo(() => nanoSpace(model), [model])
  const dPPMI = useDeferredValue(ppmiCfg)
  const ppmi = useMemo(() => trainPPMISpace(wordCorpus, dPPMI), [dPPMI])
  const spaces = useMemo(() => ({ toy, ppmi, nano }), [toy, ppmi, nano])
  const space = spaces[spaceId]
  const dSpace = useDeferredValue(space)
  const dProj = useDeferredValue(proj)
  const projection = useMemo(() => project(dSpace, dProj), [dSpace, dProj])
  const stale = dSpace !== space || dProj !== proj || dPPMI !== ppmiCfg || projection.coords.length !== space.words.length

  const picks = fitPicks(picksBy[spaceId], space)
  const focusIdx = Math.max(0, space.words.indexOf(picks.focus))
  const selIdx = picks.selected.map((w) => space.words.indexOf(w)).filter((i) => i >= 0)
  const setPicks = (p: Partial<Picks>) => setPicksBy((all) => ({ ...all, [spaceId]: { ...fitPicks(all[spaceId], space), ...p } }))
  const setPPMI = (p: Partial<PPMIConfig>) => setPPMIRaw((c) => ({ ...c, ...p }))
  const setProj = (p: Partial<ProjConfig>) => setProjRaw((c) => ({ ...c, ...p }))

  // a guided stage opens with its recommended space and projection; the student may change both afterwards.
  // An example opened from the lesson takes precedence over the stage defaults.
  const exampleOverride = useRef<{ space: SpaceId; method?: 'pca' | 'umap' } | null>(null)
  useEffect(() => {
    if (tab !== 'guided') return
    const ov = exampleOverride.current
    exampleOverride.current = null
    const sp = ov?.space ?? stage.space
    const m = ov?.method ?? stage.method
    if (sp) setSpaceId(sp)
    if (m) setProj({ method: m, ...(stage.id === 'arithmetic' ? { dims: 2 } : {}) })
    setCell(null); setHighlightPair(null)
  }, [stage.id, tab])
  useEffect(() => { setCell(null); setHighlightPair(null) }, [spaceId])

  // keep the safe projection when a projection can't be drawn yet (fewer coords than words while recomputing)
  const shownProjection = projection.coords.length === space.words.length ? projection : project(space, { ...proj, method: 'pca' })

  const openExample = (id: string) => {
    const ex = EMB_EXAMPLES.find((e) => e.id === id)!
    if (ex.window) setPPMI({ window: ex.window })
    setSpaceId(ex.space)
    if (ex.method) setProj({ method: ex.method, dims: 2 })
    if (ex.focus || ex.analogy) setPicksBy((all) => ({ ...all, [ex.space]: { ...all[ex.space], ...(ex.focus ? { focus: ex.focus } : {}), ...(ex.analogy ? { analogy: ex.analogy } : {}) } }))
    exampleOverride.current = { space: ex.space, method: ex.method }
    const target = STAGES.findIndex((s) => s.id === ex.stage)
    if (target === engine.index && tab === 'guided') exampleOverride.current = null // already there: the stage effect will not run
    setTab('guided')
    engine.seek(target)
  }
  // examples may override the stage defaults: re-apply their projection after the stage effect ran
  const goTo = (id: string) => { setTab('guided'); engine.seek(STAGES.findIndex((s) => s.id === id)) }

  const lab: EmbLab = {
    model, spaceId, setSpaceId, space, spaces, ppmiCfg, setPPMI, proj, setProj, projection: shownProjection, stale, metric, setMetric, k, setK,
    picks, setPicks, focusIdx, selIdx, cell, setCell, highlightPair, setHighlightPair, ctxSentences, setCtxSentences, ctxToken, setCtxToken,
    stage, tab, seed, setSeed, vecA, vecB, setVecA, setVecB, scaleA, setScaleA,
  }

  useHotkeys({
    ArrowRight: () => { setTab('guided'); engine.next() }, ArrowLeft: () => { setTab('guided'); engine.prev() }, Space: engine.toggle, Home: engine.reset,
    e: () => setExportOpen(true),
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const payload = (): ExportPayload => {
    const nn = nearest(space, space.vecs[focusIdx], 10, metric, [focusIdx])
    const W = (w: string) => space.words.indexOf(w)
    const [a, b, c, e] = picks.analogy
    const an = [a, b, c].every((w) => W(w) >= 0) && new Set([a, b, c]).size === 3 ? analogy(space, W(a), W(b), W(c), 5, space.kind !== 'simulation', e && W(e) >= 0 ? W(e) : undefined) : null
    const ch = makeEmbChallenges(space, shownProjection, seed).map((q) => {
      const r = results[`${space.id}-${proj.method}-${seed}-${q.id}`]
      return { question: q.prompt, answer: q.options ? q.options[q.answer] : q.answer, student: r ? (q.options ? q.options[Number(r.given)] ?? null : r.given) : null, correct: r?.ok ?? false }
    })
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'Embedding Space — Interactive Laboratory', exportedAt: new Date().toISOString(),
      source: { kind: space.kind, label: KIND_LABEL[space.kind], title: space.title, provenance: space.provenance, words: space.words.length, dims: space.dim, stats: space.stats ?? {} },
      config: { ppmi: spaceId === 'ppmi' ? ppmiCfg : null, projection: proj, metric, k }, mode: tab, stage: stage.id, depth,
      focus: { word: picks.focus, norm: norm(space.vecs[focusIdx]), vector: space.vecs[focusIdx], neighbours: nn.map((x) => ({ word: x.word, score: x.score })) },
      analogy: an ? { a, b, c, expected: e || null, target: an.target, ranked: an.ranked.map((x) => ({ word: x.word, cosine: x.score })), expectedRank: an.expectedRank ?? null } : null,
      selectedCosineMatrix: { words: selIdx.map((i) => space.words[i]), cosine: selIdx.map((i) => selIdx.map((j) => cosine(space.vecs[i], space.vecs[j]))) },
      projection: { method: proj.method, dims: proj.dims, explained: shownProjection.explained ?? null, neighbourhoodPreservation: shownProjection.preservation.mean, chance: shownProjection.preservation.chance, shepardRho: shownProjection.shepard.rho, coords: space.words.map((w, i) => ({ word: w, xy: shownProjection.coords[i] })) },
      challenge: { seed, questions: ch },
    }
    const md = [
      '# Embedding Space · experiment record', '',
      `- Source: **${space.title}** — ${KIND_LABEL[space.kind]}`,
      `- ${space.provenance}`,
      `- ${space.words.length} words × ${space.dim} dimensions${spaceId === 'ppmi' ? ` · window ±${ppmiCfg.window}, d = ${ppmiCfg.dim}, α = ${ppmiCfg.alpha}, min count ${ppmiCfg.minCount}` : ''}`,
      `- Projection: ${proj.method.toUpperCase()} ${proj.dims}D${proj.method === 'umap' ? ` (n_neighbors ${proj.nNeighbors}, min_dist ${proj.minDist}, seed ${proj.seed}, ${proj.nEpochs} epochs)` : ''}`, '',
      `## Nearest neighbours of “${picks.focus}” (${METRIC_LABEL[metric]}, full dimension)`, '| rank | word | score |', '|---|---|---|',
      ...nn.map((x, i) => `| ${i + 1} | ${x.word} | ${f4(x.score)} |`), '',
      ...(an ? [`## Analogy: ${a} : ${b} :: ${c} : ?`, ...an.ranked.map((x, i) => `${i + 1}. ${x.word} (cos ${f4(x.score)})`), e ? `Expected “${e}” ranked ${an.expectedRank ?? 'n/a'}.` : '', ''] : []),
      '## How faithful is the 2D/3D picture?',
      `- Neighbourhood preservation (k = 5): ${pct(shownProjection.preservation.mean)} (chance ${pct(shownProjection.preservation.chance)})`,
      `- Shepard rank correlation ρ: ${f3(shownProjection.shepard.rho)}`,
      ...(shownProjection.explained ? [`- Variance shown: ${pct(shownProjection.explained.reduce((x, y) => x + y, 0))}; hidden: ${pct(1 - shownProjection.explained.reduce((x, y) => x + y, 0))}`] : []), '',
      `## Challenge (seed ${seed}): ${ch.filter((q) => q.correct).length} / ${ch.length} correct`,
      ...ch.map((q, i) => `${i + 1}. ${q.question} · answer ${typeof q.answer === 'number' ? (Number.isInteger(q.answer) ? q.answer : q.answer.toFixed(4)) : q.answer}${q.student !== null ? ` · student: ${q.student} ${q.correct ? '✓' : '✗'}` : ''}`), '',
      space.kind === 'simulation' ? 'Note: these vectors are a hand-built educational simulation, not a trained model.' : 'All values were computed from the trained vectors listed above.',
    ].join('\n')
    return { json, markdown: md, filenameBase: `embedding-${space.id}-${proj.method}-${stage.id}` }
  }

  const tabs: { id: EmbTab; label: string; icon: React.ReactNode }[] = [
    { id: 'guided', label: 'Guided', icon: <Route size={15} /> }, { id: 'explore', label: 'Explore', icon: <Compass size={15} /> },
    { id: 'compare', label: 'Compare', icon: <SplitSquareHorizontal size={15} /> }, { id: 'challenge', label: 'Challenge', icon: <Target size={15} /> },
    { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]
  const heading = tab === 'guided' ? stage.title : tab === 'explore' ? 'Exploration mode' : tab === 'compare' ? 'Compare two pictures' : tab === 'challenge' ? 'Student challenge mode' : 'Lesson guide'

  return (
    <EmbCtx.Provider value={lab}>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 06 · embedding space</span>
                  <span className={`sim-badge ${space.kind === 'simulation' ? 'toy' : ''}`}>{KIND_LABEL[space.kind]}</span>
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
        inspector={<EmbInspector />}
        timeline={<Timeline engine={{ ...engine, seek: (i: number) => { setTab('guided'); engine.seek(i) } }} steps={STAGES} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </EmbCtx.Provider>
  )
}

/** Animated pipeline: text → vectors → similarity → relations → projection → context. */
function Pipeline({ index, onSeek }: { index: number; onSeek: (i: number) => void }) {
  const c = useColors()
  const n = STAGES.length
  const W = 1100, step = (W - 80) / (n - 1), y = 44
  const x = (i: number) => 40 + i * step
  const groups: { name: string; a: number; b: number }[] = []
  STAGES.forEach((s, i) => { const g = groups[groups.length - 1]; if (g?.name === s.group) g.b = i; else groups.push({ name: s.group, a: i, b: i }) })
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg viewBox={`0 0 ${W} 92`} width="100%" style={{ minWidth: 640 }} role="img" aria-label="embedding pipeline" data-snapshot="skip">
        {groups.map((g) => (
          <g key={g.name}>
            <line x1={x(g.a) - 14} x2={x(g.b) + 14} y1={12} y2={12} stroke={c.lineStrong} />
            <text x={(x(g.a) + x(g.b)) / 2} y={9} fontSize={10} textAnchor="middle" fill={c.faint} fontFamily="JetBrains Mono, monospace" letterSpacing="1.2">{g.name.toUpperCase()}</text>
          </g>
        ))}
        <line x1={x(0)} x2={x(n - 1)} y1={y} y2={y} stroke={c.line} strokeWidth={3} />
        <motion.line x1={x(0)} y1={y} y2={y} stroke={c.violet} strokeWidth={3} animate={{ x2: x(index) }} transition={{ duration: 0.5 }} />
        {index > 0 && [0, 1, 2].map((kk) => (
          <motion.circle key={`${index}-${kk}`} r={3} cy={y} fill={c.amber}
            initial={{ cx: x(Math.max(0, index - 1)), opacity: 0 }} animate={{ cx: x(index), opacity: [0, 1, 0] }}
            transition={{ duration: 1.2, delay: kk * 0.35, repeat: Infinity, repeatDelay: 0.6 }} />
        ))}
        {STAGES.map((s, i) => (
          <g key={s.id} onClick={() => onSeek(i)} style={{ cursor: 'pointer' }} role="button" aria-label={`Stage ${i + 1}: ${s.title}`}>
            <circle cx={x(i)} cy={y} r={i === index ? 12 : 8} fill={i <= index ? (i === index ? c.violet : c.teal) : c.panel2} stroke={i === index ? c.violet : c.lineStrong} strokeWidth={1.5} />
            {i === index && <circle cx={x(i)} cy={y} r={18} fill="none" stroke={c.violet} strokeOpacity={0.35} />}
            <text x={x(i)} y={y + 4} fontSize={10} textAnchor="middle" fill={i <= index ? c.bg : c.muted} fontFamily="JetBrains Mono, monospace">{i + 1}</text>
            <text x={x(i)} y={y + 34} fontSize={11} textAnchor="middle" fill={i === index ? c.violet : c.muted} fontFamily="JetBrains Mono, monospace">{s.short}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}
