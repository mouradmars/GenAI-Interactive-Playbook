/** Module 11 · RAG Pipeline — Interactive Laboratory. */
import { AnimatePresence, motion } from 'framer-motion'
import { Compass, Download, GraduationCap, Route, SplitSquareHorizontal, Target } from 'lucide-react'
import { useRef, useState } from 'react'
import { useHotkeys, Workspace } from '../../components/Shell'
import { Timeline } from '../../components/Timeline'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useStepEngine } from '../../sim/stepEngine'
import { corpusById } from '../../data/rag-corpora'
import { evaluate, summarize } from '../../engine/rag'
import { CorpusPicker, METHOD_LABEL, PipelineStrip, RERANK_LABEL } from './common'
import { useLab } from './derive'
import { Inspector } from './Inspector'
import { STAGES, type StageId } from './lesson'
import { ChallengePanel, ComparePanel, ExplorePanel, LessonPanel } from './panels'
import { StageView } from './views'
import { ragStore, useRag, type RagTab } from './store'
import { makeChallenges, check } from './challenges'

export default function RagLab() {
  const engine = useStepEngine(STAGES.length)
  const stage = STAGES[engine.index]
  const tab = useRag((s) => s.tab), setTab = useRag((s) => s.setTab)
  const lab = useLab()
  const [exportOpen, setExportOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)
  const seek = (id: StageId) => { setTab('guided'); engine.seek(STAGES.findIndex((s) => s.id === id)) }

  useHotkeys({
    ArrowRight: () => { setTab('guided'); engine.next() }, ArrowLeft: () => { setTab('guided'); engine.prev() }, Space: engine.toggle, Home: engine.reset, e: () => setExportOpen(true),
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const payload = (): ExportPayload => {
    const s = ragStore.getState(), corpus = corpusById(s.corpusId)
    const ev = evaluate(lab.pipe, corpus.questions, s.cfg), sum = summarize(ev, corpus.questions)
    const qs = makeChallenges(s.challenge.seed, { pipe: lab.pipe, q: lab.run.q, chunk: lab.run.reranked[0]?.hit.i ?? 0 })
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'RAG Pipeline — Interactive Laboratory', exportedAt: new Date().toISOString(),
      provenance: 'retrieval computed exactly (TF-IDF, BM25, LSA trained on this corpus by SVD, k-means IVF, RRF, MMR); answer generator = extractive simulation; closed-book answers = scripted simulation',
      corpus: corpus.title, extraDocuments: s.extraDocs, question: lab.query, config: s.cfg, stage: stage.id, mode: tab,
      chunks: lab.pipe.chunks.length, rewrite: lab.run.rewrite,
      ranking: lab.run.hits.slice(0, 10).map((h) => ({ chunk: h.i, doc: lab.pipe.chunks[h.i].docTitle, rank: h.rank, dense: h.dense, tfidf: h.tfidf, bm25: h.bm25, rrf: h.fused })),
      reranked: lab.run.reranked.map((r) => ({ chunk: r.hit.i, before: r.before, after: r.after, score: r.rerankScore })),
      prompt: lab.run.prompt.text, answer: lab.run.answer.text, abstained: lab.run.answer.abstained, support: lab.run.answer.support,
      closedBook: lab.question?.closedBook ?? null, evaluation: { ...sum, perQuestion: ev.map((e) => ({ qid: e.qid, goldRank: Number.isFinite(e.goldRank) ? e.goldRank : null, recall: e.recall, split: e.split, inPrompt: e.inPrompt, correct: e.answerCorrect })) },
      challenge: { seed: s.challenge.seed, score: qs.filter((q) => s.challenge.answers[q.id] && check(q, s.challenge.answers[q.id].given)).length, total: qs.length },
    }
    const md = [
      '# RAG laboratory · experiment record', '',
      `- Knowledge base: ${corpus.title}${s.extraDocs.length ? ` (+${s.extraDocs.length} added document(s))` : ''}`,
      `- Question: ${lab.query}${s.cfg.rewrite ? ` → rewritten: ${lab.run.rewrite.text}` : ''}`,
      `- Chunking: ${s.cfg.chunk.strategy}, size ${s.cfg.chunk.size}, overlap ${s.cfg.chunk.overlap} → ${lab.pipe.chunks.length} chunks; LSA k = ${lab.pipe.index.lsa.k}`,
      `- Retrieval: ${METHOD_LABEL[s.cfg.retrieve.method]}, k = ${s.cfg.retrieve.k}, index ${s.cfg.retrieve.index}; reranker ${RERANK_LABEL[s.cfg.reranker]}; budget ${s.cfg.prompt.budget} tokens`, '',
      '| rank | chunk | doc | dense | TF-IDF | BM25 |', '|---|---|---|---|---|---|',
      ...lab.run.hits.slice(0, 6).map((h) => `| ${h.rank} | #${h.i} | ${lab.pipe.chunks[h.i].docTitle} | ${h.dense.toFixed(4)} | ${h.tfidf.toFixed(4)} | ${h.bm25.toFixed(4)} |`), '',
      `**Grounded answer (extractive simulation):** ${lab.run.answer.text}`,
      lab.question ? `**Closed-book (scripted simulation):** ${lab.question.closedBook}` : '', '',
      `## Evaluation on ${corpus.questions.length} questions`, `Recall@${s.cfg.retrieve.k} ${(sum.recallAtK * 100).toFixed(0)}%, MRR ${sum.mrr.toFixed(3)}, answers correct ${(sum.answerAccuracy * 100).toFixed(0)}%, evidence splits ${sum.splits}`, '',
      ...ev.map((e, n) => `- ${corpus.questions[n].q} — gold rank ${Number.isFinite(e.goldRank) ? e.goldRank : '—'}${e.split ? ' (split)' : ''}, ${e.answerCorrect ? 'correct' : 'incorrect'}`),
    ].join('\n')
    return { json, markdown: md, filenameBase: `rag-${s.corpusId}-${s.cfg.retrieve.method}-${s.cfg.chunk.size}` }
  }

  const tabs: { id: RagTab; label: string; icon: React.ReactNode }[] = [
    { id: 'guided', label: 'Guided', icon: <Route size={15} /> }, { id: 'explore', label: 'Explore', icon: <Compass size={15} /> },
    { id: 'compare', label: 'Compare', icon: <SplitSquareHorizontal size={15} /> }, { id: 'challenge', label: 'Challenge', icon: <Target size={15} /> }, { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]
  const heading = tab === 'guided' ? stage.title : { explore: 'Exploration mode', compare: 'Compare two configurations', challenge: 'Student challenge mode', lesson: 'Lesson guide' }[tab as Exclude<RagTab, 'guided'>]

  return (
    <>
      <Workspace
        stage={
          <div className="@container grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef}>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 11 · retrieval-augmented generation</span>
                  <span className="sim-badge">exact retrieval · TF-IDF, BM25, LSA trained on the corpus</span>
                  <span className="sim-badge toy" data-testid="rag-sim-badge">simulated generator · no LLM</span>
                </div>
                <h1 className="stage-title mt-1 font-display text-[25px] leading-tight" data-testid="rag-heading">{heading}</h1>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Lab modes">
                  {tabs.map((t) => <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>)}
                </div>
                <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
              </div>
            </div>
            <CorpusPicker />
            {(tab === 'guided' || tab === 'explore') && <PipelineStrip active={stage.id} onSeek={(i) => { setTab('guided'); engine.seek(i) }} />}
            {tab === 'guided' && (
              <AnimatePresence mode="wait">
                <motion.div key={stage.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                  <StageView id={stage.id} />
                </motion.div>
              </AnimatePresence>
            )}
            {tab === 'explore' && <ExplorePanel />}
            {tab === 'compare' && <ComparePanel />}
            {tab === 'challenge' && <ChallengePanel />}
            {tab === 'lesson' && <LessonPanel seek={seek} />}
          </div>
        }
        inspector={<Inspector stage={stage.id} />}
        timeline={<Timeline engine={{ ...engine, seek: (i: number) => { setTab('guided'); engine.seek(i) } }} steps={STAGES} />}
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </>
  )
}
