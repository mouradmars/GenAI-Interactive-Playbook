/** Right panel: the numerical inspector — current stage formula, explanation by depth, and the key numbers. */
import { AnimatePresence, motion } from 'framer-motion'
import { useApp } from '../../sim/appState'
import { approxTokens, bm25, cosine, sparseDot, words } from '../../engine/rag'
import { Tex } from '../../components/viz'
import { METHOD_LABEL, RERANK_LABEL, f3, f4 } from './common'
import { useLab } from './derive'
import { STAGES, type StageId } from './lesson'
import { focusChunk } from './views'
import { useRag } from './store'

function Row({ k, v, testid }: { k: string; v: React.ReactNode; testid?: string }) {
  return <div className="flex items-baseline justify-between gap-3 border-t border-line py-1 first:border-t-0"><span className="text-muted">{k}</span><span className="text-right font-mono" data-testid={testid}>{v}</span></div>
}

export function Inspector({ stage }: { stage: StageId }) {
  const { depth } = useApp()
  const lab = useLab()
  const cfg = useRag((s) => s.cfg), sel = useRag((s) => s.selChunk)
  const st = STAGES.find((s) => s.id === stage)!
  const ix = lab.pipe.index, i = focusChunk(lab, sel), ch = ix.chunks[i], h = lab.run.hits.find((x) => x.i === i)!
  const numbers: [string, React.ReactNode][] = (() => {
    switch (stage) {
      case 'docs': return [['documents', lab.docs.length], ['words', lab.docs.reduce((s, d) => s + words(d.text).length, 0)], ['question', lab.query]]
      case 'chunk': return [['strategy', cfg.chunk.strategy], ['size / overlap', `${cfg.chunk.size} / ${cfg.chunk.overlap}`], ['chunks', lab.pipe.chunks.length], ['step between starts', cfg.chunk.size - cfg.chunk.overlap], ['evidence', lab.question?.gold ? (lab.gold.split ? 'split ✂' : `in #${lab.gold.ids.join(', #')}`) : 'none (unanswerable)']]
      case 'embed': return [['vocabulary |V|', ix.vocab.length], ['chunk', `#${i} (${ch.terms.length} terms)`], ['non-zero TF-IDF', ix.tfidf[i].size], ['LSA dims k', ix.lsa.k], ['energy kept', `${(ix.lsa.explained * 100).toFixed(1)}%`], ['σ₁', f4(ix.lsa.sigma[0])]]
      case 'index': return [['index', cfg.retrieve.index === 'flat' ? 'exact (flat)' : `IVF ${cfg.retrieve.nlist} clusters, ${cfg.retrieve.nprobe} probe(s)`], ['vectors', ix.chunks.length], ['dims', ix.lsa.k], ['searched now', lab.run.hits.filter((x) => x.searched).length]]
      case 'query': return [['query terms', lab.run.q.terms.join(', ') || '—'], ['unknown terms', lab.run.q.unknown.join(', ') || 'none'], ['rewriting', cfg.rewrite ? (lab.run.rewrite.applied.length ? lab.run.rewrite.applied.length + ' rule(s)' : 'no rule matched') : 'off'], ['‖q̂‖', f4(Math.hypot(...lab.run.q.dense))]]
      case 'similarity': return [['chunk', `#${i}`], ['TF-IDF cosine', f4(sparseDot(lab.run.q.tfidf, ix.tfidf[i]))], ['dense cosine', f4(cosine(lab.run.q.dense, ix.lsa.coords[i]))], ['BM25', f4(bm25(ix, lab.run.q, i).score)]]
      case 'retrieve': return [['method', METHOD_LABEL[cfg.retrieve.method]], ['k', cfg.retrieve.k], ['top-k', lab.run.hits.slice(0, cfg.retrieve.k).map((x) => `#${x.i}`).join(' ')], [`#${i} rank`, Number.isFinite(h.rank) ? h.rank : '—'], [`#${i} RRF`, f4(h.fused)]]
      case 'rerank': return [['reranker', RERANK_LABEL[cfg.reranker]], ['pool', cfg.pool], ['final order', lab.run.reranked.map((r) => `#${r.hit.i}`).join(' ')], ['moved', lab.run.reranked.filter((r) => r.before !== r.after).length]]
      case 'prompt': return [['passages included', lab.run.prompt.included.length], ['dropped', lab.run.prompt.dropped.length], ['passage tokens', `${lab.run.prompt.contextTokens} / ${cfg.prompt.budget}`], ['prompt tokens ≈', lab.run.prompt.tokens], ['tokens of #' + i, approxTokens(ch.text)]]
      case 'generate': case 'cite': return [['support', f3(lab.run.answer.support)], ['threshold', cfg.threshold], ['abstained', lab.run.answer.abstained ? 'yes' : 'no'], ['citations', lab.run.answer.used.map((u) => `[${u.cite}]`).join(' ') || '—']]
      case 'failures': return [['config', `${cfg.chunk.size}/${cfg.chunk.overlap} · ${cfg.retrieve.method} · k=${cfg.retrieve.k}`], ['reranker', RERANK_LABEL[cfg.reranker]], ['rewriting', cfg.rewrite ? 'on' : 'off']]
    }
  })()
  return (
    <div className="grid min-w-0 gap-4 text-[13px]">
      <div>
        <div className="eyebrow">numerical inspector</div>
        <AnimatePresence mode="wait">
          <motion.div key={stage} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} className="mt-1 grid gap-2">
            <h2 className="font-display text-[19px] leading-tight" data-testid="rag-inspector-title">{st.title}</h2>
            <div className="overflow-x-auto scroll-thin rounded-lg border border-line bg-panel-2/60 px-3 py-2"><Tex block tex={st.tex} /></div>
            <p className="leading-relaxed text-muted">{st.text}</p>
            {depth === 'beginner' && <p className="rounded-md border-l-2 border-cyan bg-cyan/5 px-2 py-1 text-[12.5px]">{st.beginner}</p>}
            {depth === 'advanced' && <p className="rounded-md border-l-2 border-violet bg-violet/5 px-2 py-1 text-[12.5px]">{st.advanced}</p>}
          </motion.div>
        </AnimatePresence>
      </div>
      <section className="grid" data-testid="rag-numbers">
        <div className="eyebrow mb-1">numbers</div>
        {numbers.map(([k, v]) => <Row key={k} k={k} v={v} />)}
      </section>
      <section className="rounded-lg border border-line bg-panel-2/40 p-2 text-[12px] text-muted">
        <div className="eyebrow mb-1">what is real here</div>
        Retrieval maths is exact and deterministic: TF-IDF, BM25, LSA (a dense embedding trained on this corpus by SVD), k-means IVF, RRF, MMR. The answer generator is an extractive simulation and the closed-book model is scripted; no neural language model runs in this lab.
      </section>
    </div>
  )
}
