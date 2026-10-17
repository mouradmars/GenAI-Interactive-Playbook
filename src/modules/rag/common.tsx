/** Shared UI pieces for module 11: document colours, the animated pipeline strip, small formatters, a controls panel. */
import { motion, useReducedMotion } from 'framer-motion'
import { useColors } from '../../sim/appState'
import { CORPORA, corpusById } from '../../data/rag-corpora'
import type { Method, PipelineConfig, Reranker } from '../../engine/rag'
import { STAGES, type StageId } from './lesson'
import { useRag } from './store'

export const f3 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) < 5e-13 ? 0 : v).toFixed(3) : '—')
export const f4 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) < 5e-13 ? 0 : v).toFixed(4) : '—')
export const pct = (v: number) => `${(v * 100).toFixed(0)}%`

export function useDocColors(docIds: string[]) {
  const c = useColors()
  const pal = [c.cyan, c.amber, c.violet, c.teal, c.rose, '#8fb3ff', '#b6e36a', '#f59ad1', '#ffd166', '#7dd3fc']
  return (id: string) => pal[Math.max(0, docIds.indexOf(id)) % pal.length]
}

const NODES: { id: StageId; label: string }[] = [
  { id: 'docs', label: 'documents' }, { id: 'chunk', label: 'chunks' }, { id: 'embed', label: 'vectors' }, { id: 'index', label: 'index' },
  { id: 'query', label: 'query' }, { id: 'similarity', label: 'similarity' }, { id: 'retrieve', label: 'top-k' }, { id: 'rerank', label: 'rerank' },
  { id: 'prompt', label: 'prompt' }, { id: 'generate', label: 'answer' }, { id: 'cite', label: 'citations' }, { id: 'failures', label: 'evaluate' },
]
/** the animated pipeline: offline (ingest/index) path and online (query) path; click a node to jump */
export function PipelineStrip({ active, onSeek }: { active: StageId; onSeek: (i: number) => void }) {
  const c = useColors()
  const reduce = useReducedMotion()
  const W = 1000, H = 120, n = NODES.length, gap = W / n, ai = NODES.findIndex((x) => x.id === active)
  const groupCol = (id: StageId) => { const g = STAGES.find((s) => s.id === id)!.group; return g === 'ingest' ? c.teal : g === 'index' ? c.violet : g === 'retrieve' ? c.cyan : g === 'generate' ? c.amber : c.rose }
  return (
    <div className="overflow-x-auto scroll-thin rounded-xl border border-line bg-panel/60 p-2" aria-label="RAG pipeline">
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" style={{ minWidth: 720 }} role="img" aria-label="Pipeline stages">
        <text x={gap * 2} y={14} textAnchor="middle" fontSize={10.5} fill={c.faint} fontFamily="var(--font-mono)">offline: ingest and index</text>
        <text x={gap * 8} y={14} textAnchor="middle" fontSize={10.5} fill={c.faint} fontFamily="var(--font-mono)">online: for every question</text>
        <line x1={gap * 4} x2={gap * 4} y1={20} y2={H - 8} stroke={c.line} strokeDasharray="3 4" />
        {NODES.slice(0, -1).map((_, k) => <line key={k} x1={gap * k + gap / 2 + 26} x2={gap * (k + 1) + gap / 2 - 26} y1={62} y2={62} stroke={k < ai ? groupCol(NODES[k + 1].id) : c.lineStrong} strokeWidth={k < ai ? 2 : 1.2} />)}
        {!reduce && ai > 0 && (
          <motion.circle key={active} r={4} fill={groupCol(active)} initial={{ cx: gap * (ai - 1) + gap / 2, cy: 62 }} animate={{ cx: gap * ai + gap / 2 }} transition={{ duration: 0.8, repeat: Infinity, repeatDelay: 0.6 }} />
        )}
        {NODES.map((nd, k) => {
          const on = k === ai, done = k < ai, col = groupCol(nd.id)
          return (
            <g key={nd.id} transform={`translate(${gap * k + gap / 2}, 62)`} style={{ cursor: 'pointer' }} onClick={() => onSeek(STAGES.findIndex((s) => s.id === nd.id))} data-testid={`pipe-${nd.id}`} role="button" aria-label={`Go to ${nd.label}`}>
              <rect x={-26} y={-20} width={52} height={40} rx={10} fill={on ? col : c.panel} stroke={col} strokeWidth={on ? 2.5 : 1.2} opacity={done || on ? 1 : 0.65} />
              <text y={5} textAnchor="middle" fontSize={11} fill={on ? c.bg : c.fg} fontFamily="var(--font-mono)">{k + 1}</text>
              <text y={38} textAnchor="middle" fontSize={10.5} fill={on ? col : c.muted} fontFamily="var(--font-mono)">{nd.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export const METHOD_LABEL: Record<Method, string> = { dense: 'Dense (LSA cosine)', tfidf: 'TF-IDF cosine', bm25: 'BM25', hybrid: 'Hybrid (dense + BM25)' }
export const RERANK_LABEL: Record<Reranker, string> = { none: 'none', mmr: 'MMR (diversity)', coverage: 'coverage heuristic' }

/** compact parameter editor, used by explore and compare */
export function ConfigEditor({ cfg, set, idp = 'a', compact = false }: { cfg: PipelineConfig; set: (f: (c: PipelineConfig) => PipelineConfig) => void; idp?: string; compact?: boolean }) {
  const lab = 'flex items-center gap-2 text-[12px] text-muted'
  const sel = 'rounded border border-line bg-panel-2 px-1 py-0.5 text-[12px]'
  return (
    <div className={`grid gap-1.5 ${compact ? '' : '@3xl:grid-cols-2'}`} aria-label={`Configuration ${idp.toUpperCase()}`}>
      <label className={lab}>chunk size <span className="w-8 font-mono text-fg">{cfg.chunk.size}</span><input type="range" min={8} max={120} value={cfg.chunk.size} onChange={(e) => set((c) => ({ ...c, chunk: { ...c.chunk, size: +e.target.value, overlap: Math.min(c.chunk.overlap, +e.target.value - 1) } }))} className="flex-1" aria-label={`Chunk size ${idp}`} /></label>
      <label className={lab}>overlap <span className="w-8 font-mono text-fg">{cfg.chunk.overlap}</span><input type="range" min={0} max={Math.max(0, cfg.chunk.size - 1)} value={cfg.chunk.overlap} onChange={(e) => set((c) => ({ ...c, chunk: { ...c.chunk, overlap: +e.target.value } }))} className="flex-1" aria-label={`Overlap ${idp}`} /></label>
      <label className={lab}>chunking<select className={sel} value={cfg.chunk.strategy} onChange={(e) => set((c) => ({ ...c, chunk: { ...c.chunk, strategy: e.target.value as 'fixed' } }))} aria-label={`Chunking strategy ${idp}`}><option value="fixed">fixed word windows</option><option value="sentence">sentence packing</option></select></label>
      <label className={lab}>retrieval<select className={sel} value={cfg.retrieve.method} onChange={(e) => set((c) => ({ ...c, retrieve: { ...c.retrieve, method: e.target.value as Method } }))} aria-label={`Retrieval method ${idp}`}>{(Object.keys(METHOD_LABEL) as Method[]).map((m) => <option key={m} value={m}>{METHOD_LABEL[m]}</option>)}</select></label>
      <label className={lab}>top-k <span className="w-6 font-mono text-fg">{cfg.retrieve.k}</span><input type="range" min={1} max={10} value={cfg.retrieve.k} onChange={(e) => set((c) => ({ ...c, retrieve: { ...c.retrieve, k: +e.target.value } }))} className="flex-1" aria-label={`Top k ${idp}`} /></label>
      <label className={lab}>reranker<select className={sel} value={cfg.reranker} onChange={(e) => set((c) => ({ ...c, reranker: e.target.value as Reranker }))} aria-label={`Reranker ${idp}`}>{(Object.keys(RERANK_LABEL) as Reranker[]).map((m) => <option key={m} value={m}>{RERANK_LABEL[m]}</option>)}</select></label>
      <label className={lab}>LSA dims <span className="w-6 font-mono text-fg">{cfg.dims}</span><input type="range" min={2} max={32} value={cfg.dims} onChange={(e) => set((c) => ({ ...c, dims: +e.target.value }))} className="flex-1" aria-label={`LSA dimensions ${idp}`} /></label>
      <label className={lab}><input type="checkbox" checked={cfg.rewrite} onChange={(e) => set((c) => ({ ...c, rewrite: e.target.checked }))} aria-label={`Query rewriting ${idp}`} />query rewriting (synonym table)</label>
    </div>
  )
}

export function CorpusPicker() {
  const corpusId = useRag((s) => s.corpusId), setCorpus = useRag((s) => s.setCorpus), qid = useRag((s) => s.qid), setQuestion = useRag((s) => s.setQuestion), custom = useRag((s) => s.custom), setCustom = useRag((s) => s.setCustom)
  const corpus = corpusById(corpusId)
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-panel/60 px-3 py-2 text-[12.5px]" aria-label="Corpus and question">
      <label className="flex items-center gap-1 text-muted">knowledge base
        <select className="rounded border border-line bg-panel-2 px-1" value={corpusId} onChange={(e) => setCorpus(e.target.value)} aria-label="Knowledge base">{CORPORA.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}</select>
      </label>
      <label className="flex min-w-0 flex-1 items-center gap-1 text-muted">question
        <select className="min-w-0 flex-1 rounded border border-line bg-panel-2 px-1" value={qid} onChange={(e) => setQuestion(e.target.value)} aria-label="Question">
          {corpus.questions.map((q) => <option key={q.id} value={q.id}>{q.q}</option>)}
          <option value="custom">✎ your own question…</option>
        </select>
      </label>
      {qid === 'custom' && <input value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="type a question" className="min-w-[220px] flex-1 rounded border border-line bg-panel-2 px-2 py-0.5" aria-label="Your question" data-testid="custom-question" />}
    </div>
  )
}
