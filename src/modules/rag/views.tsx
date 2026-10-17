/** Stage views of the RAG laboratory. All numbers come from engine/rag via the memoized pipeline. */
import { motion } from 'framer-motion'
import { AlertTriangle, BookOpen, CheckCircle2, Plus, Trash2, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { pca } from '../../engine/linalg'
import {
  BM25_B, BM25_K1, STOP, approxTokens, bm25, buildPipeline, cosine, evaluate, fixedChunkCount, probeClusters, scoreAll, sparseDot, stem, summarize, supportRatio,
  topK, verifyCitations, words, type Method, type PipelineConfig,
} from '../../engine/rag'
import { corpusById } from '../../data/rag-corpora'
import { Card, Tex } from '../../components/viz'
import { LineChart } from '../mlp/charts'
import { METHOD_LABEL, RERANK_LABEL, f3, f4, pct, useDocColors } from './common'
import { useEvaluation, useLab, type Lab } from './derive'
import { diagnose } from './diagnose'
import { OBJECTIVES, type StageId } from './lesson'
import { useRag } from './store'

export function StageView({ id }: { id: StageId }) {
  const lab = useLab()
  switch (id) {
    case 'docs': return <DocsView lab={lab} />
    case 'chunk': return <ChunkView lab={lab} />
    case 'embed': return <EmbedView lab={lab} />
    case 'index': return <IndexView lab={lab} />
    case 'query': return <QueryView lab={lab} />
    case 'similarity': return <SimilarityView lab={lab} />
    case 'retrieve': return <RetrieveView lab={lab} />
    case 'rerank': return <RerankView lab={lab} />
    case 'prompt': return <PromptView lab={lab} />
    case 'generate': return <GenerateView lab={lab} />
    case 'cite': return <CiteView lab={lab} />
    case 'failures': return <FailuresView lab={lab} />
  }
}

/** the chunk the inspector/similarity views look at: the selected one, else the top retrieved */
export const focusChunk = (lab: Lab, sel: number | null) => (sel !== null && sel < lab.pipe.chunks.length ? sel : lab.run.reranked[0]?.hit.i ?? 0)
const chunkLabel = (lab: Lab, i: number) => `#${i} ${lab.pipe.chunks[i].docTitle} · part ${lab.pipe.chunks[i].index + 1}`

/* ------------------------------------------------------------------ 1 · documents */
function Highlight({ text, span, color }: { text: string; span: string; color: string }) {
  const k = span ? text.indexOf(span) : -1
  if (k < 0) return <>{text}</>
  return <>{text.slice(0, k)}<mark style={{ background: 'transparent', color, textDecoration: 'underline', textDecorationThickness: 2 }}>{span}</mark>{text.slice(k + span.length)}</>
}
function DocsView({ lab }: { lab: Lab }) {
  const c = useColors()
  const extra = useRag((s) => s.extraDocs), addDoc = useRag((s) => s.addDoc), removeExtra = useRag((s) => s.removeExtra), corpusId = useRag((s) => s.corpusId)
  const col = useDocColors(lab.docs.map((d) => d.id))
  const [title, setTitle] = useState(''), [text, setText] = useState('')
  return (
    <div className="grid gap-3">
      <Card title="learning objectives">
        <ol className="grid list-decimal gap-0.5 pl-5 text-[12.5px] text-muted @3xl:grid-cols-2">{OBJECTIVES.map((o) => <li key={o}>{o}</li>)}</ol>
      </Card>
      <Card title={`${corpusById(corpusId).title} · ${lab.docs.length} documents · ${lab.docs.reduce((s, d) => s + words(d.text).length, 0)} words`}>
        <p className="mb-2 text-[12.5px] text-muted">{corpusById(corpusId).blurb} {lab.question?.gold ? <>The evidence for the current question is <span className="text-amber underline">underlined</span>.</> : null}</p>
        <div className="grid gap-2 @3xl:grid-cols-2">
          {lab.docs.map((d) => (
            <article key={d.id} className="rounded-lg border bg-panel-2/40 p-2.5" style={{ borderColor: col(d.id) }} data-testid={`doc-${d.id}`}>
              <div className="mb-1 flex items-center justify-between gap-2">
                <b className="text-[13px]" style={{ color: col(d.id) }}>{d.title}</b>
                <span className="flex items-center gap-1 font-mono text-[11px] text-faint">{words(d.text).length} words
                  {d.note && <span className="chip text-rose">{d.note}</span>}
                  {extra.some((x) => x.id === d.id) && <button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => removeExtra(d.id)} aria-label={`Remove ${d.title}`}><Trash2 size={12} /></button>}
                </span>
              </div>
              <p className="text-[12.5px] leading-relaxed text-muted"><Highlight text={d.text} span={lab.question?.gold ?? ''} color={c.amber} /></p>
            </article>
          ))}
        </div>
      </Card>
      <Card title="add your own document (ingested into this knowledge base)">
        <div className="grid gap-2">
          <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="title" className="rounded border border-line bg-panel-2 px-2 py-1 text-[13px]" aria-label="New document title" />
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} placeholder="paste a few sentences…" className="rounded border border-line bg-panel-2 px-2 py-1 text-[13px]" aria-label="New document text" data-testid="new-doc-text" />
          <button type="button" className="btn justify-self-start" onClick={() => { addDoc(title, text); setTitle(''); setText('') }} disabled={!text.trim()}><Plus size={13} />Add document</button>
        </div>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 2 · chunking */
function ChunkView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg), sel = useRag((s) => s.selChunk), selectChunk = useRag((s) => s.selectChunk)
  const col = useDocColors(lab.docs.map((d) => d.id))
  const goldSet = new Set(lab.gold.ids)
  const avg = lab.pipe.chunks.reduce((s, ch) => s + ch.words, 0) / lab.pipe.chunks.length
  return (
    <div className="grid gap-3">
      <Card title="chunking parameters">
        <div className="grid gap-2 text-[12.5px] text-muted @3xl:grid-cols-3">
          <label className="flex items-center gap-2">size s <span className="w-8 font-mono text-fg">{cfg.chunk.size}</span><input type="range" min={8} max={120} value={cfg.chunk.size} onChange={(e) => setCfg((x) => ({ ...x, chunk: { ...x.chunk, size: +e.target.value, overlap: Math.min(x.chunk.overlap, +e.target.value - 1) } }))} className="flex-1" aria-label="Chunk size" /></label>
          <label className="flex items-center gap-2">overlap o <span className="w-8 font-mono text-fg">{cfg.chunk.overlap}</span><input type="range" min={0} max={cfg.chunk.size - 1} value={cfg.chunk.overlap} onChange={(e) => setCfg((x) => ({ ...x, chunk: { ...x.chunk, overlap: +e.target.value } }))} className="flex-1" aria-label="Overlap" /></label>
          <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Chunking strategy">
            {(['fixed', 'sentence'] as const).map((st) => <button key={st} type="button" role="radio" aria-checked={cfg.chunk.strategy === st} onClick={() => setCfg((x) => ({ ...x, chunk: { ...x.chunk, strategy: st } }))} className={`flex-1 rounded px-2 py-0.5 text-[12px] ${cfg.chunk.strategy === st ? 'bg-raise text-cyan' : ''}`}>{st === 'fixed' ? 'fixed word windows' : 'sentence packing'}</button>)}
          </div>
        </div>
        <div className="mt-2 flex flex-wrap gap-4 font-mono text-[12px]" data-testid="chunk-stats">
          <span>{lab.pipe.chunks.length} chunks</span><span>mean {f3(avg)} words</span>
          <span>{lab.pipe.chunks.length - lab.docs.length} boundaries inside documents</span>
          {lab.question?.gold && <span className={lab.gold.split ? 'text-rose' : 'text-teal'}>{lab.gold.split ? '✗ evidence split across chunks' : `✓ evidence inside chunk${lab.gold.ids.length > 1 ? 's' : ''} ${lab.gold.ids.map((i) => `#${i}`).join(', ')}`}</span>}
        </div>
      </Card>
      {lab.docs.map((d) => {
        const ws = words(d.text), chs = lab.pipe.chunks.filter((ch) => ch.docId === d.id)
        const member = ws.map((_, w) => chs.filter((ch) => w >= ch.start && w < ch.end).map((ch) => ch.id))
        const goldStart = lab.question?.gold ? words(d.text.slice(0, Math.max(0, d.text.indexOf(lab.question.gold)))).length : -1
        const goldLen = lab.question?.gold && d.text.includes(lab.question.gold) ? words(lab.question.gold).length : 0
        return (
          <Card key={d.id} title={<span style={{ color: col(d.id) }}>{d.title} · {ws.length} words → {chs.length} chunk{chs.length > 1 ? 's' : ''}{cfg.chunk.strategy === 'fixed' ? ` (⌈(${ws.length} − ${cfg.chunk.overlap}) / (${cfg.chunk.size} − ${cfg.chunk.overlap})⌉ = ${fixedChunkCount(ws.length, cfg.chunk.size, cfg.chunk.overlap)})` : ''}</span>}>
            <p className="text-[13px] leading-[1.9]" data-testid={`chunked-${d.id}`}>
              {ws.map((w, k) => {
                const m = member[k], first = chs.find((ch) => ch.start === k)
                const isGold = goldLen > 0 && k >= goldStart && k < goldStart + goldLen
                const on = m.includes(sel ?? -1)
                const bg = m.length > 1 ? `color-mix(in oklab, ${c.amber} 30%, transparent)` : `color-mix(in oklab, ${(m[0] ?? 0) % 2 ? c.violet : c.cyan} ${on ? 34 : 14}%, transparent)`
                return (
                  <span key={k}>
                    {first && <button type="button" onClick={() => selectChunk(first.id)} className="mx-0.5 rounded px-1 align-middle font-mono text-[10px]" style={{ background: goldSet.has(first.id) ? (lab.gold.split ? c.rose : c.amber) : c.lineStrong, color: c.bg }} title={`chunk #${first.id}${goldSet.has(first.id) ? (lab.gold.split ? ' · part of the evidence' : ' · contains the evidence') : ''}`}>#{first.id}</button>}
                    <span onClick={() => selectChunk(m[m.length - 1])} className="cursor-pointer rounded-sm px-[1px]" style={{ background: bg, textDecoration: isGold ? 'underline' : undefined, textDecorationColor: c.amber, textDecorationThickness: 2, outline: on ? `1px solid ${c.cyan}` : undefined }}>{w}</span>{' '}
                  </span>
                )
              })}
            </p>
          </Card>
        )
      })}
      <p className="text-[12px] text-muted">Alternating tints mark chunks; <span className="text-amber">amber</span> words belong to two chunks (overlap); the underline is the gold evidence. A chunk badge in amber contains the whole evidence; rose badges hold only part of a split evidence span.</p>
    </div>
  )
}

/* ------------------------------------------------------------------ 3 · embedding */
function EmbedView({ lab }: { lab: Lab }) {
  const c = useColors()
  const sel = useRag((s) => s.selChunk), selectChunk = useRag((s) => s.selectChunk), cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg)
  const ix = lab.pipe.index, i = focusChunk(lab, sel), ch = ix.chunks[i]
  const counts = new Map<string, number>(); ch.terms.forEach((t) => counts.set(t, (counts.get(t) ?? 0) + 1))
  const rows = [...counts.entries()].map(([t, n]) => { const id = ix.termId.get(t)!; return { t, n, tf: 1 + Math.log(n), df: ix.df[id], idf: ix.idf[id], w: ix.tfidf[i].get(id)! } }).sort((a, b) => b.w - a.w)
  const raw = rows.map((r) => r.tf * r.idf), nrm = Math.hypot(...raw)
  const sig = ix.lsa.sigma, maxS = Math.max(...sig), v = ix.lsa.coords[i], maxV = Math.max(1e-9, ...v.map(Math.abs))
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">chunk
        <select className="max-w-[320px] rounded border border-line bg-panel-2 px-1" value={i} onChange={(e) => selectChunk(+e.target.value)} aria-label="Chunk">{ix.chunks.map((x) => <option key={x.id} value={x.id}>{chunkLabel(lab, x.id)}</option>)}</select>
        <span className="font-mono text-faint">vocabulary: {ix.vocab.length} terms · N = {ix.chunks.length} chunks</span>
      </div>
      <Card title="the chunk and its terms (lowercased, stop words removed, stemmed)">
        <p className="mb-2 text-[12.5px] text-muted">{ch.text}</p>
        <div className="flex flex-wrap gap-1">{ch.terms.map((t, k) => <span key={k} className="rounded border border-line px-1.5 font-mono text-[11px]">{t}</span>)}</div>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Card title="sparse TF-IDF vector (non-zero entries)">
          <div className="max-h-[300px] overflow-auto scroll-thin">
            <table className="w-full font-mono text-[12px]" data-testid="tfidf-table">
              <thead className="sticky top-0 bg-panel text-faint"><tr><th className="px-1 text-left">term</th><th className="px-1 text-right">count</th><th className="px-1 text-right">1+ln tf</th><th className="px-1 text-right">df</th><th className="px-1 text-right">idf</th><th className="px-1 text-right">tf·idf</th><th className="px-1 text-right">÷‖·‖</th></tr></thead>
              <tbody>{rows.map((r) => <tr key={r.t} className="border-t border-line"><td className="px-1">{r.t}</td><td className="px-1 text-right">{r.n}</td><td className="px-1 text-right">{f3(r.tf)}</td><td className="px-1 text-right">{r.df}</td><td className="px-1 text-right">{f3(r.idf)}</td><td className="px-1 text-right">{f3(r.tf * r.idf)}</td><td className="px-1 text-right text-cyan">{f4(r.w)}</td></tr>)}</tbody>
            </table>
          </div>
          <div className="mt-1 overflow-x-auto scroll-thin text-[12.5px]"><Tex block tex={`\\mathrm{idf}_t = \\ln\\frac{N+1}{\\mathrm{df}_t+1} + 1 = \\ln\\frac{${ix.chunks.length + 1}}{${rows[0]?.df ?? 0}+1} + 1 = ${f3(rows[0]?.idf ?? 0)}\\;(\\text{“${rows[0]?.t}”}),\\quad \\lVert w\\rVert = ${f3(nrm)}`} /></div>
        </Card>
        <Card title={`dense LSA vector (k = ${ix.lsa.k}), learned from this corpus`}>
          <svg viewBox={`0 0 300 ${ix.lsa.k * 16 + 6}`} className="block w-full max-w-[360px]" role="img" aria-label="Dense vector" data-testid="dense-bars">
            <line x1={150} x2={150} y1={0} y2={ix.lsa.k * 16 + 6} stroke={c.lineStrong} />
            {v.map((x, k) => <g key={k}><rect x={x >= 0 ? 150 : 150 - (Math.abs(x) / maxV) * 120} y={k * 16 + 3} width={(Math.abs(x) / maxV) * 120} height={11} rx={2} fill={x >= 0 ? c.cyan : c.violet} /><text x={4} y={k * 16 + 12} fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">d{k + 1}</text><text x={296} y={k * 16 + 12} textAnchor="end" fontSize={9.5} fill={c.muted} fontFamily="var(--font-mono)">{x.toFixed(3)}</text></g>)}
          </svg>
          <div className="mt-2 text-[11.5px] text-muted">singular values σ₁ … σ_k ({pct(ix.lsa.explained)} of the TF-IDF energy kept)</div>
          <div className="flex h-12 items-end gap-0.5">{sig.map((s, k) => <div key={k} className="flex-1 rounded-t" style={{ height: `${(s / maxS) * 100}%`, background: c.violet }} title={`σ${k + 1} = ${s.toFixed(4)}`} />)}</div>
          <label className="mt-2 flex items-center gap-2 text-[12px] text-muted">dimensions k <span className="w-6 font-mono text-fg">{cfg.dims}</span><input type="range" min={2} max={32} value={cfg.dims} onChange={(e) => setCfg((x) => ({ ...x, dims: +e.target.value }))} className="flex-1" aria-label="LSA dimensions" /></label>
        </Card>
      </div>
      <Card title="how the dense embedding is trained (truncated SVD)">
        <div className="overflow-x-auto scroll-thin text-[13px]"><Tex block tex={`A_{N\\times|V|} = U\\Sigma V^{\\top},\\quad \\text{chunk}_i \\mapsto (U_k\\Sigma_k)_i \\in \\mathbb{R}^{${ix.lsa.k}},\\quad \\text{query } q \\mapsto q\\,V_k`} /></div>
        <p className="text-[12px] text-muted">LSA compresses co-occurrence: terms that appear in the same chunks end up in similar directions, so a query can match a chunk that shares related words. It only knows this corpus (N = {ix.chunks.length} chunks, {ix.vocab.length} terms), so it cannot relate words that never occur here. At k = N it reproduces TF-IDF dot products exactly (tested).</p>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 4 · index */
export function IndexView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg), selectChunk = useRag((s) => s.selectChunk), sel = useRag((s) => s.selChunk)
  const col = useDocColors(lab.docs.map((d) => d.id))
  const ix = lab.pipe.index, coords = ix.lsa.coords
  const proj = useMemo(() => pca(coords, 2), [coords])
  const q2 = proj.components.map((comp) => lab.run.q.dense.reduce((s, x, k) => s + (x - proj.mean[k]) * comp[k], 0))
  const pts = proj.coords
  const xs = [...pts.map((p) => p[0]), q2[0] ?? 0], ys = [...pts.map((p) => p[1] ?? 0), q2[1] ?? 0]
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const W = 520, H = 340, sx = (v: number) => 30 + ((v - x0) / (x1 - x0 || 1)) * (W - 60), sy = (v: number) => H - 25 - ((v - y0) / (y1 - y0 || 1)) * (H - 50)
  const top = new Set(lab.run.reranked.map((r) => r.hit.i))
  const ivf = lab.pipe.ivf, probes = ivf ? new Set(probeClusters(ivf, lab.run.q.dense, cfg.retrieve.nprobe)) : null
  const clusterCol = [c.cyan, c.amber, c.violet, c.teal, c.rose, '#8fb3ff', '#b6e36a', '#f59ad1']
  const recallVsFlat = useMemo(() => {
    if (!ivf) return 1
    const exact = topK(scoreAll(ix, lab.run.q, { ...cfg.retrieve, index: 'flat' }), cfg.retrieve.k).map((h) => h.i)
    const got = new Set(topK(lab.run.hits, cfg.retrieve.k).map((h) => h.i))
    return exact.length ? exact.filter((i) => got.has(i)).length / exact.length : 1
  }, [ivf, ix, lab.run, cfg.retrieve])
  return (
    <div className="grid gap-3">
      <Card title="index type">
        <div className="flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
          <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Index type">
            {(['flat', 'ivf'] as const).map((t) => <button key={t} type="button" role="radio" aria-checked={cfg.retrieve.index === t} onClick={() => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, index: t } }))} className={`rounded px-2 py-0.5 text-[12px] ${cfg.retrieve.index === t ? 'bg-raise text-cyan' : ''}`}>{t === 'flat' ? 'exact (flat)' : 'IVF (clusters)'}</button>)}
          </div>
          {cfg.retrieve.index === 'ivf' && <>
            <label className="flex items-center gap-2">clusters n_list <span className="w-4 font-mono text-fg">{cfg.retrieve.nlist}</span><input type="range" min={2} max={8} value={cfg.retrieve.nlist} onChange={(e) => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, nlist: +e.target.value, nprobe: Math.min(x.retrieve.nprobe, +e.target.value) } }))} aria-label="Number of clusters" /></label>
            <label className="flex items-center gap-2">probes n_probe <span className="w-4 font-mono text-fg">{cfg.retrieve.nprobe}</span><input type="range" min={1} max={cfg.retrieve.nlist} value={cfg.retrieve.nprobe} onChange={(e) => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, nprobe: +e.target.value } }))} aria-label="Number of probes" /></label>
            <span className="font-mono" data-testid="ivf-recall">searched {lab.run.hits.filter((h) => h.searched).length}/{ix.chunks.length} chunks · top-k overlap with exact search {pct(recallVsFlat)}</span>
          </>}
        </div>
      </Card>
      <Card title={`the dense vectors, projected to 2-D by PCA (${pct(proj.explained.reduce((a, b) => a + b, 0))} of their variance) · ★ = question`}>
        <svg viewBox={`0 0 ${W} ${H}`} className="block w-full max-w-[680px]" role="img" aria-label="Vector space map" data-testid="vector-map">
          {lab.run.reranked.map((r) => <line key={r.hit.i} x1={sx(q2[0])} y1={sy(q2[1] ?? 0)} x2={sx(pts[r.hit.i][0])} y2={sy(pts[r.hit.i][1] ?? 0)} stroke={c.amber} strokeWidth={1.4} strokeDasharray="4 3" />)}
          {pts.map((p, i) => {
            const ch = ix.chunks[i], cl = ivf ? ivf.assign[i] : -1, faded = probes && !probes.has(cl)
            return (
              <g key={i} onClick={() => selectChunk(i)} style={{ cursor: 'pointer' }}>
                <circle cx={sx(p[0])} cy={sy(p[1] ?? 0)} r={top.has(i) ? 7 : 5} fill={ivf ? clusterCol[cl % 8] : col(ch.docId)} opacity={faded ? 0.18 : 0.9} stroke={i === sel ? c.fg : top.has(i) ? c.amber : 'none'} strokeWidth={2} />
                <title>{`#${i} ${ch.docTitle}: ${ch.text.slice(0, 80)}…`}</title>
              </g>
            )
          })}
          <text x={sx(q2[0])} y={sy(q2[1] ?? 0) + 6} textAnchor="middle" fontSize={20} fill={c.amber}>★</text>
        </svg>
        <div className="flex flex-wrap gap-2 text-[11px] text-muted">{ivf ? <span>colours = clusters; faded = clusters not probed for this question</span> : lab.docs.map((d) => <span key={d.id} className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: col(d.id) }} />{d.title}</span>)}</div>
        <p className="mt-1 text-[11.5px] text-faint">A 2-D picture of {ix.lsa.k}-dimensional vectors: closeness here is approximate. Retrieval uses the full vectors.</p>
      </Card>
    </div>
  )
}
/* ------------------------------------------------------------------ 5 · query */
function QueryView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg)
  const ix = lab.pipe.index, raw = lab.query.split(/[^A-Za-z0-9:']+/).filter(Boolean)
  const qw = [...lab.run.q.tfidf.entries()].map(([id, w]) => ({ t: ix.vocab[id], w })).sort((a, b) => b.w - a.w)
  const maxV = Math.max(1e-9, ...lab.run.q.dense.map(Math.abs))
  return (
    <div className="grid gap-3">
      <Card title="from question to query terms">
        <div className="grid gap-2 text-[12.5px]" data-testid="query-steps">
          <div><span className="eyebrow mr-2">1 · words</span>{raw.map((w, k) => <span key={k} className="mr-1 rounded border border-line px-1.5 font-mono text-[11.5px]">{w}</span>)}</div>
          <div><span className="eyebrow mr-2">2 · stop words removed</span>{raw.map((w, k) => { const s = STOP.has(w.toLowerCase()); return <span key={k} className={`mr-1 rounded px-1.5 font-mono text-[11.5px] ${s ? 'text-faint line-through' : 'border border-line'}`}>{w.toLowerCase()}</span> })}</div>
          <div><span className="eyebrow mr-2">3 · stemmed, in vocabulary?</span>{raw.filter((w) => !STOP.has(w.toLowerCase())).map((w, k) => { const t = stem(w.toLowerCase()), known = ix.termId.has(t); return <span key={k} className="mr-1 rounded border px-1.5 font-mono text-[11.5px]" style={{ borderColor: known ? c.teal : c.rose, color: known ? c.teal : c.rose }}>{t}{known ? ` · df ${ix.df[ix.termId.get(t)!]}` : ' · unknown'}</span> })}</div>
        </div>
        {lab.run.q.unknown.length > 0 && <div className="mt-2 flex items-start gap-2 rounded-lg border border-rose/50 bg-rose/5 p-2 text-[12.5px]"><AlertTriangle size={15} className="mt-0.5 shrink-0 text-rose" />{lab.run.q.unknown.length} term{lab.run.q.unknown.length > 1 ? 's' : ''} ({lab.run.q.unknown.join(', ')}) never occur in the corpus. TF-IDF, BM25 and this LSA embedding ignore them: a lexical gap.</div>}
      </Card>
      <Card title="query rewriting (educational: a hand-written synonym table standing in for an LLM rewriter)">
        <label className="flex items-center gap-2 text-[12.5px]"><input type="checkbox" checked={cfg.rewrite} onChange={(e) => setCfg((x) => ({ ...x, rewrite: e.target.checked }))} data-testid="rewrite-toggle" />rewrite the question before retrieval</label>
        {cfg.rewrite && <div className="mt-1 font-mono text-[12px]">{lab.run.rewrite.applied.length ? lab.run.rewrite.applied.map((a) => <div key={a} className="text-teal">{a}</div>) : <span className="text-faint">no rule matched this question</span>}<div className="mt-1 text-muted">query used: “{lab.run.rewrite.text}”</div></div>}
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="query TF-IDF vector">
          <table className="w-full font-mono text-[12px]"><tbody>{qw.map((r) => <tr key={r.t} className="border-t border-line"><td>{r.t}</td><td className="text-right text-cyan">{f4(r.w)}</td></tr>)}</tbody></table>
          {!qw.length && <p className="text-[12px] text-rose">Empty: no query term is in the vocabulary.</p>}
        </Card>
        <Card title={`query in the LSA space (q V, k = ${ix.lsa.k})`}>
          <svg viewBox={`0 0 300 ${lab.run.q.dense.length * 16 + 6}`} className="block w-full max-w-[360px]" role="img" aria-label="Query dense vector" data-testid="query-dense">
            <line x1={150} x2={150} y1={0} y2={lab.run.q.dense.length * 16 + 6} stroke={c.lineStrong} />
            {lab.run.q.dense.map((x, k) => <g key={k}><rect x={x >= 0 ? 150 : 150 - (Math.abs(x) / maxV) * 120} y={k * 16 + 3} width={(Math.abs(x) / maxV) * 120} height={11} rx={2} fill={x >= 0 ? c.cyan : c.violet} /><text x={4} y={k * 16 + 12} fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">d{k + 1}</text><text x={296} y={k * 16 + 12} textAnchor="end" fontSize={9.5} fill={c.muted} fontFamily="var(--font-mono)">{x.toFixed(3)}</text></g>)}
          </svg>
          <p className="text-[11.5px] text-muted">‖q̂‖ = {f4(Math.hypot(...lab.run.q.dense))}</p>
        </Card>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 6 · similarity */
function SimilarityView({ lab }: { lab: Lab }) {
  const c = useColors()
  const sel = useRag((s) => s.selChunk), selectChunk = useRag((s) => s.selectChunk)
  const ix = lab.pipe.index, i = focusChunk(lab, sel), q = lab.run.q
  const shared = [...q.tfidf.entries()].filter(([id]) => ix.tfidf[i].has(id)).map(([id, w]) => ({ t: ix.vocab[id], qw: w, cw: ix.tfidf[i].get(id)!, p: w * ix.tfidf[i].get(id)! }))
  const v = ix.lsa.coords[i], prods = q.dense.map((x, k) => x * v[k])
  const b = bm25(ix, q, i)
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">compare the question with
        <select className="max-w-[320px] rounded border border-line bg-panel-2 px-1" value={i} onChange={(e) => selectChunk(+e.target.value)} aria-label="Chunk to compare">{ix.chunks.map((x) => <option key={x.id} value={x.id}>{chunkLabel(lab, x.id)}</option>)}</select>
      </div>
      <p className="rounded-lg border border-line bg-panel-2/40 p-2 text-[12.5px] text-muted">{ix.chunks[i].text}</p>
      <div className="grid gap-3 @4xl:grid-cols-3">
        <Card title="TF-IDF cosine (both vectors have length 1)">
          <table className="w-full font-mono text-[12px]" data-testid="sparse-products"><thead className="text-faint"><tr><th className="text-left">term</th><th className="text-right">q</th><th className="text-right">c</th><th className="text-right">q·c</th></tr></thead>
            <tbody>{shared.map((r) => <tr key={r.t} className="border-t border-line"><td>{r.t}</td><td className="text-right">{f3(r.qw)}</td><td className="text-right">{f3(r.cw)}</td><td className="text-right">{f4(r.p)}</td></tr>)}</tbody></table>
          <div className="mt-1 font-mono text-[12.5px]">cos = Σ = <span className="text-cyan" data-testid="tfidf-cos">{f4(sparseDot(q.tfidf, ix.tfidf[i]))}</span></div>
          {!shared.length && <p className="text-[11.5px] text-muted">No shared terms: the TF-IDF cosine is 0.</p>}
        </Card>
        <Card title="dense (LSA) cosine">
          <div className="overflow-x-auto scroll-thin text-[12.5px]"><Tex block tex={`\\cos = \\frac{\\sum_k \\hat q_k c_k}{\\lVert\\hat q\\rVert\\lVert c\\rVert} = \\frac{${f4(prods.reduce((a, x) => a + x, 0))}}{${f3(Math.hypot(...q.dense))}\\times${f3(Math.hypot(...v))}} = ${f4(cosine(q.dense, v))}`} /></div>
          <div className="mt-1 flex h-16 items-end gap-0.5">{prods.map((p, k) => <div key={k} className="flex-1 rounded-t" style={{ height: `${Math.min(100, (Math.abs(p) / Math.max(1e-9, ...prods.map(Math.abs))) * 100)}%`, background: p >= 0 ? c.teal : c.rose }} title={`d${k + 1}: ${q.dense[k].toFixed(3)} × ${v[k].toFixed(3)} = ${p.toFixed(4)}`} />)}</div>
          <p className="text-[11px] text-faint">bars: per-dimension products (teal +, rose −)</p>
        </Card>
        <Card title={`BM25 (k₁ = ${BM25_K1}, b = ${BM25_B})`}>
          <table className="w-full font-mono text-[12px]" data-testid="bm25-table"><thead className="text-faint"><tr><th className="text-left">term</th><th className="text-right">tf</th><th className="text-right">idf</th><th className="text-right">score</th></tr></thead>
            <tbody>{b.parts.map((p) => <tr key={p.term} className="border-t border-line"><td>{p.term}</td><td className="text-right">{p.tf}</td><td className="text-right">{f3(p.idf)}</td><td className="text-right">{f4(p.contrib)}</td></tr>)}</tbody></table>
          <div className="mt-1 font-mono text-[12.5px]">BM25 = <span className="text-amber">{f4(b.score)}</span> · length {ix.lengths[i]} vs average {f3(ix.avgLen)}</div>
        </Card>
      </div>
      <div className="overflow-x-auto scroll-thin text-[13px]"><Tex block tex={`\\mathrm{BM25}(q,c) = \\sum_{t\\in q} \\mathrm{idf}_t\\,\\frac{\\mathrm{tf}_{t,c}(k_1+1)}{\\mathrm{tf}_{t,c} + k_1\\left(1 - b + b\\,\\frac{|c|}{\\overline{|c|}}\\right)},\\quad \\mathrm{idf}_t = \\ln\\left(1 + \\frac{N - \\mathrm{df}_t + 0.5}{\\mathrm{df}_t + 0.5}\\right)`} /></div>
    </div>
  )
}

/* ------------------------------------------------------------------ 7 · retrieval */
export function RetrieveView({ lab, compact = false }: { lab: Lab; compact?: boolean }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg), selectChunk = useRag((s) => s.selectChunk), sel = useRag((s) => s.selChunk)
  const col = useDocColors(lab.docs.map((d) => d.id))
  const gold = new Set(lab.gold.ids), topIds = new Set(lab.run.hits.slice(0, cfg.retrieve.k).filter((h) => Number.isFinite(h.rank)).map((h) => h.i))
  const shown = compact ? lab.run.hits.slice(0, Math.max(6, cfg.retrieve.k + 2)) : lab.run.hits
  const maxS = Math.max(1e-9, ...lab.run.hits.map((h) => (Number.isFinite(h.score) ? h.score : 0)))
  return (
    <div className="grid gap-3">
      {!compact && (
        <Card title="retrieval settings">
          <div className="flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
            <div className="flex flex-wrap rounded-md border border-line p-0.5" role="radiogroup" aria-label="Retrieval method">
              {(Object.keys(METHOD_LABEL) as Method[]).map((m) => <button key={m} type="button" role="radio" aria-checked={cfg.retrieve.method === m} onClick={() => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, method: m } }))} className={`rounded px-2 py-0.5 text-[12px] ${cfg.retrieve.method === m ? 'bg-raise text-cyan' : ''}`}>{METHOD_LABEL[m]}</button>)}
            </div>
            <label className="flex items-center gap-2">top-k <span className="w-5 font-mono text-fg">{cfg.retrieve.k}</span><input type="range" min={1} max={10} value={cfg.retrieve.k} onChange={(e) => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, k: +e.target.value } }))} aria-label="Top k" /></label>
            {cfg.retrieve.method === 'hybrid' && <>
              <select className="rounded border border-line bg-panel-2 px-1" value={cfg.retrieve.fusion} onChange={(e) => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, fusion: e.target.value as 'rrf' } }))} aria-label="Fusion"><option value="rrf">reciprocal rank fusion</option><option value="weighted">weighted sum (min-max)</option></select>
              {cfg.retrieve.fusion === 'weighted' && <label className="flex items-center gap-2">α (dense weight) <span className="w-8 font-mono text-fg">{cfg.retrieve.alpha.toFixed(2)}</span><input type="range" min={0} max={1} step={0.05} value={cfg.retrieve.alpha} onChange={(e) => setCfg((x) => ({ ...x, retrieve: { ...x.retrieve, alpha: +e.target.value } }))} aria-label="Alpha" /></label>}
            </>}
          </div>
        </Card>
      )}
      <Card title={`${METHOD_LABEL[cfg.retrieve.method]} · all ${lab.pipe.chunks.length} chunks ranked · top-${cfg.retrieve.k} highlighted · ★ = gold evidence`}>
        <div className={`${compact ? '' : 'max-h-[460px]'} overflow-auto scroll-thin`}>
          <table className="w-full text-[12px]" data-testid="ranking-table">
            <thead className="sticky top-0 bg-panel text-left text-faint"><tr><th className="px-1">rank</th><th className="px-1">chunk</th><th className="px-1 text-right">dense</th><th className="px-1 text-right">TF-IDF</th><th className="px-1 text-right">BM25</th>{!compact && <><th className="px-1 text-right">rank d / b</th><th className="px-1 text-right">RRF / fused</th></>}<th className="px-1">score</th></tr></thead>
            <tbody>
              {shown.map((h) => {
                const ch = lab.pipe.chunks[h.i], inTop = topIds.has(h.i)
                return (
                  <motion.tr key={h.i} layout className={`cursor-pointer border-t border-line ${inTop ? 'bg-raise' : ''} ${!h.searched ? 'opacity-35' : ''} ${h.i === sel ? 'outline outline-1 outline-cyan' : ''}`} onClick={() => selectChunk(h.i)} data-testid={`rank-row-${h.i}`}>
                    <td className="px-1 font-mono">{Number.isFinite(h.rank) ? h.rank : '—'}</td>
                    <td className="max-w-[300px] px-1"><span className="font-mono" style={{ color: col(ch.docId) }}>#{h.i}{gold.has(h.i) ? ' ★' : ''}</span> <span className="text-muted">{ch.docTitle}: {ch.text.slice(0, compact ? 60 : 90)}…</span></td>
                    <td className="px-1 text-right font-mono">{f3(h.dense)}</td><td className="px-1 text-right font-mono">{f3(h.tfidf)}</td><td className="px-1 text-right font-mono">{f3(h.bm25)}</td>
                    {!compact && <><td className="px-1 text-right font-mono text-faint">{Number.isFinite(h.rankDense) ? h.rankDense : '—'} / {Number.isFinite(h.rankBm25) ? h.rankBm25 : '—'}</td><td className="px-1 text-right font-mono">{f4(h.fused)}</td></>}
                    <td className="w-[90px] px-1"><div className="h-2 rounded bg-panel-2"><div className="h-2 rounded" style={{ width: `${Math.max(0, (Number.isFinite(h.score) ? h.score : 0) / maxS) * 100}%`, background: inTop ? c.amber : c.lineStrong }} /></div></td>
                  </motion.tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {!compact && cfg.retrieve.method === 'hybrid' && cfg.retrieve.fusion === 'rrf' && lab.run.hits[0] && (
          <div className="mt-1 overflow-x-auto scroll-thin text-[12.5px]"><Tex block tex={`\\mathrm{RRF}(\\#${lab.run.hits[0].i}) = \\frac{1}{60 + ${lab.run.hits[0].rankDense}} + \\frac{1}{60 + ${lab.run.hits[0].rankBm25}} = ${f4(lab.run.hits[0].fused)}`} /></div>
        )}
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 8 · rerank */
function RerankView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg)
  const col = useDocColors(lab.docs.map((d) => d.id))
  const pool = lab.run.hits.filter((h) => Number.isFinite(h.rank)).slice(0, Math.max(cfg.retrieve.k, cfg.pool))
  const after = lab.run.reranked
  const W = 560, rowH = 26, H = Math.max(pool.length, 1) * rowH + 30
  const ix = lab.pipe.index
  return (
    <div className="grid gap-3">
      <Card title="reranker">
        <div className="flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
          <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Reranker">
            {(['none', 'mmr', 'coverage'] as const).map((m) => <button key={m} type="button" role="radio" aria-checked={cfg.reranker === m} onClick={() => setCfg((x) => ({ ...x, reranker: m }))} className={`rounded px-2 py-0.5 text-[12px] ${cfg.reranker === m ? 'bg-raise text-cyan' : ''}`}>{RERANK_LABEL[m]}</button>)}
          </div>
          <label className="flex items-center gap-2">candidate pool <span className="w-5 font-mono text-fg">{cfg.pool}</span><input type="range" min={cfg.retrieve.k} max={15} value={cfg.pool} onChange={(e) => setCfg((x) => ({ ...x, pool: +e.target.value }))} aria-label="Candidate pool" /></label>
          {cfg.reranker === 'mmr' && <label className="flex items-center gap-2">λ <span className="w-8 font-mono text-fg">{cfg.lambda.toFixed(2)}</span><input type="range" min={0} max={1} step={0.05} value={cfg.lambda} onChange={(e) => setCfg((x) => ({ ...x, lambda: +e.target.value }))} aria-label="MMR lambda" /></label>}
        </div>
        {cfg.reranker === 'coverage' && <p className="mt-1 text-[11.5px] text-amber">Educational heuristic, not a trained cross-encoder: score = share of query terms present + 0.5 × (terms present / shortest window containing them).</p>}
      </Card>
      <Card title={`first stage (left, ${pool.length} candidates) → after reranking (right, top-${cfg.retrieve.k})`}>
        <svg viewBox={`0 0 ${W} ${H}`} className="block w-full max-w-[680px]" role="img" aria-label="Rerank slope chart" data-testid="rerank-chart">
          {pool.map((h, r) => <g key={h.i}><text x={150} y={30 + r * rowH} textAnchor="end" fontSize={11} fill={c.fg} fontFamily="var(--font-mono)">{r + 1}. #{h.i} {lab.pipe.chunks[h.i].docTitle.slice(0, 12)}</text><circle cx={160} cy={26 + r * rowH} r={4} fill={col(lab.pipe.chunks[h.i].docId)} /></g>)}
          {after.map((a) => {
            const r0 = pool.findIndex((h) => h.i === a.hit.i)
            return <g key={a.hit.i}><line x1={164} y1={26 + r0 * rowH} x2={396} y2={26 + (a.after - 1) * rowH} stroke={a.after === r0 + 1 ? c.lineStrong : c.amber} strokeWidth={2} /><circle cx={400} cy={26 + (a.after - 1) * rowH} r={4} fill={col(lab.pipe.chunks[a.hit.i].docId)} /><text x={410} y={30 + (a.after - 1) * rowH} fontSize={11} fill={c.fg} fontFamily="var(--font-mono)">{a.after}. #{a.hit.i} ({f3(a.rerankScore)})</text></g>
          })}
        </svg>
        <table className="mt-1 w-full font-mono text-[11.5px]" data-testid="rerank-detail"><tbody>{after.map((a) => <tr key={a.hit.i} className="border-t border-line"><td>#{a.hit.i}</td><td>{a.before} → {a.after}</td><td className="text-muted">{a.detail}</td></tr>)}</tbody></table>
      </Card>
      <Card title="redundancy in the candidate pool (dense cosine between candidates)">
        <div className="overflow-x-auto scroll-thin">
          <table className="font-mono text-[10.5px]"><tbody>{pool.map((a) => <tr key={a.i}><td className="pr-1 text-faint">#{a.i}</td>{pool.map((b) => { const v = cosine(ix.lsa.coords[a.i], ix.lsa.coords[b.i]); return <td key={b.i} className="h-6 w-9 text-center" style={{ background: `color-mix(in oklab, ${c.violet} ${Math.round(Math.max(0, v) * 70)}%, transparent)` }}>{v.toFixed(2)}</td> })}</tr>)}</tbody></table>
        </div>
        <p className="mt-1 text-[11.5px] text-muted">High off-diagonal values mean near-duplicate passages (often neighbouring chunks of the same paragraph, because of overlap). MMR penalizes them.</p>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 9 · prompt */
function PromptView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg)
  const p = lab.run.prompt
  return (
    <div className="grid gap-3">
      <Card title="context budget">
        <div className="flex flex-wrap items-center gap-3 text-[12.5px] text-muted">
          <label className="flex items-center gap-2">budget for passages <span className="w-10 font-mono text-fg">{cfg.prompt.budget}</span><input type="range" min={40} max={600} step={10} value={cfg.prompt.budget} onChange={(e) => setCfg((x) => ({ ...x, prompt: { ...x.prompt, budget: +e.target.value } }))} aria-label="Context budget" /></label>
          <label className="flex items-center gap-1"><input type="checkbox" checked={cfg.prompt.instructions} onChange={(e) => setCfg((x) => ({ ...x, prompt: { ...x.prompt, instructions: e.target.checked } }))} />grounding instructions</label>
        </div>
        <div className="mt-2 h-3 rounded bg-panel-2"><div className="h-3 rounded" style={{ width: `${Math.min(100, (p.contextTokens / cfg.prompt.budget) * 100)}%`, background: c.cyan }} /></div>
        <div className="mt-1 font-mono text-[12px]" data-testid="prompt-tokens">passages: {p.contextTokens} / {cfg.prompt.budget} tokens · whole prompt ≈ {p.tokens} tokens (≈ 1.3 tokens per word) · {p.included.length} included{p.dropped.length ? `, ${p.dropped.length} dropped (${p.dropped.map((i) => `#${i}`).join(', ')})` : ''}</div>
      </Card>
      <Card title="the prompt sent to the generator">
        <div className="grid gap-2 font-mono text-[12px] leading-relaxed" data-testid="prompt-text">
          {p.parts.map((part, k) => (
            <motion.div key={k} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: k * 0.08 }} className="rounded-md border-l-2 px-2 py-1" style={{ borderColor: part.kind === 'system' ? c.muted : part.kind === 'context' ? c.cyan : c.amber, background: 'var(--panel-2)' }}>
              {part.kind === 'system' && <><span className="text-faint">System: </span>{part.text}</>}
              {part.kind === 'context' && <><span className="text-cyan">[{part.cite}]</span> <span className="text-faint">({lab.pipe.chunks[part.chunk!].docTitle} · #{part.chunk} · {approxTokens(part.text)} tok)</span> {part.text}</>}
              {part.kind === 'question' && <><span className="text-amber">Question: </span>{part.text}<div className="text-faint">Answer:</div></>}
            </motion.div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 10 · generation */
export function GenerateView({ lab, compact = false }: { lab: Lab; compact?: boolean }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg)
  const a = lab.run.answer, q = lab.question
  const ctx = lab.run.prompt.included.map((i) => lab.pipe.chunks[i].text)
  const gSup = supportRatio(a.text, ctx), cSup = q ? supportRatio(q.closedBook, ctx) : null
  const correct = q ? (q.key ? !a.abstained && a.text.toLowerCase().includes(q.key.toLowerCase()) : a.abstained) : null
  return (
    <div className="grid gap-3">
      {!compact && <label className="flex items-center gap-2 text-[12.5px] text-muted">abstain when support &lt; <span className="w-10 font-mono text-fg">{cfg.threshold.toFixed(2)}</span><input type="range" min={0} max={1} step={0.05} value={cfg.threshold} onChange={(e) => setCfg((x) => ({ ...x, threshold: +e.target.value }))} aria-label="Abstention threshold" /></label>}
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={<span className="text-teal">grounded · with retrieved passages</span>} right={<span className="sim-badge toy">extractive simulation, not an LLM</span>}>
          <motion.p key={a.text} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[14px] leading-relaxed" data-testid="grounded-answer">{a.text}</motion.p>
          <div className="mt-2 flex flex-wrap gap-3 font-mono text-[11.5px] text-muted">
            <span>support {f3(a.support)} (threshold {a.threshold})</span><span>terms in context {pct(gSup.ratio)}</span>
            {correct !== null && <span className={correct ? 'text-teal' : 'text-rose'} data-testid="grounded-verdict">{correct ? '✓ matches the gold evidence' : '✗ does not match the gold evidence'}</span>}
          </div>
        </Card>
        <Card title={<span className="text-rose">ungrounded · closed book, no retrieval</span>} right={<span className="sim-badge toy">scripted simulation</span>}>
          {q ? <>
            <p className="text-[14px] leading-relaxed" data-testid="closed-answer">{q.closedBook}</p>
            <div className="mt-2 flex flex-wrap gap-3 font-mono text-[11.5px] text-muted"><span>no citations</span><span>terms supported by the retrieved passages {pct(cSup!.ratio)}</span><span className={q.closedBookCorrect ? 'text-teal' : 'text-rose'}>{q.closedBookCorrect ? '✓ happens to be right' : '✗ wrong or unsupported'}</span></div>
          </> : <p className="text-[12.5px] text-muted">Scripted closed-book answers exist only for the example questions. The point stands for any question: without sources, a fluent answer cannot be checked.</p>}
        </Card>
      </div>
      {!compact && (
        <Card title="how the grounded answer was chosen: sentence support = idf of matched query terms ÷ idf of all query terms">
          <table className="w-full text-[12px]" data-testid="sentence-scores"><thead className="text-left text-faint"><tr><th className="px-1">[n]</th><th className="px-1">sentence</th><th className="px-1">matched</th><th className="px-1 text-right">support</th></tr></thead>
            <tbody>{a.candidates.slice(0, 6).map((s, k) => <tr key={k} className={`border-t border-line ${a.used.includes(s) ? 'text-teal' : ''}`}><td className="px-1 font-mono">[{s.cite}]</td><td className="px-1">{s.text}</td><td className="px-1 font-mono text-[11px]">{s.matched.join(', ')}</td><td className="px-1 text-right font-mono">{f3(s.score)}</td></tr>)}</tbody></table>
          <p className="mt-1 text-[11.5px] text-muted">A real LLM would read all passages and write its own sentences; it can combine evidence (and reason, e.g. 5 days &gt; 4 days) but can also misread passages or ignore the instruction to abstain. This simulation is transparent and exactly reproducible instead.</p>
        </Card>
      )}
      {!compact && <p className="text-[11.5px]" style={{ color: c.faint }}>Both answer panels are simulations. Only the retrieval mathematics (TF-IDF, BM25, LSA trained on this corpus) is computed from real models.</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ 11 · citations */
function CiteView({ lab }: { lab: Lab }) {
  const c = useColors()
  const a = lab.run.answer, p = lab.run.prompt
  const checks = verifyCitations(a, p, lab.pipe.index)
  const sup = supportRatio(a.text, p.included.map((i) => lab.pipe.chunks[i].text))
  const cb = lab.question ? supportRatio(lab.question.closedBook, p.included.map((i) => lab.pipe.chunks[i].text)) : null
  return (
    <div className="grid gap-3">
      {a.abstained ? <Card title="no citations"><p className="text-[13px]">The generator abstained (support {f3(a.support)} &lt; {a.threshold}). For an unanswerable question this is the correct, grounded behaviour.</p></Card> : (
        <Card title="each cited sentence, inside the passage it cites">
          <div className="grid gap-2" data-testid="citations">
            {a.used.map((s, k) => {
              const passage = lab.pipe.chunks[p.included[s.cite - 1]].text, at = passage.indexOf(s.text)
              return (
                <div key={k} className="rounded-lg border border-line p-2 text-[12.5px]">
                  <div className="mb-1 flex items-center gap-2 font-mono text-[11.5px]">{checks[k].valid ? <CheckCircle2 size={14} className="text-teal" /> : <XCircle size={14} className="text-rose" />}[{s.cite}] → #{p.included[s.cite - 1]} {lab.pipe.chunks[p.included[s.cite - 1]].docTitle} · {checks[k].valid ? 'verified: the sentence occurs in the cited passage' : 'not found in the cited passage'}</div>
                  <p className="text-muted">{at >= 0 ? <>{passage.slice(0, at)}<mark style={{ background: `color-mix(in oklab, ${c.teal} 30%, transparent)`, color: 'inherit' }}>{s.text}</mark>{passage.slice(at + s.text.length)}</> : passage}</p>
                </div>
              )
            })}
          </div>
        </Card>
      )}
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="grounded answer: faithfulness proxy">
          <div className="font-mono text-[13px]">{pct(sup.ratio)} of its content terms occur in the prompt passages</div>
          {sup.unsupported.length > 0 && <div className="mt-1 text-[12px] text-rose">unsupported: {sup.unsupported.join(', ')}</div>}
        </Card>
        {cb && <Card title="closed-book simulation: the same check">
          <div className="font-mono text-[13px]">{pct(cb.ratio)} of its content terms occur in the passages</div>
          <div className="mt-1 text-[12px] text-rose">not supported: {cb.unsupported.join(', ') || '—'}</div>
        </Card>}
      </div>
      <p className="text-[11.5px] text-muted">Term overlap is a crude proxy: an answer can reuse words and still misstate a fact. Production systems check claims with entailment models or human review.</p>
    </div>
  )
}

/* ------------------------------------------------------------------ 12 · failures and evaluation */
function FailuresView({ lab }: { lab: Lab }) {
  const c = useColors()
  const cfg = useRag((s) => s.cfg), setQuestion = useRag((s) => s.setQuestion), corpusId = useRag((s) => s.corpusId)
  const { ev, summary, questions } = useEvaluation()
  const sizes = [10, 15, 20, 30, 40, 60, 80, 120]
  const sweep = useMemo(() => {
    const qs = corpusById(corpusId).questions
    const run = (size: number, overlap: number, method: Method) => { const cc: PipelineConfig = { ...cfg, chunk: { ...cfg.chunk, size, overlap: Math.min(overlap, size - 1) }, retrieve: { ...cfg.retrieve, method } }; const p = buildPipeline(lab.docs, cc); return summarize(evaluate(p, qs, cc), qs) }
    return { lines: [
      { label: 'overlap 0 · hybrid', pts: sizes.map((s) => [s, run(s, 0, 'hybrid').recallAtK]) },
      { label: `overlap ${cfg.chunk.overlap} · hybrid`, pts: sizes.map((s) => [s, run(s, cfg.chunk.overlap, 'hybrid').recallAtK]) },
      { label: `overlap ${cfg.chunk.overlap} · BM25`, pts: sizes.map((s) => [s, run(s, cfg.chunk.overlap, 'bm25').recallAtK]) },
      { label: `overlap ${cfg.chunk.overlap} · dense`, pts: sizes.map((s) => [s, run(s, cfg.chunk.overlap, 'dense').recallAtK]) },
    ] }
  }, [lab.docs, corpusId, cfg]) // eslint-disable-line react-hooks/exhaustive-deps
  const tiles: [string, string][] = [[`Recall@${cfg.retrieve.k}`, pct(summary.recallAtK)], ['MRR', f3(summary.mrr)], ['answers correct', pct(summary.answerAccuracy)], ['evidence splits', String(summary.splits)]]
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-4" data-testid="eval-tiles">
        {tiles.map(([k, v]) => <div key={k} className="rounded-xl border border-line bg-panel/70 p-3"><div className="eyebrow">{k}</div><div className="font-display text-[26px]">{v}</div></div>)}
      </div>
      <Card title={`every example question of this knowledge base, with the current configuration`}>
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-[12px]" data-testid="eval-table">
            <thead className="text-left text-faint"><tr><th className="px-1">question</th><th className="px-1">designed to test</th><th className="px-1 text-right">gold rank</th><th className="px-1">top-k</th><th className="px-1">answer</th><th className="px-1">diagnosis</th></tr></thead>
            <tbody>{ev.map((e, n) => {
              const q = questions[n], run = e.answer
              const d = diagnose(e, q, lab.pipe, cfg.retrieve.k, lab.docs, e.retrieved.map((i) => lab.pipe.chunks[i].docId))
              return (
                <tr key={e.qid} className={`cursor-pointer border-t border-line align-top ${lab.question?.id === q.id ? 'bg-raise' : ''}`} onClick={() => setQuestion(q.id)}>
                  <td className="px-1">{q.q}</td><td className="px-1 font-mono text-[11px] text-muted">{q.tag}</td>
                  <td className="px-1 text-right font-mono">{q.gold ? (Number.isFinite(e.goldRank) ? e.goldRank : '—') : 'n/a'}</td>
                  <td className="px-1">{q.gold ? (e.recall ? '✓' : '✗') : '—'}{e.split ? ' ✂' : ''}</td>
                  <td className="px-1">{e.answerCorrect ? <span className="text-teal">✓</span> : <span className="text-rose">✗</span>} <span className="text-faint">{run.abstained ? 'abstained' : ''}</span></td>
                  <td className="px-1" style={{ color: d.good ? c.teal : c.rose }} data-testid={`diag-${e.qid}`}>{d.text}</td>
                </tr>
              )
            })}</tbody>
          </table>
        </div>
        <p className="mt-1 text-[11.5px] text-muted">Click a row to load that question. ✂ = the gold evidence is split across chunks. Correct for an unanswerable question means abstaining.</p>
      </Card>
      <Card title={`Recall@${cfg.retrieve.k} as a function of chunk size (every point rebuilds the index and re-runs all questions)`}>
        <div className="max-w-[640px]"><LineChart ariaLabel="Recall versus chunk size" xLabel="chunk size (words)" yLabel="Recall@k" height={220} yDomain={[0, 1]}
          series={sweep.lines.map((l, k) => ({ points: l.pts, color: [c.rose, c.cyan, c.amber, c.violet][k], label: l.label, dashed: k === 0 }))} markers={[{ x: cfg.chunk.size, y: summary.recallAtK, color: c.fg, label: 'now' }]} /></div>
      </Card>
      <Card title="failure modes to look for"><ul className="grid list-disc gap-0.5 pl-5 text-[12.5px] text-muted @3xl:grid-cols-2">
        <li><b className="text-fg">Retrieval miss</b>: the evidence is not in the top-k (lexical gap, weak embedding, k too small).</li>
        <li><b className="text-fg">Chunk boundary</b>: the evidence is split; increase overlap or use sentence packing.</li>
        <li><b className="text-fg">Context budget</b>: retrieved but cut from the prompt.</li>
        <li><b className="text-fg">Distractor</b>: a similar passage outranks the evidence.</li>
        <li><b className="text-fg">Conflicting or stale sources</b>: retrieval cannot tell which source is correct.</li>
        <li><b className="text-fg">Reasoning</b>: evidence present, but the answer needs inference.</li>
        <li><b className="text-fg">Unanswerable</b>: the right output is “I don't know”.</li>
        <li><b className="text-fg">Hallucination</b>: fluent text without support (the closed-book panel).</li>
      </ul></Card>
      <div className="flex items-center gap-2 text-[11.5px] text-faint"><BookOpen size={13} />Evaluation uses the gold evidence span of each example question; answers are scored by whether they contain the gold key phrase.</div>
    </div>
  )
}
