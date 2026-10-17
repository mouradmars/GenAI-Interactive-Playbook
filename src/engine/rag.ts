/**
 * Retrieval-augmented generation, computed exactly and deterministically, for module 11.
 *
 *  - Text processing: lowercase word tokens, a fixed stop-word list, a small conservative suffix stemmer.
 *  - Chunking: fixed word windows (size, overlap) or sentence packing (sentences up to `size` words, `overlap` words repeated).
 *  - Sparse vectors: TF-IDF with tf = 1 + ln(count), idf = ln((N + 1)/(df + 1)) + 1, L2-normalized.
 *  - BM25: idf = ln(1 + (N − df + 0.5)/(df + 0.5)), score = Σ idf · tf(k1+1) / (tf + k1(1 − b + b·len/avglen)).
 *  - Dense vectors: LSA, a k-dimensional embedding learned from this corpus by truncated SVD of the TF-IDF matrix
 *    (A = UΣVᵀ, chunk i ↦ (UΣ)ᵢ, query q ↦ q V). It is a real (unsupervised, linear) trained embedding, not a neural one.
 *  - Index: exact (flat) or IVF (seeded k-means over the dense vectors, search the nprobe nearest clusters).
 *  - Fusion: reciprocal rank fusion Σ 1/(60 + rank) or a weighted sum of min-max normalized scores.
 *  - Reranking: maximal marginal relevance, or a transparent query-coverage heuristic (not a trained cross-encoder).
 *  - Generation: an extractive simulation (no language model): the best-supported sentences of the prompt context,
 *    cited by passage number, or an abstention when support is below a threshold.
 */
import { eigh } from './linalg'
import { rng } from './decoding'
import type { RagDoc, RagQuestion } from '../data/rag-corpora'

/* ------------------------------------------------------------------ text processing */
export const STOP = new Set(('a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that these those ' +
  'there here what which who whom whose when where why how do does did can could may might must shall should will would i you he she we they me him her us them ' +
  'my your his our their not no nor so than too very just about into over after before also any each all both such only own same other some more most ' +
  'up down out off again further once has have had having am per one many much happen happens get got ever').split(' '))

/** conservative stemmer: plurals and a few verb endings; stems are shown in the UI */
export function stem(w: string): string {
  if (w.length <= 3) return w
  if (w.endsWith("'s")) w = w.slice(0, -2)
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y'
  if (w.endsWith('sses')) return w.slice(0, -2)
  if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3)
  if (w.endsWith('ed') && w.length > 4 && !w.endsWith('eed')) return w.slice(0, -2)
  if (w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('us') && !w.endsWith('is')) return w.slice(0, -1)
  return w
}
export const words = (text: string): string[] => text.split(/\s+/).filter(Boolean)
export const normWord = (w: string) => w.toLowerCase().replace(/[^a-z0-9:'.-]/g, '').replace(/^[.'-]+|[.'-]+$/g, '')
/** content terms (stemmed, stop words removed) of a text */
export const terms = (text: string): string[] => text.split(/[^A-Za-z0-9:']+/).map((w) => w.toLowerCase()).filter((w) => w && !STOP.has(w)).map(stem)
export const sentences = (text: string): string[] => text.split(/(?<=[.!?])\s+(?=[A-Z0-9])/).map((s) => s.trim()).filter(Boolean)

/* ------------------------------------------------------------------ query rewriting (educational) */
/**
 * A small hand-written synonym table standing in for LLM-based query rewriting. It is labelled as such in the UI:
 * it shows the mechanism (adding corpus vocabulary to a query), not a learned model.
 */
export const REWRITES: [RegExp, string][] = [
  [/\bchatgpt\b|\bgpt\b|\bchatbots?\b/i, 'generative AI assistants chatbots'],
  [/\bhomework\b/i, 'problem sets'],
  [/\bhottest\b|\bwarmest\b/i, 'highest surface temperature'],
  [/\bstand for\b/i, 'short for'],
  [/\bworth\b/i, 'counts percent grade'],
  [/\bbook\b|\btextbook\b/i, 'textbook reading'],
]
export function rewriteQuery(text: string): { text: string; applied: string[] } {
  const applied: string[] = []
  let out = text
  for (const [re, add] of REWRITES) { const m = text.match(re); if (m) { applied.push(`${m[0]} → +“${add}”`); out += ' ' + add } }
  return { text: out, applied }
}

/* ------------------------------------------------------------------ chunking */
export type ChunkStrategy = 'fixed' | 'sentence'
export interface ChunkConfig { size: number; overlap: number; strategy: ChunkStrategy }
export interface Chunk { id: number; docId: string; docTitle: string; index: number; text: string; start: number; end: number; terms: string[]; words: number }

/** number of fixed windows for n words: ceil((n − overlap) / (size − overlap)), at least 1 */
export const fixedChunkCount = (n: number, size: number, overlap: number) => (n <= size ? 1 : Math.ceil((n - overlap) / (size - overlap)))

export function chunkDoc(doc: RagDoc, c: ChunkConfig, idStart = 0): Chunk[] {
  const w = words(doc.text), size = Math.max(5, Math.round(c.size)), ov = Math.max(0, Math.min(size - 1, Math.round(c.overlap)))
  const out: Chunk[] = []
  const push = (s: number, e: number) => { const text = w.slice(s, e).join(' '); out.push({ id: idStart + out.length, docId: doc.id, docTitle: doc.title, index: out.length, text, start: s, end: e, terms: terms(text), words: e - s }) }
  if (c.strategy === 'fixed') {
    if (w.length <= size) push(0, w.length)
    else for (let s = 0; ; s += size - ov) { const e = Math.min(w.length, s + size); push(s, e); if (e >= w.length) break }
    return out
  }
  // sentence packing: whole sentences up to `size` words (a longer sentence becomes its own chunk); the next chunk
  // starts `overlap` words before the end of the previous one, rounded back to a sentence start when possible
  const sents = sentences(doc.text)
  const bounds: [number, number][] = []
  let pos = 0
  for (const s of sents) { const n = words(s).length; bounds.push([pos, pos + n]); pos += n }
  let i = 0
  while (i < bounds.length) {
    let j = i, end = bounds[i][1]
    while (j + 1 < bounds.length && bounds[j + 1][1] - bounds[i][0] <= size) { j++; end = bounds[j][1] }
    push(bounds[i][0], end)
    if (j + 1 >= bounds.length) break
    // first sentence of the next chunk: the latest sentence starting at or after end − overlap, but always progress
    let k = j + 1
    while (k - 1 > i && ov > 0 && bounds[k - 1][0] >= end - ov) k--
    i = Math.max(i + 1, k)
  }
  return out
}
export function chunkCorpus(docs: RagDoc[], c: ChunkConfig): Chunk[] {
  const out: Chunk[] = []
  for (const d of docs) out.push(...chunkDoc(d, c, out.length))
  return out
}
/** does some chunk contain the whole gold span (word-normalized)? */
export const containsSpan = (chunkText: string, span: string) => !!span && collapse(chunkText).includes(collapse(span))
const collapse = (s: string) => ` ${words(s).map(normWord).join(' ')} `

/* ------------------------------------------------------------------ index */
export type SparseVec = Map<number, number>
export interface Index {
  chunks: Chunk[]
  vocab: string[]
  termId: Map<string, number>
  df: number[]
  idf: number[]
  bm25idf: number[]
  tfidf: SparseVec[]
  lengths: number[]
  avgLen: number
  /** LSA: k dims, singular values, chunk coordinates (UΣ), term projection V (|vocab| × k) */
  lsa: { k: number; sigma: number[]; coords: number[][]; V: number[][]; explained: number }
}
export const BM25_K1 = 1.5, BM25_B = 0.75

function tfidfOf(ts: string[], termId: Map<string, number>, idf: number[]): SparseVec {
  const counts = new Map<number, number>()
  for (const t of ts) { const id = termId.get(t); if (id !== undefined) counts.set(id, (counts.get(id) ?? 0) + 1) }
  const v: SparseVec = new Map()
  let n2 = 0
  for (const [id, c] of counts) { const w = (1 + Math.log(c)) * idf[id]; v.set(id, w); n2 += w * w }
  const n = Math.sqrt(n2) || 1
  for (const [id, w] of v) v.set(id, w / n)
  return v
}
export const sparseDot = (a: SparseVec, b: SparseVec) => { let s = 0; const [x, y] = a.size < b.size ? [a, b] : [b, a]; for (const [k, v] of x) { const w = y.get(k); if (w !== undefined) s += v * w } return s }
export const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
export const norm = (a: number[]) => Math.sqrt(dot(a, a))
export const cosine = (a: number[], b: number[]) => { const n = norm(a) * norm(b); return n === 0 ? 0 : dot(a, b) / n }

export function buildIndex(chunks: Chunk[], k = 16): Index {
  const termId = new Map<string, number>(), vocab: string[] = []
  for (const c of chunks) for (const t of c.terms) if (!termId.has(t)) { termId.set(t, vocab.length); vocab.push(t) }
  const N = chunks.length, df = new Array(vocab.length).fill(0)
  for (const c of chunks) for (const id of new Set(c.terms.map((t) => termId.get(t)!))) df[id]++
  const idf = df.map((d) => Math.log((N + 1) / (d + 1)) + 1)
  const bm25idf = df.map((d) => Math.log(1 + (N - d + 0.5) / (d + 0.5)))
  const tfidf = chunks.map((c) => tfidfOf(c.terms, termId, idf))
  const lengths = chunks.map((c) => c.terms.length), avgLen = lengths.reduce((a, b) => a + b, 0) / (N || 1)
  // LSA via the N × N Gram matrix G = A Aᵀ = U Σ² Uᵀ
  const G = chunks.map((_, i) => chunks.map((__, j) => sparseDot(tfidf[i], tfidf[j])))
  const { values, vectors } = eigh(G)
  const kk = Math.max(1, Math.min(k, N, values.filter((v) => v > 1e-10).length))
  const sigma = values.slice(0, kk).map((v) => Math.sqrt(Math.max(0, v)))
  const U = chunks.map((_, i) => Array.from({ length: kk }, (__, c) => vectors[i][c]))
  // deterministic sign: make the largest-magnitude entry of each component positive
  for (let c = 0; c < kk; c++) { let m = 0; for (let i = 0; i < N; i++) if (Math.abs(U[i][c]) > Math.abs(m)) m = U[i][c]; if (m < 0) for (let i = 0; i < N; i++) U[i][c] = -U[i][c] }
  const coords = U.map((row) => row.map((u, c) => u * sigma[c]))
  // V = Aᵀ U Σ⁻¹  (|vocab| × k)
  const V = vocab.map(() => new Array(kk).fill(0))
  tfidf.forEach((vec, i) => { for (const [t, w] of vec) for (let c = 0; c < kk; c++) V[t][c] += (w * U[i][c]) / (sigma[c] || 1) })
  const total = values.reduce((s, v) => s + Math.max(0, v), 0)
  return { chunks, vocab, termId, df, idf, bm25idf, tfidf, lengths, avgLen, lsa: { k: kk, sigma, coords, V, explained: total ? sigma.reduce((s, x) => s + x * x, 0) / total : 0 } }
}

export interface QueryVec { text: string; terms: string[]; known: string[]; unknown: string[]; tfidf: SparseVec; dense: number[] }
export function embedQuery(ix: Index, text: string): QueryVec {
  const ts = terms(text)
  const tf = tfidfOf(ts, ix.termId, ix.idf)
  const dense = new Array(ix.lsa.k).fill(0)
  for (const [t, w] of tf) for (let c = 0; c < ix.lsa.k; c++) dense[c] += w * ix.lsa.V[t][c]
  return { text, terms: ts, known: ts.filter((t) => ix.termId.has(t)), unknown: ts.filter((t) => !ix.termId.has(t)), tfidf: tf, dense }
}

/** BM25 with a per-term breakdown */
export function bm25(ix: Index, q: QueryVec, i: number): { score: number; parts: { term: string; tf: number; idf: number; contrib: number }[] } {
  const parts: { term: string; tf: number; idf: number; contrib: number }[] = []
  const counts = new Map<string, number>()
  for (const t of ix.chunks[i].terms) counts.set(t, (counts.get(t) ?? 0) + 1)
  const len = ix.lengths[i]
  for (const t of new Set(q.terms)) {
    const id = ix.termId.get(t)
    if (id === undefined) continue
    const tf = counts.get(t) ?? 0
    const contrib = tf ? (ix.bm25idf[id] * tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + (BM25_B * len) / (ix.avgLen || 1))) : 0
    parts.push({ term: t, tf, idf: ix.bm25idf[id], contrib })
  }
  return { score: parts.reduce((s, p) => s + p.contrib, 0), parts }
}

/* ------------------------------------------------------------------ IVF (approximate) index */
export interface IVF { nlist: number; centroids: number[][]; assign: number[] }
/** seeded k-means (k-means++ style seeding with a fixed RNG, Lloyd iterations to convergence or 50 steps) */
export function kmeans(X: number[][], k: number, seed: number): IVF {
  const r = rng(seed * 92821 + 7), n = X.length, kk = Math.max(1, Math.min(k, n))
  const cents: number[][] = [X[Math.floor(r() * n)].slice()]
  while (cents.length < kk) {
    const d2 = X.map((x) => Math.min(...cents.map((c) => x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0))))
    const tot = d2.reduce((a, b) => a + b, 0)
    let u = r() * tot, pick = 0
    for (; pick < n - 1; pick++) { u -= d2[pick]; if (u <= 0) break }
    cents.push(X[pick].slice())
  }
  let assign = new Array(n).fill(0)
  for (let it = 0; it < 50; it++) {
    const next = X.map((x) => { let best = 0, bd = Infinity; cents.forEach((c, j) => { const d = x.reduce((s, v, q) => s + (v - c[q]) ** 2, 0); if (d < bd) { bd = d; best = j } }); return best })
    const changed = next.some((a, i) => a !== assign[i]) || it === 0
    assign = next
    cents.forEach((c, j) => { const mem = X.filter((_, i) => assign[i] === j); if (mem.length) for (let q = 0; q < c.length; q++) c[q] = mem.reduce((s, m) => s + m[q], 0) / mem.length })
    if (!changed) break
  }
  return { nlist: kk, centroids: cents, assign }
}
/** clusters searched for a query: the nprobe centroids with the highest cosine similarity */
export const probeClusters = (ivf: IVF, q: number[], nprobe: number) => ivf.centroids.map((c, j) => ({ j, s: cosine(q, c) })).sort((a, b) => b.s - a.s || a.j - b.j).slice(0, Math.max(1, nprobe)).map((x) => x.j)

/* ------------------------------------------------------------------ retrieval */
export type Method = 'dense' | 'tfidf' | 'bm25' | 'hybrid'
export type Fusion = 'rrf' | 'weighted'
export interface RetrieveConfig { method: Method; k: number; fusion: Fusion; alpha: number; index: 'flat' | 'ivf'; nlist: number; nprobe: number; seed: number }
export interface Hit { i: number; dense: number; tfidf: number; bm25: number; rankDense: number; rankBm25: number; fused: number; score: number; rank: number; searched: boolean }
export const RRF_K = 60

const ranksOf = (scores: number[], allowed: boolean[]) => {
  const order = scores.map((s, i) => ({ s, i })).filter((x) => allowed[x.i]).sort((a, b) => b.s - a.s || a.i - b.i)
  const rk = new Array(scores.length).fill(Infinity)
  order.forEach((x, r) => (rk[x.i] = r + 1))
  return rk
}
const minmax = (v: number[], allowed: boolean[]) => { const a = v.filter((_, i) => allowed[i]); const lo = Math.min(...a), hi = Math.max(...a); return v.map((x) => (hi - lo > 1e-12 ? (x - lo) / (hi - lo) : 0)) }

/** score every chunk with every method, then rank by the chosen method. Returns all chunks, ranked. */
export function scoreAll(ix: Index, q: QueryVec, c: RetrieveConfig, ivf?: IVF | null): Hit[] {
  const N = ix.chunks.length
  const probes = c.index === 'ivf' && ivf ? new Set(probeClusters(ivf, q.dense, c.nprobe)) : null
  const allowed = ix.chunks.map((_, i) => !probes || probes.has(ivf!.assign[i]))
  const dense = ix.lsa.coords.map((v) => cosine(q.dense, v))
  const tfidf = ix.tfidf.map((v) => sparseDot(q.tfidf, v))
  const bm = ix.chunks.map((_, i) => bm25(ix, q, i).score)
  const rD = ranksOf(dense, allowed), rB = ranksOf(bm, allowed)
  const nD = minmax(dense, allowed), nB = minmax(bm, allowed)
  const fused = ix.chunks.map((_, i) => (c.fusion === 'rrf' ? (Number.isFinite(rD[i]) ? 1 / (RRF_K + rD[i]) : 0) + (Number.isFinite(rB[i]) ? 1 / (RRF_K + rB[i]) : 0) : c.alpha * nD[i] + (1 - c.alpha) * nB[i]))
  const score = c.method === 'dense' ? dense : c.method === 'tfidf' ? tfidf : c.method === 'bm25' ? bm : fused
  const rank = ranksOf(score, allowed)
  return Array.from({ length: N }, (_, i) => ({ i, dense: dense[i], tfidf: tfidf[i], bm25: bm[i], rankDense: rD[i], rankBm25: rB[i], fused: fused[i], score: score[i], rank: rank[i], searched: allowed[i] }))
    .sort((a, b) => a.rank - b.rank || a.i - b.i)
}
export const topK = (hits: Hit[], k: number) => hits.filter((h) => Number.isFinite(h.rank)).slice(0, k)

/* ------------------------------------------------------------------ reranking */
export type Reranker = 'none' | 'mmr' | 'coverage'
export interface Reranked { hit: Hit; before: number; after: number; rerankScore: number; detail: string }
/**
 * Rerank the first `pool` candidates and keep `k`.
 * MMR: argmax λ·rel(d) − (1 − λ)·max_{s ∈ S} cos(d, s), rel = min-max-normalized first-stage score, cos on dense vectors.
 * Coverage (educational heuristic, not a trained model): fraction of distinct query terms present + 0.5 · shortest
 * window containing all present query terms, scored as present/window.
 */
export function rerank(ix: Index, q: QueryVec, hits: Hit[], how: Reranker, k: number, pool: number, lambda = 0.7): Reranked[] {
  const cand = topK(hits, Math.max(k, pool))
  if (how === 'none' || !cand.length) return cand.slice(0, k).map((h, r) => ({ hit: h, before: r + 1, after: r + 1, rerankScore: h.score, detail: 'first-stage score' }))
  if (how === 'mmr') {
    const lo = Math.min(...cand.map((h) => h.score)), hi = Math.max(...cand.map((h) => h.score))
    const rel = (h: Hit) => (hi - lo > 1e-12 ? (h.score - lo) / (hi - lo) : 1)
    const chosen: Reranked[] = [], left = cand.slice()
    while (chosen.length < Math.min(k, cand.length)) {
      let best = 0, bestV = -Infinity, bestRed = 0
      left.forEach((h, n) => {
        const red = chosen.length ? Math.max(...chosen.map((c) => cosine(ix.lsa.coords[h.i], ix.lsa.coords[c.hit.i]))) : 0
        const v = lambda * rel(h) - (1 - lambda) * red
        if (v > bestV + 1e-15) { bestV = v; best = n; bestRed = red }
      })
      const h = left.splice(best, 1)[0]
      chosen.push({ hit: h, before: cand.indexOf(h) + 1, after: chosen.length + 1, rerankScore: bestV, detail: `λ·rel ${(lambda * rel(h)).toFixed(3)} − (1−λ)·max cos ${((1 - lambda) * bestRed).toFixed(3)}` })
    }
    return chosen
  }
  const qs = [...new Set(q.terms)]
  const scored = cand.map((h, n) => {
    const ts = ix.chunks[h.i].terms, present = qs.filter((t) => ts.includes(t))
    let win = Infinity
    if (present.length) for (let a = 0; a < ts.length; a++) { const seen = new Set<string>(); for (let b = a; b < ts.length; b++) { if (present.includes(ts[b])) seen.add(ts[b]); if (seen.size === present.length) { win = Math.min(win, b - a + 1); break } } }
    const cov = qs.length ? present.length / qs.length : 0, prox = present.length ? present.length / win : 0
    return { hit: h, before: n + 1, after: 0, rerankScore: cov + 0.5 * prox, detail: `coverage ${present.length}/${qs.length} + 0.5 × proximity ${prox.toFixed(3)}` }
  })
  scored.sort((a, b) => b.rerankScore - a.rerankScore || a.before - b.before)
  return scored.slice(0, k).map((s, r) => ({ ...s, after: r + 1 }))
}

/* ------------------------------------------------------------------ prompt */
export interface PromptConfig { budget: number; instructions: boolean }
export interface Prompt { text: string; included: number[]; dropped: number[]; tokens: number; contextTokens: number; parts: { kind: 'system' | 'context' | 'question'; text: string; chunk?: number; cite?: number }[] }
/** approximate token count used throughout the lab: words × 1.3, rounded up (stated in the UI) */
export const approxTokens = (s: string) => Math.ceil(words(s).length * 1.3)
export const SYSTEM_PROMPT = 'Answer the question using only the numbered context passages. Cite passages like [1]. If the context does not contain the answer, say that you do not know.'
export function buildPrompt(ix: Index, question: string, ranked: number[], c: PromptConfig): Prompt {
  const parts: Prompt['parts'] = []
  if (c.instructions) parts.push({ kind: 'system', text: SYSTEM_PROMPT })
  const included: number[] = [], dropped: number[] = []
  let ctx = 0, full = false
  // passages are added in rank order; the first one that does not fit closes the context (everything after it is dropped)
  for (const i of ranked) {
    const t = approxTokens(ix.chunks[i].text) + 4
    if (!full && ctx + t <= c.budget) { included.push(i); ctx += t; parts.push({ kind: 'context', text: ix.chunks[i].text, chunk: i, cite: included.length }) } else { full = true; dropped.push(i) }
  }
  parts.push({ kind: 'question', text: question })
  const text = parts.map((p) => (p.kind === 'system' ? `System: ${p.text}` : p.kind === 'context' ? `[${p.cite}] (${ix.chunks[p.chunk!].docTitle}) ${p.text}` : `Question: ${p.text}\nAnswer:`)).join('\n\n')
  return { text, included, dropped, tokens: approxTokens(text), contextTokens: ctx, parts }
}

/* ------------------------------------------------------------------ generation (extractive simulation) */
export interface SentenceScore { text: string; chunk: number; cite: number; score: number; matched: string[] }
export interface Answer { text: string; abstained: boolean; used: SentenceScore[]; candidates: SentenceScore[]; support: number; threshold: number }
/**
 * Extractive answer: score every sentence of the included passages by the BM25 idf of the distinct query terms it
 * contains, divided by the total idf of all distinct query terms (terms absent from the corpus count with df = 0), so
 * support ∈ [0, 1]. Answer with the best sentence (and a
 * second one from another passage if it scores ≥ 80 % of the best and is not a near-duplicate), citing passage numbers;
 * abstain below the threshold.
 */
export function generate(ix: Index, q: QueryVec, prompt: Prompt, threshold: number): Answer {
  const known = [...new Set(q.known)]
  // query terms the corpus never contains count with the largest possible idf (df = 0): they are unsupported
  const unknownIdf = Math.log(1 + (ix.chunks.length + 0.5) / 0.5)
  const totalIdf = known.reduce((s, t) => s + ix.bm25idf[ix.termId.get(t)!], 0) + new Set(q.unknown).size * unknownIdf
  const candidates: SentenceScore[] = []
  prompt.included.forEach((ci, n) => {
    for (const s of sentences(ix.chunks[ci].text)) {
      const ts = new Set(terms(s)), matched = known.filter((t) => ts.has(t))
      const score = totalIdf ? matched.reduce((a, t) => a + ix.bm25idf[ix.termId.get(t)!], 0) / totalIdf : 0
      candidates.push({ text: s, chunk: ci, cite: n + 1, score, matched })
    }
  })
  candidates.sort((a, b) => b.score - a.score || a.cite - b.cite)
  const best = candidates[0]
  if (!best || best.score < threshold) return { text: 'I don\'t know: the retrieved passages do not contain enough support to answer this question.', abstained: true, used: [], candidates, support: best?.score ?? 0, threshold }
  const used = [best]
  // a second sentence from another passage only if it adds information (term Jaccard with the first < 0.5;
  // overlapping chunks otherwise produce near-duplicate fragments)
  const jac = (a: string, b: string) => { const A = new Set(terms(a)), B = new Set(terms(b)); const inter = [...A].filter((t) => B.has(t)).length; return inter / (new Set([...A, ...B]).size || 1) }
  const second = candidates.find((s) => s.chunk !== best.chunk && s.score >= 0.8 * best.score && jac(s.text, best.text) < 0.5)
  if (second) used.push(second)
  return { text: used.map((s) => `${s.text.replace(/\s+$/, '')} [${s.cite}]`).join(' '), abstained: false, used, candidates, support: best.score, threshold }
}

/* ------------------------------------------------------------------ grounding / citation checks */
/** share of the answer's content terms that occur in the given passages (a simple faithfulness proxy) */
export function supportRatio(answer: string, passages: string[]): { ratio: number; supported: string[]; unsupported: string[] } {
  const a = [...new Set(terms(answer.replace(/\[\d+\]/g, '')))]
  const pool = new Set(passages.flatMap((p) => terms(p)))
  const supported = a.filter((t) => pool.has(t)), unsupported = a.filter((t) => !pool.has(t))
  return { ratio: a.length ? supported.length / a.length : 1, supported, unsupported }
}
/** every [n] cites an included passage, and the cited passage contains the cited sentence */
export function verifyCitations(ans: Answer, prompt: Prompt, ix: Index) {
  return ans.used.map((s) => ({ cite: s.cite, valid: s.cite >= 1 && s.cite <= prompt.included.length && ix.chunks[prompt.included[s.cite - 1]].text.includes(s.text) }))
}

/* ------------------------------------------------------------------ evaluation */
export interface QEval { qid: string; retrieved: number[]; goldChunks: number[]; goldRank: number; recall: boolean; rr: number; split: boolean; inPrompt: boolean; answerCorrect: boolean; abstained: boolean; answer: Answer }
export interface PipelineConfig { chunk: ChunkConfig; dims: number; retrieve: RetrieveConfig; reranker: Reranker; pool: number; lambda: number; prompt: PromptConfig; threshold: number; rewrite: boolean }
export interface Pipeline { chunks: Chunk[]; index: Index; ivf: IVF | null }
export function buildPipeline(docs: RagDoc[], c: Pick<PipelineConfig, 'chunk' | 'dims' | 'retrieve'>): Pipeline {
  const chunks = chunkCorpus(docs, c.chunk)
  const index = buildIndex(chunks, c.dims)
  return { chunks, index, ivf: c.retrieve.index === 'ivf' ? kmeans(index.lsa.coords, c.retrieve.nlist, c.retrieve.seed) : null }
}
export interface RunResult { q: QueryVec; rewrite: { text: string; applied: string[] }; hits: Hit[]; reranked: Reranked[]; prompt: Prompt; answer: Answer }
export function runQuery(p: Pipeline, question: string, c: PipelineConfig): RunResult {
  const rw = c.rewrite ? rewriteQuery(question) : { text: question, applied: [] }
  const q = embedQuery(p.index, rw.text)
  const hits = scoreAll(p.index, q, c.retrieve, p.ivf)
  const reranked = rerank(p.index, q, hits, c.reranker, c.retrieve.k, c.pool, c.lambda)
  const prompt = buildPrompt(p.index, question, reranked.map((r) => r.hit.i), c.prompt)
  return { q, rewrite: rw, hits, reranked, prompt, answer: generate(p.index, q, prompt, c.threshold) }
}
/** gold chunks contain the whole gold span; if none does, the span was split by chunking and partial chunks count */
export function goldChunksFor(chunks: Chunk[], gold: string): { ids: number[]; split: boolean } {
  if (!gold) return { ids: [], split: false }
  const whole = chunks.filter((c) => containsSpan(c.text, gold)).map((c) => c.id)
  if (whole.length) return { ids: whole, split: false }
  const g = terms(gold), part = chunks.filter((c) => { const t = new Set(c.terms); return g.filter((x) => t.has(x)).length / g.length >= 0.5 }).map((c) => c.id)
  return { ids: part, split: true }
}
export function evaluate(p: Pipeline, qs: RagQuestion[], c: PipelineConfig): QEval[] {
  return qs.map((qq) => {
    const run = runQuery(p, qq.q, c)
    const g = goldChunksFor(p.chunks, qq.gold)
    const ranked = run.reranked.map((r) => r.hit.i)
    const firstStage = run.hits.filter((h) => Number.isFinite(h.rank)).map((h) => h.i)
    const goldRank = g.ids.length ? Math.min(...g.ids.map((i) => { const r = firstStage.indexOf(i); return r < 0 ? Infinity : r + 1 })) : Infinity
    const recall = g.ids.some((i) => ranked.includes(i))
    const inPrompt = g.ids.some((i) => run.prompt.included.includes(i))
    const correct = qq.key ? !run.answer.abstained && run.answer.text.toLowerCase().includes(qq.key.toLowerCase()) : run.answer.abstained
    return { qid: qq.id, retrieved: ranked, goldChunks: g.ids, goldRank, recall, rr: Number.isFinite(goldRank) ? 1 / goldRank : 0, split: g.split, inPrompt, answerCorrect: correct, abstained: run.answer.abstained, answer: run.answer }
  })
}
export function summarize(ev: QEval[], qs: RagQuestion[]) {
  const ans = ev.filter((_, i) => qs[i].gold)
  return {
    recallAtK: ans.length ? ans.filter((e) => e.recall).length / ans.length : 0,
    mrr: ans.length ? ans.reduce((s, e) => s + e.rr, 0) / ans.length : 0,
    answerAccuracy: ev.length ? ev.filter((e) => e.answerCorrect).length / ev.length : 0,
    splits: ev.filter((e) => e.split).length,
  }
}

/** the default experiment (used by the lab, the lessons and the tests) */
export const DEFAULT_CONFIG: PipelineConfig = {
  chunk: { size: 40, overlap: 10, strategy: 'sentence' }, dims: 8,
  retrieve: { method: 'hybrid', k: 3, fusion: 'rrf', alpha: 0.5, index: 'flat', nlist: 4, nprobe: 1, seed: 1 },
  reranker: 'none', pool: 8, lambda: 0.7, prompt: { budget: 220, instructions: true }, threshold: 0.4, rewrite: false,
}
