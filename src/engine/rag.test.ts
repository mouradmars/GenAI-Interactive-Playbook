import { describe, expect, it } from 'vitest'
import { CORPORA, corpusById, type RagDoc } from '../data/rag-corpora'
import {
  RRF_K, approxTokens, bm25, buildIndex, buildPipeline, buildPrompt, chunkCorpus, chunkDoc, containsSpan, cosine, embedQuery, evaluate, fixedChunkCount,
  generate, goldChunksFor, kmeans, probeClusters, rerank, rewriteQuery, runQuery, scoreAll, sentences, sparseDot, stem, summarize, supportRatio, terms,
  topK, verifyCitations, words, DEFAULT_CONFIG,
} from './rag'

const DEFAULT = DEFAULT_CONFIG
const toy: RagDoc[] = [
  { id: 'a', title: 'A', text: 'Cats chase mice. Cats sleep a lot.' },
  { id: 'b', title: 'B', text: 'Dogs chase cats in the park.' },
  { id: 'c', title: 'C', text: 'Mice eat cheese.' },
]

describe('text processing', () => {
  it('stemming and stop words', () => {
    expect(stem('planets')).toBe('planet'); expect(stem('moons')).toBe('moon'); expect(stem('studies')).toBe('study'); expect(stem('class')).toBe('class')
    expect(stem('reclassified')).toBe('reclassifi'); expect(terms('The cats are chasing mice')).toEqual(['cat', 'chas', 'mice'])
    expect(sentences('One rule. Two rules! Is it 14:00? Yes.')).toEqual(['One rule.', 'Two rules!', 'Is it 14:00?', 'Yes.'])
  })
  it('query rewriting adds corpus vocabulary and reports what it did', () => {
    const r = rewriteQuery('Can I use ChatGPT for homework?')
    expect(r.applied.length).toBe(2); expect(r.text).toContain('generative AI assistants'); expect(r.text).toContain('problem sets')
    expect(rewriteQuery('How many moons?').applied).toEqual([])
  })
})

describe('chunking', () => {
  const doc = corpusById('course').docs[1]
  it('fixed windows: count formula, full coverage, exact overlap', () => {
    const n = words(doc.text).length
    for (const [size, ov] of [[10, 0], [20, 5], [25, 10], [40, 0], [200, 20]]) {
      const ch = chunkDoc(doc, { size, overlap: ov, strategy: 'fixed' })
      expect(ch.length).toBe(fixedChunkCount(n, size, ov))
      expect(ch[0].start).toBe(0); expect(ch.at(-1)!.end).toBe(n)
      ch.slice(1).forEach((c, k) => expect(c.start).toBe(ch[k].end - ov))
      ch.forEach((c) => expect(c.words).toBeLessThanOrEqual(size))
    }
  })
  it('sentence packing keeps whole sentences and covers the document', () => {
    for (const d of CORPORA.flatMap((c) => c.docs)) {
      const ch = chunkDoc(d, { size: 30, overlap: 8, strategy: 'sentence' }), sents = sentences(d.text)
      ch.forEach((c) => { const ss = sentences(c.text); ss.forEach((s) => expect(sents).toContain(s)); if (ss.length > 1) expect(c.words).toBeLessThanOrEqual(30) })
      expect(ch[0].start).toBe(0); expect(ch.at(-1)!.end).toBe(words(d.text).length)
      ch.slice(1).forEach((c, k) => expect(c.start).toBeLessThanOrEqual(ch[k].end))
    }
  })
  it('gold spans: contained when chunks are large; split detection when a boundary cuts them', () => {
    const g = corpusById('course').questions[0].gold
    expect(goldChunksFor(chunkCorpus(corpusById('course').docs, { size: 200, overlap: 0, strategy: 'fixed' }), g).split).toBe(false)
    const small = chunkCorpus(corpusById('course').docs, { size: 12, overlap: 0, strategy: 'fixed' })
    const r = goldChunksFor(small, g)
    expect(r.split).toBe(true); expect(r.ids.length).toBeGreaterThan(0)
    // overlap rescues it
    expect(goldChunksFor(chunkCorpus(corpusById('course').docs, { size: 20, overlap: 14, strategy: 'fixed' }), g).split).toBe(false)
    expect(containsSpan('a b, C d.', 'b c')).toBe(true)
  })
})

describe('scoring', () => {
  const ch = chunkCorpus(toy, { size: 50, overlap: 0, strategy: 'fixed' }), ix = buildIndex(ch, 8)
  it('TF-IDF by hand: idf = ln((N+1)/(df+1)) + 1, tf = 1 + ln(count), unit vectors', () => {
    const cat = ix.termId.get('cat')!
    expect(ix.df[cat]).toBe(2); expect(ix.idf[cat]).toBeCloseTo(Math.log(4 / 3) + 1, 14)
    ix.tfidf.forEach((v) => expect(Math.sqrt([...v.values()].reduce((s, x) => s + x * x, 0))).toBeCloseTo(1, 14))
    // chunk a: cat ×2, chase, mice, sleep, lot
    const raw = { cat: (1 + Math.log(2)) * ix.idf[cat], chase: ix.idf[ix.termId.get('chase')!], mice: ix.idf[ix.termId.get('mice')!], sleep: ix.idf[ix.termId.get('sleep')!], lot: ix.idf[ix.termId.get('lot')!] }
    const n = Math.hypot(...Object.values(raw))
    expect(ix.tfidf[0].get(cat)).toBeCloseTo(raw.cat / n, 14)
  })
  it('BM25 by hand', () => {
    const q = embedQuery(ix, 'cheese'), r = bm25(ix, q, 2)
    const N = 3, df = 1, idf = Math.log(1 + (N - df + 0.5) / (df + 0.5)), len = ix.lengths[2], avg = ix.avgLen
    expect(r.score).toBeCloseTo((idf * 1 * 2.5) / (1 + 1.5 * (1 - 0.75 + (0.75 * len) / avg)), 14)
    expect(bm25(ix, q, 0).score).toBe(0)
  })
  it('LSA: at full rank, coordinates reproduce the Gram matrix and query·doc dot products equal TF-IDF dot products', () => {
    const big = buildIndex(chunkCorpus(corpusById('solar').docs, { size: 40, overlap: 0, strategy: 'fixed' }), 100)
    const N = big.chunks.length
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) expect(big.lsa.coords[i].reduce((s, x, c) => s + x * big.lsa.coords[j][c], 0)).toBeCloseTo(sparseDot(big.tfidf[i], big.tfidf[j]), 9)
    const q = embedQuery(big, 'Which planet has two moons')
    big.lsa.coords.forEach((v, i) => expect(q.dense.reduce((s, x, c) => s + x * v[c], 0)).toBeCloseTo(sparseDot(q.tfidf, big.tfidf[i]), 9))
    expect(big.lsa.explained).toBeCloseTo(1, 9)
  })
  it('a truncated LSA is deterministic and keeps the largest singular values first', () => {
    const a = buildIndex(chunkCorpus(corpusById('solar').docs, { size: 40, overlap: 0, strategy: 'fixed' }), 4)
    const b = buildIndex(chunkCorpus(corpusById('solar').docs, { size: 40, overlap: 0, strategy: 'fixed' }), 4)
    expect(a.lsa.coords).toEqual(b.lsa.coords)
    a.lsa.sigma.slice(1).forEach((s, k) => expect(s).toBeLessThanOrEqual(a.lsa.sigma[k] + 1e-12))
    expect(a.lsa.explained).toBeGreaterThan(0); expect(a.lsa.explained).toBeLessThan(1)
  })
  it('reciprocal rank fusion by hand', () => {
    const q = embedQuery(ix, 'cats chase'), hits = scoreAll(ix, q, DEFAULT.retrieve)
    for (const h of hits) expect(h.fused).toBeCloseTo(1 / (RRF_K + h.rankDense) + 1 / (RRF_K + h.rankBm25), 14)
    const w = scoreAll(ix, q, { ...DEFAULT.retrieve, fusion: 'weighted', alpha: 1 })
    const dmax = Math.max(...w.map((h) => h.dense)); expect(w.find((h) => h.dense === dmax)!.fused).toBeCloseTo(1, 12)
    expect(cosine([1, 0], [0, 2])).toBe(0)
  })
})

describe('index, rerank, prompt, generation', () => {
  const corpus = corpusById('solar'), p = buildPipeline(corpus.docs, DEFAULT)
  it('IVF: seeded k-means; probing every cluster equals the exact search', () => {
    const iv = kmeans(p.index.lsa.coords, 4, 1)
    expect(kmeans(p.index.lsa.coords, 4, 1)).toEqual(iv)
    const q = embedQuery(p.index, 'moons of Mars')
    const flat = scoreAll(p.index, q, DEFAULT.retrieve).map((h) => h.i)
    const all = scoreAll(p.index, q, { ...DEFAULT.retrieve, index: 'ivf', nprobe: 4 }, iv).map((h) => h.i)
    expect(all).toEqual(flat)
    const one = scoreAll(p.index, q, { ...DEFAULT.retrieve, index: 'ivf', nprobe: 1 }, iv)
    const probed = new Set(probeClusters(iv, q.dense, 1))
    one.filter((h) => h.searched).forEach((h) => expect(probed.has(iv.assign[h.i])).toBe(true))
  })
  it('MMR with λ = 1 keeps the order; the first pick is always the top hit', () => {
    const q = embedQuery(p.index, 'Pluto planet'), hits = scoreAll(p.index, q, DEFAULT.retrieve)
    expect(rerank(p.index, q, hits, 'mmr', 3, 8, 1).map((r) => r.hit.i)).toEqual(topK(hits, 3).map((h) => h.i))
    expect(rerank(p.index, q, hits, 'mmr', 3, 8, 0.3)[0].hit.i).toBe(hits[0].i)
    const cov = rerank(p.index, q, hits, 'coverage', 3, 8)
    cov.slice(1).forEach((r, k) => expect(r.rerankScore).toBeLessThanOrEqual(cov[k].rerankScore))
  })
  it('the prompt respects the token budget, in rank order', () => {
    const order = topK(scoreAll(p.index, embedQuery(p.index, 'Venus temperature'), DEFAULT.retrieve), 6).map((h) => h.i)
    const pr = buildPrompt(p.index, 'q', order, { budget: 120, instructions: true })
    expect(pr.contextTokens).toBeLessThanOrEqual(120); expect([...pr.included, ...pr.dropped]).toEqual(order)
    expect(pr.included.length).toBeGreaterThan(0); expect(pr.dropped.length).toBeGreaterThan(0)
    expect(approxTokens('one two three')).toBe(4)
  })
  it('generated sentences are verbatim from the cited passages; citations verify', () => {
    const r = runQuery(p, 'How many moons does Mars have?', DEFAULT)
    expect(r.answer.abstained).toBe(false); expect(r.answer.text).toContain('Phobos')
    expect(verifyCitations(r.answer, r.prompt, p.index).every((c) => c.valid)).toBe(true)
    expect(supportRatio(r.answer.text, r.prompt.included.map((i) => p.index.chunks[i].text)).ratio).toBe(1)
  })
  it('unanswerable questions abstain; closed-book simulations are not supported by the context', () => {
    for (const c of CORPORA) {
      const pp = buildPipeline(c.docs, DEFAULT)
      for (const q of c.questions.filter((x) => !x.gold)) {
        const r = runQuery(pp, q.q, DEFAULT)
        expect(r.answer.abstained).toBe(true)
        expect(supportRatio(q.closedBook, r.prompt.included.map((i) => pp.index.chunks[i].text)).ratio).toBeLessThan(0.7)
      }
    }
  })
  it('the generator abstains when its support threshold is above the best sentence', () => {
    const q = embedQuery(p.index, 'Mars moons'), pr = buildPrompt(p.index, 'x', topK(scoreAll(p.index, q, DEFAULT.retrieve), 3).map((h) => h.i), DEFAULT.prompt)
    const a = generate(p.index, q, pr, 0.4)
    expect(generate(p.index, q, pr, a.support + 0.01).abstained).toBe(true)
  })
})

describe('evaluation (behaviour quoted in the lessons)', () => {
  const run = (id: string, cfg = DEFAULT) => { const c = corpusById(id), p = buildPipeline(c.docs, cfg); return { c, ev: evaluate(p, c.questions, cfg) } }
  it('default config: lexical-gap questions fail, query rewriting fixes them', () => {
    for (const [cid, qid] of [['course', 'c2'], ['solar', 's2'], ['glossary', 'g1']]) {
      expect(run(cid).ev.find((e) => e.qid === qid)!.answerCorrect).toBe(false)
      expect(run(cid, { ...DEFAULT, rewrite: true }).ev.find((e) => e.qid === qid)!.answerCorrect).toBe(true)
    }
  })
  it('the "five days late" question is retrieved but answered wrongly by the extractive simulation (needs reasoning)', () => {
    const e = run('course').ev.find((x) => x.qid === 'c3')!
    expect(e.recall).toBe(true); expect(e.answerCorrect).toBe(false)
  })
  it('the conflict question retrieves both Pluto documents', () => {
    const c = corpusById('solar'), p = buildPipeline(c.docs, DEFAULT), r = runQuery(p, 'Is Pluto a planet?', DEFAULT)
    const docs = r.reranked.map((x) => p.index.chunks[x.hit.i].docId)
    expect(docs).toContain('pluto2006'); expect(docs).toContain('pluto1999')
  })
  it('summary metrics and determinism', () => {
    const { c, ev } = run('glossary'), s = summarize(ev, c.questions)
    expect(s.recallAtK).toBe(1); expect(s.mrr).toBeGreaterThan(0.8)
    expect(run('glossary').ev.map((e) => e.answer.text)).toEqual(ev.map((e) => e.answer.text))
  })
  it('tiny chunks without overlap split the late-penalty span; overlap fixes it', () => {
    const cfg = { ...DEFAULT, chunk: { size: 12, overlap: 0, strategy: 'fixed' as const } }
    expect(run('course', cfg).ev.find((e) => e.qid === 'c1')!.split).toBe(true)
    expect(run('course', { ...DEFAULT, chunk: { size: 40, overlap: 20, strategy: 'fixed' } }).ev.find((e) => e.qid === 'c1')!.split).toBe(false)
  })
})

describe('the MMR example of the lesson', () => {
  it('Mars moons with 25/10 chunks: top-3 from one document; MMR (λ = 0.5) brings in three documents and keeps the evidence first', () => {
    const cfg = { ...DEFAULT_CONFIG, chunk: { size: 25, overlap: 10, strategy: 'fixed' as const } }
    const c = corpusById('solar'), p = buildPipeline(c.docs, cfg)
    const base = runQuery(p, c.questions[0].q, cfg), mmr = runQuery(p, c.questions[0].q, { ...cfg, reranker: 'mmr', lambda: 0.5 })
    expect(new Set(base.reranked.map((r) => p.chunks[r.hit.i].docId)).size).toBe(1)
    expect(new Set(mmr.reranked.map((r) => p.chunks[r.hit.i].docId)).size).toBe(3)
    expect(mmr.reranked[0].hit.i).toBe(base.reranked[0].hit.i)
    expect(mmr.answer.text).toContain('Phobos')
  })
})
