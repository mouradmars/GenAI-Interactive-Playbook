import { describe, expect, it } from 'vitest'
import { CORPORA, corpusById } from '../../data/rag-corpora'
import { DEFAULT_CONFIG, buildPipeline, cosine, embedQuery, evaluate, fixedChunkCount, runQuery, sparseDot } from '../../engine/rag'
import { check, makeChallenges } from './challenges'
import { diagnose } from './diagnose'
import { EXAMPLES, STAGES } from './lesson'
import { createRagStore, docsOf, queryText } from './store'

describe('corpora and examples', () => {
  it('every gold span is an exact substring of exactly one document; every corpus has an unanswerable question', () => {
    for (const c of CORPORA) {
      for (const q of c.questions.filter((x) => x.gold)) expect(c.docs.filter((d) => d.text.includes(q.gold)).length).toBe(1)
      expect(c.questions.some((q) => !q.gold)).toBe(true)
    }
    expect(CORPORA.length).toBeGreaterThanOrEqual(3); expect(EXAMPLES.length).toBeGreaterThanOrEqual(3); expect(STAGES.length).toBe(12)
  })
  it('example 1: the grounded answer quotes the handbook (five points), the closed-book simulation does not', () => {
    const c = corpusById('course'), p = buildPipeline(c.docs, DEFAULT_CONFIG), r = runQuery(p, c.questions[0].q, DEFAULT_CONFIG)
    expect(r.answer.text).toContain('five percentage points'); expect(c.questions[0].closedBook).toContain('10%')
  })
})

describe('store', () => {
  it('added documents are ingested; custom questions are used as queries', () => {
    const st = createRagStore()
    st.getState().addDoc('Cafeteria', 'The cafeteria serves vegetarian lasagna every Wednesday.')
    expect(docsOf(st.getState()).length).toBe(corpusById('course').docs.length + 1)
    st.getState().setCustom('What does the cafeteria serve on Wednesday?')
    const s = st.getState(), p = buildPipeline(docsOf(s), s.cfg), r = runQuery(p, queryText(s), s.cfg)
    expect(r.answer.text).toContain('lasagna')
    st.getState().setCorpus('solar'); expect(st.getState().extraDocs).toEqual([])
  })
})

describe('challenges', () => {
  const c = corpusById('glossary'), p = buildPipeline(c.docs, DEFAULT_CONFIG), q = embedQuery(p.index, c.questions[0].q)
  it('deterministic; answers agree with the engine; hand-rounded answers accepted', () => {
    for (let seed = 0; seed < 30; seed++) {
      const qs = makeChallenges(seed, { pipe: p, q, chunk: 2 })
      expect(makeChallenges(seed, { pipe: p, q, chunk: 2 })).toEqual(qs)
      const ch = qs.find((x) => x.id === 'chunks')!, m = /has (\d+) words\. Fixed windows of (\d+) words with an overlap of (\d+)/.exec(ch.prompt)!
      expect(ch.answer).toBe(fixedChunkCount(+m[1], +m[2], +m[3]))
      const cs = qs.find((x) => x.id === 'cos')!, v = /a = \[(.*?)\] and b = \[(.*?)\]/.exec(cs.prompt)!
      expect(cs.answer).toBeCloseTo(cosine(v[1].split(', ').map(Number), v[2].split(', ').map(Number)), 12)
      expect(qs.find((x) => x.id === 'live')!.answer).toBe(sparseDot(q.tfidf, p.index.tfidf[2]))
      for (const x of qs) if (x.kind === 'num') { expect(check(x, (x.answer as number).toFixed(4))).toBe(true); expect(check(x, String((x.answer as number) + 1))).toBe(false) }
      for (const x of qs) if (x.kind === 'mc') { expect(x.options).toContain(x.answer); expect(check(x, x.options![1])).toBe(x.options![1] === x.answer) }
    }
  })
})

describe('diagnosis', () => {
  it('labels each designed failure mode with the right verdict (default config)', () => {
    const verdicts: Record<string, string> = {}
    for (const c of CORPORA) {
      const p = buildPipeline(c.docs, DEFAULT_CONFIG), ev = evaluate(p, c.questions, DEFAULT_CONFIG)
      ev.forEach((e, n) => { verdicts[e.qid] = diagnose(e, c.questions[n], p, 3, c.docs, e.retrieved.map((i) => p.chunks[i].docId)).verdict })
    }
    expect(verdicts.c6).toBe('abstain-ok'); expect(verdicts.s5).toBe('abstain-ok')
    expect(verdicts.c3).toBe('reasoning'); expect(verdicts.c4).toBe('ok')
    expect(verdicts.s3).toBe('stale')
    expect(['miss', 'over-abstain']).toContain(verdicts.s2); expect(['miss', 'over-abstain']).toContain(verdicts.c2)
    expect(verdicts.s1).toBe('ok')
  })
  it('with fixed 40-word windows the final-exam question is answered from a distractor sentence', () => {
    const cfg = { ...DEFAULT_CONFIG, chunk: { size: 40, overlap: 10, strategy: 'fixed' as const } }
    const c = corpusById('course'), p = buildPipeline(c.docs, cfg), e = evaluate(p, c.questions, cfg), k = c.questions.findIndex((q) => q.id === 'c4')
    expect(diagnose(e[k], c.questions[k], p, 3, c.docs, e[k].retrieved.map((i) => p.chunks[i].docId)).verdict).toBe('distractor')
  })
  it('tiny chunks without overlap: the late-penalty question is diagnosed as a split or still answered', () => {
    const cfg = { ...DEFAULT_CONFIG, chunk: { size: 12, overlap: 0, strategy: 'fixed' as const } }
    const c = corpusById('course'), p = buildPipeline(c.docs, cfg), e = evaluate(p, c.questions, cfg)[0]
    expect(e.split).toBe(true)
    const d = diagnose(e, c.questions[0], p, 3, c.docs, e.retrieved.map((i) => p.chunks[i].docId))
    expect(['split', 'ok']).toContain(d.verdict)
  })
})
