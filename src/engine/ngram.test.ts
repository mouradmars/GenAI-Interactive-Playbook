import { describe, expect, it } from 'vitest'
import {
  BOS, EOS, UNK, NGramModel, defaultConfig, distSum, generate, isCorrect, makeChallenges, makeScorer, parseAnswer,
  scoreSentences, sentencesOf, tuneLambdas, type Method, type ModelConfig, type Order,
} from './ngram'
import { NGRAM_EXAMPLES } from '../modules/ngram/examples'

// Same structure as the classic textbook example (Jurafsky & Martin, ch. 3), with our own words
const SAM = 'I am Lina. Lina I am. I do not like cold tea and figs.'
const sam = new NGramModel(SAM)
const cfg = (p: Partial<ModelConfig>): ModelConfig => ({ ...defaultConfig, ...p })

describe('counting conventions', () => {
  it('splits sentences and lower-cases words', () => {
    expect(sentencesOf('Hello world! Bye.\nok')).toEqual([['hello', 'world'], ['bye'], ['ok']])
  })
  it('vocabulary = training words + </s> + <unk>', () => {
    expect(sam.V).toBe(10 + 2)
    expect(sam.vocab).toContain(EOS)
    expect(sam.vocab).toContain(UNK)
    expect(sam.vocab).not.toContain(BOS)
  })
  it('counts predicted positions only', () => {
    expect(sam.N).toBe(3 + 1 + 3 + 1 + 8 + 1) // 14 words + 3 </s>
    expect([...sam.uni.values()].reduce((a, b) => a + b, 0)).toBe(sam.N)
    expect([...sam.bi.values()].reduce((a, b) => a + b, 0)).toBe(sam.N)
    expect([...sam.tri.values()].reduce((a, b) => a + b, 0)).toBe(sam.N)
    expect(sam.ctx1.get(BOS)).toBe(3)
  })
})

describe('maximum likelihood reproduces the textbook bigram example', () => {
  const s = makeScorer(sam, cfg({ order: 2, method: 'mle' }))
  const P = (w: string, v: string) => s.p(w, [BOS, v])
  it.each([
    ['i', BOS, 2 / 3], ['lina', BOS, 1 / 3], ['am', 'i', 2 / 3], [EOS, 'lina', 1 / 2], ['lina', 'am', 1 / 2], ['do', 'i', 1 / 3],
  ])('P(%s | %s)', (w, v, p) => expect(P(w, v)).toBeCloseTo(p, 12))
  it('undefined context gives NaN and an unseen bigram gives 0', () => {
    expect(Number.isNaN(s.p('i', [BOS, UNK]))).toBe(true)
    expect(P('figs', 'i')).toBe(0)
  })
  it('sentence probability is the product of bigram probabilities', () => {
    const r = scoreSentences(s, 'I am Lina')
    expect(r.prob).toBeCloseTo((2 / 3) * (2 / 3) * (1 / 2) * (1 / 2), 12)
    expect(r.crossEntropy).toBeCloseTo(-Math.log2(r.prob) / 4, 12)
    expect(r.perplexity).toBeCloseTo(2 ** r.crossEntropy, 12)
    expect(r.zeroAt).toEqual([])
  })
  it('an unseen bigram makes perplexity infinite', () => {
    const r = scoreSentences(s, 'I like Lina')
    expect(r.zeroAt.length).toBeGreaterThan(0)
    expect(r.perplexity).toBe(Infinity)
  })
})

describe('every smoothed method is a proper distribution over V', () => {
  const methods: Method[] = ['laplace', 'interp', 'backoff', 'kn']
  const corpora = [SAM, ...NGRAM_EXAMPLES.map((e) => e.corpus)]
  for (const text of corpora.slice(0, 3)) {
    const m = new NGramModel(text)
    const words = m.vocab.slice(0, 6)
    const hists: string[][] = [[BOS, BOS], [BOS, words[0]], [words[0], words[1]], [UNK, UNK], [words[2], UNK], [BOS, EOS]]
    for (const method of methods)
      for (const order of [1, 2, 3] as Order[])
        it(`${method} order ${order} · ${text.slice(0, 18)}…`, () => {
          const s = makeScorer(m, cfg({ method, order, k: 0.37, dBackoff: 0.6, dKN: 0.75 }))
          for (const h of hists) {
            expect(distSum(s, h)).toBeCloseTo(1, 10)
            for (const w of m.vocab) expect(s.p(w, h)).toBeGreaterThan(0)
          }
        })
  }
})

describe('smoothing specifics', () => {
  it('add-k with k = 0 equals maximum likelihood', () => {
    const a = makeScorer(sam, cfg({ method: 'laplace', k: 0 })), b = makeScorer(sam, cfg({ method: 'mle' }))
    for (const w of sam.vocab) expect(a.p(w, [BOS, 'i'])).toBeCloseTo(b.p(w, [BOS, 'i']), 14)
  })
  it('add-one bigram: (c + 1) / (c(v·) + V)', () => {
    const s = makeScorer(sam, cfg({ method: 'laplace', k: 1 }))
    expect(s.p('am', [BOS, 'i'])).toBeCloseTo((2 + 1) / (3 + 12), 14)
    expect(s.p('figs', [BOS, 'i'])).toBeCloseTo(1 / 15, 14)
  })
  it('interpolation is the weighted sum of its components', () => {
    const s = makeScorer(sam, cfg({ method: 'interp', order: 2, lambdas: [0.1, 0.3, 0.6, 0] }))
    expect(s.p('am', [BOS, 'i'])).toBeCloseTo(0.1 / 12 + 0.3 * (2 / 17) + 0.6 * (2 / 3), 14)
  })
  it('backoff uses the discounted higher order when it was seen', () => {
    const s = makeScorer(sam, cfg({ method: 'backoff', order: 2, dBackoff: 0.5 }))
    const b = s.explain('am', [BOS, 'i'])
    expect(b.usedOrder).toBe(2)
    expect(b.p).toBeCloseTo((2 - 0.5) / 3, 14)
    expect(s.explain('figs', [BOS, 'i']).usedOrder).toBe(1)
  })
})

describe('Kneser-Ney continuation statistics', () => {
  // "francisco" is frequent but only ever follows "san"; "glasses" is rarer but follows many words
  const text = 'san francisco is big. san francisco is foggy. i like san francisco. i saw san francisco. ' +
    'my glasses broke. new glasses help. his glasses shine. reading glasses work.'
  const m = new NGramModel(text)
  it('N₁₊(• w) counts distinct left neighbours', () => {
    expect(m.c1('francisco')).toBe(4)
    expect(m.contUni.get('francisco')).toBe(1)
    expect(m.c1('glasses')).toBe(4)
    expect(m.contUni.get('glasses')).toBe(4)
    expect(m.contBiTypes).toBe(m.bi.size)
  })
  it('N₁₊(• v w) counts distinct u in trigrams', () => {
    // trigrams ending in "san francisco" start with <s> (twice), "like" and "saw": 3 distinct u
    expect(m.contBi.get(['san', 'francisco'].join('\u0001'))).toBe(3)
  })
  it('continuation unigram prefers the word seen in more contexts', () => {
    const s = makeScorer(m, cfg({ method: 'kn', order: 2, dKN: 0.75 }))
    // unseen context → pure continuation distribution
    expect(s.p('glasses', [BOS, 'without'])).toBeGreaterThan(s.p('francisco', [BOS, 'without']))
    const raw = makeScorer(m, cfg({ method: 'mle', order: 1 }))
    expect(raw.p('glasses', [])).toBeCloseTo(raw.p('francisco', []), 12)
  })
  it('matches a hand computation', () => {
    const D = 0.75
    const s = makeScorer(m, cfg({ method: 'kn', order: 2, dKN: D }))
    const B = m.bi.size, T = m.contUni.size, V = m.V
    const pcont = (w: string) => Math.max((m.contUni.get(w) ?? 0) - D, 0) / B + (D * T) / B / V
    const cv = m.ctx1.get('san')!, types = m.next1.get('san')!.size
    expect(s.p('francisco', [BOS, 'san'])).toBeCloseTo(Math.max(4 - D, 0) / cv + (D * types) / cv * pcont('francisco'), 14)
  })
})

describe('Kneser-Ney trigram and backoff identities', () => {
  const text = NGRAM_EXAMPLES[1].corpus // weather reports: many repeated trigram contexts
  const m = new NGramModel(text)
  const K = (...t: string[]) => t.join('\u0001')
  it('continuation counts satisfy their summation identities', () => {
    let s1 = 0
    for (const v of m.contUni.values()) s1 += v
    expect(s1).toBe(m.bi.size) // Σ_w N₁₊(• w) = N₁₊(• •)
    const byV = new Map<string, number>()
    for (const [k, c] of m.contBi) { const v = k.split('\u0001')[0]; byV.set(v, (byV.get(v) ?? 0) + c) }
    for (const [v, c] of byV) expect(m.contBiCtx.get(v)).toBe(c) // Σ_w N₁₊(• v w) = N₁₊(• v •)
    let s2 = 0
    for (const c of m.contBi.values()) s2 += c
    expect(s2).toBe(m.tri.size) // Σ_{v,w} N₁₊(• v w) = trigram types
  })
  it('trigram KN matches a hand computation through all three levels', () => {
    const D = 0.6
    const s = makeScorer(m, cfg({ method: 'kn', order: 3, dKN: D }))
    const [u, v] = ['will', 'be']
    const V = m.V, B = m.bi.size, T = m.contUni.size
    const pc1 = (w: string) => Math.max((m.contUni.get(w) ?? 0) - D, 0) / B + (D * T) / B / V
    const nvv = m.contBiCtx.get(v)!, typesV = m.contNext.get(v)!.size
    const pc2 = (w: string) => Math.max((m.contBi.get(K(v, w)) ?? 0) - D, 0) / nvv + (D * typesV) / nvv * pc1(w)
    const cuv = m.ctx2.get(K(u, v))!, typesUV = m.next2.get(K(u, v))!.size
    const p3 = (w: string) => Math.max(m.c3(u, v, w) - D, 0) / cuv + (D * typesUV) / cuv * pc2(w)
    for (const w of ['sunny', 'clear', 'rainy', 'hot', UNK]) expect(s.p(w, [u, v])).toBeCloseTo(p3(w), 14)
    expect(m.c3(u, v, 'sunny')).toBeGreaterThan(0)
    expect(m.c3(u, v, 'hot')).toBe(0)
  })
  it('backoff α hands exactly the discounted mass to unseen words', () => {
    const d = 0.4
    const s = makeScorer(m, cfg({ method: 'backoff', order: 2, dBackoff: d }))
    const v = 'and', seen = m.next1.get(v)!, c = m.ctx1.get(v)!
    let unseenMass = 0
    for (const w of m.vocab) if (!seen.has(w)) unseenMass += s.p(w, [BOS, v])
    expect(unseenMass).toBeCloseTo((d * seen.size) / c, 12)
  })
  it('a backoff discount of 1 never zeroes a seen n-gram', () => {
    const s = makeScorer(sam, cfg({ method: 'backoff', order: 2, dBackoff: 1 }))
    expect(s.p('do', [BOS, 'i'])).toBeGreaterThan(0) // c(i do) = 1
    expect(distSum(s, [BOS, 'i'])).toBeCloseTo(1, 10)
  })
})

describe('determinism', () => {
  it('seeded generation repeats exactly', () => {
    const s = makeScorer(sam, cfg({ method: 'kn', order: 3 }))
    expect(generate(s, 11)).toEqual(generate(s, 11))
    expect(generate(s, 11).at(-1)?.w === EOS || generate(s, 11).length === 20).toBe(true)
  })
  it('challenge sets are a pure function of corpus and seed', () => {
    expect(makeChallenges(sam, 3)).toEqual(makeChallenges(sam, 3))
    for (let seed = 0; seed < 40; seed++) expect(makeChallenges(sam, seed).find((q) => q.id === 'laplace')!.prompt).not.toContain(UNK)
    for (const c of makeChallenges(sam, 3)) expect(Number.isFinite(c.answer)).toBe(true)
  })
  it('answer parsing accepts fractions and checks tolerance', () => {
    expect(parseAnswer('2/3')).toBeCloseTo(2 / 3)
    expect(isCorrect(2 / 3, parseAnswer('0.667'))).toBe(true)
    expect(isCorrect(2 / 3, parseAnswer('0.6'))).toBe(false)
  })
  it('lambda tuning finds weights at least as good as the defaults', () => {
    const ex = NGRAM_EXAMPLES[0]
    const m = new NGramModel(ex.corpus)
    const c = cfg({ method: 'interp', order: 3 })
    const best = tuneLambdas(m, c, ex.heldOut)
    expect(best.pp).toBeLessThanOrEqual(scoreSentences(makeScorer(m, c), ex.heldOut).perplexity + 1e-9)
  })
})
