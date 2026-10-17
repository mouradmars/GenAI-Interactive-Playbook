import { describe, expect, it } from 'vitest'
import corpus from '../data/embed-corpus.txt?raw'
import { angleDeg, cosine, dot, jacobiEigen, neighbourPreservation, norm, pca, ranks, spearman, symEigen } from './linalg'
import {
  analogy, contextualEmbeddings, cooccurrence, defaultPPMI, defaultProj, makeEmbChallenges, misleadingPairs, nanoSpace, nearest, numCorrect,
  parseNum, ppmi, project, similarityMatrix, svdEmbed, tokenizeWords, toySpace, trainPPMISpace,
} from './embeddings'
import { getModel } from './model'
import { rng } from './decoding'

describe('vector basics', () => {
  it('dot, norm, cosine, angle', () => {
    expect(dot([1, 2, 3], [4, -5, 6])).toBe(12)
    expect(norm([3, 4])).toBe(5)
    expect(cosine([1, 0], [0, 1])).toBe(0)
    expect(cosine([1, 2], [2, 4])).toBeCloseTo(1, 15)
    expect(cosine([1, 2], [-1, -2])).toBeCloseTo(-1, 15)
    expect(angleDeg([1, 0], [1, 1])).toBeCloseTo(45, 12)
    expect(() => dot([1], [1, 2])).toThrow()
  })
  it('cosine is scale invariant, the dot product is not', () => {
    const a = [0.3, -1.2, 2], b = [1, 0.5, -0.25]
    expect(cosine(a.map((x) => x * 7), b)).toBeCloseTo(cosine(a, b), 14)
    expect(dot(a.map((x) => x * 7), b)).toBeCloseTo(7 * dot(a, b), 12)
  })
})

describe('Jacobi eigen-decomposition', () => {
  const A = [[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]]
  const { values, vectors } = jacobiEigen(A)
  it('reconstructs A = Q Λ Qᵀ with orthonormal Q', () => {
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let s = 0, o = 0
        for (let k = 0; k < 4; k++) { s += vectors[i][k] * values[k] * vectors[j][k]; o += vectors[k][i] * vectors[k][j] }
        expect(s).toBeCloseTo(A[i][j], 10)
        expect(o).toBeCloseTo(i === j ? 1 : 0, 10)
      }
  })
  it('sorts eigenvalues descending and preserves the trace', () => {
    for (let k = 1; k < 4; k++) expect(values[k - 1]).toBeGreaterThanOrEqual(values[k])
    expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(4 + 2 + 3 - 1, 10)
  })
  it('rejects non-symmetric input', () => expect(() => jacobiEigen([[1, 2], [0, 1]])).toThrow(/symmetric/))
  it('the fast tridiagonal-QL solver agrees with Jacobi', () => {
    const r = rng(11)
    for (const n of [1, 2, 5, 17, 40]) {
      const B = Array.from({ length: n }, () => Array.from({ length: n }, () => r() - 0.5))
      const S = B.map((row, i) => row.map((_, j) => B[i][j] + B[j][i]))
      const J = jacobiEigen(S), Q = symEigen(S)
      J.values.forEach((v, k) => expect(Q.values[k]).toBeCloseTo(v, 9))
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let s = 0
        for (let k = 0; k < n; k++) s += Q.vectors[i][k] * Q.values[k] * Q.vectors[j][k]
        expect(s).toBeCloseTo(S[i][j], 9)
      }
    }
  })
})

describe('PCA', () => {
  it('finds the axis of a line of points and explains 100% of its variance', () => {
    const rows = [-2, -1, 0, 1, 2].map((t) => [3 * t + 1, 4 * t - 2, 0])
    const r = pca(rows, 2)
    expect(Math.abs(r.components[0][0])).toBeCloseTo(0.6, 10)
    expect(Math.abs(r.components[0][1])).toBeCloseTo(0.8, 10)
    expect(r.explained[0]).toBeCloseTo(1, 10)
    expect(r.variance[0]).toBeCloseTo((25 * (4 + 1 + 0 + 1 + 4)) / 5, 10)
  })
  it('explained variance of all components sums to 1 and coordinates are centered', () => {
    const s = toySpace()
    const r = pca(s.vecs, s.dim)
    expect(r.explained.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    for (let c = 0; c < 3; c++) expect(r.coords.reduce((a, x) => a + x[c], 0)).toBeCloseTo(0, 10)
  })
})

describe('rank statistics and neighbourhood preservation', () => {
  it('ranks average ties; Spearman is 1 for any monotone map', () => {
    expect(ranks([10, 20, 20, 5])).toEqual([2, 3.5, 3.5, 1])
    expect(spearman([1, 2, 3, 4], [1, 8, 27, 64])).toBeCloseTo(1, 12)
    expect(spearman([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1, 12)
  })
  it('a "projection" that keeps every distance scores 1; a scrambled one scores near chance', () => {
    // on unit vectors ‖u − v‖² = 2 − 2 cos(u, v), so Euclidean order = cosine order
    const r = rng(3)
    const high = Array.from({ length: 30 }, () => Array.from({ length: 6 }, () => r() - 0.5))
    const units = high.map((v) => { const n = norm(v); return v.map((x) => x / n) })
    expect(neighbourPreservation(high, units, 4).mean).toBe(1)
    const scrambled = units.map((_, i) => units[(i * 7) % 30])
    expect(neighbourPreservation(high, scrambled, 4).mean).toBeLessThan(0.4)
  })
})

describe('hand-built simulation space', () => {
  const s = toySpace()
  const W = (w: string) => s.words.indexOf(w)
  it('king − man + woman lands exactly on queen', () => {
    const r = analogy(s, W('man'), W('king'), W('woman'), 3, false, W('queen'))
    expect(r.target).toEqual(s.vecs[W('queen')])
    expect(r.ranked[0].word).toBe('queen')
    expect(r.ranked[0].score).toBeCloseTo(1, 14)
    expect(r.expectedRank).toBe(1)
  })
  it('analogies exclude their inputs', () => {
    const r = analogy(s, W('man'), W('king'), W('woman'), 20)
    expect(r.ranked.map((x) => x.word)).not.toContain('king')
    expect(r.ranked.map((x) => x.word)).not.toContain('woman')
  })
  it('similarity matrix is symmetric with unit diagonal', () => {
    const idx = [0, 1, 2, 8, 12]
    const M = similarityMatrix(s, idx)
    idx.forEach((_, i) => { expect(M[i][i]).toBeCloseTo(1, 14); idx.forEach((__, j) => expect(M[i][j]).toBeCloseTo(M[j][i], 14)) })
  })
  it('nearest neighbours are sorted and exclude the query', () => {
    const nn = nearest(s, s.vecs[W('king')], 5, 'cosine', [W('king')])
    for (let i = 1; i < nn.length; i++) expect(nn[i - 1].score).toBeGreaterThanOrEqual(nn[i].score)
    expect(nn.map((x) => x.word)).not.toContain('king')
    // dot product prefers longer vectors: banana (sweet = 1.5) beats apple for the "fruit+sweet" direction
    expect(nearest(s, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 1, 'dot')[0].word).toBe('banana')
  })
})

describe('PPMI + SVD', () => {
  // tiny corpus, computed by hand
  const sents = tokenizeWords('a b. a c. b c.')
  const { words, C } = cooccurrence(sents, 1, 1)
  it('counts symmetric, sentence-bounded co-occurrences', () => {
    expect(words).toEqual(['a', 'b', 'c'])
    expect(C).toEqual([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
  })
  it('PPMI with α = 1 matches log2(P(w,c) / P(w)P(c))', () => {
    const M = ppmi(C, 1)
    // total 6, row = col = 2 → PMI = log2((1/6) / ((2/6)(2/6))) = log2(1.5)
    expect(M[0][1]).toBeCloseTo(Math.log2(1.5), 14)
    expect(M[0][0]).toBe(0)
  })
  it('context smoothing α < 1 raises the probability of rare contexts', () => {
    const C2 = [[0, 10, 1], [10, 0, 1], [1, 1, 0]]
    expect(ppmi(C2, 0.75)[0][2]).toBeLessThan(ppmi(C2, 1)[0][2] + 1e-12)
  })
  it('SVD embedding reproduces M Mᵀ from W Wᵀ when p = 1 and k = n', () => {
    const M = [[0, 1.2, 0.3], [1.2, 0, 0.5], [0.3, 0.5, 0]]
    const { W } = svdEmbed(M, 3, 1)
    // W = U Σ, so W Wᵀ = U Σ² Uᵀ = M Mᵀ
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(dot(W[i], W[j])).toBeCloseTo(dot(M[i], M[j]), 10)
  })
  it('training on the Observatory corpus is deterministic and groups categories', () => {
    const a = trainPPMISpace(corpus, defaultPPMI), b = trainPPMISpace(corpus, defaultPPMI)
    expect(a.vecs).toEqual(b.vecs)
    const W = (w: string) => a.words.indexOf(w)
    const nn = nearest(a, a.vecs[W('cat')], 4, 'cosine', [W('cat')]).map((x) => a.groups[x.i])
    expect(nn.every((g) => g === 'animals')).toBe(true)
    // with a ±4 window the capital-country pairing is learned
    expect(analogy(a, W('paris'), W('france'), W('rome'), 3, true, W('italy')).expectedRank).toBe(1)
    expect(analogy(a, W('france'), W('french'), W('germany'), 3, true, W('german')).expectedRank).toBe(1)
    // the near miss quoted in the lesson: princess beats queen in the trained space
    const kq = analogy(a, W('man'), W('king'), W('woman'), 3, true, W('queen'))
    expect(kq.ranked[0].word).toBe('princess')
    expect(kq.expectedRank).toBe(3)
    // with ±1 the capital and its country never share a window: italy drops to 8th (quoted in the example text)
    const narrow = trainPPMISpace(corpus, { ...defaultPPMI, window: 1 })
    const N = (w: string) => narrow.words.indexOf(w)
    expect(analogy(narrow, N('paris'), N('france'), N('rome'), 3, true, N('italy')).expectedRank).toBe(8)
  }, 30000)
})

describe('projection and distortion', () => {
  const s = trainPPMISpace(corpus, defaultPPMI)
  it('PCA projection reports explained variance and a Shepard correlation', () => {
    const p = project(s, defaultProj)
    expect(p.coords.length).toBe(s.words.length)
    expect(p.explained!.length).toBe(2)
    expect(p.explained![0]).toBeLessThan(0.5) // 2D keeps only a small share of 24-dimensional variance
    // the numbers quoted in the "beautiful picture that lies" example
    const u = project(s, { ...defaultProj, method: 'umap' })
    expect(u.preservation.mean).toBeGreaterThan(p.preservation.mean)
    expect(u.shepard.rho).toBeLessThan(p.shepard.rho)
    expect(p.shepard.high.length).toBe((s.words.length * (s.words.length - 1)) / 2)
    expect(p.shepard.rho).toBeGreaterThan(0)
    expect(p.shepard.rho).toBeLessThan(1)
    const m = misleadingPairs(p, 3)
    expect(m.falseFriends[0].gap).toBeGreaterThan(0)
  })
  it('UMAP is deterministic for a fixed seed and differs for another seed', () => {
    const cfg = { ...defaultProj, method: 'umap' as const, nEpochs: 120 }
    const a = project(s, cfg), b = project(s, cfg), c = project(s, { ...cfg, seed: 99 })
    expect(a.coords).toEqual(b.coords)
    expect(c.coords).not.toEqual(a.coords)
  }, 30000)
})

describe('Observatory-Nano embeddings', () => {
  const model = getModel()
  it('static space reads rows of W_E exactly', () => {
    const s = nanoSpace(model)
    const i = s.words.indexOf('·cat')
    const id = model.vocab.indexOf('▁cat')
    expect(s.vecs[i]).toEqual(Array.from(model.W.tok.data.subarray(id * 48, (id + 1) * 48)))
  })
  it('contextual embeddings: identical before position, position-only after adding W_P, diverging after attention', () => {
    const ce = contextualEmbeddings(model, ['the sky is blue', 'the sea is blue', 'paris is the capital of france'], '▁is')
    expect(ce.occ.length).toBe(3)
    const [tok, x0, b1, b2] = ce.layers
    expect(cosine(tok.vecs[0], tok.vecs[2])).toBeCloseTo(1, 12) // same token, same row of W_E
    expect(cosine(x0.vecs[0], x0.vecs[1])).toBeCloseTo(1, 12) // same token at the same position
    expect(cosine(x0.vecs[0], x0.vecs[2])).toBeLessThan(0.99) // different position
    expect(cosine(b2.vecs[0], b2.vecs[2])).toBeLessThan(cosine(b1.vecs[0], b1.vecs[2]))
    // matches a direct forward pass
    const tr = model.forward(model.encodePrompt('the sky is blue').ids)
    const p = ce.occ[0].position
    expect(b2.vecs[0]).toEqual(Array.from(tr.layers[1].xOut.data.subarray(p * 48, (p + 1) * 48)))
  })
})

describe('challenges', () => {
  const s = toySpace()
  const p = project(s, defaultProj)
  it('are a pure function of space, projection and seed', () => {
    expect(makeEmbChallenges(s, p, 4)).toEqual(makeEmbChallenges(s, p, 4))
    expect(makeEmbChallenges(s, p, 4)).not.toEqual(makeEmbChallenges(s, p, 5))
  })
  it('have self-consistent answers', () => {
    for (let seed = 0; seed < 25; seed++)
      for (const q of makeEmbChallenges(s, p, seed)) {
        expect(Number.isFinite(q.answer)).toBe(true)
        if (q.options) { expect(q.answer).toBeGreaterThanOrEqual(0); expect(q.answer).toBeLessThan(q.options.length) }
        if (q.id === 'dot') expect(q.answer).toBe(dot(q.vectors![0].v, q.vectors![1].v))
        if (q.id === 'cosine') expect(q.answer).toBeCloseTo(cosine(q.vectors![0].v, q.vectors![1].v), 14)
      }
  })
  it('answer checking accepts rounding and fractions', () => {
    expect(numCorrect(0.70711, parseNum('0.707'))).toBe(true)
    expect(numCorrect(0.70711, parseNum('0.69'))).toBe(false)
    expect(numCorrect(0.5, parseNum('1/2'))).toBe(true)
    expect(numCorrect(19.6, parseNum('20%'), 'projection')).toBe(true)
  })
})
