import ref from '../data/attention-reference.json'
import {
  PRESET_PARAMS, PRESET_X, aggregationTerms, attentionCost, buildMask, compareTraces, concatCols, defaultConfig, dotTerms, entropy,
  headSimilarity, jsDivergence, matmulCell, presetInput, runAttention, scalingExperiment, shapeList, sinusoidal, softmaxJacobianNorm,
  stableSoftmax, tokenize, validateAttention, type AttnConfig, type AttnParams, type Matrix,
} from './attention'

const close = (a: Matrix, b: Matrix, tol = 1e-12) => {
  expect(a.length).toBe(b.length)
  a.forEach((r, i) => r.forEach((v, j) => expect(Math.abs(v - b[i][j])).toBeLessThan(tol)))
}
const cfgFor = (c: (typeof ref.cases)[number]): AttnConfig => ({ H: c.H, causal: c.causal, scale: c.scale, pad: c.pad, blocked: c.blocked as boolean[][] | null, headsOn: c.headsOn })
const P = PRESET_PARAMS
const X = PRESET_X

describe('attention engine · PyTorch parity', () => {
  for (const c of ref.cases) {
    it(`matches PyTorch: ${c.name}`, () => {
      const t = runAttention(c.X, c.params as AttnParams, cfgFor(c), c.Xkv ?? undefined)
      c.heads.forEach((h, k) => {
        close(t.heads[k].Q, h.Q); close(t.heads[k].K, h.K); close(t.heads[k].V, h.V)
        close(t.heads[k].S, h.S); close(t.heads[k].A, h.A); close(t.heads[k].O, h.O)
      })
      close(t.concat, c.concat)
      close(t.out, c.out)
    })
  }
  it('the reference was cross-checked against nn.MultiheadAttention', () => {
    expect(ref.cases.filter((c) => c.checkedAgainstMultiheadAttention).length).toBeGreaterThanOrEqual(6)
  })
})

describe('attention engine · building blocks', () => {
  it('dot products and matrix-multiplication cells expose their terms', () => {
    const d = dotTerms([0.2, 0.8], [0.5, 0.4])
    expect(d.products).toEqual([0.2 * 0.5, 0.8 * 0.4])
    expect(d.sum).toBeCloseTo(0.42, 15)
    const cell = matmulCell(X, P.Wq, 0, 0)
    expect(cell.a).toEqual(X[0])
    expect(cell.b).toEqual(P.Wq.map((r) => r[0]))
    expect(cell.sum).toBeCloseTo(1.0 * 2 + 0 + 0.5 * 1 + 0.2 * 0, 15)
    expect(() => dotTerms([1], [1, 2])).toThrow()
  })
  it('Q/K/V projections use each head\'s own column block', () => {
    const t = runAttention(X, P, defaultConfig(4))
    expect(t.heads[1].Wq).toEqual(P.Wq.map((r) => r.slice(2, 4)))
    // Q of head 1, token 1 = x₁ W_Q^{(1)}
    expect(t.heads[0].Q[0][0]).toBeCloseTo(2.5, 14)
    expect(t.heads[0].Q[0][1]).toBeCloseTo(0.2, 14)
  })
  it('stable softmax: rows sum to 1, masked entries are exactly 0, huge logits do not overflow', () => {
    const r = stableSoftmax([1000, 999, -Infinity])
    expect(r.p[0] + r.p[1]).toBeCloseTo(1, 15)
    expect(r.p[2]).toBe(0)
    expect(r.max).toBe(1000)
    expect(r.shifted[0]).toBe(0)
    expect(r.p[0]).toBeCloseTo(1 / (1 + Math.exp(-1)), 15)
    expect(() => stableSoftmax([NaN, 1])).toThrow()
  })
  it('a fully masked row gives zero weights, a zero output and a flag (never NaN)', () => {
    const t = runAttention(X, P, { ...defaultConfig(4), pad: [true, false, false, false] })
    expect(t.fullyMaskedRows).toEqual([0])
    for (const h of t.heads) {
      expect(h.rows[0].fullyMasked).toBe(true)
      expect(h.A[0]).toEqual([0, 0, 0, 0])
      expect(h.O[0].every((v) => v === 0)).toBe(true)
    }
    expect(t.out.flat().every(Number.isFinite)).toBe(true)
  })
  it('causal, padding and custom masks combine, with reasons', () => {
    const { M, reasons } = buildMask(3, 3, { causal: true, pad: [false, true, false], blocked: [[false, false, false], [false, false, false], [true, false, false]] })
    expect(M[0]).toEqual([0, -Infinity, -Infinity])
    expect(M[2]).toEqual([-Infinity, -Infinity, 0])
    expect(reasons[0][1]).toEqual(['causal', 'pad'])
    expect(reasons[2][0]).toEqual(['custom'])
  })
  it('scaling divides every score by √d_k and switching it off keeps raw scores', () => {
    const on = runAttention(X, P, defaultConfig(4)), off = runAttention(X, P, { ...defaultConfig(4), scale: false })
    on.heads[0].S.forEach((r, i) => r.forEach((v, j) => expect(on.heads[0].scaled[i][j]).toBeCloseTo(v / Math.SQRT2, 14)))
    expect(off.heads[0].scaled).toEqual(off.heads[0].S)
  })
  it('weighted value aggregation equals Σ_j a_ij v_j', () => {
    const t = runAttention(X, P, defaultConfig(4))
    const g = aggregationTerms(t.heads[1], 3)
    g.o.forEach((v, k) => expect(v).toBeCloseTo(t.heads[1].O[3][k], 14))
    expect(g.contrib[0][0]).toBeCloseTo(t.heads[1].A[3][0] * t.heads[1].V[0][0], 15)
  })
  it('splitting, concatenation and output projection have the right shapes for every valid h', () => {
    for (const H of [1, 2, 4]) {
      const t = runAttention(presetInput(6), P, defaultConfig(6, H))
      expect(t.dk).toBe(4 / H)
      expect(t.heads).toHaveLength(H)
      t.heads.forEach((h) => { expect(h.Q[0]).toHaveLength(4 / H); expect(h.A).toHaveLength(6); expect(h.A[0]).toHaveLength(6) })
      expect(t.concat[0]).toHaveLength(4)
      expect(t.out).toHaveLength(6)
      expect(t.out[0]).toHaveLength(4)
      const sh = shapeList(t)
      expect(sh.find((s) => s.name === 'H')!.shape).toEqual([6, 4])
      expect(sh.find((s) => s.name === 'Q_h')!.shape).toEqual([6, 4 / H])
    }
    expect(concatCols([[[1], [2]], [[3], [4]]])).toEqual([[1, 3], [2, 4]])
  })
  it('validation rejects bad shapes and head counts', () => {
    expect(validateAttention(X, P, { ...defaultConfig(4), H: 3, headsOn: [true, true, true] })[0]).toMatch(/must divide/)
    expect(validateAttention(X, { ...P, Wo: [[1]] }, defaultConfig(4)).some((e) => /W_O must be 4×4/.test(e))).toBe(true)
    expect(validateAttention([[1, NaN, 0, 0]], P, defaultConfig(1)).some((e) => /non-finite/.test(e))).toBe(true)
    expect(() => runAttention(X, P, { ...defaultConfig(4), H: 3, headsOn: [true, true, true] })).toThrow()
  })
})

describe('attention engine · invariants', () => {
  const base = runAttention(X, P, defaultConfig(4))
  it('1 · every valid softmax row sums to 1', () => {
    for (const cfg of [defaultConfig(4), { ...defaultConfig(4), causal: false }, { ...defaultConfig(4), scale: false }]) {
      runAttention(X, P, cfg).heads.forEach((h) => h.A.forEach((r) => expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)))
    }
  })
  it('2 · masked positions have exactly zero probability', () => {
    base.heads.forEach((h) => h.A.forEach((r, i) => r.forEach((v, j) => { if (j > i) expect(v).toBe(0) })))
    const padded = runAttention(X, P, { ...defaultConfig(4), causal: false, pad: [false, true, false, false] })
    padded.heads.forEach((h) => h.A.forEach((r) => expect(r[1]).toBe(0)))
  })
  it('3 · changing only V never changes the attention weights (but changes the output)', () => {
    const Wv = P.Wv.map((r) => r.map((v) => v * -1.5 + 0.3))
    const t = runAttention(X, { ...P, Wv }, defaultConfig(4))
    const d = compareTraces(base, t)
    d.heads.forEach((h) => { expect(h.changed.Q).toBe(false); expect(h.changed.K).toBe(false); expect(h.changed.S).toBe(false); expect(h.changed.A).toBe(false); expect(h.changed.V).toBe(true) })
    expect(d.outChanged).toBe(true)
  })
  it('4 · changing Q or K can change the attention weights', () => {
    const Wq = P.Wq.map((r) => r.slice()); Wq[0][0] = -1
    const Wk = P.Wk.map((r) => r.slice()); Wk[1][1] = 3
    expect(compareTraces(base, runAttention(X, { ...P, Wq }, defaultConfig(4))).heads[0].changed.A).toBe(true)
    expect(compareTraces(base, runAttention(X, { ...P, Wk }, defaultConfig(4))).heads[0].changed.A).toBe(true)
  })
  it('editing only head 1\'s projections leaves head 2 untouched', () => {
    const Wq = P.Wq.map((r) => r.slice()); Wq[2][1] = -2
    const d = compareTraces(base, runAttention(X, { ...P, Wq }, defaultConfig(4)))
    expect(d.heads[0].changed.A).toBe(true)
    expect(Object.values(d.heads[1].changed).some(Boolean)).toBe(false)
    expect(d.outChanged).toBe(true)
  })
  it('5 · concatenated width is h·d_v, and 6 · the output has d_model columns', () => {
    expect(base.concat[0]).toHaveLength(2 * base.dk)
    expect(base.out[0]).toHaveLength(base.dModel)
  })
  it('7 · replaying identical inputs and parameters gives identical outputs', () => {
    const again = runAttention(X.map((r) => r.slice()), JSON.parse(JSON.stringify(P)), defaultConfig(4))
    expect(again.out).toEqual(base.out)
    expect(again.heads.map((h) => h.A)).toEqual(base.heads.map((h) => h.A))
    expect(scalingExperiment(16, 4).raw).toEqual(scalingExperiment(16, 4).raw)
  })
  it('head ablation replaces a head\'s block by zeros and only that block', () => {
    const t = runAttention(X, P, { ...defaultConfig(4), headsOn: [true, false] })
    t.concat.forEach((r, i) => { expect(r.slice(2)).toEqual([0, 0]); expect(r.slice(0, 2)).toEqual(base.concat[i].slice(0, 2)) })
    expect(compareTraces(base, t).outChanged).toBe(true)
  })
  it('cross-attention: queries from one sequence, keys and values from another; no causal mask', () => {
    const Y = presetInput(3, 4, 7)
    const t = runAttention(X, P, { ...defaultConfig(3), causal: true }, Y)
    expect(t.self).toBe(false)
    expect(t.heads[0].A).toHaveLength(4)
    expect(t.heads[0].A[0]).toHaveLength(3)
    expect(t.heads[0].A[0].every((v) => v > 0)).toBe(true)
  })
})

describe('attention engine · analysis tools', () => {
  it('entropy, JS divergence and the softmax Jacobian behave as expected', () => {
    expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(Math.log(4), 14)
    expect(entropy([1, 0, 0])).toBe(0)
    expect(jsDivergence([1, 0], [0, 1])).toBeCloseTo(Math.log(2), 14)
    expect(jsDivergence([0.3, 0.7], [0.3, 0.7])).toBe(0)
    expect(softmaxJacobianNorm([1, 0, 0])).toBe(0)
    expect(softmaxJacobianNorm([0.5, 0.5])).toBeCloseTo(Math.sqrt(4 * 0.25 ** 2), 14)
    const s = headSimilarity([[1, 0], [0.5, 0.5]], [[1, 0], [0.5, 0.5]])
    expect(s.js).toBe(0); expect(s.tv).toBe(0); expect(s.cosine).toBeCloseTo(1, 14)
  })
  it('the scaling experiment: Var(q·k) ≈ d_k, ≈ 1 after scaling, and unscaled softmax sharpens with d_k', () => {
    const small = scalingExperiment(4), big = scalingExperiment(256)
    expect(big.rawVar / 256).toBeGreaterThan(0.85); expect(big.rawVar / 256).toBeLessThan(1.15)
    expect(big.scaledVar).toBeCloseTo(big.rawVar / 256, 12)
    expect(big.entropyRaw).toBeLessThan(0.2)
    expect(big.maxPRaw).toBeGreaterThan(0.9)
    expect(big.entropyScaled).toBeGreaterThan(1)
    expect(big.jacRaw).toBeLessThan(big.jacScaled)
    expect(small.entropyRaw).toBeGreaterThan(big.entropyRaw)
  })
  it('the cost model grows quadratically in T for the score and mixing terms', () => {
    const a = attentionCost(8, 4, 2), b = attentionCost(16, 4, 2)
    expect(b.scores / a.scores).toBe(4)
    expect(b.proj / a.proj).toBe(2)
    expect(b.attnEntries).toBe(2 * 16 * 16)
  })
  it('helpers: preset input, sinusoidal positions and the toy tokenizer', () => {
    expect(presetInput(4)).toEqual(PRESET_X)
    expect(presetInput(6)[5]).toEqual(presetInput(6)[5])
    expect(presetInput(6)[5].every((v) => v >= 0 && v <= 1)).toBe(true)
    const pe = sinusoidal(3, 4)
    expect(pe[0]).toEqual([0, 1, 0, 1])
    expect(pe[1][0]).toBeCloseTo(Math.sin(1), 15)
    expect(pe[1][2]).toBeCloseTo(Math.sin(1 / 100), 15)
    expect(tokenize('The cat chased the mouse.')).toEqual(['The', 'cat', 'chased', 'the', 'mouse', '.'])
  })
})
