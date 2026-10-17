import { describe, expect, it } from 'vitest'
import ref from '../data/block-reference.json'
import { rng } from './decoding'
import {
  ACT_FN, blockCost, causalMask, defaultInterventions, erf, kvCacheRun, layerNorm, matmul, runBlock, runStack, shapeTable, softmaxRow, toBf16, toFp16,
  toyEmbeddings, toyParams, transpose, validateBlock, vdot,
  type BlockConfig, type BlockParams, type Interventions, type Matrix,
} from './block'

const maxErr = (a: Matrix, b: number[][]) => Math.max(...a.flatMap((r, i) => r.map((v, j) => Math.abs(v - b[i][j]))))
const base: BlockConfig = { T: 4, d: 4, H: 2, dff: 8, arch: 'pre', causal: true, act: 'gelu', eps: 1e-5 }

describe('building blocks', () => {
  it('matmul and transpose', () => {
    expect(matmul([[1, 2], [3, 4]], [[5], [6]])).toEqual([[17], [39]])
    expect(transpose([[1, 2, 3]])).toEqual([[1], [2], [3]])
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow(/shape mismatch/)
  })
  it('the dot product quoted in the brief: [0.2, 0.8] · [0.5, 0.4] = 0.42', () => { expect(vdot([0.2, 0.8], [0.5, 0.4])).toBeCloseTo(0.42, 15) })
  it('LayerNorm: mean 0, variance ≈ 1 (biased, over features), then γ and β', () => {
    const t = layerNorm([[1, 2, 3, 6]], { g: [1, 1, 1, 1], b: [0, 0, 0, 0] }, 1e-5)
    expect(t.mean[0]).toBe(3); expect(t.variance[0]).toBe(3.5)
    expect(t.xhat[0].reduce((s, v) => s + v, 0)).toBeCloseTo(0, 14)
    expect(t.xhat[0].reduce((s, v) => s + v * v, 0) / 4).toBeCloseTo(3.5 / (3.5 + 1e-5), 12)
    expect(t.xhat[0][0]).toBeCloseTo(-2 / Math.sqrt(3.5 + 1e-5), 14)
    const g = layerNorm([[1, 2, 3, 6]], { g: [2, 2, 2, 2], b: [1, 1, 1, 1] }, 1e-5)
    g.out[0].forEach((v, j) => expect(v).toBeCloseTo(2 * t.xhat[0][j] + 1, 14))
  })
  it('stable softmax: shift invariance, masked entries exactly 0, huge logits finite', () => {
    const a = softmaxRow([1, 2, 3]).p, b = softmaxRow([1001, 1002, 1003]).p
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 14))
    expect(softmaxRow([0.5, -Infinity, 1]).p[1]).toBe(0)
    expect(softmaxRow([1e6, 0]).p).toEqual([1, 0])
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 15)
  })
  it('erf / GELU are exact to ~1e-12 (reference values)', () => {
    expect(erf(0.5)).toBeCloseTo(0.5204998778130465, 13)
    expect(erf(1)).toBeCloseTo(0.8427007929497149, 13)
    expect(erf(3)).toBeCloseTo(0.9999779095030014, 13)
    expect(erf(-2)).toBeCloseTo(-0.9953222650189527, 13)
    expect(ACT_FN.gelu(1)).toBeCloseTo(0.8413447460685429, 12)
    expect(ACT_FN.silu(1)).toBeCloseTo(0.7310585786300049, 14)
  })
  it('causal mask: −∞ strictly above the diagonal; none when bidirectional', () => {
    const m = causalMask(3, true)
    expect(m[0]).toEqual([0, -Infinity, -Infinity]); expect(m[2]).toEqual([0, 0, 0])
    expect(causalMask(3, false).flat().every((v) => v === 0)).toBe(true)
  })
})

describe('PyTorch parity (training/block_reference.py, torch ' + (ref as { torch: string }).torch + ')', () => {
  for (const c of (ref as unknown as { cases: { name: string; cfg: BlockConfig; iv: Omit<Interventions, 'precision'>; X: Matrix; params: BlockParams; ref: Record<string, unknown> & { heads: { S: number[][]; A: number[][]; O: number[][] }[]; torchLayerMaxErr?: number } }[] }).cases) {
    it(c.name, () => {
      const t = runBlock(c.cfg, c.params, c.X, { ...c.iv, precision: 'fp64' })
      const R = c.ref as unknown as Record<string, number[][]>
      for (const k of ['Q', 'K', 'V', 'concat', 'attn', 'H', 'Z1', 'act', 'mlp', 'Y'] as const) expect(maxErr(t[k] as Matrix, R[k])).toBeLessThan(1e-12)
      c.ref.heads.forEach((h, i) => {
        expect(maxErr(t.heads[i].S, h.S)).toBeLessThan(1e-12)
        expect(maxErr(t.heads[i].A, h.A)).toBeLessThan(1e-12)
        expect(maxErr(t.heads[i].O, h.O)).toBeLessThan(1e-12)
      })
      if (c.ref.torchLayerMaxErr !== undefined) expect(c.ref.torchLayerMaxErr).toBeLessThan(1e-12)
    })
  }
})

describe('the block', () => {
  const p = toyParams(base, 1), X = toyEmbeddings(['The', 'cat', 'sat', 'on'], 4, 1).X
  const t = runBlock(base, p, X)
  it('shapes are consistent everywhere', () => {
    const sh = (m: Matrix) => [m.length, m[0].length]
    expect(sh(t.Q)).toEqual([4, 4]); expect(sh(t.heads[0].q)).toEqual([4, 2]); expect(sh(t.heads[0].A)).toEqual([4, 4])
    expect(sh(t.concat)).toEqual([4, 4]); expect(sh(t.Z1)).toEqual([4, 8]); expect(sh(t.Y)).toEqual([4, 4])
    expect(shapeTable(t).find((r) => r.name === 'φ(Z₁)')!.shape).toEqual([4, 8])
    expect(validateBlock({ ...base, d: 6, H: 4 })).toMatch(/not divisible/)
  })
  it('attention rows sum to 1; masked (future) weights are exactly 0', () => {
    for (const h of t.heads) h.A.forEach((row, i) => {
      expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 14)
      row.forEach((v, j) => { if (j > i) expect(v).toBe(0) })
    })
    expect(t.heads[0].A[0][0]).toBe(1) // the first token can only attend to itself
  })
  it('scores are q·k, scaled by 1/√d_h', () => {
    const h = t.heads[1]
    expect(h.S[2][1]).toBeCloseTo(vdot(h.q[2], h.k[1]), 14)
    expect(h.scaled[2][1]).toBeCloseTo(h.S[2][1] / Math.SQRT2, 14)
  })
  it('residual identities: H = X + Attn, Y = H + MLP (pre-LN)', () => {
    expect(maxErr(t.H, X.map((r, i) => r.map((v, j) => v + t.attn[i][j])))).toBe(0)
    expect(maxErr(t.Y, t.H.map((r, i) => r.map((v, j) => v + t.mlp[i][j])))).toBe(0)
  })
  it('causal: changing a later token never changes earlier outputs; bidirectional: it does', () => {
    const X2 = X.map((r) => r.slice()); X2[3] = [2, -2, 1, 0]
    const t2 = runBlock(base, p, X2)
    for (let i = 0; i < 3; i++) expect(maxErr([t2.Y[i]], [t.Y[i]])).toBe(0)
    const enc = { ...base, causal: false }
    expect(maxErr([runBlock(enc, p, X2).Y[0]], [runBlock(enc, p, X).Y[0]])).toBeGreaterThan(1e-3)
  })
  it('the MLP is position-wise: permuting tokens permutes MLP outputs (same weights per token)', () => {
    const perm = [2, 0, 3, 1]
    const tp = runBlock({ ...base, causal: false }, p, perm.map((i) => X[i]))
    const t0 = runBlock({ ...base, causal: false }, p, X)
    perm.forEach((src, i) => expect(maxErr([tp.Y[i]], [t0.Y[src]])).toBeLessThan(1e-12)) // bidirectional attention + MLP are permutation-equivariant
  })
  it('interventions: a disabled head contributes zero; α = 0 removes attention; disabled units', () => {
    const iv = defaultInterventions(base)
    const off = runBlock(base, p, X, { ...iv, heads: [false, true] })
    off.concat.forEach((r) => expect(r.slice(0, 2)).toEqual([0, 0]))
    const noAttn = runBlock(base, p, X, { ...iv, attnScale: 0 })
    expect(maxErr(noAttn.H, X)).toBe(0)
    const units = runBlock(base, p, X, { ...iv, units: iv.units.map((_, u) => u !== 3) })
    units.act.forEach((r) => expect(r[3]).toBe(0))
  })
  it('post-LN normalizes after each residual addition', () => {
    const tp = runBlock({ ...base, arch: 'post', causal: false }, p, X)
    expect(tp.ln1).toBeNull()
    tp.H.forEach((r) => expect(r.reduce((s, v) => s + v, 0) / 4).toBeCloseTo(0, 12))
    tp.Y.forEach((r) => expect(r.reduce((s, v) => s + v, 0) / 4).toBeCloseTo(0, 12))
  })
  it('deterministic toy parameters and embeddings', () => {
    expect(toyParams(base, 1)).toEqual(toyParams(base, 1))
    expect(toyParams(base, 1).Wq.flat().every((v) => Math.abs(v * 10 - Math.round(v * 10)) < 1e-9)).toBe(true)
    const e = toyEmbeddings(['the', 'cat', 'the'], 4, 1)
    expect(e.E[0]).toEqual(e.E[2]) // same token, same embedding
    expect(e.X[0]).not.toEqual(e.X[2]) // different positions
  })
})

describe('reduced precision', () => {
  it('fp16 and bf16 rounding', () => {
    expect(toFp16(1 + 2 ** -11)).toBe(1); expect(toFp16(1 + 2 ** -10)).toBe(1 + 2 ** -10)
    expect(toFp16(0.1)).toBeCloseTo(0.0999755859375, 15); expect(toFp16(70000)).toBe(Infinity)
    expect(toBf16(1 + 2 ** -8)).toBe(1); expect(toBf16(0.1)).toBeCloseTo(0.10009765625, 15)
  })
  it('the block in fp32 / bf16 / fp16 stays close to fp64, with bf16 the coarsest', () => {
    const p = toyParams(base, 1), X = toyEmbeddings(['The', 'cat', 'sat', 'on'], 4, 1).X
    const ref64 = runBlock(base, p, X).Y
    const err = (pr: Interventions['precision']) => maxErr(runBlock(base, p, X, { ...defaultInterventions(base), precision: pr }).Y, ref64)
    expect(err('fp32')).toBeLessThan(1e-5); expect(err('fp16')).toBeLessThan(2e-2); expect(err('bf16')).toBeLessThan(1e-1)
    expect(err('bf16')).toBeGreaterThan(err('fp16')); expect(err('fp16')).toBeGreaterThan(err('fp32'))
  })
})

describe('stacks, KV cache and cost', () => {
  const cfg = { ...base, T: 5 }
  const ps = [1, 2, 3].map((s) => toyParams(cfg, s)), X = toyEmbeddings(['The', 'cat', 'sat', 'on', 'the'], 4, 1).X
  it('a stack feeds each block the previous output; a disabled block is the identity', () => {
    const s = runStack(cfg, ps, X)
    expect(s.inputs[1]).toEqual(s.outputs[0]); expect(s.outputs[2]).toEqual(runBlock(cfg, ps[2], s.outputs[1]).Y)
    const d = runStack(cfg, ps, X, [true, false, true])
    expect(d.traces[1]).toBeNull(); expect(d.outputs[1]).toEqual(d.outputs[0])
  })
  it('KV-cache decoding reproduces every row of full causal attention, with fewer MACs', () => {
    const p = ps[0], full = runBlock(cfg, p, X), steps = kvCacheRun(cfg, p, X)
    steps.forEach((s, t) => {
      expect(maxErr([s.out], [full.attn[t]])).toBeLessThan(1e-12)
      s.weights.forEach((w, h) => expect(maxErr([w], [full.heads[h].A[t].slice(0, t + 1)])).toBeLessThan(1e-12))
      expect(s.cacheK[0].length).toBe(t + 1)
      if (t > 0) expect(s.newMacs).toBeLessThan(s.fullMacs)
    })
  })
  it('cost: attention scores grow with T², projections with T', () => {
    const a = blockCost(128, 64, 4, 256), b = blockCost(256, 64, 4, 256)
    expect(b.scores / a.scores).toBe(4); expect(b.qkv / a.qkv).toBe(2); expect(b.attnMatrixEntries).toBe(4 * 256 * 256)
  })
  it('scaling: the variance of q·k for random unit-variance vectors grows like d_k', () => {
    let s1 = 0, s2 = 0, n = 4000
    const rr = rng(12345), g = () => { let u = 0; for (let k = 0; k < 12; k++) u += rr(); return u - 6 }
    for (const dk of [4, 64]) { let m = 0, v = 0; for (let i = 0; i < n; i++) { const q = Array.from({ length: dk }, g), k = Array.from({ length: dk }, g); const x = vdot(q, k); m += x; v += x * x } const varr = v / n - (m / n) ** 2; if (dk === 4) s1 = varr; else s2 = varr }
    expect(s2 / s1).toBeGreaterThan(10); expect(s2 / s1).toBeLessThan(22)
  })
})

import { attnSublayer, dependencyMap, mlpSublayer } from './block'
describe('attention vs MLP dependency', () => {
  const p = toyParams(base, 1), X = toyEmbeddings(['The', 'cat', 'sat', 'on'], 4, 1).X
  it('the sublayer helpers reproduce the block', () => {
    const t = runBlock(base, p, X)
    expect(maxErr(attnSublayer(base, p, t.attnIn), t.attn)).toBeLessThan(1e-12)
    expect(maxErr(mlpSublayer(base, p, t.mlpIn), t.mlp)).toBeLessThan(1e-12)
  })
  it('MLP: output i depends only on input i; causal attention: on inputs j ≤ i; bidirectional: on all', () => {
    const dm = dependencyMap(base, p, X)
    dm.mlp.forEach((row, i) => row.forEach((v, j) => (i === j ? expect(v).toBeGreaterThan(1e-3) : expect(v).toBe(0))))
    dm.attn.forEach((row, i) => row.forEach((v, j) => (j > i ? expect(v).toBe(0) : expect(v).toBeGreaterThan(0))))
    const bi = dependencyMap({ ...base, causal: false }, p, X)
    bi.attn.flat().forEach((v) => expect(v).toBeGreaterThan(0))
  })
})
