import { describe, expect, it } from 'vitest'
import ref from '../data/nano-reference.json'
import raw from '../data/nano-model.json'
import corpusSample from '../data/corpus-sample.txt?raw'
import { getModel } from './model'
import { decodeStep, defaultDecode, generate, rng } from './decoding'
import { layerNorm, linear, mat, pca3, softmax, gelu } from './tensor'
import { pretokenize, trainBPE } from './bpe'

const model = getModel()

describe('tensor maths', () => {
  it('softmax sums to one and is shift invariant', () => {
    const p = softmax([1, 2, 3, -Infinity])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(p[3]).toBe(0)
    const q = softmax([1001, 1002, 1003, -Infinity])
    p.forEach((v, i) => expect(q[i]).toBeCloseTo(v, 12))
  })
  it('temperature sharpens and flattens', () => {
    expect(softmax([1, 2], 0.5)[1]).toBeGreaterThan(softmax([1, 2], 1)[1])
    expect(softmax([1, 2], 5)[1]).toBeLessThan(softmax([1, 2], 1)[1])
  })
  it('linear checks shapes and computes x@W+b', () => {
    const y = linear(mat(1, 2, [1, 2]), mat(2, 2, [1, 0, 0, 1]), new Float32Array([10, 20]))
    expect([...y.data]).toEqual([11, 22])
    expect(() => linear(mat(1, 3), mat(2, 2))).toThrow(/shape mismatch/)
  })
  it('layerNorm gives zero mean and unit variance (with g=1,b=0)', () => {
    const { out } = layerNorm(mat(1, 4, [1, 2, 3, 10]), new Float32Array(4).fill(1), new Float32Array(4))
    const m = out.data.reduce((a, b) => a + b, 0) / 4
    const v = out.data.reduce((a, b) => a + (b - m) ** 2, 0) / 4
    expect(m).toBeCloseTo(0, 5)
    expect(v).toBeCloseTo(1, 3)
  })
  it('gelu matches known values', () => {
    expect(gelu(0)).toBe(0)
    expect(gelu(1)).toBeCloseTo(0.841192, 5)
  })
  it('pca explains a decreasing share of variance', () => {
    const { explained } = pca3(model.W.tok)
    expect(explained[0]).toBeGreaterThanOrEqual(explained[1])
    expect(explained[1]).toBeGreaterThanOrEqual(explained[2])
  })
})

describe('BPE matches the Python trainer', () => {
  it('reproduces the exported merge list from the same corpus', async () => {
    // retrain on the 6 kB sample: merges must be a deterministic function of the text
    const a = trainBPE(corpusSample, 40), b = trainBPE(corpusSample, 40)
    expect(a.merges).toEqual(b.merges)
    expect(a.merges.map((m) => m.pair)).toEqual(ref.sampleMerges)
    expect(pretokenize('The cat.')).toEqual(['the', '▁cat', '.'])
  })
  it('encodes reference prompts to the same ids as Python', () => {
    for (const r of ref.refs) expect(model.encodePrompt(r.prompt).ids).toEqual(r.ids)
  })
  it('vocab = <bos> + chars + merges', () => {
    expect(model.vocab.length).toBe(1 + raw.chars.length + raw.merges.length)
  })
})

describe('forward pass matches PyTorch', () => {
  for (const r of ref.refs) {
    it(`"${r.prompt}"`, () => {
      const tr = model.forward(r.ids)
      const n = r.ids.length
      const V = model.meta.vocabSize
      const last = tr.logits.data.subarray((n - 1) * V, n * V)
      let maxErr = 0
      r.lastLogits.forEach((v, i) => (maxErr = Math.max(maxErr, Math.abs(v - last[i]))))
      expect(maxErr).toBeLessThan(2e-3)
      const att = tr.layers[0].heads[0].probs
      r.att0.forEach((rowv, i) => rowv.forEach((v, j) => expect(att.data[i * n + j]).toBeCloseTo(v, 4)))
      // invariants
      for (const L of tr.layers)
        for (const h of L.heads)
          for (let i = 0; i < n; i++) {
            let s = 0
            for (let j = 0; j < n; j++) {
              s += h.probs.data[i * n + j]
              if (j > i) expect(h.probs.data[i * n + j]).toBe(0) // causal mask
            }
            expect(s).toBeCloseTo(1, 5)
          }
      expect(tr.logits.rows).toBe(n)
      expect(tr.logits.cols).toBe(V)
      const top = softmax(last)
      const best = top.indexOf(Math.max(...top))
      expect(model.vocab[best]).toBe(r.top5[0][0])
    })
  }
})

describe('decoding', () => {
  const logits = [2, 1, 0.5, -1]
  it('greedy picks argmax', () => {
    expect(decodeStep(logits, { ...defaultDecode, strategy: 'greedy' }, 0.99).chosen).toBe(0)
  })
  it('top-k keeps exactly k tokens and renormalises', () => {
    const s = decodeStep(logits, { ...defaultDecode, topK: 2 }, 0.5)
    expect(s.kept.filter(Boolean).length).toBe(2)
    expect(s.final.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('top-p keeps the smallest prefix reaching p', () => {
    const s = decodeStep(logits, { ...defaultDecode, topP: 0.6 }, 0.5)
    expect(s.kept).toEqual([true, false, false, false])
  })
  it('sampling inverts the CDF', () => {
    expect(decodeStep(logits, defaultDecode, 0.0).chosen).toBe(0)
    expect(decodeStep(logits, defaultDecode, 0.9999).chosen).toBe(3)
  })
  it('seeded generation is reproducible', () => {
    const ids = model.encodePrompt('sara went to the').ids
    const a = generate(model, ids, 4, { ...defaultDecode, seed: 7 })
    const b = generate(model, ids, 4, { ...defaultDecode, seed: 7 })
    expect(a.ids).toEqual(b.ids)
    expect(rng(1)()).toBe(rng(1)())
  })
})
