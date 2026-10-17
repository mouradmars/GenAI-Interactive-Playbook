import { describe, expect, it } from 'vitest'
import {
  accuracyAt, answerOk, backward, profileAgreement, rankAgreement, cost, defaultSim, forward, makeSeqQuestions, occlusion, parseSeq, schedule, sigmoid, simParams, softmax, trained,
  type Arch, type Params, type Regime,
} from './seq'

const archs: Arch[] = ['rnn', 'lstm', 'attn']

describe('forward pass matches PyTorch for all six trained models', () => {
  for (const regime of ['mixed', 'long'] as Regime[])
    for (const arch of archs)
      it(`${regime} · ${arch}`, () => {
        const m = trained(regime, arch)
        for (const r of m.refs) {
          const tr = forward(m.params, parseSeq(r.seq))
          r.logits.forEach((v, j) => expect(tr.logits[j]).toBeCloseTo(v, 4))
        }
      })
})

describe('structural invariants', () => {
  const ids = parseSeq('cabadbcdaab')
  it('LSTM: c_t = f⊙c_{t−1} + i⊙g and h_t = o⊙tanh(c_t); gates in (0, 1)', () => {
    const tr = forward(simParams('lstm', defaultSim), ids)
    if (tr.arch !== 'lstm') throw new Error()
    for (let t = 0; t < ids.length; t++)
      for (let j = 0; j < 8; j++) {
        const cp = t ? tr.c[t - 1][j] : 0
        expect(tr.c[t][j]).toBeCloseTo(tr.f[t][j] * cp + tr.i[t][j] * tr.g[t][j], 14)
        expect(tr.h[t][j]).toBeCloseTo(tr.o[t][j] * Math.tanh(tr.c[t][j]), 14)
        for (const gate of [tr.i, tr.f, tr.o]) { expect(gate[t][j]).toBeGreaterThan(0); expect(gate[t][j]).toBeLessThan(1) }
      }
  })
  it('attention: rows sum to 1, future positions get exactly 0', () => {
    const tr = forward(trained('mixed', 'attn').params, ids)
    if (tr.arch !== 'attn') throw new Error()
    tr.A.forEach((row, t) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      row.forEach((w, s) => { if (s > t) expect(w).toBe(0) })
    })
  })
  it('output probabilities sum to 1; sigmoid and softmax basics', () => {
    for (const a of archs) expect(forward(simParams(a, defaultSim), ids).probs.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12)
    expect(sigmoid(0)).toBe(0.5)
    expect(softmax([1000, 1000])).toEqual([0.5, 0.5])
  })
})

describe('backpropagation matches finite differences', () => {
  // perturb the (continuous) input vectors and compare dL/dx with central differences
  function check(p: Params, ids: number[], target: number) {
    const tr = forward(p, ids)
    const g = backward(p, tr, target)
    const gl = backward(p, tr, target, 'logit')
    const loss = (xs: number[][]) => {
      // re-run the forward pass with arbitrary input vectors by temporarily replacing one-hot construction
      const t2 = forwardWithInputs(p, xs)
      return -Math.log(t2.probs[target])
    }
    const xs = tr.x.map((v) => v.slice())
    const eps = 1e-6
    for (let t = 0; t < ids.length; t++)
      for (let j = 0; j < 4; j++) {
        const plus = xs.map((v) => v.slice()), minus = xs.map((v) => v.slice())
        plus[t][j] += eps; minus[t][j] -= eps
        const num = (loss(plus) - loss(minus)) / (2 * eps)
        expect(g.dx[t][j]).toBeCloseTo(num, 6)
        // logit objective: compare against central differences of z_target, recovered from the probabilities
        const z = (xs2: number[][]) => forwardWithInputs(p, xs2).logits[target]
        expect(gl.dx[t][j]).toBeCloseTo((z(plus) - z(minus)) / (2 * eps), 6)
      }
    expect(g.loss).toBeCloseTo(loss(xs), 12)
  }
  for (const arch of archs) {
    it(`${arch} · random weights`, () => check(simParams(arch, { ...defaultSim, scale: 1.3 }), parseSeq('dbacabd'), 2))
    it(`${arch} · trained weights`, () => check(trained('mixed', arch).params, parseSeq('bcadda'), 1))
  }
})

/** Forward pass with arbitrary (non one-hot) input vectors, for finite differences. Mirrors `forward`. */
import { sinusoid } from './seq'
function forwardWithInputs(p: Params, x: number[][]) {
  const mv = (W: number[][], v: number[]) => W.map((r) => r.reduce((s, w, j) => s + w * v[j], 0))
  const add = (...vs: number[][]) => vs[0].map((_, i) => vs.reduce((s, v) => s + v[i], 0))
  if (p.arch === 'rnn') {
    let h = new Array(p.Wh.length).fill(0)
    for (const xt of x) h = add(mv(p.Wx, xt), mv(p.Wh, h), p.b).map(Math.tanh)
    const logits = add(mv(p.Wy, h), p.by)
    return { probs: softmax(logits), logits }
  }
  if (p.arch === 'lstm') {
    const H = p.Wh[0].length
    let h = new Array(H).fill(0), c = new Array(H).fill(0)
    for (const xt of x) {
      const z = add(mv(p.Wx, xt), mv(p.Wh, h), p.b)
      const i = z.slice(0, H).map(sigmoid), f = z.slice(H, 2 * H).map(sigmoid), g = z.slice(2 * H, 3 * H).map(Math.tanh), o = z.slice(3 * H).map(sigmoid)
      c = c.map((cv, j) => f[j] * cv + i[j] * g[j]); h = c.map((cv, j) => o[j] * Math.tanh(cv))
    }
    const logits = add(mv(p.Wy, h), p.by)
    return { probs: softmax(logits), logits }
  }
  const H = p.We.length, T = x.length, pos = sinusoid(T, H)
  const e = x.map((xt, t) => add(mv(p.We, xt), pos[t]))
  const q = mv(p.Wq, e[T - 1]), k = e.map((et) => mv(p.Wk, et)), v = e.map((et) => mv(p.Wv, et))
  const a = softmax(k.map((ks) => ks.reduce((s, kj, j) => s + kj * q[j], 0) / Math.sqrt(H)))
  const o = Array.from({ length: H }, (_, j) => a.reduce((s, w, t) => s + w * v[t][j], 0))
  const logits = add(mv(p.Wy, add(o, e[T - 1])), p.by)
  return { probs: softmax(logits), logits }
}

describe('gradient propagation', () => {
  const ids = parseSeq('abcdabcdabcdabcdabcd') // T = 20
  const firstGrad = (p: Params) => { const g = backward(p, forward(p, ids), ids[0], 'logit'); return Math.hypot(...g.dx[0]) / Math.hypot(...g.dx[ids.length - 1]) }
  it('a small recurrent scale makes RNN gradients vanish; a large one does not', () => {
    const small = firstGrad(simParams('rnn', { ...defaultSim, scale: 0.5 }))
    const large = firstGrad(simParams('rnn', { ...defaultSim, scale: 2.5 }))
    expect(small).toBeLessThan(1e-3)
    expect(large).toBeGreaterThan(small * 100)
  })
  it('a large forget-gate bias keeps the LSTM gradient alive over 20 steps', () => {
    const lo = firstGrad(simParams('lstm', { ...defaultSim, scale: 0.5, forgetBias: -2 }))
    const hi = firstGrad(simParams('lstm', { ...defaultSim, scale: 0.5, forgetBias: 4 }))
    expect(hi).toBeGreaterThan(lo * 100)
  })
  it('attention gives position 1 a direct path: its gradient does not decay with distance', () => {
    const r = firstGrad(trained('mixed', 'attn').params)
    expect(r).toBeGreaterThan(0.05)
  })
})

describe('trained-model facts quoted in the lessons', () => {
  const acc = (r: Regime, a: Arch, T: number) => trained(r, a).accuracy.find((x) => x.T === T)!.acc
  it('mixed regime: all three architectures solve every length', () => {
    for (const a of archs) for (const T of [2, 10, 20, 30]) expect(acc('mixed', a, T)).toBeGreaterThan(0.97)
  })
  it('long-only regime: the RNN stays at chance, the LSTM fails short lengths, attention solves all', () => {
    for (const T of [2, 10, 20, 30]) expect(acc('long', 'rnn', T)).toBeLessThan(0.35)
    expect(acc('long', 'lstm', 2)).toBeLessThan(0.6)
    expect(acc('long', 'lstm', 20)).toBeGreaterThan(0.97)
    for (const T of [2, 10, 20, 30]) expect(acc('long', 'attn', T)).toBeGreaterThan(0.97)
    expect(trained('long', 'rnn').history.at(-1)!.loss).toBeCloseTo(Math.log(4), 2)
  })
  it('the browser re-evaluation agrees with the Python evaluation (independent random sequences)', () => {
    expect(accuracyAt(trained('mixed', 'rnn').params, 15, 150)).toBeGreaterThan(0.97)
    expect(accuracyAt(trained('long', 'rnn').params, 15, 150)).toBeLessThan(0.4)
    expect(accuracyAt(trained('long', 'lstm').params, 2, 150)).toBeLessThan(0.65)
  })
})

describe('attribution and cost', () => {
  it('occlusion: in the trained models only the first symbol matters', () => {
    const ids = parseSeq('cbdaabdc')
    for (const a of archs) {
      const occ = occlusion(trained('mixed', a).params, ids, ids[0])
      const top = occ.indexOf(Math.max(...occ))
      expect(top).toBe(0)
    }
  })
  it('cost formulas', () => {
    expect(cost('rnn', 10, 8)).toMatchObject({ macs: 10 * (8 * 4 + 64) + 32, sequentialSteps: 11, pathLength: 9, parallelWidth: 1 })
    expect(cost('attn', 10, 8)).toMatchObject({ pathLength: 1, sequentialSteps: 5, parallelWidth: 10 })
    expect(cost('attn', 10, 8).macs).toBe(10 * 8 * 4 + 3 * 10 * 64 + 2 * 55 * 8 + 32)
    expect(schedule('rnn', 6).filter((o) => o.stage === 3).length).toBe(1)
    expect(schedule('attn', 6).filter((o) => o.stage === 1).length).toBe(6)
  })
})

describe('profile agreement (attention vs occlusion)', () => {
  it('trained attention agrees with occlusion; a random-weight model can disagree', () => {
    const ids = parseSeq('cbdaabdc')
    const tp = trained('mixed', 'attn').params, tt = forward(tp, ids)
    if (tt.arch !== 'attn') throw new Error()
    const good = profileAgreement(tt.A[7], occlusion(tp, ids, ids[0]))
    expect(good.sameTop).toBe(true)
    expect(good.pearson).toBeGreaterThan(0.95)
    const sp = simParams('attn', { ...defaultSim, seed: 1, scale: 2 }), st = forward(sp, ids)
    if (st.arch !== 'attn') throw new Error()
    const bad = profileAgreement(st.A[7], occlusion(sp, ids, ids[0]))
    expect(bad.sameTop).toBe(false)
    expect(bad.pearson).toBeLessThan(0)
  })
})

describe('rank agreement', () => {
  it('is 1 for the same order and −1 for the reverse', () => {
    expect(rankAgreement([1, 2, 3], [10, 20, 30])).toBeCloseTo(1, 12)
    expect(rankAgreement([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1, 12)
  })
})

describe('challenges', () => {
  it('are seeded and self-consistent', () => {
    expect(makeSeqQuestions(5)).toEqual(makeSeqQuestions(5))
    expect(makeSeqQuestions(5)).not.toEqual(makeSeqQuestions(6))
    for (let s = 0; s < 30; s++)
      for (const q of makeSeqQuestions(s)) {
        expect(Number.isFinite(q.answer)).toBe(true)
        if (q.options) expect(q.answer).toBeLessThan(q.options.length)
        else expect(answerOk(q, q.answer.toFixed(3))).toBe(true)
      }
    const q = makeSeqQuestions(1).find((x) => x.id === 'gradient')!
    expect(answerOk(q, String(q.answer * 1.2))).toBe(false)
  })
})
