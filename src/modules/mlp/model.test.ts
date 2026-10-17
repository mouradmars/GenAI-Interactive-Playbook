import { describe, expect, it } from 'vitest'
import { WORKED, applyUpdate, backward, chainPaths, datasetLoss, forward, getParam, initParams, lossValue, trainStep, validateConfig, zeroOptState } from '../../engine/nn'
import { PRESETS, factorTex, microSteps, pathChain, repairConfig, revealAt } from './model'
import { createMlpStore, isModified } from './store'

describe('presets', () => {
  it('every preset is a valid configuration with matching parameter shapes', () => {
    for (const p of PRESETS) {
      expect(validateConfig(p.cfg).ok).toBe(true)
      if (p.params) expect(p.params.W.map((w) => [w.length, w[0].length])).toEqual(p.cfg.sizes.slice(1).map((n, l) => [n, p.cfg.sizes[l]]))
    }
  })
})

describe('repairConfig', () => {
  it('BCE requested on a linear output switches the output to sigmoid', () => {
    const r = repairConfig({ sizes: [2, 3, 1], acts: ['relu', 'linear'], loss: 'mse' }, { loss: 'bce' })
    expect(r.cfg.acts[1]).toBe('sigmoid'); expect(r.cfg.loss).toBe('bce'); expect(r.notes.length).toBe(1)
  })
  it('changing a BCE output to linear switches the loss to MSE', () => {
    const r = repairConfig({ sizes: [2, 3, 1], acts: ['relu', 'linear'], loss: 'bce' })
    expect(r.cfg.loss).toBe('mse')
  })
  it('CCE forces softmax; one output cannot use CCE or softmax', () => {
    expect(repairConfig({ sizes: [2, 3, 3], acts: ['relu', 'sigmoid'], loss: 'mse' }, { loss: 'cce' }).cfg.acts[1]).toBe('softmax')
    const r = repairConfig({ sizes: [2, 3, 1], acts: ['relu', 'softmax'], loss: 'cce' })
    expect(r.cfg.loss).toBe('bce'); expect(r.cfg.acts[1]).toBe('sigmoid')
  })
  it('softmax on a hidden layer is replaced and every result is valid', () => {
    const r = repairConfig({ sizes: [2, 3, 2], acts: ['softmax', 'softmax'], loss: 'cce' })
    expect(r.cfg.acts[0]).toBe('tanh'); expect(validateConfig(r.cfg).ok).toBe(true)
  })
  it('clamps neuron counts', () => {
    expect(repairConfig({ sizes: [2, 99, 1], acts: ['relu', 'sigmoid'], loss: 'mse' }).cfg.sizes[1]).toBe(16)
    expect(repairConfig({ sizes: [2, 0, 1], acts: ['relu', 'sigmoid'], loss: 'mse' }).cfg.sizes[1]).toBe(1)
  })
})

describe('micro-steps', () => {
  it('2-2-1: x, z¹, a¹, z², a², L, δ², ∇W², δ¹, ∇W¹, update', () => {
    expect(microSteps(2).map((s) => s.kind + ('layer' in s ? s.layer : ''))).toEqual(['input', 'z1', 'act1', 'z2', 'act2', 'loss', 'dout', 'grad2', 'delta1', 'grad1', 'update'])
    expect(microSteps(2, 'forward').length).toBe(6)
    expect(microSteps(2, 'backprop').length).toBe(10)
  })
  it('reveal tracks what has been computed', () => {
    const s = microSteps(3)
    const r0 = revealAt(s, 0, 3)
    expect(r0).toEqual({ zUpTo: 0, aUpTo: 0, loss: false, deltaFrom: 4, gradFrom: 4, update: false })
    const r = revealAt(s, s.findIndex((x) => x.kind === 'delta' && x.layer === 2), 3)
    expect(r.aUpTo).toBe(3); expect(r.loss).toBe(true); expect(r.deltaFrom).toBe(2); expect(r.gradFrom).toBe(3)
  })
})

describe('pathChain', () => {
  it('worked example: one path, node grads equal edge tails (single-path chain)', () => {
    const { cfg, params, x, y } = WORKED, t = forward(cfg, params, x), b = backward(cfg, params, t, y)
    const start = { kind: 'w' as const, layer: 1, j: 0, i: 0 }
    const cp = chainPaths(cfg, params, t, y, start)
    const { nodes, edges } = pathChain(cfg, params, t, b, lossValue(cfg, t, y), start, cp.paths[0])
    expect(nodes.map((n) => n.kind)).toEqual(['w', 'z', 'a', 'z', 'a', 'L'])
    expect(edges.length).toBe(5)
    // with a single path, the running product from the loss end equals each node's total gradient
    edges.forEach((e, k) => expect(e.tail).toBeCloseTo(nodes[k].grad, 12))
    expect(nodes[1].value).toBeCloseTo(0.36, 15)
  })
  it('softmax paths walk through output activations', () => {
    const cfg = { sizes: [2, 2, 3], acts: ['tanh', 'softmax'] as const, loss: 'cce' as const }
    const c = { ...cfg, acts: [...cfg.acts] }, p = initParams(c.sizes, 3), t = forward(c, p, [0.3, -0.2]), b = backward(c, p, t, [0, 1, 0])
    const start = { kind: 'b' as const, layer: 1, j: 1 }
    const cp = chainPaths(c, p, t, [0, 1, 0], start)
    expect(cp.paths.length).toBe(9)
    const { nodes } = pathChain(c, p, t, b, lossValue(c, t, [0, 1, 0]), start, cp.paths[4])
    expect(nodes.map((n) => n.kind)).toEqual(['b', 'z', 'a', 'z', 'a', 'L'])
  })
  it('factor labels convert to TeX', () => {
    expect(factorTex('w₁₂⁽²⁾')).toBe('w_{12}^{(2)}')
    expect(factorTex('σ′(z₁⁽¹⁾)')).toBe("\\sigma'(z_{1}^{(1)})")
    expect(factorTex('∂L/∂a₁⁽²⁾')).toBe('\\partial L/\\partial a_{1}^{(2)}')
  })
})

describe('store', () => {
  it('starts on the worked example', () => {
    const s = createMlpStore('worked').getState()
    expect(s.cfg.sizes).toEqual([2, 2, 1]); expect(s.data).toEqual([{ x: [1, 0.8], y: [1], label: 1 }])
    expect(s.history.length).toBe(1); expect(s.history[0].loss).toBeCloseTo(lossValue(s.cfg, forward(s.cfg, s.params, [1, 0.8]), [1]), 15)
  })
  it('next past the last micro-step commits exactly one optimizer update', () => {
    const st = createMlpStore('worked'), s0 = st.getState()
    const g = backward(s0.cfg, s0.params, forward(s0.cfg, s0.params, [1, 0.8]), [1])
    const expected = applyUpdate(s0.params, g, zeroOptState(s0.params), s0.opt).params
    for (let k = 0; k < microSteps(2).length; k++) st.getState().next()
    const s = st.getState()
    expect(s.history.length).toBe(2); expect(s.micro).toBe(0); expect(s.params).toEqual(expected)
    expect(s.history[1].loss).toBeLessThan(s.history[0].loss)
  })
  it('rewind restores a snapshot and a new step branches (future is discarded)', () => {
    const st = createMlpStore('xor')
    st.getState().train(10)
    const at3 = st.getState().history[3]
    st.getState().rewind(3)
    expect(st.getState().params).toEqual(at3.params)
    st.getState().train(1)
    expect(st.getState().history.length).toBe(5)
    // replay determinism: the branch equals the original step 4
    const st2 = createMlpStore('xor'); st2.getState().train(4)
    expect(st.getState().params).toEqual(st2.getState().params)
  })
  it('training matches engine trainStep', () => {
    const st = createMlpStore('xor'), s0 = st.getState()
    let ts = { params: s0.params, opt: zeroOptState(s0.params), epoch: 0, iter: 0, cursor: 0 }
    for (let k = 0; k < 7; k++) ts = trainStep(s0.cfg, ts, s0.data, s0.opt, s0.batch, s0.trainSeed).state
    st.getState().train(7)
    expect(st.getState().params).toEqual(ts.params)
    expect(st.getState().history.at(-1)!.loss).toBeCloseTo(datasetLoss(s0.cfg, ts.params, s0.data), 15)
  })
  it('manual edits are tracked, can be reverted or recorded', () => {
    const st = createMlpStore('worked'), ref = { kind: 'w' as const, layer: 2, j: 0, i: 1 }
    st.getState().setParamValue(ref, 1.25)
    expect(isModified(st.getState())).toBe(true)
    st.getState().revertEdits(); expect(getParam(st.getState().params, ref)).toBe(-0.4)
    st.getState().setParamValue(ref, 1.25); st.getState().recordEdits()
    expect(isModified(st.getState())).toBe(false); expect(st.getState().history.at(-1)!.kind).toBe('edit')
  })
  it('architecture edits rebuild parameters, data and history, and repair the loss', () => {
    const st = createMlpStore('worked')
    st.getState().setNeurons(1, 4)
    expect(st.getState().params.W[0].length).toBe(4); expect(st.getState().history.length).toBe(1)
    st.getState().setActivation(2, 'linear'); expect(st.getState().cfg.loss).toBe('mse')
    st.getState().setLoss('bce'); expect(st.getState().cfg.acts[1]).toBe('sigmoid')
    st.getState().addLayer(); expect(st.getState().cfg.sizes.length).toBe(4)
    st.getState().removeLayer(1); expect(st.getState().cfg.sizes.length).toBe(3)
    st.getState().setNeurons(0, 3); expect(st.getState().data[0].x.length).toBe(3)
    st.getState().setDataset('xor'); expect(st.getState().dataset).toBe('random')
  })
  it('the digit-sized preset builds with 64 inputs and 10 classes', () => {
    const s = createMlpStore('digits').getState()
    expect(s.data[0].x.length).toBe(64); expect(s.data[0].y.length).toBe(10)
    expect(s.data[0].y.reduce((a, b) => a + b, 0)).toBe(1)
  })
})

import { narrate } from './narrate'
import { accuracy, defaultOpt, lossValue as lv } from '../../engine/nn'
describe('narration uses the real numbers', () => {
  const { cfg, params, x, y } = WORKED, t = forward(cfg, params, x), b = backward(cfg, params, t, y)
  const input = { cfg, params, trace: t, bwd: b, sample: { x, y }, loss: lv(cfg, t, y), opt: defaultOpt, records: applyUpdate(params, b, zeroOptState(params), defaultOpt).records, lossAfter: 0.1676, depth: 'intermediate' as const }
  it('z¹ for neuron 1 shows 0.5·1.0 + (−0.3)·0.8 + 0.1 = 0.3600', () => {
    const n = narrate({ kind: 'z', layer: 1 }, 0, input)
    expect(n.numbers[0]).toBe('z_{1}^{(1)} = 0.5000\\cdot1.0000 + (-0.3000)\\cdot0.8000 + 0.1000 = 0.3600')
  })
  it('output error: ∂L/∂ŷ = −0.9192, σ′ = 0.2484, δ² = −0.2283', () => {
    const s = narrate({ kind: 'dout' }, 0, input).numbers.join(' ')
    expect(s).toContain('-0.9192'); expect(s).toContain('0.2484'); expect(s).toContain('-0.2283')
  })
  it('update shows the SGD arithmetic for w₁₁⁽¹⁾', () => {
    expect(narrate({ kind: 'update' }, 0, input).numbers[0]).toBe('0.5000 - 0.5\\cdot(-0.0332) = 0.5166')
  })
})
describe('presets learn', () => {
  it('the XOR preset reaches 100 % with its own optimizer settings', () => {
    const st = createMlpStore('xor'); st.getState().train(1500)
    const s = st.getState(); expect(accuracy(s.cfg, s.params, s.data)).toBe(1)
  })
  it('the single neuron learns AND; the moons preset gets above 95 %', () => {
    const a = createMlpStore('neuron'); a.getState().train(800)
    expect(accuracy(a.getState().cfg, a.getState().params, a.getState().data)).toBe(1)
    const m = createMlpStore('deep'); m.getState().train(5 * 200)
    expect(accuracy(m.getState().cfg, m.getState().params, m.getState().data)).toBeGreaterThan(0.95)
  })
})

import { DEEP_PRESETS, deepData, deepNet, deepVerdict } from './VanishLab'
import { initParams as ip, layerStats } from '../../engine/nn'
describe('vanishing-gradient presets give the verdict their label claims', () => {
  for (const p of DEEP_PRESETS) it(p.label, () => {
    const cfg = deepNet(p.cfg), st = layerStats(cfg, ip(cfg.sizes, p.cfg.seed, p.cfg.scheme, p.cfg.scale), deepData())
    const v = deepVerdict(st).verdict
    const want = /vanishing/.test(p.label) ? 'vanishing' : /exploding/.test(p.label) ? 'exploding' : 'healthy'
    expect(v).toBe(want)
  })
})

import { trainEpochs as te } from '../../engine/nn'
describe('vanishing lab training comparison (the claim in the UI)', () => {
  it('deep sigmoid barely moves in 200 steps; ReLU + He of the same depth drops well below it', () => {
    const run = (k: number) => { const p = DEEP_PRESETS[k].cfg, c = deepNet(p); return te(c, ip(c.sizes, p.seed, p.scheme, p.scale), deepData(), { ...defaultOpt, lr: 0.02 }, 32, 1, 200).losses }
    const s = run(0), r = run(2)
    expect(s[0] - s.at(-1)!).toBeLessThan(0.05)
    expect(r.at(-1)!).toBeLessThan(s.at(-1)! - 0.2)
  })
})

import { challengeNet, isCorrect, makeChallenges } from './challenges'
import { actProps, ELEMENTWISE } from './ActivationLab'
describe('challenges', () => {
  it('are seeded: same seed → same questions, different seed → different numbers', () => {
    expect(makeChallenges(3)).toEqual(makeChallenges(3))
    expect(makeChallenges(3).map((q) => q.answer)).not.toEqual(makeChallenges(4).map((q) => q.answer))
    expect(makeChallenges(3).length).toBe(9)
  })
  it('numeric answers agree with the engine and accept hand-rounded values', () => {
    for (let seed = 0; seed < 20; seed++) {
      const qs = makeChallenges(seed), { cfg, p, x } = challengeNet(seed), t = forward(cfg, p, x)
      const z = qs.find((q) => q.id === 'z')!
      expect(z.answer).toBeCloseTo(p.W[0][0][0] * x[0] + p.W[0][0][1] * x[1] + p.b[0][0], 12)
      expect(z.answer).toBe(t.z[1][0])
      for (const q of qs.filter((q) => !q.options)) {
        expect(isCorrect(q, (q.answer as number).toFixed(3))).toBe(true)
        expect(isCorrect(q, String((q.answer as number) + 0.5))).toBe(false)
      }
      const paths = qs.find((q) => q.id === 'paths')!, nb = paths.answer as number
      const m = /3 → (\d) → (\d) → 1/.exec(paths.prompt)!, c2 = { sizes: [3, +m[1], +m[2], 1], acts: ['tanh', 'tanh', 'sigmoid'] as const, loss: 'mse' as const }
      const cc = { ...c2, acts: [...c2.acts] }, pp = ip(cc.sizes, 1), tt = forward(cc, pp, [0.1, 0.2, 0.3])
      expect(chainPaths(cc, pp, tt, [1], { kind: 'w', layer: 1, j: 0, i: 0 }).paths.length).toBe(nb)
    }
  })
  it('multiple choice answers are among the options; accepts commas and the minus sign', () => {
    for (const q of makeChallenges(9)) if (q.options) expect(q.options).toContain(q.answer)
    const q = { id: 'x', prompt: '', answer: -0.25, explain: '' }
    expect(isCorrect(q, '−0,25')).toBe(true); expect(isCorrect(q, 'abc')).toBe(false)
  })
})
describe('activation explorer facts', () => {
  it('measured properties: sigmoid max slope 0.25, tanh 1, ReLU half flat, linear never flat', () => {
    const p = Object.fromEntries(ELEMENTWISE.map((f) => [f, actProps(f)]))
    expect(p.sigmoid.maxDeriv).toBeCloseTo(0.25, 10); expect(p.tanh.maxDeriv).toBeCloseTo(1, 10)
    expect(p.sigmoid.zeroCentred).toBe(false); expect(p.tanh.zeroCentred).toBe(true)
    expect(p.sigmoid.saturating && p.tanh.saturating).toBe(true); expect(p.relu.saturating).toBe(false)
    expect(p.relu.flatShare).toBeGreaterThan(0.49); expect(p.linear.flatShare).toBe(0); expect(p.leaky.flatShare).toBe(0)
  })
})
