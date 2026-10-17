import { describe, expect, it } from 'vitest'
import { DIGITS, DIGITS_CFG, digitSample, digitTest, digitTrain, isTrainedDigits, trainedDigitParams } from './digits'
import {
  accuracy, backward, confusion, datasetLoss, forward, gradientCheck, initParams, inputGrad, layerStats, lossValue, output, setParam, targetsFor, makeDataset,
  type Act, type NetConfig,
} from './nn'

describe('trained digit classifier', () => {
  const p = trainedDigitParams()
  it('data: 1,797 images, 1,347 / 450 disjoint stratified split, pixels 0..16', () => {
    expect(DIGITS.X.length).toBe(1797); expect(DIGITS.train.length).toBe(1347); expect(DIGITS.test.length).toBe(450)
    expect(new Set([...DIGITS.train, ...DIGITS.test]).size).toBe(1797)
    expect(Math.max(...DIGITS.X.flat())).toBe(16); expect(Math.min(...DIGITS.X.flat())).toBe(0)
    for (let k = 0; k < 10; k++) expect(DIGITS.test.filter((i) => DIGITS.y[i] === k).length).toBeGreaterThanOrEqual(43)
  })
  it('the browser forward pass reproduces the NumPy reference outputs', () => {
    DIGITS.reference.ids.forEach((id, r) => {
      const t = forward(DIGITS_CFG, p, digitSample(id).x)
      t.z[3].forEach((v, k) => expect(v).toBeCloseTo(DIGITS.reference.logits[r][k], 6))
      output(t).forEach((v, k) => expect(v).toBeCloseTo(DIGITS.reference.probs[r][k], 7))
    })
  })
  it('test accuracy and loss match the exported metrics (the numbers shown in the UI)', () => {
    expect(accuracy(DIGITS_CFG, p, digitTest())).toBeCloseTo(DIGITS.metrics.testAcc, 12)
    expect(datasetLoss(DIGITS_CFG, p, digitTest())).toBeCloseTo(DIGITS.metrics.testLoss, 9)
    expect(accuracy(DIGITS_CFG, p, digitTrain())).toBeCloseTo(DIGITS.metrics.trainAcc, 12)
    expect((DIGITS.metrics.testAcc * 100).toFixed(1)).toBe('97.3')
    expect(confusion(DIGITS_CFG, p, digitTest(), 10).matrix).toEqual(DIGITS.metrics.confusion)
  })
  it('the training curve shows over-fitting: train loss → ~0 while test loss stays ≥ 0.1', () => {
    const last = DIGITS.curve.at(-1)!
    expect(last.trainLoss).toBeLessThan(0.01); expect(last.testLoss).toBeGreaterThan(0.1)
    expect(Math.min(...DIGITS.curve.map((c) => c.testLoss))).toBeLessThan(last.testLoss)
  })
  it('backprop on the trained network passes the gradient check (sampled parameters)', () => {
    const refs = [{ kind: 'w' as const, layer: 1, j: 3, i: 20 }, { kind: 'w' as const, layer: 2, j: 7, i: 1 }, { kind: 'b' as const, layer: 3, j: 4 }, { kind: 'w' as const, layer: 3, j: 9, i: 15 }]
    for (const r of gradientCheck(DIGITS_CFG, p, digitTest().slice(0, 5), 1e-5, refs)) expect(r.relErr < 1e-5 || r.absErr < 1e-9).toBe(true)
  })
  it('input saliency: moving x along −∂L/∂x raises p(true class)', () => {
    const s = digitTest()[0], g = inputGrad(DIGITS_CFG, p, s.x, s.y)
    const x2 = s.x.map((v, i) => v - 0.01 * g[i])
    expect(output(forward(DIGITS_CFG, p, x2))[s.label!]).toBeGreaterThan(output(forward(DIGITS_CFG, p, s.x))[s.label!])
  })
  it('isTrainedDigits detects any change', () => {
    expect(isTrainedDigits(DIGITS_CFG, p)).toBe(true)
    expect(isTrainedDigits(DIGITS_CFG, setParam(p, { kind: 'b', layer: 1, j: 0 }, 1))).toBe(false)
  })
})

describe('depth diagnostics', () => {
  const deep = (act: Act, depth: number, width = 16): NetConfig => ({ sizes: [8, ...new Array(depth).fill(width), 1], acts: [...new Array(depth).fill(act), 'sigmoid'], loss: 'bce' })
  const data = targetsFor(makeDataset('random', 3, 8, 32), 1)
  it('a deep sigmoid network: gradients vanish towards the input (over 1000× smaller)', () => {
    const cfg = deep('sigmoid', 12), st = layerStats(cfg, initParams(cfg.sizes, 1, 'xavier'), data)
    expect(st[st.length - 1].gradNorm / st[0].gradNorm).toBeGreaterThan(1000)
    expect(st.length).toBe(13)
  })
  it('ReLU with He initialization keeps gradient norms within two orders of magnitude', () => {
    const cfg = deep('relu', 12), st = layerStats(cfg, initParams(cfg.sizes, 1, 'he'), data)
    const n = st.slice(0, -1).map((s) => s.gradNorm)
    expect(Math.max(...n) / Math.min(...n)).toBeLessThan(100)
  })
  it('large initial weights explode the backpropagated error δ towards the input', () => {
    const cfg = deep('leaky', 12), st = layerStats(cfg, initParams(cfg.sizes, 1, 'normal', 1.5), data)
    expect(st[0].deltaRms / st[st.length - 2].deltaRms).toBeGreaterThan(100)
  })
  it('saturation and dead-unit fractions are measured from the forward pass', () => {
    const big = deep('tanh', 4), st = layerStats(big, initParams(big.sizes, 2, 'normal', 4), data)
    expect(st[1].saturated).toBeGreaterThan(0.5)
    const r = deep('relu', 4), sr = layerStats(r, initParams(r.sizes, 2, 'he'), data)
    sr.slice(0, -1).forEach((s) => { expect(s.dead).toBeGreaterThan(0.2); expect(s.dead).toBeLessThan(0.8) })
  })
  it('the gradient norm of the last layer equals the Frobenius norm of the batch gradient', () => {
    const cfg = deep('tanh', 2), p = initParams(cfg.sizes, 4), st = layerStats(cfg, p, data)
    const bs = data.map((d) => backward(cfg, p, forward(cfg, p, d.x), d.y))
    const m = bs[0].dW[2].map((row, j) => row.map((_, i) => bs.reduce((s, b) => s + b.dW[2][j][i], 0) / data.length))
    expect(st[2].gradNorm).toBeCloseTo(Math.sqrt(m.flat().reduce((s, v) => s + v * v, 0)), 12)
    expect(lossValue(cfg, forward(cfg, p, data[0].x), data[0].y)).toBeGreaterThan(0)
  })
})
