import { describe, expect, it } from 'vitest'
import {
  WORKED, accuracy, actDeriv, actFn, allParamRefs, applyUpdate, backward, batchGrad, chainPaths, cloneParams, datasetLoss, decisionGrid,
  defaultOpt, epochOrder, forward, getParam, gradOf, gradientCheck, initParams, lossCompat, lossGrad, lossValue, makeDataset, paramCount,
  paramName, setParam, sigmoid, softmax, softmaxJacobian, targetsFor, trainEpochs, trainStep, validateConfig, zeroOptState,
  type Act, type LossKind, type NetConfig, type OptConfig, type TrainState,
} from './nn'

const close = (a: number, b: number, tol = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(tol)

describe('activations', () => {
  it('sigmoid is stable at extremes and symmetric', () => {
    expect(sigmoid(800)).toBe(1); expect(sigmoid(-800)).toBe(0)
    close(sigmoid(0.7) + sigmoid(-0.7), 1, 1e-15)
  })
  it('derivatives match central differences', () => {
    for (const f of ['sigmoid', 'tanh', 'relu', 'leaky', 'linear'] as Act[]) for (const z of [-1.7, -0.2, 0.3, 2.1]) {
      const h = 1e-6
      close(actDeriv(f, z), (actFn(f, z + h) - actFn(f, z - h)) / (2 * h), 1e-7)
    }
  })
  it('ReLU′(0) is taken as 0', () => { expect(actDeriv('relu', 0)).toBe(0) })
  it('softmax sums to 1, is shift invariant, and its Jacobian matches finite differences', () => {
    const z = [0.3, -1.2, 2.0], a = softmax(z)
    close(a.reduce((s, v) => s + v, 0), 1, 1e-15)
    softmax(z.map((v) => v + 1000)).forEach((v, k) => close(v, a[k], 1e-12))
    const J = softmaxJacobian(a), h = 1e-6
    for (let r = 0; r < 3; r++) {
      const zp = z.slice(); zp[r] += h; const zm = z.slice(); zm[r] -= h
      const ap = softmax(zp), am = softmax(zm)
      for (let k = 0; k < 3; k++) close(J[k][r], (ap[k] - am[k]) / (2 * h), 1e-8)
    }
  })
})

describe('the worked example (numbers quoted in the lesson)', () => {
  const { cfg, params, x, y } = WORKED
  const t = forward(cfg, params, x)
  it('hidden neuron 1: z = 0.5·1.0 − 0.3·0.8 + 0.1 = 0.36', () => {
    close(t.z[1][0], 0.36, 1e-15)
    close(t.a[1][0], 1 / (1 + Math.exp(-0.36)), 1e-15)
    close(t.z[1][1], 0.2 * 1 + 0.4 * 0.8 - 0.1, 1e-15)
  })
  it('output and loss are reproduced by hand', () => {
    const h = t.a[1], z2 = 0.6 * h[0] - 0.4 * h[1] + 0.05, yh = sigmoid(z2)
    close(t.z[2][0], z2); close(t.a[2][0], yh)
    close(lossValue(cfg, t, y), (yh - 1) ** 2)
    close(lossGrad(cfg, [yh], y)[0], 2 * (yh - 1))
  })
  it('rounded values used in the text', () => {
    expect(t.a[1][0].toFixed(4)).toBe('0.5890')
    expect(t.a[1][1].toFixed(4)).toBe('0.6035')
    expect(t.a[2][0].toFixed(4)).toBe('0.5404')
    expect(lossValue(cfg, t, y).toFixed(4)).toBe('0.2112')
  })
  it('backprop by hand for w₁₁⁽²⁾ and w₁₁⁽¹⁾', () => {
    const g = backward(cfg, params, t, y)
    const yh = t.a[2][0], d2 = 2 * (yh - 1) * yh * (1 - yh)
    close(g.delta[2][0], d2)
    close(g.dW[1][0][0], d2 * t.a[1][0])
    const d1 = 0.6 * d2 * t.a[1][0] * (1 - t.a[1][0])
    close(g.delta[1][0], d1)
    close(g.dW[0][0][0], d1 * 1.0)
    close(g.dW[0][0][1], d1 * 0.8)
    close(g.db[0][0], d1)
    expect(g.outputRule).toBe('chain')
  })
})

/** every activation / loss combination that the builder allows */
const combos: { name: string; cfg: NetConfig; y: number[] }[] = []
for (const h of ['sigmoid', 'tanh', 'relu', 'leaky', 'linear'] as Act[]) {
  combos.push({ name: `${h} → sigmoid / MSE`, cfg: { sizes: [3, 4, 2], acts: [h, 'sigmoid'], loss: 'mse' }, y: [1, 0] })
  combos.push({ name: `${h} → linear / MSE`, cfg: { sizes: [3, 4, 2], acts: [h, 'linear'], loss: 'mse' }, y: [0.3, -0.7] })
  combos.push({ name: `${h} → sigmoid / BCE`, cfg: { sizes: [3, 4, 2], acts: [h, 'sigmoid'], loss: 'bce' }, y: [1, 0] })
  combos.push({ name: `${h} → softmax / CCE`, cfg: { sizes: [3, 4, 3], acts: [h, 'softmax'], loss: 'cce' }, y: [0, 0, 1] })
}
combos.push({ name: 'deep 3-4-4-2 tanh/relu/sigmoid', cfg: { sizes: [3, 4, 4, 2], acts: ['tanh', 'relu', 'sigmoid'], loss: 'mse' }, y: [0, 1] })
combos.push({ name: 'single neuron sigmoid / BCE', cfg: { sizes: [2, 1], acts: ['sigmoid'], loss: 'bce' }, y: [1] })

describe('backpropagation equals finite differences', () => {
  for (const c of combos) it(c.name, () => {
    expect(validateConfig(c.cfg).ok).toBe(true)
    const p = initParams(c.cfg.sizes, 7, 'xavier', 1.3)
    // biases away from zero so ReLU kinks are not hit exactly
    p.b = p.b.map((v, l) => v.map((_, j) => 0.11 * (j + 1) - 0.05 * l))
    const data = [{ x: [0.4, -0.9, 1.3], y: c.y }, { x: [-0.2, 0.5, 0.1], y: c.y.slice().reverse() }].map((d) => ({ ...d, x: d.x.slice(0, c.cfg.sizes[0]) }))
    const rows = gradientCheck(c.cfg, p, data, 1e-5)
    expect(rows.length).toBe(paramCount(c.cfg.sizes))
    for (const r of rows) expect(r.absErr).toBeLessThan(1e-7)
  })
})

describe('matched output pairs', () => {
  it('sigmoid + BCE gives δ = (ŷ − y)/K, identical to the generic chain', () => {
    const cfg: NetConfig = { sizes: [2, 3, 2], acts: ['tanh', 'sigmoid'], loss: 'bce' }
    const p = initParams(cfg.sizes, 3), t = forward(cfg, p, [0.5, -0.2]), y = [1, 0]
    const g = backward(cfg, p, t, y)
    expect(g.outputRule).toBe('sigmoid+bce')
    const dA = lossGrad(cfg, t.a[2], y)
    g.delta[2].forEach((d, k) => { close(d, (t.a[2][k] - y[k]) / 2); close(d, dA[k] * actDeriv('sigmoid', t.z[2][k]), 1e-12) })
  })
  it('softmax + CCE gives δ = ŷ − y, identical to Jacobianᵀ · ∂L/∂ŷ', () => {
    const cfg: NetConfig = { sizes: [2, 3, 3], acts: ['relu', 'softmax'], loss: 'cce' }
    const p = initParams(cfg.sizes, 5), t = forward(cfg, p, [0.5, -0.2]), y = [0, 1, 0]
    const g = backward(cfg, p, t, y)
    expect(g.outputRule).toBe('softmax+cce')
    const J = softmaxJacobian(t.a[2]), dA = lossGrad(cfg, t.a[2], y)
    g.delta[2].forEach((d, r) => { close(d, t.a[2][r] - y[r]); close(d, J.reduce((s, row, k) => s + row[r] * dA[k], 0), 1e-12) })
  })
  it('BCE and CCE stay finite for extreme logits', () => {
    const cfg: NetConfig = { sizes: [1, 2], acts: ['softmax'], loss: 'cce' }
    const p = { W: [[[400], [-400]]], b: [[0, 0]] }
    const l = lossValue(cfg, forward(cfg, p, [1]), [0, 1])
    close(l, 800, 1e-9)
    const cb: NetConfig = { sizes: [1, 1], acts: ['sigmoid'], loss: 'bce' }
    close(lossValue(cb, forward(cb, { W: [[[500]]], b: [[0]] }, [1]), [0]), 500, 1e-9)
  })
})

describe('chain-rule paths', () => {
  it('the sum over all paths equals the backpropagated gradient (every parameter, deep net)', () => {
    const cfg: NetConfig = { sizes: [3, 4, 3, 2], acts: ['tanh', 'sigmoid', 'sigmoid'], loss: 'mse' }
    const p = initParams(cfg.sizes, 11), x = [0.2, -0.5, 0.9], y = [1, 0]
    const t = forward(cfg, p, x), g = backward(cfg, p, t, y)
    for (const ref of allParamRefs(cfg.sizes)) close(chainPaths(cfg, p, t, y, ref).total, gradOf(g, ref), 1e-12)
  })
  it('path count: a first-layer weight in 3-4-3-2 reaches the loss through 3·2 = 6 paths', () => {
    const cfg: NetConfig = { sizes: [3, 4, 3, 2], acts: ['tanh', 'sigmoid', 'sigmoid'], loss: 'mse' }
    const p = initParams(cfg.sizes, 11), t = forward(cfg, p, [0.2, -0.5, 0.9])
    const r = chainPaths(cfg, p, t, [1, 0], { kind: 'w', layer: 1, j: 2, i: 0 })
    expect(r.paths.length).toBe(6)
    expect(r.truncated).toBe(false)
    for (const q of r.paths) { close(q.product, q.factors.reduce((s, f) => s * f.value, 1), 1e-15); expect(q.factors.length).toBe(7) }
  })
  it('works through softmax outputs and for activations (∂L/∂a)', () => {
    const cfg: NetConfig = { sizes: [2, 3, 3], acts: ['leaky', 'softmax'], loss: 'cce' }
    const p = initParams(cfg.sizes, 2), t = forward(cfg, p, [0.7, -0.4]), y = [0, 0, 1], g = backward(cfg, p, t, y)
    for (const ref of allParamRefs(cfg.sizes)) close(chainPaths(cfg, p, t, y, ref).total, gradOf(g, ref), 1e-12)
    for (let j = 0; j < 3; j++) close(chainPaths(cfg, p, t, y, { kind: 'a', layer: 1, j }).total, g.dA[1][j], 1e-12)
  })
  it('the worked example w₁₁⁽¹⁾ has exactly one path with the factors of the lesson', () => {
    const { cfg, params, x, y } = WORKED, t = forward(cfg, params, x)
    const r = chainPaths(cfg, params, t, y, { kind: 'w', layer: 1, j: 0, i: 0 })
    expect(r.paths.length).toBe(1)
    expect(r.paths[0].factors.map((f) => f.kind)).toEqual(['local-input', 'activation', 'weight', 'activation', 'loss'])
  })
  it('truncates huge path sets', () => {
    const cfg: NetConfig = { sizes: [2, 10, 10, 10, 2], acts: ['tanh', 'tanh', 'tanh', 'sigmoid'], loss: 'mse' }
    const p = initParams(cfg.sizes, 1), t = forward(cfg, p, [0.1, 0.2])
    const r = chainPaths(cfg, p, t, [1, 0], { kind: 'w', layer: 1, j: 0, i: 0 }, 50)
    expect(r.truncated).toBe(true); expect(r.paths.length).toBe(50)
  })
})

describe('batches', () => {
  it('batch gradient is the mean of per-sample gradients', () => {
    const cfg: NetConfig = { sizes: [2, 3, 1], acts: ['tanh', 'sigmoid'], loss: 'bce' }
    const p = initParams(cfg.sizes, 4), data = targetsFor(makeDataset('xor', 1, 2), 1)
    const bg = batchGrad(cfg, p, data)
    const each = data.map((d) => backward(cfg, p, forward(cfg, p, d.x), d.y))
    for (const ref of allParamRefs(cfg.sizes)) close(gradOf(bg, ref), each.reduce((s, g) => s + gradOf(g, ref), 0) / 4, 1e-14)
    close(bg.loss, datasetLoss(cfg, p, data), 1e-14)
  })
})

describe('optimizers', () => {
  const cfg = WORKED.cfg, p = WORKED.params, g = backward(cfg, p, forward(cfg, p, WORKED.x), WORKED.y)
  it('SGD: θ ← θ − η·g', () => {
    const o: OptConfig = { ...defaultOpt, kind: 'sgd', lr: 0.5 }
    const u = applyUpdate(p, g, zeroOptState(p), o)
    for (const r of u.records) { close(r.next, r.old - 0.5 * r.grad, 1e-15); expect(getParam(u.params, r.ref)).toBe(r.next) }
    expect(u.records.length).toBe(paramCount(cfg.sizes))
    expect(getParam(p, { kind: 'w', layer: 1, j: 0, i: 0 })).toBe(0.5) // input not mutated
  })
  it('momentum: v ← μv + g, θ ← θ − ηv (two steps)', () => {
    const o: OptConfig = { ...defaultOpt, kind: 'momentum', lr: 0.1, momentum: 0.9 }
    const u1 = applyUpdate(p, g, zeroOptState(p), o), u2 = applyUpdate(u1.params, g, u1.state, o)
    const r = u2.records[0]
    close(r.detail.velocity, 0.9 * r.grad + r.grad, 1e-15)
    close(r.next, u1.records[0].next - 0.1 * 1.9 * r.grad, 1e-15)
  })
  it('Adam: the first step has magnitude ≈ η for every non-zero gradient (bias correction)', () => {
    const o: OptConfig = { ...defaultOpt, kind: 'adam', lr: 0.01 }
    const u = applyUpdate(p, g, zeroOptState(p), o)
    for (const r of u.records) { close(Math.abs(r.step), 0.01, 1e-7); close(r.detail.mHat, r.grad, 1e-15); close(r.detail.vHat, r.grad ** 2, 1e-15) }
    expect(u.state.t).toBe(1)
  })
})

describe('initialization and determinism', () => {
  it('same seed → identical parameters; different seed → different', () => {
    expect(initParams([64, 16, 16, 10], 42, 'he')).toEqual(initParams([64, 16, 16, 10], 42, 'he'))
    expect(initParams([2, 3, 1], 1)).not.toEqual(initParams([2, 3, 1], 2))
  })
  it('xavier bound respected, biases zero, counts right', () => {
    const p = initParams([4, 6, 2], 9, 'xavier'), a = Math.sqrt(6 / 10)
    p.W[0].flat().forEach((w) => expect(Math.abs(w)).toBeLessThanOrEqual(a))
    expect(p.b.flat().every((v) => v === 0)).toBe(true)
    expect(paramCount([64, 16, 16, 10])).toBe(64 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10)
    expect(allParamRefs([2, 3, 1]).length).toBe(paramCount([2, 3, 1]))
  })
  it('setParam copies, cloneParams is deep', () => {
    const p = initParams([2, 2, 1], 1), q = setParam(p, { kind: 'b', layer: 2, j: 0 }, 3)
    expect(p.b[1][0]).toBe(0); expect(q.b[1][0]).toBe(3)
    const c = cloneParams(p); c.W[0][0][0] = 99; expect(p.W[0][0][0]).not.toBe(99)
    expect(paramName({ kind: 'w', layer: 2, j: 0, i: 1 })).toBe('w₁₂⁽²⁾')
  })
  it('epoch shuffles are seeded permutations', () => {
    const o = epochOrder(10, 3, 0)
    expect(o.slice().sort((a, b) => a - b)).toEqual([...Array(10).keys()])
    expect(epochOrder(10, 3, 0)).toEqual(o); expect(epochOrder(10, 3, 1)).not.toEqual(o)
  })
  it('training replays exactly from any recorded snapshot', () => {
    const cfg: NetConfig = { sizes: [2, 4, 1], acts: ['tanh', 'sigmoid'], loss: 'bce' }
    const data = targetsFor(makeDataset('moons', 2, 2, 40), 1), o: OptConfig = { ...defaultOpt, kind: 'adam', lr: 0.05 }
    const p0 = initParams(cfg.sizes, 8)
    let ts: TrainState = { params: p0, opt: zeroOptState(p0), epoch: 0, iter: 0, cursor: 0 }
    const snaps: TrainState[] = [ts]
    for (let k = 0; k < 30; k++) { ts = trainStep(cfg, ts, data, o, 8, 5).state; snaps.push(ts) }
    let rs = snaps[12]
    for (let k = 12; k < 30; k++) rs = trainStep(cfg, rs, data, o, 8, 5).state
    expect(rs).toEqual(ts)
    expect(ts.epoch).toBe(6); expect(ts.iter).toBe(30)
  })
})

describe('learning', () => {
  const xor = targetsFor(makeDataset('xor', 1, 2), 1)
  it('XOR: no hidden layer cannot solve it (accuracy ≤ 75 %)', () => {
    const cfg: NetConfig = { sizes: [2, 1], acts: ['sigmoid'], loss: 'bce' }
    const r = trainEpochs(cfg, initParams(cfg.sizes, 1), xor, { ...defaultOpt, lr: 1 }, 4, 1, 2000)
    expect(accuracy(cfg, r.state.params, xor)).toBeLessThanOrEqual(0.75)
    expect(r.losses[r.losses.length - 1]).toBeGreaterThan(0.6)
  })
  it('XOR: 2 → 4 → 1 tanh learns it', () => {
    const cfg: NetConfig = { sizes: [2, 4, 1], acts: ['tanh', 'sigmoid'], loss: 'bce' }
    const r = trainEpochs(cfg, initParams(cfg.sizes, 1), xor, { ...defaultOpt, lr: 1 }, 4, 1, 1500)
    expect(accuracy(cfg, r.state.params, xor)).toBe(1)
    expect(r.losses[r.losses.length - 1]).toBeLessThan(0.05)
  })
  it('loss decreases on the worked example with plain SGD', () => {
    const r = trainEpochs(WORKED.cfg, WORKED.params, [{ x: WORKED.x, y: WORKED.y }], defaultOpt, 1, 1, 20)
    for (let k = 1; k < r.losses.length; k++) expect(r.losses[k]).toBeLessThan(r.losses[k - 1])
  })
  it('decision grid has n² cells with probabilities', () => {
    const cfg: NetConfig = { sizes: [2, 3, 1], acts: ['tanh', 'sigmoid'], loss: 'bce' }
    const g = decisionGrid(cfg, initParams(cfg.sizes, 1), [-1, 1], [-1, 1], 10)
    expect(g.length).toBe(100); g.forEach((c) => { expect(c.p).toBeGreaterThan(0); expect(c.p).toBeLessThan(1) })
  })
})

describe('configuration rules', () => {
  const base = { sizes: [2, 3, 1], acts: ['relu', 'sigmoid'] as Act[] }
  it.each<[LossKind, Act, number, boolean]>([
    ['mse', 'sigmoid', 1, true], ['mse', 'linear', 1, true], ['bce', 'sigmoid', 1, true], ['bce', 'linear', 1, false],
    ['cce', 'softmax', 3, true], ['cce', 'sigmoid', 3, false], ['cce', 'softmax', 1, false], ['mse', 'softmax', 3, false],
  ])('%s with %s output (%i units) → %s', (loss, out, K, ok) => {
    expect(lossCompat({ sizes: [2, 3, K], acts: ['relu', out] }, loss).ok).toBe(ok)
  })
  it('softmax is rejected on hidden layers', () => {
    expect(validateConfig({ sizes: [2, 3, 1], acts: ['softmax', 'sigmoid'], loss: 'mse' }).ok).toBe(false)
    expect(validateConfig({ ...base, loss: 'mse' }).ok).toBe(true)
  })
  it('forward rejects wrong input sizes', () => { expect(() => forward(WORKED.cfg, WORKED.params, [1])).toThrow() })
})
