/**
 * Fully connected neural networks, computed exactly, for module 04.
 *
 * Conventions
 *  - sizes = [n0, n1, …, nL]; layer l = 1…L has weights W[l−1] of shape [n_l × n_{l−1}] (row j = neuron j)
 *    and biases b[l−1] of length n_l. z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾), a⁽⁰⁾ = x.
 *  - Losses are averaged over output units (MSE, BCE) and over the samples of a batch.
 *    With one output, MSE is L = (ŷ − y)² and ∂L/∂ŷ = 2(ŷ − y).
 *  - Everything the UI shows (activations, every chain-rule factor, gradients, updates) comes from here.
 *    The visualization never computes values of its own.
 */
import { rng } from './decoding'

export type Act = 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'linear' | 'softmax'
export type LossKind = 'mse' | 'bce' | 'cce'
export type InitScheme = 'xavier' | 'he' | 'uniform' | 'normal'
export type OptKind = 'sgd' | 'momentum' | 'adam'

export const ACT_LABEL: Record<Act, string> = { sigmoid: 'Sigmoid', tanh: 'Tanh', relu: 'ReLU', leaky: 'Leaky ReLU', linear: 'Linear', softmax: 'Softmax' }
export const LOSS_LABEL: Record<LossKind, string> = { mse: 'Mean squared error', bce: 'Binary cross-entropy', cce: 'Categorical cross-entropy' }
export const LEAK = 0.01

export interface NetConfig { sizes: number[]; acts: Act[]; loss: LossKind }
export interface Params { W: number[][][]; b: number[][] }

/* ------------------------------------------------------------------ activations */
export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))
export function actFn(f: Act, z: number): number {
  switch (f) {
    case 'sigmoid': return sigmoid(z)
    case 'tanh': return Math.tanh(z)
    case 'relu': return z > 0 ? z : 0
    case 'leaky': return z > 0 ? z : LEAK * z
    case 'linear': return z
    case 'softmax': throw new Error('softmax is a vector function')
  }
}
/** f′(z). ReLU′(0) is taken as 0 (a subgradient), stated in the UI. */
export function actDeriv(f: Act, z: number): number {
  switch (f) {
    case 'sigmoid': { const s = sigmoid(z); return s * (1 - s) }
    case 'tanh': { const t = Math.tanh(z); return 1 - t * t }
    case 'relu': return z > 0 ? 1 : 0
    case 'leaky': return z > 0 ? 1 : LEAK
    case 'linear': return 1
    case 'softmax': throw new Error('softmax derivative is a Jacobian')
  }
}
export function softmax(z: number[]): number[] { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }
export const logSumExp = (z: number[]) => { const m = Math.max(...z); return m + Math.log(z.reduce((s, v) => s + Math.exp(v - m), 0)) }
export function activate(f: Act, z: number[]): number[] { return f === 'softmax' ? softmax(z) : z.map((v) => actFn(f, v)) }
/** Softmax Jacobian J[k][r] = ∂a_k/∂z_r = a_k (δ_kr − a_r). */
export const softmaxJacobian = (a: number[]) => a.map((ak, k) => a.map((ar, r) => ak * ((k === r ? 1 : 0) - ar)))

/* ------------------------------------------------------------------ configuration checks */
export interface Compat { ok: boolean; reason?: string }
/** Only mathematically compatible output/loss combinations are allowed. */
export function lossCompat(cfg: Pick<NetConfig, 'sizes' | 'acts'>, loss: LossKind): Compat {
  const out = cfg.acts[cfg.acts.length - 1], K = cfg.sizes[cfg.sizes.length - 1]
  if (loss === 'mse') return out === 'softmax' ? { ok: false, reason: 'MSE on a softmax output is possible but not offered here; use categorical cross-entropy.' } : { ok: true }
  if (loss === 'bce') return out === 'sigmoid' ? { ok: true } : { ok: false, reason: 'Binary cross-entropy needs sigmoid outputs in (0, 1).' }
  if (out !== 'softmax') return { ok: false, reason: 'Categorical cross-entropy needs a softmax output layer.' }
  return K >= 2 ? { ok: true } : { ok: false, reason: 'Categorical cross-entropy needs at least 2 output classes.' }
}
export function actAllowed(f: Act, isOutput: boolean, nOut: number): Compat {
  if (f === 'softmax' && !isOutput) return { ok: false, reason: 'Softmax is only offered on the output layer.' }
  if (f === 'softmax' && nOut < 2) return { ok: false, reason: 'Softmax needs at least 2 outputs.' }
  return { ok: true }
}
export function validateConfig(cfg: NetConfig): Compat {
  if (cfg.sizes.length < 2) return { ok: false, reason: 'A network needs an input and an output layer.' }
  if (cfg.acts.length !== cfg.sizes.length - 1) return { ok: false, reason: 'One activation per non-input layer.' }
  if (cfg.sizes.some((n) => !Number.isInteger(n) || n < 1)) return { ok: false, reason: 'Every layer needs at least one neuron.' }
  for (let l = 0; l < cfg.acts.length; l++) { const c = actAllowed(cfg.acts[l], l === cfg.acts.length - 1, cfg.sizes[l + 1]); if (!c.ok) return c }
  return lossCompat(cfg, cfg.loss)
}

/* ------------------------------------------------------------------ parameters */
function gauss(r: () => number) { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
/** Reproducible initialization. Biases start at 0. */
export function initParams(sizes: number[], seed: number, scheme: InitScheme = 'xavier', scale = 1): Params {
  const r = rng(seed * 2654435761 % 4294967296 + 17)
  const W: number[][][] = [], b: number[][] = []
  for (let l = 1; l < sizes.length; l++) {
    const fi = sizes[l - 1], fo = sizes[l]
    const draw = () => {
      if (scheme === 'xavier') { const a = Math.sqrt(6 / (fi + fo)); return (r() * 2 - 1) * a * scale }
      if (scheme === 'he') return gauss(r) * Math.sqrt(2 / fi) * scale
      if (scheme === 'uniform') return (r() * 2 - 1) * scale
      return gauss(r) * scale
    }
    W.push(Array.from({ length: fo }, () => Array.from({ length: fi }, draw)))
    b.push(new Array(fo).fill(0))
  }
  return { W, b }
}
export const cloneParams = (p: Params): Params => ({ W: p.W.map((L) => L.map((r) => r.slice())), b: p.b.map((v) => v.slice()) })
export const paramCount = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + n * sizes[l] + n, 0)

/** A reference to one trainable parameter. layer is 1-based (layer of the neuron that owns it). */
export type ParamRef = { kind: 'w'; layer: number; j: number; i: number } | { kind: 'b'; layer: number; j: number }
export const getParam = (p: Params, r: ParamRef) => (r.kind === 'w' ? p.W[r.layer - 1][r.j][r.i] : p.b[r.layer - 1][r.j])
export function setParam(p: Params, r: ParamRef, v: number): Params {
  const q = cloneParams(p)
  if (r.kind === 'w') q.W[r.layer - 1][r.j][r.i] = v; else q.b[r.layer - 1][r.j] = v
  return q
}
export function allParamRefs(sizes: number[]): ParamRef[] {
  const out: ParamRef[] = []
  for (let l = 1; l < sizes.length; l++) for (let j = 0; j < sizes[l]; j++) { for (let i = 0; i < sizes[l - 1]; i++) out.push({ kind: 'w', layer: l, j, i }); out.push({ kind: 'b', layer: l, j }) }
  return out
}
export const sub = (n: number) => String(n).split('').map((d) => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('')
export const sup = (n: number) => String(n).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('')
export const paramName = (r: ParamRef) => (r.kind === 'w' ? `w${sub(r.j + 1)}${sub(r.i + 1)}⁽${sup(r.layer)}⁾` : `b${sub(r.j + 1)}⁽${sup(r.layer)}⁾`)

/* ------------------------------------------------------------------ forward */
export interface Trace { a: number[][]; z: number[][] }
/** z[l] and a[l] are indexed by layer: z[0] is unused (empty), a[0] = x. */
export function forward(cfg: NetConfig, p: Params, x: number[]): Trace {
  if (x.length !== cfg.sizes[0]) throw new Error(`input has ${x.length} values, the network expects ${cfg.sizes[0]}`)
  const a: number[][] = [x.slice()], z: number[][] = [[]]
  for (let l = 1; l < cfg.sizes.length; l++) {
    const W = p.W[l - 1], b = p.b[l - 1], prev = a[l - 1]
    if (W.length !== cfg.sizes[l] || W[0].length !== prev.length) throw new Error(`W⁽${sup(l)}⁾ has shape ${W.length}×${W[0].length}, expected ${cfg.sizes[l]}×${prev.length}`)
    const zl = W.map((row, j) => row.reduce((s, w, i) => s + w * prev[i], b[j]))
    z.push(zl); a.push(activate(cfg.acts[l - 1], zl))
  }
  return { a, z }
}
export const output = (t: Trace) => t.a[t.a.length - 1]

/* ------------------------------------------------------------------ losses */
const EPS = 1e-12
/** Loss for one sample. CCE is computed from the logits with log-sum-exp (numerically stable). */
export function lossValue(cfg: NetConfig, t: Trace, y: number[]): number {
  const yh = output(t), K = yh.length
  if (cfg.loss === 'mse') return yh.reduce((s, v, k) => s + (v - y[k]) ** 2, 0) / K
  if (cfg.loss === 'bce') {
    const zL = t.z[t.z.length - 1]
    // −[y log σ(z) + (1−y) log(1−σ(z))] = softplus(z) − y z  (stable for any z)
    return zL.reduce((s, zk, k) => s + (Math.max(zk, 0) + Math.log1p(Math.exp(-Math.abs(zk))) - y[k] * zk), 0) / K
  }
  const zL = t.z[t.z.length - 1], lse = logSumExp(zL)
  return -y.reduce((s, yk, k) => s + yk * (zL[k] - lse), 0)
}
/** ∂L/∂ŷ for one sample (the first factor of the chain rule). */
export function lossGrad(cfg: NetConfig, yh: number[], y: number[]): number[] {
  const K = yh.length
  if (cfg.loss === 'mse') return yh.map((v, k) => (2 * (v - y[k])) / K)
  if (cfg.loss === 'bce') return yh.map((v, k) => { const c = Math.min(1 - EPS, Math.max(EPS, v)); return (c - y[k]) / (c * (1 - c)) / K })
  return yh.map((v, k) => -y[k] / Math.max(v, EPS))
}

/* ------------------------------------------------------------------ backward */
export interface Backward {
  /** ∂L/∂a⁽ˡ⁾ for every layer (dA[0] is ∂L/∂x) */
  dA: number[][]
  /** f′(z⁽ˡ⁾) for elementwise activations (empty for softmax layers) */
  fprime: number[][]
  /** δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ */
  delta: number[][]
  /** ∂L/∂W⁽ˡ⁾ and ∂L/∂b⁽ˡ⁾, stored at index l − 1 like the parameters */
  dW: number[][][]
  db: number[][]
  /** how δ⁽ᴸ⁾ was obtained: generic chain rule, or the simplified ŷ − y of a matched pair */
  outputRule: 'chain' | 'sigmoid+bce' | 'softmax+cce'
}
export function backward(cfg: NetConfig, p: Params, t: Trace, y: number[]): Backward {
  const L = cfg.sizes.length - 1
  const dA: number[][] = new Array(L + 1), fprime: number[][] = new Array(L + 1).fill([]), delta: number[][] = new Array(L + 1).fill([])
  const yh = t.a[L], K = yh.length
  dA[L] = lossGrad(cfg, yh, y)
  const outAct = cfg.acts[L - 1]
  let rule: Backward['outputRule'] = 'chain'
  if (outAct === 'softmax') {
    if (cfg.loss === 'cce') { rule = 'softmax+cce'; delta[L] = yh.map((v, k) => v - y[k]) }
    else { const J = softmaxJacobian(yh); delta[L] = yh.map((_, r) => J.reduce((s, row, k) => s + row[r] * dA[L][k], 0)) }
  } else {
    fprime[L] = t.z[L].map((zv) => actDeriv(outAct, zv))
    if (outAct === 'sigmoid' && cfg.loss === 'bce') { rule = 'sigmoid+bce'; delta[L] = yh.map((v, k) => (v - y[k]) / K) }
    else delta[L] = dA[L].map((g, k) => g * fprime[L][k])
  }
  for (let l = L - 1; l >= 0; l--) {
    const W = p.W[l]
    dA[l] = new Array(cfg.sizes[l]).fill(0)
    W.forEach((row, j) => row.forEach((w, i) => (dA[l][i] += w * delta[l + 1][j])))
    if (l > 0) {
      fprime[l] = t.z[l].map((zv) => actDeriv(cfg.acts[l - 1], zv))
      delta[l] = dA[l].map((g, i) => g * fprime[l][i])
    }
  }
  const dW = p.W.map((Wl, l) => Wl.map((row, j) => row.map((_, i) => delta[l + 1][j] * t.a[l][i])))
  const db = p.b.map((_, l) => delta[l + 1].slice())
  return { dA, fprime, delta, dW, db, outputRule: rule }
}
export const gradOf = (g: Pick<Backward, 'dW' | 'db'>, r: ParamRef) => (r.kind === 'w' ? g.dW[r.layer - 1][r.j][r.i] : g.db[r.layer - 1][r.j])

/* ------------------------------------------------------------------ batches */
export interface Sample { x: number[]; y: number[]; label?: number }
export interface BatchResult { loss: number; dW: number[][][]; db: number[][]; perSample: { loss: number; yhat: number[] }[] }
/** Mean loss and mean gradients over a batch (gradient accumulation over samples). */
export function batchGrad(cfg: NetConfig, p: Params, data: Sample[]): BatchResult {
  const dW = p.W.map((L) => L.map((r) => r.map(() => 0))), db = p.b.map((v) => v.map(() => 0))
  let loss = 0
  const perSample: BatchResult['perSample'] = []
  for (const s of data) {
    const t = forward(cfg, p, s.x), g = backward(cfg, p, t, s.y), l = lossValue(cfg, t, s.y)
    loss += l; perSample.push({ loss: l, yhat: output(t) })
    g.dW.forEach((L, li) => L.forEach((r, j) => r.forEach((v, i) => (dW[li][j][i] += v))))
    g.db.forEach((v, li) => v.forEach((x, j) => (db[li][j] += x)))
  }
  const n = data.length || 1
  return { loss: loss / n, dW: dW.map((L) => L.map((r) => r.map((v) => v / n))), db: db.map((v) => v.map((x) => x / n)), perSample }
}
export const datasetLoss = (cfg: NetConfig, p: Params, data: Sample[]) => data.reduce((s, d) => s + lossValue(cfg, forward(cfg, p, d.x), d.y), 0) / (data.length || 1)
export function accuracy(cfg: NetConfig, p: Params, data: Sample[]) {
  if (!data.length || data.every((d) => d.label === undefined)) return NaN
  return data.filter((d) => predictLabel(output(forward(cfg, p, d.x))) === d.label).length / data.length
}
export const predictLabel = (yh: number[]) => (yh.length === 1 ? (yh[0] >= 0.5 ? 1 : 0) : yh.indexOf(Math.max(...yh)))

/* ------------------------------------------------------------------ gradient check */
export interface GradCheckRow { ref: ParamRef; analytic: number; numeric: number; absErr: number; relErr: number }
/** Central differences (L(θ+ε) − L(θ−ε)) / 2ε for every parameter, against backprop. */
export function gradientCheck(cfg: NetConfig, p: Params, data: Sample[], eps = 1e-5, refs = allParamRefs(cfg.sizes)): GradCheckRow[] {
  const g = batchGrad(cfg, p, data)
  return refs.map((ref) => {
    const v = getParam(p, ref)
    const lp = datasetLoss(cfg, setParam(p, ref, v + eps), data), lm = datasetLoss(cfg, setParam(p, ref, v - eps), data)
    const numeric = (lp - lm) / (2 * eps), analytic = gradOf(g, ref)
    const absErr = Math.abs(analytic - numeric)
    return { ref, analytic, numeric, absErr, relErr: absErr / Math.max(1e-12, Math.abs(analytic) + Math.abs(numeric)) }
  })
}

/* ------------------------------------------------------------------ chain-rule paths */
export interface Factor { label: string; value: number; kind: 'local-input' | 'activation' | 'weight' | 'softmax' | 'loss' | 'one' }
export interface Path { nodes: { layer: number; j: number }[]; factors: Factor[]; product: number }
/**
 * Every path from a parameter (or a neuron's activation) to the loss, as products of local derivatives.
 * The sum over all paths equals the backpropagated gradient (tested). Paths are listed for one sample.
 */
export function chainPaths(cfg: NetConfig, p: Params, t: Trace, y: number[], start: ParamRef | { kind: 'a'; layer: number; j: number }, limit = 400): { paths: Path[]; total: number; truncated: boolean } {
  const L = cfg.sizes.length - 1
  const dLda = lossGrad(cfg, t.a[L], y)
  const out: Path[] = []
  let truncated = false
  const first: Factor[] = start.kind === 'w' ? [{ label: `∂z${sub(start.j + 1)}⁽${sup(start.layer)}⁾/∂w = a${sub(start.i + 1)}⁽${sup(start.layer - 1)}⁾`, value: t.a[start.layer - 1][start.i], kind: 'local-input' }]
    : start.kind === 'b' ? [{ label: `∂z${sub(start.j + 1)}⁽${sup(start.layer)}⁾/∂b = 1`, value: 1, kind: 'one' }] : []
  // walk forward from neuron (layer, j). `atZ` = we are at z_j (need the activation factor next)
  const walk = (layer: number, j: number, atZ: boolean, factors: Factor[], nodes: Path['nodes']) => {
    if (out.length >= limit) { truncated = true; return }
    const act = cfg.acts[layer - 1]
    if (atZ) {
      if (act === 'softmax') {
        // z_j influences every output a_k through the Jacobian; at the output layer the next factor is ∂L/∂a_k
        const J = softmaxJacobian(t.a[layer])
        for (let k = 0; k < cfg.sizes[layer]; k++) {
          const f2 = [...factors, { label: `∂a${sub(k + 1)}/∂z${sub(j + 1)} (softmax)`, value: J[k][j], kind: 'softmax' as const }]
          if (layer === L) { const fl = [...f2, { label: `∂L/∂a${sub(k + 1)}⁽${sup(L)}⁾`, value: dLda[k], kind: 'loss' as const }]; out.push({ nodes: [...nodes, { layer, j: k }], factors: fl, product: fl.reduce((s, f) => s * f.value, 1) }) }
          else walkFromA(layer, k, f2, [...nodes, { layer, j: k }])
        }
        return
      }
      const fp = actDeriv(act, t.z[layer][j])
      walkFromA(layer, j, [...factors, { label: `${act === 'sigmoid' ? 'σ′' : act === 'tanh' ? 'tanh′' : act === 'relu' ? 'ReLU′' : act === 'leaky' ? 'LReLU′' : 'id′'}(z${sub(j + 1)}⁽${sup(layer)}⁾)`, value: fp, kind: 'activation' }], nodes)
    } else walkFromA(layer, j, factors, nodes)
  }
  const walkFromA = (layer: number, j: number, factors: Factor[], nodes: Path['nodes']) => {
    if (layer === L) { const fl = [...factors, { label: `∂L/∂a${sub(j + 1)}⁽${sup(L)}⁾`, value: dLda[j], kind: 'loss' as const }]; out.push({ nodes, factors: fl, product: fl.reduce((s, f) => s * f.value, 1) }); return }
    for (let k = 0; k < cfg.sizes[layer + 1]; k++) {
      walk(layer + 1, k, true, [...factors, { label: `w${sub(k + 1)}${sub(j + 1)}⁽${sup(layer + 1)}⁾`, value: p.W[layer][k][j], kind: 'weight' }], [...nodes, { layer: layer + 1, j: k }])
    }
  }
  if (start.kind === 'a') walkFromA(start.layer, start.j, [], [{ layer: start.layer, j: start.j }])
  else walk(start.layer, start.j, true, first, [{ layer: start.layer, j: start.j }])
  // matched output pairs: the generic chain (∂L/∂a · ∂a/∂z) is mathematically identical to ŷ − y (tested)
  return { paths: out, total: out.reduce((s, q) => s + q.product, 0), truncated }
}

/* ------------------------------------------------------------------ optimizers */
export interface OptConfig { kind: OptKind; lr: number; momentum: number; beta1: number; beta2: number; eps: number }
export const defaultOpt: OptConfig = { kind: 'sgd', lr: 0.5, momentum: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
export interface OptState { t: number; mW: number[][][]; mb: number[][]; vW: number[][][]; vb: number[][] }
export const zeroOptState = (p: Params): OptState => ({ t: 0, mW: p.W.map((L) => L.map((r) => r.map(() => 0))), mb: p.b.map((v) => v.map(() => 0)), vW: p.W.map((L) => L.map((r) => r.map(() => 0))), vb: p.b.map((v) => v.map(() => 0)) })
export interface UpdateRecord { ref: ParamRef; old: number; grad: number; step: number; next: number; detail: Record<string, number> }

/** One optimizer step. Returns new parameters, new state and a record per parameter (what the inspector shows). */
export function applyUpdate(p: Params, g: Pick<Backward, 'dW' | 'db'>, st: OptState, o: OptConfig): { params: Params; state: OptState; records: UpdateRecord[] } {
  const q = cloneParams(p)
  const s: OptState = { t: st.t + 1, mW: st.mW.map((L) => L.map((r) => r.slice())), mb: st.mb.map((v) => v.slice()), vW: st.vW.map((L) => L.map((r) => r.slice())), vb: st.vb.map((v) => v.slice()) }
  const records: UpdateRecord[] = []
  const one = (ref: ParamRef, old: number, grad: number, m: number, v: number): { next: number; m: number; v: number; detail: Record<string, number> } => {
    if (o.kind === 'sgd') return { next: old - o.lr * grad, m, v, detail: {} }
    if (o.kind === 'momentum') { const vel = o.momentum * m + grad; return { next: old - o.lr * vel, m: vel, v, detail: { velocity: vel } } }
    const m2 = o.beta1 * m + (1 - o.beta1) * grad, v2 = o.beta2 * v + (1 - o.beta2) * grad * grad
    const mh = m2 / (1 - o.beta1 ** s.t), vh = v2 / (1 - o.beta2 ** s.t)
    void ref
    return { next: old - (o.lr * mh) / (Math.sqrt(vh) + o.eps), m: m2, v: v2, detail: { m: m2, v: v2, mHat: mh, vHat: vh } }
  }
  p.W.forEach((L, l) => L.forEach((row, j) => row.forEach((old, i) => {
    const ref: ParamRef = { kind: 'w', layer: l + 1, j, i }
    const r = one(ref, old, g.dW[l][j][i], st.mW[l][j][i], st.vW[l][j][i])
    q.W[l][j][i] = r.next; s.mW[l][j][i] = r.m; s.vW[l][j][i] = r.v
    records.push({ ref, old, grad: g.dW[l][j][i], step: r.next - old, next: r.next, detail: r.detail })
  })))
  p.b.forEach((v, l) => v.forEach((old, j) => {
    const ref: ParamRef = { kind: 'b', layer: l + 1, j }
    const r = one(ref, old, g.db[l][j], st.mb[l][j], st.vb[l][j])
    q.b[l][j] = r.next; s.mb[l][j] = r.m; s.vb[l][j] = r.v
    records.push({ ref, old, grad: g.db[l][j], step: r.next - old, next: r.next, detail: r.detail })
  }))
  return { params: q, state: s, records }
}

/* ------------------------------------------------------------------ datasets */
export type DatasetId = 'worked' | 'xor' | 'and' | 'or' | 'circles' | 'moons' | 'spiral' | 'blobs' | 'random' | 'digits'
export const DATASET_LABEL: Record<DatasetId, string> = {
  worked: 'Worked example (one sample)', xor: 'XOR', and: 'AND', or: 'OR', circles: 'Two circles', moons: 'Two moons', spiral: 'Two spirals', blobs: 'Two blobs (linearly separable)', random: 'Random samples (any input size)', digits: '8×8 handwritten digits (UCI)',
}
export const is2D = (d: DatasetId) => d !== 'random' && d !== 'worked' && d !== 'digits'
/** Class-labelled 2-D data (labels 0/1) or random samples. Targets are shaped to the output layer by `targetsFor`. */
export function makeDataset(id: DatasetId, seed: number, nIn: number, n = 80, nClasses = 2): { x: number[]; label: number }[] {
  const r = rng(seed * 7919 + 101)
  const noise = (s: number) => gauss(r) * s
  if (id === 'worked') return [{ x: nIn === 2 ? [1.0, 0.8] : Array.from({ length: nIn }, (_, i) => +((i % 5) * 0.2 + 0.2).toFixed(1)), label: 1 }]
  if (id === 'xor') return [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].map(([a, b, y]) => ({ x: [a, b], label: y }))
  if (id === 'and') return [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]].map(([a, b, y]) => ({ x: [a, b], label: y }))
  if (id === 'or') return [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]].map(([a, b, y]) => ({ x: [a, b], label: y }))
  if (id === 'digits') throw new Error('digits come from engine/digits.ts')
  if (id === 'random') return Array.from({ length: Math.min(n, 24) }, () => ({ x: Array.from({ length: nIn }, () => +r().toFixed(2)), label: Math.floor(r() * Math.max(2, nClasses)) }))
  return Array.from({ length: n }, (_, k) => {
    const label = k % 2
    if (id === 'circles') { const a = r() * 2 * Math.PI, rad = label ? 0.35 : 0.85; return { x: [rad * Math.cos(a) + noise(0.06), rad * Math.sin(a) + noise(0.06)], label } }
    if (id === 'moons') { const a = r() * Math.PI; return { x: label ? [1 - Math.cos(a) + noise(0.08) - 0.5, 0.5 - Math.sin(a) + noise(0.08) - 0.25] : [Math.cos(a) + noise(0.08) - 0.5, Math.sin(a) + noise(0.08) - 0.25], label } }
    if (id === 'spiral') { const t = (k / n) * 2 + 0.15, a = t * 2.6 * Math.PI + (label ? Math.PI : 0); return { x: [t * 0.45 * Math.cos(a) + noise(0.03), t * 0.45 * Math.sin(a) + noise(0.03)], label } }
    return { x: label ? [0.5 + noise(0.25), 0.5 + noise(0.25)] : [-0.5 + noise(0.25), -0.5 + noise(0.25)], label }
  })
}
/** Shape targets to the output layer: one output → [label]; K outputs → one-hot (label mod K). */
export function targetsFor(points: { x: number[]; label: number }[], nOut: number): Sample[] {
  return points.map((d) => ({ x: d.x, label: nOut === 1 ? d.label : d.label % nOut, y: nOut === 1 ? [d.label] : Array.from({ length: nOut }, (_, k) => (k === d.label % nOut ? 1 : 0)) }))
}

/* ------------------------------------------------------------------ training */
export interface TrainState { params: Params; opt: OptState; epoch: number; iter: number; cursor: number }
export interface TrainStepInfo { batch: number[]; loss: number; grads: Pick<Backward, 'dW' | 'db'>; records: UpdateRecord[] }
/** Deterministic mini-batch order: a seeded shuffle per epoch. */
export function epochOrder(n: number, seed: number, epoch: number) {
  const r = rng(seed * 1000003 + epoch * 7907 + 3)
  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]] }
  return idx
}
export function trainStep(cfg: NetConfig, ts: TrainState, data: Sample[], o: OptConfig, batchSize: number, seed: number): { state: TrainState; info: TrainStepInfo } {
  const n = data.length, bs = Math.max(1, Math.min(batchSize, n))
  const order = epochOrder(n, seed, ts.epoch)
  const batch = order.slice(ts.cursor, ts.cursor + bs)
  const g = batchGrad(cfg, ts.params, batch.map((i) => data[i]))
  const u = applyUpdate(ts.params, g, ts.opt, o)
  let cursor = ts.cursor + bs, epoch = ts.epoch
  if (cursor >= n) { cursor = 0; epoch++ }
  return { state: { params: u.params, opt: u.state, epoch, iter: ts.iter + 1, cursor }, info: { batch, loss: g.loss, grads: g, records: u.records } }
}
/** Train for a number of epochs and return the loss after every step (pure; used by experiments and tests). */
export function trainEpochs(cfg: NetConfig, p: Params, data: Sample[], o: OptConfig, batchSize: number, seed: number, epochs: number) {
  let ts: TrainState = { params: p, opt: zeroOptState(p), epoch: 0, iter: 0, cursor: 0 }
  const losses = [datasetLoss(cfg, p, data)]
  while (ts.epoch < epochs) { ts = trainStep(cfg, ts, data, o, batchSize, seed).state; losses.push(datasetLoss(cfg, ts.params, data)) }
  return { state: ts, losses }
}

/** p(class 1) (one output) or argmax class probabilities over a grid, for decision boundaries. */
export function decisionGrid(cfg: NetConfig, p: Params, xr: [number, number], yr: [number, number], n = 40) {
  const cells: { x: number; y: number; p: number; cls: number }[] = []
  for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) {
    const x = xr[0] + ((ix + 0.5) / n) * (xr[1] - xr[0]), y = yr[0] + ((iy + 0.5) / n) * (yr[1] - yr[0])
    const yh = output(forward(cfg, p, [x, y]))
    cells.push({ x, y, p: yh.length === 1 ? yh[0] : yh[1] ?? 0, cls: predictLabel(yh) })
  }
  return cells
}

/* ------------------------------------------------------------------ the worked example */
/** 2 → 2 → 1, sigmoid everywhere, MSE, x = [1.0, 0.8], y = 1. Hidden neuron 1 gives z = 0.5·1.0 − 0.3·0.8 + 0.1 = 0.36. */
export const WORKED: { cfg: NetConfig; params: Params; x: number[]; y: number[] } = {
  cfg: { sizes: [2, 2, 1], acts: ['sigmoid', 'sigmoid'], loss: 'mse' },
  params: { W: [[[0.5, -0.3], [0.2, 0.4]], [[0.6, -0.4]]], b: [[0.1, -0.1], [0.05]] },
  x: [1.0, 0.8], y: [1],
}

/* ------------------------------------------------------------------ depth diagnostics (vanishing / exploding gradients) */
export interface LayerStat { layer: number; gradNorm: number; deltaRms: number; actMean: number; actStd: number; saturated: number; dead: number }
/**
 * Per-layer statistics over a batch: ‖∂L/∂W⁽ˡ⁾‖_F of the mean gradient, RMS of δ⁽ˡ⁾ over samples and units,
 * activation mean/std, the fraction of units with |f′(z)| < 0.05 (saturated) and of ReLU units with z ≤ 0 (dead).
 */
export function layerStats(cfg: NetConfig, p: Params, data: Sample[]): LayerStat[] {
  const L = cfg.sizes.length - 1
  const g = batchGrad(cfg, p, data)
  const acc = Array.from({ length: L + 1 }, () => ({ d2: 0, n: 0, s: 0, s2: 0, sat: 0, dead: 0, m: 0 }))
  for (const s of data) {
    const t = forward(cfg, p, s.x), b = backward(cfg, p, t, s.y)
    for (let l = 1; l <= L; l++) {
      const A = acc[l], f = cfg.acts[l - 1]
      t.a[l].forEach((v, j) => {
        A.s += v; A.s2 += v * v; A.m++
        A.d2 += b.delta[l][j] ** 2; A.n++
        if (f !== 'softmax') { if (Math.abs(actDeriv(f, t.z[l][j])) < 0.05) A.sat++; if ((f === 'relu') && t.z[l][j] <= 0) A.dead++ }
      })
    }
  }
  return Array.from({ length: L }, (_, k) => {
    const l = k + 1, A = acc[l], mean = A.s / A.m
    const gn = Math.sqrt(g.dW[l - 1].reduce((s, row) => s + row.reduce((u, v) => u + v * v, 0), 0))
    return { layer: l, gradNorm: gn, deltaRms: Math.sqrt(A.d2 / A.n), actMean: mean, actStd: Math.sqrt(Math.max(0, A.s2 / A.m - mean * mean)), saturated: A.sat / A.m, dead: A.dead / A.m }
  })
}
/** Input sensitivity ∂L/∂x for one sample (with a one-hot target for class c this is −∂ log p_c / ∂x under CCE). */
export const inputGrad = (cfg: NetConfig, p: Params, x: number[], y: number[]) => backward(cfg, p, forward(cfg, p, x), y).dA[0]
export function confusion(cfg: NetConfig, p: Params, data: Sample[], K: number) {
  const m = Array.from({ length: K }, () => new Array(K).fill(0))
  const pred: number[] = []
  for (const d of data) { const k = predictLabel(output(forward(cfg, p, d.x))); pred.push(k); m[d.label ?? 0][k]++ }
  return { matrix: m, pred }
}
