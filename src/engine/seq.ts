/**
 * Sequence models for module 09: a vanilla RNN, an LSTM and a one-layer self-attention model,
 * with full forward traces, exact gradients (backpropagation through time) and a cost model.
 *
 * Two sources, labelled differently in the UI:
 *  - 'trained'    weights trained by training/train_seq.py on the recall task (real models)
 *  - 'simulation' seeded random weights with editable scale/bias (for exploring dynamics)
 * The same forward/backward code runs both.
 */
import raw from '../data/seq-models.json'
import { rng } from './decoding'

export type Arch = 'rnn' | 'lstm' | 'attn'
export type Regime = 'mixed' | 'long'
export const ARCH_LABEL: Record<Arch, string> = { rnn: 'Vanilla RNN', lstm: 'LSTM', attn: 'Self-attention' }
export const SYMBOLS = 'abcd'
export const V = 4

type M = number[][]
type Vec = number[]

/* ------------------------------------------------------------- helpers */
const matvec = (W: M, x: Vec): Vec => W.map((r) => r.reduce((s, w, j) => s + w * x[j], 0))
const tmatvec = (W: M, y: Vec): Vec => { const out = new Array(W[0].length).fill(0); W.forEach((r, i) => r.forEach((w, j) => (out[j] += w * y[i]))); return out }
const addv = (...vs: Vec[]): Vec => vs[0].map((_, i) => vs.reduce((s, v) => s + v[i], 0))
const mul = (a: Vec, b: Vec): Vec => a.map((x, i) => x * b[i])
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
export const vnorm = (v: Vec) => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
export function softmax(z: Vec): Vec { const m = Math.max(...z); const e = z.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s) }
export const oneHot = (id: number): Vec => Array.from({ length: V }, (_, i) => (i === id ? 1 : 0))
export const parseSeq = (s: string) => [...s.toLowerCase()].filter((ch) => SYMBOLS.includes(ch)).map((ch) => SYMBOLS.indexOf(ch))
export const seqText = (ids: number[]) => ids.map((i) => SYMBOLS[i]).join('')

export function sinusoid(T: number, d: number): M {
  return Array.from({ length: T }, (_, t) => Array.from({ length: d }, (_, i) => {
    const k = i - (i % 2)
    const w = 1 / 10000 ** (k / d)
    return i % 2 === 0 ? Math.sin(t * w) : Math.cos(t * w)
  }))
}

/* ------------------------------------------------------------- parameters */
export interface RNNParams { arch: 'rnn'; Wx: M; Wh: M; b: Vec; Wy: M; by: Vec }
export interface LSTMParams { arch: 'lstm'; Wx: M; Wh: M; b: Vec; Wy: M; by: Vec }
export interface AttnParams { arch: 'attn'; We: M; Wq: M; Wk: M; Wv: M; Wy: M; by: Vec }
export type Params = RNNParams | LSTMParams | AttnParams
export const hiddenOf = (p: Params) => (p.arch === 'attn' ? p.We.length : p.arch === 'lstm' ? p.Wx.length / 4 : p.Wx.length)

interface RawModel { params: Record<string, number[] | number[][]>; history: { step: number; loss: number }[]; accuracy: Record<string, number>; refs: { seq: string; logits: number[] }[]; nParams: number }
interface RawFile { task: { symbols: string; regimes: Record<Regime, [number, number]>; steps: number; batch: number; lr: number; hidden: number; optimizer: string; evaluation: string }; regimes: Record<Regime, Record<Arch, RawModel>> }
const R = raw as unknown as RawFile
export const TASK = R.task

export interface TrainedModel { params: Params; history: RawModel['history']; accuracy: { T: number; acc: number }[]; refs: RawModel['refs']; nParams: number }
export function trained(regime: Regime, arch: Arch): TrainedModel {
  const m = R.regimes[regime][arch]
  const P = m.params as Record<string, number[][] & number[]>
  const params: Params = arch === 'attn'
    ? { arch, We: P.We, Wq: P.Wq, Wk: P.Wk, Wv: P.Wv, Wy: P.Wy, by: P.by }
    : { arch, Wx: P.Wx, Wh: P.Wh, b: P.b, Wy: P.Wy, by: P.by }
  return { params, history: m.history, accuracy: Object.entries(m.accuracy).map(([T, acc]) => ({ T: +T, acc })).sort((a, b) => a.T - b.T), refs: m.refs, nParams: m.nParams }
}

export interface SimConfig { seed: number; hidden: number; scale: number; forgetBias: number }
export const defaultSim: SimConfig = { seed: 7, hidden: 8, scale: 1, forgetBias: 1 }

/** Seeded Gaussian (Box–Muller on the shared mulberry32 generator). */
function gauss(r: () => number) { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
const randM = (r: () => number, rows: number, cols: number, s: number): M => Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r) * s))

/** Random weights. `scale` multiplies the recurrent / attention matrices (scale 1 ≈ spectral radius 1 for N(0, 1/H)). */
export function simParams(arch: Arch, cfg: SimConfig): Params {
  const r = rng(cfg.seed * 7919 + { rnn: 1, lstm: 2, attn: 3 }[arch])
  const H = cfg.hidden
  if (arch === 'rnn') return { arch, Wx: randM(r, H, V, 0.8), Wh: randM(r, H, H, cfg.scale / Math.sqrt(H)), b: new Array(H).fill(0), Wy: randM(r, V, H, 0.5), by: new Array(V).fill(0) }
  if (arch === 'lstm') {
    const b = new Array(4 * H).fill(0)
    for (let i = H; i < 2 * H; i++) b[i] = cfg.forgetBias
    return { arch, Wx: randM(r, 4 * H, V, 0.8), Wh: randM(r, 4 * H, H, cfg.scale / Math.sqrt(H)), b, Wy: randM(r, V, H, 0.5), by: new Array(V).fill(0) }
  }
  return { arch, We: randM(r, H, V, 0.8), Wq: randM(r, H, H, cfg.scale / Math.sqrt(H)), Wk: randM(r, H, H, cfg.scale / Math.sqrt(H)), Wv: randM(r, H, H, 1 / Math.sqrt(H)), Wy: randM(r, V, H, 0.5), by: new Array(V).fill(0) }
}

/* ------------------------------------------------------------- forward traces */
export interface RNNTrace { arch: 'rnn'; x: Vec[]; a: Vec[]; h: Vec[]; logits: Vec; probs: Vec }
export interface LSTMTrace { arch: 'lstm'; x: Vec[]; z: Vec[]; i: Vec[]; f: Vec[]; g: Vec[]; o: Vec[]; c: Vec[]; tc: Vec[]; h: Vec[]; logits: Vec; probs: Vec }
export interface AttnTrace { arch: 'attn'; x: Vec[]; pos: M; e: Vec[]; q: Vec[]; k: Vec[]; v: Vec[]; scores: M; A: M; o: Vec[]; logits: Vec; probs: Vec }
export type Trace = RNNTrace | LSTMTrace | AttnTrace

export function forward(p: Params, ids: number[]): Trace {
  if (!ids.length) throw new Error('empty sequence')
  const x = ids.map(oneHot)
  const H = hiddenOf(p)
  if (p.arch === 'rnn') {
    const a: Vec[] = [], h: Vec[] = []
    let prev = new Array(H).fill(0)
    for (const xt of x) { const at = addv(matvec(p.Wx, xt), matvec(p.Wh, prev), p.b); const ht = at.map(Math.tanh); a.push(at); h.push(ht); prev = ht }
    const logits = addv(matvec(p.Wy, prev), p.by)
    return { arch: 'rnn', x, a, h, logits, probs: softmax(logits) }
  }
  if (p.arch === 'lstm') {
    const tr: LSTMTrace = { arch: 'lstm', x, z: [], i: [], f: [], g: [], o: [], c: [], tc: [], h: [], logits: [], probs: [] }
    let hp = new Array(H).fill(0), cp = new Array(H).fill(0)
    for (const xt of x) {
      const z = addv(matvec(p.Wx, xt), matvec(p.Wh, hp), p.b)
      const i = z.slice(0, H).map(sigmoid), f = z.slice(H, 2 * H).map(sigmoid), g = z.slice(2 * H, 3 * H).map(Math.tanh), o = z.slice(3 * H).map(sigmoid)
      const c = addv(mul(f, cp), mul(i, g)), tc = c.map(Math.tanh), h = mul(o, tc)
      tr.z.push(z); tr.i.push(i); tr.f.push(f); tr.g.push(g); tr.o.push(o); tr.c.push(c); tr.tc.push(tc); tr.h.push(h)
      hp = h; cp = c
    }
    tr.logits = addv(matvec(p.Wy, hp), p.by); tr.probs = softmax(tr.logits)
    return tr
  }
  const T = ids.length
  const pos = sinusoid(T, H)
  const e = x.map((xt, t) => addv(matvec(p.We, xt), pos[t]))
  const q = e.map((et) => matvec(p.Wq, et)), k = e.map((et) => matvec(p.Wk, et)), v = e.map((et) => matvec(p.Wv, et))
  const sq = Math.sqrt(H)
  // full causal attention: row t attends to positions s ≤ t. Only the last row feeds the prediction.
  const scores = q.map((qt, t) => k.map((ks, s) => (s <= t ? qt.reduce((acc, qi, j) => acc + qi * ks[j], 0) / sq : -Infinity)))
  const A = scores.map((row, t) => { const w = softmax(row.slice(0, t + 1)); return row.map((_, s) => (s <= t ? w[s] : 0)) })
  const o = A.map((row) => Array.from({ length: H }, (_, j) => row.reduce((acc, w, s) => acc + w * v[s][j], 0)))
  const logits = addv(matvec(p.Wy, addv(o[T - 1], e[T - 1])), p.by)
  return { arch: 'attn', x, pos, e, q, k, v, scores, A, o, logits, probs: softmax(logits) }
}

/* ------------------------------------------------------------- gradients */
export interface Grads {
  loss: number
  /** ∂L/∂x_t for every input position (one-hot inputs, so dx[t][id] is gradient × input) */
  dx: Vec[]
  /** ∂L/∂(state at t): h_t for RNN/LSTM, e_t for attention */
  dstate: Vec[]
  /** LSTM only: ∂L/∂c_t */
  dc?: Vec[]
}

/**
 * What the gradient is taken of:
 *  - 'loss':  L = −log p(target), the training signal (≈ 0 once a model is confident)
 *  - 'logit': L = z_target, the sensitivity of the target's score to each input (useful for trained models)
 */
export type Objective = 'loss' | 'logit'
/** Exact gradients by backpropagation (through time for the recurrent models). */
export function backward(p: Params, tr: Trace, target: number, objective: Objective = 'loss'): Grads {
  const dy = objective === 'loss' ? tr.probs.map((pi, j) => pi - (j === target ? 1 : 0)) : tr.probs.map((_, j) => (j === target ? 1 : 0))
  const loss = objective === 'loss' ? -Math.log(Math.max(tr.probs[target], 1e-300)) : tr.logits[target]
  const T = tr.x.length
  if (p.arch === 'rnn' && tr.arch === 'rnn') {
    const dx: Vec[] = new Array(T), dstate: Vec[] = new Array(T)
    let dh = tmatvec(p.Wy, dy)
    for (let t = T - 1; t >= 0; t--) {
      dstate[t] = dh
      const da = dh.map((g, j) => g * (1 - tr.h[t][j] ** 2))
      dx[t] = tmatvec(p.Wx, da)
      dh = tmatvec(p.Wh, da)
    }
    return { loss, dx, dstate }
  }
  if (p.arch === 'lstm' && tr.arch === 'lstm') {
    const H = hiddenOf(p)
    const dx: Vec[] = new Array(T), dstate: Vec[] = new Array(T), dcs: Vec[] = new Array(T)
    let dh = tmatvec(p.Wy, dy), dc = new Array(H).fill(0)
    for (let t = T - 1; t >= 0; t--) {
      dstate[t] = dh
      const { i, f, g, o, tc } = { i: tr.i[t], f: tr.f[t], g: tr.g[t], o: tr.o[t], tc: tr.tc[t] }
      const cPrev = t > 0 ? tr.c[t - 1] : new Array(H).fill(0)
      const dct = dc.map((d, j) => d + dh[j] * o[j] * (1 - tc[j] ** 2))
      dcs[t] = dct
      const dz = [
        ...dct.map((d, j) => d * g[j] * i[j] * (1 - i[j])),
        ...dct.map((d, j) => d * cPrev[j] * f[j] * (1 - f[j])),
        ...dct.map((d, j) => d * i[j] * (1 - g[j] ** 2)),
        ...dh.map((d, j) => d * tc[j] * o[j] * (1 - o[j])),
      ]
      dx[t] = tmatvec(p.Wx, dz)
      dh = tmatvec(p.Wh, dz)
      dc = dct.map((d, j) => d * f[j])
    }
    return { loss, dx, dstate, dc: dcs }
  }
  if (p.arch === 'attn' && tr.arch === 'attn') {
    const H = hiddenOf(p), sq = Math.sqrt(H), L = T - 1
    const de: Vec[] = tr.e.map(() => new Array(H).fill(0))
    const du = tmatvec(p.Wy, dy) // u = o_L + e_L
    du.forEach((g, j) => (de[L][j] += g))
    const a = tr.A[L].slice(0, T)
    const da = tr.v.map((vs) => vs.reduce((s, x, j) => s + x * du[j], 0))
    const mean = a.reduce((s, w, t) => s + w * da[t], 0)
    const ds = a.map((w, t) => w * (da[t] - mean))
    const dq = new Array(H).fill(0)
    for (let s = 0; s < T; s++) {
      const dv = du.map((g) => g * a[s])
      const dk = tr.q[L].map((qj) => (ds[s] * qj) / sq)
      tr.k[s].forEach((kj, j) => (dq[j] += (ds[s] * kj) / sq))
      addv(tmatvec(p.Wv, dv), tmatvec(p.Wk, dk)).forEach((g, j) => (de[s][j] += g))
    }
    tmatvec(p.Wq, dq).forEach((g, j) => (de[L][j] += g))
    return { loss, dx: de.map((d) => tmatvec(p.We, d)), dstate: de }
  }
  throw new Error('parameter / trace mismatch')
}

/**
 * Occlusion attribution: how much does p(target) drop when symbol s is replaced by each other symbol
 * (averaged)? Model-agnostic and exact; contrasts with attention weights and gradients.
 */
/** Spearman rank correlation (ties averaged); used to compare attention with attribution. */
export function rankAgreement(a: number[], b: number[]) {
  const rk = (x: number[]) => { const o = x.map((v, i) => [v, i] as const).sort((p, q) => p[0] - q[0]); const r = new Array(x.length); for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && Math.abs(o[j + 1][0] - o[i][0]) < 1e-12) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2; i = j + 1 } return r }
  const ra = rk(a), rb = rk(b), n = a.length
  const ma = ra.reduce((s, x) => s + x, 0) / n, mb = rb.reduce((s, x) => s + x, 0) / n
  let sab = 0, saa = 0, sbb = 0
  for (let i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - mb) ** 2 }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0
}

/**
 * Do two importance profiles agree? Pearson correlation of the raw values plus whether they pick the same
 * top position. (Rank correlation is avoided here: profiles that are ~0 almost everywhere make ranks pure noise.)
 */
export function profileAgreement(a: number[], b: number[]) {
  const n = a.length
  const ma = a.reduce((s, x) => s + x, 0) / n, mb = b.reduce((s, x) => s + x, 0) / n
  let sab = 0, saa = 0, sbb = 0
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2 }
  const top = (x: number[]) => x.indexOf(Math.max(...x))
  return { pearson: saa > 1e-24 && sbb > 1e-24 ? sab / Math.sqrt(saa * sbb) : 0, sameTop: top(a) === top(b), topA: top(a), topB: top(b) }
}

export function occlusion(p: Params, ids: number[], target: number): number[] {
  const base = forward(p, ids).probs[target]
  return ids.map((id, s) => {
    let drop = 0
    for (let alt = 0; alt < V; alt++) {
      if (alt === id) continue
      const changed = ids.slice(); changed[s] = alt
      drop += base - forward(p, changed).probs[target]
    }
    return drop / (V - 1)
  })
}

/* ------------------------------------------------------------- evaluation */
export function randomSeq(T: number, seed: number) { const r = rng(seed); return Array.from({ length: T }, () => Math.floor(r() * V)) }
/** Accuracy on n seeded random sequences of length T (target = first symbol). */
export function accuracyAt(p: Params, T: number, n = 200, seed = 1) {
  let ok = 0
  for (let k = 0; k < n; k++) { const ids = randomSeq(T, seed * 100003 + T * 1009 + k); const pr = forward(p, ids).probs; if (pr.indexOf(Math.max(...pr)) === ids[0]) ok++ }
  return ok / n
}

/* ------------------------------------------------------------- cost model */
export interface Cost {
  /** multiply-adds for the whole forward pass */
  macs: number
  /** longest chain of operations that must run one after another */
  sequentialSteps: number
  /** number of operations a signal from position 1 passes through to reach the last position */
  pathLength: number
  /** activations stored for backpropagation (floats) */
  activationMemory: number
  /** largest number of positions that can be computed at the same time */
  parallelWidth: number
}
export function cost(arch: Arch, T: number, H: number): Cost {
  const tri = (T * (T + 1)) / 2
  if (arch === 'rnn') return { macs: T * (H * V + H * H) + V * H, sequentialSteps: T + 1, pathLength: T - 1, activationMemory: 2 * T * H, parallelWidth: 1 }
  if (arch === 'lstm') return { macs: T * 4 * H * (V + H) + T * 3 * H + V * H, sequentialSteps: T + 1, pathLength: T - 1, activationMemory: T * (6 * H + H), parallelWidth: 1 }
  return { macs: T * H * V + 3 * T * H * H + 2 * tri * H + V * H, sequentialSteps: 5, pathLength: 1, activationMemory: 4 * T * H + 2 * tri, parallelWidth: T }
}

/** A schedule of operations: which can run at the same time (same stage). Drives the parallelism animation. */
export interface Op { stage: number; label: string; position: number | null }
export function schedule(arch: Arch, T: number): Op[] {
  if (arch !== 'attn') {
    const ops: Op[] = []
    for (let t = 0; t < T; t++) ops.push({ stage: t, label: arch === 'rnn' ? `h${t + 1}` : `c${t + 1},h${t + 1}`, position: t })
    ops.push({ stage: T, label: 'output', position: null })
    return ops
  }
  const ops: Op[] = []
  for (let t = 0; t < T; t++) ops.push({ stage: 0, label: `e${t + 1}`, position: t })
  for (let t = 0; t < T; t++) ops.push({ stage: 1, label: `q,k,v${t + 1}`, position: t })
  for (let t = 0; t < T; t++) ops.push({ stage: 2, label: `scores${t + 1}`, position: t })
  for (let t = 0; t < T; t++) ops.push({ stage: 3, label: `softmax·V${t + 1}`, position: t })
  ops.push({ stage: 4, label: 'output', position: null })
  return ops
}

/* ------------------------------------------------------------- challenges */
export type QKind = 'rnn-step' | 'lstm-cell' | 'sigmoid' | 'attention' | 'gradient' | 'steps' | 'trained'
export interface SeqQuestion { id: string; kind: QKind; prompt: string; answer: number; options?: string[]; explain: string; given?: string }

/** Seeded, hand-computable questions. Numbers are rounded to 1 decimal so the arithmetic stays doable on paper. */
export function makeSeqQuestions(seed: number): SeqQuestion[] {
  const r = rng(seed)
  const pick = (lo: number, hi: number) => Math.round((lo + r() * (hi - lo)) * 10) / 10
  const qs: SeqQuestion[] = []
  { // scalar RNN step
    const w = pick(0.5, 1.5), u = pick(-1, 1), b = pick(-0.5, 0.5), x = 1, hp = pick(-0.9, 0.9)
    const a = w * x + u * hp + b
    qs.push({ id: 'rnn-step', kind: 'rnn-step', prompt: `Scalar RNN: h_t = tanh(w·x_t + u·h_{t−1} + b) with w = ${w}, u = ${u}, b = ${b}, x_t = ${x}, h_{t−1} = ${hp}. Compute h_t (3 decimals).`, answer: Math.tanh(a),
      explain: `pre-activation = ${w}·${x} + ${u}·${hp} + ${b} = ${a.toFixed(3)}; tanh(${a.toFixed(3)}) = ${Math.tanh(a).toFixed(4)}.` })
  }
  { // LSTM cell update
    const f = pick(0.1, 0.95), i = pick(0.1, 0.95), g = pick(-0.9, 0.9), cp = pick(-1.5, 1.5)
    const c = f * cp + i * g
    qs.push({ id: 'lstm-cell', kind: 'lstm-cell', prompt: `LSTM cell (one unit): forget gate f = ${f}, input gate i = ${i}, candidate g = ${g}, previous cell c_{t−1} = ${cp}. Compute c_t = f·c_{t−1} + i·g (3 decimals).`, answer: c,
      explain: `${f}·${cp} + ${i}·${g} = ${(f * cp).toFixed(3)} + ${(i * g).toFixed(3)} = ${c.toFixed(4)}. The forget gate decides how much of the old cell survives; nothing is multiplied by a weight matrix on this path.` })
  }
  { // sigmoid
    const z = pick(-3, 3)
    qs.push({ id: 'sigmoid', kind: 'sigmoid', prompt: `A gate's pre-activation is ${z}. What is the gate value σ(${z}) (3 decimals)?`, answer: sigmoid(z),
      explain: `σ(z) = 1 / (1 + e^{−z}) = 1 / (1 + ${Math.exp(-z).toFixed(4)}) = ${sigmoid(z).toFixed(4)}. Gates live in (0, 1): 0 blocks, 1 lets everything through.` })
  }
  { // attention softmax + weighted value
    const s = [pick(-1, 2), pick(-1, 2), pick(-1, 2)], v = [pick(-2, 2), pick(-2, 2), pick(-2, 2)]
    const a = softmax(s), out = a.reduce((acc, w, k) => acc + w * v[k], 0)
    qs.push({ id: 'attention', kind: 'attention', prompt: `Attention with three keys: scores (already divided by √d) = [${s.join(', ')}], scalar values = [${v.join(', ')}]. Compute the output Σ softmax(score)·value (3 decimals).`, answer: out,
      explain: `softmax = [${a.map((x) => x.toFixed(3)).join(', ')}] (sums to 1); output = ${a.map((x, k) => `${x.toFixed(3)}·${v[k]}`).join(' + ')} = ${out.toFixed(4)}.` })
  }
  { // gradient through time, linear scalar RNN
    const u = pick(0.5, 0.95), T = 6 + Math.floor(r() * 10)
    qs.push({ id: 'gradient', kind: 'gradient', prompt: `A linear scalar RNN h_t = u·h_{t−1} + x_t has u = ${u}. How large is ∂h_${T}/∂h_1 (4 decimals)?`, answer: u ** (T - 1),
      explain: `Each step multiplies the gradient by u: ∂h_${T}/∂h_1 = u^${T - 1} = ${u}^${T - 1} = ${(u ** (T - 1)).toFixed(5)}. With |u| < 1 it shrinks exponentially (vanishing); with |u| > 1 it would explode.` })
  }
  { // sequential steps
    const T = 8 + Math.floor(r() * 25)
    const opts = ['1', String(T), String(T * T), String(Math.ceil(Math.log2(T)))]
    qs.push({ id: 'steps', kind: 'steps', prompt: `A sequence has ${T} tokens. How many hidden-state updates must an RNN perform one after another (cannot be parallelized) to read it?`, options: opts, answer: 1,
      explain: `h_t needs h_{t−1}, so all ${T} updates are sequential. A self-attention layer computes all ${T} positions at once (a constant number of dependent matrix operations), at the price of ${T}² attention scores.` })
  }
  { // trained models: read the experiment
    const m = trained('long', 'lstm'), a = trained('long', 'rnn')
    const short = m.accuracy.find((x) => x.T === 2)!.acc, rnnAcc = a.accuracy.find((x) => x.T === 20)!.acc
    const opts = ['It reaches near 100% at every length', 'It stays near 25% (chance) at every length', 'It is perfect on short sequences only', 'It is perfect only at length 20']
    qs.push({ id: 'trained', kind: 'trained', prompt: 'Trained models, "long-only" regime (lengths 15–25 in training). What does the vanilla RNN\'s accuracy look like? (Check the Lengths stage.)', options: opts, answer: 1,
      explain: `The RNN scores ${(rnnAcc * 100).toFixed(0)}% at length 20, the same as guessing among 4 symbols: with every training example at least 15 steps long, the gradient reaching the first symbol is too small to learn from. The LSTM learns the long task but scores only ${(short * 100).toFixed(0)}% at length 2, which it never saw.` })
  }
  return qs
}
export const parseNumber = (s: string) => { const t = s.trim().replace(',', '.'); const f = t.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/); return f ? +f[1] / +f[2] : t === '' ? NaN : Number(t) }
export const answerOk = (q: SeqQuestion, given: string) => q.options ? Number(given) === q.answer : Math.abs(parseNumber(given) - q.answer) <= Math.max(0.0015, Math.abs(q.answer) * 0.005)
