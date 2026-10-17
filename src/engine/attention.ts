/**
 * Scaled dot-product and multi-head attention, computed exactly (float64), for module 03.
 *
 * Conventions (row vectors, as in PyTorch's nn.Linear applied to rows):
 *  - X ∈ R^{T×d_model}, one row per position.
 *  - W_Q, W_K, W_V ∈ R^{d_model × h·d_k}. Head h owns the column block h·d_k … (h+1)·d_k − 1, so
 *    W_Q^{(h)} ∈ R^{d_model×d_k} is its own learned matrix: Q_h = X W_Q^{(h)}, K_h = X W_K^{(h)}, V_h = X W_V^{(h)}.
 *  - S_h = Q_h K_hᵀ, S̃_h = S_h / √d_k (or S_h when scaling is switched off), Z_h = S̃_h + M,
 *    A_h = softmax_row(Z_h) over the permitted keys, head_h = A_h V_h.
 *  - MultiHead(X) = Concat(head_1, …, head_h) W_O with W_O ∈ R^{h·d_v × d_model}.
 *  - M is 0 where a key is permitted and −∞ where it is masked (causal, padding or a custom knock-out).
 * A row whose keys are all masked has no valid distribution: this engine returns all-zero weights and a
 * zero output for it and flags it (`fullyMasked`) instead of producing NaN.
 * Every number the UI shows comes from this file; the visual components never compute model values.
 */
import { matmul, transpose, type Matrix } from './block'
import { rng } from './decoding'

export type { Matrix }
export interface AttnParams { Wq: Matrix; Wk: Matrix; Wv: Matrix; Wo: Matrix }
export interface AttnConfig {
  /** number of heads; d_k = d_v = d_model / H */
  H: number
  causal: boolean
  /** divide by √d_k */
  scale: boolean
  /** pad[j] = true: key position j is padding and is masked for every query */
  pad: boolean[]
  /** blocked[i][j] = true: the (query i, key j) pair is knocked out by an experiment */
  blocked: boolean[][] | null
  /** head ablation: a disabled head contributes a zero block to the concatenation */
  headsOn: boolean[]
}
export type MaskReason = 'causal' | 'pad' | 'custom'

export const defaultConfig = (T: number, H = 2): AttnConfig => ({ H, causal: true, scale: true, pad: new Array(T).fill(false), blocked: null, headsOn: new Array(H).fill(true) })

/* ------------------------------------------------------------------ matrix utilities (shared with module 02) */
export const shape = (m: Matrix): [number, number] => [m.length, m[0]?.length ?? 0]
export const colBlock = (m: Matrix, c0: number, n: number): Matrix => m.map((r) => r.slice(c0, c0 + n))
export const concatCols = (parts: Matrix[]): Matrix => parts[0].map((_, i) => parts.flatMap((p) => p[i]))
export const zerosM = (r: number, c: number): Matrix => Array.from({ length: r }, () => new Array(c).fill(0))
export { matmul, transpose }

/** the terms of one dot product: a·b = Σ a_k b_k */
export function dotTerms(a: number[], b: number[]): { products: number[]; sum: number } {
  if (a.length !== b.length) throw new Error(`dot product of vectors of length ${a.length} and ${b.length}`)
  const products = a.map((x, k) => x * b[k])
  return { products, sum: products.reduce((s, v) => s + v, 0) }
}
/** the terms of one output cell of A·B: (AB)_ij = Σ_k A_ik B_kj */
export function matmulCell(A: Matrix, B: Matrix, i: number, j: number) {
  const a = A[i], b = B.map((r) => r[j])
  const { products, sum } = dotTerms(a, b)
  return { a, b, products, sum }
}

/* ------------------------------------------------------------------ softmax */
export interface RowSoftmax {
  /** the softmax input (scaled score + mask) */
  logits: number[]
  allowed: boolean[]
  /** max over permitted (finite) logits; 0 for a fully masked row */
  max: number
  shifted: number[]
  exps: number[]
  sum: number
  p: number[]
  fullyMasked: boolean
}
/** numerically stable softmax over the finite entries of a row; −∞ → probability exactly 0 */
export function stableSoftmax(logits: number[]): RowSoftmax {
  const allowed = logits.map((v) => v !== -Infinity)
  const finite = logits.filter((_, k) => allowed[k])
  if (finite.some((v) => !Number.isFinite(v))) throw new Error('softmax input contains NaN or +∞')
  const fullyMasked = finite.length === 0
  const max = fullyMasked ? 0 : Math.max(...finite)
  const shifted = logits.map((v, k) => (allowed[k] ? v - max : -Infinity))
  const exps = shifted.map((v, k) => (allowed[k] ? Math.exp(v) : 0))
  const sum = exps.reduce((s, v) => s + v, 0)
  return { logits, allowed, max, shifted, exps, sum, p: exps.map((e) => (sum > 0 ? e / sum : 0)), fullyMasked }
}
/** Shannon entropy in nats of a distribution (0·ln 0 = 0) */
export const entropy = (p: number[]) => 0 - p.reduce((s, v) => (v > 0 ? s + v * Math.log(v) : s), 0)
/** Frobenius norm of the softmax Jacobian diag(p) − p pᵀ: how strongly the probabilities respond to the logits */
export function softmaxJacobianNorm(p: number[]): number {
  let s = 0
  for (let i = 0; i < p.length; i++) for (let j = 0; j < p.length; j++) { const v = (i === j ? p[i] : 0) - p[i] * p[j]; s += v * v }
  return Math.sqrt(s)
}

/* ------------------------------------------------------------------ masks */
export function buildMask(Tq: number, Tk: number, cfg: Pick<AttnConfig, 'causal' | 'pad' | 'blocked'>): { M: Matrix; reasons: MaskReason[][][] } {
  const reasons: MaskReason[][][] = Array.from({ length: Tq }, (_, i) => Array.from({ length: Tk }, (_, j) => {
    const r: MaskReason[] = []
    if (cfg.causal && j > i) r.push('causal')
    if (cfg.pad[j]) r.push('pad')
    if (cfg.blocked?.[i]?.[j]) r.push('custom')
    return r
  }))
  return { M: reasons.map((row) => row.map((r) => (r.length ? -Infinity : 0))), reasons }
}

/* ------------------------------------------------------------------ attention */
export interface HeadTrace {
  h: number
  /** this head's own projection matrices, d_model × d_k */
  Wq: Matrix; Wk: Matrix; Wv: Matrix
  Q: Matrix; K: Matrix; V: Matrix
  /** raw scores S = Q Kᵀ */
  S: Matrix
  /** S / √d_k (equal to S when scaling is off) */
  scaled: Matrix
  /** scaled + M: the softmax input */
  logits: Matrix
  rows: RowSoftmax[]
  A: Matrix
  O: Matrix
  entropy: number[]
  enabled: boolean
}
export interface AttnTrace {
  cfg: AttnConfig
  dModel: number; dk: number; Tq: number; Tk: number
  self: boolean
  X: Matrix
  /** source of keys and values (equal to X for self-attention) */
  Xkv: Matrix
  M: Matrix
  reasons: MaskReason[][][]
  heads: HeadTrace[]
  /** Concat(head_1 … head_h), with zero blocks for disabled heads */
  concat: Matrix
  out: Matrix
  fullyMaskedRows: number[]
}

export function validateAttention(X: Matrix, P: AttnParams, cfg: AttnConfig, Xkv: Matrix = X): string[] {
  const err: string[] = []
  const [T, d] = shape(X)
  if (T < 1 || d < 1) err.push('X must have at least one row and one column')
  if (X.some((r) => r.length !== d) || Xkv.some((r) => r.length !== d)) err.push('every input row must have d_model entries')
  if (!Number.isInteger(cfg.H) || cfg.H < 1 || d % cfg.H !== 0) err.push(`h = ${cfg.H} must divide d_model = ${d}`)
  const chk = (name: string, m: Matrix, r: number, c: number) => {
    const [mr, mc] = shape(m)
    if (mr !== r || mc !== c) err.push(`${name} must be ${r}×${c}, got ${mr}×${mc}`)
    if (m.some((row) => row.some((v) => !Number.isFinite(v)))) err.push(`${name} contains a non-finite value`)
  }
  chk('W_Q', P.Wq, d, d); chk('W_K', P.Wk, d, d); chk('W_V', P.Wv, d, d); chk('W_O', P.Wo, d, d)
  if ([...X, ...Xkv].some((row) => row.some((v) => !Number.isFinite(v)))) err.push('the input contains a non-finite value')
  if (cfg.pad.length !== Xkv.length) err.push(`the padding mask must have one entry per key (${Xkv.length})`)
  if (cfg.headsOn.length !== cfg.H) err.push(`the head switches must have ${cfg.H} entries`)
  return err
}

/** One head: projections, scores, scaling, mask, stable softmax, weighted values. */
export function attentionHead(h: number, Xq: Matrix, Xkv: Matrix, P: Pick<AttnParams, 'Wq' | 'Wk' | 'Wv'>, dk: number, M: Matrix, scale: boolean, enabled = true): HeadTrace {
  const Wq = colBlock(P.Wq, h * dk, dk), Wk = colBlock(P.Wk, h * dk, dk), Wv = colBlock(P.Wv, h * dk, dk)
  const Q = matmul(Xq, Wq), K = matmul(Xkv, Wk), V = matmul(Xkv, Wv)
  const S = matmul(Q, transpose(K))
  const c = scale ? 1 / Math.sqrt(dk) : 1
  const scaled = S.map((r) => r.map((v) => v * c))
  const logits = scaled.map((r, i) => r.map((v, j) => v + M[i][j]))
  const rows = logits.map(stableSoftmax)
  const A = rows.map((r) => r.p)
  const O = matmul(A, V)
  return { h, Wq, Wk, Wv, Q, K, V, S, scaled, logits, rows, A, O, entropy: A.map(entropy), enabled }
}

/** Multi-head attention. Self-attention when Xkv is omitted; cross-attention (queries from X, keys/values from Xkv) otherwise. */
export function runAttention(X: Matrix, P: AttnParams, cfg: AttnConfig, Xkv?: Matrix): AttnTrace {
  const self = Xkv === undefined
  const src = Xkv ?? X
  const errors = validateAttention(X, P, cfg, src)
  if (errors.length) throw new Error(errors.join('; '))
  const [Tq, d] = shape(X), Tk = src.length, dk = d / cfg.H
  const { M, reasons } = buildMask(Tq, Tk, { ...cfg, causal: cfg.causal && self })
  const heads = Array.from({ length: cfg.H }, (_, h) => attentionHead(h, X, src, P, dk, M, cfg.scale, cfg.headsOn[h]))
  const concat = concatCols(heads.map((t) => (t.enabled ? t.O : zerosM(Tq, dk))))
  const out = matmul(concat, P.Wo)
  const fullyMaskedRows = heads[0].rows.flatMap((r, i) => (r.fullyMasked ? [i] : []))
  return { cfg, dModel: d, dk, Tq, Tk, self, X, Xkv: src, M, reasons, heads, concat, out, fullyMaskedRows }
}

/** contributions a_ij v_j to output row i of one head: o_i = Σ_j a_ij v_j */
export function aggregationTerms(head: HeadTrace, i: number) {
  const a = head.A[i]
  const contrib = head.V.map((v, j) => v.map((x) => a[j] * x))
  const o = head.V[0].map((_, k) => contrib.reduce((s, c) => s + c[k], 0))
  return { a, contrib, o }
}

/** tensor shapes for the shape table (n = T_q, m = T_k) */
export function shapeList(t: AttnTrace): { name: string; tex: string; shape: [number, number]; note: string }[] {
  const { Tq, Tk, dModel: d, dk, cfg } = t
  return [
    { name: 'X', tex: 'X', shape: [Tq, d], note: 'input hidden states' },
    { name: 'W_Q^(h)', tex: 'W_Q^{(h)}', shape: [d, dk], note: 'per head (W_K, W_V alike)' },
    { name: 'Q_h', tex: 'Q_h', shape: [Tq, dk], note: 'queries of one head' },
    { name: 'K_h', tex: 'K_h', shape: [Tk, dk], note: 'keys of one head' },
    { name: 'V_h', tex: 'V_h', shape: [Tk, dk], note: 'values of one head (d_v = d_k)' },
    { name: 'S_h', tex: 'Q_hK_h^{\\top}', shape: [Tq, Tk], note: 'raw scores' },
    { name: 'M', tex: 'M', shape: [Tq, Tk], note: 'additive mask' },
    { name: 'A_h', tex: 'A_h', shape: [Tq, Tk], note: 'rows sum to 1' },
    { name: 'head_h', tex: 'A_hV_h', shape: [Tq, dk], note: 'one head output' },
    { name: 'H', tex: '\\mathrm{Concat}', shape: [Tq, cfg.H * dk], note: `${cfg.H} heads side by side` },
    { name: 'W_O', tex: 'W_O', shape: [cfg.H * dk, d], note: 'output projection' },
    { name: 'O', tex: 'HW_O', shape: [Tq, d], note: 'multi-head output' },
  ]
}

/* ------------------------------------------------------------------ comparisons */
export const maxAbsDiff = (a: Matrix, b: Matrix) => {
  let m = 0
  a.forEach((r, i) => r.forEach((v, j) => { const w = b[i]?.[j]; if (Number.isFinite(v) && Number.isFinite(w)) m = Math.max(m, Math.abs(v - w)); else if (v !== w) m = Infinity }))
  return m
}
/** which tensors differ between two traces of the same shape (per head and overall) */
export function compareTraces(a: AttnTrace, b: AttnTrace, tol = 1e-12) {
  const heads = a.heads.map((ha, h) => {
    const hb = b.heads[h]
    const d = (k: 'Q' | 'K' | 'V' | 'S' | 'A' | 'O') => (hb ? maxAbsDiff(ha[k], hb[k]) : Infinity)
    const r = { Q: d('Q'), K: d('K'), V: d('V'), S: d('S'), A: d('A'), O: d('O') }
    return { ...r, changed: Object.fromEntries(Object.entries(r).map(([k, v]) => [k, v > tol])) as Record<keyof typeof r, boolean> }
  })
  const concat = a.concat[0].length === b.concat[0]?.length ? maxAbsDiff(a.concat, b.concat) : Infinity
  const out = maxAbsDiff(a.out, b.out)
  return { heads, concat, out, outChanged: out > tol }
}
/** Jensen–Shannon divergence in nats (symmetric, bounded by ln 2) */
export function jsDivergence(p: number[], q: number[]) {
  const m = p.map((v, k) => (v + q[k]) / 2)
  const kl = (a: number[]) => a.reduce((s, v, k) => (v > 0 ? s + v * Math.log(v / m[k]) : s), 0)
  return 0.5 * kl(p) + 0.5 * kl(q)
}
/** similarity of two heads' attention matrices: mean row JS divergence, mean total variation, cosine of the flattened matrices */
export function headSimilarity(A: Matrix, B: Matrix, rows?: number[]) {
  const idx = rows ?? A.map((_, i) => i).filter((i) => A[i].some((v) => v > 0) && B[i].some((v) => v > 0))
  const n = Math.max(1, idx.length)
  const js = idx.reduce((s, i) => s + jsDivergence(A[i], B[i]), 0) / n
  const tv = idx.reduce((s, i) => s + 0.5 * A[i].reduce((t, v, j) => t + Math.abs(v - B[i][j]), 0), 0) / n
  const fa = idx.flatMap((i) => A[i]), fb = idx.flatMap((i) => B[i])
  const dot = fa.reduce((s, v, k) => s + v * fb[k], 0), na = Math.hypot(...fa), nb = Math.hypot(...fb)
  return { js, tv, cosine: na && nb ? dot / (na * nb) : 0 }
}

/* ------------------------------------------------------------------ cost model */
/** multiply–accumulate counts and memory for one multi-head attention layer (batch 1) */
export function attentionCost(T: number, d: number, H: number) {
  const proj = 3 * T * d * d, scores = T * T * d, mix = T * T * d, out = T * d * d
  return { proj, scores, mix, out, total: proj + scores + mix + out, attnEntries: H * T * T, quadraticShare: (scores + mix) / (proj + scores + mix + out) }
}

/* ------------------------------------------------------------------ the scaling experiment */
function normals(seed: number) {
  const r = rng(seed)
  let spare: number | null = null
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s }
    let u = 0, v = 0
    while (u <= 1e-300) u = r()
    v = r()
    const m = Math.sqrt(-2 * Math.log(u))
    spare = m * Math.sin(2 * Math.PI * v)
    return m * Math.cos(2 * Math.PI * v)
  }
}
export interface ScalingResult {
  dk: number
  /** sample mean / variance of q·k for q, k with i.i.d. N(0, 1) entries (theory: 0 and d_k) */
  rawMean: number; rawVar: number; scaledVar: number
  raw: number[]; scaled: number[]
  /** softmax statistics over rows of `nKeys` logits, averaged over the samples */
  entropyRaw: number; entropyScaled: number
  maxPRaw: number; maxPScaled: number
  jacRaw: number; jacScaled: number
  exampleRaw: number[]; exampleScaled: number[]
}
/**
 * Draws `samples` queries and `nKeys` keys per query with i.i.d. standard normal entries (a common modelling
 * assumption, not a property of trained models) and measures the dot products and the resulting softmax rows,
 * with and without the 1/√d_k factor. Deterministic for a given seed.
 */
export function scalingExperiment(dk: number, seed = 1, samples = 400, nKeys = 8): ScalingResult {
  const z = normals(seed * 7919 + dk)
  const raw: number[] = []
  let eR = 0, eS = 0, mR = 0, mS = 0, jR = 0, jS = 0
  let exampleRaw: number[] = [], exampleScaled: number[] = []
  for (let s = 0; s < samples; s++) {
    const q = Array.from({ length: dk }, z)
    const row = Array.from({ length: nKeys }, () => { const k = Array.from({ length: dk }, z); return dotTerms(q, k).sum })
    raw.push(...row)
    const pr = stableSoftmax(row).p, ps = stableSoftmax(row.map((v) => v / Math.sqrt(dk))).p
    eR += entropy(pr); eS += entropy(ps); mR += Math.max(...pr); mS += Math.max(...ps); jR += softmaxJacobianNorm(pr); jS += softmaxJacobianNorm(ps)
    if (s === 0) { exampleRaw = pr; exampleScaled = ps }
  }
  const n = raw.length, mean = raw.reduce((a, b) => a + b, 0) / n
  const rawVar = raw.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)
  const scaled = raw.map((v) => v / Math.sqrt(dk))
  return {
    dk, rawMean: mean, rawVar, scaledVar: rawVar / dk, raw, scaled,
    entropyRaw: eR / samples, entropyScaled: eS / samples, maxPRaw: mR / samples, maxPScaled: mS / samples, jacRaw: jR / samples, jacScaled: jS / samples,
    exampleRaw, exampleScaled,
  }
}

/* ------------------------------------------------------------------ the default educational model */
/** hand-chosen hidden states for the 4-token preset (one row per position) */
export const PRESET_X: Matrix = [
  [1.0, 0.0, 0.5, 0.2],
  [0.2, 1.0, 0.0, 0.4],
  [0.0, 0.3, 1.0, 0.0],
  [0.8, 0.1, 0.4, 1.0],
]
/** hand-chosen projections; head 1 uses columns 1–2, head 2 columns 3–4 */
export const PRESET_PARAMS: AttnParams = {
  Wq: [[2, 0, 1, 0], [0, 2, 0, -1], [1, 0, 2, 0], [0, 1, 0, 2]],
  Wk: [[1, 0, 0, 0.5], [0, 1.5, 1, 0], [-0.5, -0.5, 1, 0], [0.5, 0, -0.5, 1]],
  Wv: [[1, 0, 0, 1], [0, 1, 0.5, 0], [0.5, 0, 1, 0], [0, 0.5, 0, 0.5]],
  Wo: [[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]],
}
/** a second, different hand-chosen input for comparison experiments */
export const PRESET_X_B: Matrix = [
  [0.0, 1.0, 0.0, 0.5],
  [1.0, 0.0, 0.5, 0.0],
  [0.5, 0.5, 0.0, 1.0],
  [0.0, 0.2, 1.0, 0.3],
]
/** rows beyond the preset: seeded one-decimal values in [0, 1] (depend on the position only, never on the token text) */
export function extraRow(pos: number, d: number, seed = 1): number[] {
  const r = rng(seed * 1009 + pos * 9176 + 3)
  return Array.from({ length: d }, () => Math.round(r() * 10) / 10)
}
export function presetInput(T: number, d = 4, seed = 1): Matrix {
  return Array.from({ length: T }, (_, i) => (i < PRESET_X.length && d === 4 && seed === 1 ? PRESET_X[i].slice() : extraRow(i, d, seed)))
}

/** sinusoidal positional encoding (Vaswani et al. 2017): PE[pos, 2i] = sin(pos/10000^{2i/d}), PE[pos, 2i+1] = cos(…) */
export function sinusoidal(T: number, d: number): Matrix {
  return Array.from({ length: T }, (_, pos) => Array.from({ length: d }, (_, k) => {
    const ang = pos / 10000 ** ((2 * Math.floor(k / 2)) / d)
    return k % 2 === 0 ? Math.sin(ang) : Math.cos(ang)
  }))
}
/** toy word-level tokenizer: words and punctuation marks */
export const tokenize = (s: string) => s.match(/[A-Za-z0-9'’]+|[^\sA-Za-z0-9]/g) ?? []
/** a toy embedding table: one seeded one-decimal row per lower-cased word type (illustration only, not trained) */
export function toyEmbedding(word: string, d: number): number[] {
  let h = 2166136261
  for (const c of word.toLowerCase()) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) }
  const r = rng(h >>> 0)
  return Array.from({ length: d }, () => Math.round((r() * 2 - 1) * 10) / 10)
}
