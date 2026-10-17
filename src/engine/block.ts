/**
 * One Transformer block, computed exactly (float64), for module 02.
 *
 * Conventions (row vectors, like PyTorch's nn.Linear applied to rows):
 *  - X ∈ R^{T×d}: one row per token. Q = X_n W_Q + b_Q with W_Q ∈ R^{d×d}; head h uses columns h·d_h … (h+1)·d_h − 1.
 *  - Attention for head h: A = softmax(Q_h K_hᵀ / √d_h + M), O_h = A V_h. M is 0 where allowed and −∞ where masked.
 *  - MLP(U) = φ(U W₁ + b₁) W₂ + b₂ with W₁ ∈ R^{d×d_ff}, W₂ ∈ R^{d_ff×d}. SwiGLU: (SiLU(U W₁ + b₁) ⊙ (U W₃ + b₃)) W₂ + b₂.
 *  - Pre-LN (GPT):  H = X + α·Attn(LN₁(X)),  Y = H + β·MLP(LN₂(H)).
 *    Post-LN (BERT): H = LN₁(X + α·Attn(X)), Y = LN₂(H + β·MLP(H)).
 * α, β, head and hidden-unit switches are interventions for experiments (defaults: 1, 1, all on).
 * Every number the UI shows comes from `runBlock`; the visualization never computes values itself.
 */
import { rng } from './decoding'

export type Matrix = number[][]
export type Arch = 'pre' | 'post'
export type MlpAct = 'relu' | 'gelu' | 'silu' | 'swiglu'
export type Precision = 'fp64' | 'fp32' | 'fp16' | 'bf16'

export interface BlockConfig { T: number; d: number; H: number; dff: number; arch: Arch; causal: boolean; act: MlpAct; eps: number }
export interface LNParams { g: number[]; b: number[] }
export interface BlockParams {
  ln1: LNParams; ln2: LNParams
  Wq: Matrix; Wk: Matrix; Wv: Matrix; bq: number[]; bk: number[]; bv: number[]
  Wo: Matrix; bo: number[]
  W1: Matrix; b1: number[]; W2: Matrix; b2: number[]
  /** gate projection, used only by SwiGLU */
  W3: Matrix; b3: number[]
}
export interface Interventions { heads: boolean[]; attnScale: number; mlpScale: number; units: boolean[]; precision: Precision }
export const defaultInterventions = (cfg: Pick<BlockConfig, 'H' | 'dff'>): Interventions => ({ heads: new Array(cfg.H).fill(true), attnScale: 1, mlpScale: 1, units: new Array(cfg.dff).fill(true), precision: 'fp64' })

/* ------------------------------------------------------------------ small linear algebra */
export const zeros = (r: number, c: number): Matrix => Array.from({ length: r }, () => new Array(c).fill(0))
export const shapeOf = (m: Matrix): [number, number] => [m.length, m[0]?.length ?? 0]
export const transpose = (m: Matrix): Matrix => (m[0] ?? []).map((_, j) => m.map((r) => r[j]))
export function matmul(a: Matrix, b: Matrix): Matrix {
  if ((a[0]?.length ?? 0) !== b.length) throw new Error(`matmul shape mismatch [${shapeOf(a)}] @ [${shapeOf(b)}]`)
  return a.map((r) => b[0].map((_, j) => r.reduce((s, v, k) => s + v * b[k][j], 0)))
}
export const addRow = (m: Matrix, v: number[]) => m.map((r) => r.map((x, j) => x + v[j]))
export const addM = (a: Matrix, b: Matrix) => a.map((r, i) => r.map((x, j) => x + b[i][j]))
export const scaleM = (a: Matrix, s: number) => a.map((r) => r.map((x) => x * s))
export const cols = (m: Matrix, c0: number, n: number) => m.map((r) => r.slice(c0, c0 + n))
export const vdot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
export const vnorm = (a: number[]) => Math.sqrt(vdot(a, a))

/* ------------------------------------------------------------------ numerical precision emulation */
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer)
/** round to IEEE half precision (round-to-nearest-even, with subnormals and overflow to ±∞) */
export function toFp16(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x
  const a = Math.abs(x)
  if (a >= 65520) return Math.sign(x) * Infinity
  const e = Math.max(Math.floor(Math.log2(a)), -14)
  const q = 2 ** (e - 10)
  let m = a / q
  const fl = Math.floor(m), fr = m - fl
  m = fr > 0.5 || (fr === 0.5 && fl % 2 === 1) ? fl + 1 : fl
  return Math.sign(x) * m * q
}
/** round to bfloat16 (8-bit exponent, 7-bit mantissa), round-to-nearest-even via float32 bits */
export function toBf16(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x
  f32[0] = x
  const bits = u32[0], lsb = (bits >>> 16) & 1
  u32[0] = ((bits + 0x7fff + lsb) >>> 16) << 16
  return f32[0]
}
export const rounder = (p: Precision): ((x: number) => number) => (p === 'fp32' ? Math.fround : p === 'fp16' ? toFp16 : p === 'bf16' ? toBf16 : (x) => x)

/* ------------------------------------------------------------------ pieces */
export interface LNTrace { mean: number[]; variance: number[]; centered: Matrix; xhat: Matrix; out: Matrix }
export function layerNorm(x: Matrix, p: LNParams, eps: number, r: (x: number) => number = (v) => v): LNTrace {
  const d = x[0].length
  const mean = x.map((row) => r(row.reduce((s, v) => s + v, 0) / d))
  const centered = x.map((row, i) => row.map((v) => r(v - mean[i])))
  const variance = centered.map((row) => r(row.reduce((s, v) => s + v * v, 0) / d))
  const xhat = centered.map((row, i) => row.map((v) => r(v / Math.sqrt(variance[i] + eps))))
  const out = xhat.map((row) => row.map((v, j) => r(p.g[j] * v + p.b[j])))
  return { mean, variance, centered, xhat, out }
}
/** numerically stable softmax of one row; −∞ entries get exactly 0 */
export function softmaxRow(z: number[]): { max: number; shifted: number[]; exps: number[]; sum: number; p: number[] } {
  const finite = z.filter(Number.isFinite)
  const max = finite.length ? Math.max(...finite) : 0
  const shifted = z.map((v) => (Number.isFinite(v) ? v - max : -Infinity))
  const exps = shifted.map((v) => (Number.isFinite(v) ? Math.exp(v) : 0))
  const sum = exps.reduce((s, v) => s + v, 0)
  return { max, shifted, exps, sum, p: exps.map((e) => (sum > 0 ? e / sum : 0)) }
}
/** Gauss error function to ~1e-13: Taylor series for |x| ≤ 2.5, continued fraction for erfc beyond */
export function erf(x: number): number {
  const z = Math.abs(x)
  if (z <= 2.5) {
    let term = z, sum = z
    for (let n = 1; n < 80; n++) { term *= -z * z / n; const add = term / (2 * n + 1); sum += add; if (Math.abs(add) < 1e-17) break }
    return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum
  }
  // erfc(z) = exp(−z²)/√π · 1/(z + 1/2/(z + 1/(z + 3/2/(z + …)))), evaluated bottom-up
  let f = 0
  for (let k = 60; k >= 1; k--) f = (k / 2) / (z + f)
  const erfc = Math.exp(-z * z) / Math.sqrt(Math.PI) / (z + f)
  return Math.sign(x) * (1 - erfc)
}
export const ACT_FN: Record<Exclude<MlpAct, 'swiglu'>, (z: number) => number> = {
  relu: (z) => Math.max(0, z),
  gelu: (z) => 0.5 * z * (1 + erf(z / Math.SQRT2)),
  silu: (z) => z / (1 + Math.exp(-z)),
}
export const causalMask = (T: number, causal: boolean): Matrix => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (causal && j > i ? -Infinity : 0)))

/* ------------------------------------------------------------------ the block */
export interface HeadTrace {
  q: Matrix; k: Matrix; v: Matrix
  /** raw scores S = q kᵀ */
  S: Matrix
  /** S / √d_h */
  scaled: Matrix
  mask: Matrix
  /** scaled + mask (the softmax input) */
  logits: Matrix
  rows: ReturnType<typeof softmaxRow>[]
  A: Matrix
  O: Matrix
  enabled: boolean
}
export interface BlockTrace {
  cfg: BlockConfig
  X: Matrix
  ln1: LNTrace | null
  /** input to the attention sublayer: LN₁(X) for pre-LN, X for post-LN */
  attnIn: Matrix
  Q: Matrix; K: Matrix; V: Matrix
  heads: HeadTrace[]
  concat: Matrix
  attn: Matrix
  /** X + α·attn (before LN₁ in post-LN) */
  sum1: Matrix
  H: Matrix
  ln2: LNTrace | null
  mlpIn: Matrix
  Z1: Matrix
  /** SwiGLU only: U W₃ + b₃ */
  gate: Matrix | null
  act: Matrix
  mlp: Matrix
  sum2: Matrix
  Y: Matrix
  postLn1: LNTrace | null
  postLn2: LNTrace | null
}

export function validateBlock(cfg: BlockConfig): string | null {
  if (cfg.d % cfg.H !== 0) return `d = ${cfg.d} is not divisible by H = ${cfg.H}`
  if (cfg.T < 1 || cfg.d < 1 || cfg.dff < 1) return 'sizes must be positive'
  return null
}

export function runBlock(cfg: BlockConfig, p: BlockParams, X: Matrix, iv: Interventions = defaultInterventions(cfg)): BlockTrace {
  const err = validateBlock(cfg)
  if (err) throw new Error(err)
  if (X.length !== cfg.T || X[0].length !== cfg.d) throw new Error(`X has shape [${shapeOf(X)}], expected [${cfg.T}, ${cfg.d}]`)
  const r = rounder(iv.precision), R = (m: Matrix) => m.map((row) => row.map(r))
  const dh = cfg.d / cfg.H
  const ln1 = cfg.arch === 'pre' ? layerNorm(X, p.ln1, cfg.eps, r) : null
  const attnIn = ln1 ? ln1.out : X
  const Q = R(addRow(matmul(attnIn, p.Wq), p.bq)), K = R(addRow(matmul(attnIn, p.Wk), p.bk)), V = R(addRow(matmul(attnIn, p.Wv), p.bv))
  const mask = causalMask(cfg.T, cfg.causal)
  const heads: HeadTrace[] = Array.from({ length: cfg.H }, (_, h) => {
    const q = cols(Q, h * dh, dh), k = cols(K, h * dh, dh), v = cols(V, h * dh, dh)
    const S = R(matmul(q, transpose(k)))
    const scaled = R(scaleM(S, 1 / Math.sqrt(dh)))
    const logits = scaled.map((row, i) => row.map((s, j) => s + mask[i][j]))
    const rows = logits.map(softmaxRow)
    const A = R(rows.map((x) => x.p))
    const O = R(matmul(A, v))
    return { q, k, v, S, scaled, mask, logits, rows, A, O, enabled: iv.heads[h] !== false }
  })
  const concat = Array.from({ length: cfg.T }, (_, i) => heads.flatMap((hd) => (hd.enabled ? hd.O[i] : new Array(dh).fill(0))))
  const attn = R(addRow(matmul(concat, p.Wo), p.bo))
  const sum1 = R(addM(X, scaleM(attn, iv.attnScale)))
  const postLn1 = cfg.arch === 'post' ? layerNorm(sum1, p.ln1, cfg.eps, r) : null
  const H = postLn1 ? postLn1.out : sum1
  const ln2 = cfg.arch === 'pre' ? layerNorm(H, p.ln2, cfg.eps, r) : null
  const mlpIn = ln2 ? ln2.out : H
  const Z1 = R(addRow(matmul(mlpIn, p.W1), p.b1))
  let gate: Matrix | null = null, act: Matrix
  if (cfg.act === 'swiglu') { gate = R(addRow(matmul(mlpIn, p.W3), p.b3)); act = R(Z1.map((row, i) => row.map((z, u) => (iv.units[u] === false ? 0 : ACT_FN.silu(z) * gate![i][u])))) }
  else { const f = ACT_FN[cfg.act]; act = R(Z1.map((row) => row.map((z, u) => (iv.units[u] === false ? 0 : f(z))))) }
  const mlp = R(addRow(matmul(act, p.W2), p.b2))
  const sum2 = R(addM(H, scaleM(mlp, iv.mlpScale)))
  const postLn2 = cfg.arch === 'post' ? layerNorm(sum2, p.ln2, cfg.eps, r) : null
  const Y = postLn2 ? postLn2.out : sum2
  return { cfg, X, ln1, attnIn, Q, K, V, heads, concat, attn, sum1, H, ln2, mlpIn, Z1, gate, act, mlp, sum2, Y, postLn1, postLn2 }
}

/** every intermediate tensor with its shape, in execution order (the tensor shape tracker) */
export function shapeTable(t: BlockTrace): { name: string; tex: string; shape: [number, number]; note: string }[] {
  const { T, d, H, dff } = t.cfg, dh = d / H
  const pre = t.cfg.arch === 'pre'
  return [
    { name: 'X', tex: 'X', shape: [T, d], note: 'input hidden states' },
    ...(pre ? [{ name: 'LN₁(X)', tex: '\\mathrm{LN}_1(X)', shape: [T, d] as [number, number], note: 'normalized per token' }] : []),
    { name: 'W_Q, W_K, W_V', tex: 'W_Q, W_K, W_V', shape: [d, d], note: `all heads; head h uses ${dh} columns` },
    { name: 'Q, K, V', tex: 'Q, K, V', shape: [T, d], note: `split into ${H} heads of [${T}, ${dh}]` },
    { name: 'Q_h, K_h, V_h', tex: 'Q_h, K_h, V_h', shape: [T, dh], note: 'per head' },
    { name: 'S_h = Q_h K_hᵀ', tex: 'Q_h K_h^\\top', shape: [T, T], note: 'one score per (query, key) pair' },
    { name: 'A_h', tex: 'A_h', shape: [T, T], note: 'rows sum to 1' },
    { name: 'O_h = A_h V_h', tex: 'A_h V_h', shape: [T, dh], note: 'per head' },
    { name: 'Concat', tex: '\\mathrm{Concat}(O_1,\\dots,O_H)', shape: [T, d], note: `${H} × ${dh} columns` },
    { name: 'W_O', tex: 'W_O', shape: [d, d], note: 'output projection' },
    { name: 'Attn', tex: '\\mathrm{Attn}', shape: [T, d], note: 'attention sublayer output' },
    { name: 'H', tex: 'H', shape: [T, d], note: pre ? 'X + Attn' : 'LN₁(X + Attn)' },
    { name: 'W₁', tex: 'W_1', shape: [d, dff], note: 'expand' },
    { name: 'φ(Z₁)', tex: '\\varphi(Z_1)', shape: [T, dff], note: 'hidden features' },
    { name: 'W₂', tex: 'W_2', shape: [dff, d], note: 'project back' },
    { name: 'MLP', tex: '\\mathrm{MLP}', shape: [T, d], note: 'per token, shared weights' },
    { name: 'Y', tex: 'Y', shape: [T, d], note: pre ? 'H + MLP' : 'LN₂(H + MLP)' },
  ]
}

/* ------------------------------------------------------------------ deterministic toy parameters */
const r1 = (r: () => number, s = 1) => Math.round((r() * 2 - 1) * s * 10) / 10
/** one-decimal weights (readable by hand), seeded; γ = 1, β = 0 */
export function toyParams(cfg: Pick<BlockConfig, 'd' | 'dff'>, seed: number): BlockParams {
  const r = rng(seed * 2246822519 % 4294967296 + 3)
  const M = (a: number, b: number, s: number) => Array.from({ length: a }, () => Array.from({ length: b }, () => r1(r, s)))
  const v = (n: number, s: number) => Array.from({ length: n }, () => r1(r, s))
  const { d, dff } = cfg
  return {
    ln1: { g: new Array(d).fill(1), b: new Array(d).fill(0) }, ln2: { g: new Array(d).fill(1), b: new Array(d).fill(0) },
    Wq: M(d, d, 1), Wk: M(d, d, 1), Wv: M(d, d, 1), bq: new Array(d).fill(0), bk: new Array(d).fill(0), bv: new Array(d).fill(0),
    Wo: M(d, d, 0.8), bo: new Array(d).fill(0),
    W1: M(d, dff, 0.8), b1: v(dff, 0.2), W2: M(dff, d, 0.6), b2: new Array(d).fill(0),
    W3: M(d, dff, 0.8), b3: new Array(dff).fill(0),
  }
}
/** upstream (outside the block): toy token embeddings and positional vectors, one decimal; X = E[token] + P[position] */
export function toyEmbeddings(tokens: string[], d: number, seed: number): { E: Matrix; P: Matrix; X: Matrix } {
  const E = tokens.map((tok) => { const r = rng(hashStr(tok.toLowerCase()) ^ (seed * 7919)); return Array.from({ length: d }, () => r1(r, 1)) })
  const r = rng(seed * 31337 + 1)
  const P = tokens.map(() => Array.from({ length: d }, () => r1(r, 0.3)))
  return { E, P, X: E.map((row, i) => row.map((v, j) => Math.round((v + P[i][j]) * 10) / 10)) }
}
export function hashStr(s: string) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) } return h >>> 0 }

/* ------------------------------------------------------------------ stacks */
export interface StackTrace { inputs: Matrix[]; traces: (BlockTrace | null)[]; outputs: Matrix[] }
/** run N blocks in sequence; a disabled block passes its input through unchanged */
export function runStack(cfg: BlockConfig, params: BlockParams[], X: Matrix, enabled: boolean[] = params.map(() => true)): StackTrace {
  const inputs: Matrix[] = [], traces: (BlockTrace | null)[] = [], outputs: Matrix[] = []
  let x = X
  params.forEach((p, n) => {
    inputs.push(x)
    if (enabled[n] === false) { traces.push(null); outputs.push(x); return }
    const t = runBlock(cfg, p, x)
    traces.push(t); outputs.push(t.Y); x = t.Y
  })
  return { inputs, traces, outputs }
}

/* ------------------------------------------------------------------ KV cache */
export interface KVStep { t: number; q: number[][]; cacheK: Matrix[]; cacheV: Matrix[]; weights: number[][]; out: number[]; newMacs: number; fullMacs: number }
/**
 * Incremental (autoregressive) causal attention: at step t only the new token's q, k, v are computed;
 * k and v are appended to the cache and the new query attends over the cache. The attention-sublayer
 * output row equals row t of the full-sequence causal computation (tested). MAC counts are for the
 * attention sublayer only.
 */
export function kvCacheRun(cfg: BlockConfig, p: BlockParams, X: Matrix): KVStep[] {
  const dh = cfg.d / cfg.H, steps: KVStep[] = []
  const cacheK: Matrix[] = Array.from({ length: cfg.H }, () => []), cacheV: Matrix[] = Array.from({ length: cfg.H }, () => [])
  for (let t = 0; t < cfg.T; t++) {
    const x = [X[t]]
    const xin = cfg.arch === 'pre' ? layerNorm(x, p.ln1, cfg.eps).out : x
    const q = addRow(matmul(xin, p.Wq), p.bq)[0], k = addRow(matmul(xin, p.Wk), p.bk)[0], v = addRow(matmul(xin, p.Wv), p.bv)[0]
    const qs: number[][] = [], weights: number[][] = [], outs: number[] = []
    for (let h = 0; h < cfg.H; h++) {
      const qh = q.slice(h * dh, (h + 1) * dh)
      cacheK[h].push(k.slice(h * dh, (h + 1) * dh)); cacheV[h].push(v.slice(h * dh, (h + 1) * dh))
      const sm = softmaxRow(cacheK[h].map((kk) => vdot(qh, kk) / Math.sqrt(dh)))
      qs.push(qh); weights.push(sm.p)
      for (let c = 0; c < dh; c++) outs.push(sm.p.reduce((s, w, j) => s + w * cacheV[h][j][c], 0))
    }
    const out = addRow(matmul([outs], p.Wo), p.bo)[0]
    const n = t + 1
    // with the cache: 3 projections of one row + n scores + n weighted sums + W_O, per head totals
    const newMacs = 3 * cfg.d * cfg.d + 2 * n * cfg.d + cfg.d * cfg.d
    // without: recompute the whole prefix of length n
    const fullMacs = n * 3 * cfg.d * cfg.d + 2 * n * n * cfg.d + n * cfg.d * cfg.d
    steps.push({ t, q: qs, cacheK: cacheK.map((m) => m.map((r) => r.slice())), cacheV: cacheV.map((m) => m.map((r) => r.slice())), weights, out, newMacs, fullMacs })
  }
  return steps
}

/* ------------------------------------------------------------------ cost model */
/** multiply-accumulate counts and memory (numbers stored) for one block on a sequence of length T */
export function blockCost(T: number, d: number, H: number, dff: number, swiglu = false) {
  const qkv = 3 * T * d * d, scores = T * T * d, av = T * T * d, proj = T * d * d, mlp = (swiglu ? 3 : 2) * T * d * dff
  return { qkv, scores, av, proj, mlp, total: qkv + scores + av + proj + mlp, attnMatrixEntries: H * T * T, kvCacheEntries: 2 * T * d }
}

/* ------------------------------------------------------------------ attention vs MLP: measured dependency maps */
/** the attention sublayer alone, applied to its input (already normalized in pre-LN) */
export function attnSublayer(cfg: BlockConfig, p: BlockParams, xin: Matrix): Matrix {
  const dh = cfg.d / cfg.H, T = xin.length
  const Q = addRow(matmul(xin, p.Wq), p.bq), K = addRow(matmul(xin, p.Wk), p.bk), V = addRow(matmul(xin, p.Wv), p.bv)
  const mask = causalMask(T, cfg.causal)
  const outs = Array.from({ length: cfg.H }, (_, h) => {
    const q = cols(Q, h * dh, dh), k = cols(K, h * dh, dh), v = cols(V, h * dh, dh)
    const A = matmul(q, transpose(k)).map((row, i) => softmaxRow(row.map((s, j) => s / Math.sqrt(dh) + mask[i][j])).p)
    return matmul(A, v)
  })
  const concat = Array.from({ length: T }, (_, i) => outs.flatMap((o) => o[i]))
  return addRow(matmul(concat, p.Wo), p.bo)
}
/** the MLP sublayer alone (position-wise) */
export function mlpSublayer(cfg: BlockConfig, p: BlockParams, u: Matrix): Matrix {
  const Z = addRow(matmul(u, p.W1), p.b1)
  const act = cfg.act === 'swiglu' ? (() => { const g = addRow(matmul(u, p.W3), p.b3); return Z.map((row, i) => row.map((z, k) => ACT_FN.silu(z) * g[i][k])) })() : Z.map((row) => row.map(ACT_FN[cfg.act as Exclude<MlpAct, 'swiglu'>]))
  return addRow(matmul(act, p.W2), p.b2)
}
/**
 * Perturb the sublayer input of token j (a fixed small vector) and measure how much every output row i moves.
 * dep.attn[i][j] > 0 means output i depends on input j through attention; the MLP map is diagonal.
 */
export function dependencyMap(cfg: BlockConfig, p: BlockParams, xin: Matrix, eps = 1e-4) {
  const T = xin.length, delta = xin[0].map((_, k) => eps * (k % 2 ? -1 : 1))
  const a0 = attnSublayer(cfg, p, xin), m0 = mlpSublayer(cfg, p, xin)
  const attn = zeros(T, T), mlp = zeros(T, T)
  for (let j = 0; j < T; j++) {
    const x2 = xin.map((r, i) => (i === j ? r.map((v, k) => v + delta[k]) : r))
    const a = attnSublayer(cfg, p, x2), m = mlpSublayer(cfg, p, x2)
    for (let i = 0; i < T; i++) { attn[i][j] = vnorm(a[i].map((v, k) => v - a0[i][k])) / eps; mlp[i][j] = vnorm(m[i].map((v, k) => v - m0[i][k])) / eps }
  }
  return { attn, mlp }
}
