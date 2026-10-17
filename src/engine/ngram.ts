/**
 * N-gram language models, computed exactly from a corpus.
 *
 * Conventions (shared by every method so results are comparable):
 * - Each sentence is padded as  <s> <s> w1 … wm </s>.
 * - Every count is taken over PREDICTED positions only (w1 … </s>), so the
 *   context count of h is c(h·) = Σ_w c(h w), never the raw frequency of h.
 * - The prediction vocabulary V is every training word, plus </s> and <unk>.
 *   Test words outside V are mapped to <unk>. <s> is never predicted.
 * Every smoothed method returns a proper distribution over V (tested).
 */
import { rng } from './decoding'

export const BOS = '<s>'
export const EOS = '</s>'
export const UNK = '<unk>'
const SEP = '\u0001'
export const key = (...t: string[]) => t.join(SEP)
export const unkey = (k: string) => k.split(SEP)

export type Order = 1 | 2 | 3
export type Method = 'mle' | 'laplace' | 'interp' | 'backoff' | 'kn'
export interface ModelConfig {
  order: Order
  method: Method
  k: number // add-k
  lambdas: [number, number, number, number] // weights for [uniform, unigram, bigram, trigram]; normalized over the orders in use
  dBackoff: number // absolute discount for backoff, clamped to [0.001, 0.999]
  dKN: number // Kneser-Ney discount, clamped to [0.001, 0.999]
}
export const defaultConfig: ModelConfig = {
  order: 2, method: 'mle', k: 1, lambdas: [0.02, 0.18, 0.4, 0.4], dBackoff: 0.5, dKN: 0.75,
}

export const METHOD_LABEL: Record<Method, string> = {
  mle: 'Maximum likelihood', laplace: 'Add-k (Laplace)', interp: 'Linear interpolation', backoff: 'Backoff (absolute discount)', kn: 'Interpolated Kneser-Ney',
}

export function sentencesOf(text: string): string[][] {
  return text.toLowerCase().split(/[.!?;\n]+/).map((s) => s.match(/[a-z0-9']+/g) ?? []).filter((s) => s.length > 0)
}

const inc = (m: Map<string, number>, k: string, by = 1) => m.set(k, (m.get(k) ?? 0) + by)
const addSet = (m: Map<string, Set<string>>, k: string, v: string) => { let s = m.get(k); if (!s) m.set(k, (s = new Set())); s.add(v) }

export class NGramModel {
  readonly sents: string[][]
  readonly vocab: string[]
  readonly vocabSet: Set<string>
  readonly V: number
  readonly N: number // predicted tokens
  readonly uni = new Map<string, number>()
  readonly bi = new Map<string, number>()
  readonly tri = new Map<string, number>()
  /** c(v·) and c(uv·): context counts */
  readonly ctx1 = new Map<string, number>()
  readonly ctx2 = new Map<string, number>()
  /** distinct followers: {w : c(v w) > 0}, {w : c(u v w) > 0} */
  readonly next1 = new Map<string, Set<string>>()
  readonly next2 = new Map<string, Set<string>>()
  /** Kneser-Ney continuation statistics */
  readonly contUni = new Map<string, number>() // N1+(• w) = |{v : c(v w) > 0}|
  readonly contBiTypes: number // N1+(• •) = number of bigram types
  readonly contBi = new Map<string, number>() // N1+(• v w) = |{u : c(u v w) > 0}|
  readonly contBiCtx = new Map<string, number>() // N1+(• v •) = Σ_w N1+(• v w)
  readonly contNext = new Map<string, Set<string>>() // {w : N1+(• v w) > 0}
  readonly uniTypes: number // |{w : c(w) > 0}|
  readonly contUniTypes: number // |{w : N1+(• w) > 0}|

  constructor(corpus: string) {
    this.sents = sentencesOf(corpus)
    const words = new Set<string>()
    for (const s of this.sents) for (const w of s) words.add(w)
    this.vocab = [...[...words].sort(), EOS, UNK]
    this.vocabSet = new Set(this.vocab)
    this.V = this.vocab.length
    let N = 0
    for (const s of this.sents) {
      const t = [BOS, BOS, ...s, EOS]
      for (let i = 2; i < t.length; i++) {
        const [u, v, w] = [t[i - 2], t[i - 1], t[i]]
        N++
        inc(this.uni, w)
        inc(this.bi, key(v, w)); inc(this.ctx1, v); addSet(this.next1, v, w)
        inc(this.tri, key(u, v, w)); inc(this.ctx2, key(u, v)); addSet(this.next2, key(u, v), w)
      }
    }
    this.N = N
    for (const k of this.bi.keys()) inc(this.contUni, unkey(k)[1])
    this.contBiTypes = this.bi.size
    for (const k of this.tri.keys()) {
      const [, v, w] = unkey(k)
      inc(this.contBi, key(v, w)); inc(this.contBiCtx, v); addSet(this.contNext, v, w)
    }
    this.uniTypes = this.uni.size
    this.contUniTypes = this.contUni.size
  }

  mapWord = (w: string) => (this.vocabSet.has(w) && w !== BOS ? w : UNK)
  c1 = (w: string) => this.uni.get(w) ?? 0
  c2 = (v: string, w: string) => this.bi.get(key(v, w)) ?? 0
  c3 = (u: string, v: string, w: string) => this.tri.get(key(u, v, w)) ?? 0

  /** Padded, vocabulary-mapped token sequence for a sentence. */
  pad(words: string[]) { return [BOS, BOS, ...words.map(this.mapWord), EOS] }
}

/* ---------------------------------------------------------------- scorers */

export interface Breakdown {
  p: number
  /** highest order whose counts produced the probability (backoff) */
  usedOrder?: number
  parts: { label: string; value: number }[]
}

export interface Scorer {
  model: NGramModel
  cfg: ModelConfig
  /** P(w | h) where h holds the previous (order-1) tokens, oldest first. Undefined MLE contexts return NaN. */
  p(w: string, h: string[]): number
  explain(w: string, h: string[]): Breakdown
}

export function normLambdas(cfg: ModelConfig): number[] {
  const l = cfg.lambdas.slice(0, cfg.order + 1).map((x) => Math.max(0, x))
  const s = l.reduce((a, b) => a + b, 0) || 1
  return l.map((x) => x / s)
}

export function makeScorer(model: NGramModel, cfg: ModelConfig): Scorer {
  const { V, N } = model
  const n = cfg.order
  const ctxOf = (h: string[]) => h.slice(h.length - (n - 1))
  const alphaCache = new Map<string, number>()

  // ---- maximum likelihood components (NaN when the context was never seen)
  const mle1 = (w: string) => model.c1(w) / N
  const mle2 = (w: string, v: string) => { const c = model.ctx1.get(v) ?? 0; return c ? model.c2(v, w) / c : NaN }
  const mle3 = (w: string, u: string, v: string) => { const c = model.ctx2.get(key(u, v)) ?? 0; return c ? model.c3(u, v, w) / c : NaN }

  // ---- add-k
  const lap = (w: string, h: string[]): Breakdown => {
    const k = cfg.k
    if (n === 1) return { p: (model.c1(w) + k) / (N + k * V), parts: [{ label: 'c(w)', value: model.c1(w) }, { label: 'N', value: N }, { label: 'k·V', value: k * V }] }
    const [cw, ch] = n === 2 ? [model.c2(h[0], w), model.ctx1.get(h[0]) ?? 0] : [model.c3(h[0], h[1], w), model.ctx2.get(key(h[0], h[1])) ?? 0]
    const den = ch + k * V
    return { p: den ? (cw + k) / den : NaN, parts: [{ label: 'c(h,w)', value: cw }, { label: 'c(h·)', value: ch }, { label: 'k·V', value: k * V }] }
  }

  // ---- linear interpolation; an undefined component falls back to the next lower one, uniform always defined
  const interp = (w: string, h: string[]): Breakdown => {
    const lam = normLambdas(cfg)
    const comps: number[] = [1 / V, mle1(w)]
    if (n >= 2) { const x = mle2(w, h[h.length - 1]); comps.push(Number.isNaN(x) ? comps[1] : x) }
    if (n >= 3) { const x = mle3(w, h[0], h[1]); comps.push(Number.isNaN(x) ? comps[2] : x) }
    const names = ['uniform 1/V', 'P_ML(w)', 'P_ML(w|v)', 'P_ML(w|u,v)']
    let p = 0
    const parts: Breakdown['parts'] = []
    comps.forEach((c, i) => { p += lam[i] * c; parts.push({ label: `λ${i}·${names[i]}`, value: lam[i] * c }) })
    return { p, parts }
  }

  // ---- normalized backoff with absolute discounting; unigram floor interpolates with uniform
  // d < 1 keeps every seen n-gram strictly positive; d > 0 leaves mass for unseen ones
  const d = Math.min(0.999, Math.max(0.001, cfg.dBackoff))
  const bo1 = (w: string) => Math.max(model.c1(w) - d, 0) / N + ((d * model.uniTypes) / N) / V
  const bo2 = (w: string, v: string): Breakdown => {
    const c = model.ctx1.get(v) ?? 0
    if (!c) return { p: bo1(w), usedOrder: 1, parts: [{ label: 'context unseen → P_bo(w)', value: bo1(w) }] }
    const cw = model.c2(v, w)
    if (cw > 0) return { p: (cw - d) / c, usedOrder: 2, parts: [{ label: '(c(v,w) − d) / c(v·)', value: (cw - d) / c }] }
    const a = alpha2(v, c)
    return { p: a * bo1(w), usedOrder: 1, parts: [{ label: 'α(v)', value: a }, { label: 'P_bo(w)', value: bo1(w) }] }
  }
  const alpha2 = (v: string, c: number) => {
    const ck = `2${SEP}${v}`
    let a = alphaCache.get(ck)
    if (a === undefined) {
      const seen = model.next1.get(v) ?? new Set()
      let lower = 0
      for (const x of seen) lower += bo1(x)
      a = ((d * seen.size) / c) / (1 - lower)
      alphaCache.set(ck, a)
    }
    return a
  }
  const bo3 = (w: string, u: string, v: string): Breakdown => {
    const c = model.ctx2.get(key(u, v)) ?? 0
    if (!c) { const b = bo2(w, v); return { ...b, parts: [{ label: 'trigram context unseen', value: 0 }, ...b.parts] } }
    const cw = model.c3(u, v, w)
    if (cw > 0) return { p: (cw - d) / c, usedOrder: 3, parts: [{ label: '(c(u,v,w) − d) / c(uv·)', value: (cw - d) / c }] }
    const ck = `3${SEP}${u}${SEP}${v}`
    let a = alphaCache.get(ck)
    if (a === undefined) {
      const seen = model.next2.get(key(u, v)) ?? new Set()
      let lower = 0
      for (const x of seen) lower += bo2(x, v).p
      a = ((d * seen.size) / c) / (1 - lower)
      alphaCache.set(ck, a)
    }
    const b = bo2(w, v)
    return { p: a * b.p, usedOrder: b.usedOrder, parts: [{ label: 'α(u,v)', value: a }, { label: 'P_bo(w|v)', value: b.p }] }
  }

  // ---- interpolated Kneser-Ney. Highest order uses raw counts; lower orders use continuation counts.
  const D = Math.min(0.999, Math.max(0.001, cfg.dKN))
  const kn1 = (w: string, top: boolean) => {
    const [cnt, tot, types] = top ? [model.c1(w), N, model.uniTypes] : [model.contUni.get(w) ?? 0, model.contBiTypes, model.contUniTypes]
    return Math.max(cnt - D, 0) / tot + ((D * types) / tot) / V
  }
  const kn2 = (w: string, v: string, top: boolean): Breakdown => {
    const [cnt, tot, types] = top
      ? [model.c2(v, w), model.ctx1.get(v) ?? 0, model.next1.get(v)?.size ?? 0]
      : [model.contBi.get(key(v, w)) ?? 0, model.contBiCtx.get(v) ?? 0, model.contNext.get(v)?.size ?? 0]
    const lower = kn1(w, false)
    if (!tot) return { p: lower, parts: [{ label: 'context unseen → P_cont(w)', value: lower }] }
    const disc = Math.max(cnt - D, 0) / tot
    const lam = (D * types) / tot
    return { p: disc + lam * lower, parts: [{ label: top ? 'max(c(v,w)−D,0)/c(v·)' : 'max(N₁₊(•v w)−D,0)/N₁₊(•v•)', value: disc }, { label: 'λ(v)', value: lam }, { label: 'P_cont(w)', value: lower }] }
  }
  const kn3 = (w: string, u: string, v: string): Breakdown => {
    const tot = model.ctx2.get(key(u, v)) ?? 0
    const lower = kn2(w, v, false)
    if (!tot) return { p: lower.p, parts: [{ label: 'context unseen → P_KN(w|v)', value: lower.p }] }
    const disc = Math.max(model.c3(u, v, w) - D, 0) / tot
    const lam = (D * (model.next2.get(key(u, v))?.size ?? 0)) / tot
    return { p: disc + lam * lower.p, parts: [{ label: 'max(c(u,v,w)−D,0)/c(uv·)', value: disc }, { label: 'λ(u,v)', value: lam }, { label: 'P_KN(w|v)', value: lower.p }] }
  }

  const explain = (w0: string, hist: string[]): Breakdown => {
    const w = w0 === EOS || w0 === UNK ? w0 : model.mapWord(w0)
    const h = ctxOf(hist.map((x) => (x === BOS ? x : model.mapWord(x))))
    switch (cfg.method) {
      case 'mle': {
        const p = n === 1 ? mle1(w) : n === 2 ? mle2(w, h[0]) : mle3(w, h[0], h[1])
        const cw = n === 1 ? model.c1(w) : n === 2 ? model.c2(h[0], w) : model.c3(h[0], h[1], w)
        const ch = n === 1 ? N : n === 2 ? model.ctx1.get(h[0]) ?? 0 : model.ctx2.get(key(h[0], h[1])) ?? 0
        return { p, parts: [{ label: n === 1 ? 'c(w)' : 'c(h,w)', value: cw }, { label: n === 1 ? 'N' : 'c(h·)', value: ch }] }
      }
      case 'laplace': return lap(w, h)
      case 'interp': return interp(w, h)
      case 'backoff': return n === 1 ? { p: bo1(w), usedOrder: 1, parts: [{ label: 'P_bo(w)', value: bo1(w) }] } : n === 2 ? bo2(w, h[0]) : bo3(w, h[0], h[1])
      case 'kn': return n === 1 ? { p: kn1(w, true), parts: [{ label: 'max(c(w)−D,0)/N + D·T/N·1/V', value: kn1(w, true) }] } : n === 2 ? kn2(w, h[0], true) : kn3(w, h[0], h[1])
    }
  }
  return { model, cfg, explain, p: (w, h) => explain(w, h).p }
}

/** Σ_{w∈V} P(w|h): must be 1 for every smoothed method and every context. */
export function distSum(s: Scorer, h: string[]) {
  let t = 0
  for (const w of s.model.vocab) t += s.p(w, h)
  return t
}

export function distribution(s: Scorer, h: string[]) {
  return s.model.vocab.map((w) => ({ w, p: s.p(w, h) })).sort((a, b) => (b.p || 0) - (a.p || 0) || (a.w < b.w ? -1 : 1))
}

/* --------------------------------------------------------------- scoring */

export interface TokenScore {
  i: number
  word: string // as written
  token: string // after <unk> mapping
  history: string[] // previous order-1 tokens
  p: number
  log2p: number
  breakdown: Breakdown
  seenNgram: boolean
}
export interface SentenceScore {
  tokens: TokenScore[]
  log2prob: number
  prob: number
  crossEntropy: number // bits per token
  perplexity: number
  zeroAt: number[] // indices whose probability is 0 (or undefined)
}

export function scoreSentences(s: Scorer, text: string): SentenceScore {
  const m = s.model
  const n = s.cfg.order
  const rows: TokenScore[] = []
  for (const words of sentencesOf(text)) {
    const t = m.pad(words)
    const raw = [BOS, BOS, ...words, EOS]
    for (let i = 2; i < t.length; i++) {
      const history = t.slice(i - (n - 1), i)
      const bd = s.explain(t[i], t.slice(i - 2, i))
      const p = Number.isNaN(bd.p) ? 0 : bd.p
      const seen = n === 1 ? m.c1(t[i]) > 0 : n === 2 ? m.c2(t[i - 1], t[i]) > 0 : m.c3(t[i - 2], t[i - 1], t[i]) > 0
      rows.push({ i: rows.length, word: raw[i], token: t[i], history: n === 1 ? [] : history, p, log2p: Math.log2(p), breakdown: bd, seenNgram: seen })
    }
  }
  const log2prob = rows.reduce((a, r) => a + r.log2p, 0)
  const H = rows.length ? -log2prob / rows.length : 0
  return { tokens: rows, log2prob, prob: 2 ** log2prob, crossEntropy: H, perplexity: 2 ** H, zeroAt: rows.filter((r) => r.p === 0).map((r) => r.i) }
}

/* ------------------------------------------------------------ generation */

export function generate(s: Scorer, seed: number, maxLen = 20) {
  const r = rng(seed)
  const out: { w: string; p: number }[] = []
  const hist = [BOS, BOS]
  const vocab = s.model.vocab
  for (let step = 0; step < maxLen; step++) {
    const h = hist.slice(-2)
    const ps = vocab.map((w) => { const p = s.p(w, h); return Number.isNaN(p) ? 0 : p })
    const z = ps.reduce((a, b) => a + b, 0)
    if (z <= 0) break // MLE with an unseen context: nothing to sample
    const u = r() * z
    let acc = 0, k = 0
    for (; k < vocab.length - 1; k++) { acc += ps[k]; if (u < acc) break }
    out.push({ w: vocab[k], p: ps[k] / z })
    hist.push(vocab[k])
    if (vocab[k] === EOS) break
  }
  return out
}

/* ------------------------------------------------------- tuning on held-out */

/** Grid search of interpolation weights (step 0.1, uniform weight fixed) minimising held-out perplexity. Deterministic. */
export function tuneLambdas(model: NGramModel, cfg: ModelConfig, heldOut: string) {
  const l0 = Math.max(0.01, cfg.lambdas[0])
  let best = { pp: Infinity, lambdas: cfg.lambdas }
  const steps = 10
  for (let a = 0; a <= steps; a++)
    for (let b = 0; a + b <= steps; b++) {
      const c = steps - a - b
      if (cfg.order < 3 && c !== 0) continue
      if (cfg.order < 2 && b !== 0) continue
      const rest = 1 - l0
      const lambdas: ModelConfig['lambdas'] = [l0, (rest * a) / steps, (rest * b) / steps, (rest * c) / steps]
      const pp = scoreSentences(makeScorer(model, { ...cfg, method: 'interp', lambdas }), heldOut).perplexity
      if (pp < best.pp - 1e-12) best = { pp, lambdas }
    }
  return best
}

/* ----------------------------------------------------------- challenges */

export interface Challenge {
  id: string
  kind: 'count' | 'mle' | 'laplace' | 'types' | 'continuation' | 'perplexity' | 'unseen'
  prompt: string
  answer: number
  explain: string
}

export function parseAnswer(s: string): number {
  const t = s.trim().replace(/,/g, '.')
  const frac = t.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/)
  if (frac) return Number(frac[1]) / Number(frac[2])
  return t === '' ? NaN : Number(t)
}
export const isCorrect = (answer: number, given: number) =>
  Number.isFinite(given) && Math.abs(given - answer) <= Math.max(1e-3, Math.abs(answer) * 0.005)

/** Deterministic question set for a corpus and seed. */
export function makeChallenges(model: NGramModel, seed: number): Challenge[] {
  const r = rng(seed)
  const pick = <T,>(a: T[]) => a[Math.floor(r() * a.length)]
  const show = (t: string) => (t === BOS ? '<s>' : t === EOS ? '</s>' : t)
  const bigrams = [...model.bi.keys()].map(unkey).filter(([v]) => v !== BOS)
  const anyBigrams = [...model.bi.keys()].map(unkey)
  if (!anyBigrams.length) return []
  const out: Challenge[] = []
  const [v1, w1] = pick(bigrams.length ? bigrams : anyBigrams)
  out.push({ id: 'count', kind: 'count', prompt: `How many times does the bigram “${show(v1)} ${show(w1)}” occur in the corpus (with sentence padding)?`, answer: model.c2(v1, w1),
    explain: `Slide a two-word window over every padded sentence and count matches: c(${show(v1)}, ${show(w1)}) = ${model.c2(v1, w1)}.` })
  const [v2, w2] = pick(anyBigrams)
  const c2 = model.c2(v2, w2), cv = model.ctx1.get(v2)!
  out.push({ id: 'mle', kind: 'mle', prompt: `Maximum-likelihood bigram probability P(${show(w2)} | ${show(v2)})? A fraction like 2/3 is fine.`, answer: c2 / cv,
    explain: `P_ML = c(${show(v2)} ${show(w2)}) / c(${show(v2)} ·) = ${c2} / ${cv} = ${(c2 / cv).toFixed(4)}. The denominator counts “${show(v2)}” only where it is followed by something.` })
  const unseenW = model.vocab.filter((w) => w !== UNK && model.c2(v2, w) === 0)
  const w3 = unseenW.length ? pick(unseenW) : w2
  const c3 = model.c2(v2, w3)
  out.push({ id: 'laplace', kind: 'laplace', prompt: `With add-one (Laplace) smoothing, what is P(${show(w3)} | ${show(v2)})? |V| = ${model.V}.`, answer: (c3 + 1) / (cv + model.V),
    explain: `(c + 1) / (c(${show(v2)}·) + |V|) = (${c3} + 1) / (${cv} + ${model.V}) = ${((c3 + 1) / (cv + model.V)).toFixed(5)}.` })
  out.push({ id: 'types', kind: 'types', prompt: 'How many distinct bigram types occur in the padded corpus?', answer: model.bi.size,
    explain: `Count unique (v, w) pairs over predicted positions: ${model.bi.size}. Out of |V_context| × |V| possible pairs, that is sparse.` })
  const contCands = model.vocab.filter((w) => (model.contUni.get(w) ?? 0) > 0)
  const w5 = pick(contCands)
  out.push({ id: 'continuation', kind: 'continuation', prompt: `Kneser-Ney continuation count: after how many distinct words does “${show(w5)}” appear? (N₁₊(• ${show(w5)}))`, answer: model.contUni.get(w5)!,
    explain: `N₁₊(• ${show(w5)}) = |{v : c(v ${show(w5)}) > 0}| = ${model.contUni.get(w5)}, while its raw count is ${model.c1(w5)}. KN uses the first to judge how readily a word appears in new contexts.` })
  const s = model.sents.length ? pick(model.sents) : []
  const short = s.slice(0, Math.min(3, s.length)).join(' ')
  const sc = scoreSentences(makeScorer(model, { ...defaultConfig, order: 2, method: 'laplace', k: 1 }), short)
  out.push({ id: 'perplexity', kind: 'perplexity', prompt: `Perplexity of “${short}” under an add-one bigram model (include </s>; round to 2 decimals)?`, answer: sc.perplexity,
    explain: `Multiply ${sc.tokens.length} token probabilities (${sc.tokens.map((t) => t.p.toFixed(4)).join(' × ')}), take −log₂ of the product divided by ${sc.tokens.length} to get H = ${sc.crossEntropy.toFixed(4)} bits, then PP = 2^H = ${sc.perplexity.toFixed(3)}.` })
  return out
}
