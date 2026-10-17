import { softmax } from './tensor'
import type { NanoGPT } from './model'

export interface DecodeConfig {
  strategy: 'greedy' | 'sample'
  temperature: number
  topK: number // 0 = off
  topP: number // 1 = off
  seed: number
}
export const defaultDecode: DecodeConfig = { strategy: 'sample', temperature: 1, topK: 0, topP: 1, seed: 42 }

/** mulberry32: small deterministic PRNG so classroom runs are reproducible. */
export function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface DecodeStep {
  raw: Float64Array // softmax(logits) at T=1
  tempered: Float64Array // softmax(logits / T)
  final: Float64Array // after top-k / top-p filtering, renormalised
  kept: boolean[]
  chosen: number
  u: number | null // the uniform random number used for sampling
}

export function decodeStep(logits: ArrayLike<number>, cfg: DecodeConfig, u: number): DecodeStep {
  const raw = softmax(logits)
  const T = Math.max(cfg.temperature, 1e-4)
  const tempered = softmax(logits, T)
  const order = Array.from(tempered.keys()).sort((a, b) => tempered[b] - tempered[a])
  const kept = new Array<boolean>(tempered.length).fill(false)
  let cum = 0
  order.forEach((idx, rank) => {
    const inK = cfg.topK <= 0 || rank < cfg.topK
    const inP = cfg.topP >= 1 || cum < cfg.topP // always keeps at least the top token
    if (inK && inP) kept[idx] = true
    cum += tempered[idx]
  })
  const final = new Float64Array(tempered.length)
  let z = 0
  for (let i = 0; i < final.length; i++) if (kept[i]) z += tempered[i]
  for (let i = 0; i < final.length; i++) final[i] = kept[i] ? tempered[i] / z : 0
  let chosen = order[0]
  if (cfg.strategy === 'sample') {
    let c = 0
    for (const idx of order) {
      if (!kept[idx]) continue
      c += final[idx]
      chosen = idx
      if (u < c) break
    }
  }
  return { raw, tempered, final, kept, chosen, u: cfg.strategy === 'sample' ? u : null }
}

/** Autoregressive generation that returns every step for inspection. */
export function generate(model: NanoGPT, ids: number[], nNew: number, cfg: DecodeConfig) {
  const r = rng(cfg.seed)
  const seq = [...ids]
  const steps: (DecodeStep & { context: number[] })[] = []
  for (let s = 0; s < nNew; s++) {
    const ctx = seq.slice(-model.meta.ctx)
    const tr = model.forward(ctx)
    const last = tr.logits.data.subarray((ctx.length - 1) * tr.logits.cols, ctx.length * tr.logits.cols)
    const st = decodeStep(last, cfg, r())
    steps.push({ ...st, context: ctx })
    seq.push(st.chosen)
  }
  return { ids: seq, steps }
}

/** Entropy in bits. */
export const entropy = (p: ArrayLike<number>) => {
  let h = 0
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i])
  return h
}
