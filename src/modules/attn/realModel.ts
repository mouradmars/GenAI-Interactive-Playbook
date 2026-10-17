/**
 * Real-model adapter: attention matrices of Observatory-Nano (the 2-layer, 4-head GPT trained in module 01).
 * The probabilities come from the model's own forward pass (engine/model.ts, tested against PyTorch). As a
 * cross-check, the same weights are recomputed here with this module's attention engine from the model's real
 * per-head q and k: softmax(q kᵀ/√d_k + causal mask).
 */
import { getModel, showToken } from '../../engine/model'
import { row as matRow, type Mat } from '../../engine/tensor'
import { buildMask, entropy, matmul, stableSoftmax, transpose, type Matrix } from '../../engine/attention'

const toM = (m: Mat): Matrix => Array.from({ length: m.rows }, (_, i) => Array.from(matRow(m, i)))

export function realAttention(text: string, layer: number, head: number) {
  const model = getModel()
  const enc = model.encodePrompt(text.trim() || 'the')
  const tr = model.forward(enc.ids)
  const L = Math.max(0, Math.min(layer, tr.layers.length - 1)), H = Math.max(0, Math.min(head, model.meta.nHeads - 1))
  const hd = tr.layers[L].heads[H]
  const q = toM(hd.q), k = toM(hd.k), v = toM(hd.v), A = toM(hd.probs)
  const dk = q[0].length, n = q.length
  const { M } = buildMask(n, n, { causal: true, pad: new Array(n).fill(false), blocked: null })
  const Z = matmul(q, transpose(k)).map((r, i) => r.map((s, j) => s / Math.sqrt(dk) + M[i][j]))
  const recomputed = Z.map((r) => stableSoftmax(r).p)
  let maxDiff = 0
  A.forEach((r, i) => r.forEach((p, j) => { maxDiff = Math.max(maxDiff, Math.abs(p - recomputed[i][j])) }))
  return { meta: model.meta, tokens: tr.tokens.map(showToken), truncated: enc.truncated, layer: L, head: H, dk, q, k, v, A, recomputed, maxDiff, entropy: A.map(entropy) }
}
