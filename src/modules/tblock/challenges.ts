/**
 * Ten progressive challenges for module 02. Numbers are seeded; every answer is computed with the engine.
 */
import { rng } from '../../engine/decoding'
import { ACT_FN, matmul, runBlock, softmaxRow, vdot, type BlockConfig, type BlockParams, type Matrix } from '../../engine/block'

export type Kind = 'num' | 'vec' | 'set' | 'shape' | 'mc'
export interface Challenge {
  id: string; level: number; title: string; prompt: string; given?: string; kind: Kind
  answer: number | number[] | string; options?: string[]; tol?: number
  solution: string[]; explanation: string; misconception: string
  /** a stage of the microscope where the result can be verified */
  verify?: string
}

const r1 = (r: () => number, s = 1) => Math.round((r() * 2 - 1) * s * 10) / 10
const pn = (v: number) => (v < 0 ? `(${v})` : `${v}`)
const fmt = (v: number, d = 4) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d)
const parseNums = (s: string) => s.replace(/[[\]()]/g, '').split(/[,;\s]+/).filter(Boolean).map((x) => parseFloat(x.replace('−', '-')))

export function checkAnswer(c: Challenge, given: string): boolean {
  const g = given.trim()
  if (!g) return false
  if (c.kind === 'mc') return g === c.answer
  if (c.kind === 'shape') { const m = g.replace(/\s/g, '').match(/^\[?(\d+)[x×,*](\d+)\]?$/i); return !!m && `${m[1]}×${m[2]}` === c.answer }
  if (c.kind === 'set') { const a = new Set(parseNums(g).map(Math.round)), b = c.answer as number[]; return a.size === b.length && b.every((x) => a.has(x)) }
  const tol = (a: number) => c.tol ?? Math.max(0.005, 0.01 * Math.abs(a))
  if (c.kind === 'num') { const v = parseNums(g)[0]; return Number.isFinite(v) && Math.abs(v - (c.answer as number)) <= tol(c.answer as number) }
  const v = parseNums(g), a = c.answer as number[]
  return v.length === a.length && v.every((x, k) => Number.isFinite(x) && Math.abs(x - a[k]) <= tol(a[k]))
}

export function makeChallenges(seed: number, toy: { cfg: BlockConfig; params: BlockParams; X: Matrix; tokens: string[] }): Challenge[] {
  const r = rng(seed * 6151 + 17)
  const pick = <T,>(a: T[]) => a[Math.floor(r() * a.length)]
  // 1 · shapes
  const T1 = pick([4, 6, 8, 10]), d1 = pick([8, 12, 16]), H1 = pick([2, 4].filter((h) => d1 % h === 0)), dh1 = d1 / H1
  // 2 · dot product
  const q2 = [r1(r), r1(r)], k2 = [r1(r), r1(r)], s2 = vdot(q2, k2)
  // 3 · scaling
  const dk3 = pick([4, 16, 64]), s3 = Math.round((r() * 8 - 2) * 10) / 10
  // 4 · causal mask
  const T4 = pick([5, 6, 7]), i4 = 1 + Math.floor(r() * (T4 - 1))
  // 5 · softmax
  const z5 = [r1(r, 2), r1(r, 2), r1(r, 2)], p5 = softmaxRow(z5).p
  // 6 · weighted values
  const a6raw = [r(), r(), r()].map((x) => 0.1 + x), sum6 = a6raw.reduce((a, b) => a + b, 0)
  const a6 = a6raw.map((x) => Math.round((x / sum6) * 10) / 10); a6[2] = Math.round((1 - a6[0] - a6[1]) * 10) / 10
  const V6 = [[r1(r), r1(r)], [r1(r), r1(r)], [r1(r), r1(r)]], o6 = [0, 1].map((c) => a6.reduce((s, w, j) => s + w * V6[j][c], 0))
  // 7 · residual
  const x7 = [r1(r), r1(r), r1(r), r1(r)], at7 = [r1(r, 0.5), r1(r, 0.5), r1(r, 0.5), r1(r, 0.5)], h7 = x7.map((v, k) => v + at7[k])
  // 8 · MLP 2 → 3 → 2 with ReLU
  const u8 = [r1(r), r1(r)], W18 = [[r1(r), r1(r), r1(r)], [r1(r), r1(r), r1(r)]], W28 = [[r1(r), r1(r)], [r1(r), r1(r)], [r1(r), r1(r)]]
  const z8 = matmul([u8], W18)[0], a8 = z8.map(ACT_FN.relu), m8 = matmul([a8], W28)[0]
  // 10 · from the simulator
  const t10 = runBlock(toy.cfg, toy.params, toy.X), i10 = Math.min(toy.cfg.T - 1, 1 + Math.floor(r() * (toy.cfg.T - 1))), f10 = Math.floor(r() * toy.cfg.d)
  const y10 = t10.Y[i10][f10]
  const pre10 = toy.cfg.arch === 'pre'

  return [
    { id: 'shapes', level: 1, title: 'Identify tensor shapes', kind: 'shape', answer: `${T1}×${dh1}`, verify: 'qkv',
      prompt: `A block has T = ${T1} tokens, d = ${d1} and H = ${H1} heads. What is the shape of one head's query matrix Q_h? (answer like ${T1}×?)`,
      solution: [`Q = X_n W_Q has shape [T, d] = [${T1}, ${d1}].`, `d is split evenly among the heads: d_k = d / H = ${d1} / ${H1} = ${dh1}.`, `So Q_h is [${T1}, ${dh1}], and S_h = Q_h K_hᵀ is [${T1}, ${T1}].`],
      explanation: 'Heads do not add parameters to Q, K, V: they partition the d columns.', misconception: 'Thinking each head has full width d (that would make the concatenation H·d wide).' },
    { id: 'dot', level: 2, title: 'Calculate an attention score', kind: 'num', answer: s2, verify: 'dot',
      prompt: `Compute the raw score q · k for q = [${q2.join(', ')}] and k = [${k2.join(', ')}].`,
      solution: [`q · k = ${pn(q2[0])}×${pn(k2[0])} + ${pn(q2[1])}×${pn(k2[1])}`, `= ${pn(+fmt(q2[0] * k2[0]))} + ${pn(+fmt(q2[1] * k2[1]))} = ${fmt(s2)}`],
      explanation: 'Each score multiplies matching coordinates of one query and one key and adds them up.', misconception: 'Multiplying q by k element-wise and stopping there: a score is a single number.' },
    { id: 'scale', level: 3, title: 'Apply scaling', kind: 'num', answer: s3 / Math.sqrt(dk3), verify: 'scale',
      prompt: `A raw score is ${s3} and the head dimension is d_k = ${dk3}. What is the scaled score?`,
      solution: [`√d_k = √${dk3} = ${Math.sqrt(dk3)}`, `${s3} / ${Math.sqrt(dk3)} = ${fmt(s3 / Math.sqrt(dk3))}`],
      explanation: 'Dividing by √d_k keeps the variance of the logits roughly independent of the head size.', misconception: 'Dividing by d_k instead of √d_k.' },
    { id: 'mask', level: 4, title: 'Apply causal masking', kind: 'set', answer: Array.from({ length: i4 + 1 }, (_, k) => k + 1), verify: 'mask',
      prompt: `In a causal decoder with ${T4} tokens, which positions can token ${i4 + 1} attend to? (list them, e.g. 1, 2)`,
      solution: [`Causal mask: M[i, j] = −∞ for j > i.`, `Token ${i4 + 1} keeps positions 1 … ${i4 + 1}; positions ${i4 + 2} … ${T4} get zero weight after softmax.`],
      explanation: 'Each position sees itself and everything before it.', misconception: 'Excluding the token itself: the diagonal is allowed.' },
    { id: 'softmax', level: 5, title: 'Calculate softmax', kind: 'vec', answer: p5, verify: 'softmax',
      prompt: `Convert the scores [${z5.join(', ')}] into attention weights (three numbers).`,
      solution: [`Subtract the max (${Math.max(...z5)}): [${z5.map((z) => fmt(z - Math.max(...z5), 2)).join(', ')}]`, `Exponentiate: [${z5.map((z) => fmt(Math.exp(z - Math.max(...z5)))).join(', ')}], sum = ${fmt(softmaxRow(z5).sum)}`, `Divide: [${p5.map((p) => fmt(p)).join(', ')}]`],
      explanation: 'Softmax is exp(z) / Σ exp(z); subtracting the maximum does not change the result.', misconception: 'Dividing each score by the sum of the scores (that is not softmax and can give negative weights).' },
    { id: 'av', level: 6, title: 'Weighted Value aggregation', kind: 'vec', answer: o6, verify: 'av',
      prompt: `Attention weights [${a6.join(', ')}] over value vectors v₁ = [${V6[0].join(', ')}], v₂ = [${V6[1].join(', ')}], v₃ = [${V6[2].join(', ')}]. Compute the output (two numbers).`,
      solution: [`o = ${a6[0]}·v₁ + ${a6[1]}·v₂ + ${a6[2]}·v₃`, `first coordinate: ${a6.map((w, j) => `${w}×${pn(V6[j][0])}`).join(' + ')} = ${fmt(o6[0])}`, `second coordinate: ${a6.map((w, j) => `${w}×${pn(V6[j][1])}`).join(' + ')} = ${fmt(o6[1])}`],
      explanation: 'The output is an average of the value vectors, weighted by attention.', misconception: 'Weighting the key vectors: keys only decide the weights; values are mixed.' },
    { id: 'residual', level: 7, title: 'Residual addition', kind: 'vec', answer: h7, verify: 'res1',
      prompt: `x = [${x7.join(', ')}] and the attention sublayer outputs [${at7.join(', ')}]. What is the updated hidden state in a pre-LN block?`,
      solution: [`H = x + Attn(LN(x)) is an element-wise sum.`, `[${h7.map((v) => fmt(v, 1)).join(', ')}]`],
      explanation: 'The residual connection adds the update to the unchanged input.', misconception: 'Replacing x by the attention output, or normalizing the sum (that is post-LN).' },
    { id: 'mlp', level: 8, title: 'Feedforward network', kind: 'vec', answer: m8, verify: 'ffn2',
      prompt: `u = [${u8.join(', ')}], W₁ = [${W18.map((row) => `[${row.join(', ')}]`).join(', ')}] (2×3), ReLU, W₂ = [${W28.map((row) => `[${row.join(', ')}]`).join(', ')}] (3×2), no biases. Compute ReLU(u W₁) W₂.`,
      solution: [`u W₁ = [${z8.map((v) => fmt(v, 2)).join(', ')}]`, `ReLU → [${a8.map((v) => fmt(v, 2)).join(', ')}]`, `× W₂ = [${m8.map((v) => fmt(v)).join(', ')}]`],
      explanation: 'The MLP expands, applies a non-linearity, and projects back, for each token separately.', misconception: 'Applying ReLU after W₂ as well, or mixing in other tokens: the MLP is position-wise.' },
    { id: 'arch', level: 9, title: 'Architecture comparison', kind: 'mc', answer: 'The decoder masks future positions; the encoder lets every token attend to every token.', verify: 'mask',
      options: ['The decoder masks future positions; the encoder lets every token attend to every token.', 'The encoder masks future positions; the decoder is bidirectional.', 'Both are causal; they differ only in LayerNorm position.', 'The encoder has no attention, only MLPs.'],
      prompt: 'What is the key difference between attention in a GPT-style decoder block and a BERT-style encoder block?',
      solution: ['Decoder (GPT): causal mask, M[i, j] = −∞ for j > i, needed for left-to-right generation.', 'Encoder (BERT): no causal mask; each output uses the whole sequence.', 'Separately, the presets here also differ in LayerNorm placement: pre-LN (A) vs post-LN (B).'],
      explanation: 'Masking decides which positions can exchange information.', misconception: 'Assuming every Transformer block has the same order and mask.' },
    { id: 'block', level: 10, title: 'Complete Transformer block', kind: 'num', answer: y10, verify: 'res2',
      prompt: `Using the simulator's current block (${pre10 ? 'pre-LN' : 'post-LN'}), predict Y[“${toy.tokens[i10]}”, x${f10 + 1}]. Given: ${pre10 ? `H = ${fmt(t10.H[i10][f10])} and MLP output ${fmt(t10.mlp[i10][f10])}` : `H + MLP = ${fmt(t10.sum2[i10][f10])} before LN₂ (the row mean is ${fmt(t10.postLn2!.mean[i10])}, variance ${fmt(t10.postLn2!.variance[i10])}, γ = ${toy.params.ln2.g[f10]}, β = ${toy.params.ln2.b[f10]})`}.`,
      solution: pre10 ? [`Pre-LN: Y = H + MLP(LN₂(H)).`, `${fmt(t10.H[i10][f10])} + ${fmt(t10.mlp[i10][f10])} = ${fmt(y10)}`] : [`Post-LN: Y = LN₂(H + MLP).`, `(${fmt(t10.sum2[i10][f10])} − ${fmt(t10.postLn2!.mean[i10])}) / √(${fmt(t10.postLn2!.variance[i10])} + ε) × γ + β = ${fmt(y10)}`],
      explanation: 'Verify it on the second residual stage of the microscope.', misconception: 'Forgetting that the residual adds H, not X, in the second addition.' },
  ]
}
