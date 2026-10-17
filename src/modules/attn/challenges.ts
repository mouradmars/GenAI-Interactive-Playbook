/**
 * Progressive challenges (beginner → intermediate → advanced). Seeded numbers, plus a few questions about the
 * live lab; every answer is computed by the engine or by the formulas it implements.
 */
import { rng } from '../../engine/decoding'
import { dotTerms, stableSoftmax, type AttnTrace } from '../../engine/attention'
import type { StageId } from './stages'

export type Kind = 'num' | 'vec' | 'mc' | 'set' | 'shape'
export interface Challenge {
  id: string; level: 1 | 2 | 3; title: string; prompt: string; kind: Kind
  answer: number | number[] | string; options?: string[]; tol?: number
  hints: string[]; solution: string[]; misconception: string; verify?: StageId
}

const parseNum = (s: string) => {
  const t = s.trim().replace('−', '-').replace(',', '.').toLowerCase()
  if (['-inf', '-∞', '-infinity'].includes(t)) return -Infinity
  return t === '' ? NaN : Number(t)
}
export function check(c: Challenge, given: string): boolean {
  const close = (v: number, a: number) => (a === -Infinity ? v === -Infinity : Number.isFinite(v) && Math.abs(v - a) <= (c.tol ?? Math.max(0.005, 0.01 * Math.abs(a))))
  switch (c.kind) {
    case 'mc': return given === c.answer
    case 'num': return close(parseNum(given), c.answer as number)
    case 'vec': {
      const parts = given.replace(/[[\]()]/g, '').split(/[;,]|\s+(?=[-−\d.∞i])/).map((x) => x.trim()).filter(Boolean).map((x) => parseNum(x.replace('−', '-')))
      const a = c.answer as number[]
      return parts.length === a.length && parts.every((v, k) => close(v, a[k]))
    }
    case 'set': {
      const t = given.trim().toLowerCase()
      const got = t === '' || t === 'none' || t === '-' ? [] : t.split(/[\s,;]+/).map(Number)
      const a = c.answer as number[]
      return got.every(Number.isInteger) && got.length === a.length && [...got].sort((x, y) => x - y).every((v, k) => v === [...a].sort((x, y) => x - y)[k])
    }
    case 'shape': {
      const m = given.replace(/\s/g, '').match(/^\(?(\d+)[x×,*](\d+)\)?$/i)
      return !!m && `${m[1]}×${m[2]}` === c.answer
    }
  }
}

const f = (v: number, d = 4) => (v === -Infinity ? '−∞' : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d))
const fv = (v: number[], d = 2) => `[${v.map((x) => f(x, d)).join(', ')}]`

export function makeChallenges(seed: number, t: AttnTrace, tokens: string[]): Challenge[] {
  const r = rng(seed * 7717 + 29)
  const int = (a: number, b: number) => a + Math.floor(r() * (b - a + 1))
  const one = () => { const v = Math.round((r() * 2 - 1) * 10) / 10; return v === 0 ? 0.5 : v }
  const h = Math.min(1, t.cfg.H - 1), hd = t.heads[h]
  const qi = Math.min(t.Tq - 1, int(1, t.Tq - 1))

  // level 1
  const hr = int(1, 4), hc = int(1, 4)
  const argmax = hd.A[qi].reduce((b, v, k) => (v > hd.A[qi][b] ? k : b), 0)
  const N = int(4, 6), cq = int(1, N - 1)
  // level 2
  const a3 = [one(), one(), one()], b3 = [one(), one(), one()], dot = dotTerms(a3, b3).sum
  const sRaw = Math.round((r() * 6 - 1) * 100) / 100, dkS = [2, 4, 8, 16, 64][int(0, 4)]
  const mrow = Array.from({ length: 4 }, () => Math.round((r() * 4 - 2) * 10) / 10), mr = int(1, 3)
  const masked = mrow.map((v, k) => (k + 1 > mr ? -Infinity : v))
  const lg = [Math.round((r() * 3 - 1) * 10) / 10, Math.round((r() * 3 - 1) * 10) / 10, Math.round((r() * 3 - 1) * 10) / 10], lk = int(0, 2)
  const sm = stableSoftmax(lg)
  const w1 = int(1, 5) / 10, w2 = int(1, 4) / 10, w = [w1, w2, Math.round((1 - w1 - w2) * 10) / 10]
  const vs = [[int(-2, 2), int(-2, 2)], [int(-2, 2), int(-2, 2)], [int(-2, 2), int(-2, 2)]]
  const agg = [0, 1].map((c) => w.reduce((s, a, j) => s + a * vs[j][c], 0))
  // level 3
  const n = int(3, 12), dm = [8, 12, 16, 64][int(0, 3)], divs = [1, 2, 4, 8].filter((x) => dm % x === 0 && x < dm), hh = divs[int(0, divs.length - 1)]
  const h1 = [one(), one()], h2 = [one(), one()]
  const Hrow = [one(), one(), one(), one()], wcol = [one(), one(), one(), one()], proj = dotTerms(Hrow, wcol).sum
  const bug = (['noscale', 'after', 'column'] as const)[int(0, 2)]
  const bs = [Math.round((r() * 4 + 1) * 10) / 10, Math.round((r() * 4 - 1) * 10) / 10, Math.round((r() * 4 - 2) * 10) / 10]
  const bugText = bug === 'noscale'
    ? `With d_k = 16 the scores of one row are ${fv(bs, 1)}. A student reports the weights ${fv(stableSoftmax(bs).p, 3)}.`
    : bug === 'after'
      ? `Query 1 under a causal mask, scaled scores ${fv(bs, 1)}. A student reports the weights ${fv(stableSoftmax(bs).p.map((p, k) => (k === 0 ? p : 0)), 3)}.`
      : `A 2×2 score matrix [[${f(bs[0], 1)}, ${f(bs[1], 1)}], [${f(bs[2], 1)}, ${f(bs[0], 1)}]]. A student reports weights whose columns each sum to 1: [[${f(stableSoftmax([bs[0], bs[2]]).p[0], 3)}, ${f(stableSoftmax([bs[1], bs[0]]).p[0], 3)}], [${f(stableSoftmax([bs[0], bs[2]]).p[1], 3)}, ${f(stableSoftmax([bs[1], bs[0]]).p[1], 3)}]].`
  const bugAns = { noscale: 'forgot to divide by √d_k', after: 'masked after softmax without renormalizing', column: 'normalized over queries (columns) instead of keys (rows)' }[bug]
  const tk = (k: number) => `${k + 1} (“${tokens[k]}”)`

  const list: Challenge[] = [
    { id: 'b1', level: 1, title: 'Identify Q, K and V', kind: 'mc', prompt: 'Which vectors are weighted by the attention probabilities and added up to form a head\'s output?',
      options: ['the value vectors V', 'the key vectors K', 'the query vectors Q', 'the raw scores S'], answer: 'the value vectors V',
      hints: ['Queries and keys only meet in the dot products.', 'o_i = Σ_j a_ij · ?'], solution: ['A = softmax(QKᵀ/√d_k + M) uses Q and K to make the weights.', 'The output is o_i = Σ_j a_ij v_j: the values are mixed.'], misconception: 'Thinking the keys that matched are what gets copied into the output.', verify: 'aggregate' },
    { id: 'b2', level: 1, title: 'Query and key positions', kind: 'mc', prompt: `In an attention heatmap (rows = queries, columns = keys), you look at the cell in row ${hr}, column ${hc}. Which token is the query?`,
      options: [`token ${hr}`, `token ${hc}`, 'both', 'neither'], answer: `token ${hr}`,
      hints: ['Each row is one query\'s distribution.'], solution: [`Row ${hr} belongs to query ${hr}; column ${hc} is the key it is compared with.`], misconception: 'Reading heatmaps column-first.', verify: 'heatmap' },
    { id: 'b3', level: 1, title: 'Interpret the heatmap (live lab)', kind: 'num', tol: 0, prompt: `In the lab right now, head ${h + 1}: which key token does query ${tk(qi)} attend to most? Answer with the token number.`, answer: argmax + 1,
      hints: [`Look along row ${qi + 1} of head ${h + 1}'s heatmap.`, 'The brightest cell in that row.'], solution: [`Row ${qi + 1} of A: ${fv(hd.A[qi], 3)}.`, `The largest weight is at key ${argmax + 1}.`], misconception: 'Using the largest value in a column instead of a row.', verify: 'heatmap' },
    { id: 'b4', level: 1, title: 'Causally masked positions', kind: 'set', prompt: `A causal mask over n = ${N} tokens. Which key positions are masked for query ${cq}? (comma-separated, or “none”)`,
      answer: Array.from({ length: N - cq }, (_, k) => cq + 1 + k), hints: ['A query may look at itself and earlier positions.', `Masked: j > ${cq}.`],
      solution: [`M_ij = −∞ when j > i. For i = ${cq}: j = ${Array.from({ length: N - cq }, (_, k) => cq + 1 + k).join(', ')}.`], misconception: 'Masking the diagonal too: a token may attend to itself.', verify: 'mask' },

    { id: 'i1', level: 2, title: 'A dot product', kind: 'num', prompt: `q = ${fv(a3, 1)}, k = ${fv(b3, 1)}. Compute q · k.`, answer: dot,
      hints: ['Multiply matching coordinates.', 'Then add the three products.'], solution: [`${a3.map((x, k) => `${f(x, 1)}×${f(b3[k], 1)}`).join(' + ')}`, `= ${dotTerms(a3, b3).products.map((p) => f(p, 2)).join(' + ')} = ${f(dot, 2)}`], misconception: 'Adding the vectors instead of multiplying coordinates.', verify: 'dot' },
    { id: 'i2', level: 2, title: 'A scaled score', kind: 'num', prompt: `A raw score s = ${f(sRaw, 2)} in a head with d_k = ${dkS}. Compute s / √d_k.`, answer: sRaw / Math.sqrt(dkS),
      hints: [`√${dkS} = ${f(Math.sqrt(dkS), 4)}.`], solution: [`${f(sRaw, 2)} / ${f(Math.sqrt(dkS), 4)} = ${f(sRaw / Math.sqrt(dkS))}`], misconception: 'Dividing by d_k instead of √d_k.', verify: 'scale' },
    { id: 'i3', level: 2, title: 'Apply a causal mask', kind: 'vec', prompt: `Row ${mr} of the scaled scores (4 tokens) is ${fv(mrow, 1)}. Write the row after adding the causal mask (use -inf for masked entries).`, answer: masked,
      hints: [`Query ${mr} may use keys 1…${mr}.`, 'Masked entries become −∞; the others are unchanged.'], solution: [`M_${mr},: = ${fv(mrow.map((_, k) => (k + 1 > mr ? -Infinity : 0)), 0)}`, `z = ${fv(masked, 1)}`], misconception: 'Setting masked entries to 0 instead of −∞ (0 would still get probability).', verify: 'mask' },
    { id: 'i4', level: 2, title: 'Softmax probabilities', kind: 'num', prompt: `Logits of one row: ${fv(lg, 1)}. Compute the probability of entry ${lk + 1}.`, answer: sm.p[lk],
      hints: ['Subtract the maximum first (optional, same result).', 'p_k = exp(z_k) / Σ exp(z_j).'], solution: [`max = ${f(sm.max, 1)}; exp(z − max) = ${fv(sm.exps, 4)}`, `Σ = ${f(sm.sum)}`, `p_${lk + 1} = ${f(sm.exps[lk])} / ${f(sm.sum)} = ${f(sm.p[lk])}`], misconception: 'Dividing the logits by their sum (they can be negative; that is not softmax).', verify: 'softmax' },
    { id: 'i5', level: 2, title: 'A weighted value sum', kind: 'vec', prompt: `Weights a = ${fv(w, 1)}; values v₁ = ${fv(vs[0], 0)}, v₂ = ${fv(vs[1], 0)}, v₃ = ${fv(vs[2], 0)}. Compute o = Σ a_j v_j (two numbers).`, answer: agg,
      hints: ['Do one dimension at a time.', `o₁ = ${f(w[0], 1)}·${vs[0][0]} + ${f(w[1], 1)}·${vs[1][0]} + ${f(w[2], 1)}·${vs[2][0]}.`], solution: [`o₁ = ${w.map((a, j) => `${f(a, 1)}×${vs[j][0]}`).join(' + ')} = ${f(agg[0], 2)}`, `o₂ = ${w.map((a, j) => `${f(a, 1)}×${vs[j][1]}`).join(' + ')} = ${f(agg[1], 2)}`], misconception: 'Picking the value with the largest weight instead of blending.', verify: 'aggregate' },

    { id: 'a1', level: 3, title: 'Tensor shapes', kind: 'shape', prompt: `n = ${n} tokens, d_model = ${dm}, h = ${hh} heads. What is the shape of one head's Q? (rows×cols)`, answer: `${n}×${dm / hh}`,
      hints: ['d_k = d_model / h.', 'One row per token.'], solution: [`d_k = ${dm} / ${hh} = ${dm / hh}`, `Q_h = X W_Q^(h): (${n}×${dm})(${dm}×${dm / hh}) = ${n}×${dm / hh}`, `(and A_h is ${n}×${n}, the concatenation ${n}×${dm})`], misconception: 'Giving each head the full d_model width.', verify: 'qkv' },
    { id: 'a2', level: 3, title: 'A head output (live lab)', kind: 'vec', prompt: `Head ${h + 1}, query ${tk(qi)}: weights ${fv(hd.A[qi], 4)}, values ${hd.V.map((v, k) => `v${k + 1} = ${fv(v, 3)}`).join(', ')}. Compute the head output o (${t.dk} numbers).`, answer: hd.O[qi],
      hints: ['o = Σ_j a_j v_j.', 'Masked keys have weight 0 and drop out.'], solution: hd.O[qi].map((o, c) => `o_${c + 1} = ${hd.A[qi].map((a, j) => `${f(a, 3)}×${f(hd.V[j][c], 3)}`).join(' + ')} = ${f(o)}`), misconception: 'Forgetting that the weights already sum to 1 (no further division).', verify: 'aggregate' },
    { id: 'a3', level: 3, title: 'Concatenate head outputs', kind: 'vec', prompt: `For one token, head 1 outputs ${fv(h1, 1)} and head 2 outputs ${fv(h2, 1)}. Write the concatenated row.`, answer: [...h1, ...h2],
      hints: ['Place them side by side, head 1 first.'], solution: [`[${[...h1, ...h2].map((x) => f(x, 1)).join(', ')}]`], misconception: 'Averaging or adding the heads.', verify: 'concat' },
    { id: 'a4', level: 3, title: 'Apply the output projection', kind: 'num', prompt: `A concatenated row H = ${fv(Hrow, 1)} and column c of W_O = ${fv(wcol, 1)}. Compute that output cell.`, answer: proj,
      hints: ['It is a dot product.'], solution: [`${Hrow.map((x, k) => `${f(x, 1)}×${f(wcol[k], 1)}`).join(' + ')} = ${f(proj, 2)}`], misconception: 'Stopping after the concatenation.', verify: 'proj' },
    { id: 'a5', level: 3, title: 'Diagnose an incorrect calculation', kind: 'mc', prompt: `${bugText} What went wrong?`,
      options: ['forgot to divide by √d_k', 'masked after softmax without renormalizing', 'normalized over queries (columns) instead of keys (rows)', 'nothing: the result is correct'], answer: bugAns,
      hints: ['Check whether each query row sums to 1.', 'Recompute the softmax yourself from the given scores.'],
      solution: bug === 'noscale' ? [`Correct: softmax(s/4) = ${fv(stableSoftmax(bs.map((x) => x / 4)).p, 3)}.`, 'The reported weights are softmax(s), i.e. without the 1/√16 factor (too peaked).']
        : bug === 'after' ? ['The reported row sums to less than 1.', `Correct: mask first (keys 2, 3 → −∞), then softmax → [1, 0, 0].`]
          : ['Columns sum to 1 but rows do not.', 'Softmax must run across the keys of each query (each row).'],
      misconception: 'Trusting a result because each step looks plausible: always check that rows sum to 1.', verify: 'softmax' },
    { id: 'a6', level: 3, title: 'Predict the effect of changing V', kind: 'mc', prompt: 'Only head 1\'s W_V is changed (everything else fixed, h = 2). What happens?',
      options: ['head 1\'s weights unchanged, head 2 unchanged, final output can change', 'head 1\'s weights change, final output changes', 'everything stays the same', 'both heads\' weights change'], answer: 'head 1\'s weights unchanged, head 2 unchanged, final output can change',
      hints: ['Which quantities depend on W_V?', 'A = softmax(QKᵀ/√d_k + M).'], solution: ['W_V affects only V₁ and therefore head₁ = A₁V₁.', 'A₁ uses only Q₁ and K₁; head 2 uses its own matrices.', 'The concatenation and W_O pass the change on to the output.'], misconception: 'Thinking any weight change alters the attention pattern.', verify: 'heads' },
  ]
  // seeded shuffle of the multiple-choice options so the right answer is not always first
  return list.map((c, k) => (c.options ? { ...c, options: shuffle(c.options, seed * 131 + k) } : c))
}
export function shuffle<T>(a: T[], seed: number): T[] {
  const r = rng(seed + 17), out = a.slice()
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [out[i], out[j]] = [out[j], out[i]] }
  return out
}
