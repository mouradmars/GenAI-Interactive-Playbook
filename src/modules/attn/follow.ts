/**
 * "Follow one token through attention": the twelve steps of one query's journey, each with its equation, the
 * substituted numbers and the cells to highlight. Pure function of the engine trace (unit-tested).
 */
import { aggregationTerms, matmulCell, type AttnTrace } from '../../engine/attention'
import type { StageId } from './stages'

/** the guided-pipeline operation each follow step corresponds to (keeps the inspector and microscope C in sync) */
export const FOLLOW_IDS = ['input', 'qproj', 'compare', 'scores', 'scale', 'mask', 'softmax', 'weights', 'values', 'head', 'concat', 'proj']
export const FOLLOW_STAGE: Record<string, StageId> = {
  input: 'embed', qproj: 'qkv', compare: 'dot', scores: 'dot', scale: 'scale', mask: 'mask', softmax: 'softmax', weights: 'heatmap', values: 'aggregate', head: 'aggregate', concat: 'concat', proj: 'proj',
}

export type FollowMatrix = 'X' | 'Q' | 'K' | 'S' | 'Ss' | 'M' | 'A' | 'V' | 'O' | 'H' | 'out'
export interface FollowStep {
  id: string
  title: string
  tex: string
  /** one or more lines of substituted numbers (TeX) */
  numbers: string[]
  purpose: string
  /** the matrix to display and the cells to highlight */
  show: FollowMatrix
  cells: [number, number][]
  rows: number[]
}

const t3 = (v: number, d = 3) => (v === -Infinity ? '-\\infty' : Math.abs(v) < 5e-13 ? (0).toFixed(d) : v < 0 ? `(${v.toFixed(d)})` : v.toFixed(d))
const vec = (v: number[], d = 3) => `[${v.map((x) => t3(x, d)).join(',\\,')}]`

export function followSteps(t: AttnTrace, i: number, h: number, Wo: number[][]): FollowStep[] {
  const hd = t.heads[h], n = t.Tk, row = hd.rows[i], dk = t.dk
  const allowed = row.allowed.flatMap((a, j) => (a ? [j] : []))
  const g = aggregationTerms(hd, i)
  const qc = matmulCell(t.X, hd.Wq, i, 0)
  const oc = matmulCell(t.concat, Wo, i, 0)
  const all = Array.from({ length: n }, (_, j) => j)
  return [
    { id: 'input', title: 'Input vector', tex: `x_{${i + 1}} = X_{${i + 1},:}`, numbers: [`x_{${i + 1}} = ${vec(t.X[i], 2)}`], purpose: 'The token enters as one row of X: a vector of d_model numbers.', show: 'X', cells: [], rows: [i] },
    { id: 'qproj', title: 'Query projection', tex: `q_{${i + 1}} = x_{${i + 1}} W_Q^{(${h + 1})}`, numbers: [`q_{${i + 1},1} = ${qc.a.map((a, k) => `${t3(a, 2)}\\cdot${t3(qc.b[k], 2)}`).join(' + ')} = ${t3(qc.sum)}`, `q_{${i + 1}} = ${vec(hd.Q[i])}`], purpose: 'A learned matrix turns the input into the vector used to compare against keys.', show: 'Q', cells: [], rows: [i] },
    { id: 'compare', title: 'Comparison with all keys', tex: `q_{${i + 1}} \\cdot k_j \\;\\text{for } j = 1..${n}`, numbers: all.map((j) => `q_{${i + 1}}\\cdot k_{${j + 1}} = ${hd.Q[i].map((q, c) => `${t3(q, 2)}\\cdot${t3(hd.K[j][c], 2)}`).join(' + ')}`), purpose: 'The query is compared with the key of every position, including its own.', show: 'K', cells: [], rows: all },
    { id: 'scores', title: 'Dot-product scores', tex: `s_{${i + 1},:} = q_{${i + 1}} K^{\\top}`, numbers: [`s_{${i + 1},:} = ${vec(hd.S[i])}`], purpose: 'One compatibility score per key; not yet probabilities.', show: 'S', cells: all.map((j) => [i, j] as [number, number]), rows: [] },
    { id: 'scale', title: 'Scaling', tex: t.cfg.scale ? `\\tilde s_{${i + 1},:} = s_{${i + 1},:} / \\sqrt{${dk}}` : `\\tilde s = s \\;(\\text{scaling off})`, numbers: [`\\tilde s_{${i + 1},:} = ${vec(hd.scaled[i])}`], purpose: 'Divide by √d_k so the logits stay in a moderate range.', show: 'Ss', cells: all.map((j) => [i, j] as [number, number]), rows: [] },
    { id: 'mask', title: 'Masking', tex: `z_{${i + 1},:} = \\tilde s_{${i + 1},:} + M_{${i + 1},:}`, numbers: [`M_{${i + 1},:} = ${vec(t.M[i], 0)}`, `z_{${i + 1},:} = ${vec(hd.logits[i])}`], purpose: allowed.length ? `Allowed keys: ${allowed.map((j) => j + 1).join(', ')}. Masked keys get −∞.` : 'Every key is masked: this row has no valid distribution.', show: 'M', cells: all.map((j) => [i, j] as [number, number]), rows: [] },
    { id: 'softmax', title: 'Softmax', tex: `a_{${i + 1}j} = \\frac{e^{z_{${i + 1}j} - m}}{\\sum_k e^{z_{${i + 1}k} - m}}`, numbers: row.fullyMasked ? ['\\text{fully masked: all weights } 0'] : [`m = ${t3(row.max)},\\; e^{z-m} = ${vec(row.exps, 4)}`, `\\Sigma = ${t3(row.sum, 4)}`], purpose: 'Exponentiate the stabilised logits and normalise them to sum to 1.', show: 'Ss', cells: allowed.map((j) => [i, j] as [number, number]), rows: [] },
    { id: 'weights', title: 'Attention weights', tex: `a_{${i + 1},:} = \\operatorname{softmax}(z_{${i + 1},:})`, numbers: [`a_{${i + 1},:} = ${vec(hd.A[i], 4)}`, `\\textstyle\\sum_j a_{${i + 1}j} = ${t3(hd.A[i].reduce((a, b) => a + b, 0), 6)}`], purpose: 'A probability distribution over the permitted keys: where this query reads from.', show: 'A', cells: all.map((j) => [i, j] as [number, number]), rows: [] },
    { id: 'values', title: 'Weighted values', tex: `a_{${i + 1}j}\\, v_j`, numbers: all.map((j) => `${t3(hd.A[i][j], 3)}\\cdot${vec(hd.V[j], 2)} = ${vec(g.contrib[j], 4)}`), purpose: 'Each value vector is scaled by its weight.', show: 'V', cells: [], rows: all },
    { id: 'head', title: 'Head output', tex: `o_{${i + 1}} = \\sum_j a_{${i + 1}j} v_j`, numbers: [`o_{${i + 1}} = ${vec(g.o, 4)}`], purpose: 'The weighted contributions add up to this head\'s output for the token.', show: 'O', cells: [], rows: [i] },
    { id: 'concat', title: 'Multi-head concatenation', tex: `H_{${i + 1}} = [\\text{head}_1 \\,|\\, \\dots \\,|\\, \\text{head}_{${t.cfg.H}}]_{${i + 1}}`, numbers: [`H_{${i + 1}} = ${vec(t.concat[i])}`], purpose: `All ${t.cfg.H} heads ran in parallel; their outputs for this token are placed side by side.`, show: 'H', cells: [], rows: [i] },
    { id: 'proj', title: 'Output projection', tex: `\\text{out}_{${i + 1}} = H_{${i + 1}} W_O`, numbers: [`\\text{out}_{${i + 1},1} = ${oc.a.map((a, k) => `${t3(a, 2)}\\cdot${t3(oc.b[k], 1)}`).join(' + ')} = ${t3(oc.sum, 4)}`, `\\text{out}_{${i + 1}} = ${vec(t.out[i], 4)}`], purpose: 'A learned matrix mixes the heads and maps back to d_model: the token\'s attention output.', show: 'out', cells: [], rows: [i] },
  ]
}
