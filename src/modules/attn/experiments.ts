/**
 * Eight interactive experiments. Each `run` is a pure function of the current lab state: it builds the modified
 * state, runs both through the engine and reports numerical results. Nothing here is scripted.
 */
import { PRESET_X, PRESET_X_B, compareTraces, presetInput, runAttention, maxAbsDiff, type AttnConfig, type AttnParams, type AttnTrace, type Matrix } from '../../engine/attention'

export interface ExpState { X: Matrix; params: AttnParams; cfg: AttnConfig }
export interface ExpResult { before: AttnTrace; after: AttnTrace; afterState: ExpState; metrics: { label: string; value: string }[]; observed: string; change: string }
export interface Experiment {
  id: string; n: number; title: string; hypothesis: string; question: string; options: string[]; answer: string; explanation: string
  visual: 'heatmaps' | 'entropy' | 'mask' | 'shapes' | 'ablation' | 'inputs'
  run: (s: ExpState, h: number) => ExpResult
}

const cloneP = (p: AttnParams): AttnParams => ({ Wq: p.Wq.map((r) => r.slice()), Wk: p.Wk.map((r) => r.slice()), Wv: p.Wv.map((r) => r.slice()), Wo: p.Wo.map((r) => r.slice()) })
const f = (v: number, d = 4) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d)
const rowsChanged = (a: Matrix, b: Matrix) => a.flatMap((r, i) => (r.some((v, j) => Math.abs(v - b[i][j]) > 1e-12) ? [i + 1] : []))
const colsChanged = (a: Matrix, b: Matrix) => a[0].flatMap((_, j) => (a.some((r, i) => Math.abs(r[j] - b[i][j]) > 1e-12) ? [j + 1] : []))
const list = (v: number[]) => (v.length ? v.join(', ') : 'none')
const dk = (s: ExpState) => s.X[0].length / s.cfg.H

function edit(s: ExpState, key: keyof AttnParams, r: number, c: number, delta: number): ExpState {
  const params = cloneP(s.params); params[key][r][c] += delta
  return { ...s, params }
}
function result(s: ExpState, a: ExpState, change: string, metrics: (b: AttnTrace, t: AttnTrace) => { label: string; value: string }[], observed: (b: AttnTrace, t: AttnTrace) => string): ExpResult {
  const before = runAttention(s.X, s.params, s.cfg), after = runAttention(a.X, a.params, a.cfg)
  return { before, after, afterState: a, change, metrics: metrics(before, after), observed: observed(before, after) }
}

export const EXPERIMENTS: Experiment[] = [
  { id: 'query', n: 1, title: 'Change a query projection', visual: 'heatmaps',
    hypothesis: 'Queries decide how strongly each position attends to the others, so changing W_Q should change the scores, the weights and the output.',
    question: 'You add 1 to one entry of this head\'s W_Q. What changes?', options: ['scores, weights and output', 'only the output', 'nothing'], answer: 'scores, weights and output',
    explanation: 'S = Q Kᵀ depends on Q, so the logits of every query whose q changed move; softmax turns that into new weights and A·V into a new output. Keys and values are untouched.',
    run: (s, h) => { const c = h * dk(s); const a = edit(s, 'Wq', 0, c, 1); return result(s, a, `W_Q[1, ${c + 1}] += 1 (head ${h + 1})`,
      (b, t) => { const d = compareTraces(b, t).heads[h]; return [{ label: 'max |ΔQ|', value: f(d.Q) }, { label: 'max |ΔK|', value: f(d.K) }, { label: 'max |ΔS|', value: f(d.S) }, { label: 'max |ΔA|', value: f(d.A) }, { label: 'max |Δoutput|', value: f(compareTraces(b, t).out) }] },
      (b, t) => `Weight rows that changed: ${list(rowsChanged(b.heads[h].A, t.heads[h].A))}. Keys and values are identical.`) } },
  { id: 'key', n: 2, title: 'Change a key projection', visual: 'heatmaps',
    hypothesis: 'Keys determine compatibility with queries. Changing one entry of W_K changes the keys, and with them some query–key scores.',
    question: 'You add 1 to entry [2, c] of W_K. Which scores s_ij change?', options: ['those where q_i,c ≠ 0 and x_j,2 ≠ 0', 'all of them, always', 'only the diagonal'], answer: 'those where q_i,c ≠ 0 and x_j,2 ≠ 0',
    explanation: 'Δk_j = x_j ΔW_K = x_j,2 · e_c, so Δs_ij = q_i · Δk_j = q_i,c · x_j,2. A key whose input has 0 in feature 2 keeps exactly the same scores.',
    run: (s, h) => { const c = h * dk(s); const a = edit(s, 'Wk', 1, c, 1); return result(s, a, `W_K[2, ${c + 1}] += 1 (head ${h + 1})`,
      (b, t) => { const d = compareTraces(b, t).heads[h]; return [{ label: 'max |ΔK|', value: f(d.K) }, { label: 'max |ΔS|', value: f(d.S) }, { label: 'max |ΔA|', value: f(d.A) }, { label: 'key columns with changed scores', value: list(colsChanged(b.heads[h].S, t.heads[h].S)) }, { label: 'x_j,2 per key', value: s.X.map((r) => f(r[1], 1)).join(', ') }] },
      (b, t) => `Score columns that changed: ${list(colsChanged(b.heads[h].S, t.heads[h].S))}; unchanged columns belong to keys with x_j,2 = 0 (or rows with q_i,c = 0).`) } },
  { id: 'value', n: 3, title: 'Change a value projection', visual: 'heatmaps',
    hypothesis: 'Values are what gets mixed, not how it is mixed. Changing only W_V should leave the attention weights exactly unchanged.',
    question: 'You add 1 to one entry of this head\'s W_V. What happens to the attention weights?', options: ['they stay exactly the same; the output may change', 'they change', 'they all become uniform'], answer: 'they stay exactly the same; the output may change',
    explanation: 'A = softmax(QKᵀ/√d_k + M) does not involve V. Only head = A·V, the concatenation and the output change.',
    run: (s, h) => { const c = h * dk(s); const a = edit(s, 'Wv', 0, c, 1); return result(s, a, `W_V[1, ${c + 1}] += 1 (head ${h + 1})`,
      (b, t) => { const d = compareTraces(b, t); return [{ label: 'max |ΔA| (all heads)', value: f(Math.max(...d.heads.map((x) => x.A))) }, { label: 'max |ΔV|', value: f(d.heads[h].V) }, { label: 'max |Δhead output|', value: f(d.heads[h].O) }, { label: 'max |Δoutput|', value: f(d.out) }] },
      (b, t) => (compareTraces(b, t).heads.every((x) => !x.changed.A) ? 'Attention weights identical (to 1e−12); the output changed.' : 'Unexpected: weights changed.')) } },
  { id: 'scaling', n: 4, title: 'Remove the √dₖ scaling', visual: 'entropy',
    hypothesis: 'Without scaling the logits are √d_k times larger, so each softmax row becomes more peaked (lower entropy).',
    question: 'Switch scaling off. What happens to the entropy of the attention rows?', options: ['it decreases (sharper distributions)', 'it increases', 'it stays the same'], answer: 'it decreases (sharper distributions)',
    explanation: 'Multiplying all logits of a row by √d_k > 1 sharpens softmax. The ranking of keys is unchanged. (With d_k = 1 nothing changes.)',
    run: (s, h) => { const a = { ...s, cfg: { ...s.cfg, scale: !s.cfg.scale } }; return result(s, a, `scaling ${s.cfg.scale ? 'off' : 'on'}`,
      (b, t) => [{ label: `entropy per row before (head ${h + 1})`, value: b.heads[h].entropy.map((e) => f(e, 3)).join(' · ') }, { label: 'entropy per row after', value: t.heads[h].entropy.map((e) => f(e, 3)).join(' · ') }, { label: 'largest weight before → after', value: `${f(Math.max(...b.heads[h].A.flat()), 3)} → ${f(Math.max(...t.heads[h].A.flat()), 3)}` }],
      (b, t) => { const m0 = mean(b.heads[h].entropy), m1 = mean(t.heads[h].entropy); return `Mean entropy ${f(m0, 3)} → ${f(m1, 3)} nats.` }) } },
  { id: 'causal', n: 5, title: 'Toggle the causal mask', visual: 'mask',
    hypothesis: 'The causal mask hides future tokens. Removing it gives earlier tokens access to later ones; the last token already sees everything.',
    question: 'Remove the causal mask. Which output row is guaranteed not to change?', options: ['the last token\'s row', 'the first token\'s row', 'none of them'], answer: 'the last token\'s row',
    explanation: 'Under the causal mask the last query already attends to all keys, so its row of M is all zeros either way. Every earlier row gains access to later keys.',
    run: (s) => { const a = { ...s, cfg: { ...s.cfg, causal: !s.cfg.causal } }; return result(s, a, `causal mask ${s.cfg.causal ? 'off' : 'on'}`,
      (b, t) => [{ label: 'permitted pairs before → after', value: `${count(b.M)} → ${count(t.M)}` }, { label: 'output rows that changed', value: list(rowsChanged(b.out, t.out)) }, { label: 'max |Δoutput|', value: f(maxAbsDiff(b.out, t.out)) }],
      (b, t) => `Output rows that changed: ${list(rowsChanged(b.out, t.out))}.`) } },
  { id: 'heads', n: 6, title: 'Change the head count', visual: 'shapes',
    hypothesis: 'With d_model fixed, more heads means smaller heads (d_k = d_model / h). The total size of W_Q, W_K, W_V, W_O stays the same.',
    question: 'Switch from h heads to a different valid count. What happens to the number of projection parameters?', options: ['it stays the same', 'it grows with h', 'it shrinks with h'], answer: 'it stays the same',
    explanation: 'Each W is d_model × d_model whatever h is; h only decides how its columns are grouped. The attention patterns and the output change, because each head now has a different number of dimensions.',
    run: (s) => { const H = s.cfg.H === 2 ? 4 : 2; const a = { ...s, cfg: { ...s.cfg, H, headsOn: new Array(H).fill(true) } }; return result(s, a, `h = ${s.cfg.H} → ${H}`,
      (b, t) => [{ label: 'd_k before → after', value: `${b.dk} → ${t.dk}` }, { label: 'shape of one head\'s Q', value: `${b.Tq}×${b.dk} → ${t.Tq}×${t.dk}` }, { label: 'attention matrices (h × n × n)', value: `${b.cfg.H}×${b.Tq}×${b.Tk} → ${t.cfg.H}×${t.Tq}×${t.Tk}` }, { label: 'parameters in W_Q, W_K, W_V, W_O', value: `${4 * s.X[0].length ** 2} → ${4 * s.X[0].length ** 2}` }, { label: 'max |Δoutput|', value: f(maxAbsDiff(b.out, t.out)) }],
      (b, t) => `${b.cfg.H} heads of size ${b.dk} → ${t.cfg.H} heads of size ${t.dk}; same parameters, different computation.`) } },
  { id: 'ablate', n: 7, title: 'Disable one head', visual: 'ablation',
    hypothesis: 'A disabled head contributes zeros to the concatenation, so the output loses that head\'s contribution head_h W_O^(h).',
    question: 'Disable this head. By how much does each output row change?', options: ['by exactly that head\'s contribution head_h W_O^(h)', 'not at all', 'by a random amount'], answer: 'by exactly that head\'s contribution head_h W_O^(h)',
    explanation: 'O = Σ_h head_h W_O^(h). Removing one term subtracts exactly its contribution; other heads are unaffected.',
    run: (s, h) => { const headsOn = s.cfg.headsOn.slice(); headsOn[h] = false; const a = { ...s, cfg: { ...s.cfg, headsOn } }; return result(s, a, `head ${h + 1} disabled`,
      (b, t) => [{ label: '‖Δoutput‖ per row', value: b.out.map((r, i) => f(Math.hypot(...r.map((v, k) => v - t.out[i][k])), 3)).join(' · ') }, { label: 'other heads changed?', value: compareTraces(b, t).heads.some((x, k) => k !== h && Object.values(x.changed).some(Boolean)) ? 'yes' : 'no' }],
      (b, t) => `Mean ‖Δoutput‖ = ${f(mean(b.out.map((r, i) => Math.hypot(...r.map((v, k) => v - t.out[i][k])))), 3)}.`) } },
  { id: 'inputs', n: 8, title: 'Compare two input sequences', visual: 'inputs',
    hypothesis: 'With the same weights, a different input X gives different queries, keys and values, and therefore a different attention pattern.',
    question: 'Same weights, a different X. What changes?', options: ['the attention pattern and the output', 'only the output', 'nothing: the weights decide everything'], answer: 'the attention pattern and the output',
    explanation: 'Attention weights are computed from the input (Q = XW_Q, K = XW_K); they are not fixed parameters. That is why attention is called content-based.',
    run: (s, h) => {
      const T = s.X.length
      const same = (m: Matrix) => m.length === T && m.every((r, i) => r.every((v, k) => v === s.X[i][k]))
      const alt = T === 4 ? (same(PRESET_X_B) ? PRESET_X : PRESET_X_B) : presetInput(T, 4, 7)
      const a = { ...s, X: alt.map((r) => r.slice()) }
      return result(s, a, 'input X replaced by a second hand-chosen input',
        (b, t) => [{ label: `max |ΔA| (head ${h + 1})`, value: f(maxAbsDiff(b.heads[h].A, t.heads[h].A)) }, { label: 'max |Δoutput|', value: f(maxAbsDiff(b.out, t.out)) }, { label: 'most-attended key per query, before', value: b.heads[h].A.map((r) => argmax(r) + 1).join(', ') }, { label: '… after', value: t.heads[h].A.map((r) => argmax(r) + 1).join(', ') }],
        () => 'Same parameters, different input → different attention.')
    } },
]
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
const count = (M: Matrix) => M.flat().filter((v) => v === 0).length
export const argmax = (r: number[]) => r.reduce((b, v, k) => (v > r[b] ? k : b), 0)
