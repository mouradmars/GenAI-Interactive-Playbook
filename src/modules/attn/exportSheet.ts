/** Export: every tensor as JSON plus a Markdown calculation sheet for the selected query, key and head. */
import { aggregationTerms, shapeList, type AttnTrace, type Matrix } from '../../engine/attention'
import type { ExportPayload } from '../../components/ExportDialog'
import type { AttnState } from './store'
import type { StageId } from './stages'

const f = (v: number, d = 4) => (v === -Infinity ? '−∞' : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d))
const mdMatrix = (m: Matrix, rows: string[], cols: string[], d = 4) =>
  [`| | ${cols.join(' | ')} |`, `|---|${cols.map(() => '---').join('|')}|`, ...m.map((r, i) => `| ${rows[i]} | ${r.map((v) => f(v, d)).join(' | ')} |`)].join('\n')
const jsonM = (m: Matrix) => m.map((r) => r.map((v) => (v === -Infinity ? '-inf' : v)))

export function calcSheet(t: AttnTrace, tokens: string[], sel: AttnState['sel']): string {
  const h = Math.min(sel.h, t.cfg.H - 1), hd = t.heads[h], i = Math.min(sel.i, t.Tq - 1), j = Math.min(sel.j, t.Tk - 1)
  const rl = tokens.map((x, k) => `${k + 1}·${x}`), kl = tokens.map((_, k) => `k${k + 1}`)
  const row = hd.rows[i], g = aggregationTerms(hd, i)
  const dc = (k: number) => `c${k + 1}`
  return [
    '# Attention calculation sheet', '',
    'Educational simulation: hand-chosen toy weights (not trained), computed exactly in float64 by src/engine/attention.ts (checked against PyTorch).', '',
    `- Tokens: ${tokens.join(' · ')}`, `- n = ${t.Tq}, d_model = ${t.dModel}, h = ${t.cfg.H}, d_k = d_v = ${t.dk}`,
    `- Mask: ${t.cfg.causal ? 'causal' : 'bidirectional'}${t.cfg.pad.some(Boolean) ? `, padding keys ${t.cfg.pad.flatMap((p, k) => (p ? [k + 1] : [])).join(', ')}` : ''}${t.cfg.blocked ? ', custom knock-outs' : ''} · scaling ${t.cfg.scale ? 'on (÷√d_k)' : 'off'} · heads on: ${t.cfg.headsOn.map((o, k) => `${k + 1}:${o ? 'yes' : 'no'}`).join(' ')}`, '',
    '## Shapes', ...shapeList(t).map((s) => `- ${s.name}: ${s.shape[0]}×${s.shape[1]} (${s.note})`), '',
    '## Input X', mdMatrix(t.X, rl, Array.from({ length: t.dModel }, (_, k) => `f${k + 1}`)), '',
    `## Head ${h + 1}`, '### Q', mdMatrix(hd.Q, rl, Array.from({ length: t.dk }, (_, k) => dc(k))), '', '### K', mdMatrix(hd.K, rl, Array.from({ length: t.dk }, (_, k) => dc(k))), '', '### V', mdMatrix(hd.V, rl, Array.from({ length: t.dk }, (_, k) => dc(k))), '',
    '### S = QKᵀ', mdMatrix(hd.S, rl, kl), '', `### Z = S${t.cfg.scale ? '/√d_k' : ''} + M`, mdMatrix(hd.logits, rl, kl), '', '### A = softmax(Z)', mdMatrix(hd.A, rl, kl), '', '### head output = AV', mdMatrix(hd.O, rl, Array.from({ length: t.dk }, (_, k) => dc(k))), '',
    `## Worked example: query ${i + 1} (“${tokens[i]}”), key ${j + 1} (“${tokens[j]}”), head ${h + 1}`,
    `1. q${i + 1} = [${hd.Q[i].map((v) => f(v)).join(', ')}], k${j + 1} = [${hd.K[j].map((v) => f(v)).join(', ')}]`,
    `2. s = ${hd.Q[i].map((v, k) => `${f(v, 3)}×${f(hd.K[j][k], 3)}`).join(' + ')} = ${f(hd.S[i][j])}`,
    `3. scaled: ${t.cfg.scale ? `${f(hd.S[i][j])} / √${t.dk} = ` : ''}${f(hd.scaled[i][j])}`,
    `4. mask: ${t.M[i][j] === 0 ? 'allowed (0)' : `−∞ (${t.reasons[i][j].join(' + ')})`} → z = ${f(hd.logits[i][j])}`,
    row.fullyMasked ? '5. row fully masked: weights 0, output 0' : `5. softmax: m = ${f(row.max)}, Σ exp(z − m) = ${f(row.sum)}, a = ${f(row.exps[j])} / ${f(row.sum)} = ${f(hd.A[i][j])}`,
    `6. o${i + 1} = ${g.a.map((a, k) => `${f(a, 4)}·v${k + 1}`).join(' + ')} = [${g.o.map((v) => f(v)).join(', ')}]`,
    `7. concat row ${i + 1} = [${t.concat[i].map((v) => f(v)).join(', ')}]`,
    `8. output row ${i + 1} = [${t.out[i].map((v) => f(v)).join(', ')}]`, '',
    '## Multi-head output O = Concat(heads) W_O', mdMatrix(t.out, rl, Array.from({ length: t.dModel }, (_, k) => `f${k + 1}`)), '',
  ].join('\n')
}

export function exportPayload(s: AttnState, t: AttnTrace, tokens: string[], stage: StageId): ExportPayload {
  const json = {
    observatory: 'AI Deep Learning Observatory', module: '03 · Attention — scaled dot-product and multi-head attention from scratch', exportedAt: new Date().toISOString(),
    provenance: 'educational simulation: hand-chosen toy weights (not trained), computed exactly in float64 by src/engine/attention.ts, tested against torch.nn.MultiheadAttention and F.scaled_dot_product_attention',
    stage, mode: s.tab, tokens, sentence: s.sentence, windowStart: s.windowStart, inputSource: s.xSource,
    config: { n: t.Tq, dModel: t.dModel, heads: t.cfg.H, dk: t.dk, causal: t.cfg.causal, scale: t.cfg.scale, pad: t.cfg.pad, blocked: t.cfg.blocked, headsOn: t.cfg.headsOn },
    selection: s.sel, params: s.params, X: t.X, mask: jsonM(t.M),
    heads: t.heads.map((h) => ({ head: h.h + 1, enabled: h.enabled, Wq: h.Wq, Wk: h.Wk, Wv: h.Wv, Q: h.Q, K: h.K, V: h.V, S: h.S, scaled: h.scaled, logits: jsonM(h.logits), A: h.A, output: h.O, entropy: h.entropy })),
    concat: t.concat, output: t.out, fullyMaskedRows: t.fullyMaskedRows.map((r) => r + 1),
    shapes: shapeList(t).map((x) => ({ name: x.name, shape: x.shape })),
  }
  return { json, markdown: calcSheet(t, tokens, s.sel), filenameBase: `attention-${stage}-h${t.cfg.H}-${t.cfg.causal ? 'causal' : 'bidir'}` }
}
