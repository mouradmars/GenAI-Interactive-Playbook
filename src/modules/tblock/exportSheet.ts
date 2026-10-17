/** Markdown calculation sheet: every intermediate tensor of the current block, plus the selected attention chain. */
import { shapeTable, type BlockTrace, type Matrix } from '../../engine/block'
import { ARCH_LABEL, type ArchPreset } from './stages'
import type { Sel } from './store'

const n = (v: number, d = 4) => (v === -Infinity ? '−∞' : (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d))
function table(m: Matrix, rows: string[], cols: string[], d = 4) {
  return [`| | ${cols.join(' | ')} |`, `|---|${cols.map(() => '---').join('|')}|`, ...m.map((r, i) => `| ${rows[i]} | ${r.map((v) => n(v, d)).join(' | ')} |`)].join('\n')
}

export function calcSheet(t: BlockTrace, tokens: string[], arch: ArchPreset, sel: Sel): string {
  const { d, H } = t.cfg, dh = d / H
  const f = Array.from({ length: d }, (_, k) => `x${k + 1}`), keys = tokens.map((_, k) => `k${k + 1}`)
  const hd = t.heads[sel.h], i = sel.i, j = sel.j
  const out: string[] = [
    '# Inside the Transformer block · calculation sheet', '',
    `- Architecture: ${ARCH_LABEL[arch]}`,
    `- T = ${t.cfg.T}, d = ${d}, H = ${H} (dₖ = ${dh}), d_ff = ${t.cfg.dff}, activation ${t.cfg.act}, ε = ${t.cfg.eps}${!t.cfg.causal && arch === 'gpt' ? ', causal mask DISABLED (comparison mode)' : ''}`,
    '- Educational simulation: seeded one-decimal toy weights (not trained); computed in float64 by the engine, which is tested against PyTorch.', '',
    '## Tensor shapes', ...shapeTable(t).map((s) => `- ${s.name}: [${s.shape.join(' × ')}] — ${s.note}`), '',
    '## Input X', table(t.X, tokens, f),
  ]
  if (t.ln1) out.push('', '## LN₁(X)', table(t.ln1.out, tokens, f), '', `means: ${t.ln1.mean.map((v) => n(v)).join(', ')}; variances: ${t.ln1.variance.map((v) => n(v)).join(', ')}`)
  out.push('', '## Q', table(t.Q, tokens, f), '', '## K', table(t.K, tokens, f), '', '## V', table(t.V, tokens, f))
  t.heads.forEach((h, k) => out.push('', `## Head ${k + 1}${h.enabled ? '' : ' (disabled)'}`, '### S = Q_h K_hᵀ', table(h.S, tokens, keys), '', '### logits = S/√dₖ + M', table(h.logits, tokens, keys), '', '### A = softmax', table(h.A, tokens, keys), '', '### O = A V_h', table(h.O, tokens, Array.from({ length: dh }, (_, c) => `c${c + 1}`))))
  out.push('', '## Concat', table(t.concat, tokens, f), '', '## Attn = Concat W_O + b_O', table(t.attn, tokens, f), '', `## H (${arch === 'gpt' ? 'X + Attn' : 'LN₁(X + Attn)'})`, table(t.H, tokens, f))
  out.push('', '## φ(Z₁)', table(t.act, tokens, Array.from({ length: t.cfg.dff }, (_, k) => `u${k + 1}`), 3), '', '## MLP', table(t.mlp, tokens, f), '', `## Y (${arch === 'gpt' ? 'H + MLP' : 'LN₂(H + MLP)'})`, table(t.Y, tokens, f))
  out.push('', `## Selected chain: query “${tokens[i]}”, key “${tokens[j]}”, head ${sel.h + 1}`,
    `1. S = q·k = ${hd.q[i].map((v, c) => `${n(v, 3)}×${n(hd.k[j][c], 3)}`).join(' + ')} = ${n(hd.S[i][j])}`,
    `2. scaled: ${n(hd.S[i][j])} / √${dh} = ${n(hd.scaled[i][j])}`,
    `3. mask: ${hd.mask[i][j] === 0 ? 'allowed (+0)' : 'masked (−∞)'} → logit ${n(hd.logits[i][j])}`,
    `4. softmax: exp(${n(hd.rows[i].shifted[j])}) / ${n(hd.rows[i].sum)} = ${n(hd.A[i][j])}`,
    `5. contribution to O_${i + 1}: ${n(hd.A[i][j])} × v_${j + 1} = (${hd.v[j].map((v) => n(hd.A[i][j] * v)).join(', ')})`,
    `6. O_${i + 1} = (${hd.O[i].map((v) => n(v)).join(', ')})`)
  return out.join('\n')
}
