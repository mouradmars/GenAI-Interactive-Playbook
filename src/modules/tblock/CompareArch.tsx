/** Preset A (GPT, pre-LN, causal) vs preset B (BERT, post-LN, bidirectional) on the same input and weights. */
import { useMemo } from 'react'
import { runBlock, vnorm } from '../../engine/block'
import { Card, Tex } from '../../components/viz'
import { MatrixView, f3, useRoles } from './common'
import { ARCH_LABEL } from './stages'
import { cfgOf, useTb } from './store'

export function CompareArch() {
  const r = useRoles()
  const s = useTb((x) => x)
  const tA = useMemo(() => runBlock(cfgOf({ ...s, arch: 'gpt', maskOff: false }), s.params, s.X), [s.T, s.d, s.H, s.dff, s.act, s.eps, s.params, s.X]) // eslint-disable-line react-hooks/exhaustive-deps
  const tB = useMemo(() => runBlock(cfgOf({ ...s, arch: 'bert', maskOff: false }), s.params, s.X), [s.T, s.d, s.H, s.dff, s.act, s.eps, s.params, s.X]) // eslint-disable-line react-hooks/exhaustive-deps
  const h = Math.min(s.sel.h, s.H - 1)
  const rowStats = (Y: number[][]) => Y.map((y) => { const m = y.reduce((a, b) => a + b, 0) / y.length; return { m, sd: Math.sqrt(y.reduce((a, b) => a + (b - m) ** 2, 0) / y.length) } })
  const sa = rowStats(tA.Y), sb = rowStats(tB.Y)
  const rows: [string, string, string][] = [
    ['typical model', 'GPT-style decoder (language modelling)', 'BERT-style encoder (understanding / masked LM)'],
    ['attention', 'causal: token i sees positions ≤ i', 'bidirectional: every token sees every token'],
    ['mask M', '−∞ above the diagonal', 'all zeros (no padding here)'],
    ['LayerNorm position', 'before each sublayer (pre-LN)', 'after each residual addition (post-LN)'],
    ['first update', 'H = X + Attn(LN₁(X))', 'H = LN₁(X + Attn(X))'],
    ['second update', 'Y = H + MLP(LN₂(H))', 'Y = LN₂(H + MLP(H))'],
    ['residual stream', 'never normalized inside the block; its scale can grow with depth (a final LN is applied after the last block)', 'renormalized after every sublayer: output rows have mean β̄ and std ≈ γ'],
    ['information flow', 'left to right only; suitable for generation', 'both directions; output for a token uses the whole sentence'],
  ]
  return (
    <div className="grid gap-3">
      <Card title="the two block orders">
        <div className="grid gap-3 @3xl:grid-cols-2">
          {(['gpt', 'bert'] as const).map((a) => (
            <div key={a} className={`rounded-lg border p-2 ${s.arch === a ? 'border-cyan' : 'border-line'}`}>
              <div className="mb-1 text-[12.5px] font-medium" style={{ color: a === 'gpt' ? r.q : r.k }}>{ARCH_LABEL[a]}</div>
              <FlowStrip arch={a} />
              <div className="mt-1 overflow-x-auto scroll-thin text-[13px]"><Tex block tex={a === 'gpt' ? 'H = X + \\mathrm{Attn}_{\\text{causal}}(\\mathrm{LN}_1(X))' : 'H = \\mathrm{LN}_1(X + \\mathrm{Attn}_{\\text{bidir}}(X))'} /><Tex block tex={a === 'gpt' ? 'Y = H + \\mathrm{MLP}(\\mathrm{LN}_2(H))' : 'Y = \\mathrm{LN}_2(H + \\mathrm{MLP}(H))'} /></div>
            </div>
          ))}
        </div>
      </Card>
      <Card title="differences">
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-[12.5px]" data-testid="arch-diff">
            <thead className="text-left text-faint"><tr><th className="px-2 py-1" /><th className="px-2 py-1">A · GPT pre-LN</th><th className="px-2 py-1">B · BERT post-LN</th></tr></thead>
            <tbody>{rows.map(([k, a, b]) => <tr key={k} className="border-t border-line align-top"><td className="px-2 py-1 text-muted">{k}</td><td className="px-2 py-1">{a}</td><td className="px-2 py-1">{b}</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`A · attention weights, head ${h + 1} (causal)`}><MatrixView m={tA.heads[h].A} mode="prob" maxAbs={1} rowLabels={s.tokens} colLabels={s.tokens.map((_, n) => `k${n + 1}`)} color={r.score} ariaLabel="Causal attention" testid="cmp-A" cell={46} /></Card>
        <Card title={`B · attention weights, head ${h + 1} (bidirectional)`}><MatrixView m={tB.heads[h].A} mode="prob" maxAbs={1} rowLabels={s.tokens} colLabels={s.tokens.map((_, n) => `k${n + 1}`)} color={r.score} ariaLabel="Bidirectional attention" testid="cmp-B" cell={46} /></Card>
      </div>
      <p className="text-[12px] text-muted">Same input X and the same weights in both. B's first query row differs from A's because B lets it see later tokens; A's upper triangle is exactly zero. B also has no LayerNorm before attention, so its scores use raw X.</p>
      <div className="grid gap-3 @3xl:grid-cols-2">
        {[{ t: tA, st: sa, l: 'A · output Y (residual stream, not normalized)' }, { t: tB, st: sb, l: 'B · output Y (after LN₂)' }].map((x) => (
          <Card key={x.l} title={x.l}>
            <MatrixView m={x.t.Y} rowLabels={s.tokens} colLabels={x.t.Y[0].map((_, k) => `x${k + 1}`)} color={r.tok} ariaLabel={x.l} cell={46} />
            <table className="mt-1 font-mono text-[11.5px]"><thead className="text-faint"><tr><th className="px-1 text-left">token</th><th className="px-1 text-right">row mean</th><th className="px-1 text-right">row std</th><th className="px-1 text-right">‖y‖</th></tr></thead>
              <tbody>{x.st.map((q, i) => <tr key={i} className="border-t border-line"><td className="px-1">{s.tokens[i]}</td><td className="px-1 text-right">{f3(q.m)}</td><td className="px-1 text-right">{f3(q.sd)}</td><td className="px-1 text-right">{f3(vnorm(x.t.Y[i]))}</td></tr>)}</tbody></table>
          </Card>
        ))}
      </div>
      <p className="text-[12px] text-muted">With γ = 1 and β = 0, every row of B's output has mean 0 and standard deviation ≈ 1 (post-LN renormalizes the stream). A's rows keep whatever scale the residual additions produce. This is why pre-LN models add one final LayerNorm after the last block, before the LM head.</p>
    </div>
  )
}

function FlowStrip({ arch }: { arch: 'gpt' | 'bert' }) {
  const r = useRoles()
  const items: [string, string][] = arch === 'gpt'
    ? [['X', r.tok], ['LN', r.ln], ['masked MHSA', r.score], ['⊕', r.res], ['LN', r.ln], ['MLP', r.ffn], ['⊕', r.res], ['Y', r.tok]]
    : [['X', r.tok], ['MHSA', r.score], ['⊕', r.res], ['LN', r.ln], ['MLP', r.ffn], ['⊕', r.res], ['LN', r.ln], ['Y', r.tok]]
  return (
    <div className="flex flex-wrap items-center gap-1 font-mono text-[11.5px]">
      {items.map(([l, c], k) => <span key={k} className="flex items-center gap-1"><span className="rounded border px-1.5 py-0.5" style={{ borderColor: c, color: c }}>{l}</span>{k < items.length - 1 && <span className="text-faint">→</span>}</span>)}
    </div>
  )
}
