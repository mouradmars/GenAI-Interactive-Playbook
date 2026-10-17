/**
 * The signature view: three synchronized microscopes driven by one selection. Clicking a heatmap cell (or a
 * connection, a matrix cell or a token) updates the token view (A), the matrix chain (B), the pipeline (C) and
 * the eight-step account of how that weight was produced and where it goes.
 */
import { Card } from '../../components/viz'
import { aggregationTerms, type AttnTrace } from '../../engine/attention'
import { GLYPH, Grid, HeadPicker, fx, pct, tokLabel, useRoles, useTokens } from './common'
import { TokenGraph } from './TokenGraph'
import { PipelineView } from './PipelineView'
import { ColorScale } from './viewsMix'
import { stageById } from './stages'
import { useAttn } from './store'

export function MicroscopesTab({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const h = Math.min(sel.h, trace.cfg.H - 1), hd = trace.heads[h], i = sel.i, j = sel.j
  const rl = tokens.map((_, k) => tokLabel(tokens, k)), kl = tokens.map((_, k) => `k${k + 1}`)
  const masked = trace.M.map((x) => x.map((v) => v === -Infinity))
  const g = aggregationTerms(hd, i)
  const pick = (a: number, b: number) => select({ i: a, j: b, cell: null })
  const steps: [string, React.ReactNode][] = [
    ['query', <>token {i + 1} “{tokens[i]}” is the query <span style={{ color: r.q }}>{GLYPH.q}</span></>],
    ['key', <>token {j + 1} “{tokens[j]}” is the key <span style={{ color: r.k }}>{GLYPH.k}</span></>],
    ['vectors', <>q{i + 1} = [{hd.Q[i].map((v) => fx(v, 3)).join(', ')}], k{j + 1} = [{hd.K[j].map((v) => fx(v, 3)).join(', ')}]</>],
    ['dot product', <>s = {hd.Q[i].map((v, c) => `${fx(v, 2)}×${fx(hd.K[j][c], 2)}`).join(' + ')} = <b>{fx(hd.S[i][j], 4)}</b></>],
    ['score cell', <>S[{i + 1}, {j + 1}] highlighted in the score matrix (B)</>],
    ['scaling & mask', <>s̃ = {trace.cfg.scale ? `${fx(hd.S[i][j], 4)} / √${trace.dk} = ` : ''}{fx(hd.scaled[i][j], 4)} · mask: {trace.M[i][j] === 0 ? 'allowed' : `⊘ masked (${trace.reasons[i][j].join(' + ')})`}</>],
    ['attention weight', <>a = softmax(row {i + 1})[{j + 1}] = <b style={{ color: r.prob }}>{fx(hd.A[i][j], 4)}</b> ({pct(hd.A[i][j])})</>],
    ['contribution', <>a · v{j + 1} = {fx(hd.A[i][j], 3)} × [{hd.V[j].map((v) => fx(v, 2)).join(', ')}] = [{g.contrib[j].map((v) => fx(v, 4)).join(', ')}] of o{i + 1} = [{g.o.map((v) => fx(v, 3)).join(', ')}]</>],
  ]
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <Card title="click any heatmap cell" right={<HeadPicker />}>
          <Grid m={hd.A} role="prob" mode="prob" rowLabels={rl.map((l) => `${GLYPH.q}${l}`)} colLabels={tokens.map((t, k) => `${k + 1}·${t.slice(0, 5)}`)} masked={masked} selected={[i, j]} hlRows={[i]} hlCols={[j]} onCell={pick} cell={60} digits={3} ariaLabel="Attention heatmap driving the microscopes" testid="sync-heatmap" />
          <ColorScale color={r.prob} />
        </Card>
        <Card title={`what selecting token ${i + 1} → token ${j + 1} shows`}>
          <ol className="grid gap-1 text-[12.5px]" data-testid="sync-steps">
            {steps.map(([k, v], n) => <li key={k} className="grid grid-cols-[22px_112px_1fr] items-baseline gap-1"><span className="font-mono text-[11px] text-faint">{n + 1}.</span><span className="text-muted">{k}</span><span className="font-mono text-[12px]">{v}</span></li>)}
          </ol>
        </Card>
      </div>
      <div className="grid gap-3 @5xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <Card title="microscope A · token relationships">
          <TokenGraph trace={trace} tokens={tokens} mode="prob" />
          <div className="mt-1 grid gap-0.5 font-mono text-[11.5px]">
            {g.contrib.map((cv, k) => <div key={k} className={k === j ? 'text-fg' : 'text-faint'}>{k === j ? '▸ ' : ''}{fx(hd.A[i][k], 3)} · v{k + 1} → [{cv.map((v) => fx(v, 3)).join(', ')}]</div>)}
          </div>
        </Card>
        <Card title="microscope B · matrix calculations (row of the query, column of the key)">
          <div className="flex flex-wrap items-start gap-2">
            <Grid m={hd.Q} role="q" title={`${GLYPH.q} Q`} rowLabels={tokens.map((_, k) => String(k + 1))} hlRows={[i]} cell={40} ariaLabel="Q" onCell={(a) => select({ i: a })} />
            <Grid m={hd.K} role="k" title={`${GLYPH.k} K`} hlRows={[j]} cell={40} ariaLabel="K" onCell={(a) => select({ j: a })} />
            <Grid m={hd.S} role="score" title="S = QKᵀ" colLabels={kl} selected={[i, j]} cell={40} digits={2} ariaLabel="S" onCell={pick} testid="sync-S" />
            <Grid m={hd.logits} role="score" title={trace.cfg.scale ? 'S/√dₖ + M' : 'S + M'} colLabels={kl} selected={[i, j]} masked={masked} cell={40} digits={2} ariaLabel="Masked scaled scores" onCell={pick} testid="sync-Z" />
            <Grid m={hd.A} role="prob" mode="prob" title="A" colLabels={kl} selected={[i, j]} masked={masked} cell={40} ariaLabel="A" onCell={pick} testid="sync-A" />
            <Grid m={hd.V} role="v" title={`${GLYPH.v} V`} hlRows={[j]} cell={40} ariaLabel="V" />
            <Grid m={hd.O} role="out" title="head output" hlRows={[i]} cell={44} ariaLabel="head output" />
          </div>
        </Card>
      </div>
      <Card title="microscope C · the computational pipeline, with the selected element of each tensor">
        <PipelineView trace={trace} stage={stageById('aggregate')} />
      </Card>
    </div>
  )
}
