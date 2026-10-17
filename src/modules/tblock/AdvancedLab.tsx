/** Advanced experiments: precision, KV cache, cost, attention vs MLP dependency, residual stream contributions. */
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { blockCost, dependencyMap, kvCacheRun, runBlock, vnorm, type Precision } from '../../engine/block'
import { Card } from '../../components/viz'
import { MatrixView, f3, useRoles, useTrace } from './common'
import { cfgOf, useTb } from './store'

type Sub = 'precision' | 'kv' | 'cost' | 'dependency' | 'residual'
const SUBS: [Sub, string][] = [['dependency', 'Attention vs MLP'], ['residual', 'Residual stream'], ['kv', 'KV cache'], ['cost', 'Cost vs length'], ['precision', 'Numerical precision']]

export function AdvancedLab() {
  const [sub, setSub] = useState<Sub>('dependency')
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap rounded-lg border border-line p-0.5" role="tablist" aria-label="Advanced experiments">
        {SUBS.map(([id, l]) => <button key={id} type="button" role="tab" aria-selected={sub === id} onClick={() => setSub(id)} className={`rounded-md px-2.5 py-1 text-[12.5px] ${sub === id ? 'bg-raise text-cyan' : 'text-muted'}`}>{l}</button>)}
      </div>
      {sub === 'dependency' && <Dependency />}
      {sub === 'residual' && <ResidualStream />}
      {sub === 'kv' && <KVCache />}
      {sub === 'cost' && <Cost />}
      {sub === 'precision' && <PrecisionLab />}
    </div>
  )
}

function Dependency() {
  const r = useRoles()
  const s = useTb((x) => x), trace = useTrace()
  const dm = useMemo(() => dependencyMap(trace.cfg, s.params, trace.attnIn), [trace, s.params])
  const lbl = s.tokens.map((_, k) => `in ${k + 1}`)
  return (
    <>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="attention: which inputs move each output?"><MatrixView m={dm.attn} rowLabels={s.tokens.map((t) => `out ${t}`)} colLabels={lbl} color={r.score} mode="prob" maxAbs={Math.max(...dm.attn.flat())} ariaLabel="Attention dependency" testid="dep-attn" cell={50} /></Card>
        <Card title="MLP: which inputs move each output?"><MatrixView m={dm.mlp} rowLabels={s.tokens.map((t) => `out ${t}`)} colLabels={lbl} color={r.ffn} mode="prob" maxAbs={Math.max(...dm.mlp.flat())} ariaLabel="MLP dependency" testid="dep-mlp" cell={50} /></Card>
      </div>
      <p className="text-[12.5px] text-muted">Measured, not drawn: the input row of token j is nudged by a tiny vector, each sublayer is recomputed, and cell (i, j) shows how far output i moved (per unit of nudge). Attention spreads information across positions ({trace.cfg.causal ? 'here only from earlier positions: a lower-triangular map' : 'here in both directions'}); the MLP's map is exactly diagonal, because it transforms each position on its own with shared weights.</p>
    </>
  )
}

function ResidualStream() {
  const c = useColors(), r = useRoles()
  const s = useTb((x) => x), t = useTrace()
  const rows = s.tokens.map((tok, i) => ({ tok, x: vnorm(t.X[i]), a: vnorm(t.attn[i].map((v) => v * s.iv.attnScale)), m: vnorm(t.mlp[i].map((v) => v * s.iv.mlpScale)), y: vnorm(t.Y[i]) }))
  const max = Math.max(...rows.flatMap((q) => [q.x, q.a, q.m, q.y]))
  return (
    <Card title="residual stream: size of the input and of each sublayer's contribution, per token">
      <div className="grid gap-2" data-testid="residual-stream">
        {rows.map((q) => (
          <div key={q.tok} className="grid grid-cols-[80px_minmax(0,1fr)] items-center gap-2 text-[11.5px]">
            <span className="truncate font-mono">{q.tok}</span>
            <div className="grid gap-0.5">
              {([['‖X‖', q.x, r.tok], ['‖attention update‖', q.a, r.score], ['‖MLP update‖', q.m, r.ffn], ['‖Y‖', q.y, r.res]] as [string, number, string][]).map(([l, v, col]) => (
                <div key={l} className="flex items-center gap-2"><div className="h-2.5 rounded" style={{ width: `${(v / max) * 70}%`, background: col }} /><span className="font-mono text-faint">{l} {f3(v)}</span></div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[12px] text-muted" style={{ color: c.muted }}>{t.cfg.arch === 'pre' ? 'Pre-LN: Y = X + attention update + MLP update, so the stream is the running sum of every update written into it.' : 'Post-LN: each addition is followed by a LayerNorm, so the stream is rescaled after every sublayer.'} Change the scales on the residual stages to see the bars respond.</p>
    </Card>
  )
}

function KVCache() {
  const c = useColors(), r = useRoles()
  const s = useTb((x) => x)
  const cfg = { ...cfgOf(s), arch: 'pre' as const, causal: true }
  const steps = useMemo(() => kvCacheRun(cfg, s.params, s.X), [s.params, s.X, s.T, s.d, s.H, s.dff]) // eslint-disable-line react-hooks/exhaustive-deps
  const full = useMemo(() => runBlock(cfg, s.params, s.X), [s.params, s.X, s.T, s.d, s.H, s.dff]) // eslint-disable-line react-hooks/exhaustive-deps
  const [t, setT] = useState(steps.length - 1)
  const st = steps[Math.min(t, steps.length - 1)], h = Math.min(s.sel.h, s.H - 1)
  const err = Math.max(...st.out.map((v, k) => Math.abs(v - full.attn[st.t][k])))
  const totNew = steps.slice(0, t + 1).reduce((a, q) => a + q.newMacs, 0), totFull = steps.slice(0, t + 1).reduce((a, q) => a + q.fullMacs, 0)
  return (
    <div className="grid gap-3">
      <Card title="autoregressive decoding with a KV cache (decoder, causal)">
        <label className="flex items-center gap-2 text-[12.5px] text-muted">generation step <span className="w-6 font-mono text-fg">{t + 1}</span><input type="range" min={0} max={steps.length - 1} value={t} onChange={(e) => setT(+e.target.value)} className="flex-1" aria-label="Decoding step" /></label>
        <div className="mt-2 flex flex-wrap gap-1">{s.tokens.map((tok, j) => <span key={j} className={`rounded-md border px-2 py-0.5 font-mono text-[12px] ${j === st.t ? 'border-cyan text-cyan' : j < st.t ? 'border-line-strong' : 'border-line text-faint opacity-40'}`}>{tok}</span>)}</div>
        <p className="mt-2 text-[12.5px] text-muted">At step {st.t + 1} only the new token “{s.tokens[st.t]}” is projected. Its key and value are appended to the cache; its query attends over all {st.t + 1} cached keys. Earlier tokens are never recomputed.</p>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-3">
        <Card title={<span style={{ color: r.k }}>K cache, head {h + 1} [{st.t + 1} × {s.d / s.H}]</span>}><MatrixView m={st.cacheK[h]} rowLabels={s.tokens.slice(0, st.t + 1)} colLabels={st.cacheK[h][0].map((_, k) => `c${k + 1}`)} color={r.k} ariaLabel="Key cache" testid="kv-K" cell={44} /></Card>
        <Card title={<span style={{ color: r.v }}>V cache, head {h + 1}</span>}><MatrixView m={st.cacheV[h]} rowLabels={s.tokens.slice(0, st.t + 1)} colLabels={st.cacheV[h][0].map((_, k) => `c${k + 1}`)} color={r.v} ariaLabel="Value cache" cell={44} /></Card>
        <Card title={<span style={{ color: r.score }}>weights of the new query</span>}>
          <MatrixView m={[st.weights[h]]} rowLabels={[s.tokens[st.t]]} colLabels={s.tokens.slice(0, st.t + 1).map((x) => x.slice(0, 5))} color={r.score} mode="prob" maxAbs={1} ariaLabel="New query weights" cell={44} />
          <p className="mt-1 font-mono text-[11.5px]" data-testid="kv-check">max |cached − full| = {err.toExponential(1)} ✓</p>
        </Card>
      </div>
      <Card title="multiply-accumulates in the attention sublayer, cumulative up to this step">
        <div className="grid gap-1 text-[12px]">
          {[['with KV cache', totNew, c.teal], ['recomputing the whole prefix every step', totFull, c.rose]].map(([l, v, col]) => (
            <div key={l as string} className="grid grid-cols-[230px_1fr_90px] items-center gap-2"><span className="text-muted">{l}</span><div className="h-3 rounded bg-panel-2"><div className="h-3 rounded" style={{ width: `${((v as number) / totFull) * 100}%`, background: col as string }} /></div><span className="text-right font-mono">{(v as number).toLocaleString()}</span></div>
          ))}
        </div>
        <p className="mt-1 text-[11.5px] text-muted">Full-sequence processing (training, or reading a prompt) computes all rows at once. Incremental generation with a cache does the same arithmetic as the last row only, at the price of storing 2 × T × d numbers per block.</p>
      </Card>
    </div>
  )
}

function Cost() {
  const c = useColors(), r = useRoles()
  const [logT, setLogT] = useState(9)
  const [d, setD] = useState(768)
  const H = 12, dff = 4 * d, T = 2 ** logT
  const cost = blockCost(T, d, H, dff)
  const parts: [string, number, string][] = [['Q, K, V projections', cost.qkv, r.q], ['scores QKᵀ', cost.scores, r.score], ['A·V', cost.av, r.v], ['output projection', cost.proj, r.k], ['MLP', cost.mlp, r.ffn]]
  const curve = Array.from({ length: 12 }, (_, k) => { const tt = 2 ** (k + 4), q = blockCost(tt, d, H, dff); return { tt, attn: (q.scores + q.av) / q.total } })
  return (
    <div className="grid gap-3">
      <Card title="one block · how sequence length changes the work">
        <div className="grid gap-2 text-[12.5px] text-muted @3xl:grid-cols-2">
          <label className="flex items-center gap-2">sequence length T = <span className="w-14 font-mono text-fg">{T.toLocaleString()}</span><input type="range" min={4} max={15} value={logT} onChange={(e) => setLogT(+e.target.value)} className="flex-1" aria-label="Sequence length (log2)" /></label>
          <label className="flex items-center gap-2">d = <span className="w-12 font-mono text-fg">{d}</span><select className="rounded border border-line bg-panel-2 px-1" value={d} onChange={(e) => setD(+e.target.value)} aria-label="Model dimension for cost"><option value={256}>256</option><option value={768}>768</option><option value={4096}>4096</option></select><span className="text-faint">H = 12, d_ff = 4d</span></label>
        </div>
        <div className="mt-2 grid gap-1" data-testid="cost-bars">
          {parts.map(([l, v, col]) => <div key={l} className="grid grid-cols-[150px_1fr_110px] items-center gap-2 text-[12px]"><span className="text-muted">{l}</span><div className="h-3 rounded bg-panel-2"><div className="h-3 rounded" style={{ width: `${(v / cost.total) * 100}%`, background: col }} /></div><span className="text-right font-mono">{v.toExponential(2)}</span></div>)}
        </div>
        <p className="mt-2 font-mono text-[12px]">attention matrices: {H} × {T.toLocaleString()} × {T.toLocaleString()} = {cost.attnMatrixEntries.toExponential(2)} entries · KV cache: {cost.kvCacheEntries.toLocaleString()} numbers per block</p>
      </Card>
      <Card title="share of the block's multiply-adds spent on the T × T part (scores and A·V)">
        <svg viewBox="0 0 460 170" className="block w-full max-w-[560px]" role="img" aria-label="Quadratic share">
          {[0, 0.25, 0.5, 0.75, 1].map((v) => <g key={v}><line x1={40} x2={450} y1={150 - v * 130} y2={150 - v * 130} stroke={c.line} strokeDasharray="2 3" /><text x={34} y={153 - v * 130} textAnchor="end" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">{v * 100}%</text></g>)}
          <path d={curve.map((q, k) => `${k ? 'L' : 'M'}${40 + (k / 11) * 410},${150 - q.attn * 130}`).join('')} fill="none" stroke={r.score} strokeWidth={2} />
          {curve.map((q, k) => k % 2 === 0 && <text key={k} x={40 + (k / 11) * 410} y={166} textAnchor="middle" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">{q.tt >= 1024 ? `${q.tt / 1024}k` : q.tt}</text>)}
        </svg>
        <p className="text-[12px] text-muted">Projections and the MLP grow linearly with T; scores and A·V grow with T². For short sequences the MLP dominates; past T ≈ {(6 * d).toLocaleString()} (for these sizes) the quadratic attention part takes over.</p>
      </Card>
    </div>
  )
}

function PrecisionLab() {
  const r = useRoles()
  const s = useTb((x) => x)
  const cfg = cfgOf(s)
  const ref = useMemo(() => runBlock(cfg, s.params, s.X), [s, cfg.T]) // eslint-disable-line react-hooks/exhaustive-deps
  const precs: Precision[] = ['fp32', 'bf16', 'fp16']
  const res = useMemo(() => precs.map((p) => {
    const t = runBlock(cfg, s.params, s.X, { ...s.iv, precision: p, heads: s.iv.heads.map(() => true), units: s.iv.units.map(() => true), attnScale: 1, mlpScale: 1 })
    const md = (a: number[][], b: number[][]) => Math.max(...a.flatMap((row, i) => row.map((v, j) => Math.abs(v - b[i][j]))))
    return { p, Y: t.Y, errQ: md(t.Q, ref.Q), errA: Math.max(...t.heads.map((h, k) => md(h.A, ref.heads[k].A))), errY: md(t.Y, ref.Y) }
  }), [ref]) // eslint-disable-line react-hooks/exhaustive-deps
  const [sel, setSel] = useState<Precision>('bf16')
  const cur = res.find((x) => x.p === sel)!
  return (
    <div className="grid gap-3">
      <Card title="the same block, every intermediate tensor rounded to a lower precision (controlled emulation)">
        <table className="font-mono text-[12px]" data-testid="precision-table">
          <thead className="text-faint"><tr><th className="px-2 text-left">format</th><th className="px-2 text-right">mantissa bits</th><th className="px-2 text-right">max |ΔQ|</th><th className="px-2 text-right">max |ΔA|</th><th className="px-2 text-right">max |ΔY|</th></tr></thead>
          <tbody>
            <tr className="border-t border-line"><td className="px-2">fp64 (reference)</td><td className="px-2 text-right">52</td><td className="px-2 text-right">0</td><td className="px-2 text-right">0</td><td className="px-2 text-right">0</td></tr>
            {res.map((x) => <tr key={x.p} className={`cursor-pointer border-t border-line ${x.p === sel ? 'text-cyan' : ''}`} onClick={() => setSel(x.p)}><td className="px-2">{x.p}</td><td className="px-2 text-right">{x.p === 'fp32' ? 23 : x.p === 'fp16' ? 10 : 7}</td><td className="px-2 text-right">{x.errQ.toExponential(1)}</td><td className="px-2 text-right">{x.errA.toExponential(1)}</td><td className="px-2 text-right">{x.errY.toExponential(1)}</td></tr>)}
          </tbody>
        </table>
        <p className="mt-1 text-[12px] text-muted">bf16 keeps fp32's range (8 exponent bits) with only 7 mantissa bits; fp16 has 10 mantissa bits but a much smaller range (max ≈ 65,504). Real mixed-precision kernels accumulate sums in fp32, so their errors are smaller than this every-step rounding.</p>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="Y in fp64"><MatrixView m={ref.Y} rowLabels={s.tokens} colLabels={ref.Y[0].map((_, k) => `x${k + 1}`)} color={r.tok} ariaLabel="Y fp64" digits={4} cell={60} /></Card>
        <Card title={`Y in ${sel}`}><MatrixView m={cur.Y} rowLabels={s.tokens} colLabels={cur.Y[0].map((_, k) => `x${k + 1}`)} color={r.tok} ariaLabel={`Y ${sel}`} digits={4} cell={60} /></Card>
      </div>
    </div>
  )
}
