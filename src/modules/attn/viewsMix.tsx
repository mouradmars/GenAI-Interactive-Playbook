/** Stages 7–9: the softmax microscope, the interactive heatmap and weighted value aggregation. */
import { motion, useReducedMotion } from 'framer-motion'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { useColors } from '../../sim/appState'
import { aggregationTerms, maxAbsDiff, stableSoftmax, type AttnTrace } from '../../engine/attention'
import { EditableMatrix, GLYPH, Grid, HeadPicker, RoleTag, fx, pct, tokLabel, useBaselineTrace, useRoles, useTokens } from './common'
import { TokenGraph } from './TokenGraph'
import { BaselineDiff } from './BaselineDiff'
import { attnStore, useAttn } from './store'

/* ------------------------------------------------------------------ 7 · softmax */
const SM_STEPS = ['masked, scaled logits', 'subtract the largest finite logit', 'exponentiate', 'add up the denominator', 'divide: probabilities', 'check: the row sums to 1']

export function SoftmaxView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const [step, setStep] = useState(5)
  const [offset, setOffset] = useState<number[]>(() => new Array(trace.Tk).fill(0))
  const hd = trace.heads[sel.h], i = sel.i, row = hd.rows[i]
  useEffect(() => setOffset(new Array(trace.Tk).fill(0)), [i, sel.h, trace.Tk])
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const sandbox = useMemo(() => stableSoftmax(row.logits.map((z, k) => (Number.isFinite(z) ? z + (offset[k] ?? 0) : z))), [row, offset])
  const changed = offset.some((v) => v !== 0)
  const col = (k: number) => step >= k
  return (
    <div className="grid gap-3">
      <Card title={`softmax of query ${i + 1}'s row (“${tokens[i]}”), step by step`} right={<HeadPicker />}>
        <div className="mb-2 flex flex-wrap items-center gap-1" role="group" aria-label="Softmax steps">
          <button type="button" className="btn icon-btn" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} aria-label="Previous softmax step"><ChevronLeft size={15} /></button>
          {SM_STEPS.map((s, k) => <button key={k} type="button" className={`btn !h-7 text-[11.5px] ${k === step ? 'active' : ''}`} onClick={() => setStep(k)} aria-pressed={k === step}>{k + 1}. {s}</button>)}
          <button type="button" className="btn icon-btn" onClick={() => setStep(Math.min(5, step + 1))} disabled={step === 5} aria-label="Next softmax step"><ChevronRight size={15} /></button>
        </div>
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full min-w-[520px] font-mono text-[12.5px]" data-testid="softmax-table">
            <thead><tr className="text-left text-faint">
              <th className="font-normal">key</th><th className="font-normal">zᵢⱼ</th>
              <th className={`font-normal ${col(1) ? '' : 'opacity-20'}`}>zᵢⱼ − m</th><th className={`font-normal ${col(2) ? '' : 'opacity-20'}`}>exp(·)</th>
              <th className={`font-normal ${col(4) ? '' : 'opacity-20'}`}>aᵢⱼ = exp / Σ</th><th className="font-normal" />
            </tr></thead>
            <tbody>
              {row.logits.map((z, k) => (
                <tr key={k} className={k === sel.j ? 'bg-cyan/5' : ''} onClick={() => select({ j: k })} style={{ cursor: 'pointer' }}>
                  <td><span style={{ color: r.k }}>{GLYPH.k}</span> {rl[k]}</td>
                  <td style={{ color: row.allowed[k] ? r.score : r.mask }}>{row.allowed[k] ? fx(z, 4) : '−∞ ⊘'}</td>
                  <td className={col(1) ? '' : 'opacity-20'}>{row.allowed[k] ? fx(row.shifted[k], 4) : '−∞'}</td>
                  <td className={col(2) ? '' : 'opacity-20'}>{fx(row.exps[k], 4)}</td>
                  <td className={col(4) ? '' : 'opacity-20'} style={{ color: r.prob }}>{fx(row.p[k], 4)}</td>
                  <td className="w-[30%]"><div className="h-3 rounded bg-panel-2"><motion.div className="h-3 rounded" style={{ background: r.prob }} animate={{ width: col(4) ? `${row.p[k] * 100}%` : '0%' }} transition={{ duration: 0.4 }} /></div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2 grid gap-0.5 font-mono text-[12px] text-muted">
          {col(1) && <div>m = max of the permitted logits = <b className="text-fg">{row.fullyMasked ? '— (no permitted key)' : fx(row.max, 4)}</b> (subtracting it changes nothing mathematically but prevents overflow)</div>}
          {col(3) && <div>Σ = {row.exps.filter((_, k) => row.allowed[k]).map((e) => fx(e, 4)).join(' + ') || '0'} = <b className="text-fg">{fx(row.sum, 4)}</b></div>}
          {col(5) && <div data-testid="row-sum">Σⱼ aᵢⱼ = {row.p.map((p) => fx(p, 4)).join(' + ')} = <b className={row.fullyMasked ? 'text-rose' : 'text-teal'}>{fx(row.p.reduce((a, b) => a + b, 0), 6)}</b>{row.fullyMasked ? ' (fully masked row: no distribution)' : ' ✓'}</div>}
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="the same row in the heatmap · every valid row sums to 1">
          <Grid m={hd.A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} masked={trace.M.map((x) => x.map((v) => v === -Infinity))} hlRows={[i]} selected={[i, sel.j]} onCell={(a, b) => select({ i: a, j: b })} ariaLabel="Attention weights" cell={48} testid="g-A-sm" />
          <div className="mt-1 font-mono text-[11.5px] text-muted">row sums: {hd.A.map((rr, k) => <span key={k} className={hd.rows[k].fullyMasked ? 'text-rose' : 'text-teal'}>{fx(rr.reduce((a, b) => a + b, 0), 6)}{k < hd.A.length - 1 ? ' · ' : ''}</span>)}</div>
          <p className="mt-2 text-[12px] text-muted">Softmax runs across the keys of each query separately: every query needs its own distribution over where to read from. Normalizing the whole matrix at once would make different queries compete for one shared budget.</p>
        </Card>
        <Card title="experiment · nudge the logits of this row and watch the competition" right={changed ? <button type="button" className="btn !h-7" onClick={() => setOffset(new Array(trace.Tk).fill(0))}><RotateCcw size={13} />reset</button> : null}>
          <div className="grid gap-1">
            {row.logits.map((_, k) => {
              const d = sandbox.p[k] - row.p[k]
              return (
                <div key={k} className="grid grid-cols-[70px_auto_1fr_64px_64px] items-center gap-1.5 font-mono text-[12px]">
                  <span className="truncate">{rl[k]}</span>
                  <span className="flex gap-0.5">
                    <button type="button" className="btn icon-btn !h-6 !w-6" disabled={!row.allowed[k]} onClick={() => setOffset(offset.map((o, kk) => (kk === k ? o - 0.5 : o)))} aria-label={`Decrease logit ${k + 1}`}>−</button>
                    <button type="button" className="btn icon-btn !h-6 !w-6" disabled={!row.allowed[k]} onClick={() => setOffset(offset.map((o, kk) => (kk === k ? o + 0.5 : o)))} aria-label={`Increase logit ${k + 1}`} data-testid={`nudge-up-${k}`}>+</button>
                  </span>
                  <div className="h-3 rounded bg-panel-2"><div className="h-3 rounded transition-[width]" style={{ width: `${sandbox.p[k] * 100}%`, background: r.prob }} /></div>
                  <span className="text-right">{row.allowed[k] ? pct(sandbox.p[k]) : '⊘'}</span>
                  <span className={`text-right ${Math.abs(d) < 5e-4 ? 'text-faint' : d > 0 ? 'text-teal' : 'text-rose'}`} data-testid={`nudge-delta-${k}`}>{row.allowed[k] ? `${d >= 0 ? '+' : '−'}${(Math.abs(d) * 100).toFixed(1)}` : ''}</span>
                </div>
              )
            })}
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            <button type="button" className="btn !h-7 text-[11.5px]" onClick={() => setOffset(row.logits.map((z) => (Number.isFinite(z) ? z * 4 : 0)))}>saturate (×5 all logits)</button>
            <button type="button" className="btn !h-7 text-[11.5px]" onClick={() => setOffset(row.logits.map(() => 2))}>add 2 to every logit</button>
          </div>
          <p className="mt-1 text-[11.5px] text-faint">Offsets: [{offset.map((o) => fx(o, 1)).join(', ')}]. Raising one logit takes probability from all the others; adding the same constant to every logit changes nothing. This sandbox does not change the model.</p>
        </Card>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 8 · heatmap */
export function ColorScale({ color, label = 'attention probability' }: { color: string; label?: string }) {
  const c = useColors()
  return (
    <div className="flex items-center gap-2 font-mono text-[10.5px] text-faint" aria-label={`colour scale: ${label} from 0 to 1`}>
      0 <div className="h-2.5 w-36 rounded" style={{ background: `linear-gradient(90deg, ${c.panel2}, ${color})` }} /> 1 · {label} · ⊘ hatched = masked
    </div>
  )
}
export function HeatmapView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const base = useBaselineTrace()
  const hd = trace.heads[sel.h], i = sel.i, j = sel.j
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const masked = trace.M.map((x) => x.map((v) => v === -Infinity))
  const bh = base && base.Tq === trace.Tq && base.cfg.H === trace.cfg.H ? base.heads[sel.h] : null
  const reasons = trace.reasons[i][j]
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <Card title={<span>attention heatmap <RoleTag role="prob">A</RoleTag> · head {sel.h + 1} · rows = queries, columns = keys</span>} right={<HeadPicker />}>
          <Grid m={hd.A} role="prob" mode="prob" rowLabels={rl.map((l) => `${GLYPH.q}${l}`)} colLabels={tokens.map((t, k) => `${k + 1}·${t.slice(0, 5)}`)} masked={masked} selected={[i, j]} hlRows={[i]} hlCols={[j]}
            onCell={(a, b) => select({ i: a, j: b, cell: null })} ariaLabel="Attention heatmap" cell={66} digits={3} testid="heatmap" />
          <ColorScale color={r.prob} />
        </Card>
        <Card title="microscope A · the same relationship as connections">
          <TokenGraph trace={trace} tokens={tokens} mode="prob" />
        </Card>
      </div>
      <Card title={`query ${i + 1} “${tokens[i]}” → key ${j + 1} “${tokens[j]}”: where this probability comes from`}>
        <div className="grid gap-2 font-mono text-[12.5px] @3xl:grid-cols-4" data-testid="cell-detail">
          <div className="rounded-md border border-line p-2"><div className="text-[10.5px] text-faint">dot product</div><span style={{ color: r.score }}>s = q{i + 1}·k{j + 1} = {fx(hd.S[i][j], 4)}</span></div>
          <div className="rounded-md border border-line p-2"><div className="text-[10.5px] text-faint">scaled score</div><span style={{ color: r.score }}>s̃ = {trace.cfg.scale ? `s/√${trace.dk} = ` : ''}{fx(hd.scaled[i][j], 4)}</span></div>
          <div className="rounded-md border border-line p-2"><div className="text-[10.5px] text-faint">mask</div><span style={{ color: reasons.length ? r.mask : r.prob }}>{reasons.length ? `⊘ masked (${reasons.join(' + ')}) → −∞` : '✓ allowed (M = 0)'}</span></div>
          <div className="rounded-md border border-line p-2"><div className="text-[10.5px] text-faint">probability</div><span style={{ color: r.prob }}>a = {fx(hd.A[i][j], 4)} ({pct(hd.A[i][j])})</span></div>
        </div>
        <p className="mt-2 text-[11.5px] text-muted">Rows sum to 1, columns need not. A bright cell means this query reads much of its value mixture from this key in this head; it is not, by itself, an explanation of the model's reasoning.</p>
      </Card>
      <Card title="before / after a parameter change">
        {bh ? (
          <div className="flex flex-wrap items-start gap-3">
            <Grid m={bh.A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="before (saved state)" masked={base!.M.map((x) => x.map((v) => v === -Infinity))} cell={46} ariaLabel="Attention before" />
            <Grid m={hd.A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="after (current)" masked={masked} cell={46} ariaLabel="Attention after" />
            <Grid m={hd.A.map((rr, a) => rr.map((v, b) => v - bh.A[a][b]))} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title={`difference A_after − A_before (max |Δ| = ${fx(maxAbsDiff(hd.A, bh.A), 4)})`} cell={46} digits={3} ariaLabel="Attention difference" testid="g-dA" />
          </div>
        ) : <p className="text-[12.5px] text-muted">Save the current state, then change a weight (stage 3), the input (stage 2), the mask (stage 6) or scaling (stage 5) to compare heatmaps side by side.</p>}
        <BaselineDiff trace={trace} />
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 9 · weighted value aggregation */
export function AggregateView({ trace }: { trace: AttnTrace }) {
  const r = useRoles(), reduce = useReducedMotion()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), blocked = useAttn((s) => s.blocked)
  const st = attnStore.getState()
  const hd = trace.heads[sel.h], i = sel.i, dk = trace.dk
  const g = aggregationTerms(hd, i)
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const [nudge, setNudge] = useState<number[]>(() => new Array(trace.Tk).fill(0))
  useEffect(() => setNudge(new Array(trace.Tk).fill(0)), [i, sel.h, trace.Tk])
  const what = useMemo(() => {
    const sm = stableSoftmax(hd.rows[i].logits.map((z, k) => (Number.isFinite(z) ? z + nudge[k] : z)))
    const o = hd.V[0].map((_, c) => sm.p.reduce((s, a, k) => s + a * hd.V[k][c], 0))
    return { p: sm.p, o }
  }, [hd, i, nudge])
  return (
    <div className="grid gap-3">
      <Card title={`o${i + 1} = Σⱼ a${i + 1}ⱼ vⱼ · query ${i + 1} “${tokens[i]}” · head ${sel.h + 1}`} right={<HeadPicker />}>
        <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          {dk === 2 ? <VectorPlane trace={trace} /> : <p className="text-[12px] text-muted">With d_k = {dk} the vectors cannot be drawn in a plane; the bars on the right show every dimension.</p>}
          <div className="grid content-start gap-2">
            {g.contrib.map((cv, j) => {
              const allowed = trace.M[i][j] === 0
              return (
                <motion.div key={`${j}-${g.a[j]}`} initial={reduce ? false : { opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: j * 0.12 }}
                  className={`grid grid-cols-[84px_70px_1fr] items-center gap-2 rounded-md border px-2 py-1 font-mono text-[12px] ${j === sel.j ? 'border-cyan' : 'border-line'}`} onClick={() => select({ j })} style={{ cursor: 'pointer', opacity: allowed ? 1 : 0.45 }}>
                  <span><span style={{ color: r.v }}>{GLYPH.v}</span> v{j + 1} <span className="text-faint">{tokens[j].slice(0, 5)}</span></span>
                  <span style={{ color: allowed ? r.prob : r.mask }}>{allowed ? `× ${fx(g.a[j], 3)}` : '⊘ × 0'}</span>
                  <span className="text-muted">[{hd.V[j].map((v) => fx(v, 2)).join(', ')}] → <span className="text-fg">[{cv.map((v) => fx(v, 3)).join(', ')}]</span></span>
                </motion.div>
              )
            })}
            <div className="rounded-md border px-2 py-1.5 font-mono text-[13px]" style={{ borderColor: r.out }} data-testid="agg-output">
              <span style={{ color: r.out }}>{GLYPH.out} o{i + 1}</span> = [{g.o.map((v) => fx(v, 4)).join(', ')}]
            </div>
          </div>
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="numerical microscope · one output dimension at a time">
          <div className="grid gap-1.5 font-mono text-[12px]" data-testid="agg-dims">
            {g.o.map((o, c) => (
              <div key={c}>
                <span style={{ color: r.out }}>o{i + 1},{c + 1}</span> = {g.a.map((a, j) => `${fx(a, 3)}×${fx(hd.V[j][c], 2)}`).join(' + ')}
                <div className="pl-4 text-muted">= {g.a.map((a, j) => fx(a * hd.V[j][c], 4)).join(' + ')} = <b className="text-fg">{fx(o, 4)}</b></div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[12px] text-muted">The output is a weighted combination of all permitted value vectors, not a copy of one token's value: the weights are non-negative and sum to 1, so o lies inside the region spanned by the values.</p>
        </Card>
        <Card title="experiments">
          <div className="grid gap-2 text-[12px]">
            <div>
              <b>Disable a permitted token</b> for this query (a custom knock-out in the mask). The remaining weights renormalize because the mask acts before softmax.
              <div className="mt-1 flex flex-wrap gap-1">
                {tokens.map((t, j) => <button key={j} type="button" className="btn !h-7 !px-2 text-[11.5px]" aria-pressed={!!blocked?.[i]?.[j]} onClick={() => st.toggleBlocked(i, j)} data-testid={`knock-${j}`}>{blocked?.[i]?.[j] ? '⊘ ' : ''}{j + 1}·{t}</button>)}
              </div>
            </div>
            <div>
              <b>Change the weights through the logits</b> (what-if, does not change the model): nudge the logit of a key and see the new weights and output.
              <div className="mt-1 grid gap-0.5">
                {tokens.map((t, j) => (
                  <div key={j} className="flex items-center gap-1 font-mono text-[11.5px]">
                    <span className="w-16 truncate">{j + 1}·{t}</span>
                    <button type="button" className="btn icon-btn !h-6 !w-6" disabled={trace.M[i][j] !== 0} onClick={() => setNudge(nudge.map((v, k) => (k === j ? v - 1 : v)))} aria-label={`Lower logit of key ${j + 1}`}>−</button>
                    <button type="button" className="btn icon-btn !h-6 !w-6" disabled={trace.M[i][j] !== 0} onClick={() => setNudge(nudge.map((v, k) => (k === j ? v + 1 : v)))} aria-label={`Raise logit of key ${j + 1}`}>+</button>
                    <span className="text-faint">{nudge[j] ? `${nudge[j] > 0 ? '+' : ''}${nudge[j]}` : ''}</span>
                    <span className="ml-auto">{fx(hd.A[i][j], 3)} → <span style={{ color: r.prob }}>{fx(what.p[j], 3)}</span></span>
                  </div>
                ))}
              </div>
              <div className="mt-1 font-mono text-[11.5px]">o{i + 1}: [{g.o.map((v) => fx(v, 3)).join(', ')}] → <span style={{ color: r.out }}>[{what.o.map((v) => fx(v, 3)).join(', ')}]</span></div>
            </div>
            <div>
              <b>Modify the value vectors</b> through head {sel.h + 1}'s W_V (Q and K stay fixed, so the weights cannot change):
              <EditableMatrix m={hd.Wv} onSet={(rr, c, v) => st.setW('Wv', rr, sel.h * dk + c, v)} rowLabels={['f1', 'f2', 'f3', 'f4']} colLabels={Array.from({ length: dk }, (_, k) => `c${k + 1}`)} role="v" label={`W_V head ${sel.h + 1}`} testid="agg-edit-Wv" />
            </div>
            <BaselineDiff trace={trace} compact />
          </div>
        </Card>
      </div>
      <Grid m={hd.V} role="v" rowLabels={rl} colLabels={Array.from({ length: dk }, (_, k) => `c${k + 1}`)} title={`V${sel.h + 1} (values of head ${sel.h + 1})`} hlRows={[sel.j]} ariaLabel="Value matrix" cell={50} />
    </div>
  )
}

function VectorPlane({ trace }: { trace: AttnTrace }) {
  const r = useRoles(), c = useColors(), reduce = useReducedMotion()
  const sel = useAttn((s) => s.sel)
  const hd = trace.heads[sel.h], i = sel.i
  const g = aggregationTerms(hd, i)
  const pts = [...hd.V, g.o, [0, 0]]
  const lim = Math.max(0.5, ...pts.flat().map(Math.abs)) * 1.15
  const S = 300, cx = S / 2, cy = S / 2, k = (S / 2 - 16) / lim
  const P = (v: number[]) => [cx + v[0] * k, cy - v[1] * k] as const
  // tip-to-tail chain of weighted contributions
  const starts = g.contrib.map((_, j) => g.contrib.slice(0, j).reduce((s, cv) => [s[0] + cv[0], s[1] + cv[1]], [0, 0]))
  const chain = g.contrib.map((cv, j) => [starts[j], [starts[j][0] + cv[0], starts[j][1] + cv[1]]] as [number[], number[]])
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto block w-full max-w-[340px]" role="img" aria-label={`Value vectors and output o${i + 1} in the plane of head ${sel.h + 1}`} data-testid="vector-plane">
      <defs>
        <marker id="vp-v" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={r.v} /></marker>
        <marker id="vp-o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={r.out} /></marker>
      </defs>
      <line x1={0} x2={S} y1={cy} y2={cy} stroke={c.line} /><line x1={cx} x2={cx} y1={0} y2={S} stroke={c.line} />
      <text x={S - 4} y={cy - 4} textAnchor="end" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">c1</text>
      <text x={cx + 4} y={10} fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">c2</text>
      {hd.V.map((v, j) => {
        const [x, y] = P(v), allowed = trace.M[i][j] === 0
        return (
          <g key={j} opacity={allowed ? 1 : 0.3}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={r.v} strokeWidth={1.4} strokeDasharray={allowed ? undefined : '3 3'} markerEnd="url(#vp-v)" opacity={0.65} />
            <text x={x + 5} y={y - 4} fontSize={10.5} fill={r.v} fontFamily="var(--font-mono)">v{j + 1}{allowed ? '' : ' ⊘'}</text>
          </g>
        )
      })}
      {chain.map(([a, b], j) => {
        const [x1, y1] = P(a), [x2, y2] = P(b)
        if (g.a[j] === 0) return null
        return <motion.line key={`${j}-${g.a[j]}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={r.prob} strokeWidth={2.6} strokeLinecap="round"
          initial={reduce ? false : { pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.9 }} transition={{ delay: 0.2 + j * 0.35, duration: 0.35 }} />
      })}
      {(() => { const [x, y] = P(g.o); return (
        <motion.g initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + chain.length * 0.35 }}>
          <line x1={cx} y1={cy} x2={x} y2={y} stroke={r.out} strokeWidth={2.8} markerEnd="url(#vp-o)" />
          <text x={x + 6} y={y + 12} fontSize={11.5} fill={r.out} fontFamily="var(--font-mono)">o{i + 1}</text>
        </motion.g>
      ) })()}
      <text x={6} y={S - 6} fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">teal: a·v contributions, tip to tail</text>
    </svg>
  )
}
