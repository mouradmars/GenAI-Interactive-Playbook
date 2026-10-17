/** Right panel — the mathematical inspector, synchronized with the stage, the selection and the token bar. */
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { shapeTable, type BlockTrace } from '../../engine/block'
import { Tex } from '../../components/viz'
import { f4, tn, useRoles } from './common'

/** a vector entry for TeX (no parentheses) */
const vt = (v: number, d: number) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d)
import { EXPLAIN, type Stage } from './stages'
import { useTb, type ParamKey } from './store'

const ATT_STAGES = new Set(['dot', 'scores', 'scale', 'mask', 'softmax', 'weights', 'av'])

export function Inspector({ trace, stage }: { trace: BlockTrace; stage: Stage }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel)
  const [allShapes, setAllShapes] = useState(false)
  const shapes = shapeTable(trace)
  const { T, d, H, dff } = trace.cfg, dh = d / H
  const stageShapes: Record<string, string[]> = {
    x: ['X'], ln1: ['X', 'LN₁(X)'], qkv: ['LN₁(X)', 'W_Q, W_K, W_V', 'Q, K, V', 'Q_h, K_h, V_h'], dot: ['Q_h, K_h, V_h'], scores: ['Q_h, K_h, V_h', 'S_h = Q_h K_hᵀ'],
    scale: ['S_h = Q_h K_hᵀ'], mask: ['S_h = Q_h K_hᵀ'], softmax: ['A_h'], weights: ['A_h'], av: ['A_h', 'O_h = A_h V_h'], concat: ['O_h = A_h V_h', 'Concat'],
    proj: ['Concat', 'W_O', 'Attn'], res1: ['X', 'Attn', 'H'], ln2: ['H'], ffn1: ['W₁', 'φ(Z₁)'], act: ['φ(Z₁)'], ffn2: ['φ(Z₁)', 'W₂', 'MLP'], res2: ['H', 'MLP', 'Y'], out: ['Y'],
  }
  const mine = shapes.filter((s) => (stageShapes[stage.id] ?? []).includes(s.name))
  return (
    <div className="grid min-w-0 gap-4 text-[13px]">
      <div>
        <div className="eyebrow">mathematical inspector</div>
        <h2 className="mt-1 font-display text-[19px] leading-tight" data-testid="tb-inspector-title">{stage.title}</h2>
        <div className="mt-2 overflow-x-auto scroll-thin rounded-lg border border-line bg-panel-2/60 px-3 py-2 text-[15px]"><Tex block tex={stage.tex} /></div>
      </div>
      <section>
        <div className="eyebrow mb-1">tensor shapes here</div>
        <div className="grid gap-0.5 font-mono text-[12px]">
          {mine.map((s) => <div key={s.name} className="flex justify-between gap-2 border-t border-line py-0.5 first:border-t-0"><span><Tex tex={s.tex} /></span><span className="text-cyan">[{s.shape.join(' × ')}]</span></div>)}
        </div>
      </section>
      <section className="grid gap-1" data-testid="tb-calc">
        <div className="eyebrow">calculation for the current selection</div>
        <Calc trace={trace} stage={stage} />
      </section>
      <EditBox />
      <section>
        <button type="button" className="flex items-center gap-1 text-[12px] text-cyan" onClick={() => setAllShapes(!allShapes)} aria-expanded={allShapes}>{allShapes ? <ChevronDown size={13} /> : <ChevronRight size={13} />}tensor shape tracker (whole block)</button>
        {allShapes && (
          <table className="mt-1 w-full font-mono text-[11.5px]" data-testid="shape-tracker">
            <tbody>{shapes.map((s) => <tr key={s.name} className="border-t border-line"><td className="py-0.5 pr-2"><Tex tex={s.tex} /></td><td className="pr-2 text-right text-cyan">[{s.shape.join('×')}]</td><td className="text-faint">{s.note}</td></tr>)}</tbody>
          </table>
        )}
        <p className="mt-1 text-[11px] text-faint">T = {T} tokens, d = {d}, H = {H} heads of dₖ = {dh}, d_ff = {dff}. Row-vector convention: Q = X W_Q.</p>
      </section>
      <p className="text-[11px] text-faint" style={{ color: r.tok }}>Selected: query “{tokens[sel.i]}”, key “{tokens[sel.j]}”, head {sel.h + 1}.</p>
    </div>
  )
}

function Calc({ trace, stage }: { trace: BlockTrace; stage: Stage }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), iv = useTb((s) => s.iv), params = useTb((s) => s.params)
  const { i, j, h } = sel
  if (ATT_STAGES.has(stage.id)) {
    const hd = trace.heads[h], dk = hd.q[0].length, row = hd.rows[i]
    const steps: { id: string; tex: string; color: string }[] = [
      { id: 'dot', tex: `S_{${i + 1}${j + 1}} = q_{${i + 1}}\\cdot k_{${j + 1}} = ${tn(hd.S[i][j], 4)}`, color: r.score },
      { id: 'scale', tex: `S/\\sqrt{${dk}} = ${tn(hd.scaled[i][j], 4)}`, color: r.score },
      { id: 'mask', tex: `+\\;M_{${i + 1}${j + 1}} = ${hd.mask[i][j] === 0 ? '0' : '-\\infty'}\\;\\Rightarrow\\; L = ${tn(hd.logits[i][j], 4)}`, color: r.score },
      { id: 'softmax', tex: `A_{${i + 1}${j + 1}} = \\frac{e^{${tn(row.shifted[j], 3)}}}{${tn(row.sum, 4)}} = ${tn(hd.A[i][j], 4)}`, color: r.score },
      { id: 'av', tex: `A_{${i + 1}${j + 1}}\\,v_{${j + 1}} = ${tn(hd.A[i][j], 3)}\\cdot(${hd.v[j].map((y) => vt(y, 2)).join(',\\ ')}) = (${hd.v[j].map((x) => vt(hd.A[i][j] * x, 3)).join(',\\ ')})`, color: r.v },
      { id: 'out', tex: `O_{${i + 1}} = \\textstyle\\sum_j A_{${i + 1}j} v_j = (${hd.O[i].map((y) => vt(y, 3)).join(',\\ ')})`, color: r.v },
    ]
    const activeIdx = { dot: 0, scores: 0, scale: 1, mask: 2, softmax: 3, weights: 3, av: 4 }[stage.id as 'dot'] ?? 0
    return (
      <div className="grid gap-0.5" data-testid="sync-chain">
        <div className="text-[11.5px] text-muted">query “{tokens[i]}” → key “{tokens[j]}”, head {h + 1}: from dot product to output</div>
        {steps.map((s, k) => (
          <div key={s.id} className={`overflow-x-auto scroll-thin rounded-md border-l-2 px-2 ${k === activeIdx || (stage.id === 'av' && k === 5) ? 'bg-raise' : 'opacity-75'}`} style={{ borderColor: k === activeIdx ? s.color : 'transparent' }}><Tex block tex={s.tex} /></div>
        ))}
        {!trace.heads[h].enabled && <div className="text-[11.5px] text-rose">This head is disabled downstream (its O is replaced by zeros in the concatenation).</div>}
      </div>
    )
  }
  switch (stage.id) {
    case 'x': return <><Tex block tex={`x_{${i + 1}} = (${trace.X[i].map((y) => vt(y, 2)).join(',\\ ')})`} /><Tex block tex={`\\lVert x_{${i + 1}}\\rVert = ${tn(Math.hypot(...trace.X[i]), 4)}`} /></>
    case 'ln1': case 'ln2': {
      const ln = stage.id === 'ln1' ? trace.ln1! : trace.ln2!
      return <><Tex block tex={`\\mu_{${i + 1}} = ${tn(ln.mean[i], 4)},\\quad \\sigma^2_{${i + 1}} = ${tn(ln.variance[i], 4)}`} /><Tex block tex={`\\mathrm{LN}(x_{${i + 1}}) = (${ln.out[i].map((y) => vt(y, 3)).join(',\\ ')})`} /></>
    }
    case 'qkv': {
      const c = sel.cell && ['Q', 'K', 'V'].includes(sel.cell.m) ? sel.cell : { m: 'Q', r: i, c: 0 }
      const M = c.m === 'Q' ? trace.Q : c.m === 'K' ? trace.K : trace.V
      return <Tex block tex={`${c.m}_{${c.r + 1},${c.c + 1}} = x_{n,${c.r + 1}}\\cdot W_{${c.m}}[:,${c.c + 1}] = ${tn(M[c.r][c.c], 4)}\\quad(\\text{head } ${Math.floor(c.c / (trace.cfg.d / trace.cfg.H)) + 1})`} />
    }
    case 'concat': return <Tex block tex={`\\mathrm{Concat}_{${i + 1}} = (${trace.concat[i].map((y) => vt(y, 3)).join(',\\ ')})`} />
    case 'proj': return <Tex block tex={`\\mathrm{Attn}_{${i + 1}} = (${trace.attn[i].map((y) => vt(y, 3)).join(',\\ ')})`} />
    case 'res1': return <Tex block tex={`${trace.cfg.arch === 'post' ? '\\mathrm{LN}_1(' : ''}x_{${i + 1}} + ${iv.attnScale !== 1 ? `${iv.attnScale}\\cdot` : ''}\\mathrm{Attn}_{${i + 1}}${trace.cfg.arch === 'post' ? ')' : ''} = (${trace.H[i].map((y) => vt(y, 3)).join(',\\ ')})`} />
    case 'ffn1': case 'act': case 'ffn2': {
      const u = Math.min(sel.u, trace.cfg.dff - 1)
      return <><Tex block tex={`z_{${u + 1}} = ${tn(trace.Z1[i][u], 4)},\\ \\varphi(z_{${u + 1}}) = ${tn(trace.act[i][u], 4)}${iv.units[u] === false ? '\\ (\\text{unit off})' : ''}`} /><Tex block tex={`\\mathrm{MLP}_{${i + 1}} = (${trace.mlp[i].map((y) => vt(y, 3)).join(',\\ ')})`} /><div className="text-[11px] text-faint">W₁: {params.W1.length}×{params.W1[0].length}, W₂: {params.W2.length}×{params.W2[0].length}</div></>
    }
    case 'res2': return <Tex block tex={`${trace.cfg.arch === 'post' ? '\\mathrm{LN}_2(' : ''}h_{${i + 1}} + ${iv.mlpScale !== 1 ? `${iv.mlpScale}\\cdot` : ''}\\mathrm{MLP}_{${i + 1}}${trace.cfg.arch === 'post' ? ')' : ''} = (${trace.Y[i].map((y) => vt(y, 3)).join(',\\ ')})`} />
    case 'out': return <Tex block tex={`y_{${i + 1}} = (${trace.Y[i].map((y) => vt(y, 3)).join(',\\ ')})`} />
    default: return null
  }
}

/** edit the selected input value or weight (toy mode) */
function EditBox() {
  const sel = useTb((s) => s.sel), X = useTb((s) => s.X), params = useTb((s) => s.params), setX = useTb((s) => s.setX), setParam = useTb((s) => s.setParam), tokens = useTb((s) => s.tokens)
  const resetValues = useTb((s) => s.resetValues), xEdited = useTb((s) => s.xEdited), pEdited = useTb((s) => s.paramsEdited)
  const c = sel.cell
  const isW = c && ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'W3'].includes(c.m)
  if (!c || (c.m !== 'X' && !isW)) return (xEdited || pEdited) ? <button type="button" className="btn justify-self-start" onClick={resetValues}>Reset edited values</button> : null
  const v = c.m === 'X' ? X[c.r][c.c] : params[c.m as ParamKey][c.r][c.c]
  const label = c.m === 'X' ? `X[“${tokens[c.r]}”, x${c.c + 1}]` : `${c.m.replace('W', 'W_').replace('q', 'Q').replace('k', 'K').replace('v', 'V').replace('o', 'O')}[${c.r + 1}, ${c.c + 1}]`
  const set = (x: number) => (c.m === 'X' ? setX(c.r, c.c, x) : setParam(c.m as ParamKey, c.r, c.c, x))
  return (
    <section className="grid gap-1 rounded-lg border border-amber/50 bg-amber/5 p-2" aria-label="Edit value">
      <div className="text-[12px]">edit <b className="font-mono">{label}</b> (toy mode) — everything downstream is recomputed</div>
      <div className="grid grid-cols-[1fr_84px] items-center gap-2">
        <input type="range" min={-3} max={3} step={0.1} value={Math.max(-3, Math.min(3, v))} onChange={(e) => set(+e.target.value)} aria-label={`Edit ${label}`} />
        <input type="number" step={0.1} value={+v.toFixed(4)} onChange={(e) => set(parseFloat(e.target.value))} className="rounded border border-line bg-panel-2 px-1 py-0.5 font-mono text-[12px]" aria-label={`Value of ${label}`} data-testid="tb-edit" />
      </div>
      <div className="flex items-center justify-between text-[11px] text-faint"><span>current: {f4(v)}</span>{(xEdited || pEdited) && <button type="button" className="text-cyan underline decoration-dotted" onClick={resetValues}>reset all edits</button>}</div>
    </section>
  )
}

export function ExplainPanel({ stage }: { stage: Stage; }) {
  return <ExplainInner id={stage.id} />
}
function ExplainInner({ id }: { id: Stage['id'] }) {
  const e = EXPLAIN[id]
  const rows: [string, string][] = [['What is happening?', e.what], ['Why is it necessary?', e.why], ['What is the mathematical operation?', e.math], ['What changes in the representation?', e.changes], ['What should you remember?', e.remember]]
  return (
    <section className="grid gap-2 rounded-xl border border-line bg-panel/70 p-3 @3xl:grid-cols-5" aria-label="Explanation" data-testid="explain-panel">
      {rows.map(([q, a], k) => (
        <div key={q} className={k === 4 ? 'rounded-lg border-l-2 border-amber bg-amber/5 pl-2' : ''}>
          <div className="eyebrow mb-0.5">{q}</div>
          <p className="text-[12.5px] leading-relaxed text-muted">{a}</p>
        </div>
      ))}
    </section>
  )
}
