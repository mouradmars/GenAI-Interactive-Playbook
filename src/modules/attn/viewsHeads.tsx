/** Stages 10–13: the multi-head explorer, concatenation, the output projection and the final output. */
import { motion, useReducedMotion } from 'framer-motion'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { useColors } from '../../sim/appState'
import { headSimilarity, matmul, matmulCell, type AttnTrace } from '../../engine/attention'
import { EditableMatrix, GLYPH, Grid, HeadPicker, RoleTag, Seg, fx, headColors, pct, tokLabel, useRoles, useTokens } from './common'
import { TermSum } from './viewsScore'
import { BaselineDiff } from './BaselineDiff'
import { PipelineView } from './PipelineView'
import { stageById } from './stages'
import { D_MODEL, attnStore, useAttn } from './store'

const feat = Array.from({ length: D_MODEL }, (_, k) => `f${k + 1}`)
const masks = (t: AttnTrace) => t.M.map((x) => x.map((v) => v === -Infinity))

/** per-head contribution to the output: head_h W_O^{(h)} (zero for a disabled head) */
export function headContributions(t: AttnTrace, Wo: number[][]) {
  return t.heads.map((h) => (h.enabled ? matmul(h.O, Wo.slice(h.h * t.dk, (h.h + 1) * t.dk)) : h.O.map(() => new Array(t.dModel).fill(0))))
}

/* ------------------------------------------------------------------ 10 · heads */
type HeadsMode = 'chambers' | 'compare' | 'overlay'
export function HeadsView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), H = useAttn((s) => s.H), headsOn = useAttn((s) => s.headsOn), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), params = useAttn((s) => s.params)
  const st = attnStore.getState()
  const [mode, setMode] = useState<HeadsMode>('chambers')
  const [pair, setPair] = useState<[number, number]>([0, 1])
  const a = Math.min(pair[0], H - 1), b = Math.min(pair[1], H - 1)
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const hc = headColors(r)
  const contrib = useMemo(() => headContributions(trace, params.Wo), [trace, params.Wo])
  const sim = H > 1 ? headSimilarity(trace.heads[a].A, trace.heads[b].A) : null
  return (
    <div className="grid gap-3">
      <Card title={`${H} parallel heads · d_model = ${D_MODEL}, h = ${H}, d_k = d_v = ${trace.dk}`} right={
        <div className="flex flex-wrap items-center gap-2">
          <Seg label="Number of heads" value={H} onChange={(v) => st.setH(v)} options={[1, 2, 4].map((v) => ({ v, label: `h = ${v}`, title: `d_k = ${D_MODEL / v}` }))} />
          <Seg label="Head view" value={mode} onChange={setMode} options={[{ v: 'chambers', label: 'all heads' }, { v: 'compare', label: 'compare two' }, { v: 'overlay', label: 'overlay' }]} />
        </div>
      }>
        <div className="flex flex-wrap items-center gap-2 text-[12px]">
          <span className="text-muted">enter a head:</span><HeadPicker />
          <span className="ml-2 text-muted">ablate:</span>
          {headsOn.map((on, h) => <button key={h} type="button" className="btn !h-7 text-[11.5px]" aria-pressed={!on} onClick={() => st.toggleHead(h)} data-testid={`ablate-${h}`}>{on ? `disable head ${h + 1}` : `head ${h + 1} disabled ⊘`}</button>)}
        </div>
        <p className="mt-1 text-[11.5px] text-muted">Each head has its own projection matrices (its own column block of W_Q, W_K, W_V). Heads are not assigned linguistic jobs; any specialization would come from training, and these weights are hand-chosen toys.</p>
      </Card>
      {mode === 'chambers' && (
        <div className={`grid gap-3 ${H === 1 ? '' : H === 2 ? '@3xl:grid-cols-2' : '@3xl:grid-cols-2 @7xl:grid-cols-4'}`}>
          {trace.heads.map((hd) => (
            <Card key={hd.h} className={sel.h === hd.h ? 'ring-1 ring-cyan/60' : ''} title={<span style={{ color: hc[hd.h % 4] }}>head {hd.h + 1}{hd.enabled ? '' : ' · disabled (contributes zeros)'}</span>}
              right={<button type="button" className="btn !h-6 !px-1.5 text-[11px]" onClick={() => select({ h: hd.h })} aria-pressed={sel.h === hd.h}>inspect</button>}>
              <div className={`grid gap-2 ${hd.enabled ? '' : 'opacity-45'}`} data-testid={`chamber-${hd.h}`}>
                <div className="flex flex-wrap gap-2">
                  <Grid m={hd.Q} role="q" title={`${GLYPH.q} Q${hd.h + 1}`} rowLabels={tokens.map((_, k) => String(k + 1))} cell={38} digits={1} ariaLabel={`Q of head ${hd.h + 1}`} hlRows={[sel.i]} />
                  <Grid m={hd.K} role="k" title={`${GLYPH.k} K${hd.h + 1}`} cell={38} digits={1} ariaLabel={`K of head ${hd.h + 1}`} hlRows={[sel.j]} />
                  <Grid m={hd.V} role="v" title={`${GLYPH.v} V${hd.h + 1}`} cell={38} digits={1} ariaLabel={`V of head ${hd.h + 1}`} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Grid m={hd.scaled} role="score" title="scores S̃" masked={masks(trace)} cell={38} digits={1} ariaLabel={`Scaled scores of head ${hd.h + 1}`} selected={[sel.i, sel.j]} onCell={(i, j) => select({ i, j, h: hd.h })} />
                  <Grid m={hd.A} role="prob" mode="prob" title="weights A" masked={masks(trace)} cell={38} ariaLabel={`Attention weights of head ${hd.h + 1}`} selected={[sel.i, sel.j]} onCell={(i, j) => select({ i, j, h: hd.h })} testid={`chamber-A-${hd.h}`} />
                  <Grid m={hd.O} role="out" title={`head${hd.h + 1} = AV`} cell={38} ariaLabel={`Output of head ${hd.h + 1}`} hlRows={[sel.i]} />
                </div>
                <div className="font-mono text-[11px] text-muted">entropy per query (nats): {hd.entropy.map((e) => fx(e, 2)).join(' · ')}</div>
              </div>
            </Card>
          ))}
        </div>
      )}
      {mode === 'compare' && H > 1 && (
        <Card title="compare two heads">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[12px]">
            <Seg label="First head" value={a} onChange={(v) => setPair([v, b])} options={trace.heads.map((h) => ({ v: h.h, label: `head ${h.h + 1}` }))} />
            <span className="text-faint">vs</span>
            <Seg label="Second head" value={b} onChange={(v) => setPair([a, v])} options={trace.heads.map((h) => ({ v: h.h, label: `head ${h.h + 1}` }))} />
          </div>
          <div className="flex flex-wrap items-start gap-3">
            <Grid m={trace.heads[a].A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title={`head ${a + 1}`} masked={masks(trace)} cell={50} ariaLabel={`Head ${a + 1} weights`} />
            <Grid m={trace.heads[b].A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title={`head ${b + 1}`} masked={masks(trace)} cell={50} ariaLabel={`Head ${b + 1} weights`} />
            <Grid m={trace.heads[a].A.map((row, i) => row.map((v, j) => v - trace.heads[b].A[i][j]))} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title={`difference (head ${a + 1} − head ${b + 1})`} cell={50} digits={3} ariaLabel="Difference of the two heads" testid="g-head-diff" />
          </div>
          {sim && (
            <table className="mt-2 font-mono text-[12px]" data-testid="head-metrics">
              <tbody>
                <tr><td className="pr-3 text-muted">mean Jensen–Shannon divergence per query row (0 = identical, ln 2 = 0.693 max)</td><td>{fx(sim.js, 4)}</td></tr>
                <tr><td className="pr-3 text-muted">mean total-variation distance per row (0…1)</td><td>{fx(sim.tv, 4)}</td></tr>
                <tr><td className="pr-3 text-muted">cosine similarity of the flattened weight matrices</td><td>{fx(sim.cosine, 4)}</td></tr>
                <tr><td className="pr-3 text-muted">mean entropy · head {a + 1} vs head {b + 1}</td><td>{fx(mean(trace.heads[a].entropy), 3)} vs {fx(mean(trace.heads[b].entropy), 3)}</td></tr>
                <tr><td className="pr-3 text-muted">‖contribution to the output‖ of query {sel.i + 1}: head {a + 1} vs head {b + 1}</td><td>{fx(Math.hypot(...contrib[a][sel.i]), 3)} vs {fx(Math.hypot(...contrib[b][sel.i]), 3)}</td></tr>
              </tbody>
            </table>
          )}
        </Card>
      )}
      {mode === 'compare' && H === 1 && <Card><p className="text-[12.5px] text-muted">With one head there is nothing to compare; choose h = 2 or h = 4.</p></Card>}
      {mode === 'overlay' && <Card title="overlay · each cell split into one bar per head (height = weight)"><Overlay trace={trace} tokens={tokens} /></Card>}
      <Card title="advanced experiment · modify only head 1's projection matrices">
        <p className="text-[12.5px] text-muted">Saves the current state, then changes one entry of W_Q inside head 1's column block. Head 1's queries, scores and weights change; every tensor of the other heads stays identical; the final output changes after concatenation and W_O.</p>
        <button type="button" className="btn mt-1" onClick={() => { st.saveBaseline('before editing head 1'); const p = attnStore.getState().params; st.setW('Wq', 2, 0, p.Wq[2][0] - 2) }} data-testid="edit-head1">Run: W_Q[3, 1] −= 2 (head 1 only)</button>
        <BaselineDiff trace={trace} />
      </Card>
    </div>
  )
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)

function Overlay({ trace, tokens }: { trace: AttnTrace; tokens: string[] }) {
  const r = useRoles(), c = useColors()
  const hc = headColors(r)
  const n = trace.Tq, m = trace.Tk, cw = 64, ch = 50, lw = 74, th = 20
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg width={lw + m * cw + 4} height={th + n * ch + 24} role="img" aria-label="Overlay of all heads' attention weights" data-testid="overlay">
        {tokens.map((t, j) => <text key={j} x={lw + j * cw + cw / 2} y={13} textAnchor="middle" fontSize={10.5} fill={c.muted} fontFamily="var(--font-mono)">k{j + 1}·{t.slice(0, 4)}</text>)}
        {tokens.map((t, i) => <text key={i} x={lw - 6} y={th + i * ch + ch / 2 + 4} textAnchor="end" fontSize={11} fill={c.muted} fontFamily="var(--font-mono)">{i + 1}·{t.slice(0, 6)}</text>)}
        {Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => {
          const x0 = lw + j * cw + 2, y0 = th + i * ch + 2, w = cw - 4, hgt = ch - 4, bw = w / trace.cfg.H
          const masked = trace.M[i][j] === -Infinity
          return (
            <g key={`${i}${j}`}>
              <rect x={x0} y={y0} width={w} height={hgt} rx={4} fill={c.panel2} stroke={masked ? r.mask : c.line} strokeDasharray={masked ? '3 3' : undefined} />
              {!masked && trace.heads.map((hd) => <rect key={hd.h} x={x0 + hd.h * bw + 1} y={y0 + hgt - hd.A[i][j] * hgt} width={bw - 2} height={hd.A[i][j] * hgt} fill={hc[hd.h % 4]} opacity={hd.enabled ? 0.9 : 0.25}><title>{`head ${hd.h + 1}: ${pct(hd.A[i][j])}`}</title></rect>)}
              {masked && <text x={x0 + w / 2} y={y0 + hgt / 2 + 4} textAnchor="middle" fontSize={11} fill={r.mask}>⊘</text>}
            </g>
          )
        }))}
        {trace.heads.map((hd) => <g key={hd.h} transform={`translate(${lw + hd.h * 80}, ${th + n * ch + 8})`}><rect width={10} height={10} fill={hc[hd.h % 4]} /><text x={14} y={9} fontSize={10.5} fill={c.fg} fontFamily="var(--font-mono)">head {hd.h + 1}</text></g>)}
      </svg>
    </div>
  )
}

/* ------------------------------------------------------------------ 11 · concat */
export function ConcatView({ trace }: { trace: AttnTrace }) {
  const r = useRoles(), reduce = useReducedMotion()
  const tokens = useTokens(), sel = useAttn((s) => s.sel)
  const [merged, setMerged] = useState(true)
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const hc = headColors(r), dk = trace.dk, H = trace.cfg.H
  const cl = Array.from({ length: H * dk }, (_, k) => `h${Math.floor(k / dk) + 1}.${(k % dk) + 1}`)
  return (
    <div className="grid gap-3">
      <Card title={`H = Concat(head₁ … head${H > 1 ? `_${H}` : '₁'}) ∈ ℝ^${trace.Tq}×${H * dk}`} right={<button type="button" className="btn !h-7" onClick={() => setMerged(!merged)} aria-pressed={merged}>{merged ? 'separate the heads' : 'merge the heads'}</button>}>
        <div className="flex flex-wrap items-start gap-3">
          {trace.heads.map((hd) => (
            <motion.div key={hd.h} animate={{ x: merged || reduce ? 0 : hd.h * 18 - 9 }} transition={{ duration: 0.5 }}>
              <Grid m={hd.enabled ? hd.O : hd.O.map((row) => row.map(() => 0))} role="out" title={<span style={{ color: hc[hd.h % 4] }}>head {hd.h + 1}{hd.enabled ? '' : ' (disabled → 0)'} · {trace.Tq}×{dk}</span>}
                rowLabels={hd.h === 0 ? rl : undefined} colLabels={Array.from({ length: dk }, (_, k) => `c${k + 1}`)} hlRows={[sel.i]} cell={48} ariaLabel={`Head ${hd.h + 1} output`} />
            </motion.div>
          ))}
          <span className="self-center font-mono text-faint">→</span>
          <motion.div initial={reduce ? false : { opacity: 0, scale: 0.96 }} animate={{ opacity: merged ? 1 : 0.35, scale: 1 }} transition={{ duration: 0.4 }}>
            <Grid m={trace.concat} role="out" rowLabels={rl} colLabels={cl} group={dk} groupColors={hc} hlRows={[sel.i]} cell={48} ariaLabel="Concatenated head outputs H" testid="g-H" title={`H · ${trace.Tq}×${H * dk}`} />
          </motion.div>
        </div>
        <p className="mt-2 font-mono text-[12px]" data-testid="concat-row">row {sel.i + 1} of H = [{trace.heads.map((hd) => (hd.enabled ? hd.O[sel.i] : hd.O[sel.i].map(() => 0)).map((v) => fx(v, 3)).join(', ')).join(' | ')}]</p>
        <p className="mt-1 text-[12px] text-muted">Concatenation only rearranges numbers: (n, h, d_v) → (n, h·d_v). It is not the end of multi-head attention; the learned output projection W_O follows.</p>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 12 · output projection */
export function ProjView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), params = useAttn((s) => s.params)
  const st = attnStore.getState()
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const hc = headColors(r), dk = trace.dk, H = trace.cfg.H
  const cell = sel.cell?.m === 'O' ? sel.cell : { m: 'O', r: sel.i, c: 0 }
  const calc = matmulCell(trace.concat, params.Wo, cell.r, cell.c)
  const contrib = useMemo(() => headContributions(trace, params.Wo), [trace, params.Wo])
  const cl = Array.from({ length: H * dk }, (_, k) => `h${Math.floor(k / dk) + 1}.${(k % dk) + 1}`)
  return (
    <div className="grid gap-3">
      <Card title={<span><RoleTag role="out">O</RoleTag> = H W_O · {trace.Tq}×{H * dk} · {H * dk}×{D_MODEL} → {trace.Tq}×{D_MODEL}</span>}>
        <div className="flex flex-wrap items-start gap-2">
          <Grid m={trace.concat} role="out" rowLabels={rl} colLabels={cl} group={dk} groupColors={hc} hlRows={[cell.r]} cell={44} ariaLabel="H" />
          <span className="self-center font-mono text-faint">×</span>
          <Grid m={params.Wo} role="w" rowLabels={cl} colLabels={feat} hlCols={[cell.c]} cell={44} digits={1} ariaLabel="W_O" />
          <span className="self-center font-mono text-faint">=</span>
          <Grid m={trace.out} role="out" rowLabels={rl} colLabels={feat} selected={[cell.r, cell.c]} onCell={(i, c) => select({ i, cell: { m: 'O', r: i, c } })} cell={50} ariaLabel="Multi-head output O" testid="g-O" />
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`output cell o${cell.r + 1},${cell.c + 1} · click any cell of O`}>
          <TermSum key={`${cell.r}${cell.c}`} a={calc.a} b={calc.b} roleA="out" roleB="w" label={`o_${cell.r + 1},${cell.c + 1} = Σₖ H_${cell.r + 1},k · W_O[k, ${cell.c + 1}]`} result={`o${cell.r + 1},${cell.c + 1}`} testid="proj-calc" />
          {H > 1 && (
            <div className="mt-2 font-mono text-[12px] text-muted">by head: {trace.heads.map((hd) => {
              const part = calc.products.slice(hd.h * dk, (hd.h + 1) * dk).reduce((s, v) => s + v, 0)
              return <span key={hd.h} style={{ color: hc[hd.h % 4] }}>head {hd.h + 1}: {fx(part, 4)}{hd.h < H - 1 ? ' + ' : ''}</span>
            })} = {fx(calc.sum, 4)}</div>
          )}
          <p className="mt-2 text-[12px] text-muted">Every output feature draws on every head: W_O is where the heads' results are mixed. Equivalently O = Σ_h head_h W_O^(h), with W_O^(h) the rows of W_O that belong to head h.</p>
        </Card>
        <Card title="edit W_O (learned output projection)">
          <EditableMatrix m={params.Wo} onSet={(rr, c, v) => st.setW('Wo', rr, c, v)} rowLabels={cl} colLabels={feat} role="w" label="W_O" testid="edit-Wo" />
        </Card>
      </div>
      {H > 1 && (
        <Card title="each head's contribution head_h W_O^(h); they add up to O">
          <div className="flex flex-wrap items-start gap-2">
            {contrib.map((m, h) => <div key={h} className="flex items-center gap-2"><Grid m={m} role="out" title={<span style={{ color: hc[h % 4] }}>head {h + 1} · W_O^({h + 1}) = rows {h * dk + 1}–{h * dk + dk}</span>} rowLabels={h === 0 ? rl : undefined} colLabels={feat} cell={44} ariaLabel={`Contribution of head ${h + 1}`} hlRows={[cell.r]} />{h < contrib.length - 1 && <span className="font-mono text-faint">+</span>}</div>)}
            <span className="self-center font-mono text-faint">=</span>
            <Grid m={trace.out} role="out" colLabels={feat} title="O" cell={44} ariaLabel="O" hlRows={[cell.r]} />
          </div>
          <p className="mt-1 text-[11px] text-faint">Each W_O^(h) is a {dk}×{D_MODEL} block of rows of W_O (rows {Array.from({ length: H }, (_, h) => `${h * dk + 1}–${h * dk + dk}`).join(', ')}).</p>
        </Card>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 13 · output */
export function OutputView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const i = sel.i
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={<span>input <RoleTag role="x">X</RoleTag> · {trace.Tq}×{D_MODEL}</span>}>
          <Grid m={trace.X} role="x" rowLabels={rl} colLabels={feat} hlRows={[i]} onCell={(a) => select({ i: a })} ariaLabel="Input X" />
        </Card>
        <Card title={<span>output <RoleTag role="out">MultiHead(X)</RoleTag> · {trace.Tq}×{D_MODEL} · same shape</span>}>
          <Grid m={trace.out} role="out" rowLabels={rl} colLabels={feat} hlRows={[i]} onCell={(a) => select({ i: a })} ariaLabel="Final attention output" testid="g-out" />
        </Card>
      </div>
      <Card title={`summary for token ${i + 1} “${tokens[i]}”`}>
        <div className="grid gap-1 font-mono text-[12px]" data-testid="output-summary">
          <div><span style={{ color: r.x }}>{GLYPH.x} x{i + 1}</span> = [{trace.X[i].map((v) => fx(v, 2)).join(', ')}]</div>
          {trace.heads.map((hd) => (
            <div key={hd.h} className="pl-2">
              head {hd.h + 1}: <span style={{ color: r.q }}>q = [{hd.Q[i].map((v) => fx(v, 2)).join(', ')}]</span> · weights <span style={{ color: r.prob }}>[{hd.A[i].map((v) => fx(v, 3)).join(', ')}]</span> · <span style={{ color: r.out }}>o = [{hd.O[i].map((v) => fx(v, 3)).join(', ')}]</span>{hd.enabled ? '' : ' (disabled)'}
            </div>
          ))}
          <div>concat row = [{trace.concat[i].map((v) => fx(v, 3)).join(', ')}]</div>
          <div><span style={{ color: r.out }}>{GLYPH.out} output</span> = [{trace.out[i].map((v) => fx(v, 4)).join(', ')}]</div>
        </div>
        <p className="mt-2 text-[12px] text-muted">{trace.cfg.causal ? `Under the causal mask this row depends only on tokens 1…${i + 1}.` : 'Without a causal mask this row can depend on every token.'} In a Transformer block the output is added to the residual stream (X + MultiHead(X)) and followed by an MLP (see module 02).</p>
      </Card>
      <Card title="microscope C · the whole computation, with shapes">
        <PipelineView trace={trace} stage={stageById('output')} />
      </Card>
    </div>
  )
}
