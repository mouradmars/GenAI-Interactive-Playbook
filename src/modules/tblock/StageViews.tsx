/**
 * The stage views of the block microscope. Each view only displays tensors from the engine trace and writes
 * selections/edits back to the store; it never computes model values of its own.
 */
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, ChevronDown, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { ACT_FN, toyEmbeddings, vdot, vnorm, type BlockTrace, type LNTrace, type Matrix, type MlpAct } from '../../engine/block'
import { rng } from '../../engine/decoding'
import { Card, Tex } from '../../components/viz'
import { MatrixView, VecBars, f3, f4, tn, useRoles } from './common'
import type { StageId } from './stages'
import { useTb, type ParamKey } from './store'

const fl = (d: number) => Array.from({ length: d }, (_, k) => `x${k + 1}`)

export function StageView({ id, trace }: { id: StageId; trace: BlockTrace }) {
  switch (id) {
    case 'x': return <InputView trace={trace} />
    case 'ln1': return <LNMicroscope ln={trace.ln1!} input={trace.X} which="ln1" editable />
    case 'ln2': return <LNMicroscope ln={trace.ln2!} input={trace.H} which="ln2" />
    case 'qkv': return <QKVView trace={trace} />
    case 'dot': return <DotView trace={trace} />
    case 'scores': return <ScoresView trace={trace} />
    case 'scale': return <ScaleView trace={trace} />
    case 'mask': return <MaskView trace={trace} />
    case 'softmax': return <SoftmaxView trace={trace} />
    case 'weights': return <WeightsView trace={trace} />
    case 'av': return <AVView trace={trace} />
    case 'concat': return <ConcatView trace={trace} />
    case 'proj': return <ProjView trace={trace} />
    case 'res1': return <ResidualView trace={trace} which={1} />
    case 'ffn1': case 'act': case 'ffn2': return <MLPView trace={trace} stage={id} />
    case 'res2': return <ResidualView trace={trace} which={2} />
    case 'out': return <OutputView trace={trace} />
  }
}

function HeadPicker({ H }: { H: number }) {
  const r = useRoles()
  const h = useTb((s) => s.sel.h), select = useTb((s) => s.select), heads = useTb((s) => s.iv.heads)
  return (
    <div className="flex items-center gap-1" role="radiogroup" aria-label="Attention head">
      {Array.from({ length: H }, (_, k) => (
        <button key={k} type="button" role="radio" aria-checked={h === k} onClick={() => select({ h: k })} className={`btn !py-0.5 text-[12px] ${h === k ? 'active' : ''}`} style={h === k ? { color: r.score, borderColor: r.score } : undefined}>
          head {k + 1}{heads[k] === false ? ' (off)' : ''}
        </button>
      ))}
    </div>
  )
}
const headColors = (r: ReturnType<typeof useRoles>) => [r.score, r.q, r.v, r.k, r.ffn, r.res, r.tok, r.ln]

/* ------------------------------------------------------------------ 1 · input */
function InputView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), showUp = useTb((s) => s.showUpstream), setShowUp = useTb((s) => s.setShowUpstream)
  const seed = useTb((s) => s.seed), xEdited = useTb((s) => s.xEdited)
  const X = trace.X, xi = X[sel.i], xj = X[sel.j]
  const cos = vdot(xi, xj) / (vnorm(xi) * vnorm(xj) || 1)
  const up = useMemo(() => toyEmbeddings(tokens, trace.cfg.d, seed), [tokens, trace.cfg.d, seed])
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-[auto_minmax(0,1fr)]">
        <Card title={`X ∈ ℝ^${trace.cfg.T}×${trace.cfg.d} · one row per token (click a cell to edit it in the inspector)`}>
          <MatrixView m={X} rowLabels={tokens} colLabels={fl(trace.cfg.d)} color={r.tok} hlRows={[sel.i]} selected={sel.cell?.m === 'X' ? [sel.cell.r, sel.cell.c] : null}
            onCell={(i, c) => select({ i, f: c, cell: { m: 'X', r: i, c } })} ariaLabel="Input hidden states X" testid="mat-X" />
          {xEdited && <p className="mt-1 text-[11.5px] text-amber">Edited by hand (toy mode).</p>}
        </Card>
        <Card title={`token “${tokens[sel.i]}” · ‖x‖ = ${f3(vnorm(xi))}`}>
          <VecBars v={xi} color={r.tok} />
          <div className="mt-2 grid gap-1 text-[12.5px]">
            <div className="flex items-center gap-2 text-muted">compare with
              <select className="rounded border border-line bg-panel-2 px-1" value={sel.j} onChange={(e) => select({ j: +e.target.value })} aria-label="Token to compare">{tokens.map((t, k) => <option key={k} value={k}>{k + 1} · {t}</option>)}</select>
            </div>
            <VecBars v={xj} color={r.k} />
            <div className="font-mono text-[12px] text-muted">cosine = {f3(cos)} · distance = {f3(vnorm(xi.map((v, k) => v - xj[k])))} · ‖x_j‖ = {f3(vnorm(xj))}</div>
          </div>
        </Card>
      </div>
      <Card>
        <button type="button" className="flex items-center gap-1 text-[13px] text-cyan" onClick={() => setShowUp(!showUp)} aria-expanded={showUp}>{showUp ? <ChevronDown size={14} /> : <ChevronRight size={14} />}Where do these vectors come from? (upstream of the block)</button>
        {showUp && (
          <div className="mt-2 grid gap-2">
            <p className="text-[12.5px] leading-relaxed text-muted">Before the first block: text → tokens → token IDs → an embedding lookup E[token] → plus positional information P[position]. In this toy, X = E + P with one-decimal values. Inside a block nothing positional is added again; deeper blocks receive the previous block's output.</p>
            <div className="flex flex-wrap items-start gap-3">
              <MatrixView m={up.E} rowLabels={tokens} colLabels={fl(trace.cfg.d)} color={r.tok} title="E[token]" ariaLabel="Token embeddings" digits={1} cell={40} />
              <span className="self-center font-mono text-lg">+</span>
              <MatrixView m={up.P} rowLabels={tokens.map((_, i) => `pos ${i + 1}`)} colLabels={fl(trace.cfg.d)} color={r.ln} title="P[position]" ariaLabel="Positional vectors" digits={1} cell={40} />
              <span className="self-center font-mono text-lg">=</span>
              <MatrixView m={up.X} rowLabels={tokens} colLabels={fl(trace.cfg.d)} color={r.tok} title="X (rounded to 0.1)" ariaLabel="Constructed X" digits={1} cell={40} />
            </div>
            <p className="text-[11.5px] text-faint">The same token at two positions has the same E row but a different P row.</p>
          </div>
        )}
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 2 · LayerNorm microscope */
const LN_STEPS = ['original values', 'mean μ', 'subtract μ', 'variance σ²', 'normalize', 'apply γ, β']
export function LNMicroscope({ ln, input, which, editable = false, title }: { ln: LNTrace; input: Matrix; which: 'ln1' | 'ln2'; editable?: boolean; title?: string }) {
  const c = useColors(), r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), params = useTb((s) => s.params), setLN = useTb((s) => s.setLN), eps = useTb((s) => s.eps), setEps = useTb((s) => s.setEps), setX = useTb((s) => s.setX)
  const [k, setK] = useState(5)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { if (!playing) return; if (k >= 5) { setPlaying(false); return } const id = setTimeout(() => setK(k + 1), 900); return () => clearTimeout(id) }, [playing, k])
  const i = sel.i, x = input[i], d = x.length, mu = ln.mean[i], v2 = ln.variance[i], sd = Math.sqrt(v2 + eps)
  const g = params[which].g, b = params[which].b
  const shown = k <= 1 ? x : k <= 3 ? ln.centered[i] : k === 4 ? ln.xhat[i] : ln.out[i]
  const all = [...x, ...ln.centered[i], ...ln.xhat[i], ...ln.out[i]].map(Math.abs)
  const ma = Math.max(1e-9, ...all)
  const W = Math.max(260, d * 56), Hh = 170, mid = Hh / 2, bw = W / d
  const y = (v: number) => mid - (v / ma) * (mid - 18)
  return (
    <div className="grid gap-3">
      <Card title={title ?? `LayerNorm ${which === 'ln1' ? '₁' : '₂'} · token “${tokens[i]}” · normalizes across its ${d} features`} right={
        <div className="flex items-center gap-1">
          <button type="button" className="btn icon-btn !h-7 !w-7" onClick={() => { setK(0); setPlaying(true) }} aria-label="Animate LayerNorm"><Play size={13} /></button>
          {playing && <button type="button" className="btn icon-btn !h-7 !w-7" onClick={() => setPlaying(false)} aria-label="Pause"><Pause size={13} /></button>}
          <button type="button" className="btn icon-btn !h-7 !w-7" onClick={() => { setPlaying(false); setK(5) }} aria-label="Show result"><RotateCcw size={13} /></button>
        </div>}>
        <div className="mb-2 flex flex-wrap gap-1">{LN_STEPS.map((s, n) => <button key={s} type="button" onClick={() => { setPlaying(false); setK(n) }} className={`rounded-md border px-2 py-0.5 text-[11.5px] ${n === k ? 'border-cyan text-cyan' : n < k ? 'border-line-strong text-fg' : 'border-line text-faint'}`}>{n + 1}. {s}</button>)}</div>
        <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <svg viewBox={`0 0 ${W} ${Hh + 18}`} className="block w-full max-w-[360px]" role="img" aria-label="LayerNorm bars" data-testid="ln-bars">
            <line x1={0} x2={W} y1={mid} y2={mid} stroke={c.lineStrong} />
            {shown.map((v, f) => (
              <g key={f}>
                <motion.rect x={f * bw + 8} width={bw - 16} rx={3} fill={k >= 5 ? r.tok : k === 4 ? r.res : r.ln} animate={{ y: Math.min(y(v), mid), height: Math.max(1, Math.abs(y(v) - mid)) }} transition={{ duration: 0.5 }} />
                <text x={f * bw + bw / 2} y={(v >= 0 ? y(v) - 4 : y(v) + 12)} textAnchor="middle" fontSize={10.5} fill={c.fg} fontFamily="var(--font-mono)">{f3(v)}</text>
                <text x={f * bw + bw / 2} y={Hh + 14} textAnchor="middle" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">x{f + 1}</text>
              </g>
            ))}
            {k >= 1 && k <= 2 && <motion.line x1={0} x2={W} initial={{ y1: mid, y2: mid }} animate={{ y1: k === 1 ? y(mu) : mid, y2: k === 1 ? y(mu) : mid }} stroke={r.score} strokeWidth={2} strokeDasharray="6 4" />}
            {k === 1 && <text x={W - 4} y={y(mu) - 4} textAnchor="end" fontSize={11} fill={r.score} fontFamily="var(--font-mono)">μ = {f4(mu)}</text>}
            {k === 3 && [1, -1].map((s) => <line key={s} x1={0} x2={W} y1={y(s * Math.sqrt(v2))} y2={y(s * Math.sqrt(v2))} stroke={r.score} strokeDasharray="3 3" />)}
            {k === 3 && <text x={W - 4} y={y(Math.sqrt(v2)) - 4} textAnchor="end" fontSize={11} fill={r.score} fontFamily="var(--font-mono)">±σ = ±{f4(Math.sqrt(v2))}</text>}
          </svg>
          <div className="grid content-start gap-1 overflow-x-auto scroll-thin text-[13px]" data-testid="ln-numbers">
            <Tex block tex={`\\mu = \\tfrac{1}{${d}}(${x.map((v) => tn(v, 2)).join(' + ')}) = ${tn(mu, 4)}`} />
            {k >= 3 && <Tex block tex={`\\sigma^2 = \\tfrac{1}{${d}}\\sum_i (x_i - \\mu)^2 = ${tn(v2, 4)}`} />}
            {k >= 4 && <Tex block tex={`\\hat x_{${sel.f + 1}} = \\frac{${tn(x[sel.f], 2)} - ${tn(mu, 4)}}{\\sqrt{${tn(v2, 4)} + ${eps}}} = ${tn(ln.xhat[i][sel.f], 4)}`} />}
            {k >= 5 && <Tex block tex={`y_{${sel.f + 1}} = ${tn(g[sel.f], 2)}\\cdot${tn(ln.xhat[i][sel.f], 4)} + ${tn(b[sel.f], 2)} = ${tn(ln.out[i][sel.f], 4)}`} />}
            <p className="text-[11.5px] text-faint">Feature shown in detail: x{sel.f + 1} (click a bar column in the matrices to change). Standard deviation used: √(σ² + ε) = {f4(sd)}.</p>
          </div>
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title="experiment: γ, β, ε and the input">
          <div className="grid gap-1.5 text-[12.5px] text-muted">
            <div className="flex flex-wrap items-center gap-1">feature
              {Array.from({ length: d }, (_, f) => <button key={f} type="button" className={`btn !px-2 !py-0.5 ${sel.f === f ? 'active' : ''}`} onClick={() => select({ f })}>x{f + 1}</button>)}
            </div>
            <label className="flex items-center gap-2">γ<sub>{sel.f + 1}</sub> <span className="w-10 font-mono text-fg">{g[sel.f].toFixed(2)}</span><input type="range" min={-2} max={3} step={0.05} value={g[sel.f]} onChange={(e) => setLN(which, 'g', sel.f, +e.target.value)} className="flex-1" aria-label="Gamma" /></label>
            <label className="flex items-center gap-2">β<sub>{sel.f + 1}</sub> <span className="w-10 font-mono text-fg">{b[sel.f].toFixed(2)}</span><input type="range" min={-2} max={2} step={0.05} value={b[sel.f]} onChange={(e) => setLN(which, 'b', sel.f, +e.target.value)} className="flex-1" aria-label="Beta" /></label>
            <label className="flex items-center gap-2">ε = 10<sup>{Math.log10(eps).toFixed(0)}</sup><input type="range" min={-8} max={1} step={1} value={Math.round(Math.log10(eps))} onChange={(e) => setEps(10 ** +e.target.value)} className="flex-1" aria-label="Epsilon exponent" /></label>
            {editable ? (
              <label className="flex items-center gap-2">x<sub>{sel.f + 1}</sub> of “{tokens[i]}” <span className="w-10 font-mono text-fg">{x[sel.f].toFixed(2)}</span><input type="range" min={-4} max={4} step={0.1} value={x[sel.f]} onChange={(e) => setX(i, sel.f, +e.target.value)} className="flex-1" aria-label="Input value" /></label>
            ) : <p className="text-[11.5px] text-faint">This LayerNorm's input is H (computed upstream in the block); edit X or the weights to change it.</p>}
            <p className="text-[11.5px] text-faint">Try a large ε (10⁰): the output shrinks, because the denominator no longer equals the standard deviation.</p>
          </div>
        </Card>
        <Card title="every token, normalized independently">
          <table className="w-full font-mono text-[12px]">
            <thead className="text-left text-faint"><tr><th className="px-1">token</th><th className="px-1 text-right">μ</th><th className="px-1 text-right">σ²</th><th className="px-1 text-right">mean of x̂</th><th className="px-1 text-right">var of x̂</th></tr></thead>
            <tbody>{tokens.map((t, n) => {
              const xh = ln.xhat[n], m = xh.reduce((s, v) => s + v, 0) / d, vv = xh.reduce((s, v) => s + (v - m) ** 2, 0) / d
              return <tr key={n} className={`border-t border-line ${n === i ? 'text-cyan' : ''}`} onClick={() => select({ i: n })} style={{ cursor: 'pointer' }}><td className="px-1">{t}</td><td className="px-1 text-right">{f3(ln.mean[n])}</td><td className="px-1 text-right">{f3(ln.variance[n])}</td><td className="px-1 text-right">{f3(m)}</td><td className="px-1 text-right">{f3(vv)}</td></tr>
            })}</tbody>
          </table>
          <p className="mt-1 text-[11.5px] text-muted">Each row is standardized over its own features (mean 0, variance just under 1 because of ε), whatever the other tokens look like.</p>
        </Card>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 3 · Q, K, V factory */
export function MatMulCell({ a, b, r, c, bias, labelA, labelOut }: { a: Matrix; b: Matrix; r: number; c: number; bias?: number; labelA: string; labelOut: string }) {
  const terms = a[r].map((v, k) => `${tn(v, 2)}\\cdot${tn(b[k][c], 2)}`)
  const sum = a[r].reduce((s, v, k) => s + v * b[k][c], 0) + (bias ?? 0)
  const shown = terms.length > 8 ? [...terms.slice(0, 7), '\\dots'] : terms
  return <div className="overflow-x-auto scroll-thin text-[13px]"><Tex block tex={`{${labelOut}}_{${r + 1},${c + 1}} = \\sum_k {${labelA}}_{${r + 1},k}W_{k,${c + 1}}${bias !== undefined ? ' + b' : ''} = ${shown.join(' + ')}${bias !== undefined && bias !== 0 ? ` + ${tn(bias, 2)}` : ''} = ${tn(sum, 4)}`} /></div>
}

function QKVView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), params = useTb((s) => s.params)
  const { d, H } = trace.cfg, dh = d / H
  const xn = trace.attnIn
  const cell = sel.cell && ['Q', 'K', 'V', 'Wq', 'Wk', 'Wv'].includes(sel.cell.m) ? sel.cell : { m: 'Q', r: sel.i, c: 0 }
  const role = cell.m.replace('W', '').replace('q', 'Q').replace('k', 'K').replace('v', 'V') as 'Q' | 'K' | 'V'
  const outRow = cell.m.startsWith('W') ? sel.i : cell.r, outCol = cell.c
  const hc = headColors(r)
  const machines: { role: 'Q' | 'K' | 'V'; W: ParamKey; out: Matrix; color: string; intuition: string }[] = [
    { role: 'Q', W: 'Wq', out: trace.Q, color: r.q, intuition: 'what this position is looking for' },
    { role: 'K', W: 'Wk', out: trace.K, color: r.k, intuition: 'what it can be matched against' },
    { role: 'V', W: 'Wv', out: trace.V, color: r.v, intuition: 'what it passes along if attended to' },
  ]
  return (
    <div className="grid gap-3">
      <Card title={`input to attention: ${trace.cfg.arch === 'pre' ? 'X_n = LN₁(X)' : 'X (post-LN: no norm before attention)'} · [${trace.cfg.T}, ${d}]`}>
        <MatrixView m={xn} rowLabels={tokens} colLabels={fl(d)} color={r.tok} hlRows={[outRow]} ariaLabel="Attention input" onCell={(i) => select({ i })} />
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-3">
        {machines.map((m) => {
          const active = role === m.role
          return (
            <Card key={m.role} title={<span style={{ color: m.color }}>{m.role} = X_n W_{m.role} · {m.intuition}</span>} className={active ? 'ring-1' : ''}>
              <div className="grid gap-2">
                <MatrixView m={params[m.W]} rowLabels={fl(d)} colLabels={Array.from({ length: d }, (_, k) => `c${k + 1}`)} color={m.color} digits={1} cell={38} colGroup={dh} groupColors={hc}
                  hlCols={active ? [outCol] : []} selected={sel.cell?.m === m.W ? [sel.cell.r, sel.cell.c] : null} onCell={(rr, cc) => select({ cell: { m: m.W, r: rr, c: cc } })} ariaLabel={`W_${m.role}`} title={`W_${m.role} [${d}, ${d}] · click to edit`} testid={`mat-${m.W}`} />
                <MatrixView m={m.out} rowLabels={tokens} colLabels={Array.from({ length: d }, (_, k) => `h${Math.floor(k / dh) + 1}.${(k % dh) + 1}`)} color={m.color} cell={38} colGroup={dh} groupColors={hc}
                  selected={active ? [outRow, outCol] : null} hlRows={active ? [outRow] : []} onCell={(rr, cc) => select({ i: rr, cell: { m: m.role, r: rr, c: cc } })} ariaLabel={m.role} title={`${m.role} [${trace.cfg.T}, ${d}] · columns grouped by head`} testid={`mat-${m.role}`} />
              </div>
            </Card>
          )
        })}
      </div>
      <Card title={`how ${role}[${tokens[outRow]}, column ${outCol + 1}] is computed: row ${outRow + 1} of X_n times column ${outCol + 1} of W_${role}`}>
        <MatMulCell a={xn} b={params[`W${role.toLowerCase()}` as ParamKey]} r={outRow} c={outCol} bias={params[`b${role.toLowerCase()}` as 'bq']?.[outCol]} labelA="X_n" labelOut={role} />
        <ProductStrip a={xn[outRow]} b={params[`W${role.toLowerCase()}` as ParamKey].map((row) => row[outCol])} colorA={r.tok} colorB={role === 'Q' ? r.q : role === 'K' ? r.k : r.v} />
        <p className="mt-1 text-[11.5px] text-muted">The words “query”, “key” and “value” are intuitions for three learned linear projections; the vectors are not literal questions, labels or facts. Column {outCol + 1} belongs to head {Math.floor(outCol / dh) + 1}.</p>
      </Card>
    </div>
  )
}

/** element-wise products and their running sum, animated */
export function ProductStrip({ a, b, colorA, colorB, label = 'product' }: { a: number[]; b: number[]; colorA: string; colorB: string; label?: string }) {
  const c = useColors()
  const prods = a.map((v, k) => v * b[k])
  let run = 0
  return (
    <div className="mt-2 overflow-x-auto scroll-thin">
      <table className="font-mono text-[12px]" data-testid="product-strip">
        <tbody>
          <tr><td className="pr-2 text-faint">a</td>{a.map((v, k) => <td key={k} className="px-2 text-right" style={{ color: colorA }}>{f3(v)}</td>)}</tr>
          <tr><td className="pr-2 text-faint">b</td>{b.map((v, k) => <td key={k} className="px-2 text-right" style={{ color: colorB }}>{f3(v)}</td>)}</tr>
          <tr className="border-t border-line"><td className="pr-2 text-faint">{label}</td>{prods.map((v, k) => <motion.td key={`${k}-${v}`} className="px-2 text-right" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: k * 0.12 }}>{f3(v)}</motion.td>)}</tr>
          <tr><td className="pr-2 text-faint">running Σ</td>{prods.map((v, k) => { run += v; return <motion.td key={`${k}-${run}`} className="px-2 text-right" style={{ color: c.amber }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 + k * 0.12 }}>{f3(run)}</motion.td> })}</tr>
        </tbody>
      </table>
    </div>
  )
}

/* ------------------------------------------------------------------ 4 · attention stages */
function DotView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h], q = hd.q[sel.i], k = hd.k[sel.j]
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2"><HeadPicker H={trace.cfg.H} />
        <span className="text-[12.5px] text-muted">query <select className="rounded border border-line bg-panel-2 px-1" value={sel.i} onChange={(e) => select({ i: +e.target.value })} aria-label="Query token">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select>
          {' '}key <select className="rounded border border-line bg-panel-2 px-1" value={sel.j} onChange={(e) => select({ j: +e.target.value })} aria-label="Key token">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select></span>
      </div>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_auto]">
        <Card title={`q of “${tokens[sel.i]}” · k of “${tokens[sel.j]}” · head ${sel.h + 1} (d_k = ${hd.q[0].length})`}>
          <div className="grid gap-2 @xl:grid-cols-2">
            <div><div className="text-[11.5px]" style={{ color: r.q }}>query q{sel.i + 1}</div><VecBars v={q} color={r.q} labels={q.map((_, n) => `c${n + 1}`)} height={80} /></div>
            <div><div className="text-[11.5px]" style={{ color: r.k }}>key k{sel.j + 1}</div><VecBars v={k} color={r.k} labels={k.map((_, n) => `c${n + 1}`)} height={80} /></div>
          </div>
          <ProductStrip a={q} b={k} colorA={r.q} colorB={r.k} />
          <div className="overflow-x-auto scroll-thin text-[14px]" data-testid="dot-formula"><Tex block tex={`q_{${sel.i + 1}}\\cdot k_{${sel.j + 1}} = ${q.map((v, n) => `${tn(v)}\\times${tn(k[n])}`).join(' + ')} = ${tn(hd.S[sel.i][sel.j], 4)}`} /></div>
        </Card>
        <Card title="its cell in S">
          <MatrixView m={hd.S} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} selected={[sel.i, sel.j]} hlRows={[sel.i]} hlCols={[sel.j]} onCell={(i, j) => select({ i, j })} ariaLabel="Score matrix" cell={44} />
        </Card>
      </div>
    </div>
  )
}

function ScoresView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h], dh = hd.q[0].length
  return (
    <div className="grid gap-3">
      <HeadPicker H={trace.cfg.H} />
      <div className="flex flex-wrap items-start gap-3">
        <Card title={`Q_h [${trace.cfg.T}, ${dh}]`}><MatrixView m={hd.q} rowLabels={tokens} colLabels={hd.q[0].map((_, n) => `c${n + 1}`)} color={r.q} hlRows={[sel.i]} ariaLabel="Q head" cell={44} /></Card>
        <span className="self-center font-mono text-xl">×</span>
        <Card title={`K_hᵀ [${dh}, ${trace.cfg.T}]`}><MatrixView m={hd.k[0].map((_, c) => hd.k.map((row) => row[c]))} rowLabels={hd.k[0].map((_, n) => `c${n + 1}`)} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.k} hlCols={[sel.j]} ariaLabel="K head transposed" cell={44} /></Card>
        <span className="self-center font-mono text-xl">=</span>
        <Card title={`S = Q_h K_hᵀ [${trace.cfg.T}, ${trace.cfg.T}] · click a cell`}>
          <MatrixView m={hd.S} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} selected={[sel.i, sel.j]} onCell={(i, j) => select({ i, j })} ariaLabel="Score matrix S" testid="mat-S" cell={52} />
        </Card>
      </div>
      <Card title={`S[${tokens[sel.i]}, ${tokens[sel.j]}]`}>
        <div className="overflow-x-auto scroll-thin text-[14px]"><Tex block tex={`S_{${sel.i + 1}${sel.j + 1}} = q_{${sel.i + 1}}\\cdot k_{${sel.j + 1}} = ${hd.q[sel.i].map((v, n) => `${tn(v)}\\times${tn(hd.k[sel.j][n])}`).join(' + ')} = ${tn(hd.S[sel.i][sel.j], 4)}`} /></div>
        <p className="text-[11.5px] text-muted">S is generally not symmetric: S₁₂ uses q₁ and k₂, S₂₁ uses q₂ and k₁ ({f3(hd.S[sel.i][sel.j])} vs {f3(hd.S[sel.j][sel.i])} here).</p>
      </Card>
    </div>
  )
}

function ScaleView({ trace }: { trace: BlockTrace }) {
  const c = useColors(), r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h], dk = hd.q[0].length
  // the variance of q·k for random unit-variance vectors, measured (seeded), for several d_k
  const demo = useMemo(() => [2, 8, 32, 128].map((n) => {
    const rr = rng(97 + n), g = () => { let u = 0; for (let t = 0; t < 12; t++) u += rr(); return u - 6 }
    const xs = Array.from({ length: 600 }, () => { const a = Array.from({ length: n }, g), b = Array.from({ length: n }, g); return vdot(a, b) })
    const m = xs.reduce((s, v) => s + v, 0) / xs.length, sd = Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / xs.length)
    return { n, sd, sdScaled: sd / Math.sqrt(n) }
  }), [])
  const row = hd.S[sel.i], rowS = hd.scaled[sel.i], allowed = hd.mask[sel.i].map((m) => m === 0)
  const soft = (z: number[]) => { const zz = z.map((v, j) => (allowed[j] ? v : -Infinity)), m = Math.max(...zz.filter(Number.isFinite)), e = zz.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }
  const pU = soft(row), pS = soft(rowS)
  const ent = (p: number[]) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0)
  return (
    <div className="grid gap-3">
      <HeadPicker H={trace.cfg.H} />
      <div className="flex flex-wrap items-start gap-3">
        <Card title="S (raw)"><MatrixView m={hd.S} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} hlRows={[sel.i]} onCell={(i, j) => select({ i, j })} ariaLabel="Raw scores" cell={50} /></Card>
        <span className="self-center font-mono">÷ √{dk} = ÷ {f3(Math.sqrt(dk))}</span>
        <Card title="S / √dₖ"><MatrixView m={hd.scaled} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} hlRows={[sel.i]} selected={[sel.i, sel.j]} onCell={(i, j) => select({ i, j })} ariaLabel="Scaled scores" testid="mat-scaled" cell={50} /></Card>
      </div>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`row “${tokens[sel.i]}”: softmax without vs with scaling (this toy has d_k = ${dk}, so the effect is mild)`}>
          {[{ p: pU, l: 'softmax(S)', e: ent(pU) }, { p: pS, l: 'softmax(S/√dₖ)', e: ent(pS) }].map((x) => (
            <div key={x.l} className="mb-1">
              <div className="font-mono text-[11.5px] text-muted">{x.l} · entropy {f3(x.e)} nats</div>
              <div className="flex gap-1">{x.p.map((v, j) => <div key={j} className="flex-1"><div className="h-12 rounded-t bg-panel-2"><div className="rounded-t" style={{ height: `${v * 100}%`, marginTop: `${(1 - v) * 48}px`, background: r.score }} /></div><div className="text-center font-mono text-[10px] text-faint">{f3(v)}</div></div>)}</div>
            </div>
          ))}
        </Card>
        <Card title="why √dₖ: dot products of random vectors grow with dₖ (measured, 600 samples each)">
          <table className="w-full font-mono text-[12px]">
            <thead className="text-left text-faint"><tr><th>d_k</th><th className="text-right">std of q·k</th><th className="text-right">√d_k</th><th className="text-right">std after ÷√d_k</th></tr></thead>
            <tbody>{demo.map((x) => <tr key={x.n} className="border-t border-line"><td>{x.n}</td><td className="text-right">{f3(x.sd)}</td><td className="text-right">{f3(Math.sqrt(x.n))}</td><td className="text-right" style={{ color: c.teal }}>{f3(x.sdScaled)}</td></tr>)}</tbody>
          </table>
          <p className="mt-1 text-[11.5px] text-muted">With unit-variance components, q·k has variance d_k. Dividing by √d_k brings it back to ≈ 1, so the softmax stays in a range where it is not almost one-hot and gradients do not vanish.</p>
        </Card>
      </div>
    </div>
  )
}

function MaskView({ trace }: { trace: BlockTrace }) {
  const c = useColors(), r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), arch = useTb((s) => s.arch), maskOff = useTb((s) => s.maskOff), setMaskOff = useTb((s) => s.setMaskOff)
  const hd = trace.heads[sel.h]
  const [pos, setPos] = useState(sel.i)
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <HeadPicker H={trace.cfg.H} />
        {arch === 'gpt' && (
          <label className="flex items-center gap-2 text-[12.5px]"><input type="checkbox" checked={maskOff} onChange={(e) => setMaskOff(e.target.checked)} data-testid="mask-toggle" />disable the causal mask (educational comparison)</label>
        )}
      </div>
      {arch === 'gpt' && maskOff && <div className="flex items-start gap-2 rounded-lg border border-rose/60 bg-rose/5 p-2 text-[12.5px]"><AlertTriangle size={15} className="mt-0.5 shrink-0 text-rose" />Mask disabled: every position now reads future tokens. This is no longer causal self-attention; a language model trained this way could copy the token it is supposed to predict, and it could not be used for left-to-right generation.</div>}
      {arch === 'bert' && <div className="rounded-lg border border-line bg-panel-2/50 p-2 text-[12.5px] text-muted">Encoder (BERT-style) attention is bidirectional: M is all zeros, every token sees every token. (Padding masks exist in practice but are not needed for a sequence without padding.)</div>}
      <div className="flex flex-wrap items-start gap-3">
        <Card title="S / √dₖ"><MatrixView m={hd.scaled} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} ariaLabel="Scaled scores" cell={50} /></Card>
        <span className="self-center font-mono text-xl">+</span>
        <Card title="M (mask)"><MatrixView m={hd.mask} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={c.rose} ariaLabel="Mask" testid="mat-M" cell={44} digits={0} /></Card>
        <span className="self-center font-mono text-xl">=</span>
        <Card title="L (softmax input)"><MatrixView m={hd.logits} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} selected={[sel.i, sel.j]} onCell={(i, j) => select({ i, j })} ariaLabel="Masked logits" testid="mat-L" cell={50} /></Card>
      </div>
      <Card title="causal mask explorer: what each position can see">
        <label className="flex items-center gap-2 text-[12.5px] text-muted">position <span className="w-6 font-mono text-fg">{pos + 1}</span><input type="range" min={0} max={tokens.length - 1} value={pos} onChange={(e) => setPos(+e.target.value)} className="flex-1" aria-label="Position" /></label>
        <div className="mt-2 flex flex-wrap gap-1">
          {tokens.map((t, j) => {
            const vis = hd.mask[pos][j] === 0
            return <motion.span key={j} layout className={`rounded-md border px-2 py-1 font-mono text-[12px] ${j === pos ? 'border-cyan text-cyan' : vis ? 'border-line-strong text-fg' : 'border-line text-faint line-through'}`} animate={{ opacity: vis ? 1 : 0.4 }}>{t}</motion.span>
          })}
        </div>
        <p className="mt-1 text-[11.5px] text-muted">Position {pos + 1} sees {hd.mask[pos].filter((m) => m === 0).length} of {tokens.length} positions. In a causal decoder the visible context grows by one with every position.</p>
      </Card>
    </div>
  )
}

function SoftmaxView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h], row = hd.rows[sel.i]
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2"><HeadPicker H={trace.cfg.H} />
        <span className="text-[12.5px] text-muted">row (query) <select className="rounded border border-line bg-panel-2 px-1" value={sel.i} onChange={(e) => select({ i: +e.target.value })} aria-label="Softmax row">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select></span>
      </div>
      <Card title={`stable softmax of row “${tokens[sel.i]}”, head ${sel.h + 1}`}>
        <div className="overflow-x-auto scroll-thin">
          <table className="font-mono text-[12.5px]" data-testid="softmax-table">
            <thead className="text-faint"><tr><th className="px-2 text-left">key</th>{tokens.map((t, j) => <th key={j} className="px-2 text-right">{t}</th>)}<th className="px-2 text-right">Σ</th></tr></thead>
            <tbody>
              <tr className="border-t border-line"><td className="px-2 text-muted">logit L</td>{hd.logits[sel.i].map((v, j) => <td key={j} className="px-2 text-right">{f3(v)}</td>)}<td /></tr>
              <tr className="border-t border-line"><td className="px-2 text-muted">L − max ({f3(row.max)})</td>{row.shifted.map((v, j) => <td key={j} className="px-2 text-right">{f3(v)}</td>)}<td /></tr>
              <tr className="border-t border-line"><td className="px-2 text-muted">exp(·)</td>{row.exps.map((v, j) => <td key={j} className="px-2 text-right">{f4(v)}</td>)}<td className="px-2 text-right" style={{ color: r.score }}>{f4(row.sum)}</td></tr>
              <tr className="border-t border-line"><td className="px-2 text-muted">÷ Σ = A</td>{row.p.map((v, j) => <td key={j} className="px-2 text-right" style={{ color: r.score }}>{f4(v)}</td>)}<td className="px-2 text-right">{f4(row.p.reduce((s, v) => s + v, 0))}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="mt-2 flex items-end gap-1" style={{ height: 70 }}>{row.p.map((v, j) => <motion.div key={`${sel.h}-${sel.i}-${j}`} className="flex-1 rounded-t" style={{ background: r.score }} initial={{ height: 0 }} animate={{ height: `${Math.max(1, v * 100)}%` }} transition={{ duration: 0.5, delay: j * 0.05 }} title={`${tokens[j]}: ${v.toFixed(4)}`} />)}</div>
        <p className="mt-1 text-[11.5px] text-muted">Subtracting the row maximum leaves the result unchanged (it cancels in the fraction) but keeps every exponent ≤ 0, so nothing overflows. e^(−∞) = 0: masked keys get exactly zero.</p>
      </Card>
      <Card title="check: every row of every head sums to 1">
        <div className="flex flex-wrap gap-3 font-mono text-[12px]" data-testid="row-sums">
          {trace.heads.map((h, k) => <div key={k}><div className="text-faint">head {k + 1}</div>{h.A.map((rw, i) => <div key={i}>{tokens[i].padEnd(6, ' ').slice(0, 6)} Σ = {rw.reduce((s, v) => s + v, 0).toFixed(6)} ✓</div>)}</div>)}
        </div>
      </Card>
    </div>
  )
}

function WeightsView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h]
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap gap-3">
        {trace.heads.map((h, k) => (
          <Card key={k} title={<button type="button" className={k === sel.h ? 'text-amber' : 'text-muted'} onClick={() => select({ h: k })}>head {k + 1}{h.enabled ? '' : ' (disabled downstream)'}</button>}>
            <MatrixView m={h.A} mode="prob" maxAbs={1} rowLabels={tokens} colLabels={tokens.map((_, n) => `k${n + 1}`)} color={r.score} selected={k === sel.h ? [sel.i, sel.j] : null} onCell={(i, j) => select({ h: k, i, j })} ariaLabel={`Attention weights head ${k + 1}`} testid={`mat-A${k}`} cell={50} />
          </Card>
        ))}
      </div>
      <Card title="selected cell">
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[12.5px] @xl:grid-cols-5" data-testid="cell-detail">
          <div><div className="text-faint">query</div>{tokens[sel.i]}</div><div><div className="text-faint">key</div>{tokens[sel.j]}</div>
          <div><div className="text-faint">raw score</div>{f4(hd.S[sel.i][sel.j])}</div><div><div className="text-faint">scaled</div>{f4(hd.logits[sel.i][sel.j])}</div>
          <div><div className="text-faint">probability</div><span style={{ color: r.score }}>{f4(hd.A[sel.i][sel.j])}</span></div>
        </div>
        <p className="mt-2 text-[11.5px] text-muted">Rows are queries, columns are keys. These toy weights are seeded random numbers, not trained, so the patterns carry no linguistic meaning; the Real model tab shows attention from a trained network. Even there, attention weights show where a head reads, not why the model decides something.</p>
      </Card>
    </div>
  )
}

function AVView({ trace }: { trace: BlockTrace }) {
  const c = useColors(), r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select)
  const hd = trace.heads[sel.h], a = hd.A[sel.i], dh = hd.v[0].length
  const contrib = hd.v.map((v, j) => v.map((x) => a[j] * x))
  const ma = Math.max(1e-9, ...hd.O[sel.i].map(Math.abs), ...contrib.flat().map(Math.abs))
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2"><HeadPicker H={trace.cfg.H} /><span className="text-[12.5px] text-muted">output for <select className="rounded border border-line bg-panel-2 px-1" value={sel.i} onChange={(e) => select({ i: +e.target.value })} aria-label="Output token">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select></span></div>
      <Card title={`O[“${tokens[sel.i]}”] = Σⱼ A[${sel.i + 1}, j] · v_j (head ${sel.h + 1})`}>
        <div className="grid gap-1.5">
          {hd.v.map((v, j) => (
            <motion.div key={`${sel.h}-${sel.i}-${j}`} className={`grid grid-cols-[90px_70px_minmax(0,1fr)] items-center gap-2 ${a[j] === 0 ? 'opacity-35' : ''}`} initial={{ opacity: 0, x: -6 }} animate={{ opacity: a[j] === 0 ? 0.35 : 1, x: 0 }} transition={{ delay: j * 0.08 }}>
              <span className="truncate font-mono text-[12px]">{tokens[j]}</span>
              <span className="font-mono text-[12px]" style={{ color: r.score }}>× {f3(a[j])}</span>
              <div className="flex flex-wrap items-center gap-2 font-mono text-[11.5px]">
                <span style={{ color: r.v }}>v = ({v.map(f3).join(', ')})</span>
                <span className="text-faint">→</span>
                <span>({contrib[j].map(f3).join(', ')})</span>
              </div>
            </motion.div>
          ))}
          <div className="grid grid-cols-[90px_70px_minmax(0,1fr)] items-center gap-2 border-t border-line pt-1.5 font-mono text-[12.5px]"><span>sum</span><span /><span style={{ color: r.v }} data-testid="av-sum">O = ({hd.O[sel.i].map(f4).join(', ')})</span></div>
        </div>
      </Card>
      <Card title="contribution of each key token to each output coordinate">
        <svg viewBox={`0 0 ${dh * 120} 150`} className="block w-full max-w-[520px]" role="img" aria-label="Stacked contributions">
          <line x1={0} x2={dh * 120} y1={75} y2={75} stroke={c.lineStrong} />
          {Array.from({ length: dh }, (_, cc) => {
            let up = 75, dn = 75
            return (
              <g key={cc}>
                {contrib.map((v, j) => {
                  const h = (Math.abs(v[cc]) / ma) * 60
                  const yy = v[cc] >= 0 ? (up -= h) : dn; if (v[cc] < 0) dn += h
                  return <rect key={j} x={cc * 120 + 30} y={yy} width={60} height={Math.max(0, h)} fill={headColors(r)[j % 8]} opacity={0.85}><title>{`${tokens[j]}: ${v[cc].toFixed(4)}`}</title></rect>
                })}
                <text x={cc * 120 + 60} y={148} textAnchor="middle" fontSize={11} fill={c.muted} fontFamily="var(--font-mono)">c{cc + 1} = {f3(hd.O[sel.i][cc])}</text>
              </g>
            )
          })}
        </svg>
        <div className="flex flex-wrap gap-2 text-[11px] text-muted">{tokens.map((t, j) => <span key={j} className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: headColors(r)[j % 8] }} />{t}</span>)}</div>
      </Card>
    </div>
  )
}

function ConcatView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), toggleHead = useTb((s) => s.toggleHead), iv = useTb((s) => s.iv)
  const { H, d } = trace.cfg, dh = d / H, hc = headColors(r)
  const off = trace.heads.flatMap((h, k) => (h.enabled ? [] : Array.from({ length: dh }, (_, c) => k * dh + c)))
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-start gap-2">
        {trace.heads.map((h, k) => (
          <Card key={k} title={<span style={{ color: hc[k] }}>O_{k + 1} [{trace.cfg.T}, {dh}]</span>} right={<label className="flex items-center gap-1 text-[11.5px]"><input type="checkbox" checked={iv.heads[k] !== false} onChange={() => toggleHead(k)} aria-label={`Head ${k + 1} enabled`} />on</label>}>
            <MatrixView m={h.O} rowLabels={k === 0 ? tokens : undefined} colLabels={h.O[0].map((_, c) => `c${c + 1}`)} color={hc[k]} ariaLabel={`Head ${k + 1} output`} cell={44} dimCols={h.enabled ? [] : h.O[0].map((_, c) => c)} />
          </Card>
        ))}
      </div>
      <Card title={`Concat(O₁…O_${H}) [${trace.cfg.T}, ${H} × ${dh} = ${d}] · coloured bars mark which head owns each column`}>
        <MatrixView m={trace.concat} rowLabels={tokens} colLabels={Array.from({ length: d }, (_, k) => `h${Math.floor(k / dh) + 1}.${(k % dh) + 1}`)} color={r.score} colGroup={dh} groupColors={hc} dimCols={off} ariaLabel="Concatenated heads" testid="mat-concat" />
        {off.length > 0 && <p className="mt-1 text-[12px] text-rose">Ablation: disabled heads contribute zeros. The output projection and everything downstream are recomputed.</p>}
      </Card>
    </div>
  )
}

function ProjView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), params = useTb((s) => s.params)
  const d = trace.cfg.d, cell = sel.cell?.m === 'Attn' ? sel.cell : { m: 'Attn', r: sel.i, c: sel.f }
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-start gap-3">
        <Card title="Concat"><MatrixView m={trace.concat} rowLabels={tokens} colLabels={fl(d)} color={r.score} hlRows={[cell.r]} ariaLabel="Concat" cell={42} /></Card>
        <span className="self-center font-mono text-xl">×</span>
        <Card title={`W_O [${d}, ${d}] · click to edit`}><MatrixView m={params.Wo} rowLabels={fl(d)} colLabels={fl(d)} color={r.score} digits={1} hlCols={[cell.c]} selected={sel.cell?.m === 'Wo' ? [sel.cell.r, sel.cell.c] : null} onCell={(rr, cc) => select({ cell: { m: 'Wo', r: rr, c: cc } })} ariaLabel="W_O" cell={40} /></Card>
        <span className="self-center font-mono text-xl">=</span>
        <Card title="Attn (sublayer output)"><MatrixView m={trace.attn} rowLabels={tokens} colLabels={fl(d)} color={r.score} selected={[cell.r, cell.c]} onCell={(rr, cc) => select({ i: rr, f: cc, cell: { m: 'Attn', r: rr, c: cc } })} ariaLabel="Attention output" testid="mat-attn" cell={46} /></Card>
      </div>
      <Card title="one output cell">
        <MatMulCell a={trace.concat} b={params.Wo} r={cell.r} c={cell.c} bias={params.bo[cell.c]} labelA="C" labelOut="\mathrm{Attn}" />
      </Card>
      <Card title={`attention sublayer: input vs output for “${tokens[cell.r]}”`}>
        <div className="grid gap-2 @xl:grid-cols-2">
          <div><div className="text-[11.5px] text-muted">input to attention ({trace.cfg.arch === 'pre' ? 'LN₁(X)' : 'X'})</div><VecBars v={trace.attnIn[cell.r]} color={r.tok} /></div>
          <div><div className="text-[11.5px] text-muted">attention output (context mixed in from {trace.heads[0].mask[cell.r].filter((m) => m === 0).length} permitted positions)</div><VecBars v={trace.attn[cell.r]} color={r.score} /></div>
        </div>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ residuals */
function ResidualView({ trace, which }: { trace: BlockTrace; which: 1 | 2 }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), iv = useTb((s) => s.iv), setIv = useTb((s) => s.setIv)
  const i = sel.i, post = trace.cfg.arch === 'post'
  const base = which === 1 ? trace.X : trace.H, contrib = which === 1 ? trace.attn : trace.mlp, scale = which === 1 ? iv.attnScale : iv.mlpScale
  const sum = which === 1 ? trace.sum1 : trace.sum2, out = which === 1 ? trace.H : trace.Y, ln = which === 1 ? trace.postLn1 : trace.postLn2
  const key = which === 1 ? 'attnScale' : 'mlpScale'
  const name = which === 1 ? 'attention' : 'MLP'
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
        token <select className="rounded border border-line bg-panel-2 px-1" value={i} onChange={(e) => select({ i: +e.target.value })} aria-label="Residual token">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select>
        <label className="ml-2 flex items-center gap-2">scale the {name} contribution <span className="w-10 font-mono text-fg">{scale.toFixed(2)}</span><input type="range" min={0} max={2} step={0.05} value={scale} onChange={(e) => setIv({ [key]: +e.target.value })} aria-label={`${name} contribution scale`} /></label>
        <label className="flex items-center gap-1"><input type="checkbox" checked={scale === 0} onChange={(e) => setIv({ [key]: e.target.checked ? 0 : 1 })} data-testid={`disable-${which}`} />disable it</label>
      </div>
      <Card title={`${which === 1 ? (post ? 'H = LN₁(X + Attn)' : 'H = X + Attn') : post ? 'Y = LN₂(H + MLP)' : 'Y = H + MLP'} · token “${tokens[i]}”`}>
        <div className="grid gap-2 @3xl:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto_minmax(0,1fr)] @3xl:items-center">
          <div><div className="text-[11.5px]" style={{ color: r.res }}>{which === 1 ? 'X (residual stream)' : 'H (residual stream)'}</div><VecBars v={base[i]} color={r.res} height={80} /></div>
          <span className="text-center font-mono text-xl">+</span>
          <div><div className="text-[11.5px]" style={{ color: which === 1 ? r.score : r.ffn }}>{scale !== 1 ? `${scale.toFixed(2)} × ` : ''}{name} contribution</div><VecBars v={contrib[i].map((x) => x * scale)} color={which === 1 ? r.score : r.ffn} height={80} /></div>
          <span className="text-center font-mono text-xl">=</span>
          <div><div className="text-[11.5px]" style={{ color: r.res }}>{post ? 'sum (then LayerNorm below)' : which === 1 ? 'H' : 'Y'}</div><VecBars v={sum[i]} color={r.res} height={80} testid={`res-sum-${which}`} /></div>
        </div>
        <table className="mt-2 font-mono text-[12px]" data-testid={`res-table-${which}`}>
          <thead className="text-faint"><tr><th className="px-2 text-left">feature</th>{base[i].map((_, f) => <th key={f} className="px-2 text-right">x{f + 1}</th>)}</tr></thead>
          <tbody>
            <tr className="border-t border-line"><td className="px-2 text-muted">stream</td>{base[i].map((v, f) => <td key={f} className="px-2 text-right">{f3(v)}</td>)}</tr>
            <tr className="border-t border-line"><td className="px-2 text-muted">+ {name}</td>{contrib[i].map((v, f) => <td key={f} className="px-2 text-right">{f3(v * scale)}</td>)}</tr>
            <tr className="border-t border-line"><td className="px-2 text-muted">= sum</td>{sum[i].map((v, f) => <td key={f} className="px-2 text-right" style={{ color: r.res }}>{f3(v)}</td>)}</tr>
            {post && <tr className="border-t border-line"><td className="px-2 text-muted">LN(sum)</td>{out[i].map((v, f) => <td key={f} className="px-2 text-right">{f3(v)}</td>)}</tr>}
          </tbody>
        </table>
      </Card>
      {post && ln && <LNMicroscope ln={ln} input={sum} which={which === 1 ? 'ln1' : 'ln2'} title={`post-LN: LayerNorm ${which === 1 ? '₁' : '₂'} applied after the addition · token “${tokens[i]}”`} />}
      {which === 2 && !post && <TokenTrace trace={trace} />}
      <p className="text-[12px] text-muted">The residual path carries the stream forward unchanged; the sublayer adds an update. Set the scale to 0 and the {name} sublayer is skipped entirely, yet the block still produces an output. Residual connections help gradients flow through deep stacks, but they do not by themselves guarantee stable training (normalization, initialization and learning rates matter too).</p>
    </div>
  )
}

function TokenTrace({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), iv = useTb((s) => s.iv)
  const i = sel.i
  const rows: [string, number[], string][] = [
    ['X (input)', trace.X[i], r.tok], ['+ attention', trace.attn[i].map((v) => v * iv.attnScale), r.score], ['= H', trace.H[i], r.res],
    ['+ MLP', trace.mlp[i].map((v) => v * iv.mlpScale), r.ffn], ['= Y (output)', trace.Y[i], r.res],
  ]
  return (
    <Card title={`complete trace of “${tokens[i]}” through the block`}>
      <table className="font-mono text-[12px]" data-testid="token-trace">
        <thead className="text-faint"><tr><th className="px-2 text-left" />{trace.X[i].map((_, f) => <th key={f} className="px-2 text-right">x{f + 1}</th>)}<th className="px-2 text-right">‖·‖</th></tr></thead>
        <tbody>{rows.map(([l, v, col]) => <tr key={l} className="border-t border-line"><td className="px-2" style={{ color: col }}>{l}</td>{v.map((x, f) => <td key={f} className="px-2 text-right">{f3(x)}</td>)}<td className="px-2 text-right text-muted">{f3(vnorm(v))}</td></tr>)}</tbody>
      </table>
      <p className="mt-1 text-[11.5px] text-muted">In a pre-LN block, Y = X + attention update + MLP update exactly (check any column).</p>
    </Card>
  )
}

/* ------------------------------------------------------------------ feedforward */
function MLPView({ trace, stage }: { trace: BlockTrace; stage: 'ffn1' | 'act' | 'ffn2' }) {
  const c = useColors(), r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), act = useTb((s) => s.act), setAct = useTb((s) => s.setAct), iv = useTb((s) => s.iv), toggleUnit = useTb((s) => s.toggleUnit), params = useTb((s) => s.params)
  const i = sel.i, { d, dff } = trace.cfg
  const u = trace.mlpIn[i], z = trace.Z1[i], a = trace.act[i], m = trace.mlp[i]
  const W = 560, Hh = Math.max(220, dff * 16 + 40)
  const yIn = (k: number) => 30 + ((Hh - 60) * (k + 0.5)) / d, yH = (k: number) => 30 + ((Hh - 60) * (k + 0.5)) / dff
  const xs = [60, W / 2, W - 60]
  const maW1 = Math.max(1e-9, ...params.W1.flat().map(Math.abs)), maW2 = Math.max(1e-9, ...params.W2.flat().map(Math.abs))
  const hv = stage === 'ffn1' ? z : a, maH = Math.max(1e-9, ...hv.map(Math.abs))
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
        token <select className="rounded border border-line bg-panel-2 px-1" value={i} onChange={(e) => select({ i: +e.target.value })} aria-label="MLP token">{tokens.map((t, n) => <option key={n} value={n}>{n + 1} · {t}</option>)}</select>
        <span className="ml-2">activation</span>
        <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="MLP activation">
          {(['relu', 'gelu', 'silu', 'swiglu'] as MlpAct[]).map((x) => <button key={x} type="button" role="radio" aria-checked={act === x} onClick={() => setAct(x)} className={`rounded px-2 py-0.5 text-[11.5px] ${act === x ? 'bg-raise text-cyan' : 'text-muted'}`}>{x === 'swiglu' ? 'SwiGLU (gated)' : x.toUpperCase()}</button>)}
        </div>
      </div>
      {act === 'swiglu' && <div className="rounded-lg border border-amber/50 bg-amber/5 p-2 text-[12px] text-muted">SwiGLU is a different MLP architecture, not only a different activation: it has a third matrix W₃ (the gate), and computes (SiLU(U W₁ + b₁) ⊙ (U W₃ + b₃)) W₂ + b₂. Used in LLaMA-style models.</div>}
      <Card title={`one token through the MLP: ${d} → ${dff} → ${d} · click a hidden unit to switch it off`}>
        <svg viewBox={`0 0 ${W} ${Hh}`} className="block w-full max-w-[620px]" role="img" aria-label="Position-wise MLP" data-testid="mlp-net">
          {u.map((_, k) => z.map((__, h) => <line key={`a${k}.${h}`} x1={xs[0]} y1={yIn(k)} x2={xs[1]} y2={yH(h)} stroke={params.W1[k][h] >= 0 ? r.ffn : c.violet} strokeWidth={0.4 + 1.6 * Math.abs(params.W1[k][h]) / maW1} opacity={stage === 'ffn1' ? 0.7 : 0.18} />))}
          {z.map((_, h) => m.map((__, k) => <line key={`b${h}.${k}`} x1={xs[1]} y1={yH(h)} x2={xs[2]} y2={yIn(k)} stroke={params.W2[h][k] >= 0 ? r.ffn : c.violet} strokeWidth={0.4 + 1.6 * Math.abs(params.W2[h][k]) / maW2} opacity={stage === 'ffn2' ? 0.7 : iv.units[h] === false ? 0.03 : 0.18} />))}
          {u.map((v, k) => <g key={`u${k}`}><circle cx={xs[0]} cy={yIn(k)} r={11} fill={c.panel2} stroke={r.tok} /><text x={xs[0] - 16} y={yIn(k) + 4} textAnchor="end" fontSize={10.5} fill={c.fg} fontFamily="var(--font-mono)">{f3(v)}</text></g>)}
          {z.map((_, h) => {
            const off = iv.units[h] === false, v = hv[h]
            return (
              <g key={`h${h}`} style={{ cursor: 'pointer' }} onClick={() => toggleUnit(h)} data-testid={`unit-${h}`}>
                <circle cx={xs[1]} cy={yH(h)} r={Math.min(7, (Hh - 60) / dff / 2 - 1)} fill={off ? c.bg : interpolate(c.panel2, r.ffn, Math.abs(v) / maH)} stroke={off ? c.rose : r.ffn} strokeDasharray={off ? '2 2' : undefined} />
                <text x={xs[1] + 12} y={yH(h) + 3.5} fontSize={9.5} fill={off ? c.rose : c.muted} fontFamily="var(--font-mono)">{off ? 'off' : `${stage === 'ffn1' ? 'z' : 'φ'} ${f3(v)}`}</text>
              </g>
            )
          })}
          {m.map((v, k) => <g key={`m${k}`}><circle cx={xs[2]} cy={yIn(k)} r={11} fill={stage === 'ffn2' ? interpolate(c.panel2, r.ffn, Math.abs(v) / Math.max(1e-9, ...m.map(Math.abs))) : c.panel2} stroke={r.ffn} /><text x={xs[2] + 16} y={yIn(k) + 4} fontSize={10.5} fill={stage === 'ffn2' ? c.fg : c.faint} fontFamily="var(--font-mono)">{f3(v)}</text></g>)}
          <text x={xs[0]} y={16} textAnchor="middle" fontSize={11} fill={c.muted}>U (d = {d})</text>
          <text x={xs[1]} y={16} textAnchor="middle" fontSize={11} fill={stage !== 'ffn2' ? r.ffn : c.muted}>{stage === 'ffn1' ? 'Z₁ = U W₁ + b₁' : 'φ(Z₁)'} (d_ff = {dff})</text>
          <text x={xs[2]} y={16} textAnchor="middle" fontSize={11} fill={stage === 'ffn2' ? r.ffn : c.muted}>MLP output (d = {d})</text>
        </svg>
        <MLPFormula trace={trace} stage={stage} i={i} />
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <ActCurve act={act} z={z} />
        <Card title="the same MLP for every position (shared weights, no mixing)">
          <MatrixView m={stage === 'ffn1' ? trace.Z1 : stage === 'act' ? trace.act : trace.mlp} rowLabels={tokens} colLabels={(stage === 'ffn2' ? m : z).map((_, k) => `${k + 1}`)} color={r.ffn} hlRows={[i]} onCell={(rr) => select({ i: rr })} ariaLabel="MLP for all tokens" cell={stage === 'ffn2' ? 44 : 30} digits={stage === 'ffn2' ? 2 : 1} />
          <p className="mt-1 text-[11.5px] text-muted">Row t depends only on row t of the input: attention mixes information across positions, the MLP transforms each position on its own. (Advanced tab: a measured dependency map makes this visible.)</p>
        </Card>
      </div>
    </div>
  )
}
const interpolate = (a: string, b: string, t: number) => { const f = (h: string) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)); try { const A = f(a), B = f(b); return `rgb(${A.map((x, k) => Math.round(x + (B[k] - x) * Math.min(1, Math.sqrt(t)))).join(',')})` } catch { return b } }

function MLPFormula({ trace, stage, i }: { trace: BlockTrace; stage: 'ffn1' | 'act' | 'ffn2'; i: number }) {
  const params = useTb((s) => s.params), u = useTb((s) => s.sel.u), sf = useTb((s) => s.sel.f), select = useTb((s) => s.select)
  const { dff, d } = trace.cfg, act = trace.cfg.act
  const h = Math.min(u, dff - 1), f = Math.min(sf, d - 1)
  const z = trace.Z1[i][h]
  return (
    <div className="mt-2 grid gap-1 overflow-x-auto scroll-thin text-[13px]">
      <div className="flex flex-wrap items-center gap-1 text-[11.5px] text-muted">hidden unit <select className="rounded border border-line bg-panel-2 px-1" value={h} onChange={(e) => select({ u: +e.target.value })} aria-label="Hidden unit">{Array.from({ length: dff }, (_, k) => <option key={k} value={k}>{k + 1}</option>)}</select></div>
      {stage === 'ffn1' && <MatMulCell a={trace.mlpIn} b={params.W1} r={i} c={h} bias={params.b1[h]} labelA="U" labelOut="Z_1" />}
      {stage === 'act' && <Tex block tex={act === 'swiglu' ? `\\mathrm{SiLU}(${tn(z)})\\cdot g = ${tn(ACT_FN.silu(z), 4)}\\cdot${tn(trace.gate![i][h], 4)} = ${tn(trace.act[i][h], 4)}` : `\\varphi(${tn(z)}) = ${tn(trace.act[i][h], 4)}\\quad(${act === 'relu' ? '\\max(0,z)' : act === 'gelu' ? 'z\\,\\Phi(z) = \\tfrac{z}{2}(1+\\mathrm{erf}(z/\\sqrt2))' : 'z\\,\\sigma(z)'})`} />}
      {stage === 'ffn2' && <div className="flex flex-wrap items-center gap-1 text-[11.5px] text-muted">output feature <select className="rounded border border-line bg-panel-2 px-1" value={f} onChange={(e) => select({ f: +e.target.value })} aria-label="Output feature">{Array.from({ length: d }, (_, k) => <option key={k} value={k}>x{k + 1}</option>)}</select></div>}
      {stage === 'ffn2' && <MatMulCell a={trace.act} b={params.W2} r={i} c={f} bias={params.b2[f]} labelA="\varphi" labelOut="\mathrm{MLP}" />}
    </div>
  )
}

function ActCurve({ act, z }: { act: MlpAct; z: number[] }) {
  const c = useColors(), r = useRoles()
  const f = act === 'swiglu' ? ACT_FN.silu : ACT_FN[act]
  const W = 300, H = 150, sx = (x: number) => ((x + 4) / 8) * W, sy = (y: number) => H - 20 - ((y + 1) / 5) * (H - 30)
  const pts = Array.from({ length: 81 }, (_, k) => -4 + k * 0.1)
  return (
    <Card title={`${act === 'swiglu' ? 'SiLU (the SwiGLU gate activation)' : act.toUpperCase()} with this token's z values`}>
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full max-w-[360px]" role="img" aria-label="Activation curve">
        <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} stroke={c.line} /><line x1={sx(0)} x2={sx(0)} y1={0} y2={H} stroke={c.line} />
        <path d={pts.map((x, k) => `${k ? 'L' : 'M'}${sx(x)},${sy(f(x))}`).join('')} fill="none" stroke={r.ffn} strokeWidth={2} />
        {z.filter((v) => v > -4 && v < 4).map((v, k) => <circle key={k} cx={sx(v)} cy={sy(f(v))} r={3.5} fill={c.amber} />)}
      </svg>
    </Card>
  )
}

/* ------------------------------------------------------------------ output */
function OutputView({ trace }: { trace: BlockTrace }) {
  const r = useRoles()
  const tokens = useTb((s) => s.tokens), sel = useTb((s) => s.sel), select = useTb((s) => s.select), setTab = useTb((s) => s.setTab)
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-start gap-3">
        <Card title="X (input)"><MatrixView m={trace.X} rowLabels={tokens} colLabels={fl(trace.cfg.d)} color={r.tok} hlRows={[sel.i]} onCell={(i) => select({ i })} ariaLabel="Input" cell={44} /></Card>
        <span className="self-center font-mono text-xl">→</span>
        <Card title="Y (output) · same shape"><MatrixView m={trace.Y} rowLabels={tokens} colLabels={fl(trace.cfg.d)} color={r.tok} hlRows={[sel.i]} onCell={(i) => select({ i })} ariaLabel="Output hidden states" testid="mat-Y" cell={44} /></Card>
      </div>
      <Card title="how much each token changed">
        <table className="font-mono text-[12px]">
          <thead className="text-faint"><tr><th className="px-2 text-left">token</th><th className="px-2 text-right">‖X‖</th><th className="px-2 text-right">‖Y‖</th><th className="px-2 text-right">‖Y − X‖</th><th className="px-2 text-right">cos(X, Y)</th><th className="px-2 text-right">positions attended</th></tr></thead>
          <tbody>{tokens.map((t, i) => { const x = trace.X[i], y = trace.Y[i]; return (
            <tr key={i} className={`border-t border-line ${i === sel.i ? 'text-cyan' : ''}`}><td className="px-2">{t}</td><td className="px-2 text-right">{f3(vnorm(x))}</td><td className="px-2 text-right">{f3(vnorm(y))}</td><td className="px-2 text-right">{f3(vnorm(y.map((v, k) => v - x[k])))}</td><td className="px-2 text-right">{f3(vdot(x, y) / (vnorm(x) * vnorm(y) || 1))}</td><td className="px-2 text-right">{trace.heads[0].mask[i].filter((m) => m === 0).length}</td></tr>) })}</tbody>
        </table>
      </Card>
      <AnimatePresence>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-lg border border-amber/50 bg-amber/5 p-3 text-[12.5px] leading-relaxed">
          <b>Y is a hidden state, not a prediction.</b> To get next-token probabilities a language model applies, after the last block, a final LayerNorm, the unembedding matrix (the LM head, d → vocabulary) and a softmax. The <button type="button" className="text-cyan underline decoration-dotted" onClick={() => setTab('real')}>Real model</button> tab shows that whole path in a trained network.
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
