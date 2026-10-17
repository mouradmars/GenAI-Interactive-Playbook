/** Stages 3–6: the Q/K/V projection factory, the dot-product microscope, scaling and masking. */
import { motion, useReducedMotion } from 'framer-motion'
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { useColors } from '../../sim/appState'
import { dotTerms, matmulCell, scalingExperiment, stableSoftmax, entropy, type AttnTrace } from '../../engine/attention'
import { EditableMatrix, GLYPH, Grid, HeadPicker, RoleTag, Seg, fx, pct, tokLabel, useRoles, useTokens, type Role } from './common'
import { TokenGraph } from './TokenGraph'
import { TokenCards } from './viewsInput'
import { BaselineDiff } from './BaselineDiff'
import { D_MODEL, attnStore, useAttn, type WKey } from './store'

const feat = Array.from({ length: D_MODEL }, (_, k) => `f${k + 1}`)
const dims = (n: number) => Array.from({ length: n }, (_, k) => `c${k + 1}`)
const pn = (v: number) => (v < 0 ? `(${fx(v, 2)})` : fx(v, 2))

/** animated term-by-term sum: a₁b₁ + a₂b₂ + … = result */
export function TermSum({ a, b, roleA, roleB, result, label, testid }: { a: number[]; b: number[]; roleA: Role; roleB: Role; result: string; label: string; testid?: string }) {
  const r = useRoles(), reduce = useReducedMotion()
  const { products, sum } = dotTerms(a, b)
  const running = products.map((_, k) => products.slice(0, k + 1).reduce((s, v) => s + v, 0))
  return (
    <div className="grid gap-1.5" data-testid={testid}>
      <div className="font-mono text-[12px] text-muted">{label}</div>
      <div className="flex flex-wrap items-center gap-1 font-mono text-[12.5px]">
        {products.map((p, k) => (
          <motion.span key={`${k}-${a[k]}-${b[k]}`} initial={reduce ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: k * 0.18 }} className="flex items-center gap-1">
            {k > 0 && <span className="text-faint">+</span>}
            <span className="rounded border border-line bg-panel-2 px-1.5 py-0.5" title={`term ${k + 1}: ${a[k]} × ${b[k]} = ${p}`}>
              <span style={{ color: r[roleA] }}>{pn(a[k])}</span><span className="text-faint">×</span><span style={{ color: r[roleB] }}>{pn(b[k])}</span>
            </span>
          </motion.span>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-1 font-mono text-[12.5px]">
        <span className="text-faint">=</span>
        {products.map((p, k) => <motion.span key={`p${k}-${p}`} initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 + k * 0.18 }}>{k > 0 && <span className="text-faint">+ </span>}{fx(p, 3)}</motion.span>)}
      </div>
      <div className="flex flex-wrap items-center gap-2 font-mono text-[12px] text-faint">
        running sum: {running.map((v, k) => <motion.span key={`r${k}-${v}`} initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + k * 0.18 }}>{fx(v, 3)}{k < running.length - 1 ? ' →' : ''}</motion.span>)}
      </div>
      <motion.div key={`s${sum}`} initial={reduce ? false : { scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.25 + products.length * 0.18 }} className="font-mono text-[14px]">
        <span className="text-faint">= </span><b>{result}</b> <span className="text-fg">= {fx(sum, 4)}</span>
      </motion.div>
    </div>
  )
}

/* ------------------------------------------------------------------ 3 · Q/K/V factory */
const PATHS: { key: 'Q' | 'K' | 'V'; w: Exclude<WKey, 'Wo'>; role: Role; name: string; what: string }[] = [
  { key: 'Q', w: 'Wq', role: 'q', name: 'Query', what: 'used to calculate how strongly this position attends to other positions' },
  { key: 'K', w: 'Wk', role: 'k', name: 'Key', what: 'used to determine compatibility with queries' },
  { key: 'V', w: 'Wv', role: 'v', name: 'Value', what: 'the information combined according to the attention weights' },
]
const PHASES = ['select a token', 'its input vector', 'query pathway', 'key pathway', 'value pathway']

export function QKVView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select)
  const st = attnStore.getState()
  const [phase, setPhase] = useState(4)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { if (!playing) return; if (phase >= 4) { setPlaying(false); return } const id = setTimeout(() => setPhase(phase + 1), 1100); return () => clearTimeout(id) }, [playing, phase])
  const h = sel.h, hd = trace.heads[h], dk = trace.dk, i = sel.i
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const cell = sel.cell && ['Q', 'K', 'V'].includes(sel.cell.m) ? sel.cell : { m: 'Q', r: i, c: 0 }
  const path = PATHS.find((p) => p.key === cell.m)!
  const W = hd[`W${path.key.toLowerCase()}` as 'Wq' | 'Wk' | 'Wv']
  const calc = matmulCell(trace.X, W, cell.r, cell.c)
  return (
    <div className="grid gap-3">
      <Card title="the projection factory · one input row, three learned matrices" right={<HeadPicker />}>
        <div className="flex flex-wrap items-center gap-2">
          <TokenCards showKey={false} />
          <div className="ml-auto flex items-center gap-1" role="group" aria-label="Projection animation">
            <button type="button" className="btn" onClick={() => { if (phase >= 4) setPhase(0); setPlaying(!playing) }} aria-label={playing ? 'Pause projection animation' : 'Play projection animation'}>{playing ? <Pause size={14} /> : <Play size={14} />}{playing ? 'Pause' : 'Animate'}</button>
            <button type="button" className="btn icon-btn" onClick={() => setPhase(Math.min(4, phase + 1))} aria-label="Next projection step"><SkipForward size={14} /></button>
            <button type="button" className="btn icon-btn" onClick={() => { setPlaying(false); setPhase(0) }} aria-label="Restart projection animation"><RotateCcw size={14} /></button>
            <span className="font-mono text-[11.5px] text-muted" aria-live="polite">{phase + 1}/5 · {PHASES[phase]}</span>
          </div>
        </div>
      </Card>
      <div className="grid gap-3 @5xl:grid-cols-3">
        {PATHS.map((p, k) => {
          const on = phase === 4 || phase === k + 2
          const out = hd[p.key]
          const Wp = hd[p.w]
          const isCell = cell.m === p.key
          return (
            <Card key={p.key} className={`transition-opacity ${on ? '' : 'opacity-35'}`} title={<span><RoleTag role={p.role}>{p.key}{trace.cfg.H > 1 ? `_${h + 1}` : ''}</RoleTag> = <RoleTag role="x">X</RoleTag> W_{p.key}{trace.cfg.H > 1 ? `^(${h + 1})` : ''} · {trace.Tq}×{D_MODEL} · {D_MODEL}×{dk} → {trace.Tq}×{dk}</span>}>
              <div className="flex flex-wrap items-start gap-2">
                <Grid m={trace.X} role="x" colLabels={feat} rowLabels={rl} hlRows={phase >= 1 ? [isCell ? cell.r : i] : []} cell={40} ariaLabel="X" digits={1} />
                <span className="self-center font-mono text-faint">×</span>
                <Grid m={Wp} role="w" colLabels={dims(dk)} hlCols={on && isCell ? [cell.c] : []} cell={40} ariaLabel={`W_${p.key} of head ${h + 1}`} digits={1} />
                <span className="self-center font-mono text-faint">=</span>
                <Grid m={out} role={p.role} colLabels={dims(dk)} rowLabels={tokens.map((_, kk) => String(kk + 1))} hlRows={phase >= 1 && on ? [i] : []} selected={isCell ? [cell.r, cell.c] : null} cell={46}
                  onCell={(rr, c) => select({ i: rr, cell: { m: p.key, r: rr, c } })} ariaLabel={`${p.name} matrix of head ${h + 1}`} testid={`g-${p.key}`} />
              </div>
              <p className="mt-1 text-[11.5px] text-muted"><b style={{ color: r[p.role] }}>{p.name}</b>: a learned projection {p.what}.</p>
            </Card>
          )
        })}
      </div>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <Card title={`matrix-multiplication microscope · ${cell.m.toLowerCase()}${cell.r + 1},${cell.c + 1} = row ${cell.r + 1} of X · column ${cell.c + 1} of W_${cell.m}${trace.cfg.H > 1 ? `^(${h + 1})` : ''}`}>
          <TermSum key={`${cell.m}${cell.r}${cell.c}${h}`} a={calc.a} b={calc.b} roleA="x" roleB="w" label={`${cell.m.toLowerCase()}_{${cell.r + 1},${cell.c + 1}} = Σₖ x_{${cell.r + 1},k} · w_{k,${cell.c + 1}}`} result={`${cell.m.toLowerCase()}${cell.r + 1},${cell.c + 1}`} testid="qkv-calc" />
          <p className="mt-2 text-[11.5px] text-faint">Click any cell of Q, K or V above to inspect it.</p>
        </Card>
        <Card title="what Q, K and V are (and are not)">
          <p className="text-[12.5px] leading-relaxed text-muted">An intuitive analogy: a query <i>seeks</i>, a key is <i>matched</i> against queries, a value is the content <i>retrieved</i>. But Q, K and V are learned numerical vectors, not literal questions or human-readable labels; here they are produced by hand-chosen toy matrices.</p>
          <div className="mt-2 rounded-md border-l-2 border-amber bg-amber/5 px-2 py-1.5 text-[12px]">
            <b>Experiment:</b> change a value weight while holding Q and K fixed. The attention weights must stay identical; only the output can change.
            <div className="mt-1 flex flex-wrap gap-1">
              <button type="button" className="btn !h-7" onClick={() => { st.saveBaseline('before editing W_V'); st.setW('Wv', 0, h * dk, attnStore.getState().params.Wv[0][h * dk] + 1) }} data-testid="try-v">Try it: add 1 to W_V[1, {h * dk + 1}]</button>
              <button type="button" className="btn !h-7" onClick={() => { st.saveBaseline('before editing W_Q'); st.setW('Wq', 0, h * dk, attnStore.getState().params.Wq[0][h * dk] + 1) }}>…or to W_Q[1, {h * dk + 1}]</button>
            </div>
          </div>
          <BaselineDiff trace={trace} />
        </Card>
      </div>
      <Card title={`edit head ${h + 1}'s own projection matrices (columns ${h * dk + 1}–${h * dk + dk} of W_Q, W_K, W_V)`}>
        <div className="grid gap-3 @3xl:grid-cols-3">
          {PATHS.map((p) => (
            <div key={p.key}>
              <div className="mb-1 font-mono text-[11.5px]" style={{ color: r[p.role] }}>{GLYPH[p.role]} W_{p.key}^({h + 1})</div>
              <EditableMatrix m={hd[p.w]} onSet={(rr, c, v) => st.setW(p.w, rr, h * dk + c, v)} rowLabels={feat} colLabels={dims(dk)} role={p.role} label={`W_${p.key} head ${h + 1}`} testid={`edit-${p.w}`} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ 4 · dot products */
export function DotView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), mode = useAttn((s) => s.graphMode), setMode = useAttn((s) => s.setGraphMode)
  const hd = trace.heads[sel.h], i = sel.i, j = sel.j
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card title="microscope A · every query compared with every key" right={
          <div className="flex flex-wrap items-center gap-1"><HeadPicker /><Seg label="Connection encoding" value={mode} onChange={setMode} options={[{ v: 'raw', label: 'raw scores' }, { v: 'prob', label: 'normalized' }]} /></div>
        }>
          <TokenGraph trace={trace} tokens={tokens} mode={mode} />
          {mode === 'prob' && <p className="mt-1 text-[11.5px] text-amber">Normalized view shows the weights after scaling, masking and softmax (stage 7). Raw dot products are not probabilities.</p>}
        </Card>
        <Card title={<span>microscope B · <RoleTag role="score">S = QKᵀ</RoleTag> · rows = queries, columns = keys</span>}>
          <Grid m={hd.S} role="score" rowLabels={rl.map((l) => `q${l}`)} colLabels={tokens.map((_, k) => `k${k + 1}`)} selected={[i, j]} hlRows={[i]} onCell={(a, b) => select({ i: a, j: b, cell: null })} ariaLabel="Score matrix S" testid="g-S" />
          <p className="text-[11.5px] text-muted">s<sub>{i + 1}{j + 1}</sub> = {fx(hd.S[i][j], 4)} but s<sub>{j + 1}{i + 1}</sub> = {fx(hd.S[j]?.[i] ?? NaN, 4)}: queries and keys use different matrices, so S is not symmetric.</p>
        </Card>
      </div>
      <Card title={`the dot product for query ${i + 1} (“${tokens[i]}”) and key ${j + 1} (“${tokens[j]}”)`}>
        <div className="grid gap-3 @3xl:grid-cols-[auto_minmax(0,1fr)]">
          <AlignedVectors q={hd.Q[i]} k={hd.K[j]} />
          <TermSum key={`${i}${j}${sel.h}`} a={hd.Q[i]} b={hd.K[j]} roleA="q" roleB="k" label={`s_${i + 1}${j + 1} = q${i + 1} · k${j + 1}`} result={`s${i + 1}${j + 1}`} testid="dot-calc" />
        </div>
        <p className="mt-2 text-[11.5px] text-faint"><span style={{ color: r.q }}>{GLYPH.q} q{i + 1}</span> = row {i + 1} of Q, <span style={{ color: r.k }}>{GLYPH.k} k{j + 1}</span> = row {j + 1} of K. Click a connection or a cell to change the pair.</p>
      </Card>
    </div>
  )
}
function AlignedVectors({ q, k }: { q: number[]; k: number[] }) {
  const r = useRoles(), c = useColors()
  const n = q.length, cw = 58, W = 70 + n * cw
  const prod = q.map((v, d) => v * k[d])
  const row = (label: string, v: number[], col: string, y: number) => (
    <g>
      <text x={0} y={y + 18} fontSize={11.5} fill={col} fontFamily="var(--font-mono)">{label}</text>
      {v.map((x, d) => <g key={d}><rect x={64 + d * cw} y={y} width={cw - 6} height={26} rx={5} fill={c.panel2} stroke={col} /><text x={64 + d * cw + (cw - 6) / 2} y={y + 17} textAnchor="middle" fontSize={11.5} fill={c.fg} fontFamily="var(--font-mono)">{fx(x, 2)}</text></g>)}
    </g>
  )
  return (
    <svg viewBox={`0 0 ${W} 132`} width={W} className="block max-w-full" role="img" aria-label="Query and key aligned by dimension">
      {row(`${GLYPH.q} q`, q, r.q, 4)}
      {row(`${GLYPH.k} k`, k, r.k, 38)}
      {q.map((_, d) => <text key={d} x={64 + d * cw + (cw - 6) / 2} y={80} textAnchor="middle" fontSize={11} fill={c.faint} fontFamily="var(--font-mono)">×  ↓</text>)}
      {row('q⊙k', prod, r.score, 92)}
    </svg>
  )
}

/* ------------------------------------------------------------------ 5 · scaling */
const DK_PRESETS = [1, 2, 4, 8, 16, 64, 256, 1024]
export function ScaleView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), scale = useAttn((s) => s.scale)
  const st = attnStore.getState()
  const [dk, setDk] = useState(64)
  const [seed, setSeed] = useState(1)
  const [mult, setMult] = useState(1)
  const res = useMemo(() => scalingExperiment(dk, seed), [dk, seed])
  const sweep = useMemo(() => DK_PRESETS.map((d) => scalingExperiment(d, seed, 200)), [seed])
  const hd = trace.heads[sel.h], i = sel.i
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const rowZ = hd.logits[i].map((v) => (Number.isFinite(v) ? v * mult : v))
  const sm = stableSoftmax(rowZ)
  return (
    <div className="grid gap-3">
      <Card title="in the live model" right={<div className="flex items-center gap-2"><HeadPicker /><Seg label="Scaling" value={scale ? 'on' : 'off'} onChange={(v) => st.setScale(v === 'on')} options={[{ v: 'on', label: '÷ √dₖ on' }, { v: 'off', label: 'off' }]} /></div>}>
        <div className="flex flex-wrap items-start gap-3">
          <Grid m={hd.S} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="S = QKᵀ (raw)" selected={[i, sel.j]} onCell={(a, b) => select({ i: a, j: b })} ariaLabel="Raw scores" cell={46} />
          <span className="self-center font-mono text-sm text-faint">{scale ? `÷ √${trace.dk} = ÷ ${fx(Math.sqrt(trace.dk), 3)}` : '× 1'}</span>
          <Grid m={hd.scaled} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title={scale ? 'S̃ = S / √dₖ' : 'S̃ = S (scaling off)'} selected={[i, sel.j]} onCell={(a, b) => select({ i: a, j: b })} ariaLabel="Scaled scores" cell={46} testid="g-Ss" />
        </div>
        <p className="mt-1 text-[12px] text-muted">s̃<sub>{i + 1}{sel.j + 1}</sub> = {fx(hd.S[i][sel.j], 4)} {scale ? `/ ${fx(Math.sqrt(trace.dk), 4)}` : ''} = {fx(hd.scaled[i][sel.j], 4)}. Scaling never changes the order of the scores within a row; it changes how peaked the softmax will be.</p>
      </Card>
      <Card title="experiment · random queries and keys with i.i.d. N(0, 1) entries (a common modelling assumption)" right={
        <div className="flex flex-wrap items-center gap-2">
          <Seg label="Head dimension d_k" value={dk} onChange={setDk} options={DK_PRESETS.map((d) => ({ v: d, label: String(d) }))} />
          <label className="flex items-center gap-1 text-[12px] text-muted">seed<input type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, +e.target.value || 0))} className="w-14 rounded border border-line px-1 font-mono" aria-label="Sampling seed" /></label>
        </div>
      }>
        <div className="grid gap-3 @3xl:grid-cols-2">
          <div>
            <Histogram values={res.raw} color={r.score} label={`raw q·k · sample variance ${fx(res.rawVar, 2)} (theory: d_k = ${dk})`} />
            <Histogram values={res.scaled} color={r.prob} label={`scaled q·k/√d_k · variance ${fx(res.scaledVar, 3)} (theory: 1)`} />
          </div>
          <div className="grid content-start gap-2">
            <table className="w-full font-mono text-[12px]" data-testid="scale-table">
              <thead><tr className="text-faint"><th className="text-left font-normal">over 8 keys, mean of 400 rows</th><th className="font-normal">unscaled</th><th className="font-normal">scaled</th></tr></thead>
              <tbody>
                <tr><td className="text-muted">entropy (max ln 8 = 2.079)</td><td className="text-center">{fx(res.entropyRaw, 3)}</td><td className="text-center">{fx(res.entropyScaled, 3)}</td></tr>
                <tr><td className="text-muted">largest probability</td><td className="text-center">{pct(res.maxPRaw)}</td><td className="text-center">{pct(res.maxPScaled)}</td></tr>
                <tr><td className="text-muted">‖softmax Jacobian‖ (gradient size)</td><td className="text-center">{fx(res.jacRaw, 3)}</td><td className="text-center">{fx(res.jacScaled, 3)}</td></tr>
              </tbody>
            </table>
            <div className="grid grid-cols-2 gap-2">
              <MiniBars p={res.exampleRaw} color={r.score} label="one row, unscaled" />
              <MiniBars p={res.exampleScaled} color={r.prob} label="same row, scaled" />
            </div>
            <SweepChart sweep={sweep} />
          </div>
        </div>
        <p className="mt-2 text-[12px] leading-relaxed text-muted">If q and k have independent entries with mean 0 and variance 1, then Var(q·k) = Σ Var(q_c k_c) = d_k. Without scaling, larger heads give larger logits, softmax approaches one-hot and its Jacobian diag(p) − ppᵀ (and with it the gradient) shrinks. Dividing by √d_k restores unit variance. This is an argument about initialization statistics, not a guarantee against every training instability.</p>
      </Card>
      <Card title={`softmax sensitivity · multiply query ${i + 1}'s logits by a factor`}>
        <label className="flex items-center gap-2 font-mono text-[12px] text-muted">factor ×{mult.toFixed(1)}
          <input type="range" min={0} max={8} step={0.1} value={mult} onChange={(e) => setMult(+e.target.value)} className="w-56" aria-label="Logit multiplier" />
        </label>
        <div className="mt-2 grid gap-1">
          {sm.p.map((p, k) => (
            <div key={k} className="grid grid-cols-[90px_1fr_70px_70px] items-center gap-2 font-mono text-[12px]">
              <span className="truncate text-right">{tokLabel(tokens, k)}</span>
              <div className="h-3 rounded bg-panel-2"><div className="h-3 rounded transition-[width]" style={{ width: `${p * 100}%`, background: sm.allowed[k] ? r.prob : 'transparent' }} /></div>
              <span className="text-right text-faint">{fx(rowZ[k], 2)}</span>
              <span className="text-right">{sm.allowed[k] ? pct(p) : '⊘'}</span>
            </div>
          ))}
        </div>
        <p className="mt-1 text-[11.5px] text-faint">entropy {fx(entropy(sm.p), 3)} nats · factor 0 gives a uniform distribution over the permitted keys; large factors approach one-hot. (Equivalent to a softmax temperature 1/factor.)</p>
      </Card>
    </div>
  )
}
function Histogram({ values, color, label }: { values: number[]; color: string; label: string }) {
  const c = useColors()
  const lim = Math.max(3, ...values.map(Math.abs)) , bins = 41, W = 360, H = 74
  const counts = new Array(bins).fill(0)
  values.forEach((v) => { const b = Math.min(bins - 1, Math.max(0, Math.floor(((v + lim) / (2 * lim)) * bins))); counts[b]++ })
  const mx = Math.max(...counts)
  return (
    <div className="mb-2">
      <div className="font-mono text-[11px] text-muted">{label}</div>
      <svg viewBox={`0 0 ${W} ${H + 14}`} className="block w-full max-w-[460px]" role="img" aria-label={label}>
        {counts.map((n, b) => <rect key={b} x={(b / bins) * W} y={H - (n / mx) * H} width={W / bins - 1} height={(n / mx) * H} fill={color} opacity={0.85} />)}
        <line x1={W / 2} x2={W / 2} y1={0} y2={H} stroke={c.lineStrong} strokeDasharray="3 3" />
        <text x={2} y={H + 12} fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">−{lim.toFixed(1)}</text>
        <text x={W / 2} y={H + 12} textAnchor="middle" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">0</text>
        <text x={W - 2} y={H + 12} textAnchor="end" fontSize={10} fill={c.faint} fontFamily="var(--font-mono)">{lim.toFixed(1)}</text>
      </svg>
    </div>
  )
}
function MiniBars({ p, color, label }: { p: number[]; color: string; label: string }) {
  return (
    <div>
      <div className="font-mono text-[10.5px] text-faint">{label}</div>
      <div className="flex h-14 items-end gap-0.5">{p.map((v, k) => <div key={k} className="flex-1 rounded-t" style={{ height: `${Math.max(2, v * 100)}%`, background: color }} title={pct(v)} />)}</div>
    </div>
  )
}
function SweepChart({ sweep }: { sweep: ReturnType<typeof scalingExperiment>[] }) {
  const r = useRoles(), c = useColors()
  const W = 360, H = 110, L = 30, B = 18
  const x = (k: number) => L + (k / (sweep.length - 1)) * (W - L - 20)
  const y = (e: number) => 6 + (1 - e / Math.log(8)) * (H - B - 6)
  const line = (f: (s: ScalingRes) => number) => sweep.map((s, k) => `${k ? 'L' : 'M'}${x(k)},${y(f(s))}`).join(' ')
  return (
    <div>
      <div className="font-mono text-[11px] text-muted">mean softmax entropy vs d_k (<span style={{ color: r.score }}>unscaled</span>, <span style={{ color: r.prob }}>scaled</span>)</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full max-w-[460px]" role="img" aria-label="Entropy versus head dimension">
        <line x1={L} x2={W} y1={y(0)} y2={y(0)} stroke={c.line} /><line x1={L} x2={L} y1={4} y2={y(0)} stroke={c.line} />
        <text x={L - 4} y={y(Math.log(8)) + 4} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">2.08</text>
        <text x={L - 4} y={y(0) + 3} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">0</text>
        <path d={line((s) => s.entropyRaw)} fill="none" stroke={r.score} strokeWidth={2} strokeDasharray="5 3" />
        <path d={line((s) => s.entropyScaled)} fill="none" stroke={r.prob} strokeWidth={2} />
        {sweep.map((s, k) => <text key={k} x={x(k)} y={H - 4} textAnchor="middle" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">{s.dk}</text>)}
      </svg>
    </div>
  )
}
type ScalingRes = ReturnType<typeof scalingExperiment>

/* ------------------------------------------------------------------ 6 · masking */
type MaskPreset = 'bidirectional' | 'causal' | 'padding' | 'combined'
export function MaskView({ trace }: { trace: AttnTrace }) {
  const r = useRoles()
  const tokens = useTokens(), sel = useAttn((s) => s.sel), select = useAttn((s) => s.select), causal = useAttn((s) => s.causal), pad = useAttn((s) => s.pad), blocked = useAttn((s) => s.blocked)
  const st = attnStore.getState()
  const hd = trace.heads[sel.h], i = sel.i, n = trace.Tq
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const anyPad = pad.some(Boolean)
  const preset: MaskPreset = causal && anyPad ? 'combined' : causal ? 'causal' : anyPad ? 'padding' : 'bidirectional'
  const choose = (p: MaskPreset) => {
    st.setCausal(p === 'causal' || p === 'combined')
    const auto = tokens.map((t) => t === '<pad>')
    const wantPad = p === 'padding' || p === 'combined'
    st.setPad(wantPad ? (auto.some(Boolean) ? auto : tokens.map((_, k) => k === n - 1)) : tokens.map(() => false))
  }
  const masked = trace.M.map((row) => row.map((v) => v === -Infinity))
  const allowedFor = (q: number) => trace.M[q].flatMap((v, k) => (v === 0 ? [k + 1] : []))
  const reasonText = (q: number, k: number) => trace.reasons[q][k].map((x) => (x === 'causal' ? 'future token (causal)' : x === 'pad' ? 'padding key' : 'knocked out')).join(' + ')
  return (
    <div className="grid gap-3">
      <Card title="choose a mask" right={<HeadPicker />}>
        <div className="flex flex-wrap items-center gap-2">
          <Seg label="Mask preset" value={preset} onChange={choose} options={[
            { v: 'bidirectional', label: 'bidirectional (encoder)' }, { v: 'causal', label: 'causal (decoder)' }, { v: 'padding', label: 'padding only' }, { v: 'combined', label: 'causal + padding' },
          ]} />
          <span className="text-[12px] text-muted">padding keys:</span>
          {tokens.map((t, k) => <button key={k} type="button" className="btn !h-7 !px-2 text-[11.5px]" aria-pressed={pad[k]} onClick={() => st.togglePad(k)} data-testid={`pad-${k}`}>{k + 1}·{t}{pad[k] ? ' ⊘' : ''}</button>)}
          {blocked && <button type="button" className="btn !h-7 text-[11.5px]" onClick={st.clearBlocked}>clear knock-outs</button>}
        </div>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <Card title="which keys may each query use?">
          <div className="grid gap-1">
            {tokens.map((t, q) => (
              <button key={q} type="button" onClick={() => select({ i: q, cell: null })} className={`grid grid-cols-[86px_1fr] items-center gap-2 rounded-md border px-2 py-1 text-left ${q === i ? 'border-cyan bg-cyan/5' : 'border-transparent hover:bg-panel-2'}`} aria-pressed={q === i} data-testid={`maskrow-${q}`}>
                <span className="font-mono text-[12px]" style={{ color: r.q }}>{GLYPH.q} {q + 1}·{t.length > 6 ? t.slice(0, 5) + '…' : t}</span>
                <span className="flex flex-wrap gap-1">
                  {tokens.map((_, k) => {
                    const ok = trace.M[q][k] === 0
                    return <span key={k} title={ok ? 'allowed' : reasonText(q, k)} className="rounded px-1.5 font-mono text-[11px]" style={{ background: ok ? `color-mix(in oklab, ${r.prob} 22%, transparent)` : 'transparent', color: ok ? 'var(--fg)' : r.mask, border: `1px ${ok ? 'solid' : 'dashed'} ${ok ? r.prob : r.mask}` }}>{ok ? '✓' : '⊘'}{k + 1}</span>
                  })}
                </span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-[12px] text-muted" data-testid="mask-sentence">Token {i + 1} can attend to {allowedFor(i).length ? `token${allowedFor(i).length > 1 ? 's' : ''} ${allowedFor(i).join(', ')}` : 'no token at all'}.</p>
          {trace.fullyMaskedRows.length > 0 && (
            <div className="mt-2 rounded-md border border-rose/50 bg-rose/5 p-2 text-[12px]" role="status" data-testid="fully-masked">
              <b className="text-rose">Fully masked row{trace.fullyMaskedRows.length > 1 ? 's' : ''}: query {trace.fullyMaskedRows.map((q) => q + 1).join(', ')}.</b> Every key is masked, so softmax would compute 0/0. A naive implementation produces NaN; recent PyTorch (scaled_dot_product_attention) returns zeros. This lab also returns zero weights and a zero output, and flags the row.
            </div>
          )}
        </Card>
        <Card title={<span>the additive mask <RoleTag role="mask">M</RoleTag> · click a cell to knock out one pair (custom mask)</span>}>
          <Grid m={trace.M.map((row) => row.map((v) => (v === 0 ? 0 : -Infinity)))} role="mask" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} masked={masked} hlRows={[i]} digits={0}
            onCell={(a, b) => { st.toggleBlocked(a, b); select({ i: a, j: b }) }} ariaLabel="Additive mask M" testid="g-M" />
          <p className="text-[11.5px] text-muted">0 = allowed, −∞ = masked (hatched ⊘). Masks combine: a cell is −∞ if any rule masks it.</p>
        </Card>
      </div>
      <Card title={`Z = S̃ + M, then softmax · head ${sel.h + 1}`}>
        <div className="flex flex-wrap items-start gap-3">
          <Grid m={hd.scaled} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="S̃ (scaled scores)" hlRows={[i]} cell={44} ariaLabel="Scaled scores" />
          <span className="self-center font-mono text-faint">+ M =</span>
          <Grid m={hd.logits} role="score" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="Z (softmax input)" masked={masked} hlRows={[i]} cell={44} ariaLabel="Masked logits" testid="g-Z" />
          <span className="self-center font-mono text-faint">→ softmax →</span>
          <Grid m={hd.A} role="prob" mode="prob" rowLabels={rl} colLabels={tokens.map((_, k) => `k${k + 1}`)} title="A" masked={masked} hlRows={[i]} cell={44} ariaLabel="Attention weights" testid="g-A-mask" />
        </div>
        <p className="mt-1 text-[12px] text-muted">Masking is a separate step from scaling (which divides every score) and from softmax (which normalizes). Because it acts before softmax, the remaining weights in each row still sum to 1. Changing the mask changes which positions contribute to the output.</p>
      </Card>
    </div>
  )
}

