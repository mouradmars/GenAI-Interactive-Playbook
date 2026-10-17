/** Advanced research mode: entropy, head ablation, head similarity, sequence-length cost, cross-attention and a real model. */
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { useColors } from '../../sim/appState'
import { PRESET_X_B, attentionCost, compareTraces, headSimilarity, presetInput, runAttention, type AttnTrace } from '../../engine/attention'
import { Grid, HeadPicker, Seg, fx, headColors, tokLabel, useRoles, useTokens } from './common'
import { realAttention } from './realModel'
import { useAttn } from './store'

type Section = 'entropy' | 'ablation' | 'similarity' | 'length' | 'cross' | 'real'
export function ResearchTab({ trace }: { trace: AttnTrace }) {
  const [sec, setSec] = useState<Section>('entropy')
  return (
    <div className="grid gap-3">
      <Seg label="Research section" value={sec} onChange={setSec} options={[
        { v: 'entropy', label: 'A · entropy' }, { v: 'ablation', label: 'B · head ablation' }, { v: 'similarity', label: 'C · head similarity' },
        { v: 'length', label: 'D · sequence length' }, { v: 'cross', label: 'E · cross-attention' }, { v: 'real', label: 'F · real model' },
      ]} />
      {sec === 'entropy' && <EntropySection trace={trace} />}
      {sec === 'ablation' && <AblationSection trace={trace} />}
      {sec === 'similarity' && <SimilaritySection trace={trace} />}
      {sec === 'length' && <LengthSection />}
      {sec === 'cross' && <CrossSection />}
      {sec === 'real' && <RealSection />}
    </div>
  )
}

function EntropySection({ trace }: { trace: AttnTrace }) {
  const r = useRoles(), c = useColors()
  const tokens = useTokens()
  const hc = headColors(r)
  const n = trace.Tq, H = trace.cfg.H, gw = 40 + H * 22, W = 50 + n * gw, Ht = 180, B = 30
  const maxE = Math.log(trace.Tk)
  const y = (e: number) => 10 + (1 - e / maxE) * (Ht - B - 10)
  return (
    <Card title="A · attention entropy per query: H(a_i) = −Σ_j a_ij ln a_ij (0 = all weight on one key)">
      <svg viewBox={`0 0 ${W} ${Ht}`} className="block w-full" style={{ maxWidth: W * 1.4 }} role="img" aria-label="Attention entropy per query and head" data-testid="entropy-chart">
        <line x1={40} x2={W} y1={y(0)} y2={y(0)} stroke={c.line} />
        <text x={36} y={y(maxE) + 4} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">ln {trace.Tk}</text>
        <text x={36} y={y(0) + 3} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">0</text>
        {Array.from({ length: n }, (_, i) => {
          const allowed = trace.M[i].filter((v) => v === 0).length
          return (
            <g key={i} transform={`translate(${50 + i * gw}, 0)`}>
              {allowed > 0 && <line x1={0} x2={H * 22} y1={y(Math.log(allowed))} y2={y(Math.log(allowed))} stroke={c.muted} strokeDasharray="3 2"><title>maximum for {allowed} permitted keys: ln {allowed}</title></line>}
              {trace.heads.map((hd) => <rect key={hd.h} x={hd.h * 22} y={y(hd.entropy[i])} width={18} height={y(0) - y(hd.entropy[i])} fill={hc[hd.h % 4]} opacity={hd.enabled ? 0.9 : 0.3}><title>{`head ${hd.h + 1}, query ${i + 1}: ${fx(hd.entropy[i], 3)} nats`}</title></rect>)}
              <text x={(H * 22) / 2} y={Ht - 12} textAnchor="middle" fontSize={10.5} fill={c.fg} fontFamily="var(--font-mono)">{tokLabel(tokens, i)}</text>
            </g>
          )
        })}
      </svg>
      <div className="flex flex-wrap gap-3 font-mono text-[11.5px]">{trace.heads.map((hd) => <span key={hd.h} style={{ color: hc[hd.h % 4] }}>■ head {hd.h + 1}: mean {fx(hd.entropy.reduce((a, b) => a + b, 0) / n, 3)} nats</span>)}<span className="text-faint">dashed: maximum ln(#permitted keys)</span></div>
      <p className="mt-1 text-[12px] text-muted">Under a causal mask the first query has one permitted key, so its entropy is always 0. Low entropy means focused attention; it says nothing by itself about what is useful.</p>
    </Card>
  )
}

function AblationSection({ trace }: { trace: AttnTrace }) {
  const tokens = useTokens()
  const X = useAttn((s) => s.X), params = useAttn((s) => s.params)
  const cfgS = trace.cfg
  const rows = useMemo(() => trace.heads.map((hd) => {
    const off = runAttention(X, params, { ...cfgS, headsOn: cfgS.headsOn.map((o, k) => (k === hd.h ? false : o)) })
    const only = runAttention(X, params, { ...cfgS, headsOn: cfgS.headsOn.map((_, k) => k === hd.h) })
    const d = trace.out.map((r, i) => Math.hypot(...r.map((v, k) => v - off.out[i][k])))
    const keep = trace.out.map((r, i) => Math.hypot(...r.map((v, k) => v - only.out[i][k])))
    return { h: hd.h, d, keep, others: compareTraces(trace, off).heads.filter((_, k) => k !== hd.h).every((x) => !Object.values(x.changed).some(Boolean)) }
  }), [trace, X, params, cfgS])
  const norms = trace.out.map((r) => Math.hypot(...r))
  return (
    <Card title="B · head ablation: disable one head at a time and measure ‖Δ output‖ per token">
      <table className="w-full font-mono text-[12px]" data-testid="ablation-table">
        <thead><tr className="text-faint"><th className="text-left font-normal">disabled head</th>{tokens.map((_, i) => <th key={i} className="font-normal">{tokLabel(tokens, i)}</th>)}<th className="font-normal">mean</th><th className="font-normal">other heads unchanged?</th></tr></thead>
        <tbody>
          {rows.map((x) => <tr key={x.h}><td>head {x.h + 1}</td>{x.d.map((v, i) => <td key={i} className="text-center">{fx(v, 3)}</td>)}<td className="text-center">{fx(x.d.reduce((a, b) => a + b, 0) / x.d.length, 3)}</td><td className="text-center">{x.others ? 'yes' : 'no'}</td></tr>)}
          <tr className="text-faint"><td>‖output‖</td>{norms.map((v, i) => <td key={i} className="text-center">{fx(v, 3)}</td>)}<td /><td /></tr>
        </tbody>
      </table>
      {trace.cfg.H > 1 && <p className="mt-1 font-mono text-[11.5px] text-muted">keeping only one head: ‖Δ‖ = {rows.map((x) => `head ${x.h + 1}: ${fx(x.keep.reduce((a, b) => a + b, 0) / x.keep.length, 3)}`).join(' · ')} (mean over tokens)</p>}
      <p className="mt-1 text-[12px] text-muted">Ablation measures how much the output changes, not what a head “means”. In trained models, effects of ablation can also be compensated by other components downstream.</p>
    </Card>
  )
}

function SimilaritySection({ trace }: { trace: AttnTrace }) {
  const H = trace.cfg.H
  if (H === 1) return <Card title="C · head similarity"><p className="text-[12.5px] text-muted">Choose h = 2 or h = 4 to compare heads.</p></Card>
  const pairs = trace.heads.flatMap((a) => trace.heads.filter((b) => b.h > a.h).map((b) => ({ a: a.h, b: b.h, ...headSimilarity(a.A, b.A) })))
  const js = trace.heads.map((a) => trace.heads.map((b) => (a.h === b.h ? 0 : headSimilarity(a.A, b.A).js)))
  return (
    <Card title="C · similarity between heads' attention matrices">
      <div className="flex flex-wrap items-start gap-4">
        <Grid m={js} role="score" rowLabels={trace.heads.map((h) => `head ${h.h + 1}`)} colLabels={trace.heads.map((h) => `h${h.h + 1}`)} title="mean Jensen–Shannon divergence (nats, 0…ln 2)" cell={54} digits={3} ariaLabel="Head similarity matrix" maxAbs={Math.log(2)} />
        <table className="font-mono text-[12px]" data-testid="similarity-table">
          <thead><tr className="text-faint"><th className="text-left font-normal">pair</th><th className="font-normal">JS</th><th className="font-normal">total variation</th><th className="font-normal">cosine</th></tr></thead>
          <tbody>{pairs.map((p) => <tr key={`${p.a}${p.b}`}><td className="pr-3">head {p.a + 1} · head {p.b + 1}</td><td className="px-2 text-center">{fx(p.js, 4)}</td><td className="px-2 text-center">{fx(p.tv, 4)}</td><td className="px-2 text-center">{fx(p.cosine, 4)}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="mt-1 text-[12px] text-muted">Rows are compared as probability distributions (JS divergence and total variation, averaged over queries that have at least one permitted key); the cosine compares the flattened matrices. Different patterns do not imply different “functions”.</p>
    </Card>
  )
}

function LengthSection() {
  const r = useRoles(), c = useColors()
  const T = useAttn((s) => s.T), H = useAttn((s) => s.H)
  const [preset, setPreset] = useState<'toy' | 'gpt2'>('gpt2')
  const d = preset === 'toy' ? 4 : 768, h = preset === 'toy' ? H : 12
  const Ts = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
  const data = Ts.map((t) => ({ t, ...attentionCost(t, d, h) }))
  const W = 520, Hh = 200, L = 52, B = 26
  const lx = (k: number) => L + (k / (Ts.length - 1)) * (W - L - 10)
  const all = data.flatMap((x) => [x.proj + x.out, x.scores + x.mix])
  const lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all))
  const ly = (v: number) => 8 + (1 - (Math.log10(v) - lo) / (hi - lo)) * (Hh - B - 8)
  const path = (f: (x: (typeof data)[number]) => number) => data.map((x, k) => `${k ? 'L' : 'M'}${lx(k)},${ly(f(x))}`).join(' ')
  const live = attentionCost(T, 4, H)
  return (
    <Card title="D · how attention cost grows with sequence length n (batch 1, multiply–accumulates)" right={<Seg label="Model size" value={preset} onChange={setPreset} options={[{ v: 'toy', label: 'toy · d = 4' }, { v: 'gpt2', label: 'GPT-2-small-like · d = 768, h = 12' }]} />}>
      <svg viewBox={`0 0 ${W} ${Hh}`} className="block w-full" style={{ maxWidth: W * 1.4 }} role="img" aria-label="Cost versus sequence length" data-testid="cost-chart">
        <line x1={L} x2={W} y1={Hh - B} y2={Hh - B} stroke={c.line} /><line x1={L} x2={L} y1={4} y2={Hh - B} stroke={c.line} />
        <path d={path((x) => x.proj + x.out)} fill="none" stroke={r.w} strokeWidth={2} />
        <path d={path((x) => x.scores + x.mix)} fill="none" stroke={r.score} strokeWidth={2} strokeDasharray="6 3" />
        {Ts.map((t, k) => (k % 2 === 0 ? <text key={t} x={lx(k)} y={Hh - 8} textAnchor="middle" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">{t}</text> : null))}
        <text x={L - 4} y={ly(10 ** hi) + 4} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">1e{Math.round(hi)}</text>
        <text x={L - 4} y={ly(10 ** lo) + 3} textAnchor="end" fontSize={9.5} fill={c.faint} fontFamily="var(--font-mono)">1e{Math.round(lo)}</text>
      </svg>
      <div className="flex flex-wrap gap-3 font-mono text-[11.5px]"><span style={{ color: r.w }}>— projections Q, K, V, O: 4·n·d² (linear in n)</span><span style={{ color: r.score }}>- - QKᵀ and A·V: 2·n²·d (quadratic in n)</span><span className="text-faint">log–log axes</span></div>
      <p className="mt-1 text-[12px] text-muted">The quadratic terms overtake the projections when 2n²d &gt; 4nd², i.e. for n &gt; 2d ({2 * d} tokens here). The attention matrices alone hold h·n² numbers: {h}·{8192}² ≈ {(h * 8192 ** 2 / 1e6).toFixed(0)} million at n = 8192. In the lab right now (n = {T}, d = 4, h = {H}): {live.total} MACs, {live.attnEntries} attention weights.</p>
    </Card>
  )
}

function CrossSection() {
  const tokens = useTokens()
  const X = useAttn((s) => s.X), params = useAttn((s) => s.params), scale = useAttn((s) => s.scale), H = useAttn((s) => s.H), sel = useAttn((s) => s.sel)
  const [m, setM] = useState(3)
  const src = useMemo(() => (m <= 4 ? PRESET_X_B.slice(0, m) : presetInput(m, 4, 7)), [m])
  const t = useMemo(() => runAttention(X, params, { H, causal: false, scale, pad: new Array(m).fill(false), blocked: null, headsOn: new Array(H).fill(true) }, src), [X, params, H, scale, m, src])
  const h = Math.min(sel.h, H - 1)
  const sl = Array.from({ length: m }, (_, k) => `src${k + 1}`)
  return (
    <Card title="E · cross-attention: queries from one sequence, keys and values from another" right={<div className="flex items-center gap-2"><HeadPicker /><Seg label="Source length" value={m} onChange={setM} options={[2, 3, 4, 5, 6].map((v) => ({ v, label: `m = ${v}` }))} /></div>}>
      <div className="flex flex-wrap items-start gap-3">
        <Grid m={src} role="k" rowLabels={sl} colLabels={['f1', 'f2', 'f3', 'f4']} title="source sequence (e.g. encoder output)" cell={42} ariaLabel="Source sequence" digits={1} />
        <Grid m={t.heads[h].A} role="prob" mode="prob" rowLabels={tokens.map((_, k) => tokLabel(tokens, k))} colLabels={sl} title={`A · ${t.Tq}×${t.Tk} (queries × source keys)`} cell={52} ariaLabel="Cross-attention weights" testid="cross-A" />
        <Grid m={t.out} role="out" colLabels={['f1', 'f2', 'f3', 'f4']} title={`output · ${t.Tq}×4`} cell={46} ariaLabel="Cross-attention output" />
      </div>
      <p className="mt-2 text-[12px] text-muted">Self-attention: Q, K and V all come from X (an n×n matrix). Cross-attention (e.g. a translation decoder reading the encoder): Q = X W_Q from the target sequence, K = Y W_K and V = Y W_V from the source sequence Y, so A is n×m and no causal mask is needed over the source. The output still has one row per query position. Same toy weights as the lab; the source rows are hand-chosen.</p>
    </Card>
  )
}

function RealSection() {
  const [text, setText] = useState('the cat sat in the')
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const res = useMemo(() => realAttention(text, layer, head), [text, layer, head])
  const m = res.meta
  return (
    <Card title="F · real-model attention inspection · Observatory-Nano (a real trained model)">
      <div className="mb-2 rounded-md border border-teal/50 bg-teal/5 px-2 py-1.5 text-[12px]"><b className="text-teal">Real model, not the toy.</b> {m.name}: a decoder-only GPT with {m.nLayers} layers × {m.nHeads} heads, d = {m.dModel}, trained in this repository (module 01) on a small corpus. These weights are read from its forward pass; nothing is invented. Attention weights show where each head reads from; they do not explain the model's reasoning on their own.</div>
      <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
        <input value={text} onChange={(e) => setText(e.target.value)} className="w-64 rounded border border-line px-2 py-1 font-mono" aria-label="Prompt for the real model" />
        {['the cat sat in the', 'paris is the capital of', 'one plus two is'].map((s) => <button key={s} type="button" className="btn !h-7 text-[11.5px]" onClick={() => setText(s)}>{s}</button>)}
        <Seg label="Layer" value={layer} onChange={setLayer} options={Array.from({ length: m.nLayers }, (_, k) => ({ v: k, label: `layer ${k + 1}` }))} />
        <Seg label="Real-model head" value={head} onChange={setHead} options={Array.from({ length: m.nHeads }, (_, k) => ({ v: k, label: `head ${k + 1}` }))} />
      </div>
      <div className="mt-2 flex flex-wrap items-start gap-3">
        <Grid m={res.A} role="prob" mode="prob" rowLabels={res.tokens.map((t, k) => `${k + 1}·${t.slice(0, 7)}`)} colLabels={res.tokens.map((_, k) => `k${k + 1}`)} masked={res.A.map((r, i) => r.map((_, j) => j > i))} cell={46} title={`layer ${res.layer + 1}, head ${res.head + 1}: attention weights (d_k = ${res.dk})`} ariaLabel="Real model attention weights" testid="real-A" />
        <div className="grid gap-1 font-mono text-[12px]">
          <div>entropy per query: {res.entropy.map((e) => fx(e, 2)).join(' · ')}</div>
          <div data-testid="real-check">cross-check: recomputing softmax(qkᵀ/√{res.dk} + causal mask) from the model's real q and k with this module's engine reproduces the weights to max |Δ| = {res.maxDiff.toExponential(1)} (the model itself runs in float32)</div>
          {res.truncated && <div className="text-amber">The prompt was truncated to the model's context length.</div>}
        </div>
      </div>
    </Card>
  )
}
