/**
 * Phase 4 · vanishing and exploding gradients. A deep network of adjustable depth, width, activation and
 * initialization; per-layer gradient norms, error RMS and activation statistics measured by the engine.
 */
import { Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { ACT_LABEL, defaultOpt, initParams, layerStats, makeDataset, targetsFor, trainEpochs, type Act, type LayerStat, type InitScheme, type NetConfig } from '../../engine/nn'
import { Card } from '../../components/viz'
import { LineChart } from './charts'

export interface DeepCfg { depth: number; width: number; act: Act; scheme: InitScheme; scale: number; seed: number }
export const deepNet = (d: DeepCfg): NetConfig => ({ sizes: [8, ...new Array(d.depth).fill(d.width), 1], acts: [...new Array(d.depth).fill(d.act), 'sigmoid'], loss: 'bce' })
export const deepData = () => targetsFor(makeDataset('random', 3, 8, 32), 1)
export const DEEP_PRESETS: { label: string; cfg: DeepCfg; say: string }[] = [
  { label: 'Deep sigmoid → vanishing', cfg: { depth: 12, width: 16, act: 'sigmoid', scheme: 'xavier', scale: 1, seed: 1 }, say: 'Each layer multiplies the error by σ′ ≤ 0.25 and by the weights: the first layers receive almost nothing.' },
  { label: 'Deep tanh, Xavier', cfg: { depth: 12, width: 16, act: 'tanh', scheme: 'xavier', scale: 1, seed: 1 }, say: 'tanh′ is up to 1 and Xavier keeps the signal scale roughly constant: much healthier than sigmoid.' },
  { label: 'Deep ReLU, He → healthy', cfg: { depth: 12, width: 16, act: 'relu', scheme: 'he', scale: 1, seed: 1 }, say: 'He initialization (variance 2/fan-in) compensates for ReLU zeroing half the units; norms stay within a narrow band.' },
  { label: 'Large weights → exploding', cfg: { depth: 12, width: 16, act: 'leaky', scheme: 'normal', scale: 1.5, seed: 1 }, say: 'Weights with std 1.5 and width 16 multiply the signal by about 6 per layer: errors grow layer by layer towards the input.' },
  { label: 'Small weights → vanishing', cfg: { depth: 12, width: 16, act: 'tanh', scheme: 'normal', scale: 0.1, seed: 1 }, say: 'Tiny weights shrink activations and errors at every layer, even with a non-saturating slope.' },
]

/** compares the backpropagated error at the first and last hidden layers */
export function deepVerdict(st: LayerStat[]) {
  const ratio = st[0].deltaRms / st[st.length - 2].deltaRms
  return { ratio, verdict: !Number.isFinite(ratio) ? 'non-finite' : ratio < 1e-2 ? 'vanishing' : ratio > 1e2 ? 'exploding' : 'healthy' }
}

function LogBars({ values, color, label, testid }: { values: number[]; color: string; label: string; testid?: string }) {
  const c = useColors()
  const W = 460, H = 180, ml = 44, mb = 22, mt = 8
  const lv = values.map((v) => Math.log10(Math.max(1e-30, v)))
  const lo = Math.floor(Math.min(...lv)), hi = Math.ceil(Math.max(...lv)) + (Math.ceil(Math.max(...lv)) === Math.floor(Math.min(...lv)) ? 1 : 0)
  const bw = (W - ml - 8) / values.length
  const y = (v: number) => mt + (1 - (v - lo) / (hi - lo || 1)) * (H - mt - mb)
  const step = Math.max(1, Math.ceil((hi - lo) / 5))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={label} data-testid={testid}>
      {Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, k) => lo + k * step).map((e) => <g key={e}><line x1={ml} x2={W - 8} y1={y(e)} y2={y(e)} stroke={c.line} strokeDasharray="2 3" /><text x={ml - 4} y={y(e) + 3} fontSize={10} textAnchor="end" fill={c.faint} fontFamily="var(--font-mono)">1e{e}</text></g>)}
      {lv.map((v, k) => (
        <g key={k}>
          <rect x={ml + k * bw + 1} y={y(v)} width={Math.max(1, bw - 2)} height={Math.max(0, H - mb - y(v))} fill={color} rx={2}><title>{`layer ${k + 1}: ${values[k].toExponential(2)}`}</title></rect>
          {(values.length <= 16 || k % 2 === 0) && <text x={ml + k * bw + bw / 2} y={H - 8} fontSize={9.5} textAnchor="middle" fill={c.faint} fontFamily="var(--font-mono)">{k + 1}</text>}
        </g>
      ))}
    </svg>
  )
}

export function VanishLab() {
  const c = useColors()
  const [d, setD] = useState<DeepCfg>(DEEP_PRESETS[0].cfg)
  const [say, setSay] = useState(DEEP_PRESETS[0].say)
  const set = (p: Partial<DeepCfg>) => { setD((x) => ({ ...x, ...p })); setSay('') }
  const data = useMemo(deepData, [])
  const cfg = useMemo(() => deepNet(d), [d])
  const params = useMemo(() => initParams(cfg.sizes, d.seed, d.scheme, d.scale), [cfg, d.seed, d.scheme, d.scale])
  const st = useMemo(() => layerStats(cfg, params, data), [cfg, params, data])
  const g = st.map((s) => s.gradNorm)
  const { ratio, verdict } = deepVerdict(st)

  const [runs, setRuns] = useState<{ label: string; losses: number[] }[] | null>(null)
  useEffect(() => setRuns(null), [d])
  const train = () => {
    const ref: DeepCfg = { ...d, act: 'relu', scheme: 'he', scale: 1 }
    const mk = (x: DeepCfg) => { const k = deepNet(x); return trainEpochs(k, initParams(k.sizes, x.seed, x.scheme, x.scale), data, { ...defaultOpt, kind: 'sgd', lr: 0.02 }, 32, 1, 200).losses }
    setRuns([{ label: `this network (${ACT_LABEL[d.act]}, ${d.scheme}${d.scheme === 'normal' || d.scheme === 'uniform' ? ` ${d.scale}` : ''})`, losses: mk(d) }, { label: 'same depth, ReLU + He', losses: mk(ref) }])
  }

  return (
    <div className="grid gap-3">
      <Card title="a deep network · 8 inputs → depth × width hidden → 1 sigmoid output · 32 random samples">
        <div className="mb-2 flex flex-wrap gap-1">
          {DEEP_PRESETS.map((p) => <button key={p.label} type="button" className="btn" onClick={() => { setD(p.cfg); setSay(p.say) }}>{p.label}</button>)}
        </div>
        <div className="grid gap-2 text-[12.5px] text-muted @3xl:grid-cols-3">
          <label className="flex items-center gap-2">depth <span className="w-6 font-mono text-fg">{d.depth}</span><input type="range" min={2} max={24} value={d.depth} onChange={(e) => set({ depth: +e.target.value })} className="flex-1" aria-label="Depth" /></label>
          <label className="flex items-center gap-2">width <span className="w-6 font-mono text-fg">{d.width}</span><input type="range" min={4} max={32} value={d.width} onChange={(e) => set({ width: +e.target.value })} className="flex-1" aria-label="Width" /></label>
          <label className="flex items-center gap-2">activation
            <select className="flex-1 rounded border border-line bg-panel-2 px-1 py-0.5" value={d.act} onChange={(e) => set({ act: e.target.value as Act })} aria-label="Hidden activation">{(['sigmoid', 'tanh', 'relu', 'leaky', 'linear'] as Act[]).map((a) => <option key={a} value={a}>{ACT_LABEL[a]}</option>)}</select>
          </label>
          <label className="flex items-center gap-2">init
            <select className="flex-1 rounded border border-line bg-panel-2 px-1 py-0.5" value={d.scheme} onChange={(e) => set({ scheme: e.target.value as InitScheme })} aria-label="Deep initialization"><option value="xavier">Xavier</option><option value="he">He</option><option value="normal">Normal(0, scale)</option><option value="uniform">Uniform ±scale</option></select>
          </label>
          <label className="flex items-center gap-2">scale <span className="w-10 font-mono text-fg">{d.scale.toFixed(2)}</span><input type="range" min={0.05} max={3} step={0.05} value={d.scale} onChange={(e) => set({ scale: +e.target.value })} className="flex-1" aria-label="Deep init scale" /></label>
          <label className="flex items-center gap-2">seed <input type="number" min={0} value={d.seed} onChange={(e) => set({ seed: Math.max(0, Math.round(+e.target.value || 0)) })} className="w-16 rounded border border-line bg-panel-2 px-1 font-mono" aria-label="Deep seed" /></label>
        </div>
        <div className={`mt-2 rounded-lg border p-2 text-[13px] ${verdict === 'healthy' ? 'border-teal/50 bg-teal/5' : 'border-rose/50 bg-rose/5'}`} data-testid="vanish-verdict">
          <b className={verdict === 'healthy' ? 'text-teal' : 'text-rose'}>{verdict}</b>: error at the first hidden layer ÷ error at the last, RMS(δ⁽¹⁾) / RMS(δ⁽{d.depth}⁾) = <span className="font-mono">{Number.isFinite(ratio) ? ratio.toExponential(2) : '—'}</span>
          {say && <span className="text-muted"> · {say}</span>}
        </div>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-2">
        <Card title="gradient norm per layer ‖∂L/∂W⁽ˡ⁾‖ (log scale; input side left)"><LogBars values={g} color={c.rose} label="Gradient norm per layer" testid="grad-bars" /></Card>
        <Card title="error size per layer: RMS of δ⁽ˡ⁾ (log scale)"><LogBars values={st.map((s) => s.deltaRms)} color={c.violet} label="Delta RMS per layer" /></Card>
        <Card title="activations per layer">
          <LineChart ariaLabel="Activation statistics per layer" xLabel="layer" height={190} series={[{ points: st.map((s) => [s.layer, s.actStd]), color: c.cyan, label: 'std of a' }, { points: st.map((s) => [s.layer, s.actMean]), color: c.amber, label: 'mean of a', dashed: true }]} />
        </Card>
        <Card title="units that block gradient">
          <LineChart ariaLabel="Saturated and dead fractions per layer" xLabel="layer" height={190} yDomain={[0, 1]} series={[{ points: st.slice(0, -1).map((s) => [s.layer, s.saturated]), color: c.rose, label: '|f′(z)| < 0.05 (saturated)' }, ...(d.act === 'relu' ? [{ points: st.slice(0, -1).map((s) => [s.layer, s.dead]), color: c.violet, label: 'ReLU with z ≤ 0' }] : [])]} />
        </Card>
      </div>
      <Card title="does it train? 200 steps of full-batch SGD (η = 0.02)" right={<button type="button" className="btn" onClick={train}><Play size={13} />Train both</button>}>
        {runs ? <LineChart ariaLabel="Deep network training curves" xLabel="step" yLabel="loss" height={200} series={runs.map((r, k) => ({ points: r.losses.map((v, i) => [i, v]), color: k ? c.teal : c.rose, label: `${r.label} → ${r.losses.at(-1)!.toFixed(3)}` }))} />
          : <p className="text-[12.5px] text-muted">Trains this network and the same depth with ReLU + He initialization from the same seed on the same 32 samples, and overlays the loss curves. A network whose early layers get no gradient learns only in its last layers.</p>}
      </Card>
    </div>
  )
}
