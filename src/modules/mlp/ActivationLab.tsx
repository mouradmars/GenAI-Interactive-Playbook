/** Phase 4 · activation explorer: f(z) and f′(z) side by side, a probe z, measured properties, and softmax. */
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { ACT_LABEL, actDeriv, actFn, softmax, type Act } from '../../engine/nn'
import { Card, Tex } from '../../components/viz'
import { LineChart } from './charts'
import { actTex, derivTex, n4 } from './narrate'

export const ELEMENTWISE: Act[] = ['sigmoid', 'tanh', 'relu', 'leaky', 'linear']
const ZS = Array.from({ length: 241 }, (_, k) => -6 + k * 0.05)

/** measured on a grid z ∈ [−6, 6] (and checked in the tests) */
export function actProps(f: Act) {
  const vals = ZS.map((z) => actFn(f, z)), ders = ZS.map((z) => actDeriv(f, z))
  return {
    min: Math.min(...vals), max: Math.max(...vals), maxDeriv: Math.max(...ders),
    zeroCentred: Math.abs(actFn(f, 0)) < 1e-12,
    saturating: Math.abs(actDeriv(f, 6)) < 0.01 && Math.abs(actDeriv(f, -6)) < 0.01,
    flatShare: ders.filter((d) => Math.abs(d) < 0.01).length / ders.length,
  }
}
const NOTES: Record<Act, string> = {
  sigmoid: 'Outputs in (0, 1): natural for probabilities. Slope at most 0.25, and near zero for |z| > 5, so deep sigmoid stacks lose gradient.',
  tanh: 'A rescaled sigmoid, zero-centred, range (−1, 1), slope up to 1. Still saturates for large |z|.',
  relu: 'Cheap and does not saturate for z > 0 (slope exactly 1). For z ≤ 0 the slope is 0: a unit stuck there receives no gradient (“dead ReLU”).',
  leaky: 'Like ReLU, but a small slope 0.01 for z < 0 keeps some gradient flowing.',
  linear: 'No non-linearity. Stacking linear layers is still a linear map, so hidden layers need a non-linear activation.',
  softmax: 'A vector function used on the output layer: turns logits into a probability distribution.',
}

export function ActivationLab() {
  const c = useColors()
  const colors: Record<Act, string> = { sigmoid: c.cyan, tanh: c.violet, relu: c.amber, leaky: c.rose, linear: c.teal, softmax: c.fg }
  const [shown, setShown] = useState<Act[]>(['sigmoid', 'tanh', 'relu'])
  const [z, setZ] = useState(1.5)
  const toggle = (f: Act) => setShown((s) => (s.includes(f) ? s.filter((x) => x !== f) : [...s, f]))
  const props = useMemo(() => Object.fromEntries(ELEMENTWISE.map((f) => [f, actProps(f)])) as Record<Act, ReturnType<typeof actProps>>, [])
  const [logits, setLogits] = useState([2, 1, -0.5])
  const sm = softmax(logits)
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        {ELEMENTWISE.map((f) => (
          <button key={f} type="button" className="btn" aria-pressed={shown.includes(f)} onClick={() => toggle(f)}><span className="inline-block h-[3px] w-4 rounded" style={{ background: colors[f] }} />{ACT_LABEL[f]}</button>
        ))}
        <label className="ml-auto flex items-center gap-2 text-[12.5px] text-muted">z = <span className="w-12 font-mono text-fg">{z.toFixed(2)}</span>
          <input type="range" min={-6} max={6} step={0.05} value={z} onChange={(e) => setZ(+e.target.value)} aria-label="Probe z" className="w-40" />
        </label>
      </div>
      <div className="grid gap-3 @4xl:grid-cols-2">
        <Card title="f(z)">
          <LineChart legendLeft ariaLabel="Activation functions" xLabel="z" height={220} xDomain={[-6, 6]} yDomain={[-1.5, 3]} onClickX={(v) => setZ(+Math.max(-6, Math.min(6, v)).toFixed(2))}
            series={shown.map((f) => ({ points: ZS.map((x) => [x, actFn(f, x)] as [number, number]), color: colors[f], label: ACT_LABEL[f] }))}
            markers={shown.map((f) => ({ x: z, y: actFn(f, z), color: colors[f] }))} />
        </Card>
        <Card title="f′(z): how much gradient passes through">
          <LineChart ariaLabel="Activation derivatives" xLabel="z" height={220} xDomain={[-6, 6]} yDomain={[-0.1, 1.1]} onClickX={(v) => setZ(+Math.max(-6, Math.min(6, v)).toFixed(2))}
            series={shown.map((f) => ({ points: ZS.map((x) => [x, actDeriv(f, x)] as [number, number]), color: colors[f], label: ACT_LABEL[f] }))}
            markers={shown.map((f) => ({ x: z, y: actDeriv(f, z), color: colors[f] }))} />
        </Card>
      </div>
      <Card title={`at z = ${z.toFixed(2)} · and measured over z ∈ [−6, 6]`}>
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full text-[12.5px]" data-testid="act-table">
            <thead className="text-left text-faint"><tr><th className="px-2 py-1">activation</th><th className="px-2 py-1 text-right">f(z)</th><th className="px-2 py-1 text-right">f′(z)</th><th className="px-2 py-1 text-right">range</th><th className="px-2 py-1 text-right">max f′</th><th className="px-2 py-1">zero-centred</th><th className="px-2 py-1">saturates (both tails)</th><th className="px-2 py-1 text-right">flat (f′ &lt; 0.01)</th></tr></thead>
            <tbody>
              {ELEMENTWISE.map((f) => { const p = props[f]; return (
                <tr key={f} className="border-t border-line font-mono">
                  <td className="px-2 py-1 font-sans" style={{ color: colors[f] }}>{ACT_LABEL[f]}</td>
                  <td className="px-2 py-1 text-right">{n4(actFn(f, z))}</td>
                  <td className="px-2 py-1 text-right">{n4(actDeriv(f, z))}</td>
                  <td className="px-2 py-1 text-right">{f === 'relu' || f === 'linear' || f === 'leaky' ? (f === 'relu' ? '[0, ∞)' : '(−∞, ∞)') : `(${p.min.toFixed(0)}, ${p.max.toFixed(0)})`}</td>
                  <td className="px-2 py-1 text-right">{p.maxDeriv.toFixed(2)}</td>
                  <td className="px-2 py-1">{p.zeroCentred ? 'yes' : 'no'}</td>
                  <td className="px-2 py-1">{p.saturating ? 'yes' : 'no'}</td>
                  <td className="px-2 py-1 text-right">{(p.flatShare * 100).toFixed(0)} %</td>
                </tr>) })}
            </tbody>
          </table>
        </div>
        <div className="mt-2 grid gap-2 @3xl:grid-cols-2">
          {shown.map((f) => (
            <div key={f} className="rounded-lg border border-line bg-panel-2/50 p-2 text-[12.5px]">
              <div className="flex flex-wrap items-center gap-3 overflow-x-auto scroll-thin"><b style={{ color: colors[f] }}>{ACT_LABEL[f]}</b><Tex tex={actTex[f]} /><Tex tex={derivTex[f]} /></div>
              <p className="mt-1 text-muted">{NOTES[f]}</p>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[12px] text-muted">Why it matters: backpropagation multiplies one f′ per layer. With sigmoid that factor is at most 0.25, so ten layers shrink the signal by at least 0.25¹⁰ ≈ 10⁻⁶. See the Vanishing tab.</p>
      </Card>
      <Card title="softmax: logits → probabilities">
        <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="grid gap-2">
            {logits.map((v, k) => (
              <label key={k} className="flex items-center gap-2 text-[12.5px] text-muted">z{['₁', '₂', '₃'][k]} <span className="w-12 font-mono text-fg">{v.toFixed(2)}</span>
                <input type="range" min={-5} max={5} step={0.05} value={v} onChange={(e) => setLogits((l) => l.map((x, i) => (i === k ? +e.target.value : x)))} className="flex-1" aria-label={`Logit ${k + 1}`} />
              </label>
            ))}
            <button type="button" className="btn justify-self-start" onClick={() => setLogits((l) => l.map((x) => +(x + 1).toFixed(2)))}>add 1 to every logit</button>
          </div>
          <div className="grid gap-1">
            {sm.map((p, k) => (
              <div key={k} className="flex items-center gap-2 text-[12.5px]"><span className="w-6 font-mono text-muted">p{['₁', '₂', '₃'][k]}</span>
                <div className="h-4 flex-1 rounded bg-panel-2"><div className="h-4 rounded" style={{ width: `${p * 100}%`, background: c.cyan }} /></div>
                <span className="w-14 text-right font-mono" data-testid={`softmax-${k}`}>{p.toFixed(4)}</span></div>
            ))}
            <p className="text-[12px] text-muted">Σ = {sm.reduce((a, b) => a + b, 0).toFixed(4)}. Adding the same number to every logit changes nothing: only differences between logits matter.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
