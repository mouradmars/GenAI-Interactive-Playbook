/**
 * Loss lab: (1) the loss as a function of one parameter, with the backprop gradient as the tangent and the
 * gradient-descent step; (2) why sigmoid + MSE learns slowly when badly wrong and sigmoid + BCE does not.
 */
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { LOSS_LABEL, backward, datasetLoss, forward, getParam, gradOf, batchGrad, lossValue, paramName, setParam, type LossKind, type NetConfig } from '../../engine/nn'
import { Card, Tex } from '../../components/viz'
import { LineChart } from './charts'
import { n4 } from './narrate'
import { shownSample, useMlp } from './store'

/** loss and ∂L/∂z for a single sigmoid output with logit z (computed by the engine on a 1 → 1 network) */
export function logitCurve(loss: LossKind, y: number, z: number) {
  const cfg: NetConfig = { sizes: [1, 1], acts: ['sigmoid'], loss }
  const p = { W: [[[1]]], b: [[0]] }, t = forward(cfg, p, [z])
  return { L: lossValue(cfg, t, [y]), dz: backward(cfg, p, t, [y]).delta[1][0], yhat: t.a[1][0] }
}

export function LossLab() {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), data = useMlp((s) => s.data), sampleIdx = useMlp((s) => s.sampleIdx), probe = useMlp((s) => s.probe), selection = useMlp((s) => s.selection), opt = useMlp((s) => s.opt)
  const setParamValue = useMlp((s) => s.setParamValue)
  const [scope, setScope] = useState<'sample' | 'data'>('sample')
  const [range, setRange] = useState(1.5)
  const ref = useMemo(() => (selection && selection.kind !== 'neuron' ? selection : { kind: 'w' as const, layer: 1, j: 0, i: 0 }), [selection])
  const set = useMemo(() => (scope === 'sample' ? [shownSample({ probe, data, sampleIdx })] : data.slice(0, 200)), [scope, data, sampleIdx, probe])
  const th = getParam(params, ref)
  const slice = useMemo(() => Array.from({ length: 121 }, (_, k) => { const v = th - range + (2 * range * k) / 120; return [v, datasetLoss(cfg, setParam(params, ref, v), set)] as [number, number] }), [cfg, params, ref, set, range, th])
  const L0 = datasetLoss(cfg, params, set), g = gradOf(batchGrad(cfg, params, set), ref)
  const nxt = th - opt.lr * g, Lnext = datasetLoss(cfg, setParam(params, ref, nxt), set)
  const dx = range * 0.45

  const [yT, setYT] = useState(1)
  const curves = useMemo(() => {
    const zs = Array.from({ length: 161 }, (_, k) => -8 + k * 0.1)
    const mse = zs.map((z) => logitCurve('mse', yT, z)), bce = zs.map((z) => logitCurve('bce', yT, z))
    return { L: [zs.map((z, k) => [z, mse[k].L] as [number, number]), zs.map((z, k) => [z, bce[k].L] as [number, number])], D: [zs.map((z, k) => [z, Math.abs(mse[k].dz)] as [number, number]), zs.map((z, k) => [z, Math.abs(bce[k].dz)] as [number, number])] }
  }, [yT])
  const zBad = yT === 1 ? -6 : 6
  const bad = { mse: logitCurve('mse', yT, zBad), bce: logitCurve('bce', yT, zBad) }

  return (
    <div className="grid gap-3 @4xl:grid-cols-2">
      <Card title={`loss along one parameter · ${paramName(ref)}`} right={
        <div className="flex rounded-md border border-line p-0.5 text-[11.5px]" role="radiogroup" aria-label="Loss over">
          {(['sample', 'data'] as const).map((s) => <button key={s} type="button" role="radio" aria-checked={scope === s} onClick={() => setScope(s)} className={`rounded px-2 py-0.5 ${scope === s ? 'bg-raise text-cyan' : 'text-muted'}`}>{s === 'sample' ? 'this sample' : data.length > 200 ? 'first 200 samples' : 'whole dataset'}</button>)}
        </div>}>
        <LineChart ariaLabel={`Loss as a function of ${paramName(ref)}`} xLabel={paramName(ref)} yLabel="L" height={230}
          series={[{ points: slice, color: c.cyan }]}
          lines={[{ x1: th - dx, y1: L0 - g * dx, x2: th + dx, y2: L0 + g * dx, color: c.rose, dashed: true }, { x1: th, y1: L0, x2: nxt, y2: Lnext, color: c.teal, arrow: true }]}
          markers={[{ x: th, y: L0, color: c.amber, label: 'now' }, { x: nxt, y: Lnext, color: c.teal, r: 3.5 }]}
          onClickX={(v) => setParamValue(ref, +v.toFixed(3))} />
        <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted">
          <span><span className="text-rose">dashed</span>: tangent with slope = backprop gradient {n4(g)}</span>
          <span><span className="text-teal">arrow</span>: one step θ − η·g (η = {opt.lr}) → L = {n4(Lnext)}</span>
          <label className="ml-auto flex items-center gap-1">range ±{range}<input type="range" min={0.5} max={8} step={0.5} value={range} onChange={(e) => setRange(+e.target.value)} aria-label="Slice range" /></label>
        </div>
        <p className="mt-2 text-[12.5px] leading-relaxed text-muted">Every point of the curve is a full forward pass with only {paramName(ref)} changed. The gradient is the slope of this curve at the current value; the tangent touches it there. Click the chart to set the parameter. Select another weight or bias on the network to slice along it.</p>
      </Card>
      <Card title="why cross-entropy for sigmoid outputs" right={
        <label className="flex items-center gap-1 text-[12px] text-muted">target y =
          <select className="rounded border border-line bg-panel-2 px-1" value={yT} onChange={(e) => setYT(+e.target.value)} aria-label="Target for the logit comparison"><option value={1}>1</option><option value={0}>0</option></select>
        </label>}>
        <div className="grid gap-2 @6xl:grid-cols-2">
          <div><div className="eyebrow mb-1">loss vs logit z</div>
            <LineChart ariaLabel="Loss against logit for MSE and BCE" xLabel="z" height={150} series={[{ points: curves.L[0], color: c.violet, label: 'MSE (σ(z) − y)²' }, { points: curves.L[1], color: c.amber, label: 'BCE' }]} markers={[{ x: zBad, y: bad.mse.L, color: c.violet }, { x: zBad, y: bad.bce.L, color: c.amber }]} />
          </div>
          <div><div className="eyebrow mb-1">|∂L/∂z|: the learning signal</div>
            <LineChart ariaLabel="Gradient magnitude against logit" xLabel="z" height={150} series={[{ points: curves.D[0], color: c.violet, label: 'MSE: 2(σ−y)σ(1−σ)' }, { points: curves.D[1], color: c.amber, label: 'BCE: σ − y' }]} markers={[{ x: zBad, y: Math.abs(bad.mse.dz), color: c.violet }, { x: zBad, y: Math.abs(bad.bce.dz), color: c.amber }]} />
          </div>
        </div>
        <p className="mt-2 text-[12.5px] leading-relaxed text-muted">
          At z = {zBad} the neuron is confidently wrong (ŷ = {n4(bad.mse.yhat)}, target {yT}). With MSE its gradient is only <span className="font-mono text-violet">{n4(Math.abs(bad.mse.dz))}</span> because σ′(z) is tiny there, so learning stalls.
          With binary cross-entropy the σ′ factor cancels and the gradient is <span className="font-mono text-amber">{n4(Math.abs(bad.bce.dz))}</span>: the more wrong, the stronger the push.
        </p>
        <div className="mt-1 overflow-x-auto scroll-thin"><Tex block tex={`\\frac{\\partial L_{\\mathrm{MSE}}}{\\partial z} = 2(\\hat y - y)\\,\\hat y(1-\\hat y), \\qquad \\frac{\\partial L_{\\mathrm{BCE}}}{\\partial z} = \\hat y - y`} /></div>
        <p className="text-[11.5px] text-faint">Current network loss: {LOSS_LABEL[cfg.loss]}. Curves computed by the engine on a single sigmoid unit.</p>
      </Card>
    </div>
  )
}
