/**
 * Phase 3: gradient descent over many iterations. Loss curve from the recorded history (click to rewind),
 * live decision boundary, a learning-rate experiment and the XOR comparison (no hidden layer vs hidden layer).
 */
import { FastForward, Pause, Play, SkipForward, StepForward } from 'lucide-react'
import { memo, useEffect, useMemo, useState } from 'react'
import { interpolateRgb } from 'd3'
import { useColors } from '../../sim/appState'
import {
  accuracy, datasetLoss, decisionGrid, defaultOpt, initParams, makeDataset, targetsFor, trainEpochs, trainStep, zeroOptState,
  type NetConfig, type Params, type Sample, type TrainState,
} from '../../engine/nn'
import { Card } from '../../components/viz'
import { LineChart } from './charts'
import { n4 } from './narrate'
import { useMlp } from './store'

export const bounds = (data: Sample[]): { xr: [number, number]; yr: [number, number] } => {
  const xs = data.map((d) => d.x[0]), ys = data.map((d) => d.x[1])
  const pad = (a: number, b: number): [number, number] => { const m = Math.max(0.25, (b - a) * 0.15); return [a - m, b + m] }
  return { xr: pad(Math.min(...xs), Math.max(...xs)), yr: pad(Math.min(...ys), Math.max(...ys)) }
}

export const DecisionMap = memo(function DecisionMap({ cfg, params, data, size = 260, n = 40, highlight }: { cfg: NetConfig; params: Params; data: Sample[]; size?: number; n?: number; highlight?: number }) {
  const c = useColors()
  const { xr, yr } = useMemo(() => bounds(data), [data])
  const cells = useMemo(() => decisionGrid(cfg, params, xr, yr, n), [cfg, params, xr, yr, n])
  const col0 = interpolateRgb(c.panel2, c.violet), col1 = interpolateRgb(c.panel2, c.amber)
  const sx = (x: number) => ((x - xr[0]) / (xr[1] - xr[0])) * size, sy = (y: number) => size - ((y - yr[0]) / (yr[1] - yr[0])) * size
  const cw = size / n
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="block w-full rounded-lg border border-line" role="img" aria-label="Decision boundary">
      {cells.map((q, k) => <rect key={k} x={sx(q.x) - cw / 2} y={sy(q.y) - cw / 2} width={cw + 0.5} height={cw + 0.5} fill={q.p >= 0.5 ? col1(Math.min(1, (q.p - 0.5) * 2) * 0.85) : col0(Math.min(1, (0.5 - q.p) * 2) * 0.85)} />)}
      {data.map((d, k) => <circle key={k} cx={sx(d.x[0])} cy={sy(d.x[1])} r={k === highlight ? 6 : data.length > 20 ? 3.2 : 5.5} fill={d.label ? c.amber : c.violet} stroke={k === highlight ? c.cyan : c.bg} strokeWidth={k === highlight ? 2.5 : 1.2} />)}
    </svg>
  )
})

export function TrainPanel() {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), data = useMlp((s) => s.data), history = useMlp((s) => s.history), hIndex = useMlp((s) => s.hIndex)
  const training = useMlp((s) => s.training), setTraining = useMlp((s) => s.setTraining), train = useMlp((s) => s.train), rewind = useMlp((s) => s.rewind)
  const batch = useMlp((s) => s.batch), opt = useMlp((s) => s.opt), sampleIdx = useMlp((s) => s.sampleIdx), trainSeed = useMlp((s) => s.trainSeed)
  const perEpoch = Math.ceil(data.length / Math.max(1, Math.min(batch, data.length)))
  const two = cfg.sizes[0] === 2 && data.length > 1
  const [logY, setLogY] = useState(false)
  // long runs are executed in chunks so the page stays responsive
  const [pending, setPending] = useState(0)
  const run = (n: number) => { if (n <= 40) train(n); else setPending(n) }
  useEffect(() => {
    if (!pending) return
    const id = setTimeout(() => { const k = Math.min(pending, data.length > 200 ? 8 : 40); train(k); setPending(pending - k) }, 0)
    return () => clearTimeout(id)
  }, [pending, train, data.length])

  useEffect(() => {
    if (!training) return
    const id = setInterval(() => { train(data.length > 200 ? 2 : perEpoch > 4 ? Math.ceil(perEpoch / 2) : 1) }, 70)
    return () => clearInterval(id)
  }, [training, train, perEpoch, data.length])
  useEffect(() => { if (training && history.length > 1400) setTraining(false) }, [training, history.length, setTraining])

  const cur = history[hIndex]
  const acc = accuracy(cfg, params, data)
  const curve = history.map((h, i) => [i, h.loss] as [number, number])

  // learning-rate experiment, from the initial state, same seed and batches
  const [lrRuns, setLrRuns] = useState<{ lr: number; losses: number[] }[] | null>(null)
  const runLr = () => {
    const sub = data.slice(0, 200), pe = Math.ceil(sub.length / Math.max(1, Math.min(batch, sub.length)))
    const p0 = history[0].params, epochs = Math.max(3, Math.min(200, Math.floor(400 / pe)))
    setLrRuns([0.1, 1, 10, 40].map((m) => { const lr = +(opt.lr * m).toPrecision(2); return { lr, losses: trainEpochs(cfg, p0, sub, { ...opt, lr }, batch, trainSeed, epochs).losses } }))
  }
  useEffect(() => setLrRuns(null), [cfg, data, opt.kind, batch])

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn" onClick={() => train(1)} disabled={training || pending > 0}><StepForward size={14} />1 batch</button>
        <button type="button" className="btn" onClick={() => run(perEpoch)} disabled={training || pending > 0}><SkipForward size={14} />1 epoch</button>
        <button type="button" className="btn" onClick={() => run(perEpoch * 10)} disabled={training || pending > 0}><FastForward size={14} />10 epochs</button>
        <button type="button" className="btn btn-primary" onClick={() => setTraining(!training)} aria-pressed={training} data-testid="train-toggle">{training ? <><Pause size={14} />Pause training</> : <><Play size={14} />Train</>}</button>
        <span className="font-mono text-[12px] text-muted" data-testid="train-status">
          iteration {cur.iter} · epoch {cur.epoch} · loss <span className="text-amber">{n4(cur.loss)}</span>{pending > 0 ? <span className="text-cyan"> · running, {pending} steps to go</span> : null}{Number.isFinite(acc) ? <> · accuracy <span className="text-teal">{(acc * 100).toFixed(0)}%</span></> : null}
        </span>
        <span className="ml-auto text-[11.5px] text-faint">{opt.kind} · η {opt.lr} · batch {Math.min(batch, data.length)} of {data.length} · {perEpoch} step{perEpoch > 1 ? 's' : ''}/epoch · seeded shuffles</span>
      </div>
      <div className={`grid gap-3 ${two ? '@4xl:grid-cols-[minmax(0,1fr)_300px]' : ''}`}>
        <Card title="loss over the recorded history (click to rewind)" right={<label className="flex items-center gap-1 text-[11.5px] text-muted"><input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} />log scale</label>}>
          <LineChart ariaLabel="Training loss curve" xLabel="recorded state" yLabel="loss (dataset)" height={230} logY={logY}
            series={[{ points: curve, color: c.amber }]} markers={[{ x: hIndex, y: cur.loss, color: c.cyan, label: n4(cur.loss) }]}
            onClickX={(x) => rewind(Math.round(x))} />
          <p className="mt-1 text-[12px] text-muted">Each point is the mean loss over all {data.length} samples after one recorded update. Rewinding restores the exact parameters and optimizer state; training again from there starts a new branch.</p>
        </Card>
        {two && (
          <Card title="decision boundary (live)">
            <DecisionMap cfg={cfg} params={params} data={data} highlight={sampleIdx} />
            <p className="mt-1 text-[12px] text-muted">Background: p(class 1) from a forward pass at each of 1,600 grid points. Amber = class 1, violet = class 0.</p>
          </Card>
        )}
      </div>
      <Card title="learning-rate experiment" right={<button type="button" className="btn" onClick={runLr}>Run from the initial weights</button>}>
        {lrRuns ? (
          <>
            <LineChart ariaLabel="Loss curves for four learning rates" xLabel="update step" yLabel="loss" height={210} logY
              series={lrRuns.map((r, k) => ({ points: r.losses.map((v, i) => [i, Math.min(v, 1e3)] as [number, number]), color: [c.violet, c.cyan, c.amber, c.rose][k], label: `η = ${r.lr} → ${Number.isFinite(r.losses.at(-1)!) ? n4(r.losses.at(-1)!) : 'diverged'}` }))} />
            <p className="mt-1 text-[12px] text-muted">Same initial weights, same batches; only η differs{data.length > 200 ? ' (first 200 training samples)' : ''}. Too small crawls, too large overshoots or oscillates.</p>
          </>
        ) : <p className="text-[12.5px] text-muted">Trains four copies of this network from its initial weights with η × 0.1, × 1, × 10 and × 40, and overlays their loss curves.</p>}
      </Card>
      <XorCompare />
    </div>
  )
}

/** XOR with and without a hidden layer, trained by the engine, scrubbable by epoch. */
export function XorCompare() {
  const data = useMemo(() => targetsFor(makeDataset('xor', 1, 2), 1), [])
  const [seed, setSeed] = useState(1)
  const runs = useMemo(() => {
    const mk = (cfg: NetConfig) => {
      const p0 = initParams(cfg.sizes, seed), o = { ...defaultOpt, lr: 1 }
      let ts: TrainState = { params: p0, opt: zeroOptState(p0), epoch: 0, iter: 0, cursor: 0 }
      const snaps = [{ epoch: 0, params: p0 }]
      for (let e = 1; e <= 1500; e++) { ts = trainStep(cfg, ts, data, o, 4, 1).state; if (e % 25 === 0) snaps.push({ epoch: e, params: ts.params }) }
      return { cfg, snaps }
    }
    return [mk({ sizes: [2, 1], acts: ['sigmoid'], loss: 'bce' }), mk({ sizes: [2, 4, 1], acts: ['tanh', 'sigmoid'], loss: 'bce' })]
  }, [data, seed])
  const [k, setK] = useState(runs[0].snaps.length - 1)
  const c = useColors()
  return (
    <Card title="XOR: why hidden layers matter" right={<label className="flex items-center gap-1 text-[12px] text-muted">seed<input type="number" className="w-14 rounded border border-line bg-panel-2 px-1 font-mono" value={seed} min={0} onChange={(e) => setSeed(Math.max(0, Math.round(+e.target.value || 0)))} aria-label="XOR initialization seed" /></label>}>
      <label className="mb-2 flex items-center gap-2 text-[12.5px] text-muted">epoch <span className="w-12 font-mono text-fg">{runs[0].snaps[k].epoch}</span>
        <input type="range" min={0} max={runs[0].snaps.length - 1} value={k} onChange={(e) => setK(+e.target.value)} className="flex-1" aria-label="XOR epoch" />
      </label>
      <div className="grid gap-3 @2xl:grid-cols-2">
        {runs.map((r, i) => {
          const p = r.snaps[k].params, acc = accuracy(r.cfg, p, data)
          const curve = r.snaps.map((s) => [s.epoch, datasetLoss(r.cfg, s.params, data)] as [number, number])
          return (
            <div key={i} className="grid gap-1">
              <div className="flex items-baseline justify-between text-[13px]"><b>{i === 0 ? '2 → 1 (no hidden layer)' : '2 → 4 → 1 (tanh hidden layer)'}</b><span className={`font-mono ${acc === 1 ? 'text-teal' : 'text-rose'}`} data-testid={`xor-acc-${i}`}>{(acc * 100).toFixed(0)}% correct</span></div>
              <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] items-center gap-2">
                <DecisionMap cfg={r.cfg} params={p} data={data} size={200} n={30} />
                <LineChart ariaLabel="XOR loss" height={150} xLabel="epoch" series={[{ points: curve, color: i ? c.teal : c.rose }]} markers={[{ x: r.snaps[k].epoch, y: curve[k][1], color: c.cyan }]} yDomain={[0, 0.8]} />
              </div>
            </div>
          )
        })}
      </div>
      <p className="mt-2 text-[12.5px] leading-relaxed text-muted">A single neuron draws one straight line, and no straight line separates XOR: it can get at most 3 of 4 points right, however long it trains. The hidden layer bends the input space so that the output neuron's line works. Both trained with full-batch SGD (η = 1) on binary cross-entropy.</p>
    </Card>
  )
}
