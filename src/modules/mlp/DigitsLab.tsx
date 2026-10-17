/**
 * Phase 4 · the trained digit classifier: draw or load an 8×8 digit, see the prediction, the hidden layers,
 * the input saliency, the first-layer weight images and the confusion matrix. All values come from the
 * exported weights (or your own, if you retrained) through engine/nn.
 */
import { Eraser, Microscope, Shuffle, Trash2, X } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { interpolateRgb } from 'd3'
import { useColors, type Colors } from '../../sim/appState'
import { DIGITS, digitTest, fromPixels, isTrainedDigits } from '../../engine/digits'
import { accuracy, confusion, datasetLoss, forward, inputGrad, output } from '../../engine/nn'
import { Card } from '../../components/viz'
import { LineChart } from './charts'
import { n4 } from './narrate'
import { useMlp } from './store'

const gray = (c: Colors) => interpolateRgb(c.panel2, c.fg)
function diverge(c: Colors, max: number) {
  const pos = interpolateRgb(c.panel2, c.amber), neg = interpolateRgb(c.panel2, c.violet)
  return (v: number) => { const t = Math.min(1, Math.abs(v) / (max || 1)); return v >= 0 ? pos(Math.sqrt(t)) : neg(Math.sqrt(t)) }
}

/** 8×8 grid of cells; `paint` makes it an editor */
export function Grid8({ values, color, size = 168, paint, label, testid }: { values: number[]; color: (v: number) => string; size?: number; paint?: (i: number, erase: boolean) => void; label: string; testid?: string }) {
  const c = useColors()
  const down = useRef<null | boolean>(null)
  const cell = size / 8
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="block w-full touch-none rounded-md border border-line" role="img" aria-label={label} data-testid={testid}
      onPointerUp={() => (down.current = null)} onPointerLeave={() => (down.current = null)} onContextMenu={(e) => paint && e.preventDefault()}>
      {values.map((v, i) => (
        <rect key={i} x={(i % 8) * cell} y={Math.floor(i / 8) * cell} width={cell} height={cell} fill={color(v)} stroke={c.line} strokeWidth={paint ? 0.6 : 0.2}
          style={paint ? { cursor: 'crosshair' } : undefined} data-testid={testid ? `${testid}-${i}` : undefined}
          onPointerDown={paint ? (e) => { const erase = e.button === 2 || e.shiftKey; down.current = erase; paint(i, erase) } : undefined}
          onPointerEnter={paint ? () => { if (down.current !== null) paint(i, down.current) } : undefined} />
      ))}
    </svg>
  )
}

export function DigitsLab() {
  const cfg = useMlp((s) => s.cfg), loadPreset = useMlp((s) => s.loadPreset)
  const fits = cfg.sizes[0] === 64 && cfg.sizes[cfg.sizes.length - 1] === 10
  if (!fits) {
    return (
      <Card title="trained digit classifier">
        <p className="mb-2 text-[13px] text-muted">The digit lab needs a network with 64 inputs and 10 outputs. Load the trained 64 → 16 → 16 → 10 classifier to continue.</p>
        <button type="button" className="btn btn-primary" onClick={() => loadPreset('digits')}>Load the trained digit classifier</button>
      </Card>
    )
  }
  return <DigitsInner />
}

function DigitsInner() {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), setProbe = useMlp((s) => s.setProbe), setTab = useMlp((s) => s.setTab), select = useMlp((s) => s.select)
  const test = digitTest()
  const [testIdx, setTestIdx] = useState(0)
  const [px, setPx] = useState<number[]>(() => DIGITS.X[DIGITS.test[0]].slice())
  const [label, setLabel] = useState<number | undefined>(DIGITS.y[DIGITS.test[0]])
  const [erase, setErase] = useState(false)
  const [cellPick, setCellPick] = useState<[number, number] | null>(null)
  const trained = isTrainedDigits(cfg, params)

  const sample = useMemo(() => fromPixels(px, label), [px, label])
  const t = useMemo(() => forward(cfg, params, sample.x), [cfg, params, sample])
  const probs = output(t), pred = probs.indexOf(Math.max(...probs))
  // saliency: ∂ log p(pred) / ∂x  (= −∂L/∂x with the predicted class as target)
  const sal = useMemo(() => inputGrad(cfg, params, sample.x, Array.from({ length: 10 }, (_, k) => (k === pred ? 1 : 0))).map((v) => -v), [cfg, params, sample, pred])
  const salMax = Math.max(1e-9, ...sal.map(Math.abs))
  const conf = useMemo(() => confusion(cfg, params, test, 10), [cfg, params, test])
  const testAcc = useMemo(() => accuracy(cfg, params, test), [cfg, params, test])
  const testLoss = useMemo(() => datasetLoss(cfg, params, test), [cfg, params, test])
  const wMax = Math.max(1e-9, ...params.W[0].flat().map(Math.abs))
  const g = gray(c)

  const load = (k: number) => { const i = DIGITS.test[k]; setTestIdx(k); setPx(DIGITS.X[i].slice()); setLabel(DIGITS.y[i]) }
  const paint = (i: number, er: boolean) => { setPx((p) => { const q = p.slice(); q[i] = er || erase ? 0 : Math.min(16, q[i] + 8); return q }); setLabel(undefined) }
  const wrong = conf.pred.map((p, k) => (p !== test[k].label ? k : -1)).filter((k) => k >= 0)
  const picked = cellPick ? conf.pred.map((p, k) => (test[k].label === cellPick[0] && p === cellPick[1] ? k : -1)).filter((k) => k >= 0) : []

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
        {trained
          ? <span className="sim-badge" data-testid="digits-badge">trained model · NumPy, Adam, {DIGITS.training.epochs} epochs · test accuracy {(DIGITS.metrics.testAcc * 100).toFixed(1)} % on 450 held-out images</span>
          : <span className="sim-badge toy" data-testid="digits-badge">your weights (not the exported trained model) · test accuracy {(testAcc * 100).toFixed(1)} %</span>}
        <span className="text-muted">Data: UCI optical digits, 8×8, 1,347 training / 450 test images.</span>
      </div>
      <div className="grid gap-3 @lg:grid-cols-[210px_minmax(0,1fr)]">
        <Card title={label === undefined ? 'your drawing' : `test image ${testIdx + 1} / 450 · true digit ${label}`}>
          <div className="mx-auto max-w-[220px]"><Grid8 values={px} color={(v) => g(v / 16)} paint={paint} label="8 by 8 digit editor" testid="digit-canvas" /></div>
          <div className="mt-2 flex flex-wrap gap-1">
            <button type="button" className="btn" aria-pressed={erase} onClick={() => setErase(!erase)} title="Erase (or right-click / shift)"><Eraser size={13} />{erase ? 'erasing' : 'erase'}</button>
            <button type="button" className="btn" onClick={() => { setPx(new Array(64).fill(0)); setLabel(undefined) }}><Trash2 size={13} />clear</button>
            <button type="button" className="btn" onClick={() => load((testIdx + 449) % 450)} aria-label="Previous test image">‹</button>
            <button type="button" className="btn" onClick={() => load((testIdx + 1) % 450)} aria-label="Next test image">›</button>
            <button type="button" className="btn" onClick={() => wrong.length && load(wrong[(wrong.indexOf(testIdx) + 1) % wrong.length])} disabled={!wrong.length} title="Next misclassified test image"><Shuffle size={13} />mistake</button>
          </div>
          <p className="mt-1 text-[11.5px] text-faint">Click or drag to paint (each pass adds 8 of 16). Right-click or shift to erase.</p>
          <button type="button" className="btn mt-2 w-full justify-center" onClick={() => { setProbe(fromPixels(px, label ?? pred)); setTab('microscope') }} data-testid="digit-to-microscope"><Microscope size={13} />Trace this digit in the microscope</button>
          {label === undefined && <p className="mt-1 text-[11px] text-faint">Unlabelled drawing: the microscope uses the predicted digit ({pred}) as the target.</p>}
        </Card>
        <div className="grid gap-3">
          <Card title="prediction">
            <div className="flex flex-wrap items-end gap-4">
              <div><div className="font-display text-[54px] leading-none text-cyan" data-testid="digit-pred">{pred}</div><div className="font-mono text-[12px] text-muted">p = {n4(probs[pred])}{label !== undefined ? (pred === label ? ' · correct' : ` · wrong (true ${label})`) : ''}</div></div>
              <div className="flex flex-1 items-end gap-1" style={{ height: 96 }} aria-label="Class probabilities">
                {probs.map((p, k) => (
                  <div key={k} className="flex flex-1 flex-col items-center gap-0.5" title={`p(${k}) = ${p.toFixed(4)}`}>
                    <div className="w-full rounded-t" style={{ height: Math.max(1, p * 74), background: k === pred ? c.cyan : k === label ? c.amber : c.lineStrong }} />
                    <span className="font-mono text-[11px] text-muted">{k}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
          <div className="grid gap-3 @5xl:grid-cols-2">
            <Card title="hidden activations (click a unit)">
              {[1, 2].map((l) => {
                const a = t.a[l], m = Math.max(1e-9, ...a)
                return (
                  <div key={l} className="mb-2">
                    <div className="mb-1 font-mono text-[11px] text-muted">layer {l} · ReLU · {a.filter((v) => v > 0).length}/16 active</div>
                    <div className="grid grid-cols-16 gap-[2px]" style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}>
                      {a.map((v, j) => <button key={j} type="button" onClick={() => { select({ kind: 'neuron', layer: l, j }); setProbe(sample); setTab('microscope') }} title={`h${j + 1}⁽${l}⁾ = ${v.toFixed(4)}`} className="h-6 rounded-sm border border-line" style={{ background: interpolateRgb(c.panel2, c.cyan)(Math.sqrt(v / m)) }} aria-label={`Layer ${l} unit ${j + 1}`} />)}
                    </div>
                  </div>
                )
              })}
              <p className="text-[11.5px] text-muted">Grey = 0: a ReLU unit that is silent for this image. Different digits switch on different units.</p>
            </Card>
            <Card title={`saliency · ∂ log p(${pred}) / ∂x`}>
              <div className="mx-auto grid max-w-[300px] grid-cols-2 gap-2">
                <Grid8 values={px} color={(v) => g(v / 16)} label="input" size={120} />
                <Grid8 values={sal} color={diverge(c, salMax)} label="saliency map" size={120} testid="saliency" />
              </div>
              <p className="mt-1 text-[11.5px] text-muted"><span className="text-amber">Amber</span>: more ink here would raise p({pred}); <span className="text-violet">violet</span>: would lower it. Computed by backpropagating to the input, one backward pass.</p>
            </Card>
          </div>
        </div>
      </div>
      <Card title="first-layer weights · what each hidden unit of layer 1 looks for (click to inspect)">
        <div className="grid grid-cols-4 gap-2 @lg:grid-cols-8">
          {params.W[0].map((row, j) => (
            <button key={j} type="button" className="grid gap-0.5 text-left" onClick={() => { select({ kind: 'neuron', layer: 1, j }); setProbe(sample); setTab('microscope') }} aria-label={`Weights of layer 1 unit ${j + 1}`}>
              <Grid8 values={row} color={diverge(c, wMax)} label={`weights of unit ${j + 1}`} size={80} />
              <span className="font-mono text-[10.5px] text-muted">h{j + 1} · a = {t.a[1][j].toFixed(2)}</span>
            </button>
          ))}
        </div>
        <p className="mt-1 text-[11.5px] text-muted">Each image is one row of W⁽¹⁾ laid out as 8×8. The unit's z is the dot product of this image with the input (plus a bias): amber pixels vote for the unit, violet against.</p>
      </Card>
      <div className="grid gap-3 @4xl:grid-cols-2">
        <Card title={`confusion matrix · 450 test images · accuracy ${(testAcc * 100).toFixed(1)} % · loss ${n4(testLoss)}`}>
          <ConfusionMatrix m={conf.matrix} onPick={(r, k) => setCellPick(cellPick && cellPick[0] === r && cellPick[1] === k ? null : [r, k])} picked={cellPick} />
          {cellPick && (
            <div className="mt-2">
              <div className="mb-1 flex items-center justify-between text-[12px]"><span>true {cellPick[0]}, predicted {cellPick[1]} · {picked.length} image{picked.length === 1 ? '' : 's'}</span><button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => setCellPick(null)} aria-label="Close"><X size={12} /></button></div>
              <div className="flex flex-wrap gap-1">{picked.slice(0, 24).map((k) => <button key={k} type="button" className="w-10" onClick={() => load(k)} aria-label={`Load test image ${k + 1}`}><Grid8 values={DIGITS.X[DIGITS.test[k]]} color={(v) => g(v / 16)} label="thumbnail" size={40} /></button>)}</div>
            </div>
          )}
        </Card>
        <Card title="how it was trained (Python log, per epoch)">
          <LineChart ariaLabel="Digit training curves" xLabel="epoch" yLabel="cross-entropy" height={200} logY
            series={[{ points: DIGITS.curve.map((p) => [p.epoch, p.trainLoss]), color: c.cyan, label: 'train loss' }, { points: DIGITS.curve.map((p) => [p.epoch, p.testLoss]), color: c.amber, label: 'test loss' }]} />
          <p className="mt-1 text-[12px] leading-relaxed text-muted">Training loss falls to {DIGITS.metrics.trainLoss.toExponential(1)} (100 % on the training images) while test loss bottoms out near {Math.min(...DIGITS.curve.map((p) => p.testLoss)).toFixed(3)} and then wanders: the network starts memorizing. {DIGITS.training.optimizer === 'adam' ? 'Adam' : DIGITS.training.optimizer}, η = {DIGITS.training.lr}, batch {DIGITS.training.batch}, He initialization, seed {DIGITS.training.seed} (training/train_digits.py).</p>
        </Card>
      </div>
    </div>
  )
}

function ConfusionMatrix({ m, onPick, picked }: { m: number[][]; onPick: (r: number, k: number) => void; picked: [number, number] | null }) {
  const c = useColors()
  const max = Math.max(...m.flat())
  const diag = interpolateRgb(c.panel2, c.teal), off = interpolateRgb(c.panel2, c.rose)
  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="mx-auto border-separate border-spacing-[2px] font-mono text-[11px]" data-testid="confusion">
        <thead><tr><th className="text-faint">t \ p</th>{m.map((_, k) => <th key={k} className="w-7 text-muted">{k}</th>)}</tr></thead>
        <tbody>
          {m.map((row, r) => (
            <tr key={r}><th className="pr-1 text-muted">{r}</th>
              {row.map((v, k) => (
                <td key={k} onClick={() => v && onPick(r, k)} className={`h-7 w-7 rounded text-center ${v ? 'cursor-pointer' : ''} ${picked && picked[0] === r && picked[1] === k ? 'outline outline-2 outline-cyan' : ''}`}
                  style={{ background: v ? (r === k ? diag(0.25 + 0.75 * v / max) : off(Math.min(1, 0.35 + v / 4))) : 'transparent', color: r === k && v / max > 0.5 ? c.bg : c.fg }}>{v || ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
