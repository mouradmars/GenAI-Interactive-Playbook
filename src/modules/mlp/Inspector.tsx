/** Right panel — the Mathematical Microscope: the current step with real numbers, then the selected element. */
import { AnimatePresence, motion } from 'framer-motion'
import { GitCompare, RotateCcw, Save } from 'lucide-react'
import { useMemo } from 'react'
import { useApp } from '../../sim/appState'
import { ACT_LABEL, gradientCheck, gradOf, getParam, paramName, type ParamRef } from '../../engine/nn'
import { Tex } from '../../components/viz'
import { neuronName, phaseOf, PHASE_LABEL, type MicroStep } from './model'
import { narrate, n4 } from './narrate'
import type { NetView } from './derive'
import { useMlp } from './store'

function Row({ k, v, accent, testid }: { k: React.ReactNode; v: React.ReactNode; accent?: string; testid?: string }) {
  return <div className="flex items-baseline justify-between gap-3 border-t border-line py-1 first:border-t-0"><span className="text-muted">{k}</span><span className={`font-mono ${accent ?? ''}`} data-testid={testid}>{v}</span></div>
}

export function Inspector({ net }: { net: NetView }) {
  const { depth } = useApp()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), selection = useMlp((s) => s.selection), opt = useMlp((s) => s.opt)
  const { step, L } = net
  const focus = selection && selection.layer === ('layer' in step ? step.layer : step.kind === 'dout' ? L : -1) ? selection.j : 0
  const nar = narrate(step, focus, { cfg, params, trace: net.trace, bwd: net.bwd, sample: net.sample, loss: net.loss, opt, records: net.update.records, lossAfter: net.update.lossAfter, depth })
  const phase = phaseOf(step)
  return (
    <div className="grid min-w-0 gap-4 text-[13px]">
      <div>
        <div className="eyebrow">mathematical microscope</div>
        <AnimatePresence mode="wait">
          <motion.div key={`${net.steps.indexOf(step)}-${focus}-${step.kind}`} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} className="mt-1 grid gap-2">
            <div className="flex items-center gap-2">
              <span className={`chip ${phase === 'backward' ? 'text-rose' : phase === 'update' ? 'text-teal' : phase === 'loss' ? 'text-amber' : 'text-cyan'}`}>{PHASE_LABEL[phase]}</span>
              <h2 className="font-display text-[19px] leading-tight" data-testid="inspector-title">{nar.title}</h2>
            </div>
            <p className="leading-relaxed text-muted">{nar.plain}</p>
            <div className="overflow-x-auto scroll-thin rounded-lg border border-line bg-panel-2/60 px-3 py-2"><Tex block tex={nar.general} /></div>
            <div className="grid min-w-0 gap-1" data-testid="inspector-numbers">
              {nar.numbers.map((n, i) => <div key={i} className="min-w-0 overflow-x-auto scroll-thin text-[13px]"><Tex block tex={n} /></div>)}
            </div>
            {nar.note && <p className="rounded-md border-l-2 border-amber bg-amber/5 px-2 py-1 text-[12.5px] text-muted">{nar.note}</p>}
          </motion.div>
        </AnimatePresence>
      </div>
      <Selected net={net} />
    </div>
  )
}

function stepIndex(steps: MicroStep[], pred: (s: MicroStep) => boolean) { return steps.findIndex(pred) }

function Selected({ net }: { net: NetView }) {
  const cfg = useMlp((s) => s.cfg), selection = useMlp((s) => s.selection), setMicro = useMlp((s) => s.setMicro), select = useMlp((s) => s.select)
  const { trace: t, bwd: g, reveal, L, steps } = net
  if (!selection) return <div className="rounded-lg border border-dashed border-line p-3 text-muted">Click a neuron, a connection or a bias diamond to inspect it.</div>
  const Jump = ({ pred, children }: { pred: (s: MicroStep) => boolean; children: React.ReactNode }) => {
    const i = stepIndex(steps, pred)
    return i < 0 ? <span className="text-faint">not in this mode</span> : <button type="button" className="text-[12px] text-cyan underline decoration-dotted" onClick={() => setMicro(i)}>{children}</button>
  }
  if (selection.kind === 'neuron') {
    const { layer: l, j } = selection
    const known = l === 0 || reveal.aUpTo >= l, zk = l > 0 && reveal.zUpTo >= l, dk = l > 0 && reveal.deltaFrom <= l
    const act = l > 0 ? cfg.acts[l - 1] : null
    return (
      <section className="grid gap-1 rounded-xl border border-line bg-panel/70 p-3" aria-label="Selected neuron">
        <div className="eyebrow">selected neuron</div>
        <div className="font-mono text-[17px]">{neuronName(l, j, L)} <span className="font-sans text-[12.5px] text-muted">· {l === 0 ? 'input' : l === L ? 'output' : `hidden layer ${l}`}{act ? ` · ${ACT_LABEL[act]}` : ''}</span></div>
        {l > 0 && <Row k="weighted sum z" v={zk ? n4(t.z[l][j]) : <Jump pred={(s) => s.kind === 'z' && s.layer === l}>compute →</Jump>} />}
        <Row k="activation a" v={known ? n4(t.a[l][j]) : <Jump pred={(s) => s.kind === 'act' && s.layer === l}>compute →</Jump>} accent="text-cyan" testid="sel-activation" />
        {l > 0 && act !== 'softmax' && <Row k="slope f′(z)" v={zk ? n4(g.fprime[l][j]) : '?'} />}
        {l > 0 && <Row k="∂L/∂a" v={dk ? n4(g.dA[l][j]) : <Jump pred={(s) => (l === L ? s.kind === 'dout' : s.kind === 'delta' && s.layer === l)}>backprop →</Jump>} />}
        {l > 0 && <Row k="δ = ∂L/∂z" v={dk ? n4(g.delta[l][j]) : '?'} accent="text-rose" testid="sel-delta" />}
        {l === 0 && <Row k="∂L/∂x (input sensitivity)" v={reveal.gradFrom <= 1 ? n4(g.dA[0][j]) : '?'} />}
        {l > 0 && <p className="mt-1 text-[12px] text-muted">Incoming: {cfg.sizes[l - 1]} weights + 1 bias (<button type="button" className="text-cyan underline decoration-dotted" onClick={() => select({ kind: 'b', layer: l, j })}>inspect the bias</button>). Outgoing: {l < L ? `${cfg.sizes[l + 1]} weights` : 'the loss'}.</p>}
      </section>
    )
  }
  return <ParamCard r={selection} net={net} />
}

function ParamCard({ r, net }: { r: ParamRef; net: NetView }) {
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), opt = useMlp((s) => s.opt), setParamValue = useMlp((s) => s.setParamValue)
  const compare = useMlp((s) => s.compare), setCompare = useMlp((s) => s.setCompare), recordEdits = useMlp((s) => s.recordEdits), revertEdits = useMlp((s) => s.revertEdits)
  const history = useMlp((s) => s.history), hIndex = useMlp((s) => s.hIndex), setMicro = useMlp((s) => s.setMicro)
  const { trace: t, bwd: g, reveal, steps, sample } = net
  const v = getParam(params, r), grad = gradOf(g, r), gk = reveal.gradFrom <= r.layer
  const rec = net.update.records.find((x) => x.ref.kind === r.kind && x.ref.layer === r.layer && x.ref.j === r.j && (x.ref.kind === 'b' || (r.kind === 'w' && x.ref.i === r.i)))!
  const num = useMemo(() => gradientCheck(cfg, params, [sample], 1e-5, [r])[0], [cfg, params, sample, r])
  const orig = getParam(history[hIndex].params, r)
  const gi = steps.findIndex((s) => s.kind === 'grad' && s.layer === r.layer)
  return (
    <section className="grid gap-1 rounded-xl border border-line bg-panel/70 p-3" aria-label="Selected parameter">
      <div className="eyebrow">selected {r.kind === 'w' ? 'weight' : 'bias'}</div>
      <div className="font-mono text-[17px]">{paramName(r)} <span className="font-sans text-[12.5px] text-muted">· {r.kind === 'w' ? `from ${neuronName(r.layer - 1, r.i, net.L)} to ${neuronName(r.layer, r.j, net.L)}` : `bias of ${neuronName(r.layer, r.j, net.L)}`}</span></div>
      <label className="mt-1 grid grid-cols-[1fr_92px] items-center gap-2">
        <input type="range" min={-3} max={3} step={0.01} value={Math.max(-3, Math.min(3, v))} onChange={(e) => setParamValue(r, +e.target.value)} aria-label={`Edit ${paramName(r)}`} />
        <input type="number" step={0.05} value={+v.toFixed(4)} onChange={(e) => setParamValue(r, parseFloat(e.target.value))} className="rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-[12.5px]" aria-label={`Value of ${paramName(r)}`} data-testid="param-input" />
      </label>
      {net.modified && (
        <div className="my-1 grid gap-1.5 rounded-lg border border-violet/50 bg-violet/5 p-2 text-[12px]" data-testid="modified-banner">
          <div>Edited (recorded value {n4(orig)}). Everything downstream was recomputed.</div>
          {net.original && <div className="font-mono">ŷ {net.original.trace.a[net.L].slice(0, 3).map((x) => n4(x)).join(', ')} → {net.yhat.slice(0, 3).map((x) => n4(x)).join(', ')} · L {n4(net.original.loss)} → {n4(net.loss)}</div>}
          <div className="flex flex-wrap gap-1">
            <button type="button" className="btn" aria-pressed={compare} onClick={() => setCompare(!compare)}><GitCompare size={13} />Compare on network</button>
            <button type="button" className="btn" onClick={revertEdits}><RotateCcw size={13} />Revert</button>
            <button type="button" className="btn" onClick={recordEdits}><Save size={13} />Record edit</button>
          </div>
        </div>
      )}
      <Row k="value" v={n4(v)} accent="text-amber" />
      <Row k="∂L/∂θ (backprop)" v={gk ? n4(grad) : gi >= 0 ? <button type="button" className="text-[12px] text-cyan underline decoration-dotted" onClick={() => setMicro(gi)}>backprop →</button> : '?'} accent="text-rose" testid="sel-grad" />
      <Row k="finite difference" v={gk ? n4(num.numeric) : '?'} />
      <Row k="relative error" v={gk ? num.relErr.toExponential(1) : '?'} />
      {r.kind === 'w' && <Row k="= δ · a" v={gk ? `${n4(g.delta[r.layer][r.j])} × ${n4(t.a[r.layer - 1][r.i])}` : '?'} />}
      <div className="mt-1 eyebrow">next update ({opt.kind}, η = {opt.lr})</div>
      <Row k="step Δθ" v={n4(rec.step)} accent={rec.step >= 0 ? 'text-rose' : 'text-teal'} />
      <Row k="new value" v={`${n4(rec.old)} → ${n4(rec.next)}`} />
      {opt.kind === 'adam' && <Row k="m̂, v̂" v={`${n4(rec.detail.mHat)}, ${rec.detail.vHat.toExponential(2)}`} />}
      {opt.kind === 'momentum' && <Row k="velocity" v={n4(rec.detail.velocity)} />}
      <p className="mt-1 text-[12px] text-muted">{gk ? (grad > 0 ? 'Positive gradient: increasing this parameter would increase the loss, so descent decreases it.' : grad < 0 ? 'Negative gradient: increasing this parameter lowers the loss, so descent increases it.' : 'Zero gradient: this parameter does not affect the loss for this sample.') : 'The gradient is computed during the backward pass.'}</p>
    </section>
  )
}
