/** Derived, memoized computations for the current state (one sample's forward and backward pass). */
import { useMemo } from 'react'
import { applyUpdate, backward, forward, lossValue, output, type Backward, type Sample, type Trace, type UpdateRecord } from '../../engine/nn'
import { microSteps, revealAt, type MicroStep, type Reveal } from './model'
import { isModified, shownSample, useMlp } from './store'

export interface NetView {
  sample: Sample
  trace: Trace
  bwd: Backward
  loss: number
  yhat: number[]
  steps: MicroStep[]
  step: MicroStep
  reveal: Reveal
  L: number
  /** update preview for the displayed sample with the chosen optimizer */
  update: { records: UpdateRecord[]; lossAfter: number }
  modified: boolean
  /** the recorded snapshot's forward pass, for "compare original vs modified" */
  original: { trace: Trace; loss: number } | null
}

/** off the microscope tab every value is shown: the step-by-step reveal belongs to the microscope */
const FULL = (L: number): Reveal => ({ zUpTo: L, aUpTo: L, loss: true, deltaFrom: 1, gradFrom: 1, update: false })

export function useNet(): NetView {
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), data = useMlp((s) => s.data), sampleIdx = useMlp((s) => s.sampleIdx), probe = useMlp((s) => s.probe)
  const micro = useMlp((s) => s.micro), mode = useMlp((s) => s.mode), opt = useMlp((s) => s.opt)
  const history = useMlp((s) => s.history), hIndex = useMlp((s) => s.hIndex), tab = useMlp((s) => s.tab)
  return useMemo(() => {
    const sample = shownSample({ probe, data, sampleIdx })
    const L = cfg.sizes.length - 1
    const trace = forward(cfg, params, sample.x)
    const bwd = backward(cfg, params, trace, sample.y)
    const loss = lossValue(cfg, trace, sample.y)
    const steps = microSteps(L, mode)
    const idx = Math.min(micro, steps.length - 1)
    const u = applyUpdate(params, bwd, history[hIndex].opt, opt)
    const lossAfter = lossValue(cfg, forward(cfg, u.params, sample.x), sample.y)
    const modified = isModified({ params, history, hIndex })
    let original: NetView['original'] = null
    if (modified) { const t0 = forward(cfg, history[hIndex].params, sample.x); original = { trace: t0, loss: lossValue(cfg, t0, sample.y) } }
    return { sample, trace, bwd, loss, yhat: output(trace), steps, step: steps[idx], reveal: tab === 'microscope' ? revealAt(steps, idx, L) : FULL(L), L, update: { records: u.records, lossAfter }, modified, original }
  }, [cfg, params, data, sampleIdx, probe, micro, mode, opt, history, hIndex, tab])
}
