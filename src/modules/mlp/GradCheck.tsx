/** Gradient checker: backprop against central finite differences for every parameter, plus the ε trade-off. */
import { CheckCircle2, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { allParamRefs, gradientCheck, paramName } from '../../engine/nn'
import { Card, Tex } from '../../components/viz'
import { selKey } from './model'
import { shownSample, useMlp } from './store'

const CAP = 50

export const PASS_REL = 1e-5
const passes = (r: { relErr: number; absErr: number }) => r.relErr < PASS_REL || r.absErr < 1e-9

export function GradCheck() {
  const c = useColors()
  const cfg = useMlp((s) => s.cfg), params = useMlp((s) => s.params), data = useMlp((s) => s.data), sampleIdx = useMlp((s) => s.sampleIdx), probe = useMlp((s) => s.probe), selection = useMlp((s) => s.selection), select = useMlp((s) => s.select)
  const [scope, setScope] = useState<'sample' | 'data'>('sample')
  const [eps, setEps] = useState(-5)
  const [layer, setLayer] = useState(0)
  const [page, setPage] = useState(0)
  const set = useMemo(() => (scope === 'sample' ? [shownSample({ probe, data, sampleIdx })] : data.slice(0, CAP)), [scope, data, sampleIdx, probe])
  const refs = useMemo(() => allParamRefs(cfg.sizes).filter((r) => layer === 0 || r.layer === layer), [cfg.sizes, layer])
  const rows = useMemo(() => gradientCheck(cfg, params, set, 10 ** eps, refs), [cfg, params, set, eps, refs])
  const failed = rows.filter((r) => !passes(r)).length
  const maxRel = Math.max(0, ...rows.map((r) => (Math.abs(r.analytic) + Math.abs(r.numeric) > 1e-9 ? r.relErr : 0)))
  const PER = 80, pages = Math.ceil(rows.length / PER)

  // ε sweep for the selected (or first) parameter: truncation error vs floating-point round-off
  const target = selection && selection.kind !== 'neuron' ? selection : refs[0]
  const sweep = useMemo(() => Array.from({ length: 13 }, (_, k) => -1 - k).map((e) => { const r = gradientCheck(cfg, params, set, 10 ** e, [target])[0]; return { e, abs: Math.max(1e-17, r.absErr) } }), [cfg, params, set, target])
  const W = 360, H = 150, lx = (e: number) => 34 + ((-e - 1) / 12) * (W - 44), ly = (v: number) => 10 + ((-Math.log10(v) - 0) / 17) * (H - 34)

  return (
    <div className="grid gap-3">
      <Card title="how it works">
        <div className="grid gap-2 text-[13px] leading-relaxed text-muted md:grid-cols-[1fr_auto] md:items-center">
          <p>Nudge one parameter up and down by ε, run the network twice, and measure the slope of the loss. If backpropagation is implemented correctly, this numerical slope agrees with the analytic gradient for every parameter. Pass: relative error below 10⁻⁵.</p>
          <Tex block tex={`\\frac{\\partial L}{\\partial \\theta} \\approx \\frac{L(\\theta+\\epsilon) - L(\\theta-\\epsilon)}{2\\epsilon},\\qquad \\text{rel.err} = \\frac{|g_{bp} - g_{num}|}{|g_{bp}| + |g_{num}|}`} />
        </div>
      </Card>
      <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
        <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Loss over">
          {(['sample', 'data'] as const).map((s) => <button key={s} type="button" role="radio" aria-checked={scope === s} onClick={() => setScope(s)} className={`rounded px-2 py-0.5 ${scope === s ? 'bg-raise text-cyan' : 'text-muted'}`}>{s === 'sample' ? (probe ? 'your sample' : `sample ${sampleIdx + 1}`) : data.length > CAP ? `first ${CAP} of ${data.length} samples` : `all ${data.length} samples`}</button>)}
        </div>
        <label className="flex items-center gap-2 text-muted">ε = 10<sup>{eps}</sup><input type="range" min={-9} max={-1} step={1} value={eps} onChange={(e) => setEps(+e.target.value)} aria-label="Finite-difference epsilon exponent" /></label>
        <select className="rounded-md border border-line bg-panel-2 px-2 py-1" value={layer} onChange={(e) => { setLayer(+e.target.value); setPage(0) }} aria-label="Layer filter">
          <option value={0}>all layers</option>{cfg.sizes.slice(1).map((_, l) => <option key={l} value={l + 1}>layer {l + 1}</option>)}
        </select>
        <span className={`ml-auto flex items-center gap-1.5 font-mono ${failed ? 'text-rose' : 'text-teal'}`} data-testid="gradcheck-summary">
          {failed ? <XCircle size={15} /> : <CheckCircle2 size={15} />}{rows.length - failed}/{rows.length} pass · max rel. error {maxRel.toExponential(1)}
        </span>
      </div>
      <div className="grid gap-3 @4xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="max-h-[440px] overflow-auto scroll-thin rounded-lg border border-line">
          <table className="w-full text-[12px]" data-testid="gradcheck-table">
            <thead className="sticky top-0 bg-panel text-left text-faint">
              <tr><th className="px-2 py-1">parameter</th><th className="px-2 py-1 text-right">backprop</th><th className="px-2 py-1 text-right">numerical</th><th className="px-2 py-1 text-right">abs. error</th><th className="px-2 py-1 text-right">rel. error</th><th className="px-2 py-1" /></tr>
            </thead>
            <tbody>
              {rows.slice(page * PER, page * PER + PER).map((r) => {
                const k = selKey(r.ref), on = k === selKey(selection)
                return (
                  <tr key={k} onClick={() => select(r.ref)} className={`cursor-pointer border-t border-line font-mono ${on ? 'bg-raise text-cyan' : 'hover:bg-panel-2'}`}>
                    <td className="px-2 py-0.5">{paramName(r.ref)}</td>
                    <td className="px-2 py-0.5 text-right">{r.analytic.toFixed(7)}</td>
                    <td className="px-2 py-0.5 text-right">{r.numeric.toFixed(7)}</td>
                    <td className="px-2 py-0.5 text-right text-muted">{r.absErr.toExponential(1)}</td>
                    <td className="px-2 py-0.5 text-right text-muted">{Math.abs(r.analytic) + Math.abs(r.numeric) > 1e-9 ? r.relErr.toExponential(1) : '—'}</td>
                    <td className="px-2 py-0.5">{passes(r) ? <CheckCircle2 size={13} className="text-teal" /> : <XCircle size={13} className="text-rose" />}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {pages > 1 && (
            <div className="flex items-center gap-2 border-t border-line p-1.5 text-[12px]">
              <button type="button" className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}>prev</button>
              <span className="font-mono text-muted">page {page + 1}/{pages}</span>
              <button type="button" className="btn" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>next</button>
            </div>
          )}
        </div>
        <Card title={`choosing ε · ${paramName(target)}`}>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Absolute error against epsilon">
            {[-1, -4, -7, -10, -13].map((e) => <g key={e}><line x1={lx(e)} x2={lx(e)} y1={10} y2={H - 24} stroke={c.line} /><text x={lx(e)} y={H - 10} fontSize={10} textAnchor="middle" fill={c.faint} fontFamily="var(--font-mono)">1e{e}</text></g>)}
            {[1e-3, 1e-7, 1e-11, 1e-15].map((v) => <g key={v}><line x1={34} x2={W - 10} y1={ly(v)} y2={ly(v)} stroke={c.line} strokeDasharray="2 3" /><text x={30} y={ly(v) + 3} fontSize={9.5} textAnchor="end" fill={c.faint} fontFamily="var(--font-mono)">{v.toExponential(0)}</text></g>)}
            <path d={sweep.map((p, i) => `${i ? 'L' : 'M'}${lx(p.e)},${ly(p.abs)}`).join('')} fill="none" stroke={c.cyan} strokeWidth={2} />
            {sweep.map((p) => <circle key={p.e} cx={lx(p.e)} cy={ly(p.abs)} r={p.e === eps ? 5 : 2.5} fill={p.e === eps ? c.amber : c.cyan} />)}
          </svg>
          <p className="mt-1 text-[12px] leading-snug text-muted">|backprop − numerical| for each ε (log scales). Large ε: the finite difference misses curvature (error ∝ ε²). Tiny ε: floating-point round-off dominates (error ∝ 1/ε). The sweet spot is around 10⁻⁵.</p>
        </Card>
      </div>
    </div>
  )
}
