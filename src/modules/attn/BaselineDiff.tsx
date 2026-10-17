/** "What changed since the saved state?" — per-tensor differences computed by compareTraces. */
import { Bookmark, RotateCcw, X as Close } from 'lucide-react'
import { compareTraces, type AttnTrace } from '../../engine/attention'
import { fx } from './common'
import { useBaselineTrace } from './common'
import { attnStore, useAttn } from './store'

export function BaselineDiff({ trace, compact = false }: { trace: AttnTrace; compact?: boolean }) {
  const base = useBaselineTrace()
  const label = useAttn((s) => s.baseline?.label)
  const st = attnStore.getState()
  if (!base) {
    return (
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[12px] text-muted">
        <button type="button" className="btn !h-7" onClick={() => st.saveBaseline('saved state')} data-testid="save-baseline"><Bookmark size={13} />Save state for comparison</button>
        {!compact && <span>then edit anything and see exactly which tensors change.</span>}
      </div>
    )
  }
  const comparable = base.Tq === trace.Tq && base.cfg.H === trace.cfg.H
  const d = comparable ? compareTraces(base, trace) : null
  const cell = (v: number) => <td className={`px-1 text-center ${v > 1e-12 ? 'text-amber' : 'text-teal'}`}>{v > 1e-12 ? fx(v, 3) : '='}</td>
  return (
    <div className="mt-2 rounded-md border border-line bg-panel-2/40 p-2 text-[12px]" data-testid="baseline-diff">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="eyebrow">compared with: {label}</span>
        <button type="button" className="btn !h-6 !px-1.5 text-[11px]" onClick={st.restoreBaseline} disabled={!comparable && base.Tq !== trace.Tq}><RotateCcw size={11} />restore</button>
        <button type="button" className="btn !h-6 !px-1.5 text-[11px]" onClick={() => st.saveBaseline('saved state')}><Bookmark size={11} />save current</button>
        <button type="button" className="btn icon-btn !h-6 !w-6" onClick={st.clearBaseline} aria-label="Stop comparing"><Close size={11} /></button>
      </div>
      {d ? (
        <>
          <table className="font-mono text-[11.5px]">
            <thead><tr className="text-faint"><th className="pr-2 text-left font-normal">max |Δ|</th>{['Q', 'K', 'V', 'S', 'A', 'head'].map((k) => <th key={k} className="px-1 font-normal">{k}</th>)}</tr></thead>
            <tbody>
              {d.heads.map((h, k) => <tr key={k}><td className="pr-2 text-muted">head {k + 1}</td>{cell(h.Q)}{cell(h.K)}{cell(h.V)}{cell(h.S)}{cell(h.A)}{cell(h.O)}</tr>)}
            </tbody>
          </table>
          <div className="mt-1 font-mono text-[11.5px]">final output: <span className={d.outChanged ? 'text-amber' : 'text-teal'} data-testid="diff-out">{d.outChanged ? `changed (max |Δ| = ${fx(d.out, 4)})` : 'unchanged'}</span> · attention weights: <span data-testid="diff-A" className={d.heads.some((h) => h.changed.A) ? 'text-amber' : 'text-teal'}>{d.heads.some((h) => h.changed.A) ? 'changed' : 'unchanged'}</span></div>
          <div className="text-[11px] text-faint">“=” means identical to 1e−12.</div>
        </>
      ) : <div className="text-muted">The saved state has a different sequence length or number of heads; only same-shape states can be compared cell by cell.</div>}
    </div>
  )
}
