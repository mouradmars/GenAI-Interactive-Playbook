/** Eight experiments: hypothesis → prediction → run → visual and numerical result → explanation → reset. */
import { CheckCircle2, ExternalLink, FlaskConical, RotateCcw, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { Grid, HeadPicker, tokLabel, useTokens } from './common'
import { EXPERIMENTS, type ExpResult, type Experiment } from './experiments'
import { shuffle } from './challenges'
import { cfgOf, attnStore, useAttn } from './store'

export function ExperimentsTab() {
  const X = useAttn((s) => s.X), params = useAttn((s) => s.params), H = useAttn((s) => s.H), causal = useAttn((s) => s.causal), scale = useAttn((s) => s.scale), pad = useAttn((s) => s.pad), blocked = useAttn((s) => s.blocked), headsOn = useAttn((s) => s.headsOn), h = useAttn((s) => s.sel.h)
  const state = useMemo(() => ({ X, params, cfg: cfgOf({ H, causal, scale, pad, blocked, headsOn }) }), [X, params, H, causal, scale, pad, blocked, headsOn])
  return (
    <div className="grid gap-3">
      <Card title="how the experiments work" right={<HeadPicker />}>
        <p className="text-[12.5px] leading-relaxed text-muted">Each experiment starts from the lab's current state (edit it in the guided pipeline first if you like), applies one change, and runs both versions through the same engine. Make a prediction, run it, then compare the numbers. <b className="text-fg">Open in the lab</b> applies the change to the main view and saves the old state for side-by-side comparison.</p>
      </Card>
      <div className="grid gap-3 @5xl:grid-cols-2">
        {EXPERIMENTS.map((e) => <ExperimentCard key={e.id} e={e} state={state} h={Math.min(h, H - 1)} />)}
      </div>
    </div>
  )
}

function ExperimentCard({ e, state, h }: { e: Experiment; state: Parameters<Experiment['run']>[0]; h: number }) {
  const tokens = useTokens()
  const [pred, setPred] = useState<string | null>(null)
  const [res, setRes] = useState<ExpResult | null>(null)
  const run = () => setRes(e.run(state, h))
  const reset = () => { setRes(null); setPred(null) }
  const open = () => {
    if (!res) return
    const st = attnStore.getState()
    st.saveBaseline(`before experiment ${e.n}`)
    st.setParams(res.afterState.params)
    if (res.afterState.X !== state.X) st.setInput(res.afterState.X)
    const c = res.afterState.cfg
    if (c.H !== state.cfg.H) st.setH(c.H)
    st.setCausal(c.causal); st.setScale(c.scale); st.setHeadsOn(c.headsOn.slice())
    st.setTab('guided')
  }
  const rl = tokens.map((_, k) => tokLabel(tokens, k))
  const masked = (t: ExpResult['before']) => t.M.map((r) => r.map((v) => v === -Infinity))
  return (
    <Card title={`experiment ${e.n} · ${e.title}`} className="@container/exp">
      <div className="grid gap-2 text-[12.5px]" data-testid={`exp-${e.id}`}>
        <p><span className="text-faint">Hypothesis:</span> {e.hypothesis}</p>
        <div>
          <p className="mb-1"><span className="text-faint">Your prediction:</span> {e.question}</p>
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={`Prediction for experiment ${e.n}`}>
            {shuffle(e.options, e.n * 7).map((o) => <button key={o} type="button" role="radio" aria-checked={pred === o} className={`btn !h-auto min-h-7 py-1 text-left text-[12px] ${pred === o ? 'active' : ''}`} onClick={() => setPred(o)}>{o}</button>)}
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          <button type="button" className="btn btn-primary" onClick={run} data-testid={`run-${e.id}`}><FlaskConical size={14} />Run experiment</button>
          <button type="button" className="btn" onClick={reset} disabled={!res && !pred}><RotateCcw size={13} />Reset</button>
          {res && <button type="button" className="btn" onClick={open}><ExternalLink size={13} />Open in the lab</button>}
        </div>
        {res && (
          <div className="grid gap-2 rounded-lg border border-line bg-panel-2/40 p-2" data-testid={`exp-result-${e.id}`}>
            <div className="font-mono text-[11.5px] text-muted">change: {res.change}</div>
            {(e.visual === 'heatmaps' || e.visual === 'entropy' || e.visual === 'mask' || e.visual === 'inputs' || e.visual === 'ablation') && (
              <div className="flex flex-wrap gap-2">
                <Grid m={res.before.heads[Math.min(h, res.before.cfg.H - 1)].A} role="prob" mode="prob" rowLabels={rl} masked={masked(res.before)} cell={40} title="weights before" ariaLabel="weights before" />
                <Grid m={res.after.heads[Math.min(h, res.after.cfg.H - 1)].A} role="prob" mode="prob" masked={masked(res.after)} cell={40} title="weights after" ariaLabel="weights after" />
                {e.visual !== 'heatmaps' && <>
                  <Grid m={res.before.out} role="out" cell={40} title="output before" ariaLabel="output before" />
                  <Grid m={res.after.out} role="out" cell={40} title="output after" ariaLabel="output after" />
                </>}
              </div>
            )}
            {e.visual === 'shapes' && (
              <div className="flex flex-wrap gap-2">
                {res.after.heads.map((hd) => <Grid key={hd.h} m={hd.A} role="prob" mode="prob" masked={masked(res.after)} cell={30} title={`head ${hd.h + 1} (d_k = ${res.after.dk})`} ariaLabel={`head ${hd.h + 1} after`} />)}
              </div>
            )}
            <table className="font-mono text-[12px]"><tbody>{res.metrics.map((m) => <tr key={m.label}><td className="pr-3 align-top text-muted">{m.label}</td><td>{m.value}</td></tr>)}</tbody></table>
            <p className="text-[12.5px]"><span className="text-faint">Observed:</span> {res.observed}</p>
            {pred && (pred === e.answer
              ? <p className="flex items-center gap-1 text-teal"><CheckCircle2 size={14} />Your prediction matches the result.</p>
              : <p className="flex items-center gap-1 text-rose"><XCircle size={14} />The result differs from your prediction (“{pred}”). Expected: {e.answer}.</p>)}
            <p className="text-[12.5px] text-muted"><span className="text-faint">Explanation:</span> {e.explanation}</p>
          </div>
        )}
      </div>
    </Card>
  )
}
