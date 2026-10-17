/** Explore, compare, challenge and lesson modes, plus the inspector, for module 09. */
import { useMemo, useState } from 'react'
import { Card, Tex } from '../../components/viz'
import { useApp, useColors } from '../../sim/appState'
import {
  ARCH_LABEL, SYMBOLS, answerOk, backward, cost, forward, hiddenOf, makeSeqQuestions, simParams, trained, vnorm, type Arch, type Regime, type SeqQuestion, type SimConfig,
} from '../../engine/seq'
import { CaveatView, CompareView, CursorBar, LSTMCellDiagram, LogChart, Predictions, SourceBar, StateHeatmap } from './views'
import { MISCONCEPTIONS, OBJECTIVES, QUIZ, STAGES } from './lesson'
import { SEQ_EXAMPLES } from './examples'
import { f3, f4, pct, sci, useSeq } from './context'

/* ---------------------------------------------------------------- explore */
export function ExplorePanel() {
  const { runs, cursor, unit, setUnit } = useSeq()
  const c = useColors()
  const r = runs.rnn.trace, l = runs.lstm.trace
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <SourceBar />
      <CursorBar />
      <Card title="predictions"><Predictions /></Card>
      <div className="grid gap-3 @5xl:grid-cols-2">
        <Card title="RNN hidden state">{r.arch === 'rnn' && <StateHeatmap rows={r.h} t={cursor.t} label="h_t" unit={unit} onUnit={setUnit} />}</Card>
        <Card title="LSTM cell at the cursor"><LSTMCellDiagram t={cursor.t} unit={unit} />{l.arch === 'lstm' && <StateHeatmap rows={l.c} t={cursor.t} label="cell state c_t" maxAbs={Math.max(1, ...l.c.flat().map(Math.abs))} unit={unit} onUnit={setUnit} />}</Card>
      </div>
      <Card title="gradient reaching each input (current objective)">
        <LogChart xLabel="input position" yLabel="‖∂L/∂x_t‖" marker={cursor.t} series={(['rnn', 'lstm', 'attn'] as Arch[]).map((a) => ({ name: ARCH_LABEL[a], ys: runs[a].grads.dx.map((d) => vnorm(d)), color: a === 'rnn' ? c.rose : a === 'lstm' ? c.cyan : c.amber }))} />
      </Card>
      <CaveatView />
    </div>
  )
}

/* ---------------------------------------------------------------- compare */
interface Cfg { label: string; source: 'trained' | 'sim'; regime: Regime; sim: SimConfig }
export function ComparePanel() {
  const { sim, regime } = useSeq()
  const [a, setA] = useState<Cfg>({ label: 'A', source: 'trained', regime: 'mixed', sim })
  const [b, setB] = useState<Cfg>({ label: 'B', source: 'trained', regime: regime === 'mixed' ? 'long' : 'mixed', sim })
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <p className="max-w-[80ch] text-[13.5px] text-muted">Two configurations on the same input sequence. Compare the two training regimes, trained against random weights, or two random settings (for example recurrent scale 0.6 against 1.8). The table and gradient curves are recomputed for each side.</p>
      <div className="grid gap-3 @5xl:grid-cols-2">
        <CompareSide cfg={a} setCfg={(p) => setA((x) => ({ ...x, ...p }))} />
        <CompareSide cfg={b} setCfg={(p) => setB((x) => ({ ...x, ...p }))} />
      </div>
      <Card title="three architectures at one time step (current settings)"><CompareView /></Card>
    </div>
  )
}

function CompareSide({ cfg, setCfg }: { cfg: Cfg; setCfg: (p: Partial<Cfg>) => void }) {
  const { ids, target, objective } = useSeq()
  const c = useColors()
  const runs = useMemo(() => (['rnn', 'lstm', 'attn'] as Arch[]).map((a) => {
    const p = cfg.source === 'trained' ? trained(cfg.regime, a).params : simParams(a, cfg.sim)
    const tr = forward(p, ids)
    return { a, tr, g: backward(p, tr, target, objective) }
  }), [cfg, ids, target, objective])
  return (
    <Card title={`configuration ${cfg.label}`} right={<span className={`sim-badge ${cfg.source === 'sim' ? 'toy' : ''}`}>{cfg.source === 'trained' ? `trained · ${cfg.regime}` : 'simulation'}</span>}>
      <div className="grid gap-2">
        <div className="flex flex-wrap gap-2">
          <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label={`source ${cfg.label}`}>
            {(['trained', 'sim'] as const).map((s) => <button key={s} type="button" role="radio" aria-checked={cfg.source === s} onClick={() => setCfg({ source: s })} className={`rounded-md px-2.5 py-1 text-[12px] ${cfg.source === s ? 'bg-raise text-cyan' : 'text-muted'}`}>{s === 'trained' ? 'trained' : 'random weights'}</button>)}
          </div>
          {cfg.source === 'trained' && <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label={`regime ${cfg.label}`}>
            {(['mixed', 'long'] as const).map((r) => <button key={r} type="button" role="radio" aria-checked={cfg.regime === r} onClick={() => setCfg({ regime: r })} className={`rounded-md px-2.5 py-1 text-[12px] ${cfg.regime === r ? 'bg-raise text-cyan' : 'text-muted'}`}>{r === 'mixed' ? 'mixed 2–20' : 'long 15–25'}</button>)}
          </div>}
        </div>
        {cfg.source === 'sim' && (
          <div className="grid grid-cols-2 gap-2">
            <label className="grid gap-1 text-[12px] text-muted font-num" htmlFor={`cs-${cfg.label}`}>scale {cfg.sim.scale.toFixed(2)}<input id={`cs-${cfg.label}`} type="range" min={0.1} max={3} step={0.05} value={cfg.sim.scale} onChange={(e) => setCfg({ sim: { ...cfg.sim, scale: +e.target.value } })} /></label>
            <label className="grid gap-1 text-[12px] text-muted font-num" htmlFor={`cf-${cfg.label}`}>forget bias {cfg.sim.forgetBias.toFixed(1)}<input id={`cf-${cfg.label}`} type="range" min={-3} max={5} step={0.1} value={cfg.sim.forgetBias} onChange={(e) => setCfg({ sim: { ...cfg.sim, forgetBias: +e.target.value } })} /></label>
          </div>
        )}
      </div>
      <table className="mt-3 w-full font-mono text-[12px] font-num">
        <thead><tr className="text-left text-faint"><th className="font-normal">model</th><th className="font-normal">says</th><th className="font-normal">p(correct)</th><th className="font-normal">‖∂/∂x₁‖ / ‖∂/∂x_T‖</th></tr></thead>
        <tbody>{runs.map(({ a, tr, g }) => {
          const pred = tr.probs.indexOf(Math.max(...tr.probs)); const n = g.dx.map((d) => vnorm(d))
          return <tr key={a} className="border-b border-line/50"><td>{ARCH_LABEL[a]}</td><td className={pred === target ? 'text-ok' : 'text-rose'}>{SYMBOLS[pred]}</td><td>{pct(tr.probs[target])}</td><td>{sci(n[n.length - 1] ? n[0] / n[n.length - 1] : 0)}</td></tr>
        })}</tbody>
      </table>
      <div className="mt-2"><LogChart height={180} xLabel="input position" yLabel="gradient norm" series={runs.map(({ a, g }) => ({ name: ARCH_LABEL[a], ys: g.dx.map((d) => vnorm(d)), color: a === 'rnn' ? c.rose : a === 'lstm' ? c.cyan : c.amber }))} /></div>
    </Card>
  )
}

/* -------------------------------------------------------------- challenge */
export type Results = Record<string, { given: string; ok: boolean; revealed: boolean }>
export function ChallengePanel({ results, setResults }: { results: Results; setResults: (r: Results) => void }) {
  const { seed, setSeed } = useSeq()
  const qs = useMemo(() => makeSeqQuestions(seed), [seed])
  const [draft, setDraft] = useState<Record<string, string>>({})
  const key = (q: SeqQuestion) => `${seed}-${q.id}`
  const score = qs.filter((q) => results[key(q)]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title={`student challenge · ${score} / ${qs.length} correct · question set ${seed}`}
        right={<div className="flex items-center gap-2">
          <label htmlFor="sch-seed" className="font-mono text-[11px] text-faint">seed</label>
          <input id="sch-seed" type="number" className="h-7 w-20 px-2 font-mono text-[12px]" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} />
          <button type="button" className="btn h-7" onClick={() => setSeed(seed + 1)}>New question set</button>
        </div>}>
        <p className="max-w-[80ch] text-[13px] text-muted">Seven questions from a seeded generator: the same seed gives the whole class the same numbers. Work them out by hand (a calculator for tanh, σ and exp is fine), then check. Answers within 0.5% (or ±0.0015) are accepted.</p>
      </Card>
      {qs.map((q, n) => {
        const k = key(q), r = results[k]
        const check = (given: string) => setResults({ ...results, [k]: { given, ok: answerOk(q, given), revealed: r?.revealed ?? false } })
        return (
          <Card key={k} title={`question ${n + 1} · ${q.kind}`}>
            <p className="text-[14.5px] leading-relaxed">{q.prompt}</p>
            {q.options ? (
              <fieldset className="mt-2 grid gap-1.5 sm:grid-cols-2">
                <legend className="sr-only">Options for question {n + 1}</legend>
                {q.options.map((o, i) => (
                  <label key={o} className={`flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1 text-[13px] ${r?.given === String(i) ? (r.ok ? 'border-ok/70' : 'border-rose/70') : 'border-line'}`}>
                    <input type="radio" name={`sq-${q.id}`} checked={r?.given === String(i)} onChange={() => check(String(i))} />{o}
                  </label>
                ))}
              </fieldset>
            ) : (
              <form className="mt-2 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); check(draft[k] ?? '') }}>
                <label htmlFor={`sans-${q.id}`} className="sr-only">Answer to question {n + 1}</label>
                <input id={`sans-${q.id}`} className="h-9 w-40 px-2 font-mono" value={draft[k] ?? ''} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} placeholder="your answer" autoComplete="off" />
                <button type="submit" className="btn btn-primary h-9">Check</button>
              </form>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <button type="button" className="btn h-8" onClick={() => setResults({ ...results, [k]: { given: r?.given ?? '', ok: r?.ok ?? false, revealed: true } })}>Reveal answer</button>
              {r && r.given !== '' && <span role="status" className={`text-[13px] ${r.ok ? 'text-ok' : 'text-rose'}`}>{r.ok ? 'Correct' : 'Not yet'}</span>}
            </div>
            {(r?.revealed || r?.ok) && <p className="mt-2 rounded-lg border border-line bg-panel-2 p-2 font-mono text-[12.5px] leading-relaxed text-muted font-num">Answer: <span className="text-amber">{q.options ? q.options[q.answer] : q.answer.toFixed(4)}</span>. {q.explain}</p>}
          </Card>
        )
      })}
    </div>
  )
}

/* ----------------------------------------------------------------- lesson */
export function LessonPanel({ goTo, openExample }: { goTo: (id: string) => void; openExample: (id: string) => void }) {
  const [ans, setAns] = useState<(number | null)[]>(QUIZ.map(() => null))
  const sc = ans.filter((a, i) => a === QUIZ[i].answer).length
  return (
    <div className="grid gap-3 @5xl:grid-cols-2 [&>*]:min-w-0">
      <Card title="learning objectives"><ol className="grid list-decimal gap-1.5 pl-5 text-[13.5px] leading-relaxed">{OBJECTIVES.map((o) => <li key={o}>{o}</li>)}</ol></Card>
      <Card title="guided walkthrough">
        <ol className="grid gap-1 text-[13px]">
          {STAGES.map((s, i) => (
            <li key={s.id}><button type="button" className="flex w-full items-baseline gap-2 rounded px-1 py-0.5 text-left hover:bg-panel-2" onClick={() => goTo(s.id)}>
              <span className="w-6 font-mono text-[11px] text-faint font-num">{i + 1}</span><span>{s.title}</span><span className="ml-auto font-mono text-[10.5px] text-faint">{s.group}</span>
            </button></li>
          ))}
        </ol>
      </Card>
      <Card title="worked examples" className="@5xl:col-span-2">
        <div className="grid gap-3 md:grid-cols-2">
          {SEQ_EXAMPLES.map((ex, k) => (
            <div key={ex.id} className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="flex flex-wrap items-center gap-2"><span className="eyebrow">example {k + 1}</span><span className={`sim-badge ${ex.source === 'sim' ? 'toy' : ''}`}>{ex.source === 'sim' ? 'simulation' : `trained · ${ex.regime}`}</span></div>
              <h3 className="mt-1 font-display text-[17px]">{ex.title}</h3>
              <p className="mt-1 font-mono text-[12.5px] tracking-[0.15em] text-cyan">{ex.seq}</p>
              <p className="mt-2 text-[13px] leading-relaxed text-muted">{ex.why}</p>
              <button type="button" className="btn mt-2" onClick={() => openExample(ex.id)}>Open this example</button>
            </div>
          ))}
        </div>
      </Card>
      <Card title={`student questions · ${sc} / ${QUIZ.length} correct`}>
        <div className="grid gap-4">
          {QUIZ.map((q, i) => (
            <fieldset key={i} className="grid gap-1.5">
              <legend className="mb-1 text-[13.5px]">{i + 1}. {q.q}</legend>
              {q.options.map((o, k) => (
                <label key={k} className={`flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1 text-[13px] ${ans[i] === k ? (k === q.answer ? 'border-ok/70' : 'border-rose/70') : 'border-line'}`}>
                  <input type="radio" name={`seqq-${i}`} checked={ans[i] === k} onChange={() => setAns(ans.map((a, j) => (j === i ? k : a)))} />{o}
                </label>
              ))}
              {ans[i] !== null && <p className={`text-[12.5px] ${ans[i] === q.answer ? 'text-ok' : 'text-rose'}`}>{ans[i] === q.answer ? 'Correct. ' : 'Not quite. '}{q.explain}</p>}
            </fieldset>
          ))}
        </div>
      </Card>
      <Card title="misconception warnings">
        <div className="grid gap-3">
          {MISCONCEPTIONS.map((m) => (
            <div key={m.myth} className="rounded-lg border border-rose/40 p-3">
              <div className="text-[13.5px] text-rose">Myth: {m.myth}</div>
              <div className="mt-1 text-[13.5px] leading-relaxed">{m.truth}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* -------------------------------------------------------------- inspector */
export function SeqInspector() {
  const { stage, tab, runs, cursor, unit, ids, target, source, regime, sim, objective } = useSeq()
  const { depth } = useApp()
  const shown = tab === 'guided' ? stage : STAGES.find((s) => s.id === (tab === 'compare' ? 'compare' : tab === 'explore' ? 'gradients' : 'task'))!
  const t = cursor.t, T = ids.length
  const r = runs.rnn.trace, l = runs.lstm.trace, a = runs.attn.trace
  const u = Math.min(unit, hiddenOf(runs.rnn.params) - 1)
  const rows: [string, string][] = []
  if (r.arch === 'rnn') rows.push([`RNN h_${t + 1}[u${u + 1}]`, f4(r.h[t][u])])
  if (l.arch === 'lstm') { const ul = Math.min(unit, l.c[0].length - 1); rows.push([`LSTM f / i / o (u${ul + 1})`, `${f3(l.f[t][ul])} / ${f3(l.i[t][ul])} / ${f3(l.o[t][ul])}`], [`LSTM c_${t + 1}`, f4(l.c[t][ul])]) }
  if (a.arch === 'attn') { const top = a.A[t].slice(0, t + 1); const m = top.indexOf(Math.max(...top)); rows.push([`attention row ${t + 1}: max`, `${f3(top[m])} on position ${m + 1}`], [`row ${t + 1} sums to`, f4(top.reduce((x, y) => x + y, 0))]) }
  return (
    <div className="grid gap-4" aria-live="polite">
      <div><div className="eyebrow">{tab === 'guided' ? shown.group : tab}</div><h2 className="mt-1 font-display text-[22px] leading-tight">{shown.title}</h2></div>
      <div className="overflow-x-auto rounded-lg border border-line bg-panel-2 px-3 py-3 scroll-thin"><Tex tex={shown.equation} block /></div>
      <p className="max-w-[60ch] text-[13.5px] leading-relaxed text-fg/90">{shown[depth]}</p>
      <section>
        <div className="eyebrow mb-1.5">source</div>
        <p className={`text-[12.5px] ${source === 'trained' ? 'text-teal' : 'text-amber'}`}>{source === 'trained' ? `Trained models, regime “${regime}”. Weights from training/train_seq.py; this forward pass reproduces PyTorch’s logits to 4 decimals.` : `Simulation: seeded random weights (seed ${sim.seed}, width ${sim.hidden}, scale ${sim.scale}, forget bias ${sim.forgetBias}). Not trained.`}</p>
      </section>
      <section>
        <div className="eyebrow mb-1.5">input · target “{SYMBOLS[target]}” · T = {T}</div>
        <dl className="grid gap-1 font-mono text-[12px] font-num">
          {(['rnn', 'lstm', 'attn'] as Arch[]).map((x) => (
            <div key={x} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{ARCH_LABEL[x]}</dt><dd className={runs[x].pred === target ? 'text-ok' : 'text-rose'}>p = {pct(runs[x].trace.probs[target])} · says {SYMBOLS[runs[x].pred]}</dd></div>
          ))}
        </dl>
      </section>
      <section>
        <div className="eyebrow mb-1.5">at the cursor · t = {t + 1}</div>
        <dl className="grid gap-1 font-mono text-[12px] font-num">{rows.map(([k, v]) => <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k}</dt><dd>{v}</dd></div>)}</dl>
      </section>
      {depth !== 'beginner' && (
        <section>
          <div className="eyebrow mb-1.5">gradient ({objective === 'loss' ? 'loss' : 'target logit'}) reaching x₁</div>
          <dl className="grid gap-1 font-mono text-[12px] font-num">{(['rnn', 'lstm', 'attn'] as Arch[]).map((x) => <div key={x} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{ARCH_LABEL[x]}</dt><dd>{sci(vnorm(runs[x].grads.dx[0]))}</dd></div>)}</dl>
        </section>
      )}
      {depth === 'advanced' && (
        <section>
          <div className="eyebrow mb-1.5">cost for T = {T}, d = {hiddenOf(runs.rnn.params)}</div>
          <dl className="grid gap-1 font-mono text-[12px] font-num">{(['rnn', 'lstm', 'attn'] as Arch[]).map((x) => { const k = cost(x, T, hiddenOf(runs[x].params)); return <div key={x} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{ARCH_LABEL[x]}</dt><dd>{k.macs.toLocaleString()} MACs · {k.sequentialSteps} rounds</dd></div> })}</dl>
        </section>
      )}
    </div>
  )
}
