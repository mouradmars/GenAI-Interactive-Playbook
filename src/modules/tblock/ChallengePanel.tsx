/** Ten progressive challenges with check, step-by-step solution, explanation, misconception and retry. */
import { CheckCircle2, Dices, Eye, RotateCcw, Search, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { checkAnswer, makeChallenges, type Challenge } from './challenges'
import { cfgOf, useTb } from './store'
import { seekRef } from './seek'

export function ChallengePanel() {
  const s = useTb((x) => x)
  const qs = useMemo(() => makeChallenges(s.challenge.seed, { cfg: cfgOf(s), params: s.params, X: s.X, tokens: s.tokens }), [s.challenge.seed, s.params, s.X, s.arch, s.T, s.d, s.H, s.dff, s.act]) // eslint-disable-line react-hooks/exhaustive-deps
  const right = qs.filter((q) => s.challenge.answers[q.id]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title="ten progressive challenges">
        <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
          <p className="flex-1 text-muted">Compute by hand, then check. Every answer comes from the same engine as the microscope; numeric answers are accepted within 1 % (or ±0.005). Vectors: separate numbers with commas. Same seed → same numbers.</p>
          <label className="flex items-center gap-1 text-muted">seed<input type="number" min={0} value={s.challenge.seed} onChange={(e) => s.setChallenge({ seed: +e.target.value || 0 })} className="w-16 rounded border border-line bg-panel-2 px-1 font-mono" aria-label="Challenge seed" /></label>
          <button type="button" className="btn" onClick={() => s.setChallenge({ seed: s.challenge.seed + 1 })}><Dices size={14} />New numbers</button>
          <span className="font-mono" data-testid="tb-score">{right} / {qs.length} solved</span>
        </div>
        <div className="mt-2 flex h-2 gap-0.5">{qs.map((q) => <div key={q.id} className={`flex-1 rounded ${s.challenge.answers[q.id]?.ok ? 'bg-teal' : s.challenge.answers[q.id] ? 'bg-rose/60' : 'bg-panel-2'}`} />)}</div>
      </Card>
      <ol className="grid gap-2">{qs.map((q) => <ChallengeCard key={`${s.challenge.seed}-${q.id}`} q={q} />)}</ol>
    </div>
  )
}

function ChallengeCard({ q }: { q: Challenge }) {
  const res = useTb((x) => x.challenge.answers[q.id]), answer = useTb((x) => x.answer)
  const [given, setGiven] = useState(res?.given ?? '')
  const [show, setShow] = useState(false)
  const check = (g: string) => answer(q.id, g, checkAnswer(q, g))
  const ansText = typeof q.answer === 'number' ? q.answer.toFixed(4) : Array.isArray(q.answer) ? q.answer.map((v) => v.toFixed(4)).join(', ') : q.answer
  return (
    <li className="grid gap-2 rounded-xl border border-line bg-panel/70 p-3 text-[13px]" data-testid={`ch-${q.id}`}>
      <div className="flex flex-wrap items-baseline gap-2"><span className="chip text-cyan">challenge {q.level}</span><b>{q.title}</b></div>
      <p className="leading-relaxed">{q.prompt}</p>
      <div className="flex flex-wrap items-center gap-2">
        {q.options ? (
          <div className="grid w-full gap-1">{q.options.map((o) => <button key={o} type="button" className="btn justify-start text-left" aria-pressed={given === o} onClick={() => { setGiven(o); check(o) }}>{o}</button>)}</div>
        ) : (
          <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); check(given) }}>
            <input value={given} onChange={(e) => setGiven(e.target.value)} placeholder={q.kind === 'vec' ? 'a, b, …' : q.kind === 'set' ? '1, 2, …' : q.kind === 'shape' ? 'rows×cols' : 'number'} className="w-40 rounded-md border border-line bg-panel-2 px-2 py-1 font-mono" aria-label={`Answer to challenge ${q.level}`} />
            <button type="submit" className="btn">Check answer</button>
          </form>
        )}
        {res && (res.ok ? <span className="flex items-center gap-1 text-teal"><CheckCircle2 size={15} />correct</span> : <span className="flex items-center gap-1 text-rose"><XCircle size={15} />not yet</span>)}
        {res && !res.ok && <button type="button" className="btn" onClick={() => { setGiven(''); setShow(false) }}><RotateCcw size={13} />Retry</button>}
        <button type="button" className="btn" aria-pressed={show} onClick={() => setShow(!show)}><Eye size={13} />{show ? 'Hide' : 'Show'} solution</button>
        {q.verify && <button type="button" className="btn" onClick={() => seekRef.current(q.verify!)}><Search size={13} />Verify in the simulator</button>}
      </div>
      {(show || res?.ok) && (
        <div className="grid gap-1 rounded-md border-l-2 border-cyan bg-cyan/5 px-3 py-2 text-[12.5px]">
          <div><span className="text-faint">answer:</span> <span className="font-mono">{ansText}</span></div>
          <ol className="list-decimal pl-5 text-muted">{q.solution.map((x, k) => <li key={k} className="font-mono text-[12px]">{x}</li>)}</ol>
          <div className="text-muted">{q.explanation}</div>
          <div className="text-amber">Common misconception: {q.misconception}</div>
        </div>
      )}
    </li>
  )
}
