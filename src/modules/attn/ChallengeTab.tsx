/** Student challenge mode: three levels, numerical entry, immediate feedback, hints, worked solutions, retry. */
import { CheckCircle2, Dices, Eye, Lightbulb, RotateCcw, Search, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import type { AttnTrace } from '../../engine/attention'
import { useTokens } from './common'
import { check, makeChallenges, type Challenge } from './challenges'
import { seekRef } from './seek'
import { attnStore, useAttn } from './store'

const LEVEL = { 1: 'Level 1 · beginner', 2: 'Level 2 · intermediate', 3: 'Level 3 · advanced' } as const

export function ChallengeTab({ trace }: { trace: AttnTrace }) {
  const tokens = useTokens(), seed = useAttn((s) => s.challenge.seed), answers = useAttn((s) => s.challenge.answers)
  const st = attnStore.getState()
  const qs = useMemo(() => makeChallenges(seed, trace, tokens), [seed, trace, tokens])
  const solved = qs.filter((q) => answers[q.id]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title="challenges · compute by hand, then check">
        <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
          <p className="flex-1 text-muted">Numeric answers are accepted within 1 % (or ±0.005). Vectors: numbers separated by commas (use -inf for masked entries). Shapes: rows×cols. Progress is kept for this session; “New numbers” draws a fresh, reproducible set (same seed → same numbers). Questions marked “live lab” use the current lab state.</p>
          <label className="flex items-center gap-1 text-muted">seed<input type="number" min={0} value={seed} onChange={(e) => st.setChallengeSeed(+e.target.value || 0)} className="w-16 rounded border border-line px-1 font-mono" aria-label="Challenge seed" /></label>
          <button type="button" className="btn" onClick={() => st.setChallengeSeed(seed + 1)}><Dices size={14} />New numbers</button>
          <span className="font-mono" data-testid="attn-score">{solved} / {qs.length} solved</span>
        </div>
        <div className="mt-2 flex h-2 gap-0.5" aria-hidden="true">{qs.map((q) => <div key={q.id} className={`flex-1 rounded ${answers[q.id]?.ok ? 'bg-teal' : answers[q.id] ? 'bg-rose/60' : 'bg-panel-2'}`} />)}</div>
      </Card>
      {([1, 2, 3] as const).map((lv) => (
        <section key={lv} className="grid gap-2" aria-label={LEVEL[lv]}>
          <h2 className="eyebrow">{LEVEL[lv]} · {qs.filter((q) => q.level === lv && answers[q.id]?.ok).length}/{qs.filter((q) => q.level === lv).length}</h2>
          <ol className="grid gap-2 @5xl:grid-cols-2">{qs.filter((q) => q.level === lv).map((q) => <ChallengeCard key={`${seed}-${q.id}`} q={q} />)}</ol>
        </section>
      ))}
    </div>
  )
}

function ChallengeCard({ q }: { q: Challenge }) {
  const res = useAttn((s) => s.challenge.answers[q.id]), hints = useAttn((s) => s.challenge.hints[q.id] ?? 0)
  const st = attnStore.getState()
  const [given, setGiven] = useState(res?.given ?? '')
  const [show, setShow] = useState(false)
  const submit = (g: string) => st.answer(q.id, g, check(q, g))
  const ans = typeof q.answer === 'number' ? (q.answer === -Infinity ? '−∞' : q.answer.toFixed(4)) : Array.isArray(q.answer) ? (q.kind === 'set' ? (q.answer.length ? q.answer.join(', ') : 'none') : q.answer.map((v) => (v === -Infinity ? '−∞' : v.toFixed(4))).join(', ')) : q.answer
  const placeholder = { num: 'number', vec: 'a, b, …', set: '3, 4 or none', shape: 'rows×cols', mc: '' }[q.kind]
  return (
    <li className="grid content-start gap-2 rounded-xl border border-line bg-panel/70 p-3 text-[13px]" data-testid={`ach-${q.id}`}>
      <div className="flex flex-wrap items-baseline gap-2"><span className="chip text-[11px] text-cyan">{q.id}</span><b>{q.title}</b>{res && <span className="text-[11px] text-faint">{res.tries} attempt{res.tries > 1 ? 's' : ''}</span>}</div>
      <p className="leading-relaxed">{q.prompt}</p>
      {q.options ? (
        <div className="grid gap-1" role="radiogroup" aria-label={`Answer to ${q.id}`}>{q.options.map((o) => <button key={o} type="button" role="radio" aria-checked={given === o} className={`btn !h-auto min-h-8 justify-start py-1 text-left ${given === o ? 'active' : ''}`} onClick={() => { setGiven(o); submit(o) }}>{o}</button>)}</div>
      ) : (
        <form className="flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); submit(given) }}>
          <input value={given} onChange={(e) => setGiven(e.target.value)} placeholder={placeholder} className="w-44 rounded-md border border-line px-2 py-1 font-mono" aria-label={`Answer to ${q.id}`} data-testid={`ans-${q.id}`} />
          <button type="submit" className="btn">Check</button>
        </form>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        {res && (res.ok ? <span className="flex items-center gap-1 text-teal" role="status"><CheckCircle2 size={15} />correct</span> : <span className="flex items-center gap-1 text-rose" role="status"><XCircle size={15} />not yet</span>)}
        {res && !res.ok && <button type="button" className="btn !h-7" onClick={() => { setGiven(''); setShow(false) }}><RotateCcw size={13} />Retry</button>}
        <button type="button" className="btn !h-7" onClick={() => st.hint(q.id)} disabled={hints >= q.hints.length}><Lightbulb size={13} />Hint{q.hints.length > 1 ? ` (${Math.min(hints, q.hints.length)}/${q.hints.length})` : ''}</button>
        <button type="button" className="btn !h-7" aria-pressed={show} onClick={() => setShow(!show)}><Eye size={13} />{show ? 'Hide' : 'Show'} worked solution</button>
        {q.verify && <button type="button" className="btn !h-7" onClick={() => seekRef.current(q.verify!)}><Search size={13} />Verify in the lab</button>}
      </div>
      {hints > 0 && <ul className="grid gap-0.5 text-[12px] text-amber">{q.hints.slice(0, hints).map((x) => <li key={x}><span className="text-faint">hint:</span> {x}</li>)}</ul>}
      {(show || res?.ok) && (
        <div className="grid gap-1 rounded-md border-l-2 border-cyan bg-cyan/5 px-3 py-2 text-[12.5px]">
          <div><span className="text-faint">answer:</span> <span className="font-mono">{ans}</span></div>
          <ol className="list-decimal pl-5 text-muted">{q.solution.map((x, k) => <li key={k} className="font-mono text-[12px]">{x}</li>)}</ol>
          <div className="text-amber">Common misconception: {q.misconception}</div>
        </div>
      )}
    </li>
  )
}
