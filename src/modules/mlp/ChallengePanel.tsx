/** Phase 5 · student challenge mode: seeded, hand-computable questions checked against the engine. */
import { CheckCircle2, Dices, Eye, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { isCorrect, makeChallenges, type Question } from './challenges'
import { useMlp } from './store'

export function ChallengePanel() {
  const challenge = useMlp((s) => s.challenge), setSeed = useMlp((s) => s.setChallengeSeed), answer = useMlp((s) => s.answer)
  const qs = useMemo(() => makeChallenges(challenge.seed), [challenge.seed])
  const [reveal, setReveal] = useState(false)
  const done = qs.filter((q) => challenge.answers[q.id]).length, right = qs.filter((q) => challenge.answers[q.id]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title="student challenge">
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <p className="flex-1 text-muted">Work these out by hand (a calculator is fine), then check. Answers are computed by the same engine as the lab; numeric answers are accepted within 1 % (or ±0.005). Share a seed so a class gets the same set.</p>
          <label className="flex items-center gap-1 text-muted">seed<input type="number" min={0} value={challenge.seed} onChange={(e) => { setSeed(+e.target.value || 0); setReveal(false) }} className="w-16 rounded border border-line bg-panel-2 px-1 font-mono" aria-label="Challenge seed" /></label>
          <button type="button" className="btn" onClick={() => { setSeed(challenge.seed + 1); setReveal(false) }}><Dices size={14} />New set</button>
          <button type="button" className="btn" aria-pressed={reveal} onClick={() => setReveal(!reveal)}><Eye size={14} />{reveal ? 'Hide' : 'Reveal'} solutions</button>
          <span className="font-mono text-[12.5px]" data-testid="challenge-score">{right} / {qs.length} correct · {done} answered</span>
        </div>
      </Card>
      <ol className="grid gap-2">
        {qs.map((q, i) => <QuestionCard key={`${challenge.seed}-${q.id}`} n={i + 1} q={q} reveal={reveal} result={challenge.answers[q.id]} onCheck={(given) => answer(q.id, given, isCorrect(q, given))} />)}
      </ol>
    </div>
  )
}

function QuestionCard({ n, q, reveal, result, onCheck }: { n: number; q: Question; reveal: boolean; result?: { given: string; ok: boolean }; onCheck: (g: string) => void }) {
  const [given, setGiven] = useState(result?.given ?? '')
  return (
    <li className="grid gap-2 rounded-xl border border-line bg-panel/70 p-3 text-[13px]" data-testid={`q-${q.id}`}>
      <div className="flex gap-2"><span className="font-mono text-cyan">{n}.</span><div className="grid gap-1"><b className="font-medium">{q.prompt}</b>{q.context && <span className="text-[12px] text-muted">{q.context}</span>}</div></div>
      <div className="flex flex-wrap items-center gap-2 pl-5">
        {q.options ? q.options.map((o) => (
          <button key={o} type="button" className="btn" aria-pressed={given === o} onClick={() => { setGiven(o); onCheck(o) }}>{o}</button>
        )) : (
          <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); if (given.trim()) onCheck(given) }}>
            <input value={given} onChange={(e) => setGiven(e.target.value)} placeholder="your answer" className="w-32 rounded-md border border-line bg-panel-2 px-2 py-1 font-mono" aria-label={`Answer to question ${n}`} />
            <button type="submit" className="btn">Check</button>
          </form>
        )}
        {result && (result.ok ? <span className="flex items-center gap-1 text-teal"><CheckCircle2 size={15} />correct</span> : <span className="flex items-center gap-1 text-rose"><XCircle size={15} />not quite</span>)}
      </div>
      {(reveal || result) && (
        <div className="ml-5 rounded-md border-l-2 border-cyan bg-cyan/5 px-2 py-1 text-[12.5px] text-muted">
          {(reveal || result?.ok) && <>Answer: <span className="font-mono text-fg">{typeof q.answer === 'number' ? q.answer.toFixed(4).replace(/\.?0+$/, '') || '0' : q.answer}</span> · </>}{reveal || result?.ok ? q.explain : 'Try again, or reveal the solutions.'}
        </div>
      )}
    </li>
  )
}
