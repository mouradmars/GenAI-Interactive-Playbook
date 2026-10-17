/** Explore, compare, challenge and lesson panels for module 11. */
import { ArrowRight, CheckCircle2, Dices, Eye, RotateCcw, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { Card } from '../../components/viz'
import { corpusById } from '../../data/rag-corpora'
import { ConfigEditor, METHOD_LABEL, RERANK_LABEL, f3, pct } from './common'
import { useEvaluation, useLab } from './derive'
import { GenerateView, RetrieveView, focusChunk } from './views'
import { check, makeChallenges, type Challenge } from './challenges'
import { EXAMPLES, OBJECTIVES, STAGES, STUDENT_QUESTIONS, type StageId } from './lesson'
import { ragStore, useRag } from './store'

/* ------------------------------------------------------------------ explore */
export function ExplorePanel() {
  const cfg = useRag((s) => s.cfg), setCfg = useRag((s) => s.setCfg), resetCfg = useRag((s) => s.resetCfg)
  const lab = useLab(), { summary } = useEvaluation()
  return (
    <div className="grid gap-3 @3xl:grid-cols-[260px_minmax(0,1fr)]">
      <Card title="all parameters" right={<button type="button" className="btn !py-0.5 text-[12px]" onClick={resetCfg}><RotateCcw size={12} />defaults</button>}>
        <ConfigEditor cfg={cfg} set={setCfg} compact />
        <div className="mt-2 grid gap-1 border-t border-line pt-2 text-[12px] text-muted">
          <label className="flex items-center gap-2">budget <span className="w-10 font-mono text-fg">{cfg.prompt.budget}</span><input type="range" min={40} max={600} step={10} value={cfg.prompt.budget} onChange={(e) => setCfg((c) => ({ ...c, prompt: { ...c.prompt, budget: +e.target.value } }))} className="flex-1" aria-label="Explore budget" /></label>
          <label className="flex items-center gap-2">abstain below <span className="w-10 font-mono text-fg">{cfg.threshold.toFixed(2)}</span><input type="range" min={0} max={1} step={0.05} value={cfg.threshold} onChange={(e) => setCfg((c) => ({ ...c, threshold: +e.target.value }))} className="flex-1" aria-label="Explore threshold" /></label>
        </div>
        <div className="mt-2 grid grid-cols-3 gap-1 border-t border-line pt-2 text-center" data-testid="explore-metrics">
          <div><div className="eyebrow">R@k</div><div className="font-mono">{pct(summary.recallAtK)}</div></div>
          <div><div className="eyebrow">MRR</div><div className="font-mono">{f3(summary.mrr)}</div></div>
          <div><div className="eyebrow">correct</div><div className="font-mono">{pct(summary.answerAccuracy)}</div></div>
        </div>
      </Card>
      <div className="grid min-w-0 content-start gap-3">
        <RetrieveView lab={lab} compact />
        <GenerateView lab={lab} compact />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ compare */
export function ComparePanel() {
  const c = useColors()
  const cfgA = useRag((s) => s.cfg), cfgB = useRag((s) => s.cfgB), setA = useRag((s) => s.setCfg), setB = useRag((s) => s.setCfgB)
  const A = useLab('A'), B = useLab('B'), eA = useEvaluation('A'), eB = useEvaluation('B')
  const side = (name: string, cfg: typeof cfgA, set: typeof setA, lab: typeof A, ev: typeof eA, col: string) => (
    <Card title={<span style={{ color: col }}>configuration {name}</span>}>
      <ConfigEditor cfg={cfg} set={set} idp={name.toLowerCase()} compact />
      <div className="mt-2 font-mono text-[11.5px] text-muted">{lab.pipe.chunks.length} chunks · {METHOD_LABEL[cfg.retrieve.method]} · k = {cfg.retrieve.k} · rerank {RERANK_LABEL[cfg.reranker]}</div>
      <div className="mt-2 grid grid-cols-3 gap-1 text-center" data-testid={`cmp-metrics-${name}`}>
        <div><div className="eyebrow">R@k</div><div className="font-display text-[22px]">{pct(ev.summary.recallAtK)}</div></div>
        <div><div className="eyebrow">MRR</div><div className="font-display text-[22px]">{f3(ev.summary.mrr)}</div></div>
        <div><div className="eyebrow">correct</div><div className="font-display text-[22px]">{pct(ev.summary.answerAccuracy)}</div></div>
      </div>
      <div className="mt-2 text-[12px]"><div className="eyebrow mb-1">top-k for the current question</div>{lab.run.reranked.map((r) => <div key={r.hit.i} className="truncate font-mono text-[11.5px]">#{r.hit.i} {lab.pipe.chunks[r.hit.i].docTitle}: <span className="text-muted">{lab.pipe.chunks[r.hit.i].text.slice(0, 70)}…</span></div>)}</div>
      <div className="mt-2 rounded-md border-l-2 px-2 py-1 text-[12.5px]" style={{ borderColor: col }} data-testid={`cmp-answer-${name}`}>{lab.run.answer.text}</div>
    </Card>
  )
  return (
    <div className="grid gap-3">
      <div className="grid gap-3 @3xl:grid-cols-2">{side('A', cfgA, setA, A, eA, c.cyan)}{side('B', cfgB, setB, B, eB, c.amber)}</div>
      <Card title="question by question">
        <table className="w-full text-[12px]" data-testid="cmp-table">
          <thead className="text-left text-faint"><tr><th className="px-1">question</th><th className="px-1">A: gold rank</th><th className="px-1">A: answer</th><th className="px-1">B: gold rank</th><th className="px-1">B: answer</th></tr></thead>
          <tbody>{eA.ev.map((a, n) => { const b = eB.ev[n], q = eA.questions[n]; return (
            <tr key={a.qid} className="border-t border-line"><td className="px-1">{q.q}</td>
              <td className="px-1 font-mono">{q.gold ? (Number.isFinite(a.goldRank) ? a.goldRank : '—') : 'n/a'}{a.split ? ' ✂' : ''}</td><td className="px-1">{a.answerCorrect ? <CheckCircle2 size={14} className="text-teal" /> : <XCircle size={14} className="text-rose" />}</td>
              <td className="px-1 font-mono">{q.gold ? (Number.isFinite(b.goldRank) ? b.goldRank : '—') : 'n/a'}{b.split ? ' ✂' : ''}</td><td className="px-1">{b.answerCorrect ? <CheckCircle2 size={14} className="text-teal" /> : <XCircle size={14} className="text-rose" />}</td>
            </tr>) })}</tbody>
        </table>
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ challenge */
export function ChallengePanel() {
  const lab = useLab(), sel = useRag((s) => s.selChunk), ch = useRag((s) => s.challenge), setSeed = useRag((s) => s.setChallengeSeed)
  const live = focusChunk(lab, sel)
  const qs = useMemo(() => makeChallenges(ch.seed, { pipe: lab.pipe, q: lab.run.q, chunk: live }), [ch.seed, lab.pipe, lab.run.q, live])
  const right = qs.filter((q) => ch.answers[q.id]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title="student challenge">
        <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
          <p className="flex-1 text-muted">Compute by hand, then check (numbers accepted within 1 % or ±0.005). Same seed → same set; share it with the class.</p>
          <label className="flex items-center gap-1 text-muted">seed<input type="number" min={0} value={ch.seed} onChange={(e) => setSeed(+e.target.value || 0)} className="w-16 rounded border border-line bg-panel-2 px-1 font-mono" aria-label="Challenge seed" /></label>
          <button type="button" className="btn" onClick={() => setSeed(ch.seed + 1)}><Dices size={14} />New set</button>
          <span className="font-mono" data-testid="rag-score">{right} / {qs.length} correct</span>
        </div>
      </Card>
      <ol className="grid gap-2">{qs.map((q, n) => <QCard key={`${ch.seed}-${q.id}`} q={q} n={n + 1} />)}</ol>
    </div>
  )
}
function QCard({ q, n }: { q: Challenge; n: number }) {
  const res = useRag((s) => s.challenge.answers[q.id]), answer = useRag((s) => s.answer)
  const [given, setGiven] = useState(res?.given ?? ''), [show, setShow] = useState(false)
  const go = (g: string) => answer(q.id, g, check(q, g))
  return (
    <li className="grid gap-2 rounded-xl border border-line bg-panel/70 p-3 text-[13px]" data-testid={`rq-${q.id}`}>
      <div><span className="mr-2 font-mono text-cyan">{n}.</span><b>{q.title}</b></div>
      <p>{q.prompt}</p>
      <div className="flex flex-wrap items-center gap-2">
        {q.options ? <div className="grid w-full gap-1">{q.options.map((o) => <button key={o} type="button" className="btn justify-start text-left" aria-pressed={given === o} onClick={() => { setGiven(o); go(o) }}>{o}</button>)}</div>
          : <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); go(given) }}><input value={given} onChange={(e) => setGiven(e.target.value)} placeholder="your answer" className="w-36 rounded-md border border-line bg-panel-2 px-2 py-1 font-mono" aria-label={`Answer ${n}`} /><button type="submit" className="btn">Check</button></form>}
        {res && (res.ok ? <span className="flex items-center gap-1 text-teal"><CheckCircle2 size={15} />correct</span> : <span className="flex items-center gap-1 text-rose"><XCircle size={15} />not yet</span>)}
        <button type="button" className="btn" aria-pressed={show} onClick={() => setShow(!show)}><Eye size={13} />{show ? 'Hide' : 'Show'} solution</button>
      </div>
      {(show || res?.ok) && (
        <div className="rounded-md border-l-2 border-cyan bg-cyan/5 px-3 py-2 text-[12.5px]">
          <div><span className="text-faint">answer:</span> <span className="font-mono">{typeof q.answer === 'number' ? (+q.answer.toFixed(5)).toString() : q.answer}</span></div>
          <ol className="list-decimal pl-5 font-mono text-[12px] text-muted">{q.solution.map((s, k) => <li key={k}>{s}</li>)}</ol>
          <div className="text-amber">Common misconception: {q.misconception}</div>
        </div>
      )}
    </li>
  )
}

/* ------------------------------------------------------------------ lesson */
export function openExample(id: string, seek: (stage: StageId) => void) {
  const ex = EXAMPLES.find((e) => e.id === id)!
  const s = ragStore.getState()
  s.setCorpus(ex.corpus); s.setQuestion(ex.qid); s.resetCfg()
  if (ex.cfg === 'small-chunks') s.setCfg((c) => ({ ...c, chunk: { size: 12, overlap: 0, strategy: 'fixed' } }))
  if (ex.cfg === 'rewrite') s.setCfg((c) => ({ ...c, rewrite: true }))
  if (ex.cfg === 'mmr') s.setCfg((c) => ({ ...c, chunk: { size: 25, overlap: 10, strategy: 'fixed' }, reranker: 'mmr', lambda: 0.5 }))
  s.setTab('guided'); seek(ex.stage)
}
export function LessonPanel({ seek }: { seek: (stage: StageId) => void }) {
  return (
    <div className="grid gap-3">
      <Card title="learning objectives"><ol className="list-decimal pl-5 text-[13px] leading-relaxed">{OBJECTIVES.map((o) => <li key={o}>{o}</li>)}</ol></Card>
      <Card title="worked examples · each sets up the lab and opens the stage it is about">
        <div className="grid gap-2 @3xl:grid-cols-2">
          {EXAMPLES.map((e) => (
            <div key={e.id} className="grid gap-1 rounded-lg border border-line bg-panel-2/40 p-2.5">
              <b className="text-[13.5px]">{e.title}</b>
              <span className="font-mono text-[11px] text-faint">{corpusById(e.corpus).title} · stage “{STAGES.find((s) => s.id === e.stage)!.title}”</span>
              <p className="text-[12.5px] text-muted">{e.what}</p>
              <button type="button" className="btn justify-self-start" onClick={() => openExample(e.id, seek)} data-testid={`example-${e.id}`}>Open example<ArrowRight size={13} /></button>
            </div>
          ))}
        </div>
      </Card>
      <Card title="guided walkthrough (12 stages)">
        <ol className="grid list-decimal gap-1 pl-5 text-[12.5px]">{STAGES.map((s) => <li key={s.id}><button type="button" className="text-left text-cyan underline decoration-dotted" onClick={() => { ragStore.getState().setTab('guided'); seek(s.id) }}>{s.title}</button> <span className="text-muted">— {s.beginner}</span></li>)}</ol>
      </Card>
      <Card title="questions for students"><ol className="list-decimal pl-5 text-[13px] leading-relaxed">{STUDENT_QUESTIONS.map((q) => <li key={q}>{q}</li>)}</ol></Card>
    </div>
  )
}
