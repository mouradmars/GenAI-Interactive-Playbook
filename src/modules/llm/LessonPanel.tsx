/** Teaching scaffold: objectives, guided walkthrough, worked examples, exercises, misconceptions, assessment. */
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { showToken } from '../../engine/model'
import { entropy } from '../../engine/decoding'
import { softmax, topk } from '../../engine/tensor'
import { MISCONCEPTIONS, OBJECTIVES, PRESETS, QUIZ } from './lesson'
import { lastRow, useLab } from './context'
import type { JourneyStep } from './journey'

export function LessonPanel({ steps, goTo }: { steps: JourneyStep[]; goTo: (i: number, prompt?: string) => void }) {
  const groups = useMemo(() => {
    const g: { name: string; items: { s: JourneyStep; i: number }[] }[] = []
    steps.forEach((s, i) => {
      const last = g[g.length - 1]
      if (last?.name === s.group) last.items.push({ s, i })
      else g.push({ name: s.group, items: [{ s, i }] })
    })
    return g
  }, [steps])
  return (
    <div className="grid gap-3 xl:grid-cols-2">
      <Card title="learning objectives">
        <ol className="grid list-decimal gap-1.5 pl-5 text-[13.5px] leading-relaxed">{OBJECTIVES.map((o) => <li key={o}>{o}</li>)}</ol>
      </Card>
      <Card title="guided walkthrough · jump to any stage">
        <div className="grid gap-2">
          {groups.map((g) => (
            <div key={g.name} className="flex flex-wrap items-baseline gap-1.5">
              <span className="w-[74px] shrink-0 font-mono text-[11px] text-faint">{g.name}</span>
              {g.items.map(({ s, i }) => <button key={s.id} type="button" className="btn h-7 px-2 text-[12px]" onClick={() => goTo(i)}>{s.short}</button>)}
            </div>
          ))}
        </div>
      </Card>
      <Card title="worked examples" className="xl:col-span-2">
        <div className="grid gap-3 md:grid-cols-2">
          {PRESETS.map((p, k) => (
            <div key={p.prompt} className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="eyebrow">example {k + 1}</div>
              <h3 className="mt-1 font-display text-[17px]">{p.title}</h3>
              <p className="mt-1 font-mono text-[13px] text-cyan">“{p.prompt}”</p>
              <p className="mt-2 text-[13px] leading-relaxed text-muted">{p.why}</p>
              <div className="mt-2 flex flex-wrap gap-2">
                <button type="button" className="btn" onClick={() => goTo(1, p.prompt)}>Open at tokenization</button>
                <button type="button" className="btn" onClick={() => goTo(steps.findIndex((s) => s.kind === 'softmax' && s.layer === 1), p.prompt)}>Open at block 2 attention</button>
                <button type="button" className="btn" onClick={() => goTo(steps.findIndex((s) => s.kind === 'probs'), p.prompt)}>Open at output</button>
              </div>
            </div>
          ))}
        </div>
      </Card>
      <Exercises />
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
      <Quiz />
    </div>
  )
}

function Exercises() {
  const { model } = useLab()
  // 1. uncertainty hunt
  const [p1, setP1] = useState('the cat sat in the')
  const h1 = useMemo(() => {
    const ids = model.encodePrompt(p1).ids
    return entropy(softmax(lastRow(model.forward(ids))))
  }, [model, p1])
  // 2. head hunt
  const dog = useMemo(() => {
    const tr = model.forward(model.encodePrompt('the dog likes to eat').ids)
    const n = tr.ids.length
    const dogPos = tr.tokens.indexOf('▁dog')
    const w = tr.layers[1].heads.map((h) => h.probs.data[(n - 1) * n + dogPos])
    return { w, best: topk(w, 1)[0] }
  }, [model])
  const [g2, setG2] = useState<number | null>(null)
  // 3. temperature
  const [T, setT] = useState(1)
  const top3 = useMemo(() => {
    const z = lastRow(model.forward(model.encodePrompt('sara went to the').ids))
    const p = softmax(z, T)
    const k = topk(p, 1)[0]
    return { p: p[k], tok: model.vocab[k] }
  }, [model, T])
  return (
    <Card title="interactive exercises" className="xl:col-span-2">
      <div className="grid gap-3 lg:grid-cols-3">
        <div className="rounded-lg border border-line bg-panel-2 p-3">
          <h3 className="font-display text-[16px]">1 · Make the model unsure</h3>
          <p className="mt-1 text-[13px] text-muted">Write a prompt whose next-token entropy is at least 2 bits. Hint: where did the corpus choose at random?</p>
          <label className="sr-only" htmlFor="ex1">prompt</label>
          <input id="ex1" className="mt-2 h-9 w-full px-2 font-mono" value={p1} onChange={(e) => setP1(e.target.value)} />
          <p className={`mt-2 font-mono text-[13px] font-num ${h1 >= 2 ? 'text-ok' : 'text-amber'}`} role="status">H = {h1.toFixed(2)} bits {h1 >= 2 ? '✓ solved' : '· keep trying'}</p>
        </div>
        <div className="rounded-lg border border-line bg-panel-2 p-3">
          <h3 className="font-display text-[16px]">2 · Find the "dog" head</h3>
          <p className="mt-1 text-[13px] text-muted">For “the dog likes to eat”, which block-2 head sends the most of the last token's attention to “dog”?</p>
          <div className="mt-2 flex gap-2">{[0, 1, 2, 3].map((h) => <button key={h} type="button" className="btn" aria-pressed={g2 === h} onClick={() => setG2(h)}>head {h + 1}</button>)}</div>
          {g2 !== null && (
            <p className={`mt-2 text-[13px] ${g2 === dog.best ? 'text-ok' : 'text-rose'}`} role="status">
              {g2 === dog.best ? 'Correct. ' : 'Not quite. '}Weights on “dog”: {dog.w.map((w, h) => `head ${h + 1}: ${(w * 100).toFixed(0)}%`).join(', ')}.
            </p>
          )}
        </div>
        <div className="rounded-lg border border-line bg-panel-2 p-3">
          <h3 className="font-display text-[16px]">3 · Tune the temperature</h3>
          <p className="mt-1 text-[13px] text-muted">For “sara went to the”, find a temperature that gives the top token at least 60% probability.</p>
          <label className="mt-2 grid gap-1 font-mono text-[12px] text-muted font-num" htmlFor="ex3">T = {T.toFixed(2)}
            <input id="ex3" type="range" min={0.05} max={2} step={0.01} value={T} onChange={(e) => setT(+e.target.value)} />
          </label>
          <p className={`mt-2 font-mono text-[13px] font-num ${top3.p >= 0.6 ? 'text-ok' : 'text-amber'}`} role="status">
            top: {showToken(top3.tok)} {(top3.p * 100).toFixed(1)}% {top3.p >= 0.6 ? '✓ solved: low T exaggerates small logit gaps' : ''}
          </p>
        </div>
      </div>
    </Card>
  )
}

function Quiz() {
  const [ans, setAns] = useState<(number | null)[]>(QUIZ.map(() => null))
  const score = ans.filter((a, i) => a === QUIZ[i].answer).length
  return (
    <Card title={`assessment · ${score} / ${QUIZ.length} correct`}>
      <div className="grid gap-4">
        {QUIZ.map((q, i) => (
          <fieldset key={i} className="grid gap-1.5">
            <legend className="mb-1 text-[13.5px]">{i + 1}. {q.q}</legend>
            {q.options.map((o, k) => (
              <label key={k} className={`flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1 text-[13px] ${ans[i] === k ? (k === q.answer ? 'border-ok/70' : 'border-rose/70') : 'border-line'}`}>
                <input type="radio" name={`quiz-${i}`} checked={ans[i] === k} onChange={() => setAns(ans.map((a, j) => (j === i ? k : a)))} />
                {o}
              </label>
            ))}
            {ans[i] !== null && <p className={`text-[12.5px] ${ans[i] === q.answer ? 'text-ok' : 'text-rose'}`}>{ans[i] === q.answer ? 'Correct. ' : 'Not quite. '}{q.explain}</p>}
          </fieldset>
        ))}
      </div>
    </Card>
  )
}
