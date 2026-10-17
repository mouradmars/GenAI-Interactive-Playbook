/** Explore, A/B compare, challenge and lesson modes, plus the inspector, for module 08. */
import { useMemo, useState } from 'react'
import { Card, Tex } from '../../components/viz'
import { useApp } from '../../sim/appState'
import {
  EOS, METHOD_LABEL, distSum, generate, isCorrect, key, makeChallenges, makeScorer, parseAnswer, scoreSentences, type ModelConfig,
} from '../../engine/ngram'
import { NGRAM_EXAMPLES } from './examples'
import { MISCONCEPTIONS, OBJECTIVES, QUIZ, STAGES } from './lesson'
import { ComparisonTable, ConfigPanel, DistExplorer, ScoreTable, SummaryTiles, LoadExample } from './views'
import { fmtP, fmtPP, useNG } from './context'

/* ------------------------------------------------------------ explore */
export function ExplorePanel() {
  const { cfg, setCfg, scorer, seed, setSeed } = useNG()
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <div className="grid gap-3 @5xl:grid-cols-[minmax(0,320px)_1fr]">
        <Card title="model settings"><ConfigPanel cfg={cfg} setCfg={setCfg} idp="ex-" /></Card>
        <DistExplorer scorer={scorer} />
      </div>
      <ScoreTable />
      <Generator scorer={scorer} seed={seed} setSeed={setSeed} />
    </div>
  )
}

function Generator({ scorer, seed, setSeed, count = 4, idp = '' }: { scorer: ReturnType<typeof makeScorer>; seed: number; setSeed?: (n: number) => void; count?: number; idp?: string }) {
  const samples = useMemo(() => Array.from({ length: count }, (_, i) => generate(scorer, seed + i, 18)), [scorer, seed, count])
  return (
    <Card title={`sampled sentences · seeds ${seed}–${seed + count - 1} · deterministic`}
      right={setSeed && (
        <div className="flex items-center gap-2">
          <label htmlFor={`${idp}seed`} className="font-mono text-[11px] text-faint">seed</label>
          <input id={`${idp}seed`} type="number" className="h-7 w-20 px-2 font-mono text-[12px]" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} />
          <button type="button" className="btn h-7" onClick={() => setSeed(seed + count)}>Next samples</button>
        </div>
      )}>
      {scorer.cfg.method === 'mle' && <p className="mb-2 text-[12px] text-muted">Maximum likelihood can only reproduce n-grams it has seen, so it often regenerates training sentences.</p>}
      <div className="grid gap-2">
        {samples.map((s, i) => (
          <p key={i} className="flex flex-wrap gap-1 font-mono text-[13px] leading-relaxed">
            {s.length === 0 && <span className="text-rose">nothing to sample (the start context has no counts)</span>}
            {s.map((t, k) => (
              <span key={k} title={`p = ${t.p.toFixed(4)}`} className={`rounded px-1 ${t.w === EOS ? 'text-teal' : t.w === '<unk>' ? 'text-rose' : ''}`}
                style={{ background: `color-mix(in oklab, var(--cyan) ${Math.round(t.p * 40)}%, transparent)` }}>{t.w}</span>
            ))}
          </p>
        ))}
      </div>
      <p className="mt-2 font-mono text-[10.5px] text-faint">Shading = probability of the sampled token. Hover for the exact value.</p>
    </Card>
  )
}

/* -------------------------------------------------------- A/B compare */
export function ABPanel() {
  const { model, test, heldOut, cfg } = useNG()
  const [a, setA] = useState<ModelConfig>({ ...cfg, order: 2, method: 'laplace', k: 1 })
  const [b, setB] = useState<ModelConfig>({ ...cfg, order: 3, method: 'kn' })
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <p className="max-w-[75ch] text-[13.5px] text-muted">Two models fitted to the same corpus. Change either side; perplexity, per-token probabilities and samples are recomputed.</p>
      <div className="grid gap-3 @3xl:grid-cols-2">
        {[{ l: 'A', c: a, s: setA }, { l: 'B', c: b, s: setB }].map(({ l, c, s }) => <ABSide key={l} label={l} cfg={c} setCfg={(p) => s((x) => ({ ...x, ...p }))} model={model} test={test} heldOut={heldOut} />)}
      </div>
      <ComparisonTable />
    </div>
  )
}

function ABSide({ label, cfg, setCfg, model, test, heldOut }: { label: string; cfg: ModelConfig; setCfg: (p: Partial<ModelConfig>) => void; model: ReturnType<typeof useNG>['model']; test: string; heldOut: string }) {
  const s = useMemo(() => makeScorer(model, cfg), [model, cfg])
  const t = useMemo(() => scoreSentences(s, test), [s, test])
  const h = useMemo(() => scoreSentences(s, heldOut), [s, heldOut])
  return (
    <Card title={`model ${label} · ${METHOD_LABEL[cfg.method]} · ${['unigram', 'bigram', 'trigram'][cfg.order - 1]}`}>
      <ConfigPanel cfg={cfg} setCfg={setCfg} idp={`ab${label}-`} />
      <div className="mt-3"><SummaryTiles s={t} /></div>
      <p className="mt-2 font-mono text-[12px] text-muted font-num">held-out perplexity <span className="text-cyan">{fmtPP(h.perplexity)}</span></p>
      <div className="mt-2 flex flex-wrap gap-1">
        {t.tokens.map((x) => <span key={x.i} className={`rounded border border-line px-1.5 py-0.5 font-mono text-[11px] font-num ${x.p === 0 ? 'text-rose' : 'text-muted'}`}>{x.token} {fmtP(x.p)}</span>)}
      </div>
      <div className="mt-3"><Generator scorer={s} seed={7} count={2} /></div>
    </Card>
  )
}

/* ---------------------------------------------------------- challenge */
export function ChallengePanel({ results, setResults }: { results: Record<string, { given: string; ok: boolean; revealed: boolean }>; setResults: (r: Record<string, { given: string; ok: boolean; revealed: boolean }>) => void }) {
  const { model, seed, setSeed } = useNG()
  const qs = useMemo(() => makeChallenges(model, seed), [model, seed])
  const [draft, setDraft] = useState<Record<string, string>>({})
  const score = qs.filter((q) => results[`${seed}-${q.id}`]?.ok).length
  if (!qs.length) return <Card title="student challenge"><p className="text-muted">The corpus needs at least one sentence.</p></Card>
  return (
    <div className="grid gap-3">
      <Card title={`student challenge · ${score} / ${qs.length} correct · question set ${seed}`}
        right={<div className="flex items-center gap-2">
          <label htmlFor="ch-seed" className="font-mono text-[11px] text-faint">seed</label>
          <input id="ch-seed" type="number" className="h-7 w-20 px-2 font-mono text-[12px]" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} />
          <button type="button" className="btn h-7" onClick={() => setSeed(seed + 1)}>New question set</button>
        </div>}>
        <p className="max-w-[75ch] text-[13px] text-muted">Questions are generated from the current corpus with a seeded generator, so the whole class gets the same set for the same seed. Compute by hand, then check. Fractions like 3/7 are accepted; answers are checked to 0.5%.</p>
      </Card>
      {qs.map((q, i) => {
        const k = `${seed}-${q.id}`
        const r = results[k]
        return (
          <Card key={k} title={`question ${i + 1} · ${q.kind}`}>
            <p className="text-[14.5px]">{q.prompt}</p>
            <form className="mt-2 flex flex-wrap items-center gap-2" onSubmit={(e) => {
              e.preventDefault()
              const given = draft[k] ?? ''
              setResults({ ...results, [k]: { given, ok: isCorrect(q.answer, parseAnswer(given)), revealed: r?.revealed ?? false } })
            }}>
              <label htmlFor={`ans-${q.id}`} className="sr-only">Answer to question {i + 1}</label>
              <input id={`ans-${q.id}`} className="h-9 w-40 px-2 font-mono" value={draft[k] ?? ''} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} placeholder="your answer" autoComplete="off" />
              <button type="submit" className="btn btn-primary h-9">Check</button>
              <button type="button" className="btn h-9" onClick={() => setResults({ ...results, [k]: { given: r?.given ?? '', ok: r?.ok ?? false, revealed: true } })}>Reveal answer</button>
              {r && r.given !== '' && <span role="status" className={`text-[13px] ${r.ok ? 'text-ok' : 'text-rose'}`}>{r.ok ? 'Correct' : `Not yet: ${r.given} ≠ expected value`}</span>}
            </form>
            {(r?.revealed || r?.ok) && <p className="mt-2 rounded-lg border border-line bg-panel-2 p-2 font-mono text-[12.5px] leading-relaxed text-muted font-num">Answer: <span className="text-amber">{Number.isInteger(q.answer) ? q.answer : q.answer.toFixed(5)}</span>. {q.explain}</p>}
          </Card>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------- lesson */
export function LessonPanel({ goTo }: { goTo: (stageId: string, exampleId?: string) => void }) {
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
          {NGRAM_EXAMPLES.map((ex, k) => (
            <div key={ex.id} className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="eyebrow">example {k + 1}</div>
              <h3 className="mt-1 font-display text-[17px]">{ex.title}</h3>
              <p className="mt-1 font-mono text-[12.5px] text-cyan">test: “{ex.test}”</p>
              <p className="mt-2 text-[13px] leading-relaxed text-muted">{ex.why}</p>
              <div className="mt-2 flex flex-wrap gap-2">
                <button type="button" className="btn" onClick={() => goTo(ex.stage, ex.id)}>Open at {STAGES.find((s) => s.id === ex.stage)!.short}</button>
                <button type="button" className="btn" onClick={() => goTo('corpus', ex.id)}>Start from the corpus</button>
              </div>
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
                  <input type="radio" name={`ngq-${i}`} checked={ans[i] === k} onChange={() => setAns(ans.map((a, j) => (j === i ? k : a)))} />{o}
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
        <div className="mt-3 flex flex-wrap gap-2">{NGRAM_EXAMPLES.slice(0, 2).map((e) => <LoadExample key={e.id} id={e.id} label={`Load: ${e.title}`} />)}</div>
      </Card>
    </div>
  )
}

/* ---------------------------------------------------------- inspector */
export function NGInspector() {
  const { stage, eff, scorer, score, focus, ctx, model, tab } = useNG()
  const { depth } = useApp()
  const t = score.tokens[Math.min(focus, score.tokens.length - 1)]
  const sum = distSum(scorer, ctx)
  const h = ctx.slice(2 - (eff.order - 1))
  const ch = eff.order === 1 ? model.N : eff.order === 2 ? model.ctx1.get(h[0]) ?? 0 : model.ctx2.get(key(h[0], h[1])) ?? 0
  const show = tab === 'guided' ? stage : { ...STAGES[0], title: METHOD_LABEL[eff.method], group: tab, equation: STAGES.find((s) => s.method === eff.method && s.id !== 'score' && s.id !== 'zeros')?.equation ?? STAGES[3].equation }
  const text = tab === 'guided' ? stage[depth] : STAGES.find((s) => s.method === eff.method && s.id !== 'score' && s.id !== 'zeros')?.[depth] ?? ''
  return (
    <div className="grid gap-4" aria-live="polite">
      <div>
        <div className="eyebrow">{show.group}</div>
        <h2 className="mt-1 font-display text-[22px] leading-tight">{show.title}</h2>
      </div>
      <div className="overflow-x-auto rounded-lg border border-line bg-panel-2 px-3 py-3 scroll-thin"><Tex tex={show.equation} block /></div>
      <p className="max-w-[60ch] text-[13.5px] leading-relaxed text-fg/90">{text}</p>
      <section>
        <div className="eyebrow mb-1.5">model · computed from your corpus</div>
        <dl className="grid gap-1 font-mono text-[12px] font-num">
          {[
            ['method', `${METHOD_LABEL[eff.method]} · n = ${eff.order}`],
            ['N (predicted tokens)', String(model.N)], ['|V|', String(model.V)],
            ['types: uni / bi / tri', `${model.uni.size} / ${model.bi.size} / ${model.tri.size}`],
          ].map(([k, v]) => <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k}</dt><dd className="text-right">{v}</dd></div>)}
        </dl>
      </section>
      <section>
        <div className="eyebrow mb-1.5">explorer context h = {eff.order === 1 ? '∅' : h.join(' ')}</div>
        <dl className="grid gap-1 font-mono text-[12px] font-num">
          <div className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">c(h·)</dt><dd>{ch}</dd></div>
          <div className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">Σ_w P(w|h) over |V|</dt>
            <dd className={Number.isNaN(sum) ? 'text-rose' : Math.abs(sum - 1) < 1e-9 ? 'text-ok' : 'text-rose'}>{Number.isNaN(sum) ? 'undefined (0/0)' : `${sum.toFixed(10)} ✓`}</dd></div>
        </dl>
      </section>
      {t && (
        <section>
          <div className="eyebrow mb-1.5">selected test token {t.i + 1}: P({t.token} | {t.history.join(' ') || '∅'})</div>
          <dl className="grid gap-1 font-mono text-[12px] font-num">
            {t.breakdown.parts.map((p) => <div key={p.label} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="min-w-0 break-words text-muted">{p.label}</dt><dd className="shrink-0">{Number.isInteger(p.value) ? p.value : p.value.toFixed(6)}</dd></div>)}
            <div className="flex justify-between gap-3 py-1"><dt className="text-muted">P</dt><dd className={t.p === 0 ? 'text-rose' : 'text-amber'}>{fmtP(t.p)}</dd></div>
            <div className="flex justify-between gap-3 py-1"><dt className="text-muted">−log₂ P</dt><dd>{Number.isFinite(t.log2p) ? (-t.log2p).toFixed(4) : '∞'} bits</dd></div>
          </dl>
          {depth === 'advanced' && t.breakdown.usedOrder && <p className="mt-1 font-mono text-[11px] text-faint">answered by the order-{t.breakdown.usedOrder} counts</p>}
        </section>
      )}
      <section>
        <div className="eyebrow mb-1.5">test text</div>
        <dl className="grid grid-cols-2 gap-1 font-mono text-[12px] font-num">
          <dt className="text-muted">T</dt><dd className="text-right">{score.tokens.length}</dd>
          <dt className="text-muted">H (bits/token)</dt><dd className="text-right">{Number.isFinite(score.crossEntropy) ? score.crossEntropy.toFixed(4) : '∞'}</dd>
          <dt className="text-muted">PP = 2ᴴ</dt><dd className="text-right text-amber">{fmtPP(score.perplexity)}</dd>
          <dt className="text-muted">tokens with P = 0</dt><dd className={`text-right ${score.zeroAt.length ? 'text-rose' : ''}`}>{score.zeroAt.length}</dd>
        </dl>
      </section>
    </div>
  )
}

