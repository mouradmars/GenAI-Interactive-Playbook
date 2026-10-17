/** Explore, compare, challenge and lesson modes, plus the inspector, for module 06. */
import { useMemo, useState } from 'react'
import { Card, Tex } from '../../components/viz'
import { useApp } from '../../sim/appState'
import { cosine, dot, norm } from '../../engine/linalg'
import {
  KIND_LABEL, METRIC_LABEL, makeEmbChallenges, nearest, numCorrect, parseNum, project, trainPPMISpace, type ProjConfig, type Space,
} from '../../engine/embeddings'
import wordCorpus from '../../data/embed-corpus.txt?raw'
import { Scatter2D } from './Scatter'
import { AnalogyBox, Faithfulness, MapPanel, MetricPicker, PPMIControls, ProjControls, SelectedChips, Slider, SourceBadge, SpacePicker, WordSelect } from './views'
import { MISCONCEPTIONS, OBJECTIVES, QUIZ, STAGES, type SpaceId } from './lesson'
import { EMB_EXAMPLES } from './examples'
import { f3, f4, pct, useEmb } from './context'

/* ---------------------------------------------------------------- explore */
export function ExplorePanel() {
  const { spaceId, setSpaceId, space, focusIdx, picks, setPicks, metric, k, setK } = useEmb()
  const nn = useMemo(() => nearest(space, space.vecs[focusIdx], k, metric, [focusIdx]), [space, focusIdx, k, metric])
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <div className="grid gap-3 @5xl:grid-cols-[minmax(0,360px)_1fr]">
        <div className="grid content-start gap-3">
          <Card title="source" right={<SourceBadge space={space} />}>
            <SpacePicker value={spaceId} onChange={setSpaceId} idp="ex-" />
            {spaceId === 'ppmi' && <div className="mt-3"><PPMIControls idp="ex-" /></div>}
          </Card>
          <Card title="projection"><ProjControls idp="ex-" /></Card>
          <Card title="query">
            <div className="grid gap-2">
              <WordSelect id="ex-focus" label="focus word" value={picks.focus} onChange={(focus) => setPicks({ focus })} />
              <MetricPicker />
              <Slider id="ex-k" label={`neighbours k = ${k}`} value={k} min={1} max={15} step={1} onChange={setK} />
            </div>
          </Card>
        </div>
        <MapPanel neighbourK={k} />
      </div>
      <div className="grid gap-3 @5xl:grid-cols-3">
        <Card title={`nearest to “${picks.focus}” · ${METRIC_LABEL[metric]}`}>
          <table className="w-full font-mono text-[12px] font-num"><tbody>{nn.map((x, i) => <tr key={x.i} className="border-b border-line/50"><td className="text-faint">{i + 1}</td><td><button type="button" className="hover:text-cyan" onClick={() => setPicks({ focus: x.word })}>{x.word}</button></td><td className="text-right">{f4(x.score)}</td></tr>)}</tbody></table>
        </Card>
        <AnalogyBox compact />
        <div className="grid content-start gap-3"><Faithfulness /><Card title="selected words"><SelectedChips /></Card></div>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- compare */
interface SideCfg { space: SpaceId; window: number; method: 'pca' | 'umap'; seed: number }
export function ComparePanel() {
  const { spaces, picks, ppmiCfg, proj } = useEmb()
  const [a, setA] = useState<SideCfg>({ space: 'ppmi', window: ppmiCfg.window, method: 'pca', seed: proj.seed })
  const [b, setB] = useState<SideCfg>({ space: 'ppmi', window: ppmiCfg.window, method: 'umap', seed: proj.seed })
  return (
    <div className="grid gap-3 [&>*]:min-w-0">
      <p className="max-w-[78ch] text-[13.5px] text-muted">Two pictures side by side. Same space with PCA versus UMAP shows what each projection keeps; the same projection on two spaces (or two context windows) shows what training changes. The focus word “{picks.focus}” is highlighted on both sides when it exists.</p>
      <div className="grid gap-3 @5xl:grid-cols-2">
        {[{ l: 'A', s: a, set: setA }, { l: 'B', s: b, set: setB }].map(({ l, s, set }) => <Side key={l} label={l} cfg={s} setCfg={(p) => set((x) => ({ ...x, ...p }))} spaces={spaces} />)}
      </div>
    </div>
  )
}

function Side({ label, cfg, setCfg, spaces }: { label: string; cfg: SideCfg; setCfg: (p: Partial<SideCfg>) => void; spaces: Record<SpaceId, Space> }) {
  const { picks, setPicks, ppmiCfg, proj } = useEmb()
  const space = useMemo(() => (cfg.space === 'ppmi' ? (cfg.window === ppmiCfg.window ? spaces.ppmi : trainPPMISpace(wordCorpus, { ...ppmiCfg, window: cfg.window })) : spaces[cfg.space]), [cfg.space, cfg.window, spaces, ppmiCfg])
  const pc: ProjConfig = { ...proj, method: cfg.method, dims: 2, seed: cfg.seed }
  const p = useMemo(() => project(space, pc), [space, cfg.method, cfg.seed, proj.nNeighbors, proj.minDist, proj.nEpochs])
  const f = space.words.indexOf(picks.focus)
  const nn = f >= 0 ? nearest(space, space.vecs[f], 5, 'cosine', [f]) : []
  return (
    <Card title={`picture ${label} · ${space.title} · ${cfg.method.toUpperCase()}`} right={<SourceBadge space={space} />}>
      <div className="grid gap-2">
        <SpacePicker value={cfg.space} onChange={(s) => setCfg({ space: s })} idp={`cmp${label}-`} />
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label={`projection ${label}`}>
            {(['pca', 'umap'] as const).map((m) => <button key={m} type="button" role="radio" aria-checked={cfg.method === m} onClick={() => setCfg({ method: m })} className={`rounded-md px-2.5 py-1 font-mono text-[12px] uppercase ${cfg.method === m ? 'bg-raise text-cyan' : 'text-muted'}`}>{m}</button>)}
          </div>
          {cfg.space === 'ppmi' && <div className="w-40"><Slider id={`cmp${label}-win`} label={`window ±${cfg.window}`} value={cfg.window} min={1} max={8} step={1} onChange={(window) => setCfg({ window })} /></div>}
          {cfg.method === 'umap' && <label htmlFor={`cmp${label}-seed`} className="grid gap-1 text-[12px] text-muted">seed<input id={`cmp${label}-seed`} type="number" className="h-8 w-20 px-2 font-mono" value={cfg.seed} onChange={(e) => setCfg({ seed: +e.target.value || 0 })} /></label>}
        </div>
      </div>
      <div className="mt-2">
        <Scatter2D space={space} coords={p.coords} focus={f} selected={[]} onFocus={(i) => setPicks({ focus: space.words[i] })} neighbourK={f >= 0 ? 5 : 0} height={360} ariaLabel={`embedding map ${label}`} />
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[12px] font-num sm:grid-cols-4">
        <dt className="text-muted">words × dims</dt><dd>{space.words.length} × {space.dim}</dd>
        <dt className="text-muted">neighbourhoods kept</dt><dd>{pct(p.preservation.mean)}</dd>
        <dt className="text-muted">Shepard ρ</dt><dd>{f3(p.shepard.rho)}</dd>
        <dt className="text-muted">variance shown</dt><dd>{p.explained ? pct(p.explained.reduce((x, y) => x + y, 0)) : 'n/a (UMAP)'}</dd>
      </dl>
      <p className="mt-2 font-mono text-[12px] text-muted">{f >= 0 ? <>neighbours of “{picks.focus}”: {nn.map((x) => `${x.word} ${f3(x.score)}`).join(', ')}</> : <>“{picks.focus}” is not in this space.</>}</p>
    </Card>
  )
}

/* -------------------------------------------------------------- challenge */
export type ChallengeResults = Record<string, { given: string; ok: boolean; revealed: boolean }>
export function ChallengePanel({ results, setResults }: { results: ChallengeResults; setResults: (r: ChallengeResults) => void }) {
  const { space, projection, seed, setSeed, proj } = useEmb()
  const qs = useMemo(() => makeEmbChallenges(space, projection, seed), [space, projection, seed])
  const [draft, setDraft] = useState<Record<string, string>>({})
  const keyOf = (id: string) => `${space.id}-${proj.method}-${seed}-${id}`
  const score = qs.filter((q) => results[keyOf(q.id)]?.ok).length
  return (
    <div className="grid gap-3">
      <Card title={`student challenge · ${score} / ${qs.length} correct · question set ${seed}`}
        right={<div className="flex items-center gap-2">
          <label htmlFor="ech-seed" className="font-mono text-[11px] text-faint">seed</label>
          <input id="ech-seed" type="number" className="h-7 w-20 px-2 font-mono text-[12px]" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} />
          <button type="button" className="btn h-7" onClick={() => setSeed(seed + 1)}>New question set</button>
        </div>}>
        <p className="max-w-[78ch] text-[13px] text-muted">Questions are generated by a seeded generator from the current space ({space.title}) and projection ({proj.method.toUpperCase()}), so a class using the same seed and settings gets identical questions. Compute by hand, then check. Numeric answers are accepted within 1% (or ±0.006); fractions like 3/5 work.</p>
      </Card>
      {qs.map((q, n) => {
        const k = keyOf(q.id)
        const r = results[k]
        const check = (given: string) => setResults({ ...results, [k]: { given, ok: q.options ? Number(given) === q.answer : numCorrect(q.answer, parseNum(given), q.kind), revealed: r?.revealed ?? false } })
        return (
          <Card key={k} title={`question ${n + 1} · ${q.kind}`}>
            <p className="text-[14.5px]">{q.prompt}</p>
            {q.vectors && <div className="mt-2 flex flex-wrap gap-2 font-mono text-[13px]">{q.vectors.map((v) => <span key={v.name} className="rounded-md border border-line bg-panel-2 px-2 py-1">{v.name} = [{v.v.map((x) => (Number.isInteger(x) ? x : x.toFixed(2))).join(', ')}]</span>)}</div>}
            {q.options ? (
              <fieldset className="mt-2 flex flex-wrap gap-2">
                <legend className="sr-only">Options for question {n + 1}</legend>
                {q.options.map((o, i) => (
                  <label key={o} className={`flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1 font-mono text-[13px] ${r?.given === String(i) ? (r.ok ? 'border-ok/70' : 'border-rose/70') : 'border-line'}`}>
                    <input type="radio" name={`eq-${q.id}`} checked={r?.given === String(i)} onChange={() => check(String(i))} />{o}
                  </label>
                ))}
              </fieldset>
            ) : (
              <form className="mt-2 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); check(draft[k] ?? '') }}>
                <label htmlFor={`eans-${q.id}`} className="sr-only">Answer to question {n + 1}</label>
                <input id={`eans-${q.id}`} className="h-9 w-40 px-2 font-mono" value={draft[k] ?? ''} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} placeholder="your answer" autoComplete="off" />
                <button type="submit" className="btn btn-primary h-9">Check</button>
              </form>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <button type="button" className="btn h-8" onClick={() => setResults({ ...results, [k]: { given: r?.given ?? '', ok: r?.ok ?? false, revealed: true } })}>Reveal answer</button>
              {r && r.given !== '' && <span role="status" className={`text-[13px] ${r.ok ? 'text-ok' : 'text-rose'}`}>{r.ok ? 'Correct' : 'Not yet'}</span>}
            </div>
            {(r?.revealed || r?.ok) && <p className="mt-2 rounded-lg border border-line bg-panel-2 p-2 font-mono text-[12.5px] leading-relaxed text-muted font-num">Answer: <span className="text-amber">{q.options ? q.options[q.answer] : Number.isInteger(q.answer) ? q.answer : q.answer.toFixed(4)}</span>. {q.explain}</p>}
          </Card>
        )
      })}
    </div>
  )
}

/* ----------------------------------------------------------------- lesson */
export function LessonPanel({ goTo, openExample }: { goTo: (stageId: string) => void; openExample: (id: string) => void }) {
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
          {EMB_EXAMPLES.map((ex, k) => (
            <div key={ex.id} className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="flex flex-wrap items-center gap-2"><span className="eyebrow">example {k + 1}</span><span className={`sim-badge ${ex.space === 'toy' ? 'toy' : ''}`}>{ex.space === 'toy' ? 'simulation' : ex.space === 'ppmi' ? 'trained · PPMI-SVD' : 'trained · transformer'}</span></div>
              <h3 className="mt-1 font-display text-[17px]">{ex.title}</h3>
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
                  <input type="radio" name={`embq-${i}`} checked={ans[i] === k} onChange={() => setAns(ans.map((a, j) => (j === i ? k : a)))} />{o}
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
export function EmbInspector() {
  const { stage, tab, space, focusIdx, picks, selIdx, cell, metric, projection, vecA, vecB, scaleA } = useEmb()
  const { depth } = useApp()
  const v = space.vecs[focusIdx]
  const nn = nearest(space, v, 3, 'cosine', [focusIdx])
  const pair: [number, number] | null = cell && selIdx[cell[0]] !== undefined && selIdx[cell[1]] !== undefined ? [selIdx[cell[0]], selIdx[cell[1]]] : nn[0] ? [focusIdx, nn[0].i] : null
  const shown = tab === 'guided' ? stage : STAGES.find((s) => s.id === (tab === 'compare' ? 'compare' : 'neighbours'))!
  const isPlane = tab === 'guided' && (stage.id === 'dot' || stage.id === 'cosine')
  const a = [vecA[0] * scaleA, vecA[1] * scaleA]
  return (
    <div className="grid gap-4" aria-live="polite">
      <div>
        <div className="eyebrow">{tab === 'guided' ? shown.group : tab}</div>
        <h2 className="mt-1 font-display text-[22px] leading-tight">{shown.title}</h2>
      </div>
      <div className="overflow-x-auto rounded-lg border border-line bg-panel-2 px-3 py-3 scroll-thin"><Tex tex={shown.equation} block /></div>
      <p className="max-w-[60ch] text-[13.5px] leading-relaxed text-fg/90">{shown[depth]}</p>
      {isPlane ? (
        <section>
          <div className="eyebrow mb-1.5">your two vectors</div>
          <dl className="grid gap-1 font-mono text-[12px] font-num">
            {[['a', `[${f3(a[0])}, ${f3(a[1])}]`], ['b', `[${f3(vecB[0])}, ${f3(vecB[1])}]`], ['a · b', f4(dot(a, vecB))], ['‖a‖ · ‖b‖', `${f4(norm(a))} · ${f4(norm(vecB))}`], ['cos θ', f4(cosine(a, vecB))]].map(([k, val]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k}</dt><dd>{val}</dd></div>
            ))}
          </dl>
        </section>
      ) : (
        <>
          <section>
            <div className="eyebrow mb-1.5">source</div>
            <p className="mb-1"><SourceBadge space={space} /></p>
            <dl className="grid gap-1 font-mono text-[12px] font-num">
              {[['space', space.title], ['words × dims', `${space.words.length} × ${space.dim}`], ...Object.entries(space.stats ?? {}).slice(0, depth === 'beginner' ? 2 : 6).map(([k2, val]) => [k2, String(val)])].map(([k2, val]) => (
                <div key={k2} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k2}</dt><dd className="text-right">{val}</dd></div>
              ))}
            </dl>
            <p className="mt-1 text-[11px] text-faint">{KIND_LABEL[space.kind]}</p>
          </section>
          <section>
            <div className="eyebrow mb-1.5">focus “{picks.focus}”</div>
            <dl className="grid gap-1 font-mono text-[12px] font-num">
              <div className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">‖e‖</dt><dd>{f4(norm(v))}</dd></div>
              {nn.map((x) => <div key={x.i} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">cos with {x.word}</dt><dd>{f4(x.score)}</dd></div>)}
            </dl>
          </section>
          {pair && depth !== 'beginner' && (
            <section>
              <div className="eyebrow mb-1.5">{cell ? 'selected cell' : 'nearest pair'}: {space.words[pair[0]]} · {space.words[pair[1]]}</div>
              <dl className="grid gap-1 font-mono text-[12px] font-num">
                {(() => { const x = space.vecs[pair[0]], y = space.vecs[pair[1]]; return [['a · b', f4(dot(x, y))], ['‖a‖ ‖b‖', `${f4(norm(x))} × ${f4(norm(y))}`], ['cos = a·b / (‖a‖‖b‖)', f4(cosine(x, y))], ['Euclidean distance', f4(Math.sqrt(x.reduce((s, xi, i) => s + (xi - y[i]) ** 2, 0)))]] })().map(([k2, val]) => (
                  <div key={k2} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k2}</dt><dd>{val}</dd></div>
                ))}
              </dl>
              {depth === 'advanced' && <p className="mt-1 font-mono text-[11px] text-faint">metric in use: {METRIC_LABEL[metric]}</p>}
            </section>
          )}
          <section>
            <div className="eyebrow mb-1.5">projection faithfulness</div>
            <dl className="grid gap-1 font-mono text-[12px] font-num">
              {[['neighbourhoods kept', pct(projection.preservation.mean)], ['chance level', pct(projection.preservation.chance)], ['Shepard ρ', f3(projection.shepard.rho)], ...(projection.explained ? [['variance shown', pct(projection.explained.reduce((x, y) => x + y, 0))]] : [])].map(([k2, val]) => (
                <div key={k2} className="flex justify-between gap-3 border-b border-line/60 py-1"><dt className="text-muted">{k2}</dt><dd>{val}</dd></div>
              ))}
            </dl>
            <p className="mt-1 text-[11.5px] text-muted">The map is a {projection.coords[0]?.length ?? 2}-D shadow of {space.dim}-D vectors. Similarities above are computed in full dimension.</p>
          </section>
        </>
      )}
    </div>
  )
}
