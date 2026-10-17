/** Stage views for module 08. Every number is computed from the current corpus by src/engine/ngram.ts. */
import { AnimatePresence, motion } from 'framer-motion'
import { Pause, Play, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card } from '../../components/viz'
import { useColors } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import {
  BOS, EOS, METHOD_LABEL, NGramModel, distSum, distribution, key, makeScorer, normLambdas, scoreSentences, tuneLambdas, unkey,
  type Method, type ModelConfig, type Order, type Scorer,
} from '../../engine/ngram'
import { NGRAM_EXAMPLES } from './examples'
import { CurveChart, stageIndex } from './charts'
import { fmtP, fmtPP, showTok, useNG } from './context'

/* ================================================================ shared controls */

export function OrderPicker() {
  const { cfg, setCfg } = useNG()
  return (
    <div className="flex items-center gap-1" role="radiogroup" aria-label="n-gram order">
      <span className="mr-1 font-mono text-[11px] text-faint">order</span>
      {([1, 2, 3] as Order[]).map((o) => (
        <button key={o} type="button" role="radio" aria-checked={cfg.order === o} className="btn h-7 px-2" aria-pressed={cfg.order === o} onClick={() => setCfg({ order: o })}>
          {['unigram', 'bigram', 'trigram'][o - 1]}
        </button>
      ))}
    </div>
  )
}

function Slider({ id, label, value, min, max, step, onChange }: { id: string; label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label htmlFor={id} className="grid min-w-0 gap-1 text-[12px] text-muted font-num">{label}
      <input id={id} type="range" className="w-full min-w-0" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}

/** Parameter controls for a method. `only` limits which method's controls are shown. */
export function ParamControls({ cfg, setCfg, only, idp = '' }: { cfg: ModelConfig; setCfg: (p: Partial<ModelConfig>) => void; only?: Method; idp?: string }) {
  const m = only ?? cfg.method
  const lam = normLambdas(cfg)
  if (m === 'laplace') return <Slider id={`${idp}k`} label={`k = ${cfg.k.toFixed(2)}${cfg.k === 1 ? ' (Laplace)' : cfg.k === 0 ? ' (= MLE)' : ''}`} value={cfg.k} min={0} max={2} step={0.01} onChange={(k) => setCfg({ k })} />
  if (m === 'backoff') return <Slider id={`${idp}d`} label={`discount d = ${cfg.dBackoff.toFixed(2)}`} value={cfg.dBackoff} min={0.05} max={0.95} step={0.01} onChange={(dBackoff) => setCfg({ dBackoff })} />
  if (m === 'kn') return <Slider id={`${idp}D`} label={`discount D = ${cfg.dKN.toFixed(2)}`} value={cfg.dKN} min={0.05} max={0.95} step={0.01} onChange={(dKN) => setCfg({ dKN })} />
  if (m === 'interp') {
    const names = ['λ₀ uniform', 'λ₁ unigram', 'λ₂ bigram', 'λ₃ trigram']
    return (
      <div className="grid gap-2 sm:grid-cols-2">
        {names.slice(0, cfg.order + 1).map((nm, i) => (
          <Slider key={i} id={`${idp}lam${i}`} label={`${nm}: raw ${cfg.lambdas[i].toFixed(2)} → ${lam[i].toFixed(3)}`} value={cfg.lambdas[i]} min={0} max={1} step={0.01}
            onChange={(v) => { const l = [...cfg.lambdas] as ModelConfig['lambdas']; l[i] = v; setCfg({ lambdas: l }) }} />
        ))}
      </div>
    )
  }
  return <p className="text-[12px] text-muted">Maximum likelihood has no parameters.</p>
}

export function ConfigPanel({ cfg, setCfg, idp = '' }: { cfg: ModelConfig; setCfg: (p: Partial<ModelConfig>) => void; idp?: string }) {
  return (
    <div className="grid gap-3">
      <div className="grid gap-2 sm:grid-cols-2">
        <label htmlFor={`${idp}order`} className="grid gap-1 text-[12px] text-muted">order
          <select id={`${idp}order`} className="h-8 w-full px-2" value={cfg.order} onChange={(e) => setCfg({ order: +e.target.value as Order })}>
            <option value={1}>unigram</option><option value={2}>bigram</option><option value={3}>trigram</option>
          </select>
        </label>
        <label htmlFor={`${idp}method`} className="grid gap-1 text-[12px] text-muted">method
          <select id={`${idp}method`} className="h-8 w-full px-2" value={cfg.method} onChange={(e) => setCfg({ method: e.target.value as Method })}>
            {(Object.keys(METHOD_LABEL) as Method[]).map((m) => <option key={m} value={m}>{METHOD_LABEL[m]}</option>)}
          </select>
        </label>
      </div>
      <ParamControls cfg={cfg} setCfg={setCfg} idp={idp} />
    </div>
  )
}

/** Choose the history for the probability explorer. */
function ContextPicker() {
  const { model, eff, ctx, setCtx } = useNG()
  const n = eff.order
  const options = useMemo(() => {
    if (n === 1) return []
    const keys = n === 2 ? [...model.ctx1.keys()].map((v) => [BOS, v]) : [...model.ctx2.keys()].map(unkey)
    const cnt = (h: string[]) => (n === 2 ? model.ctx1.get(h[1]) ?? 0 : model.ctx2.get(key(h[0], h[1])) ?? 0)
    return keys.sort((a, b) => cnt(b) - cnt(a) || (a.join(' ') < b.join(' ') ? -1 : 1)).map((h) => ({ h, label: `${n === 2 ? h[1] : h.join(' ')}  (c=${cnt(h)})` }))
  }, [model, n])
  const cur = n === 2 ? key(BOS, ctx[1]) : key(ctx[0], ctx[1])
  const inList = options.some((o) => key(...o.h) === cur)
  const [custom, setCustom] = useState('')
  if (n === 1) return <span className="font-mono text-[12px] text-muted">unigram model: no context</span>
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor="ng-ctx" className="font-mono text-[11px] text-faint">context h =</label>
      <select id="ng-ctx" className="h-8 max-w-[260px] px-2 font-mono text-[12.5px]" value={inList ? cur : ''} onChange={(e) => e.target.value && setCtx(unkey(e.target.value))}>
        {!inList && <option value="">{(n === 2 ? ctx[1] : ctx.join(' '))} (unseen)</option>}
        {options.map((o) => <option key={key(...o.h)} value={key(...o.h)}>{o.label}</option>)}
      </select>
      <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); const w = custom.toLowerCase().match(/[a-z0-9']+/g) ?? []; if (w.length) setCtx(w.length === 1 ? [BOS, w[0]] : w.slice(-2)) }}>
        <label htmlFor="ng-ctx-custom" className="sr-only">Type a context</label>
        <input id="ng-ctx-custom" placeholder="or type words…" className="h-8 w-[150px] px-2 font-mono text-[12.5px]" value={custom} onChange={(e) => setCustom(e.target.value)} />
        <button type="submit" className="btn h-8">Use</button>
      </form>
    </div>
  )
}

/** P(w | h) over V for the current method, with the MLE distribution drawn behind it. */
export function DistExplorer({ scorer: sIn, title, rows = 12 }: { scorer?: Scorer; title?: string; rows?: number }) {
  const lab = useNG()
  const c = useColors()
  const s = sIn ?? lab.scorer
  const { model, ctx } = lab
  const n = s.cfg.order
  const mle = useMemo(() => makeScorer(model, { ...s.cfg, method: 'mle' }), [model, s.cfg])
  const dist = useMemo(() => distribution(s, ctx), [s, ctx])
  const sum = useMemo(() => distSum(s, ctx), [s, ctx])
  const h = ctx.slice(2 - (n - 1))
  const count = (w: string) => (n === 1 ? model.c1(w) : n === 2 ? model.c2(h[0], w) : model.c3(h[0], h[1], w))
  const seenMass = dist.reduce((a, d) => a + (count(d.w) > 0 ? d.p : 0), 0)
  const top = dist.slice(0, rows)
  const restMass = dist.slice(rows).reduce((a, d) => a + (d.p || 0), 0)
  const undefinedCtx = Number.isNaN(sum)
  return (
    <Card title={title ?? `P(w | ${n === 1 ? '∅' : h.join(' ')}) · ${METHOD_LABEL[s.cfg.method]}`}>
      <div className="mb-3"><ContextPicker /></div>
      {undefinedCtx ? (
        <p className="py-6 text-center text-[14px] text-rose">This context never occurred in training, so c(h·) = 0 and maximum likelihood is undefined (0/0). Every smoothed method still returns a distribution.</p>
      ) : (
        <>
          <div className="grid gap-[3px]">
            <div className="grid grid-cols-[100px_52px_1fr_78px] gap-2 font-mono text-[10.5px] text-faint"><span className="text-right">w</span><span className="text-right">c(h,w)</span><span>grey = MLE · bright = this method</span><span className="text-right">P(w|h)</span></div>
            {top.map((d) => {
              const pm = mle.p(d.w, ctx)
              return (
                <motion.div layout key={d.w} className="grid grid-cols-[100px_52px_1fr_78px] items-center gap-2 font-mono text-[12px]">
                  <span className="truncate text-right">{showTok(d.w)}</span>
                  <span className={`text-right font-num ${count(d.w) ? 'text-fg' : 'text-faint'}`}>{count(d.w)}</span>
                  <div className="relative h-[15px] rounded-sm" style={{ background: c.panel2 }}>
                    {!Number.isNaN(pm) && <div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${pm * 100}%`, background: c.lineStrong }} />}
                    <div className="absolute inset-y-[4px] left-0 rounded-sm transition-[width] duration-500" style={{ width: `${d.p * 100}%`, background: count(d.w) ? c.cyan : c.violet }} />
                  </div>
                  <span className="text-right text-muted font-num">{fmtP(d.p)}</span>
                </motion.div>
              )
            })}
            {dist.length > rows && <div className="font-mono text-[11px] text-faint">+ {dist.length - rows} more words share {(restMass * 100).toFixed(2)}%</div>}
          </div>
          <div className="mt-3 grid gap-1">
            <div className="flex h-3 overflow-hidden rounded-sm" aria-label="seen versus unseen probability mass">
              <div style={{ width: `${seenMass * 100}%`, background: c.cyan }} />
              <div style={{ width: `${(1 - seenMass) * 100}%`, background: c.violet }} />
            </div>
            <div className="flex flex-wrap justify-between gap-2 font-mono text-[11px] text-muted font-num">
              <span><span className="text-cyan">■</span> seen after h: {(seenMass * 100).toFixed(2)}%</span>
              <span><span className="text-violet">■</span> unseen (reserved): {((1 - seenMass) * 100).toFixed(2)}%</span>
              <span className={Math.abs(sum - 1) < 1e-9 ? 'text-ok' : 'text-rose'}>Σ_w P(w|h) = {sum.toFixed(9)} {Math.abs(sum - 1) < 1e-9 ? '✓' : ''}</span>
            </div>
          </div>
        </>
      )}
    </Card>
  )
}

/** Per-token scoring of the test text. */
export function ScoreTable({ compact = false }: { compact?: boolean }) {
  const { score, focus, setFocus, setCtx, eff, model } = useNG()
  const c = useColors()
  const maxBits = Math.max(1, ...score.tokens.map((t) => (Number.isFinite(t.log2p) ? -t.log2p : 0)))
  let cum = 0
  return (
    <Card title={`scoring the test text · ${METHOD_LABEL[eff.method]} · ${['unigram', 'bigram', 'trigram'][eff.order - 1]}`}>
      <SummaryTiles s={score} />
      {!compact && (
        <div className="mt-3 overflow-x-auto scroll-thin">
          <table className="w-full min-w-[560px] font-mono text-[12px] font-num">
            <thead><tr className="text-left text-[10.5px] text-faint"><th className="pb-1 pr-2 font-normal">t</th><th className="pr-2 font-normal">history → token</th><th className="pr-2 text-right font-normal">c(h,w)</th><th className="pr-2 text-right font-normal">P(w|h)</th><th className="pr-2 font-normal">−log₂ P (bits)</th><th className="text-right font-normal">Σ bits</th></tr></thead>
            <tbody>
              {score.tokens.map((t) => {
                const bits = -t.log2p
                cum += bits
                const cnt = eff.order === 1 ? model.c1(t.token) : eff.order === 2 ? model.c2(t.history[0], t.token) : model.c3(t.history[0], t.history[1], t.token)
                return (
                  <tr key={t.i} onClick={() => { setFocus(t.i); setCtx(t.history.length === 2 ? t.history : [BOS, t.history[0] ?? BOS]) }}
                    className={`cursor-pointer border-t border-line/50 ${t.i === focus ? 'bg-raise' : ''} ${t.p === 0 ? 'text-rose' : ''}`} aria-selected={t.i === focus}>
                    <td className="py-1 pr-2 text-faint">{t.i + 1}</td>
                    <td className="pr-2"><span className="text-muted">{t.history.map(showTok).join(' ')}{t.history.length ? ' → ' : ''}</span><span className={t.p === 0 ? '' : 'text-fg'}>{showTok(t.token)}</span>{t.word !== t.token && <span className="text-amber"> ({t.word})</span>}</td>
                    <td className={`pr-2 text-right ${cnt ? '' : 'text-faint'}`}>{cnt}</td>
                    <td className="pr-2 text-right">{fmtP(t.p)}</td>
                    <td className="pr-2">
                      <div className="h-[10px] w-full min-w-[80px] rounded-sm" style={{ background: c.panel2 }}>
                        <div className="h-full rounded-sm" style={{ width: `${Number.isFinite(bits) ? (bits / maxBits) * 100 : 100}%`, background: Number.isFinite(bits) ? c.amber : c.rose }} />
                      </div>
                    </td>
                    <td className="text-right">{Number.isFinite(cum) ? cum.toFixed(2) : '∞'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

export function SummaryTiles({ s }: { s: ReturnType<typeof scoreSentences> }) {
  const tiles: [string, string, string?][] = [
    ['tokens T', String(s.tokens.length)],
    ['log₂ P(text)', Number.isFinite(s.log2prob) ? s.log2prob.toFixed(3) : '−∞'],
    ['P(text)', fmtP(s.prob)],
    ['cross-entropy H', Number.isFinite(s.crossEntropy) ? `${s.crossEntropy.toFixed(3)} bits` : '∞'],
    ['perplexity 2ᴴ', fmtPP(s.perplexity), Number.isFinite(s.perplexity) ? 'text-amber' : 'text-rose'],
  ]
  return (
    <div className="@container"><dl className="grid grid-cols-2 gap-2 @lg:grid-cols-3 @3xl:grid-cols-5">
      {tiles.map(([k, v, cls]) => (
        <div key={k} className="rounded-lg border border-line bg-panel-2 px-2.5 py-2">
          <dt className="eyebrow">{k}</dt>
          <dd className={`mt-0.5 font-mono text-[15px] font-num ${cls ?? 'text-fg'}`}>{v}</dd>
        </div>
      ))}
    </dl></div>
  )
}

/* ================================================================ stage views */

function CorpusView() {
  const { corpus, setCorpus, test, setTest, heldOut, setHeldOut, model, exampleId } = useNG()
  return (
    <>
      <div className="flex flex-wrap gap-2">
        {NGRAM_EXAMPLES.map((ex) => <LoadExample key={ex.id} id={ex.id} active={exampleId === ex.id} />)}
      </div>
      <div className="grid gap-3 @3xl:grid-cols-[1.4fr_1fr]">
        <Card title="training corpus · edit freely">
          <label htmlFor="ng-corpus" className="sr-only">Training corpus</label>
          <textarea id="ng-corpus" value={corpus} onChange={(e) => setCorpus(e.target.value)} rows={7} spellCheck={false} className="w-full resize-y px-3 py-2 font-mono text-[13px] leading-relaxed" />
          <dl className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[12px] text-muted font-num">
            <div><dt className="inline">sentences </dt><dd className="inline text-fg">{model.sents.length}</dd></div>
            <div><dt className="inline">predicted tokens N </dt><dd className="inline text-fg">{model.N}</dd></div>
            <div><dt className="inline">|V| </dt><dd className="inline text-fg">{model.V}</dd></div>
            <div><dt className="inline">bigram types </dt><dd className="inline text-fg">{model.bi.size}</dd></div>
            <div><dt className="inline">trigram types </dt><dd className="inline text-fg">{model.tri.size}</dd></div>
          </dl>
        </Card>
        <div className="grid gap-3">
          <Card title="test text (scored)">
            <label htmlFor="ng-test" className="sr-only">Test text</label>
            <textarea id="ng-test" value={test} onChange={(e) => setTest(e.target.value)} rows={2} spellCheck={false} className="w-full resize-y px-3 py-2 font-mono text-[13px]" />
          </Card>
          <Card title="held-out text (for tuning)">
            <label htmlFor="ng-held" className="sr-only">Held-out text</label>
            <textarea id="ng-held" value={heldOut} onChange={(e) => setHeldOut(e.target.value)} rows={2} spellCheck={false} className="w-full resize-y px-3 py-2 font-mono text-[13px]" />
          </Card>
        </div>
      </div>
      <Card title={`padded sentences · first ${Math.min(6, model.sents.length)} of ${model.sents.length}`}>
        <div className="grid gap-1.5">
          {model.sents.slice(0, 6).map((s, i) => (
            <div key={i} className="flex flex-wrap gap-1">
              {[BOS, BOS, ...s, EOS].map((t, k) => (
                <span key={k} className={`rounded border px-1.5 py-0.5 font-mono text-[12.5px] ${t === BOS ? 'border-line text-faint' : t === EOS ? 'border-teal/60 text-teal' : 'border-line-strong text-fg'}`}>{t}</span>
              ))}
            </div>
          ))}
        </div>
      </Card>
    </>
  )
}

export function LoadExample({ id, active, label }: { id: string; active?: boolean; label?: string }) {
  const { loadExample } = useNG()
  const ex = NGRAM_EXAMPLES.find((e) => e.id === id)!
  return <button type="button" className="btn" aria-pressed={!!active} onClick={() => loadExample(id)}>{label ?? ex.title}</button>
}

function CountView() {
  const { model, eff } = useNG()
  const c = useColors()
  const positions = useMemo(() => {
    const out: { s: number; i: number }[] = []
    model.sents.forEach((s, si) => { for (let i = 2; i < s.length + 3; i++) out.push({ s: si, i }) })
    return out.slice(0, 400)
  }, [model])
  const win = useStepEngine(Math.max(1, positions.length))
  const pos = positions[Math.min(win.index, positions.length - 1)]
  if (!pos) return <Card title="counting"><p className="text-muted">Add some text to the corpus first.</p></Card>
  const toks = [BOS, BOS, ...model.sents[pos.s], EOS]
  const [u, v, w] = [toks[pos.i - 2], toks[pos.i - 1], toks[pos.i]]
  // counts accumulated up to and including the cursor
  const so = { w: 0, vw: 0, uvw: 0 }
  const tally = new Map<string, number>()
  for (let k = 0; k <= win.index && k < positions.length; k++) {
    const p = positions[k], t = [BOS, BOS, ...model.sents[p.s], EOS]
    const [a, b, d] = [t[p.i - 2], t[p.i - 1], t[p.i]]
    if (d === w) so.w++
    if (b === v && d === w) so.vw++
    if (a === u && b === v && d === w) so.uvw++
    const kk = eff.order === 1 ? d : eff.order === 2 ? `${b} ${d}` : `${a} ${b} ${d}`
    tally.set(kk, (tally.get(kk) ?? 0) + 1)
  }
  const top = [...tally].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 10)
  const mx = top[0]?.[1] ?? 1
  const roleOf = (k: number) => (k === pos.i ? 'w' : k === pos.i - 1 ? 'v' : k === pos.i - 2 ? 'u' : '')
  const col: Record<string, string> = { w: c.amber, v: c.cyan, u: c.violet }
  return (
    <>
      <Card title={`window at position ${win.index + 1} of ${positions.length}${positions.length === 400 ? ' (first 400 shown)' : ''} · sentence ${pos.s + 1}`}
        right={
          <div className="flex items-center gap-1" role="group" aria-label="window playback">
            <button type="button" className="btn icon-btn h-7 w-7" onClick={win.reset} aria-label="Reset window"><RotateCcw size={13} /></button>
            <button type="button" className="btn icon-btn h-7 w-7" onClick={win.prev} aria-label="Window back"><ChevronLeft size={14} /></button>
            <button type="button" className="btn h-7" onClick={win.toggle} aria-label={win.playing ? 'Pause window' : 'Play window'}>{win.playing ? <Pause size={13} /> : <Play size={13} />}{win.playing ? 'Pause' : 'Slide'}</button>
            <button type="button" className="btn icon-btn h-7 w-7" onClick={win.next} aria-label="Window forward"><ChevronRight size={14} /></button>
            <input aria-label="window speed" type="range" min={0.5} max={8} step={0.5} value={win.speed} onChange={(e) => win.setSpeed(+e.target.value)} className="w-20" />
          </div>
        }>
        <div className="flex flex-wrap gap-1.5 py-2">
          {toks.map((t, k) => {
            const r = roleOf(k)
            return (
              <motion.span key={k} layout className="relative rounded-md border px-2 py-1 font-mono text-[15px]"
                style={{ borderColor: r ? col[r] : c.line, color: r ? col[r] : k > pos.i ? c.faint : c.fg, boxShadow: r === 'w' ? `0 0 16px -4px ${c.amber}` : undefined }}>
                {t}
                {r && <span className="absolute -top-2 left-1 rounded bg-panel px-1 font-mono text-[9px]" style={{ color: col[r] }}>{r}</span>}
              </motion.span>
            )
          })}
        </div>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          {[
            { label: `c(${w})`, now: so.w, total: model.c1(w), color: c.amber },
            { label: `c(${v} ${w})`, now: so.vw, total: model.c2(v, w), color: c.cyan },
            { label: `c(${u} ${v} ${w})`, now: so.uvw, total: model.c3(u, v, w), color: c.violet },
          ].map((x) => (
            <div key={x.label} className="rounded-lg border border-line bg-panel-2 p-2">
              <div className="truncate font-mono text-[11.5px] text-muted">{x.label}</div>
              <div className="flex items-baseline gap-2 font-mono font-num">
                <AnimatePresence mode="popLayout">
                  <motion.span key={x.now} initial={{ y: -8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="text-[20px]" style={{ color: x.color }}>{x.now}</motion.span>
                </AnimatePresence>
                <span className="text-[11px] text-faint">so far · final {x.total}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
      <Card title={`running tally · top ${['unigrams', 'bigrams', 'trigrams'][eff.order - 1]} so far`}>
        <div className="grid gap-[3px]">
          {top.map(([k, n]) => (
            <motion.div layout key={k} className="grid grid-cols-[150px_1fr_40px] items-center gap-2 font-mono text-[12px]">
              <span className="truncate text-right">{k}</span>
              <div className="h-[13px] rounded-sm" style={{ background: c.panel2 }}><div className="h-full rounded-sm transition-[width] duration-300" style={{ width: `${(n / mx) * 100}%`, background: c.teal }} /></div>
              <span className="text-right text-muted font-num">{n}</span>
            </motion.div>
          ))}
        </div>
      </Card>
    </>
  )
}

function TablesView() {
  const { model, setCtx, ctx } = useNG()
  const c = useColors()
  const uni = [...model.uni].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 16)
  const rowsK = [...model.ctx1].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 11).map(([k]) => k)
  const colsK = [...model.uni].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 11).map(([k]) => k)
  const tri = [...model.tri].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 12)
  const maxC = Math.max(1, ...rowsK.flatMap((v) => colsK.map((w) => model.c2(v, w))))
  const cw = 46, ch = 26, lw = 74, th = 62
  return (
    <>
      <div className="grid gap-3 @5xl:grid-cols-[1fr_1.35fr]">
        <Card title={`unigram counts · top ${uni.length} of ${model.uni.size} · N = ${model.N}`}>
          <div className="grid gap-[3px]">
            {uni.map(([w, n]) => (
              <div key={w} className="grid grid-cols-[84px_1fr_92px] items-center gap-2 font-mono text-[12px]">
                <span className="truncate text-right">{w}</span>
                <div className="h-[13px] rounded-sm" style={{ background: c.panel2 }}><div className="h-full rounded-sm" style={{ width: `${(n / uni[0][1]) * 100}%`, background: c.teal }} /></div>
                <span className="text-right text-muted font-num">{n} · {(n / model.N).toFixed(3)}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card title="bigram counts c(v w) · rows = context v, columns = next word w · click a cell">
          <div className="overflow-x-auto scroll-thin">
            <svg viewBox={`0 0 ${lw + colsK.length * cw} ${th + rowsK.length * ch}`} width="100%" style={{ minWidth: 520 }} role="img" aria-label="bigram count matrix">
              {colsK.map((w, j) => <text key={w} x={lw + j * cw + cw / 2} y={th - 6} fontSize={11} fill={c.muted} transform={`rotate(-45 ${lw + j * cw + cw / 2} ${th - 6})`} fontFamily="JetBrains Mono, monospace">{w}</text>)}
              {rowsK.map((v, i) => (
                <g key={v}>
                  <text x={lw - 6} y={th + i * ch + ch / 2 + 4} fontSize={11.5} textAnchor="end" fill={ctx[1] === v ? c.amber : c.fg} fontFamily="JetBrains Mono, monospace">{v}</text>
                  {colsK.map((w, j) => {
                    const n = model.c2(v, w)
                    return (
                      <g key={w} onClick={() => setCtx([BOS, v])} style={{ cursor: 'pointer' }}>
                        <rect x={lw + j * cw} y={th + i * ch} width={cw - 2} height={ch - 2} rx={3} fill={n ? c.cyan : c.panel2} fillOpacity={n ? 0.18 + 0.82 * (n / maxC) : 1} stroke={ctx[1] === v ? c.amber : c.line} />
                        <text x={lw + j * cw + cw / 2 - 1} y={th + i * ch + ch / 2 + 3} fontSize={11} textAnchor="middle" fill={n ? c.fg : c.faint} fontFamily="JetBrains Mono, monospace">{n}</text>
                      </g>
                    )
                  })}
                </g>
              ))}
            </svg>
          </div>
          <p className="mt-1 font-mono text-[11px] text-faint">{rowsK.length} most frequent contexts × {colsK.length} most frequent words. Zeros dominate even here.</p>
        </Card>
      </div>
      <Card title={`trigram counts · top ${tri.length} of ${model.tri.size}`}>
        <div className="grid gap-x-6 gap-y-1 font-mono text-[12.5px] sm:grid-cols-2 @5xl:grid-cols-3 font-num">
          {tri.map(([k, n]) => {
            const [a, b, d] = unkey(k)
            return <div key={k} className="flex justify-between gap-2 border-b border-line/50 py-0.5"><span><span className="text-muted">{a} {b}</span> {d}</span><span className="text-muted">{n} · P={(n / model.ctx2.get(key(a, b))!).toFixed(3)}</span></div>
          })}
        </div>
      </Card>
    </>
  )
}

function MLEView() {
  const { scorer, ctx, model, eff } = useNG()
  const n = eff.order
  const h = ctx.slice(2 - (n - 1))
  const ch = n === 1 ? model.N : n === 2 ? model.ctx1.get(h[0]) ?? 0 : model.ctx2.get(key(h[0], h[1])) ?? 0
  const followers = n === 1 ? [...model.uni] : [...(n === 2 ? model.next1.get(h[0]) : model.next2.get(key(h[0], h[1]))) ?? []].map((w) => [w, n === 2 ? model.c2(h[0], w) : model.c3(h[0], h[1], w)] as [string, number])
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><OrderPicker /></div>
      <DistExplorer scorer={scorer} />
      <Card title={`worked calculation · c(h·) = ${ch}`}>
        <div className="flex flex-wrap gap-2 font-mono text-[12.5px] font-num">
          {followers.sort((a, b) => b[1] - a[1]).slice(0, 12).map(([w, cnt]) => (
            <span key={w} className="rounded-md border border-line bg-panel-2 px-2 py-1">P({w}|{n === 1 ? '∅' : h.join(' ')}) = {cnt}/{ch} = <span className="text-amber">{(cnt / ch).toFixed(4)}</span></span>
          ))}
          {!followers.length && <span className="text-rose">Unseen context: there is nothing to divide by.</span>}
        </div>
      </Card>
    </>
  )
}

function ScoreView() {
  return <><div className="flex flex-wrap items-center gap-2"><OrderPicker /></div><ScoreTable /></>
}

function ZerosView() {
  const { model, score, eff } = useNG()
  const c = useColors()
  const contexts = [BOS, ...model.vocab.filter((w) => w !== EOS && w !== '<unk>')]
  const V = model.V
  const possibleBi = contexts.length * V
  const possibleTri = contexts.length * contexts.length * V
  const singletons = [...model.bi.values()].filter((x) => x === 1).length
  const cell = Math.max(3, Math.min(26, Math.floor(440 / V)))
  const idx = new Map(model.vocab.map((w, i) => [w, i]))
  const cidx = new Map(contexts.map((w, i) => [w, i]))
  const zeros = score.tokens.filter((t) => !t.seenNgram)
  return (
    <>
      <div className="flex flex-wrap items-center gap-2"><OrderPicker /></div>
      <div className="grid gap-3 @5xl:grid-cols-[1fr_1fr]">
        <Card title={`every possible bigram · ${contexts.length} contexts × ${V} words`}>
          <div className="overflow-x-auto scroll-thin">
            <svg viewBox={`0 0 ${V * cell} ${contexts.length * cell}`} width={V * cell} style={{ maxWidth: '100%', height: 'auto', background: c.panel2 }} role="img" aria-label="bigram occupancy grid">
              {[...model.bi.keys()].map((k) => {
                const [v, w] = unkey(k)
                return <rect key={k} x={idx.get(w)! * cell} y={cidx.get(v)! * cell} width={cell - (cell > 3 ? 0.5 : 0)} height={cell - (cell > 3 ? 0.5 : 0)} fill={c.cyan} />
              })}
            </svg>
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-1 font-mono text-[12px] font-num">
            <dt className="text-muted">bigram types seen</dt><dd className="text-right">{model.bi.size} / {possibleBi} = {((model.bi.size / possibleBi) * 100).toFixed(2)}%</dd>
            <dt className="text-muted">trigram types seen</dt><dd className="text-right">{model.tri.size} / {possibleTri} = {((model.tri.size / possibleTri) * 100).toFixed(3)}%</dd>
            <dt className="text-muted">Good-Turing unseen mass N₁/N</dt><dd className="text-right">{singletons} / {model.N} = {(singletons / model.N).toFixed(3)}</dd>
          </dl>
        </Card>
        <Card title={`test ${['unigrams', 'bigrams', 'trigrams'][eff.order - 1]} with zero training count`}>
          {zeros.length ? (
            <ul className="grid gap-1 font-mono text-[13px]">
              {zeros.map((t) => <li key={t.i} className="text-rose">“{[...t.history, t.token].join(' ')}”{t.word !== t.token ? ` (${t.word} → <unk>)` : ''} → P_ML = 0</li>)}
            </ul>
          ) : <p className="text-[13px] text-ok">Every test n-gram was seen in training at this order. Try a trigram model, or add a new word to the test text.</p>}
          <div className="mt-3 rounded-lg border border-line bg-panel-2 p-3 font-mono text-[13px] font-num">
            MLE perplexity on the test text: <span className={Number.isFinite(score.perplexity) ? 'text-amber' : 'text-rose'}>{fmtPP(score.perplexity)}</span>
          </div>
          <p className="mt-2 text-[12.5px] text-muted">One zero makes the product zero, the log −∞ and the perplexity infinite. The next four stages are four different ways to reserve probability for the unseen.</p>
        </Card>
      </div>
    </>
  )
}

/** Perplexity curve over one parameter for the current order. */
function useCurve(param: 'k' | 'dBackoff' | 'dKN', xs: number[]) {
  const { model, eff, test, heldOut } = useNG()
  return useMemo(() => {
    const method: Method = param === 'k' ? 'laplace' : param === 'dBackoff' ? 'backoff' : 'kn'
    const t: number[] = [], h: number[] = []
    for (const x of xs) {
      const s = makeScorer(model, { ...eff, method, [param]: x })
      t.push(scoreSentences(s, test).perplexity)
      h.push(heldOut.trim() ? scoreSentences(s, heldOut).perplexity : NaN)
    }
    return { t, h }
  }, [model, eff.order, eff.lambdas, test, heldOut, param, xs]) // eslint-disable-line react-hooks/exhaustive-deps
}

const K_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2]
const D_GRID = Array.from({ length: 19 }, (_, i) => +(0.05 * (i + 1)).toFixed(2))

function LaplaceView() {
  const { cfg, setCfg, scorer, model, ctx, eff } = useNG()
  const c = useColors()
  const curve = useCurve('k', K_GRID)
  const n = eff.order
  const h = ctx.slice(2 - (n - 1))
  const ch = n === 1 ? model.N : n === 2 ? model.ctx1.get(h[0]) ?? 0 : model.ctx2.get(key(h[0], h[1])) ?? 0
  const seen = (n === 1 ? [...model.uni] : [...(n === 2 ? model.next1.get(h[0]) : model.next2.get(key(h[0], h[1]))) ?? []].map((w) => [w, n === 2 ? model.c2(h[0], w) : model.c3(h[0], h[1], w)] as [string, number]))
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
  const k = cfg.k, V = model.V
  return (
    <>
      <div className="flex flex-wrap items-end gap-4"><OrderPicker /><div className="w-[260px]"><ParamControls cfg={cfg} setCfg={setCfg} only="laplace" idp="st-" /></div></div>
      <div className="grid gap-3 @5xl:grid-cols-2">
        <DistExplorer scorer={scorer} />
        <div className="grid gap-3">
          <Card title={`adjusted counts c* for h = ${n === 1 ? '∅' : h.join(' ')} · the tax paid by seen n-grams`}>
            {ch ? (
              <table className="w-full font-mono text-[12px] font-num">
                <thead><tr className="text-left text-[10.5px] text-faint"><th className="font-normal">w</th><th className="text-right font-normal">c</th><th className="text-right font-normal">c* = (c+k)·c(h·)/(c(h·)+kV)</th><th className="text-right font-normal">c*/c</th></tr></thead>
                <tbody>
                  {seen.map(([w, cnt]) => {
                    const cs = ((cnt + k) * ch) / (ch + k * V)
                    return <tr key={w} className="border-t border-line/50"><td className="py-0.5">{w}</td><td className="text-right">{cnt}</td><td className="text-right text-amber">{cs.toFixed(3)}</td><td className="text-right">{(cs / cnt).toFixed(3)}</td></tr>
                  })}
                </tbody>
              </table>
            ) : <p className="text-[13px] text-muted">Unseen context: add-k gives the uniform distribution 1/|V| = {(1 / V).toFixed(4)}.</p>}
          </Card>
          <Card title="perplexity versus k">
            <CurveChart xs={K_GRID} current={k || 0.01} logX xLabel="k (log scale)" series={[{ name: 'test', ys: curve.t, color: c.amber }, { name: 'held-out', ys: curve.h, color: c.cyan }]} />
          </Card>
        </div>
      </div>
      <ScoreTable compact />
    </>
  )
}

function InterpView() {
  const { cfg, setCfg, scorer, score, model, eff, heldOut, heldScore, setFocus, focus } = useNG()
  const c = useColors()
  const [tuned, setTuned] = useState<{ before: number; after: number } | null>(null)
  const palette = [c.faint, c.teal, c.cyan, c.violet]
  const tune = () => {
    const before = heldScore.perplexity
    const best = tuneLambdas(model, eff, heldOut)
    setCfg({ lambdas: best.lambdas })
    setTuned({ before, after: best.pp })
  }
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><OrderPicker />
        <div className="flex items-center gap-2">
          <button type="button" className="btn btn-primary" onClick={tune} disabled={!heldOut.trim()}>Tune λ on held-out</button>
          {tuned && <span className="font-mono text-[12px] text-muted font-num">held-out PP {fmtPP(tuned.before)} → <span className="text-ok">{fmtPP(tuned.after)}</span></span>}
        </div>
      </div>
      <Card title="weights (normalized to sum to 1)"><ParamControls cfg={cfg} setCfg={setCfg} only="interp" idp="st-" /></Card>
      <div className="grid gap-3 @5xl:grid-cols-2">
        <DistExplorer scorer={scorer} />
        <Card title="who contributed each test token's probability">
          <div className="grid gap-1">
            {score.tokens.map((t) => {
              const tot = t.p || 1
              return (
                <button type="button" key={t.i} onClick={() => setFocus(t.i)} className={`grid grid-cols-[110px_1fr_64px] items-center gap-2 rounded px-1 text-left font-mono text-[12px] ${t.i === focus ? 'bg-raise' : ''}`}>
                  <span className="truncate text-right">{t.history.slice(-1).join(' ')} → {t.token}</span>
                  <div className="flex h-[13px] overflow-hidden rounded-sm" style={{ background: c.panel2 }}>
                    {t.breakdown.parts.map((p, i) => <div key={i} title={`${p.label} = ${p.value.toFixed(5)}`} style={{ width: `${(p.value / tot) * 100}%`, background: palette[i] }} />)}
                  </div>
                  <span className="text-right text-muted font-num">{fmtP(t.p)}</span>
                </button>
              )
            })}
          </div>
          <div className="mt-2 flex flex-wrap gap-3 font-mono text-[10.5px] text-muted">
            {['uniform', 'unigram', 'bigram', 'trigram'].slice(0, eff.order + 1).map((nm, i) => <span key={nm}><span style={{ color: palette[i] }}>■</span> {nm}</span>)}
          </div>
        </Card>
      </div>
      <ScoreTable compact />
    </>
  )
}

function BackoffView() {
  const { cfg, setCfg, scorer, score, focus, setFocus } = useNG()
  const c = useColors()
  const curve = useCurve('dBackoff', D_GRID)
  const orderColor = [c.faint, c.teal, c.cyan, c.violet]
  return (
    <>
      <div className="flex flex-wrap items-end gap-4"><OrderPicker /><div className="w-[260px]"><ParamControls cfg={cfg} setCfg={setCfg} only="backoff" idp="st-" /></div></div>
      <Card title="which order answered each test token?">
        <div className="flex flex-wrap gap-1.5">
          {score.tokens.map((t) => (
            <button key={t.i} type="button" onClick={() => setFocus(t.i)} className="chip" data-selected={t.i === focus}
              style={{ borderColor: orderColor[t.breakdown.usedOrder ?? 1] }}>
              <span>{t.token}</span><span className="id" style={{ color: orderColor[t.breakdown.usedOrder ?? 1] }}>{['', 'uni', 'bi', 'tri'][t.breakdown.usedOrder ?? 1]}</span>
            </button>
          ))}
        </div>
        <p className="mt-2 font-mono text-[11px] text-faint">The border color shows the highest order whose counts produced the probability; lower orders are reached through α.</p>
      </Card>
      <div className="grid gap-3 @5xl:grid-cols-2">
        <DistExplorer scorer={scorer} />
        <Card title="perplexity versus discount d">
          <CurveChart xs={D_GRID} current={cfg.dBackoff} xLabel="d" series={[{ name: 'test', ys: curve.t, color: c.amber }, { name: 'held-out', ys: curve.h, color: c.cyan }]} />
        </Card>
      </div>
      <ScoreTable compact />
    </>
  )
}

function KNView() {
  const { cfg, setCfg, scorer, model } = useNG()
  const c = useColors()
  const curve = useCurve('dKN', D_GRID)
  const [sel, setSel] = useState<string | null>(null)
  const D = cfg.dKN
  const B = model.contBiTypes, T = model.contUniTypes, V = model.V
  const rows = model.vocab.filter((w) => model.c1(w) > 0).map((w) => {
    const cont = model.contUni.get(w) ?? 0
    return { w, c: model.c1(w), pml: model.c1(w) / model.N, cont, pc: Math.max(cont - D, 0) / B + (D * T) / B / V }
  }).sort((a, b) => b.c - a.c || (a.w < b.w ? -1 : 1))
  const rankDiff = (() => {
    const byC = [...rows].sort((a, b) => b.pml - a.pml).map((r) => r.w)
    const byP = [...rows].sort((a, b) => b.pc - a.pc).map((r) => r.w)
    return new Map(rows.map((r) => [r.w, byC.indexOf(r.w) - byP.indexOf(r.w)]))
  })()
  const shown = [...rows].sort((a, b) => Math.abs(rankDiff.get(b.w)!) - Math.abs(rankDiff.get(a.w)!) || b.c - a.c).slice(0, 12).sort((a, b) => b.c - a.c)
  const left = sel ? [...model.bi.keys()].map(unkey).filter(([, w]) => w === sel).map(([v]) => v) : []
  return (
    <>
      <div className="flex flex-wrap items-end gap-4"><OrderPicker /><div className="w-[260px]"><ParamControls cfg={cfg} setCfg={setCfg} only="kn" idp="st-" /></div></div>
      <div className="grid gap-3 @5xl:grid-cols-[1.1fr_1fr]">
        <Card title={`continuation counts · N₁₊(••) = ${B} bigram types · click a word`}>
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full min-w-[460px] font-mono text-[12px] font-num">
              <thead><tr className="text-left text-[10.5px] text-faint"><th className="font-normal">w</th><th className="text-right font-normal">c(w)</th><th className="text-right font-normal">P_ML(w)</th><th className="text-right font-normal">N₁₊(• w)</th><th className="text-right font-normal">P_cont(w)</th><th className="text-right font-normal">rank change</th></tr></thead>
              <tbody>
                {shown.map((r) => {
                  const d = rankDiff.get(r.w)!
                  return (
                    <tr key={r.w} onClick={() => setSel(r.w)} className={`cursor-pointer border-t border-line/50 ${sel === r.w ? 'bg-raise' : ''}`}>
                      <td className="py-0.5">{r.w}</td><td className="text-right">{r.c}</td><td className="text-right">{r.pml.toFixed(4)}</td>
                      <td className="text-right text-cyan">{r.cont}</td><td className="text-right text-amber">{r.pc.toFixed(4)}</td>
                      <td className={`text-right ${d > 0 ? 'text-ok' : d < 0 ? 'text-rose' : 'text-faint'}`}>{d > 0 ? `▲ ${d}` : d < 0 ? `▼ ${-d}` : '·'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[12px] text-muted">The 12 words whose rank moves most between raw frequency and continuation probability. ▼ = a frequent word that appears in few contexts.</p>
          {sel && <p className="mt-2 font-mono text-[12.5px]"><span className="text-amber">{sel}</span> follows {left.length} distinct word{left.length === 1 ? '' : 's'}: <span className="text-muted">{left.join(', ')}</span></p>}
        </Card>
        <div className="grid gap-3">
          <DistExplorer scorer={scorer} rows={8} />
          <Card title="perplexity versus discount D">
            <CurveChart xs={D_GRID} current={cfg.dKN} xLabel="D" height={170} series={[{ name: 'test', ys: curve.t, color: c.amber }, { name: 'held-out', ys: curve.h, color: c.cyan }]} />
          </Card>
        </div>
      </div>
      <ScoreTable compact />
    </>
  )
}

export function useComparison(model: NGramModel, base: ModelConfig, test: string, heldOut: string) {
  return useMemo(() => {
    const methods = Object.keys(METHOD_LABEL) as Method[]
    return methods.map((method) => ({
      method,
      byOrder: ([1, 2, 3] as Order[]).map((order) => {
        const s = makeScorer(model, { ...base, method, order })
        const t = scoreSentences(s, test), h = scoreSentences(s, heldOut)
        return { order, test: t.perplexity, held: heldOut.trim() ? h.perplexity : NaN, zeros: t.zeroAt.length }
      }),
    }))
  }, [model, base, test, heldOut])
}

export function ComparisonTable() {
  const { model, cfg, test, heldOut } = useNG()
  const c = useColors()
  const table = useComparison(model, cfg, test, heldOut)
  const allHeld = table.flatMap((r) => r.byOrder.map((o) => o.held)).filter(Number.isFinite)
  const best = Math.min(...allHeld)
  const bars = table.map((r) => ({ m: r.method, v: r.byOrder[cfg.order - 1].held }))
  const finite = bars.map((b) => b.v).filter(Number.isFinite)
  const mx = Math.max(...finite, 1), mn = Math.min(...finite, mx)
  const lg = (x: number) => Math.log(x)
  return (
    <div className="grid gap-3 @5xl:grid-cols-[1.3fr_1fr]">
      <Card title="perplexity · test / held-out · current parameters">
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full min-w-[520px] font-mono text-[12.5px] font-num">
            <thead><tr className="text-left text-[10.5px] text-faint"><th className="pb-1 font-normal">method</th>{['unigram', 'bigram', 'trigram'].map((o) => <th key={o} className="text-right font-normal">{o}</th>)}</tr></thead>
            <tbody>
              {table.map((r) => (
                <tr key={r.method} className="border-t border-line/50">
                  <td className="py-1.5">{METHOD_LABEL[r.method]}</td>
                  {r.byOrder.map((o) => (
                    <td key={o.order} className={`text-right ${o.held === best ? 'text-ok' : ''}`}>
                      <span className={Number.isFinite(o.test) ? '' : 'text-rose'}>{fmtPP(o.test)}</span>
                      <span className="text-faint"> / </span>
                      <span className={Number.isFinite(o.held) ? (o.held === best ? 'text-ok' : 'text-cyan') : 'text-rose'}>{fmtPP(o.held)}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[12px] text-muted">Green = lowest held-out perplexity. ∞ = some token got probability 0. Parameters (k, λ, d, D) are the current settings; tune them in their stages.</p>
      </Card>
      <Card title={`held-out perplexity · ${['unigram', 'bigram', 'trigram'][cfg.order - 1]} · log scale`}>
        <div className="grid gap-1.5">
          {bars.map((b) => (
            <div key={b.m} className="grid grid-cols-[120px_1fr_56px] items-center gap-2 font-mono text-[12px]">
              <span className="truncate text-right">{METHOD_LABEL[b.m].split(' (')[0]}</span>
              <div className="h-[14px] rounded-sm" style={{ background: c.panel2 }}>
                <div className="h-full rounded-sm transition-[width] duration-500" style={{ width: Number.isFinite(b.v) ? `${8 + 92 * ((lg(b.v) - lg(mn) * 0.9) / (lg(mx) - lg(mn) * 0.9 || 1))}%` : '100%', background: Number.isFinite(b.v) ? (b.v === mn ? c.teal : c.cyan) : c.rose }} />
              </div>
              <span className="text-right text-muted font-num">{fmtPP(b.v)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function CompareStage() {
  return <><div className="flex flex-wrap items-center gap-2"><OrderPicker /></div><ComparisonTable /><ScoreTable compact /></>
}

export function GuidedStage() {
  const { stage } = useNG()
  const V: Record<string, () => React.JSX.Element> = {
    corpus: CorpusView, count: CountView, tables: TablesView, mle: MLEView, score: ScoreView, zeros: ZerosView,
    laplace: LaplaceView, interp: InterpView, backoff: BackoffView, kn: KNView, compare: CompareStage,
  }
  const View = V[stage.id]
  return (
    <AnimatePresence mode="wait">
      <motion.div key={stage.id} initial={{ opacity: 0.4, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0.4, y: -6 }} transition={{ duration: 0.2 }} className="grid min-w-0 gap-3 [&>*]:min-w-0">
        <View />
      </motion.div>
    </AnimatePresence>
  )
}

export { stageIndex }
