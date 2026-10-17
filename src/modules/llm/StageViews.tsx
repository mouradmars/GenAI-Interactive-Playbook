/** One view per journey step. Every number shown is read from the live forward trace. */
import { AnimatePresence, motion } from 'framer-motion'
import { lazy, Suspense, useMemo, useState } from 'react'
import { Card, Heatmap, ProbBars, TokenStrip, VectorBars, fmt, seqColor } from '../../components/viz'
import { useApp, useColors } from '../../sim/appState'
import { showToken } from '../../engine/model'
import { gelu, row, topk, stats, type Mat } from '../../engine/tensor'
import { entropy } from '../../engine/decoding'
import { PRESETS } from './lesson'
import { lastRow, useLab } from './context'

const EmbeddingCloud = lazy(() => import('../../components/three').then((m) => ({ default: m.EmbeddingCloud })))

const tokLabels = (t: string[]) => t.map((x, i) => `${i}:${showToken(x)}`.slice(0, 11))
const Loading3D = () => <div className="grid h-full min-h-[300px] place-items-center text-muted">Loading 3D view…</div>

export function StageView() {
  const { step } = useLab()
  const views: Record<string, () => React.JSX.Element> = {
    text: TextView, bpe: BPEView, ids: IdsView, embed: EmbedView, pos: PosView, ln1: LN1View, qkv: QKVView,
    scores: ScoresView, softmax: SoftmaxView, mix: MixView, mlp: MLPView, residual: ResidualView, lnf: LnfView,
    logits: LogitsView, probs: ProbsView, decode: DecodeView,
  }
  const V = views[step.kind]
  return (
    <AnimatePresence mode="wait">
      <motion.div key={step.id} initial={{ opacity: 0.4, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0.4, y: -6 }} transition={{ duration: 0.22 }}
        className="grid min-w-0 gap-3 [&>*]:min-w-0">
        <V />
      </motion.div>
    </AnimatePresence>
  )
}

function SelectedTokenBar({ note }: { note?: string }) {
  const { trace, sel, setSel } = useLab()
  return (
    <Card title="Tokens · click to inspect one" right={note && <span className="font-mono text-[11px] text-faint">{note}</span>}>
      <TokenStrip tokens={trace.tokens} ids={trace.ids} selected={sel} onSelect={setSel} />
    </Card>
  )
}

function HeadPicker() {
  const { model, head, setHead, step } = useLab()
  return (
    <div className="flex items-center gap-1" role="group" aria-label="attention head">
      <span className="mr-1 font-mono text-[11px] text-faint">L{(step.layer ?? 0) + 1} head</span>
      {Array.from({ length: model.meta.nHeads }, (_, h) => (
        <button key={h} type="button" className="btn h-7 px-2" aria-pressed={h === head} onClick={() => setHead(h)}>{h + 1}</button>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ input */
function TextView() {
  const { prompt, setPrompt, trace, generated, model, unknown, undoToken } = useLab()
  const [draft, setDraft] = useState(prompt)
  const commit = () => draft.trim() && setPrompt(draft)
  return (
    <>
      <Card title="Prompt">
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); commit() }}>
          <label htmlFor="prompt" className="sr-only">Prompt</label>
          <input id="prompt" value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit}
            className="min-w-0 flex-1 px-3 py-2 font-mono text-[15px]" spellCheck={false} autoComplete="off" />
          <button className="btn btn-primary h-[38px]" type="submit">Run forward pass</button>
        </form>
        <div className="mt-3 flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button key={p.prompt} type="button" className="btn" aria-pressed={p.prompt === prompt} onClick={() => { setDraft(p.prompt); setPrompt(p.prompt) }}>
              {p.prompt}
            </button>
          ))}
        </div>
        {unknown.length > 0 && (
          <p className="mt-3 text-[13px] text-rose">Characters outside the model's alphabet were dropped: {[...new Set(unknown)].join(' ')}. The model only knows lower-case letters, space, "." and ",".</p>
        )}
      </Card>
      <Card title="What the model will receive">
        <p className="font-mono text-[22px] leading-relaxed">
          <span className="text-rose">{'<bos>'}</span>{' '}
          <span>{prompt.toLowerCase()}</span>
          {generated.length > 0 && <span className="text-teal">{generated.map((g) => model.vocab[g].replace(/▁/g, ' ')).join('')}</span>}
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-4 font-mono text-[12px] text-muted font-num">
          <span>{prompt.length} characters</span><span>→ {trace.ids.length} tokens incl. {'<bos>'}</span><span>context limit {model.meta.ctx}</span>
          {generated.length > 0 && <button type="button" className="btn h-7" onClick={undoToken}>Remove last generated token</button>}
        </div>
        <p className="mt-3 max-w-[70ch] text-[13px] text-muted">
          The model is <b className="text-fg">{model.meta.name}</b>: a real {model.meta.nLayers}-layer, {model.meta.nHeads}-head GPT with {model.meta.params.toLocaleString()} parameters,
          trained in this repository for {model.meta.trainSteps.toLocaleString()} steps on a {model.meta.corpusSentences.toLocaleString()}-sentence toy corpus (final loss {model.meta.finalLoss}).
          Its world is small: animals and what they eat, capitals, colors, counting and walks to places.
        </p>
      </Card>
    </>
  )
}

function BPEView() {
  const { model, prompt } = useLab()
  const { depth } = useApp()
  const enc = useMemo(() => model.tokenizer.encode(' ' + prompt.trim()), [model, prompt])
  return (
    <Card title={`Merging characters into tokens · ${model.tokenizer.merges.length} learned merges`}>
      <div className="grid gap-4 overflow-x-auto scroll-thin">
        {enc.traces.map((t, wi) => (
          <div key={wi} className="min-w-0">
            <div className="mb-1 font-mono text-[11px] text-faint">word {wi + 1}: “{showToken(t.word)}” · {t.events.length} merges</div>
            <div className="grid gap-1">
              <SymRow syms={t.start} label="chars" />
              {(depth === 'beginner' ? t.events.slice(-1) : t.events).map((e, k) => (
                <motion.div key={k} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: k * 0.05 }}>
                  <SymRow syms={e.after} merged={e.pair.join('')} label={`#${e.rank}`} note={`${showToken(e.pair[0])} + ${showToken(e.pair[1])}`} />
                </motion.div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
function SymRow({ syms, merged, label, note }: { syms: string[]; merged?: string; label: string; note?: string }) {
  let used = false
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="w-12 shrink-0 text-right font-mono text-[10.5px] text-faint font-num">{label}</span>
      {syms.map((s, i) => {
        const hit = !used && s === merged
        if (hit) used = true
        return <span key={i} className={`rounded border px-1.5 py-0.5 font-mono text-[13px] ${hit ? 'border-amber text-amber' : 'border-line text-fg'}`}>{showToken(s)}</span>
      })}
      {note && <span className="ml-2 font-mono text-[11px] text-muted">merge rank {label.slice(1)}: {note}</span>}
    </div>
  )
}

function IdsView() {
  const { model, trace, sel, setSel } = useLab()
  const c = useColors()
  const set = new Set(trace.ids)
  const cols = 18
  return (
    <>
      <SelectedTokenBar />
      <Card title={`Vocabulary · ${model.vocab.length} entries · <bos>, ${model.vocab.length - 1 - model.tokenizer.merges.length} characters, ${model.tokenizer.merges.length} merges`}>
        <div className="overflow-x-auto scroll-thin">
          <svg viewBox={`0 0 ${cols * 44} ${Math.ceil(model.vocab.length / cols) * 26}`} width="100%" style={{ minWidth: 560 }} role="img" aria-label="vocabulary grid">
            {model.vocab.map((t, i) => {
              const x = (i % cols) * 44, y = Math.floor(i / cols) * 26
              const on = set.has(i)
              const isSel = trace.ids[sel] === i
              return (
                <g key={i} onClick={() => on && setSel(trace.ids.indexOf(i))} style={{ cursor: on ? 'pointer' : 'default' }}>
                  <rect x={x + 1} y={y + 1} width={42} height={24} rx={3} fill={on ? (isSel ? c.amber : c.cyan) : c.panel2} fillOpacity={on ? 0.85 : 1} stroke={c.line} />
                  <text x={x + 22} y={y + 12} fontSize={9.5} textAnchor="middle" fill={on ? c.bg : c.muted} fontFamily="JetBrains Mono, monospace">{showToken(t).slice(0, 6)}</text>
                  <text x={x + 22} y={y + 22} fontSize={7.5} textAnchor="middle" fill={on ? c.bg : c.faint} fontFamily="JetBrains Mono, monospace">{i}</text>
                </g>
              )
            })}
          </svg>
        </div>
      </Card>
    </>
  )
}

/* -------------------------------------------------------------- embedding */
function EmbedView() {
  const { model, trace, sel } = useLab()
  return (
    <>
      <SelectedTokenBar />
      <div className="grid gap-3">
        <Card title={`token embeddings · [${trace.ids.length}, ${model.meta.dModel}]`}>
          <Heatmap m={trace.tokEmb} rowLabels={tokLabels(trace.tokens)} selectedRow={sel} ariaLabel="token embedding matrix" />
          <div className="mt-3"><VectorBars v={row(trace.tokEmb, sel)} label={<>row {trace.ids[sel]} of W_E = “{showToken(trace.tokens[sel])}”</>} /></div>
        </Card>
        <Card title={`all ${model.vocab.length} embeddings · PCA to 3D · drag to orbit`} className="min-h-[380px]">
          <div className="h-[360px]">
            <Suspense fallback={<Loading3D />}>
              <EmbeddingCloud table={model.W.tok} vocab={model.vocab} highlight={trace.ids} selected={trace.ids[sel]} />
            </Suspense>
          </div>
        </Card>
      </div>
    </>
  )
}

function PosView() {
  const { trace, sel } = useLab()
  const labels = tokLabels(trace.tokens)
  const ma = Math.max(...Array.from(trace.x0.data, Math.abs))
  return (
    <>
      <SelectedTokenBar />
      <div className="grid gap-3 lg:grid-cols-3">
        <Card title="W_E[id] · what"><Heatmap m={trace.tokEmb} rowLabels={labels} selectedRow={sel} maxAbs={ma} /></Card>
        <Card title="+ W_P[i] · where"><Heatmap m={trace.posEmb} rowLabels={trace.tokens.map((_, i) => `pos ${i}`)} selectedRow={sel} maxAbs={ma} /></Card>
        <Card title="= x⁽⁰⁾ · residual stream begins"><Heatmap m={trace.x0} rowLabels={labels} selectedRow={sel} maxAbs={ma} /></Card>
      </div>
      <Card title={`token ${sel}: the three vectors (48 dims)`}>
        <div className="grid gap-2 md:grid-cols-3">
          <VectorBars v={row(trace.tokEmb, sel)} label="token" maxAbs={ma} />
          <VectorBars v={row(trace.posEmb, sel)} label="position" maxAbs={ma} />
          <VectorBars v={row(trace.x0, sel)} label="sum" maxAbs={ma} />
        </div>
      </Card>
    </>
  )
}

/* ------------------------------------------------------------- attention */
function LN1View() {
  const { trace, step, sel } = useLab()
  const L = trace.layers[step.layer!]
  return <NormView before={L.xIn} after={L.ln1.out} means={L.ln1.means} stds={L.ln1.stds} sel={sel} labels={tokLabels(trace.tokens)} />
}
function LnfView() {
  const { trace, sel } = useLab()
  return <NormView before={trace.layers[trace.layers.length - 1].xOut} after={trace.lnf.out} means={trace.lnf.means} stds={trace.lnf.stds} sel={sel} labels={tokLabels(trace.tokens)} />
}
function NormView({ before, after, means, stds, sel, labels }: { before: Mat; after: Mat; means: number[]; stds: number[]; sel: number; labels: string[] }) {
  const { depth } = useApp()
  const b = row(before, sel)
  return (
    <>
      <SelectedTokenBar />
      <div className="grid gap-3 lg:grid-cols-2">
        <Card title="before · residual stream x"><Heatmap m={before} rowLabels={labels} selectedRow={sel} /></Card>
        <Card title="after · LN(x)"><Heatmap m={after} rowLabels={labels} selectedRow={sel} /></Card>
      </div>
      <Card title={`token ${sel} · μ = ${fmt(means[sel])} · σ = ${fmt(stds[sel])}`}>
        <div className="grid gap-3 md:grid-cols-2">
          <VectorBars v={b} label={`x: range ${fmt(stats(b).min, 2)} … ${fmt(stats(b).max, 2)}`} />
          <VectorBars v={row(after, sel)} label="LN(x) = γ ⊙ (x − μ)/σ + β" />
        </div>
        {depth !== 'beginner' && (
          <p className="mt-2 font-mono text-[12px] text-muted font-num">
            dim 0: ({fmt(b[0], 4)} − {fmt(means[sel], 4)}) / {fmt(stds[sel], 4)} = {fmt((b[0] - means[sel]) / stds[sel], 4)} → ×γ₀ + β₀ = <span className="text-amber">{fmt(after.data[sel * after.cols], 4)}</span>
          </p>
        )}
      </Card>
    </>
  )
}

function QKVView() {
  const { trace, step, sel, model } = useLab()
  const L = trace.layers[step.layer!]
  const labels = tokLabels(trace.tokens)
  const HD = model.headDim
  return (
    <>
      <SelectedTokenBar note={`dashed lines split ${model.meta.nHeads} heads × ${HD} dims`} />
      <div className="grid gap-3 xl:grid-cols-3">
        <Card title="Q = LN(x) W_Q · queries"><Heatmap m={L.q} rowLabels={labels} selectedRow={sel} colGroup={HD} /></Card>
        <Card title="K = LN(x) W_K · keys"><Heatmap m={L.k} rowLabels={labels} selectedRow={sel} colGroup={HD} /></Card>
        <Card title="V = LN(x) W_V · values"><Heatmap m={L.v} rowLabels={labels} selectedRow={sel} colGroup={HD} /></Card>
      </div>
    </>
  )
}

function ScoresView() {
  const { trace, step, head, cell, setCell, sel, model } = useLab()
  const { depth } = useApp()
  const H = trace.layers[step.layer!].heads[head]
  const labels = tokLabels(trace.tokens)
  const [i, j] = cell && cell[0] < trace.ids.length && cell[1] <= cell[0] ? cell : [sel, Math.max(0, sel - 1)]
  const qi = row(H.q, i), kj = row(H.k, j)
  const prods = Array.from(qi, (q, d) => q * kj[d])
  const sum = prods.reduce((a, b) => a + b, 0)
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><HeadPicker /><span className="font-mono text-[11px] text-faint">click a cell to expand its dot product</span></div>
      <div className="grid gap-3 xl:grid-cols-[1.1fr_1fr]">
        <Card title={`S = Q_h K_hᵀ / √${model.headDim} · head ${head + 1}`}>
          <Heatmap m={H.scores} rowLabels={labels} colLabels={trace.tokens.map(showToken)} mask showValues={trace.ids.length <= 12}
            selectedCell={[i, j]} onCell={(a, b) => b <= a && setCell([a, b])} cell={trace.ids.length <= 12 ? 38 : 22} />
        </Card>
        <Card title={`S[${i}, ${j}] · query “${showToken(trace.tokens[i])}” · key “${showToken(trace.tokens[j])}”`}>
          <div className="grid gap-2">
            <VectorBars v={qi} label={`q_${i} (${model.headDim} dims)`} color="var(--cyan)" />
            <VectorBars v={kj} label={`k_${j}`} color="var(--violet)" />
            <VectorBars v={prods} label="element-wise q · k" />
          </div>
          {depth !== 'beginner' && (
            <p className="mt-3 break-words font-mono text-[12px] leading-relaxed text-muted font-num">
              {prods.slice(0, depth === 'advanced' ? 12 : 4).map((p) => fmt(p, 3)).join(' + ')}{depth === 'advanced' ? '' : ' + …'}
              {' '}= {fmt(sum, 4)} ; ÷ √{model.headDim} = <span className="text-amber">{fmt(sum / Math.sqrt(model.headDim), 4)}</span>
            </p>
          )}
        </Card>
      </div>
    </>
  )
}

function SoftmaxView() {
  const { trace, step, head, setHead, sel, setSel, model } = useLab()
  const c = useColors()
  const L = trace.layers[step.layer!]
  const n = trace.ids.length
  const H = L.heads[head]
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><HeadPicker /><span className="font-mono text-[11px] text-faint">query token = selected chip</span></div>
      <PredictGate kind="attention">
        <Card title={`where does “${showToken(trace.tokens[sel])}” look? · layer ${step.layer! + 1}, head ${head + 1}`}>
          <AttentionArcs tokens={trace.tokens} probs={H.probs} query={sel} onSelect={setSel} />
        </Card>
        <div className="grid gap-3 xl:grid-cols-[1fr_1fr]">
          <Card title="A = softmax(S) · rows sum to 1">
            <Heatmap m={H.probs} mode="prob" rowLabels={tokLabels(trace.tokens)} colLabels={trace.tokens.map(showToken)} mask showValues={n <= 12}
              selectedRow={sel} cell={n <= 12 ? 38 : 22} />
          </Card>
          <Card title="all heads in this layer · click to focus">
            <div className="grid grid-cols-2 gap-2">
              {L.heads.map((h, k) => (
                <button key={k} type="button" onClick={() => setHead(k)} className={`rounded-lg border p-1 text-left ${k === head ? 'border-cyan' : 'border-line'}`}>
                  <div className="mb-1 font-mono text-[10.5px] text-muted">head {k + 1}</div>
                  <svg viewBox={`0 0 ${n} ${n}`} width="100%" style={{ aspectRatio: '1', maxWidth: '100%' }} aria-label={`head ${k + 1} attention`}>
                    {Array.from({ length: n * n }, (_, q) => {
                      const a = Math.floor(q / n), b = q % n
                      return <rect key={q} x={b} y={a} width={0.95} height={0.95} fill={b > a ? c.bg : seqColor(c)(h.probs.data[q])} />
                    })}
                  </svg>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[12px] text-muted">{model.meta.nHeads} heads see the same tokens but learn different patterns.</p>
          </Card>
        </div>
      </PredictGate>
    </>
  )
}

export function AttentionArcs({ tokens, probs, query, onSelect }: { tokens: string[]; probs: Mat; query: number; onSelect?: (i: number) => void }) {
  const c = useColors()
  const n = tokens.length
  const step = 86
  const W = n * step + 20
  const base = 150
  return (
    <div className="overflow-x-auto scroll-thin">
      <svg viewBox={`0 0 ${W} ${base + 54}`} width="100%" style={{ minWidth: Math.min(W, 640) }} role="img" aria-label="attention arcs">
        {Array.from({ length: query + 1 }, (_, j) => {
          const w = probs.data[query * n + j]
          const x1 = 10 + j * step + step / 2, x2 = 10 + query * step + step / 2
          const h = Math.min(base - 24, 26 + Math.abs(x2 - x1) * 0.32)
          const d = j === query ? `M ${x2 - 14} ${base} C ${x2 - 22} ${base - 46}, ${x2 + 22} ${base - 46}, ${x2 + 14} ${base}` : `M ${x1} ${base} Q ${(x1 + x2) / 2} ${base - h * 2} ${x2} ${base}`
          return (
            <motion.path key={j} d={d} fill="none" stroke={c.cyan} strokeWidth={1 + w * 14} strokeOpacity={0.15 + w * 0.85} strokeLinecap="round"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.6, delay: j * 0.04 }} />
          )
        })}
        {Array.from({ length: query + 1 }, (_, j) => {
          const w = probs.data[query * n + j]
          return (
            <text key={`l${j}`} x={10 + j * step + step / 2} y={base + 46} fontSize={12} textAnchor="middle" fill={w > 0.2 ? c.amber : c.muted}
              fontFamily="JetBrains Mono, monospace">{(w * 100).toFixed(w < 0.1 ? 1 : 0)}%</text>
          )
        })}
        {tokens.map((t, i) => (
          <g key={i} onClick={() => onSelect?.(i)} style={{ cursor: 'pointer' }}>
            <rect x={10 + i * step + 6} y={base + 4} width={step - 12} height={26} rx={6} fill={i === query ? c.amber : c.panel2}
              stroke={i > query ? c.line : c.lineStrong} strokeDasharray={i > query ? '3 3' : undefined} />
            <text x={10 + i * step + step / 2} y={base + 21} fontSize={12} textAnchor="middle" fill={i === query ? c.bg : i > query ? c.faint : c.fg}
              fontFamily="JetBrains Mono, monospace">{showToken(t)}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function MixView() {
  const { trace, step, head, sel, model } = useLab()
  const L = trace.layers[step.layer!]
  const H = L.heads[head]
  const n = trace.ids.length
  const weights = Array.from({ length: sel + 1 }, (_, j) => H.probs.data[sel * n + j])
  const order = topk(weights, Math.min(4, weights.length))
  const ma = Math.max(...Array.from(H.v.data, Math.abs))
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2"><HeadPicker /><span className="font-mono text-[11px] text-faint">o = Σⱼ Aᵢⱼ vⱼ for the selected token</span></div>
      <div className="grid gap-3 xl:grid-cols-2">
        <Card title={`head ${head + 1}: weighted values for “${showToken(trace.tokens[sel])}” (top ${order.length} of ${sel + 1})`}>
          <div className="grid gap-2">
            {order.map((j) => (
              <div key={j} className="grid grid-cols-[88px_1fr] items-center gap-2">
                <span className="font-mono text-[11.5px] text-muted font-num">{(weights[j] * 100).toFixed(1)}% × v<sub>{j}</sub><br /><span className="text-fg">{showToken(trace.tokens[j])}</span></span>
                <VectorBars v={row(H.v, j).map((x) => x * weights[j])} maxAbs={ma} height={40} />
              </div>
            ))}
            <div className="grid grid-cols-[88px_1fr] items-center gap-2 border-t border-line pt-2">
              <span className="font-mono text-[11.5px] text-amber">= head out</span>
              <VectorBars v={row(H.out, sel)} maxAbs={ma} height={40} color="var(--teal)" />
            </div>
          </div>
        </Card>
        <Card title="concat all heads → W_O → add to residual">
          <div className="grid gap-2">
            <VectorBars v={row(L.concat, sel)} label={`concat [${model.meta.dModel}] (${model.meta.nHeads} heads side by side)`} color="var(--teal)" />
            <VectorBars v={row(L.xIn, sel)} label="x (residual in)" />
            <VectorBars v={row(L.attnOut, sel)} label="+ attention update (concat · W_O + b)" />
            <VectorBars v={row(L.xMid, sel)} label="= x′ (residual out)" />
          </div>
        </Card>
      </div>
      <SelectedTokenBar />
    </>
  )
}

function MLPView() {
  const { trace, step, sel, model } = useLab()
  const c = useColors()
  const L = trace.layers[step.layer!]
  const pre = row(L.pre, sel), act = row(L.act, sel)
  const M = model.meta.dMlp
  const cols = 24
  const active = act.filter((a) => a > 0.1).length
  const maxA = Math.max(...act)
  const xs = [-4, 4], ys = [-0.4, 4]
  const px = (x: number) => ((x - xs[0]) / (xs[1] - xs[0])) * 300 + 30
  const py = (y: number) => 170 - ((y - ys[0]) / (ys[1] - ys[0])) * 150
  const curve = Array.from({ length: 81 }, (_, i) => { const x = xs[0] + (i / 80) * (xs[1] - xs[0]); return `${i ? 'L' : 'M'}${px(x)},${py(gelu(x))}` }).join(' ')
  return (
    <>
      <SelectedTokenBar note={`${active} of ${M} neurons > 0.1 for this token`} />
      <div className="grid gap-3 xl:grid-cols-2">
        <Card title="GELU · this token's pre-activations">
          <svg viewBox="0 0 340 190" width="100%" role="img" aria-label="GELU curve with activations">
            <line x1={px(xs[0])} x2={px(xs[1])} y1={py(0)} y2={py(0)} stroke={c.line} />
            <line x1={px(0)} x2={px(0)} y1={py(ys[0])} y2={py(ys[1])} stroke={c.line} />
            {[-4, -2, 0, 2, 4].map((t) => <text key={t} x={px(t)} y={186} fontSize={9} textAnchor="middle" fill={c.faint}>{t}</text>)}
            <path d={curve} fill="none" stroke={c.violet} strokeWidth={2} />
            {Array.from(pre, (x, k) => {
              const xc = Math.max(xs[0], Math.min(xs[1], x))
              return <circle key={k} cx={px(xc)} cy={py(gelu(xc))} r={2.6} fill={c.amber} fillOpacity={0.7} />
            })}
            <text x={36} y={18} fontSize={10.5} fill={c.muted} fontFamily="JetBrains Mono, monospace">GELU(x) = ½x(1 + tanh(√(2/π)(x + 0.044715x³)))</text>
          </svg>
        </Card>
        <Card title={`hidden layer · ${M} neurons after GELU`}>
          <svg viewBox={`0 0 ${cols * 14} ${Math.ceil(M / cols) * 14}`} width="100%" role="img" aria-label="neuron activations">
            {Array.from(act, (a, k) => (
              <rect key={k} x={(k % cols) * 14} y={Math.floor(k / cols) * 14} width={12} height={12} rx={2}
                fill={a > 0 ? seqColor(c, maxA)(a) : c.panel2} stroke={a > 0.1 ? c.cyan : c.line} strokeOpacity={a > 0.1 ? 0.6 : 1}>
                <title>neuron {k}: pre {fmt(pre[k])}, post {fmt(a)}</title>
              </rect>
            ))}
          </svg>
        </Card>
      </div>
      <Card title="per-token update: x′ + MLP(LN(x′))">
        <div className="grid gap-2 md:grid-cols-3">
          <VectorBars v={row(L.xMid, sel)} label="x′ (after attention)" />
          <VectorBars v={row(L.mlpOut, sel)} label="+ MLP output (act · W₂ + b₂)" />
          <VectorBars v={row(L.xOut, sel)} label="= residual after block" />
        </div>
      </Card>
    </>
  )
}

function ResidualView() {
  const { trace, step, model } = useLab()
  const l = step.layer!
  const n = trace.ids.length
  const depths = [{ name: 'embedding', m: trace.x0 }, ...trace.layers.slice(0, l + 1).map((L, k) => ({ name: `after block ${k + 1}`, m: L.xOut }))]
  return (
    <>
      <Card title="logit lens · what would each position predict if we stopped here?">
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full border-separate border-spacing-y-1 font-mono text-[12.5px] font-num">
            <thead><tr className="text-left text-faint"><th className="pr-3 font-normal">position · token</th>{depths.map((d) => <th key={d.name} className="pr-3 font-normal">{d.name}</th>)}</tr></thead>
            <tbody>
              {Array.from({ length: n }, (_, i) => (
                <tr key={i}>
                  <td className="pr-3 text-fg">{i} · {showToken(trace.tokens[i])}</td>
                  {depths.map((d) => {
                    const p = model.logitLens(d.m, i)
                    const k = topk(p, 1)[0]
                    return (
                      <td key={d.name} className="pr-3">
                        <span className="rounded px-1.5 py-0.5" style={{ background: `color-mix(in oklab, var(--cyan) ${Math.round(p[k] * 55)}%, transparent)` }}>{showToken(model.vocab[k])}</span>
                        <span className="ml-1 text-muted">{(p[k] * 100).toFixed(0)}%</span>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[12px] text-muted">Each row reads the residual vector at that position through the final LayerNorm and unembedding. The last row is the one used for generation.</p>
      </Card>
      <Card title="‖x‖ per token as the stream deepens">
        <ProbBars unit="logit" items={depths.flatMap((d) => [{ label: d.name, value: Math.sqrt(row(d.m, n - 1).reduce((a, b) => a + b * b, 0)), sub: '' }])} accent="var(--teal)" />
        <p className="mt-1 font-mono text-[11px] text-faint">norm of the last token's residual vector</p>
      </Card>
    </>
  )
}

/* ----------------------------------------------------------------- output */
function LogitsView() {
  const { trace, model } = useLab()
  const c = useColors()
  const z = lastRow(trace)
  const idx = topk(z, 14)
  const s = stats(z)
  const bins = 30
  const hist = new Array(bins).fill(0)
  z.forEach((v) => hist[Math.min(bins - 1, Math.floor(((v - s.min) / (s.max - s.min)) * bins))]++)
  const hm = Math.max(...hist)
  return (
    <div className="grid gap-3 xl:grid-cols-[1.2fr_1fr]">
      <Card title={`top logits for position ${trace.ids.length - 1} (“${showToken(trace.tokens[trace.ids.length - 1])}”)`}>
        <ProbBars unit="logit" items={idx.map((i) => ({ label: showToken(model.vocab[i]), value: z[i] }))} />
      </Card>
      <Card title={`all ${model.meta.vocabSize} logits · histogram`}>
        <svg viewBox={`0 0 ${bins * 10} 120`} width="100%" role="img" aria-label="logit histogram">
          {hist.map((h, i) => <rect key={i} x={i * 10 + 1} y={110 - (h / hm) * 100} width={8} height={(h / hm) * 100} fill={i === bins - 1 ? c.amber : c.violet} rx={1} />)}
          <line x1={0} x2={bins * 10} y1={110.5} y2={110.5} stroke={c.line} />
        </svg>
        <div className="flex justify-between font-mono text-[11px] text-muted font-num"><span>{fmt(s.min, 1)}</span><span>mean {fmt(s.mean, 2)}</span><span>{fmt(s.max, 1)}</span></div>
        <p className="mt-2 text-[12.5px] text-muted">Most of the vocabulary sits in a low cluster. Only a few tokens get a logit far to the right, and softmax will amplify that gap exponentially.</p>
      </Card>
    </div>
  )
}

function ProbsView() {
  const { trace, model } = useLab()
  const p = useMemo(() => { const z = lastRow(trace); const e = Array.from(z, (v) => Math.exp(v - Math.max(...z))); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s) }, [trace])
  const idx = topk(p, 12)
  const H = entropy(p)
  return (
    <PredictGate kind="next">
      <div className="grid gap-3 xl:grid-cols-[1.4fr_1fr]">
        <Card title="next-token probabilities · top 12">
          <ProbBars items={idx.map((i) => ({ label: showToken(model.vocab[i]), value: p[i] }))} max={1} highlight={0} />
        </Card>
        <Card title="the distribution in numbers">
          <dl className="grid grid-cols-2 gap-y-2 font-mono text-[13px] font-num">
            <dt className="text-muted">Σ p</dt><dd className="text-ok">{p.reduce((a, b) => a + b, 0).toFixed(6)}</dd>
            <dt className="text-muted">entropy</dt><dd>{H.toFixed(3)} bits</dd>
            <dt className="text-muted">perplexity 2ᴴ</dt><dd>{(2 ** H).toFixed(2)}</dd>
            <dt className="text-muted">top-1</dt><dd className="text-amber">{showToken(model.vocab[idx[0]])} · {(p[idx[0]] * 100).toFixed(2)}%</dd>
            <dt className="text-muted">tokens above 1%</dt><dd>{p.filter((x) => x > 0.01).length}</dd>
          </dl>
          <p className="mt-3 text-[12.5px] text-muted">{H < 0.3 ? 'The model is almost certain: the training data always continued this way.' : H > 2 ? 'Probability is spread over several options. The corpus really did vary here.' : 'A few candidates compete.'}</p>
        </Card>
      </div>
    </PredictGate>
  )
}

function DecodeView() {
  const { decode, setDecode, decodeResult: r, model, appendToken, sampleU, trace } = useLab()
  const c = useColors()
  const idx = topk(r.tempered, 10)
  const cdf: { id: number; a: number; b: number }[] = []
  let acc = 0
  topk(r.final, r.final.length).forEach((id) => { if (r.final[id] > 0) { cdf.push({ id, a: acc, b: acc + r.final[id] }); acc += r.final[id] } })
  return (
    <>
      <Card title="decoding rule · the model is unchanged, only the choice changes">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <label className="grid gap-1 text-[12px] text-muted" htmlFor="strategy">strategy
            <select id="strategy" className="h-8 w-full min-w-0 px-2" value={decode.strategy} onChange={(e) => setDecode({ ...decode, strategy: e.target.value as 'greedy' | 'sample' })}>
              <option value="greedy">greedy (argmax)</option><option value="sample">sample</option>
            </select>
          </label>
          <Slider id="temp" label={`temperature T = ${decode.temperature.toFixed(2)}`} min={0.1} max={3} step={0.05} value={decode.temperature} onChange={(v) => setDecode({ ...decode, temperature: v })} />
          <Slider id="topk" label={`top-k = ${decode.topK || 'off'}`} min={0} max={20} step={1} value={decode.topK} onChange={(v) => setDecode({ ...decode, topK: v })} />
          <Slider id="topp" label={`top-p = ${decode.topP >= 1 ? 'off' : decode.topP.toFixed(2)}`} min={0.05} max={1} step={0.05} value={decode.topP} onChange={(v) => setDecode({ ...decode, topP: v })} />
          <label className="grid gap-1 text-[12px] text-muted" htmlFor="seed">random seed
            <input id="seed" type="number" className="h-8 w-full min-w-0 px-2 font-mono" value={decode.seed} onChange={(e) => setDecode({ ...decode, seed: Number(e.target.value) || 0 })} />
          </label>
        </div>
      </Card>
      <div className="grid gap-3 xl:grid-cols-2">
        <Card title="raw p (T = 1) vs. after temperature + filtering">
          <div className="grid gap-[3px]">
            {idx.map((i) => (
              <div key={i} className="grid grid-cols-[90px_1fr_110px] items-center gap-2 font-mono text-[12px]">
                <span className={`truncate text-right ${i === r.chosen ? 'text-amber' : ''}`}>{showToken(model.vocab[i])}</span>
                <div className="relative h-[16px] rounded-sm" style={{ background: c.panel2 }}>
                  <div className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${r.raw[i] * 100}%`, background: c.lineStrong }} />
                  <div className="absolute inset-y-[4px] left-0 rounded-sm transition-[width] duration-500" style={{ width: `${r.final[i] * 100}%`, background: r.kept[i] ? (i === r.chosen ? c.amber : c.cyan) : 'transparent' }} />
                </div>
                <span className="text-right text-muted font-num">{(r.raw[i] * 100).toFixed(1)} → {r.kept[i] ? (r.final[i] * 100).toFixed(1) : '✕'}%</span>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[11px] text-faint">grey = original · bright = final distribution used for the choice · ✕ = filtered out</p>
        </Card>
        <Card title={decode.strategy === 'sample' ? `sampling: u = ${sampleU.toFixed(4)} lands on the cumulative distribution` : 'greedy: take the argmax'}>
          <svg viewBox="0 0 400 70" width="100%" role="img" aria-label="cumulative distribution">
            {cdf.slice(0, 40).map((s, k) => (
              <g key={s.id}>
                <rect x={s.a * 400} y={14} width={Math.max(0.5, (s.b - s.a) * 400 - 1)} height={26} fill={s.id === r.chosen ? c.amber : k % 2 ? c.violet : c.cyan} fillOpacity={s.id === r.chosen ? 1 : 0.55} />
                {(s.b - s.a) > 0.07 && <text x={(s.a + s.b) * 200} y={31} fontSize={10} textAnchor="middle" fill={c.bg} fontFamily="JetBrains Mono, monospace">{showToken(model.vocab[s.id])}</text>}
              </g>
            ))}
            {decode.strategy === 'sample' && <><line x1={sampleU * 400} x2={sampleU * 400} y1={6} y2={50} stroke={c.fg} strokeWidth={2} /><text x={Math.min(370, sampleU * 400 + 4)} y={62} fontSize={10} fill={c.fg} fontFamily="JetBrains Mono, monospace">u</text></>}
          </svg>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-muted">chosen:</span>
            <span className="chip" data-selected="true">{showToken(model.vocab[r.chosen])}<span className="id">{r.chosen}</span></span>
            <span className="font-mono text-[12px] text-muted">p̃ = {(r.final[r.chosen] * 100).toFixed(2)}%</span>
            <button type="button" className="btn btn-primary" onClick={() => appendToken(r.chosen)} disabled={trace.ids.length >= model.meta.ctx}>
              Append token and run again
            </button>
          </div>
          <p className="mt-2 text-[12px] text-muted">Appending feeds the new token back in: the whole forward pass above is recomputed for {trace.ids.length + 1} tokens. Each append draws a new u from the seeded generator.</p>
        </Card>
      </div>
    </>
  )
}

function Slider({ id, label, min, max, step, value, onChange }: { id: string; label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
  return (
    <label className="grid gap-1 text-[12px] text-muted font-num" htmlFor={id}>{label}
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

/* ------------------------------------------------------------- prediction */
function PredictGate({ kind, children }: { kind: 'attention' | 'next'; children: React.ReactNode }) {
  const { predictMode, prediction, setPrediction, trace, step, head, sel, model } = useLab()
  const options = useMemo(() => {
    if (kind === 'attention') return trace.tokens.slice(0, sel + 1).map((t, i) => ({ id: i, label: `${i}: ${showToken(t)}` }))
    const p = lastRow(trace)
    const top = topk(p, 10)
    const pick = [top[0], top[3] ?? top[1], top[1], top[6] ?? top[2]]
    const uniq = [...new Set(pick)]
    // deterministic shuffle so the answer isn't always first
    return uniq.sort((a, b) => ((a * 7919) % 13) - ((b * 7919) % 13)).map((id) => ({ id, label: showToken(model.vocab[id]) }))
  }, [kind, trace, sel, model])
  if (!predictMode || prediction.revealed) {
    return (
      <>
        {predictMode && prediction.revealed && <PredictFeedback kind={kind} />}
        {children}
      </>
    )
  }
  return (
    <Card title="predict before you reveal" className="border-amber/50">
      <p className="mb-3 max-w-[65ch] text-[14px]">
        {kind === 'attention'
          ? <>In layer {step.layer! + 1}, head {head + 1}, which earlier token will <b className="text-amber">“{showToken(trace.tokens[sel])}”</b> attend to most?</>
          : <>Which token will the model rate most likely to come next?</>}
      </p>
      <div className="flex flex-wrap gap-2">
        {options.map((o) => (
          <button key={o.id} type="button" className="btn" aria-pressed={prediction.choice === o.id} onClick={() => setPrediction({ ...prediction, choice: o.id })}>{o.label}</button>
        ))}
      </div>
      <button type="button" className="btn btn-primary mt-3" disabled={prediction.choice === null} onClick={() => setPrediction({ ...prediction, revealed: true })}>Reveal</button>

    </Card>
  )
}

function PredictFeedback({ kind }: { kind: 'attention' | 'next' }) {
  const { prediction, trace, step, head, sel, model } = useLab()
  if (prediction.choice === null) return null
  let ok: boolean, msg: string
  if (kind === 'attention') {
    const n = trace.ids.length
    const probs = trace.layers[step.layer!].heads[head].probs
    const w = Array.from({ length: sel + 1 }, (_, j) => probs.data[sel * n + j])
    const best = topk(w, 1)[0]
    ok = best === prediction.choice
    msg = ok ? `Correct: ${(w[best] * 100).toFixed(1)}% of the weight goes to “${showToken(trace.tokens[best])}”.`
      : `The head puts ${(w[best] * 100).toFixed(1)}% on “${showToken(trace.tokens[best])}” and only ${(w[prediction.choice] * 100).toFixed(1)}% on your choice. Heads often attend by position or token type rather than meaning: ${best === 0 ? 'parking weight on <bos> is a common default when a head has nothing specific to fetch.' : 'try other heads to see different patterns.'}`
  } else {
    const z = lastRow(trace)
    const e = Array.from(z, (v) => Math.exp(v - Math.max(...z)))
    const s = e.reduce((a, b) => a + b, 0)
    const best = topk(z, 1)[0]
    ok = best === prediction.choice
    msg = ok ? `Correct: “${showToken(model.vocab[best])}” has ${(e[best] / s * 100).toFixed(1)}%.`
      : `The model gives “${showToken(model.vocab[best])}” ${(e[best] / s * 100).toFixed(1)}% and your choice ${(e[prediction.choice] / s * 100).toFixed(2)}%. It only knows its training corpus: if a continuation never appeared there, it gets almost no probability.`
  }
  return <div className={`rounded-lg border px-3 py-2 text-[13.5px] ${ok ? 'border-ok/60 text-ok' : 'border-rose/60 text-rose'}`} role="status">{msg}</div>
}

