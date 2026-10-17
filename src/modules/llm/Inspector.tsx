/** Right-hand mathematical inspector: equation, tensor shapes, live checks, explanation for the chosen depth. */
import { Tex, fmt } from '../../components/viz'
import { useApp } from '../../sim/appState'
import { showToken, type ForwardTrace } from '../../engine/model'
import { row, stats, type Mat } from '../../engine/tensor'
import { entropy } from '../../engine/decoding'
import { STEP_LESSONS, type Dims } from './lesson'
import { lastRow, useLab, type LabCtx } from './context'

export interface Fact { label: string; value: string; ok?: boolean }

/** Live, recomputed facts for the current step (also exported in snapshots). */
export function liveFacts(ctx: Pick<LabCtx, 'trace' | 'step' | 'sel' | 'head' | 'decodeResult' | 'model'>): { facts: Fact[]; tensor?: { name: string; m: Mat } } {
  const { trace, step, sel, head, decodeResult, model } = ctx
  const n = trace.ids.length
  const L = step.layer !== undefined ? trace.layers[step.layer] : undefined
  const tok = `“${showToken(trace.tokens[sel])}”`
  switch (step.kind) {
    case 'text': case 'bpe': case 'ids':
      return { facts: [{ label: 'tokens (incl. <bos>)', value: String(n) }, { label: 'ids', value: trace.ids.join(', ') }] }
    case 'embed':
      return { facts: [{ label: `‖W_E[${trace.ids[sel]}]‖`, value: fmt(Math.hypot(...row(trace.tokEmb, sel))) }], tensor: { name: 'token embeddings', m: trace.tokEmb } }
    case 'pos':
      return { facts: [{ label: `‖x⁽⁰⁾_${sel}‖`, value: fmt(Math.hypot(...row(trace.x0, sel))) }], tensor: { name: 'x0', m: trace.x0 } }
    case 'ln1': case 'lnf': {
      const ln = step.kind === 'ln1' ? L!.ln1 : trace.lnf
      const x = step.kind === 'ln1' ? L!.xIn : trace.layers[trace.layers.length - 1].xOut
      const z = row(x, sel).map((v) => (v - ln.means[sel]) / ln.stds[sel])
      const s = stats(z)
      return {
        facts: [
          { label: `μ of ${tok}`, value: fmt(ln.means[sel], 4) }, { label: 'σ', value: fmt(ln.stds[sel], 4) },
          { label: 'mean of (x−μ)/σ', value: fmt(s.mean, 6), ok: Math.abs(s.mean) < 1e-4 },
          { label: 'std of (x−μ)/σ', value: fmt(s.std, 6), ok: Math.abs(s.std - 1) < 1e-3 },
        ],
        tensor: { name: 'LN output', m: ln.out },
      }
    }
    case 'qkv':
      return { facts: [{ label: '‖q‖ / ‖k‖ / ‖v‖', value: [L!.q, L!.k, L!.v].map((m) => fmt(Math.hypot(...row(m, sel)), 2)).join(' / ') }], tensor: { name: 'Q', m: L!.q } }
    case 'scores': {
      const H = L!.heads[head]
      const r = Array.from({ length: n }, (_, j) => H.scores.data[sel * n + j])
      return {
        facts: [
          { label: `scores row ${sel}`, value: r.filter(Number.isFinite).map((v) => fmt(v, 2)).join(' ') },
          { label: 'masked (j > i)', value: `${(n * (n - 1)) / 2} cells = −∞`, ok: true },
        ],
        tensor: { name: `scores L${step.layer! + 1}H${head + 1}`, m: H.scores },
      }
    }
    case 'softmax': {
      const H = L!.heads[head]
      let maxDev = 0, leak = 0
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) { s += H.probs.data[i * n + j]; if (j > i) leak = Math.max(leak, H.probs.data[i * n + j]) }
        maxDev = Math.max(maxDev, Math.abs(s - 1))
      }
      return {
        facts: [
          { label: 'max |Σⱼ Aᵢⱼ − 1|', value: maxDev.toExponential(1), ok: maxDev < 1e-5 },
          { label: 'max future weight', value: String(leak), ok: leak === 0 },
          { label: `weight on <bos> for ${tok}`, value: `${(H.probs.data[sel * n] * 100).toFixed(1)}%` },
        ],
        tensor: { name: `attention L${step.layer! + 1}H${head + 1}`, m: H.probs },
      }
    }
    case 'mix':
      return {
        facts: [
          { label: '‖attention update‖', value: fmt(Math.hypot(...row(L!.attnOut, sel)), 3) },
          { label: '‖x‖ → ‖x′‖', value: `${fmt(Math.hypot(...row(L!.xIn, sel)), 2)} → ${fmt(Math.hypot(...row(L!.xMid, sel)), 2)}` },
        ],
        tensor: { name: 'attention output', m: L!.attnOut },
      }
    case 'mlp': {
      const a = row(L!.act, sel)
      return {
        facts: [
          { label: 'neurons > 0.1', value: `${a.filter((v) => v > 0.1).length} / ${a.length}` },
          { label: 'strongest neuron', value: (() => { let b = 0; a.forEach((v, i) => { if (v > a[b]) b = i }); return `#${b} = ${fmt(a[b], 3)}` })() },
          { label: '‖MLP update‖', value: fmt(Math.hypot(...row(L!.mlpOut, sel)), 3) },
        ],
        tensor: { name: 'MLP hidden (post-GELU)', m: L!.act },
      }
    }
    case 'residual':
      return { facts: [{ label: `‖x‖ after block ${step.layer! + 1}`, value: fmt(Math.hypot(...row(L!.xOut, n - 1)), 3) }], tensor: { name: 'residual', m: L!.xOut } }
    case 'logits': {
      const s = stats(lastRow(trace))
      return { facts: [{ label: 'max / mean / min logit', value: `${fmt(s.max, 2)} / ${fmt(s.mean, 2)} / ${fmt(s.min, 2)}` }], tensor: { name: 'logits', m: trace.logits } }
    }
    case 'probs':
    case 'decode': {
      const r = decodeResult
      const sum = r.raw.reduce((a, b) => a + b, 0)
      const sumF = r.final.reduce((a, b) => a + b, 0)
      return {
        facts: [
          { label: 'Σ p', value: sum.toFixed(8), ok: Math.abs(sum - 1) < 1e-9 },
          { label: 'Σ p̃ (after filtering)', value: sumF.toFixed(8), ok: Math.abs(sumF - 1) < 1e-9 },
          { label: 'entropy H(p)', value: `${entropy(r.raw).toFixed(3)} bits` },
          { label: 'entropy H(p̃)', value: `${entropy(r.final).toFixed(3)} bits` },
          { label: 'tokens kept', value: String(r.kept.filter(Boolean).length) },
        ].concat(step.kind === 'decode' ? [{ label: 'chosen', value: `${showToken(model.vocab[r.chosen])} (id ${r.chosen})` }] : []),
      }
    }
  }
}

export function dimsFor(trace: ForwardTrace, ctx: Pick<LabCtx, 'model' | 'head' | 'step'>): Dims {
  const m = ctx.model.meta
  return { n: trace.ids.length, D: m.dModel, H: m.nHeads, HD: m.dModel / m.nHeads, M: m.dMlp, V: m.vocabSize, layer: ctx.step.layer ?? 0, head: ctx.head }
}

export function Inspector() {
  const ctx = useLab()
  const { depth } = useApp()
  const { step, trace, sel } = ctx
  const lesson = STEP_LESSONS[step.kind]
  const d = dimsFor(trace, ctx)
  const { facts, tensor } = liveFacts(ctx)
  return (
    <div className="grid gap-4" aria-live="polite">
      <div>
        <div className="eyebrow">{step.group}{step.layer !== undefined ? '' : ''}</div>
        <h2 className="mt-1 font-display text-[22px] leading-tight">{step.title}</h2>
      </div>
      <div className="overflow-x-auto rounded-lg border border-line bg-panel-2 px-3 py-3 scroll-thin"><Tex tex={lesson.equation(d)} block /></div>
      <p className="max-w-[60ch] text-[13.5px] leading-relaxed text-fg/90">{lesson[depth](d)}</p>
      {depth !== 'beginner' && (
        <section>
          <div className="eyebrow mb-1.5">tensor shapes</div>
          <table className="w-full font-mono text-[12px] font-num">
            <tbody>
              {lesson.shapes(d).map(([k, v]) => (
                <tr key={k} className="border-b border-line/60"><td className="py-1 pr-2 text-muted">{k}</td><td className="py-1 text-right text-cyan">{v}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <section>
        <div className="eyebrow mb-1.5">live values · computed now</div>
        <dl className="grid gap-1 font-mono text-[12px] font-num">
          {facts.map((f) => (
            <div key={f.label} className="flex items-baseline justify-between gap-3 border-b border-line/60 py-1">
              <dt className="text-muted">{f.label}</dt>
              <dd className={`min-w-0 break-all text-right ${f.ok === true ? 'text-ok' : f.ok === false ? 'text-rose' : 'text-fg'}`}>{f.value}{f.ok === true ? ' ✓' : f.ok === false ? ' ✗' : ''}</dd>
            </div>
          ))}
        </dl>
      </section>
      {depth === 'advanced' && tensor && tensor.m.rows > sel && (
        <section>
          <div className="eyebrow mb-1.5">{tensor.name}[{sel}, :8] · shape [{tensor.m.rows}, {tensor.m.cols}]</div>
          <div className="grid grid-cols-4 gap-1 font-mono text-[11px] font-num">
            {Array.from(row(tensor.m, sel).slice(0, 8)).map((v, i) => (
              <span key={i} className="rounded bg-panel-2 px-1.5 py-1 text-right">{fmt(v, 3)}</span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
