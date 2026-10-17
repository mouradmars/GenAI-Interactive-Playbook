/**
 * Real-model inspection with Observatory-Nano, the 2-layer, 4-head GPT trained in this repository (module 01).
 * Every value comes from its actual weights via engine/model.ts, whose forward pass is tested against PyTorch.
 */
import { useMemo, useState } from 'react'
import { useColors } from '../../sim/appState'
import { getModel, showToken } from '../../engine/model'
import { row as matRow, softmax, topk, type Mat } from '../../engine/tensor'
import { vnorm } from '../../engine/block'
import { Card } from '../../components/viz'
import { MatrixView, f3, useRoles } from './common'

const toM = (m: Mat) => Array.from({ length: m.rows }, (_, i) => Array.from(matRow(m, i)))
const EXAMPLES = ['the dog likes to eat', 'the cat sat in the', 'paris is the capital of', 'one plus two is']

export function RealModel() {
  const c = useColors(), r = useRoles()
  const model = getModel()
  const [text, setText] = useState('the dog likes to eat')
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [tok, setTok] = useState<number | null>(null)
  const enc = useMemo(() => model.encodePrompt(text || 'the'), [model, text])
  const trace = useMemo(() => model.forward(enc.ids), [model, enc.ids])
  const tokens = trace.tokens.map(showToken)
  const n = tokens.length, t = Math.min(tok ?? n - 1, n - 1)
  const L = trace.layers[layer], A = toM(L.heads[head].probs)
  const states = [trace.x0, ...trace.layers.map((l) => l.xOut)]
  const stateNames = ['embedding + position', ...trace.layers.map((_, k) => `after block ${k + 1}`)]
  const logits = Array.from(matRow(trace.logits, n - 1))
  const probs = softmax(logits)
  const top = topk(probs, 8)
  const m = model.meta
  return (
    <div className="grid gap-3">
      <Card title="the model">
        <p className="text-[12.5px] leading-relaxed text-muted">
          <b className="text-fg">{m.name}</b>: a decoder-only, pre-LN GPT with {m.nLayers} blocks, {m.nHeads} heads, d = {m.dModel}, d_ff = {m.dMlp}, {m.vocabSize}-token BPE vocabulary, {m.params.toLocaleString()} parameters, trained here for {m.trainSteps.toLocaleString()} steps ({m.optimizer}) on a {m.corpusSentences.toLocaleString()}-sentence toy corpus (animals and food, capitals, colours, counting, places). The browser forward pass reproduces PyTorch to 4 decimals (module 01 tests). It is small and knows only its corpus, but every number below is real: nothing is invented or randomized.
        </p>
      </Card>
      <Card title="input text">
        <div className="flex flex-wrap items-center gap-2">
          <input value={text} onChange={(e) => { setText(e.target.value.toLowerCase().replace(/[^a-z .]/g, '').slice(0, 90)); setTok(null) }} className="min-w-[220px] flex-1 rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-[13px]" aria-label="Real model input" data-testid="real-input" />
          {EXAMPLES.map((e) => <button key={e} type="button" className="btn !py-0.5 text-[11.5px]" onClick={() => { setText(e); setTok(null) }}>{e}</button>)}
        </div>
        <div className="mt-2 flex flex-wrap gap-1" data-testid="real-tokens">
          {tokens.map((x, i) => <button key={i} type="button" onClick={() => setTok(i)} className={`rounded-md border px-2 py-0.5 font-mono text-[12px] ${i === t ? 'border-cyan text-cyan' : 'border-line'}`} title={`id ${trace.ids[i]}`}>{x}</button>)}
        </div>
        <p className="mt-1 text-[11.5px] text-faint">{n} tokens (a &lt;bos&gt; token is prepended, as in training). Click a token to inspect it.{enc.truncated ? ' Truncated to the 32-token context.' : ''}</p>
      </Card>
      <div className="grid gap-3 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card title={`actual attention · block ${layer + 1}, head ${head + 1}`} right={
          <div className="flex flex-wrap gap-1">
            {trace.layers.map((_, k) => <button key={k} type="button" className={`btn !py-0.5 text-[11.5px] ${k === layer ? 'active' : ''}`} onClick={() => setLayer(k)}>block {k + 1}</button>)}
            {L.heads.map((_, k) => <button key={k} type="button" className={`btn !py-0.5 text-[11.5px] ${k === head ? 'active' : ''}`} onClick={() => setHead(k)} aria-label={`Real head ${k + 1}`}>h{k + 1}</button>)}
          </div>}>
          <MatrixView m={A} mode="prob" maxAbs={1} rowLabels={tokens} colLabels={tokens.map((x) => x.slice(0, 5))} color={r.score} ariaLabel="Real attention weights" testid="real-A" cell={n > 9 ? 34 : 42} digits={2} hlRows={[t]} onCell={(i) => setTok(i)} />
          <p className="mt-1 text-[11.5px] text-muted">Causal, rows sum to 1. Attention shows where this head reads from for each query; it does not by itself explain the model's prediction.</p>
        </Card>
        <Card title={`hidden state of “${tokens[t]}” (${m.dModel} features) through the model`}>
          <MatrixView m={states.map((s) => Array.from(matRow(s, t)))} rowLabels={['emb+pos', ...trace.layers.map((_, k) => `block ${k + 1}`)]} colLabels={undefined} color={r.tok} ariaLabel="Real hidden states" testid="real-h" cell={9} showValues={false} />
          <table className="mt-2 font-mono text-[11.5px]"><tbody>{states.map((s, k) => <tr key={k} className="border-t border-line"><td className="pr-3 text-muted">{stateNames[k]}</td><td className="text-right">‖h‖ = {f3(vnorm(Array.from(matRow(s, t))))}</td></tr>)}</tbody></table>
          <p className="mt-1 text-[11.5px] text-faint">Hover a cell for the exact value. These are hidden states, not word meanings: individual features are not directly interpretable.</p>
        </Card>
      </div>
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Card title={`what block ${layer + 1} writes into the residual stream, per token`}>
          <table className="w-full font-mono text-[11.5px]" data-testid="real-contrib">
            <thead className="text-faint"><tr><th className="px-1 text-left">token</th><th className="px-1 text-right">‖input‖</th><th className="px-1 text-right">‖attention update‖</th><th className="px-1 text-right">‖MLP update‖</th><th className="px-1 text-right">‖output‖</th></tr></thead>
            <tbody>{tokens.map((x, i) => <tr key={i} className={`border-t border-line ${i === t ? 'text-cyan' : ''}`}><td className="px-1">{x}</td><td className="px-1 text-right">{f3(vnorm(Array.from(matRow(L.xIn, i))))}</td><td className="px-1 text-right" style={{ color: r.score }}>{f3(vnorm(Array.from(matRow(L.attnOut, i))))}</td><td className="px-1 text-right" style={{ color: r.ffn }}>{f3(vnorm(Array.from(matRow(L.mlpOut, i))))}</td><td className="px-1 text-right">{f3(vnorm(Array.from(matRow(L.xOut, i))))}</td></tr>)}</tbody>
          </table>
          <p className="mt-1 text-[11.5px] text-muted">Same structure as the toy: x_out = x_in + attention update + MLP update (pre-LN).</p>
        </Card>
        <Card title="next-token prediction (after the complete forward pass)">
          <div className="mb-2 flex flex-wrap items-center gap-1 font-mono text-[11px] text-muted">
            {['block 2 output', 'final LayerNorm', `LM head (${m.dModel} → ${m.vocabSize})`, 'softmax'].map((x, k) => <span key={x} className="flex items-center gap-1"><span className="rounded border border-line px-1.5 py-0.5">{x}</span>{k < 3 && '→'}</span>)}
          </div>
          <div className="grid gap-1" data-testid="real-next">
            {top.map((id) => (
              <div key={id} className="grid grid-cols-[90px_1fr_60px] items-center gap-2 text-[12px]">
                <span className="truncate font-mono">{showToken(model.vocab[id])}</span>
                <div className="h-3 rounded bg-panel-2"><div className="h-3 rounded" style={{ width: `${probs[id] * 100}%`, background: c.cyan }} /></div>
                <span className="text-right font-mono">{(probs[id] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <p className="mt-1 text-[11.5px] text-muted">The block output is a hidden state; only after the final LayerNorm, the unembedding and a softmax does it become a distribution over the {m.vocabSize} tokens.</p>
        </Card>
      </div>
    </div>
  )
}
