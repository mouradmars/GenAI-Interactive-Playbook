/** Side-by-side experiment: two prompts and/or decoding configurations through the same real model. */
import { useMemo, useState } from 'react'
import { Card, ProbBars } from '../../components/viz'
import { showToken } from '../../engine/model'
import { defaultDecode, entropy, generate, type DecodeConfig } from '../../engine/decoding'
import { topk } from '../../engine/tensor'
import { useLab } from './context'

interface Cfg { prompt: string; decode: DecodeConfig }

export function ComparePanel() {
  const { prompt } = useLab()
  const [a, setA] = useState<Cfg>({ prompt, decode: { ...defaultDecode, strategy: 'greedy' } })
  const [b, setB] = useState<Cfg>({ prompt, decode: { ...defaultDecode, temperature: 1.6, seed: 3 } })
  return (
    <div className="grid gap-3">
      <p className="max-w-[75ch] text-[13.5px] text-muted">Run the same model twice. Change the prompt or the decoding rule on either side and every result below is recomputed, including 10 generated tokens per side.</p>
      <div className="grid gap-3 lg:grid-cols-2">
        <Side label="A" cfg={a} setCfg={setA} />
        <Side label="B" cfg={b} setCfg={setB} />
      </div>
    </div>
  )
}

function Side({ label, cfg, setCfg }: { label: string; cfg: Cfg; setCfg: (c: Cfg) => void }) {
  const { model } = useLab()
  const run = useMemo(() => {
    const enc = model.encodePrompt(cfg.prompt)
    const g = generate(model, enc.ids, Math.min(10, model.meta.ctx - enc.ids.length), cfg.decode)
    return { enc, g }
  }, [model, cfg])
  const first = run.g.steps[0]
  const d = cfg.decode
  const set = (p: Partial<DecodeConfig>) => setCfg({ ...cfg, decode: { ...d, ...p } })
  const avgH = run.g.steps.reduce((s, x) => s + entropy(x.final), 0) / Math.max(1, run.g.steps.length)
  const logp = run.g.steps.reduce((s, x) => s + Math.log(x.raw[x.chosen]), 0)
  return (
    <Card title={`configuration ${label}`}>
      <div className="grid gap-2">
        <label className="grid gap-1 text-[12px] text-muted" htmlFor={`cmp-prompt-${label}`}>prompt
          <input id={`cmp-prompt-${label}`} className="h-9 px-2 font-mono text-[13px]" value={cfg.prompt} onChange={(e) => setCfg({ ...cfg, prompt: e.target.value })} />
        </label>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <label className="grid gap-1 text-[11.5px] text-muted" htmlFor={`cmp-s-${label}`}>strategy
            <select id={`cmp-s-${label}`} className="h-8 w-full min-w-0 px-1" value={d.strategy} onChange={(e) => set({ strategy: e.target.value as DecodeConfig['strategy'] })}>
              <option value="greedy">greedy</option><option value="sample">sample</option>
            </select>
          </label>
          <label className="grid gap-1 text-[11.5px] text-muted font-num" htmlFor={`cmp-t-${label}`}>T = {d.temperature.toFixed(2)}
            <input id={`cmp-t-${label}`} type="range" className="w-full min-w-0" min={0.1} max={3} step={0.05} value={d.temperature} onChange={(e) => set({ temperature: +e.target.value })} />
          </label>
          <label className="grid gap-1 text-[11.5px] text-muted font-num" htmlFor={`cmp-k-${label}`}>top-k = {d.topK || 'off'}
            <input id={`cmp-k-${label}`} type="range" className="w-full min-w-0" min={0} max={20} step={1} value={d.topK} onChange={(e) => set({ topK: +e.target.value })} />
          </label>
          <label className="grid gap-1 text-[11.5px] text-muted" htmlFor={`cmp-seed-${label}`}>seed
            <input id={`cmp-seed-${label}`} type="number" className="h-8 w-full min-w-0 px-1 font-mono" value={d.seed} onChange={(e) => set({ seed: +e.target.value || 0 })} />
          </label>
        </div>
      </div>
      <div className="mt-3 rounded-lg border border-line bg-panel-2 p-3 font-mono text-[14px] leading-relaxed">
        <span className="text-muted">{cfg.prompt.toLowerCase()}</span>
        {run.g.steps.map((s, i) => (
          <span key={i} className="text-teal" title={`p̃ = ${(s.final[s.chosen] * 100).toFixed(1)}%`}>{model.vocab[s.chosen].replace(/▁/g, ' ')}</span>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {run.g.steps.map((s, i) => (
          <span key={i} className="rounded border border-line px-1.5 py-0.5 font-mono text-[10.5px] text-muted font-num">{showToken(model.vocab[s.chosen])} {(s.final[s.chosen] * 100).toFixed(0)}%</span>
        ))}
      </div>
      {first && (
        <div className="mt-3">
          <div className="eyebrow mb-1">first-step distribution p̃</div>
          <ProbBars items={topk(first.final, 6).map((i) => ({ label: showToken(model.vocab[i]), value: first.final[i] }))} max={1} />
        </div>
      )}
      <dl className="mt-3 grid grid-cols-2 gap-1 font-mono text-[12px] font-num">
        <dt className="text-muted">mean entropy of p̃</dt><dd className="text-right">{avgH.toFixed(3)} bits</dd>
        <dt className="text-muted">log-prob under the model (T=1)</dt><dd className="text-right">{logp.toFixed(3)}</dd>
      </dl>
    </Card>
  )
}
