/** Module 1 · Inside an LLM — the complete token journey through Observatory-Nano. */
import { Box, Download, GraduationCap, Microscope, SplitSquareHorizontal } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Timeline } from '../../components/Timeline'
import { useHotkeys, Workspace } from '../../components/Shell'
import { ExportDialog, type ExportPayload } from '../../components/ExportDialog'
import { useApp } from '../../sim/appState'
import { useStepEngine } from '../../sim/stepEngine'
import { getModel, showToken } from '../../engine/model'
import { decodeStep, defaultDecode, entropy, rng, type DecodeConfig } from '../../engine/decoding'
import { row, topk } from '../../engine/tensor'
import { buildJourney } from './journey'
import { Lab, lastRow, type LabCtx, type Prediction } from './context'
import { StageView } from './StageViews'
import { Inspector, liveFacts } from './Inspector'
import { ComparePanel } from './ComparePanel'
import { LessonPanel } from './LessonPanel'
import { STEP_LESSONS } from './lesson'

const ResidualTower = lazy(() => import('../../components/three').then((m) => ({ default: m.ResidualTower })))

type Tab = 'lab' | 'overview' | 'compare' | 'lesson'

export default function InsideLLM() {
  const model = getModel()
  const steps = useMemo(() => buildJourney(model.meta.nLayers), [model])
  const engine = useStepEngine(steps.length)
  const { depth } = useApp()
  const [tab, setTab] = useState<Tab>('lab')
  const [prompt, setPromptRaw] = useState('the dog likes to eat')
  const [generated, setGenerated] = useState<number[]>([])
  const [selRaw, setSel] = useState<number | null>(null)
  const [head, setHead] = useState(0)
  const [cell, setCell] = useState<[number, number] | null>(null)
  const [decode, setDecode] = useState<DecodeConfig>(defaultDecode)
  const [predictMode, setPredictMode] = useState(false)
  const [predictions, setPredictions] = useState<Record<string, Prediction>>({})
  const [exportOpen, setExportOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement>(null)

  const enc = useMemo(() => model.encodePrompt(prompt), [model, prompt])
  const ids = useMemo(() => [...enc.ids, ...generated].slice(-model.meta.ctx), [enc, generated, model])
  const trace = useMemo(() => model.forward(ids), [model, ids])
  const sel = selRaw === null || selRaw >= ids.length ? ids.length - 1 : selRaw
  const sampleU = useMemo(() => { const r = rng(decode.seed); let u = 0; for (let k = 0; k <= generated.length; k++) u = r(); return u }, [decode.seed, generated.length])
  const decodeResult = useMemo(() => decodeStep(lastRow(trace), decode, sampleU), [trace, decode, sampleU])
  const step = steps[engine.index]

  const resetContext = () => { setSel(null); setCell(null); setPredictions({}) }
  const setPrompt = useCallback((p: string) => { setPromptRaw(p); setGenerated([]); resetContext() }, [])
  const appendToken = (id: number) => { setGenerated((g) => [...g, id]); resetContext() }
  const undoToken = () => { setGenerated((g) => g.slice(0, -1)); resetContext() }
  useEffect(() => setCell(null), [head, engine.index])

  const ctx: LabCtx = {
    model, trace, step, prompt, setPrompt, generated, appendToken, undoToken, sel, setSel, head, setHead, cell, setCell,
    decode, setDecode, decodeResult, sampleU, predictMode, unknown: enc.unknown,
    prediction: predictions[`${step.id}-${head}-${sel}`] ?? { choice: null, revealed: false },
    setPrediction: (p) => setPredictions((all) => ({ ...all, [`${step.id}-${head}-${sel}`]: p })),
  }

  const goTo = (i: number, p?: string) => { if (p && p !== prompt) setPrompt(p); setTab('lab'); engine.seek(i) }

  useHotkeys({
    ArrowRight: engine.next, ArrowLeft: engine.prev, Space: engine.toggle, Home: engine.reset,
    p: () => setPredictMode((v) => !v), e: () => setExportOpen(true), g: () => setTab((t) => (t === 'overview' ? 'lab' : 'overview')),
  }, [engine.next, engine.prev, engine.toggle, engine.reset])

  const payload = (): ExportPayload => {
    const { facts, tensor } = liveFacts(ctx)
    const p = decodeResult.raw
    const top = topk(p, 8)
    const json = {
      observatory: 'AI Deep Learning Observatory', module: 'Inside an LLM', model: model.meta, exportedAt: new Date().toISOString(),
      prompt, generatedTokens: generated.map((g) => model.vocab[g]), tokens: trace.tokens, ids: trace.ids,
      step: { index: engine.index, ...step }, depth, selectedToken: sel, head, decode,
      facts, tensor: tensor ? { name: tensor.name, shape: [tensor.m.rows, tensor.m.cols], rows: Array.from({ length: tensor.m.rows }, (_, i) => Array.from(row(tensor.m, i), (v) => +v.toFixed(5))) } : null,
      nextToken: top.map((i) => ({ token: model.vocab[i], p: +p[i].toFixed(6) })),
    }
    const d = { n: trace.ids.length, D: model.meta.dModel, H: model.meta.nHeads, HD: model.headDim, M: model.meta.dMlp, V: model.meta.vocabSize, layer: step.layer ?? 0, head }
    const md = [
      `# Inside an LLM · experiment record`,
      ``,
      `- Model: ${model.meta.name} (${model.meta.nLayers} layers, ${model.meta.nHeads} heads, d_model ${model.meta.dModel}, ${model.meta.params.toLocaleString()} parameters)`,
      `- Prompt: "${prompt}"${generated.length ? ` + generated "${generated.map((g) => model.vocab[g].replace(/▁/g, ' ')).join('')}"` : ''}`,
      `- Tokens (${trace.ids.length}): ${trace.tokens.map(showToken).join(' | ')}`,
      `- Stage: ${engine.index + 1}/${steps.length} · ${step.group} · ${step.title}`,
      `- Decoding: ${decode.strategy}, T=${decode.temperature}, top-k=${decode.topK || 'off'}, top-p=${decode.topP >= 1 ? 'off' : decode.topP}, seed=${decode.seed}`,
      ``,
      `## What happens at this stage`,
      STEP_LESSONS[step.kind][depth](d),
      ``,
      `## Live values`,
      ...facts.map((f) => `- ${f.label}: ${f.value}${f.ok === true ? ' ✓' : ''}`),
      ``,
      `## Next-token distribution (top 8)`,
      `| token | probability |`, `|---|---|`,
      ...top.map((i) => `| ${showToken(model.vocab[i])} | ${(p[i] * 100).toFixed(2)}% |`),
      ``,
      `Entropy: ${entropy(p).toFixed(3)} bits. All values computed in the browser from the trained weights.`,
    ].join('\n')
    return { json, markdown: md, filenameBase: `observatory-${step.id}-${prompt.replace(/\W+/g, '-').slice(0, 30)}` }
  }

  const depthIndex = step.layer === undefined ? (step.group === 'Output' ? model.meta.nLayers * 2 : 0)
    : ['ln1', 'qkv', 'scores', 'softmax'].includes(step.kind) ? step.layer * 2 : step.kind === 'mix' ? step.layer * 2 + 1 : step.layer * 2 + 2

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'lab', label: 'Lab', icon: <Microscope size={15} /> },
    { id: 'overview', label: '3D overview', icon: <Box size={15} /> },
    { id: 'compare', label: 'Compare', icon: <SplitSquareHorizontal size={15} /> },
    { id: 'lesson', label: 'Lesson', icon: <GraduationCap size={15} /> },
  ]

  return (
    <Lab.Provider value={ctx}>
      <Workspace
        stage={
          <div className="grid gap-3 p-3 md:p-4 [&>*]:min-w-0" ref={stageRef} id="stage">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="eyebrow">module 01 · inside an LLM</span>
                  <span className="sim-badge" title="Every value is computed from trained weights">real model · {model.meta.name}</span>
                </div>
                <h1 className="stage-title mt-1 font-display text-[26px] leading-tight">
                  {tab === 'lab' ? step.title : tab === 'overview' ? 'The residual stream in 3D' : tab === 'compare' ? 'Compare two configurations' : 'Lesson guide'}
                </h1>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <div className="flex rounded-lg border border-line p-0.5" role="tablist" aria-label="Module views">
                  {tabs.map((t) => (
                    <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] ${tab === t.id ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{t.icon}{t.label}</button>
                  ))}
                </div>
                <button type="button" className="btn" onClick={() => setExportOpen(true)} title="Export (E)"><Download size={15} />Export</button>
              </div>
            </div>
            {tab === 'lab' && <StageView />}
            {tab === 'overview' && (
              <div className="grid gap-3">
                <div className="h-[min(62vh,560px)] min-h-[340px] overflow-hidden rounded-xl border border-line bg-panel/60">
                  <Suspense fallback={<div className="grid h-full place-items-center text-muted">Loading 3D view…</div>}>
                    <ResidualTower trace={trace} depthIndex={depthIndex} layer={step.layer ?? 0} head={head} selected={sel} />
                  </Suspense>
                </div>
                <p className="max-w-[75ch] text-[13px] text-muted">
                  Each column is one token; each row of boxes is the residual stream at one depth. The highlighted row follows the timeline. Arcs show where attention in
                  layer {(step.layer ?? 0) + 1}, head {head + 1} moved information. Step through the timeline below to watch the stream deepen.
                </p>
              </div>
            )}
            {tab === 'compare' && <ComparePanel />}
            {tab === 'lesson' && <LessonPanel steps={steps} goTo={goTo} />}
          </div>
        }
        inspector={<Inspector />}
        timeline={
          <Timeline engine={engine} steps={steps}
            extra={
              <label className="flex items-center gap-2 text-[12px] text-muted" title="Predict before reveal (P)">
                <input type="checkbox" checked={predictMode} onChange={(e) => setPredictMode(e.target.checked)} id="predict-mode" />
                predict before reveal
              </label>
            } />
        }
      />
      <ExportDialog open={exportOpen} onClose={() => setExportOpen(false)} payload={payload} stageEl={stageRef.current} />
    </Lab.Provider>
  )
}
