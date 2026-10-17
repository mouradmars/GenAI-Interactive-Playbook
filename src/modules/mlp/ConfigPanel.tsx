/** Left panel: presets, architecture builder, loss, initialization, data and optimizer. */
import { Minus, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { ACT_LABEL, DATASET_LABEL, LOSS_LABEL, actAllowed, lossCompat, paramCount, type Act, type DatasetId, type InitScheme, type LossKind, type OptKind } from '../../engine/nn'
import { ACT_CHOICES, MAX_HIDDEN_NEURONS, MAX_LAYERS, PRESETS, type PresetId } from './model'
import { useMlp } from './store'

const sel = 'w-full rounded-md border border-line bg-panel-2 px-2 py-1 text-[12.5px]'
const num = 'w-full rounded-md border border-line bg-panel-2 px-2 py-1 font-mono text-[12px]'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="grid min-w-0 gap-2 border-b border-line pb-3 last:border-b-0"><div className="eyebrow">{title}</div>{children}</section>
}

export function ConfigPanel() {
  const s = useMlp((x) => x)
  const L = s.cfg.sizes.length - 1
  const dsChoices: DatasetId[] = s.cfg.sizes[0] === 2 ? ['worked', 'xor', 'and', 'or', 'blobs', 'circles', 'moons', 'spiral', 'random'] : s.cfg.sizes[0] === 64 && s.cfg.sizes[L] === 10 ? ['digits', 'worked', 'random'] : ['worked', 'random']
  return (
    <div className="grid min-w-0 content-start gap-3 text-[12.5px]" aria-label="Network configuration">
      <Section title="preset">
        <select className={sel} value={s.preset} onChange={(e) => e.target.value !== 'custom' && s.loadPreset(e.target.value as PresetId)} aria-label="Preset">
          {s.preset === 'custom' && <option value="custom">Custom network</option>}
          {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
        <p className="text-[11.5px] leading-snug text-muted">{s.preset === 'custom' ? 'Your own architecture.' : PRESETS.find((p) => p.id === s.preset)!.blurb}</p>
      </Section>

      <Section title={`architecture · ${paramCount(s.cfg.sizes)} parameters`}>
        <div className="grid gap-1.5">
          {s.cfg.sizes.map((n, l) => (
            <div key={l} className="grid min-w-0 gap-1 rounded-lg border border-line bg-panel-2/50 p-1.5" data-testid={`layer-row-${l}`}>
              <div className="flex items-center gap-1">
                <span className="w-[58px] font-mono text-[11px] text-muted">{l === 0 ? 'input' : l === L ? 'output' : `hidden ${l}`}</span>
                <button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => s.setNeurons(l, n - 1)} disabled={n <= 1} aria-label={`Remove neuron from layer ${l}`}><Minus size={12} /></button>
                <span className="w-6 text-center font-mono" aria-label={`Layer ${l} size`}>{n}</span>
                <button type="button" className="btn icon-btn !h-6 !w-6 !p-0" onClick={() => s.setNeurons(l, n + 1)} disabled={l > 0 && l < L ? n >= MAX_HIDDEN_NEURONS : n >= 64} aria-label={`Add neuron to layer ${l}`}><Plus size={12} /></button>
                {l > 0 && l < L && <button type="button" className="btn icon-btn ml-auto !h-6 !w-6 !p-0" onClick={() => s.removeLayer(l)} aria-label={`Remove hidden layer ${l}`} title="Remove layer"><Trash2 size={12} /></button>}
              </div>
              {l > 0 && (
                <select className="w-full min-w-0 rounded border border-line bg-panel px-1 py-0.5 text-[11.5px]" value={s.cfg.acts[l - 1]} onChange={(e) => s.setActivation(l, e.target.value as Act)} aria-label={`Activation of layer ${l}`}>
                  {ACT_CHOICES.map((a) => { const ok = actAllowed(a, l === L, s.cfg.sizes[l]); return <option key={a} value={a} disabled={!ok.ok} title={ok.reason}>{ACT_LABEL[a]}</option> })}
                </select>
              )}
            </div>
          ))}
        </div>
        <button type="button" className="btn justify-center" onClick={s.addLayer} disabled={s.cfg.sizes.length >= MAX_LAYERS + 1}><Plus size={13} />Add hidden layer</button>
      </Section>

      <Section title="loss">
        <select className={sel} value={s.cfg.loss} onChange={(e) => s.setLoss(e.target.value as LossKind)} aria-label="Loss function">
          {(['mse', 'bce', 'cce'] as LossKind[]).map((l) => {
            const ok = l === 'cce' ? s.cfg.sizes[L] >= 2 : true
            return <option key={l} value={l} disabled={!ok}>{LOSS_LABEL[l]}{lossCompat(s.cfg, l).ok ? '' : ok ? ' (adjusts output)' : ' (needs ≥ 2 outputs)'}</option>
          })}
        </select>
        {s.notes.length > 0 && (
          <div className="rounded-md border border-amber/50 bg-amber/5 p-2 text-[11.5px] text-muted" role="status">
            {s.notes.map((n, i) => <div key={i}>{n}</div>)}
            <button type="button" className="mt-1 text-cyan underline decoration-dotted" onClick={s.dismissNotes}>dismiss</button>
          </div>
        )}
      </Section>

      <Section title="initialization">
        <div className="grid grid-cols-2 gap-1.5">
          <select className={sel} value={s.init.scheme} onChange={(e) => s.setInit({ scheme: e.target.value as InitScheme })} aria-label="Initialization scheme">
            <option value="xavier">Xavier uniform</option><option value="he">He normal</option><option value="uniform">Uniform ±scale</option><option value="normal">Normal(0, scale)</option>
          </select>
          <label className="flex items-center gap-1 text-muted">seed<input className={num} type="number" value={s.init.seed} min={0} onChange={(e) => s.setInit({ seed: Math.max(0, Math.round(+e.target.value || 0)) })} aria-label="Initialization seed" /></label>
        </div>
        <label className="flex items-center gap-2 text-muted">scale {s.init.scale.toFixed(2)}<input type="range" min={0.05} max={3} step={0.05} value={s.init.scale} onChange={(e) => s.setInit({ scale: +e.target.value })} className="flex-1" aria-label="Initialization scale" /></label>
        <button type="button" className="btn justify-center" onClick={s.reinit}><RefreshCw size={13} />Re-initialize weights</button>
        <p className="text-[11px] text-faint">Same seed → same weights, every time. Biases start at 0.</p>
      </Section>

      <Section title="data">
        <select className={sel} value={s.dataset} onChange={(e) => s.setDataset(e.target.value as DatasetId)} aria-label="Dataset">
          {dsChoices.map((d) => <option key={d} value={d}>{d === 'worked' ? 'Single custom sample' : DATASET_LABEL[d]}</option>)}
        </select>
        {s.dataset === 'worked' ? (
          <div className="grid gap-1">
            <div className="text-[11px] text-muted">input x</div>
            <div className="grid grid-cols-4 gap-1">{s.custom.x.slice(0, 8).map((v, i) => <input key={i} className={num} type="number" step={0.1} value={v} onChange={(e) => s.setCustom('x', i, parseFloat(e.target.value))} aria-label={`x${i + 1}`} />)}</div>
            {s.custom.x.length > 8 && <div className="text-[11px] text-faint">{s.custom.x.length - 8} more inputs use a fixed pattern.</div>}
            <div className="text-[11px] text-muted">target y</div>
            <div className="grid grid-cols-4 gap-1">{s.custom.y.slice(0, 8).map((v, i) => <input key={i} className={num} type="number" step={s.cfg.loss === 'mse' ? 0.1 : 1} min={s.cfg.loss === 'mse' ? undefined : 0} max={s.cfg.loss === 'mse' ? undefined : 1} value={v} onChange={(e) => s.setCustom('y', i, parseFloat(e.target.value))} aria-label={`y${i + 1}`} />)}</div>
          </div>
        ) : (
          <>
            <label className="flex items-center gap-2 text-muted">sample
              <input type="range" min={0} max={s.data.length - 1} value={s.sampleIdx} onChange={(e) => s.setSample(+e.target.value)} className="flex-1" aria-label="Sample shown in the microscope" />
              <span className="w-12 text-right font-mono">{s.sampleIdx + 1}/{s.data.length}</span>
            </label>
            {s.probe && <div className="flex items-center gap-2 text-[11.5px] text-cyan">microscope shows your sample <button type="button" className="underline decoration-dotted" onClick={() => s.setProbe(null)}>use dataset</button></div>}
            {s.dataset !== 'xor' && s.dataset !== 'and' && s.dataset !== 'or' && s.dataset !== 'digits' && (
              <label className="flex items-center gap-1 text-muted">data seed<input className={num} type="number" value={s.dataSeed} min={0} onChange={(e) => s.setDataSeed(Math.max(0, Math.round(+e.target.value || 0)))} aria-label="Data seed" /></label>
            )}
          </>
        )}
      </Section>

      <Section title="optimizer">
        <div className="grid grid-cols-2 gap-1.5">
          <select className={sel} value={s.opt.kind} onChange={(e) => s.setOpt({ kind: e.target.value as OptKind })} aria-label="Optimizer">
            <option value="sgd">SGD</option><option value="momentum">Momentum</option><option value="adam">Adam</option>
          </select>
          <label className="flex items-center gap-1 text-muted">batch<input className={num} type="number" min={1} max={s.data.length} value={s.batch} onChange={(e) => s.setBatch(+e.target.value || 1)} aria-label="Batch size" /></label>
        </div>
        <label className="flex items-center gap-2 text-muted">η = <span className="w-12 font-mono text-fg">{s.opt.lr}</span>
          <input type="range" min={-3} max={1} step={0.05} value={Math.log10(s.opt.lr)} onChange={(e) => s.setOpt({ lr: +(10 ** +e.target.value).toPrecision(2) })} className="flex-1" aria-label="Learning rate (log scale)" />
        </label>
        {s.opt.kind === 'momentum' && <label className="flex items-center gap-2 text-muted">μ {s.opt.momentum}<input type="range" min={0} max={0.99} step={0.01} value={s.opt.momentum} onChange={(e) => s.setOpt({ momentum: +e.target.value })} className="flex-1" aria-label="Momentum" /></label>}
      </Section>
    </div>
  )
}
