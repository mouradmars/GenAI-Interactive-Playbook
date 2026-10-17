/** Left · attention pipeline navigator: every operation selectable, with its output shape. */
import type { AttnTrace } from '../../engine/attention'
import { STAGES, type Stage } from './stages'

export function shapeAfter(id: Stage['id'], t: AttnTrace): string {
  const { Tq: n, Tk: m, dModel: d, dk, cfg } = t
  switch (id) {
    case 'tokens': return `${n} tokens`
    case 'embed': return `${n}×${d}`
    case 'qkv': return `3 × ${n}×${dk} / head`
    case 'dot': case 'scale': case 'mask': case 'softmax': case 'heatmap': return `${n}×${m}`
    case 'aggregate': return `${n}×${dk}`
    case 'heads': return `${cfg.H} × ${n}×${dk}`
    case 'concat': return `${n}×${cfg.H * dk}`
    case 'proj': case 'output': return `${n}×${d}`
  }
}

export function Navigator({ index, onSeek, trace }: { index: number; onSeek: (i: number) => void; trace: AttnTrace }) {
  return (
    <nav aria-label="Attention pipeline navigator" className="rounded-xl border border-line bg-panel/60 p-2" data-testid="attn-navigator">
      <div className="eyebrow px-1 pb-1">attention pipeline</div>
      <ol className="flex gap-1 overflow-x-auto pb-1 scroll-thin @5xl:grid @5xl:gap-0 @5xl:overflow-visible">
        {STAGES.map((s, k) => {
          const on = k === index, past = k < index
          return (
            <li key={s.id} className="min-w-fit @5xl:min-w-0">
              <button type="button" onClick={() => onSeek(k)} aria-current={on ? 'step' : undefined} data-testid={`navstep-${s.id}`}
                className={`grid w-full grid-cols-[22px_1fr] items-baseline gap-1 rounded-lg border px-2 py-1.5 text-left transition-colors ${on ? 'border-cyan bg-cyan/10 shadow-[var(--glow)]' : 'border-transparent hover:bg-panel-2'}`}>
                <span className={`font-mono text-[10.5px] ${on ? 'text-cyan' : past ? 'text-fg' : 'text-faint'}`}>{String(k + 1).padStart(2, '0')}</span>
                <span className="min-w-0">
                  <span className={`block whitespace-nowrap text-[12.5px] leading-tight @5xl:whitespace-normal ${on ? 'text-fg' : 'text-muted'}`}>{s.nav}</span>
                  <span className="block font-mono text-[10px] text-faint">{shapeAfter(s.id, trace)}</span>
                </span>
              </button>
              {k < STAGES.length - 1 && <div aria-hidden="true" className="hidden pl-[17px] font-mono text-[9px] leading-[9px] text-faint @5xl:block">↓</div>}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
