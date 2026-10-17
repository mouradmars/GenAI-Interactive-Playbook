import { MODULES } from './registry'

/** Placeholder for curriculum modules that are not built yet. Says so plainly. */
export function PlannedModule({ id, onOpen }: { id: string; onOpen: (id: string) => void }) {
  const i = MODULES.findIndex((m) => m.id === id)
  const m = MODULES[i]
  return (
    <div className="grid h-full place-items-center p-6">
      <div className="max-w-[60ch]">
        <div className="eyebrow">module {String(i + 1).padStart(2, '0')} · planned</div>
        <h1 className="mt-2 font-display text-[30px] leading-tight">{m.title}</h1>
        <p className="mt-3 text-[15px] leading-relaxed text-muted">{m.blurb}</p>
        <p className="mt-3 text-[14px] leading-relaxed text-muted">
          This module is not built yet. It will reuse the shared step engine, inspector, timeline and lesson format you can try now in module 01.
        </p>
        <button type="button" className="btn btn-primary mt-5" onClick={() => onOpen('llm')}>Open module 01 · Inside an LLM</button>
      </div>
    </div>
  )
}
