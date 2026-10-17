/** Application shell: header, collapsible module rail, shared workspace layout, help overlay, global shortcuts. */
import { HelpCircle, Maximize2, Minimize2, Moon, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Sun, X } from 'lucide-react'
import { Fragment, useEffect, type ReactNode } from 'react'
import { useApp, type Depth } from '../sim/appState'
import { MODULES } from '../modules/registry'

const isTyping = (e: KeyboardEvent) => {
  const t = e.target as HTMLElement | null
  return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)
}

export function useHotkeys(map: Record<string, () => void>, deps: unknown[] = []) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return
      const k = e.key === ' ' ? 'Space' : e.key
      const f = map[k]
      if (f) { e.preventDefault(); f() }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
      <circle cx="13" cy="13" r="11.5" fill="none" stroke="var(--line-strong)" />
      <circle cx="13" cy="13" r="7" fill="none" stroke="var(--violet)" strokeDasharray="2 2" />
      <circle cx="13" cy="13" r="2.6" fill="var(--cyan)" />
      <path d="M13 1.5v4M13 20.5v4M1.5 13h4M20.5 13h4" stroke="var(--amber)" strokeWidth="1.4" />
    </svg>
  )
}

export function Shell({ active, onSelect, children }: { active: string; onSelect: (id: string) => void; children: ReactNode }) {
  const app = useApp()
  const { theme, setTheme, depth, setDepth, navOpen, setNavOpen, inspectorOpen, setInspectorOpen, presenting, setPresenting, helpOpen, setHelpOpen } = app
  useHotkeys({
    t: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
    f: () => setPresenting(!presenting),
    '1': () => setDepth('beginner'), '2': () => setDepth('intermediate'), '3': () => setDepth('advanced'),
    n: () => setNavOpen(!navOpen), i: () => setInspectorOpen(!inspectorOpen),
    '?': () => setHelpOpen(!helpOpen), Escape: () => setHelpOpen(false),
  }, [theme, presenting, navOpen, inspectorOpen, helpOpen])
  const showNav = navOpen && !presenting
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)] grid-rows-[auto_minmax(0,1fr)]">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-panel/85 px-3 py-2 backdrop-blur">
        <button type="button" className="btn icon-btn" onClick={() => setNavOpen(!navOpen)} aria-label="Toggle module navigation (N)" title="Modules (N)">
          {navOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
        </button>
        <div className="flex items-center gap-2">
          <Logo />
          <div className="leading-tight">
            <div className="font-display text-[17px] tracking-tight">AI Deep Learning Observatory</div>
            <div className="hidden font-mono text-[10.5px] text-faint sm:block">neural computation, made observable</div>
          </div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="flex rounded-lg border border-line p-0.5" role="radiogroup" aria-label="Depth level">
            {(['beginner', 'intermediate', 'advanced'] as Depth[]).map((d, k) => (
              <button key={d} type="button" role="radio" aria-checked={depth === d} onClick={() => setDepth(d)} title={`${d} (${k + 1})`}
                className={`rounded-md px-2.5 py-1 text-[12px] capitalize ${depth === d ? 'bg-raise text-cyan' : 'text-muted hover:text-fg'}`}>{d}</button>
            ))}
          </div>
          <button type="button" className="btn icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle projector theme (T)" title="Theme (T)">
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <button type="button" className="btn icon-btn" onClick={() => setInspectorOpen(!inspectorOpen)} aria-label="Toggle inspector (I)" title="Inspector (I)">
            {inspectorOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
          </button>
          <button type="button" className="btn" onClick={() => setPresenting(!presenting)} aria-pressed={presenting} title="Presentation mode (F)">
            {presenting ? <Minimize2 size={15} /> : <Maximize2 size={15} />}<span className="hidden md:inline">Present</span>
          </button>
          <button type="button" className="btn icon-btn" onClick={() => setHelpOpen(true)} aria-label="Keyboard shortcuts (?)" title="Shortcuts (?)"><HelpCircle size={16} /></button>
        </div>
      </header>
      <div className="grid min-h-0" style={{ gridTemplateColumns: showNav ? 'minmax(0,auto) minmax(0,1fr)' : 'minmax(0,1fr)' }}>
        {showNav && (
          <nav aria-label="Modules" className="w-[248px] overflow-y-auto border-r border-line bg-panel/60 p-2 scroll-thin max-lg:fixed max-lg:inset-y-0 max-lg:left-0 max-lg:top-[57px] max-lg:z-40 max-lg:bg-panel">
            <div className="eyebrow px-2 pb-2 pt-1">curriculum</div>
            <ol className="grid gap-0.5">
              {MODULES.map((m, i) => (
                <li key={m.id}>
                  <button type="button" onClick={() => onSelect(m.id)} aria-current={active === m.id ? 'page' : undefined}
                    className={`grid w-full grid-cols-[26px_1fr] items-baseline gap-1 rounded-lg px-2 py-2 text-left ${active === m.id ? 'bg-raise' : 'hover:bg-panel-2'}`}>
                    <span className={`font-mono text-[11px] font-num ${active === m.id ? 'text-cyan' : 'text-faint'}`}>{String(i + 1).padStart(2, '0')}</span>
                    <span>
                      <span className={`block text-[13px] ${m.status === 'ready' ? 'text-fg' : 'text-muted'}`}>{m.title}</span>
                      {m.status === 'planned' && <span className="font-mono text-[10px] text-faint">planned</span>}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </nav>
        )}
        <main className="min-h-0 min-w-0">{children}</main>
      </div>
      {helpOpen && <HelpOverlay onClose={() => setHelpOpen(false)} />}
    </div>
  )
}

/** Shared module layout: stage (~65%), inspector (right), timeline (bottom). */
export function Workspace({ stage, inspector, timeline }: { stage: ReactNode; inspector: ReactNode; timeline: ReactNode }) {
  const { inspectorOpen } = useApp()
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)] grid-rows-[minmax(0,1fr)_auto]">
      <div className={`grid min-h-0 grid-cols-[minmax(0,1fr)] ${inspectorOpen ? 'lg:grid-cols-[minmax(0,1fr)_minmax(300px,32%)] max-lg:overflow-y-auto' : ''}`}>
        <div className="min-w-0 lg:min-h-0 lg:overflow-y-auto scroll-thin">{stage}</div>
        {inspectorOpen && (
          <aside aria-label="Mathematical inspector" className="min-w-0 lg:min-h-0 border-l border-line bg-panel/70 p-4 lg:overflow-y-auto scroll-thin max-lg:border-l-0 max-lg:border-t">
            {inspector}
          </aside>
        )}
      </div>
      <div className="border-t border-line bg-panel/90 backdrop-blur">{timeline}</div>
    </div>
  )
}

const SHORTCUTS: [string, string][] = [
  ['→ / ←', 'Step forward / backward'], ['Space', 'Play / pause'], ['Home', 'Reset to the first step'],
  ['1 · 2 · 3', 'Beginner · intermediate · advanced'], ['T', 'Dark / projector theme'], ['F', 'Presentation mode (fullscreen)'],
  ['N / I', 'Toggle modules / inspector'], ['P', 'Toggle predict-before-reveal'], ['E', 'Export snapshot'], ['G', '3D overview'], ['?', 'This help'],
]
function HelpOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-bg/70 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" onClick={onClose}>
      <div className="panel w-[min(480px,100%)] rounded-2xl p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-display text-[20px]">Keyboard shortcuts</h2>
          <button type="button" className="btn icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <dl className="grid grid-cols-[110px_1fr] gap-y-2 text-[13px]">
          {SHORTCUTS.map(([k, v]) => <Fragment key={k}><dt className="font-mono text-cyan">{k}</dt><dd className="text-muted">{v}</dd></Fragment>)}
        </dl>
      </div>
    </div>
  )
}
