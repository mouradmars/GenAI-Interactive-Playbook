import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type Depth = 'beginner' | 'intermediate' | 'advanced'
export type Theme = 'dark' | 'light'

interface AppState {
  theme: Theme; setTheme: (t: Theme) => void
  depth: Depth; setDepth: (d: Depth) => void
  navOpen: boolean; setNavOpen: (v: boolean) => void
  inspectorOpen: boolean; setInspectorOpen: (v: boolean) => void
  presenting: boolean; setPresenting: (v: boolean) => void
  helpOpen: boolean; setHelpOpen: (v: boolean) => void
}
const Ctx = createContext<AppState | null>(null)

const load = <T,>(k: string, d: T): T => {
  try { const v = localStorage.getItem(k); return v ? (JSON.parse(v) as T) : d } catch { return d }
}
const save = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* storage may be unavailable */ } }

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => load('obs.theme', 'dark'))
  const [depth, setDepth] = useState<Depth>(() => load('obs.depth', 'intermediate'))
  const [navOpen, setNavOpen] = useState(() => typeof window === 'undefined' || window.innerWidth > 1100)
  const [inspectorOpen, setInspectorOpen] = useState(true)
  const [presenting, setPresenting] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)

  useEffect(() => { document.documentElement.dataset.theme = theme; save('obs.theme', theme) }, [theme])
  useEffect(() => save('obs.depth', depth), [depth])
  useEffect(() => {
    document.documentElement.classList.toggle('presenting', presenting)
    if (presenting) document.documentElement.requestFullscreen?.().catch(() => {})
    else if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {})
  }, [presenting])
  useEffect(() => {
    const onFs = () => { if (!document.fullscreenElement) setPresenting(false) }
    document.addEventListener('fullscreenchange', onFs)
    return () => document.removeEventListener('fullscreenchange', onFs)
  }, [])

  const value = useMemo(
    () => ({ theme, setTheme, depth, setDepth, navOpen, setNavOpen, inspectorOpen, setInspectorOpen, presenting, setPresenting, helpOpen, setHelpOpen }),
    [theme, depth, navOpen, inspectorOpen, presenting, helpOpen],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useApp = () => {
  const c = useContext(Ctx)
  if (!c) throw new Error('useApp outside provider')
  return c
}

const PALETTES = {
  dark: { bg: '#060a11', panel: '#0b111b', panel2: '#101827', line: '#1d2a3d', lineStrong: '#2b3c55', fg: '#e7eef8', muted: '#8b9bb2',
    faint: '#5b6b82', cyan: '#45e1f2', violet: '#a98bff', amber: '#f6b552', teal: '#34d6bd', rose: '#ff7a90' },
  light: { bg: '#f3f6fa', panel: '#ffffff', panel2: '#f7f9fc', line: '#d6dee9', lineStrong: '#b9c6d6', fg: '#0a1322', muted: '#475569',
    faint: '#7a889c', cyan: '#0a8aa0', violet: '#6644d8', amber: '#b56a00', teal: '#0d8a78', rose: '#cf2f4f' },
}
export type Colors = (typeof PALETTES)['dark']
/** Theme colours for SVG/canvas drawing; mirrors the CSS tokens in styles/index.css. */
export function useColors(): Colors {
  return PALETTES[useApp().theme]
}
