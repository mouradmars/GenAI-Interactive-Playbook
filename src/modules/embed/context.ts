import { createContext, useContext } from 'react'
import type { Metric, PPMIConfig, ProjConfig, Projection, Space } from '../../engine/embeddings'
import type { NanoGPT } from '../../engine/model'
import type { SpaceId, Stage } from './lesson'

export type EmbTab = 'guided' | 'explore' | 'compare' | 'challenge' | 'lesson'

export interface Picks {
  focus: string
  selected: string[]
  analogy: [string, string, string, string] // a, b, c, expected ("" = none)
}

export interface EmbLab {
  model: NanoGPT
  spaceId: SpaceId; setSpaceId: (s: SpaceId) => void
  space: Space
  spaces: Record<SpaceId, Space>
  ppmiCfg: PPMIConfig; setPPMI: (p: Partial<PPMIConfig>) => void
  proj: ProjConfig; setProj: (p: Partial<ProjConfig>) => void
  projection: Projection
  /** true while a new projection is being computed for the latest settings */
  stale: boolean
  metric: Metric; setMetric: (m: Metric) => void
  k: number; setK: (k: number) => void
  picks: Picks; setPicks: (p: Partial<Picks>) => void
  /** indices into space.words */
  focusIdx: number
  selIdx: number[]
  cell: [number, number] | null; setCell: (c: [number, number] | null) => void
  highlightPair: [number, number] | null; setHighlightPair: (p: [number, number] | null) => void
  ctxSentences: string; setCtxSentences: (s: string) => void
  ctxToken: string; setCtxToken: (t: string) => void
  stage: Stage
  tab: EmbTab
  seed: number; setSeed: (n: number) => void
  /** the two draggable vectors of the Dot/Cosine stages */
  vecA: [number, number]; vecB: [number, number]; setVecA: (v: [number, number]) => void; setVecB: (v: [number, number]) => void
  scaleA: number; setScaleA: (s: number) => void
}

export const EmbCtx = createContext<EmbLab | null>(null)
export const useEmb = () => {
  const c = useContext(EmbCtx)
  if (!c) throw new Error('useEmb outside provider')
  return c
}

export const f3 = (x: number) => (Number.isFinite(x) ? (Math.abs(x) < 1e-12 ? 0 : x).toFixed(3) : '—')
export const f4 = (x: number) => (Number.isFinite(x) ? (Math.abs(x) < 1e-12 ? 0 : x).toFixed(4) : '—')
export const pct = (x: number) => `${(x * 100).toFixed(1)}%`

export const GROUP_ORDER = ['people', 'animals', 'fruit', 'foods', 'capitals', 'countries', 'colors', 'things', 'numbers', 'names', 'places', 'adjectives', 'verbs', 'function', 'subword']
