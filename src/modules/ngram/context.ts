import { createContext, useContext } from 'react'
import { BOS, EOS, UNK, type ModelConfig, type NGramModel, type Scorer, type SentenceScore } from '../../engine/ngram'
import type { Stage } from './lesson'

export type NGTab = 'guided' | 'explore' | 'compare' | 'challenge' | 'lesson'

export interface NGLab {
  model: NGramModel
  corpus: string; setCorpus: (s: string) => void
  test: string; setTest: (s: string) => void
  heldOut: string; setHeldOut: (s: string) => void
  exampleId: string | null
  loadExample: (id: string) => void
  cfg: ModelConfig; setCfg: (p: Partial<ModelConfig>) => void
  /** cfg with the guided stage's method applied */
  eff: ModelConfig
  scorer: Scorer
  score: SentenceScore
  heldScore: SentenceScore
  stage: Stage
  tab: NGTab
  focus: number; setFocus: (i: number) => void
  /** two-token history (oldest first) for the probability explorer */
  ctx: string[]; setCtx: (h: string[]) => void
  seed: number; setSeed: (n: number) => void
}

export const NGCtx = createContext<NGLab | null>(null)
export const useNG = () => {
  const c = useContext(NGCtx)
  if (!c) throw new Error('useNG outside provider')
  return c
}

export const showTok = (t: string) => (t === BOS || t === EOS || t === UNK ? t : t)
export const fmtP = (p: number) =>
  Number.isNaN(p) ? 'undef.' : p === 0 ? '0' : p === Infinity ? '∞' : p < 1e-4 ? p.toExponential(2) : p.toFixed(4)
export const fmtPP = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '∞')
