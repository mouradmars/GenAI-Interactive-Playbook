import { createContext, useContext, useEffect, useState } from 'react'
import type { Arch, Grads, Objective, Params, Regime, SimConfig, Trace } from '../../engine/seq'
import type { Stage } from './lesson'

export type SeqTab = 'guided' | 'explore' | 'compare' | 'challenge' | 'lesson'
export type Source = 'trained' | 'sim'

export interface ModelRun { arch: Arch; params: Params; trace: Trace; grads: Grads; pred: number }

export interface SeqLab {
  source: Source; setSource: (s: Source) => void
  regime: Regime; setRegime: (r: Regime) => void
  sim: SimConfig; setSim: (p: Partial<SimConfig>) => void
  seq: string; setSeq: (s: string) => void
  ids: number[]
  target: number
  objective: Objective; setObjective: (o: Objective) => void
  runs: Record<Arch, ModelRun>
  cursor: TimeCursor
  unit: number; setUnit: (u: number) => void
  stage: Stage
  tab: SeqTab
  seed: number; setSeed: (n: number) => void
}

export const SeqCtx = createContext<SeqLab | null>(null)
export const useSeq = () => { const c = useContext(SeqCtx); if (!c) throw new Error('useSeq outside provider'); return c }

/** A time cursor with its own transport (play / pause / step / reset), independent of the stage timeline. */
export interface TimeCursor { t: number; T: number; playing: boolean; set: (t: number) => void; next: () => void; prev: () => void; reset: () => void; toggle: () => void }
export function useTimeCursor(T: number, speed = 1.4): TimeCursor {
  const [t, setT] = useState(Math.max(0, T - 1))
  const [playing, setPlaying] = useState(false)
  useEffect(() => { setT((x) => Math.min(x, T - 1)) }, [T])
  useEffect(() => {
    if (!playing) return
    const id = setTimeout(() => setT((x) => { if (x >= T - 1) { setPlaying(false); return x } return x + 1 }), 1000 / speed)
    return () => clearTimeout(id)
  }, [playing, t, T, speed])
  // clamp on read as well: when the sequence gets shorter, the stored t may still point past its end for one render
  const tc = Math.max(0, Math.min(t, T - 1))
  return {
    t: tc, T, playing, set: (v) => { setPlaying(false); setT(Math.max(0, Math.min(T - 1, v))) },
    next: () => { setPlaying(false); setT((x) => Math.min(T - 1, x + 1)) }, prev: () => { setPlaying(false); setT((x) => Math.max(0, x - 1)) },
    reset: () => { setPlaying(false); setT(0) }, toggle: () => { if (!playing && tc >= T - 1) setT(0); setPlaying(!playing) },
  }
}

export const f3 = (x: number) => (Number.isFinite(x) ? (Math.abs(x) < 5e-13 ? 0 : x).toFixed(3) : '—')
export const f4 = (x: number) => (Number.isFinite(x) ? (Math.abs(x) < 5e-13 ? 0 : x).toFixed(4) : '—')
export const sci = (x: number) => (x === 0 ? '0' : Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toFixed(3) : x.toExponential(1))
export const pct = (x: number) => `${(x * 100).toFixed(1)}%`
