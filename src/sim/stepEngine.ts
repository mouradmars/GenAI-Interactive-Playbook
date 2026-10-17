/**
 * Shared step engine used by every module: a pure reducer (unit-tested) plus a
 * React hook that drives playback. Modules supply only the number of steps.
 */
import { useCallback, useEffect, useReducer } from 'react'

export interface EngineState {
  index: number
  count: number
  playing: boolean
  speed: number // steps per second
}
export type EngineAction =
  | { type: 'next' } | { type: 'prev' } | { type: 'reset' } | { type: 'play' } | { type: 'pause' } | { type: 'toggle' }
  | { type: 'seek'; index: number } | { type: 'speed'; speed: number } | { type: 'count'; count: number } | { type: 'tick' }

const clamp = (i: number, n: number) => Math.max(0, Math.min(n - 1, i))

export function engineReducer(s: EngineState, a: EngineAction): EngineState {
  switch (a.type) {
    case 'next': return { ...s, index: clamp(s.index + 1, s.count), playing: false }
    case 'prev': return { ...s, index: clamp(s.index - 1, s.count), playing: false }
    case 'reset': return { ...s, index: 0, playing: false }
    case 'seek': return { ...s, index: clamp(a.index, s.count), playing: false }
    case 'play': return { ...s, playing: true, index: s.index >= s.count - 1 ? 0 : s.index }
    case 'pause': return { ...s, playing: false }
    case 'toggle': return engineReducer(s, { type: s.playing ? 'pause' : 'play' })
    case 'speed': return { ...s, speed: Math.max(0.1, Math.min(8, a.speed)) }
    case 'count': return { ...s, count: a.count, index: clamp(s.index, a.count) }
    case 'tick': {
      if (!s.playing) return s
      const index = s.index + 1
      return index >= s.count ? { ...s, playing: false } : { ...s, index }
    }
  }
}

export function useStepEngine(count: number, initial = 0) {
  const [state, dispatch] = useReducer(engineReducer, { index: initial, count, playing: false, speed: 0.6 })
  useEffect(() => dispatch({ type: 'count', count }), [count])
  useEffect(() => {
    if (!state.playing) return
    const t = setTimeout(() => dispatch({ type: 'tick' }), 1000 / state.speed)
    return () => clearTimeout(t)
  }, [state.playing, state.index, state.speed])
  const api = {
    next: useCallback(() => dispatch({ type: 'next' }), []),
    prev: useCallback(() => dispatch({ type: 'prev' }), []),
    reset: useCallback(() => dispatch({ type: 'reset' }), []),
    toggle: useCallback(() => dispatch({ type: 'toggle' }), []),
    pause: useCallback(() => dispatch({ type: 'pause' }), []),
    seek: useCallback((index: number) => dispatch({ type: 'seek', index }), []),
    setSpeed: useCallback((speed: number) => dispatch({ type: 'speed', speed }), []),
  }
  return { ...state, ...api }
}
export type StepEngine = ReturnType<typeof useStepEngine>
