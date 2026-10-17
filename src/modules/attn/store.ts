/**
 * Zustand store for module 03 · Attention: the editable toy model (input, projections, configuration), the
 * synchronized selection shared by every view (query i, key j, head h, selected matrix cell), a saved baseline
 * for before/after comparisons, and the state of the follow, experiment and challenge modes.
 */
import { createStore, useStore } from 'zustand'
import { PRESET_PARAMS, presetInput, sinusoidal, tokenize, toyEmbedding, type AttnConfig, type AttnParams, type Matrix } from '../../engine/attention'

export type AttnTab = 'guided' | 'microscopes' | 'follow' | 'experiments' | 'challenge' | 'research' | 'lessons'
export type WKey = 'Wq' | 'Wk' | 'Wv' | 'Wo'
export type XSource = 'hand' | 'embed'
export const D_MODEL = 4
export const MAX_T = 8
export const PAD = '<pad>'
export const DEFAULT_SENTENCE = 'The cat chased the mouse.'
export const VALUE_LIMIT = 10

export interface Sel { i: number; j: number; h: number; cell: { m: string; r: number; c: number } | null }
export interface Baseline { label: string; X: Matrix; params: AttnParams; cfg: AttnConfig; tokens: string[] }

export interface AttnState {
  sentence: string
  windowStart: number
  T: number
  H: number
  xSource: XSource
  X: Matrix
  xEdited: boolean
  params: AttnParams
  paramsEdited: boolean
  causal: boolean
  scale: boolean
  pad: boolean[]
  blocked: boolean[][] | null
  headsOn: boolean[]
  sel: Sel
  graphMode: 'raw' | 'prob'
  baseline: Baseline | null
  tab: AttnTab
  followStep: number
  challenge: { seed: number; answers: Record<string, { given: string; ok: boolean; tries: number }>; hints: Record<string, number> }

  setSentence: (s: string) => void
  setWindow: (start: number, T?: number) => void
  setH: (H: number) => void
  setXSource: (s: XSource) => void
  setX: (i: number, f: number, v: number) => string | null
  setW: (k: WKey, r: number, c: number, v: number) => string | null
  setCausal: (v: boolean) => void
  setScale: (v: boolean) => void
  togglePad: (j: number) => void
  setPad: (p: boolean[]) => void
  toggleBlocked: (i: number, j: number) => void
  clearBlocked: () => void
  toggleHead: (h: number) => void
  setHeadsOn: (v: boolean[]) => void
  select: (p: Partial<Sel>) => void
  setGraphMode: (m: 'raw' | 'prob') => void
  saveBaseline: (label?: string) => void
  clearBaseline: () => void
  restoreBaseline: () => void
  setTab: (t: AttnTab) => void
  setFollowStep: (k: number) => void
  setParams: (p: AttnParams) => void
  setInput: (X: Matrix) => void
  resetAll: () => void
  setChallengeSeed: (s: number) => void
  answer: (id: string, given: string, ok: boolean) => void
  hint: (id: string) => void
}

/** the window of tokens the toy layer attends over; positions past the end of the sentence are padding tokens */
export function windowTokens(sentence: string, start: number, T: number): string[] {
  const all = tokenize(sentence)
  return Array.from({ length: T }, (_, i) => all[start + i] ?? PAD)
}
export const allTokens = (sentence: string) => tokenize(sentence)
/** the X used when the source is "toy embedding + sinusoidal position": e(token) + PE(position in the sentence) */
export function embedInput(tokens: string[], start: number): { E: Matrix; P: Matrix; X: Matrix } {
  const E = tokens.map((t) => (t === PAD ? new Array(D_MODEL).fill(0) : toyEmbedding(t, D_MODEL)))
  const P = sinusoidal(start + tokens.length, D_MODEL).slice(start)
  return { E, P, X: E.map((r, i) => r.map((v, k) => v + P[i][k])) }
}
export const validateValue = (v: number): string | null =>
  !Number.isFinite(v) ? 'Enter a number.' : Math.abs(v) > VALUE_LIMIT ? `Keep values within ±${VALUE_LIMIT} so the numbers stay readable.` : null

export const cfgOf = (s: Pick<AttnState, 'H' | 'causal' | 'scale' | 'pad' | 'blocked' | 'headsOn'>): AttnConfig => ({ H: s.H, causal: s.causal, scale: s.scale, pad: s.pad, blocked: s.blocked, headsOn: s.headsOn })
const clone = (m: Matrix) => m.map((r) => r.slice())
const cloneP = (p: AttnParams): AttnParams => ({ Wq: clone(p.Wq), Wk: clone(p.Wk), Wv: clone(p.Wv), Wo: clone(p.Wo) })

export function createAttnStore() {
  return createStore<AttnState>()((set, get) => {
    /** recompute everything that depends on the window (tokens, X, padding) */
    const window = (sentence: string, start: number, T: number, xSource: XSource): Partial<AttnState> => {
      const tokens = windowTokens(sentence, start, T)
      const X = xSource === 'embed' ? embedInput(tokens, start).X : presetInput(T)
      const s = get()
      return {
        sentence, windowStart: start, T, xSource, X, xEdited: false, pad: tokens.map((t) => t === PAD), blocked: null,
        sel: { i: Math.min(s?.sel.i ?? 3, T - 1), j: Math.min(s?.sel.j ?? 0, T - 1), h: s?.sel.h ?? 0, cell: null }, baseline: null,
      }
    }
    const initial = window(DEFAULT_SENTENCE, 1, 4, 'hand')
    return {
      sentence: DEFAULT_SENTENCE, windowStart: 1, T: 4, H: 2, xSource: 'hand', X: initial.X!, xEdited: false,
      params: cloneP(PRESET_PARAMS), paramsEdited: false, causal: true, scale: true, pad: initial.pad!, blocked: null, headsOn: [true, true],
      sel: { i: 2, j: 0, h: 0, cell: null }, graphMode: 'raw', baseline: null, tab: 'guided', followStep: 0,
      challenge: { seed: 1, answers: {}, hints: {} },

      setSentence: (sentence) => { const s = get(); const n = tokenize(sentence).length; set(window(sentence.slice(0, 160), Math.max(0, Math.min(s.windowStart, Math.max(0, n - 2))), s.T, s.xSource)) },
      setWindow: (start, T) => { const s = get(); const t = Math.max(2, Math.min(MAX_T, Math.round(T ?? s.T))); set(window(s.sentence, Math.max(0, Math.round(start)), t, s.xSource)) },
      setH: (H) => { if (D_MODEL % H !== 0) return; set({ H, headsOn: new Array(H).fill(true), sel: { ...get().sel, h: Math.min(get().sel.h, H - 1), cell: null } }) },
      setXSource: (xSource) => { const s = get(); set(window(s.sentence, s.windowStart, s.T, xSource)) },
      setX: (i, f, v) => {
        const e = validateValue(v); if (e) return e
        const X = clone(get().X); X[i][f] = v; set({ X, xEdited: true }); return null
      },
      setW: (k, r, c, v) => {
        const e = validateValue(v); if (e) return e
        const params = cloneP(get().params); params[k][r][c] = v; set({ params, paramsEdited: true }); return null
      },
      setCausal: (causal) => set({ causal }),
      setScale: (scale) => set({ scale }),
      togglePad: (j) => { const pad = get().pad.slice(); pad[j] = !pad[j]; set({ pad }) },
      setPad: (pad) => set({ pad }),
      toggleBlocked: (i, j) => {
        const T = get().T
        const b = get().blocked?.map((r) => r.slice()) ?? Array.from({ length: T }, () => new Array(T).fill(false))
        b[i][j] = !b[i][j]
        set({ blocked: b.some((r) => r.some(Boolean)) ? b : null })
      },
      clearBlocked: () => set({ blocked: null }),
      toggleHead: (h) => { const headsOn = get().headsOn.slice(); headsOn[h] = !headsOn[h]; set({ headsOn }) },
      setHeadsOn: (headsOn) => set({ headsOn }),
      select: (p) => set({ sel: { ...get().sel, ...p } }),
      setGraphMode: (graphMode) => set({ graphMode }),
      saveBaseline: (label = 'saved state') => { const s = get(); set({ baseline: { label, X: clone(s.X), params: cloneP(s.params), cfg: { ...cfgOf(s), pad: s.pad.slice(), blocked: s.blocked?.map((r) => r.slice()) ?? null, headsOn: s.headsOn.slice() }, tokens: windowTokens(s.sentence, s.windowStart, s.T) } }) },
      clearBaseline: () => set({ baseline: null }),
      restoreBaseline: () => {
        const b = get().baseline
        if (!b || b.X.length !== get().T) return
        set({ X: clone(b.X), params: cloneP(b.params), H: b.cfg.H, causal: b.cfg.causal, scale: b.cfg.scale, pad: b.cfg.pad.slice(), blocked: b.cfg.blocked?.map((r) => r.slice()) ?? null, headsOn: b.cfg.headsOn.slice(), sel: { ...get().sel, h: Math.min(get().sel.h, b.cfg.H - 1) } })
      },
      setTab: (tab) => set({ tab }),
      setFollowStep: (followStep) => set({ followStep }),
      setParams: (p) => set({ params: cloneP(p), paramsEdited: true }),
      setInput: (X) => set({ X: clone(X), xEdited: true }),
      resetAll: () => set({
        ...window(DEFAULT_SENTENCE, 1, 4, 'hand'), H: 2, headsOn: [true, true], params: cloneP(PRESET_PARAMS), paramsEdited: false, causal: true, scale: true,
        sel: { i: 2, j: 0, h: 0, cell: null }, graphMode: 'raw', followStep: 0,
      }),
      setChallengeSeed: (seed) => set({ challenge: { seed: Math.max(0, Math.round(seed)), answers: {}, hints: {} } }),
      answer: (id, given, ok) => { const c = get().challenge; set({ challenge: { ...c, answers: { ...c.answers, [id]: { given, ok, tries: (c.answers[id]?.tries ?? 0) + 1 } } } }) },
      hint: (id) => { const c = get().challenge; set({ challenge: { ...c, hints: { ...c.hints, [id]: (c.hints[id] ?? 0) + 1 } } }) },
    }
  })
}
export const attnStore = createAttnStore()
export const useAttn = <T,>(sel: (s: AttnState) => T) => useStore(attnStore, sel)
