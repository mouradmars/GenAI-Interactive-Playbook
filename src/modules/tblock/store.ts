/**
 * Zustand store for module 02. Configuration, the editable toy parameters and inputs, interventions and the
 * synchronized selection (query token, key token, head, feature) shared by every view.
 */
import { createStore, useStore } from 'zustand'
import { defaultInterventions, toyEmbeddings, toyParams, type BlockConfig, type BlockParams, type Interventions, type Matrix, type MlpAct } from '../../engine/block'
import type { ArchPreset } from './stages'

export type TbTab = 'block' | 'compare' | 'heads' | 'stack' | 'real' | 'advanced' | 'challenge' | 'lessons'
export const SENTENCES: { label: string; tokens: string[] }[] = [
  { label: 'The cat sat on the mat .', tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat', '.'] },
  { label: 'The animal didn\'t cross the street because it was tired .', tokens: ['The', 'animal', 'didn\'t', 'cross', 'the', 'street', 'because', 'it', 'was', 'tired'] },
]
export const MAX_T = 10

export interface Sel { i: number; j: number; h: number; f: number; u: number; cell: { m: string; r: number; c: number } | null }
export type ParamKey = 'Wq' | 'Wk' | 'Wv' | 'Wo' | 'W1' | 'W2' | 'W3'

export interface TbState {
  arch: ArchPreset
  sentence: number
  tokens: string[]
  T: number; d: number; H: number; dff: number; act: MlpAct; eps: number
  /** educational comparison: the decoder's causal mask switched off */
  maskOff: boolean
  seed: number
  params: BlockParams
  X: Matrix
  xEdited: boolean
  paramsEdited: boolean
  iv: Interventions
  sel: Sel
  tab: TbTab
  showUpstream: boolean
  stackN: number
  stackOn: boolean[]
  stackBlock: number
  challenge: { level: number; seed: number; answers: Record<string, { given: string; ok: boolean }> }

  setArch: (a: ArchPreset) => void
  setSentence: (k: number) => void
  setToken: (i: number, label: string) => void
  setDims: (p: Partial<Pick<TbState, 'T' | 'd' | 'H' | 'dff'>>) => void
  setAct: (a: MlpAct) => void
  setEps: (e: number) => void
  setMaskOff: (v: boolean) => void
  setSeed: (s: number) => void
  setX: (i: number, f: number, v: number) => void
  setParam: (k: ParamKey, r: number, c: number, v: number) => void
  setLN: (which: 'ln1' | 'ln2', field: 'g' | 'b', f: number, v: number) => void
  resetValues: () => void
  setIv: (p: Partial<Interventions>) => void
  toggleHead: (h: number) => void
  toggleUnit: (u: number) => void
  select: (p: Partial<Sel>) => void
  setTab: (t: TbTab) => void
  setShowUpstream: (v: boolean) => void
  setStack: (p: Partial<Pick<TbState, 'stackN' | 'stackOn' | 'stackBlock'>>) => void
  setChallenge: (p: Partial<TbState['challenge']>) => void
  answer: (id: string, given: string, ok: boolean) => void
}

export const cfgOf = (s: Pick<TbState, 'arch' | 'T' | 'd' | 'H' | 'dff' | 'act' | 'eps' | 'maskOff'>): BlockConfig => ({
  T: s.T, d: s.d, H: s.H, dff: s.dff, act: s.act, eps: s.eps, arch: s.arch === 'gpt' ? 'pre' : 'post', causal: s.arch === 'gpt' && !s.maskOff,
})
const tokensFor = (sentence: number, T: number) => { const t = SENTENCES[sentence].tokens; return Array.from({ length: T }, (_, i) => t[i] ?? `tok${i + 1}`) }

export function createTbStore() {
  return createStore<TbState>()((set, get) => {
    const fresh = (over: Partial<TbState>): Partial<TbState> => {
      const s = { ...get(), ...over }
      const tokens = over.tokens ?? (s.tokens.length === s.T ? s.tokens : tokensFor(s.sentence, s.T))
      const sel = s.sel
      return {
        ...over, tokens, params: toyParams({ d: s.d, dff: s.dff }, s.seed), X: toyEmbeddings(tokens, s.d, s.seed).X, xEdited: false, paramsEdited: false,
        iv: defaultInterventions({ H: s.H, dff: s.dff }),
        sel: { ...sel, i: Math.min(sel.i, s.T - 1), j: Math.min(sel.j, s.T - 1), h: Math.min(sel.h, s.H - 1), f: Math.min(sel.f, s.d - 1), u: Math.min(sel.u, s.dff - 1), cell: null },
      }
    }
    const T0 = 4, d0 = 4, H0 = 2, dff0 = 8, tokens0 = tokensFor(0, T0)
    return {
      arch: 'gpt', sentence: 0, tokens: tokens0, T: T0, d: d0, H: H0, dff: dff0, act: 'gelu', eps: 1e-5, maskOff: false, seed: 1,
      params: toyParams({ d: d0, dff: dff0 }, 1), X: toyEmbeddings(tokens0, d0, 1).X, xEdited: false, paramsEdited: false,
      iv: defaultInterventions({ H: H0, dff: dff0 }),
      sel: { i: 3, j: 1, h: 0, f: 0, u: 0, cell: null }, tab: 'block', showUpstream: false,
      stackN: 3, stackOn: [true, true, true, true, true, true], stackBlock: 0,
      challenge: { level: 0, seed: 1, answers: {} },

      setArch: (arch) => set({ arch, maskOff: false }),
      setSentence: (k) => { const tokens = tokensFor(k, get().T); set(fresh({ sentence: k, tokens })) },
      setToken: (i, label) => {
        const s = get(), tokens = s.tokens.slice(); tokens[i] = label.slice(0, 14) || `tok${i + 1}`
        // relabelling changes the upstream embedding lookup (unless X was edited by hand)
        set(s.xEdited ? { tokens } : { tokens, X: toyEmbeddings(tokens, s.d, s.seed).X })
      },
      setDims: (p) => {
        const s = { ...get(), ...p }
        let { T, d, H, dff } = s
        T = Math.max(2, Math.min(MAX_T, Math.round(T))); d = Math.max(2, Math.min(8, Math.round(d))); dff = Math.max(2, Math.min(32, Math.round(dff)))
        H = Math.max(1, Math.min(d, Math.round(H))); while (d % H !== 0) H--
        const tokens = T === get().tokens.length ? get().tokens : Array.from({ length: T }, (_, i) => get().tokens[i] ?? tokensFor(s.sentence, T)[i])
        set(fresh({ T, d, H, dff, tokens }))
      },
      setAct: (act) => set({ act }),
      setEps: (eps) => set({ eps: Math.max(1e-12, eps) }),
      setMaskOff: (maskOff) => set({ maskOff }),
      setSeed: (seed) => set(fresh({ seed: Math.max(0, Math.round(seed)) })),
      setX: (i, f, v) => { if (!Number.isFinite(v)) return; const X = get().X.map((r) => r.slice()); X[i][f] = v; set({ X, xEdited: true }) },
      setParam: (k, r, c, v) => { if (!Number.isFinite(v)) return; const p = get().params; set({ params: { ...p, [k]: p[k].map((row, ri) => (ri === r ? row.map((x, ci) => (ci === c ? v : x)) : row)) }, paramsEdited: true }) },
      setLN: (which, field, f, v) => { if (!Number.isFinite(v)) return; const p = get().params; set({ params: { ...p, [which]: { ...p[which], [field]: p[which][field].map((x, k) => (k === f ? v : x)) } }, paramsEdited: true }) },
      resetValues: () => set(fresh({})),
      setIv: (p) => set({ iv: { ...get().iv, ...p } }),
      toggleHead: (h) => { const heads = get().iv.heads.slice(); heads[h] = !heads[h]; set({ iv: { ...get().iv, heads } }) },
      toggleUnit: (u) => { const units = get().iv.units.slice(); units[u] = !units[u]; set({ iv: { ...get().iv, units } }) },
      select: (p) => set({ sel: { ...get().sel, ...p } }),
      setTab: (tab) => set({ tab }),
      setShowUpstream: (showUpstream) => set({ showUpstream }),
      setStack: (p) => set(p),
      setChallenge: (p) => set({ challenge: { ...get().challenge, ...p, answers: p.seed !== undefined || p.level !== undefined ? {} : get().challenge.answers } }),
      answer: (id, given, ok) => { const c = get().challenge; set({ challenge: { ...c, answers: { ...c.answers, [id]: { given, ok } } } }) },
    }
  })
}
export const tbStore = createTbStore()
export const useTb = <T,>(sel: (s: TbState) => T) => useStore(tbStore, sel)
