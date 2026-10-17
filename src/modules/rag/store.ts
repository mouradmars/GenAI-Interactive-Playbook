/** Zustand store for module 11 (RAG): corpus, question, configuration A (and B for comparison), selection, challenge state. */
import { createStore, useStore } from 'zustand'
import { CORPORA, corpusById, type RagDoc } from '../../data/rag-corpora'
import { DEFAULT_CONFIG, type PipelineConfig } from '../../engine/rag'

export type RagTab = 'guided' | 'explore' | 'compare' | 'challenge' | 'lesson'
export interface RagState {
  corpusId: string
  /** documents added by the student (appended to the corpus) */
  extraDocs: RagDoc[]
  /** a question id of the corpus, or 'custom' */
  qid: string
  custom: string
  cfg: PipelineConfig
  cfgB: PipelineConfig
  selChunk: number | null
  tab: RagTab
  view3d: boolean
  challenge: { seed: number; answers: Record<string, { given: string; ok: boolean }> }
  setCorpus: (id: string) => void
  addDoc: (title: string, text: string) => void
  removeExtra: (id: string) => void
  setQuestion: (qid: string) => void
  setCustom: (text: string) => void
  setCfg: (f: (c: PipelineConfig) => PipelineConfig) => void
  setCfgB: (f: (c: PipelineConfig) => PipelineConfig) => void
  resetCfg: () => void
  selectChunk: (i: number | null) => void
  setTab: (t: RagTab) => void
  setView3d: (v: boolean) => void
  setChallengeSeed: (s: number) => void
  answer: (id: string, given: string, ok: boolean) => void
}
const clone = (c: PipelineConfig): PipelineConfig => JSON.parse(JSON.stringify(c))

export function createRagStore() {
  return createStore<RagState>()((set, get) => ({
    corpusId: CORPORA[0].id, extraDocs: [], qid: CORPORA[0].questions[0].id, custom: '',
    cfg: clone(DEFAULT_CONFIG), cfgB: { ...clone(DEFAULT_CONFIG), chunk: { size: 15, overlap: 0, strategy: 'fixed' }, retrieve: { ...DEFAULT_CONFIG.retrieve, method: 'bm25' } },
    selChunk: null, tab: 'guided', view3d: false, challenge: { seed: 1, answers: {} },
    setCorpus: (id) => set({ corpusId: id, extraDocs: [], qid: corpusById(id).questions[0].id, selChunk: null }),
    addDoc: (title, text) => { const t = text.trim(); if (!t) return; set({ extraDocs: [...get().extraDocs, { id: `user${get().extraDocs.length + 1}`, title: title.trim() || `My document ${get().extraDocs.length + 1}`, text: t.replace(/\s+/g, ' ') }], selChunk: null }) },
    removeExtra: (id) => set({ extraDocs: get().extraDocs.filter((d) => d.id !== id), selChunk: null }),
    setQuestion: (qid) => set({ qid, selChunk: null }),
    setCustom: (custom) => set({ custom: custom.slice(0, 200), qid: 'custom', selChunk: null }),
    setCfg: (f) => set({ cfg: f(clone(get().cfg)) }),
    setCfgB: (f) => set({ cfgB: f(clone(get().cfgB)) }),
    resetCfg: () => set({ cfg: clone(DEFAULT_CONFIG) }),
    selectChunk: (selChunk) => set({ selChunk }),
    setTab: (tab) => set({ tab }),
    setView3d: (view3d) => set({ view3d }),
    setChallengeSeed: (seed) => set({ challenge: { seed: Math.max(0, Math.round(seed)), answers: {} } }),
    answer: (id, given, ok) => { const c = get().challenge; set({ challenge: { ...c, answers: { ...c.answers, [id]: { given, ok } } } }) },
  }))
}
export const ragStore = createRagStore()
export const useRag = <T,>(sel: (s: RagState) => T) => useStore(ragStore, sel)
export const docsOf = (s: Pick<RagState, 'corpusId' | 'extraDocs'>) => [...corpusById(s.corpusId).docs, ...s.extraDocs]
export const questionOf = (s: Pick<RagState, 'corpusId' | 'qid' | 'custom'>) => (s.qid === 'custom' ? null : corpusById(s.corpusId).questions.find((q) => q.id === s.qid) ?? null)
export const queryText = (s: Pick<RagState, 'corpusId' | 'qid' | 'custom'>) => questionOf(s)?.q ?? (s.custom || 'Ask a question')
