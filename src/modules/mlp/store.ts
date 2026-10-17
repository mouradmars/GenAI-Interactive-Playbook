/**
 * Zustand store for module 04. Holds the configuration, the working parameters, the recorded history of
 * every iteration (for rewind), the micro-step cursor of the current iteration and the selection.
 * Every number derives from engine/nn; the store never invents values.
 */
import { createStore, useStore } from 'zustand'
import {
  WORKED, applyUpdate, backward, cloneParams, datasetLoss, forward, initParams, makeDataset, targetsFor, trainStep, zeroOptState,
  type Act, type DatasetId, type InitScheme, type LossKind, type NetConfig, type OptConfig, type OptState, type Params, type ParamRef, type Sample,
} from '../../engine/nn'
import { digitTrain } from '../../engine/digits'
import { MAX_LAYERS, PRESETS, microSteps, presetById, repairConfig, selectionValid, type PresetId, type Selection, type TimelineMode } from './model'

export interface Snapshot { params: Params; opt: OptState; epoch: number; iter: number; cursor: number; loss: number; kind: 'init' | 'micro' | 'train' | 'edit'; note: string }
export type LabTab = 'microscope' | 'check' | 'loss' | 'train' | 'digits' | 'activations' | 'vanishing' | 'challenge' | 'lesson'
export type EdgeMode = 'auto' | 'weights' | 'grads'

export interface MlpState {
  preset: PresetId | 'custom'
  cfg: NetConfig
  init: { scheme: InitScheme; seed: number; scale: number }
  dataset: DatasetId
  dataSeed: number
  custom: { x: number[]; y: number[] }
  data: Sample[]
  sampleIdx: number
  /** a sample shown in the microscope instead of data[sampleIdx] (e.g. a digit you drew) */
  probe: Sample | null
  opt: OptConfig
  batch: number
  trainSeed: number
  /** working parameters (may contain unrecorded manual edits) */
  params: Params
  history: Snapshot[]
  hIndex: number
  selection: Selection | null
  pathIdx: number
  micro: number
  mode: TimelineMode
  playing: boolean
  speed: number
  training: boolean
  tab: LabTab
  edgeMode: EdgeMode
  compare: boolean
  view3d: boolean
  notes: string[]
  challenge: { seed: number; answers: Record<string, { given: string; ok: boolean }> }

  loadPreset: (id: PresetId) => void
  setActivation: (layer: number, act: Act) => void
  setLoss: (loss: LossKind) => void
  setNeurons: (layer: number, n: number) => void
  addLayer: () => void
  removeLayer: (layer: number) => void
  setInit: (p: Partial<MlpState['init']>) => void
  reinit: () => void
  setDataset: (d: DatasetId) => void
  setDataSeed: (s: number) => void
  setSample: (i: number) => void
  setProbe: (s: Sample | null) => void
  setCustom: (which: 'x' | 'y', i: number, v: number) => void
  setOpt: (p: Partial<OptConfig>) => void
  setBatch: (n: number) => void
  select: (s: Selection | null) => void
  setPathIdx: (i: number) => void
  setParamValue: (ref: ParamRef, v: number) => void
  recordEdits: () => void
  revertEdits: () => void
  setMicro: (i: number) => void
  next: () => void
  prev: () => void
  resetMicro: () => void
  togglePlay: () => void
  setPlaying: (v: boolean) => void
  setSpeed: (v: number) => void
  setMode: (m: TimelineMode) => void
  commitUpdate: () => void
  train: (steps: number) => void
  setTraining: (v: boolean) => void
  rewind: (h: number) => void
  setTab: (t: LabTab) => void
  setEdgeMode: (m: EdgeMode) => void
  setCompare: (v: boolean) => void
  setView3d: (v: boolean) => void
  dismissNotes: () => void
  setChallengeSeed: (seed: number) => void
  answer: (id: string, given: string, ok: boolean) => void
}

export const MAX_HISTORY = 1500

export function buildData(cfg: NetConfig, dataset: DatasetId, seed: number, custom: { x: number[]; y: number[] }): Sample[] {
  const nIn = cfg.sizes[0], nOut = cfg.sizes[cfg.sizes.length - 1]
  if (dataset === 'digits') return digitTrain()
  if (dataset === 'worked') return [{ x: custom.x.slice(0, nIn), y: custom.y.slice(0, nOut), label: nOut === 1 ? (custom.y[0] >= 0.5 ? 1 : 0) : custom.y.indexOf(Math.max(...custom.y)) }]
  return targetsFor(makeDataset(dataset, seed, nIn, 80, nOut), nOut)
}
/** default custom sample for a configuration: the worked example when it fits, otherwise a simple pattern */
export function defaultCustom(cfg: NetConfig): { x: number[]; y: number[] } {
  const nIn = cfg.sizes[0], nOut = cfg.sizes[cfg.sizes.length - 1]
  const x = nIn === 2 ? WORKED.x.slice() : Array.from({ length: nIn }, (_, i) => +(((i % 5) + 1) * 0.2).toFixed(1))
  const y = nOut === 1 ? [1] : Array.from({ length: nOut }, (_, k) => (k === 0 ? 1 : 0))
  return { x, y }
}
/** a 2-D dataset only fits two inputs; anything else falls back to random samples */
const fitDataset = (d: DatasetId, cfg: NetConfig): DatasetId =>
  d === 'digits' ? (cfg.sizes[0] === 64 && cfg.sizes[cfg.sizes.length - 1] === 10 ? d : 'random') : d === 'worked' || d === 'random' || cfg.sizes[0] === 2 ? d : 'random'

const snap = (cfg: NetConfig, data: Sample[], params: Params, opt: OptState, epoch: number, iter: number, cursor: number, kind: Snapshot['kind'], note: string): Snapshot =>
  ({ params, opt, epoch, iter, cursor, loss: datasetLoss(cfg, params, data), kind, note })

/** keep the history bounded: drop every other recorded training step from the oldest half */
function pushHistory(h: Snapshot[], hIndex: number, s: Snapshot): { history: Snapshot[]; hIndex: number } {
  let out = h.slice(0, hIndex + 1)
  out.push(s)
  if (out.length > MAX_HISTORY) {
    const half = Math.floor(out.length / 2)
    out = [...out.slice(0, half).filter((x, k) => k === 0 || x.kind !== 'train' || k % 2 === 0), ...out.slice(half)]
  }
  return { history: out, hIndex: out.length - 1 }
}

export function createMlpStore(presetId: PresetId = 'worked') {
  return createStore<MlpState>()((set, get) => {
    /** rebuild everything that depends on the architecture */
    const rebuild = (cfg: NetConfig, over: Partial<MlpState> = {}): Partial<MlpState> => {
      const s = { ...get(), ...over }
      const custom = over.custom ?? (s.custom && s.custom.x.length === cfg.sizes[0] && s.custom.y.length === cfg.sizes[cfg.sizes.length - 1] ? s.custom : defaultCustom(cfg))
      const dataset = fitDataset(s.dataset, cfg)
      const data = buildData(cfg, dataset, s.dataSeed, custom)
      const params = over.params ?? initParams(cfg.sizes, s.init.seed, s.init.scheme, s.init.scale)
      const first = snap(cfg, data, params, zeroOptState(params), 0, 0, 0, 'init', over.params ? 'preset weights' : `${s.init.scheme} init · seed ${s.init.seed}`)
      const L = cfg.sizes.length - 1
      const selection = selectionValid(s.selection, cfg.sizes) ? s.selection : { kind: 'w' as const, layer: 1, j: 0, i: 0 }
      const probe = s.probe && s.probe.x.length === cfg.sizes[0] && s.probe.y.length === cfg.sizes[L] ? s.probe : null
      return { ...over, probe, cfg, custom, dataset, data, params, history: [first], hIndex: 0, selection, pathIdx: 0, micro: Math.min(s.micro, microSteps(L, s.mode).length - 1), sampleIdx: Math.min(s.sampleIdx, data.length - 1), playing: false, training: false }
    }
    const fromPreset = (id: PresetId): Partial<MlpState> => {
      const p = presetById(id)
      const cfg = p.cfg
      const custom = id === 'worked' ? { x: WORKED.x.slice(), y: WORKED.y.slice() } : defaultCustom(cfg)
      const data = buildData(cfg, p.dataset, 1, custom)
      const params = p.params ? cloneParams(p.params) : initParams(cfg.sizes, p.init.seed, p.init.scheme, p.init.scale)
      return {
        preset: id, probe: null, cfg, init: { ...p.init }, dataset: p.dataset, dataSeed: 1, custom, data, sampleIdx: 0, opt: { ...p.opt }, batch: p.batch, params,
        history: [snap(cfg, data, params, zeroOptState(params), 0, 0, 0, 'init', p.params ? 'preset weights' : `${p.init.scheme} init · seed ${p.init.seed}`)], hIndex: 0,
        selection: { kind: 'w', layer: 1, j: 0, i: 0 }, pathIdx: 0, micro: 0, playing: false, training: false, notes: [],
      }
    }
    const changeCfg = (raw: NetConfig, prefer?: { loss?: LossKind }) => {
      const { cfg, notes } = repairConfig(raw, prefer)
      set({ ...rebuild(cfg), preset: 'custom', notes })
    }
    const steps = () => microSteps(get().cfg.sizes.length - 1, get().mode)
    const current = () => get().history[get().hIndex]

    return {
      ...(fromPreset(presetId) as MlpState),
      trainSeed: 1, challenge: { seed: 1, answers: {} }, mode: 'iteration', speed: 1, tab: 'microscope', edgeMode: 'auto', compare: false, view3d: false,
      dataSeed: 1,

      loadPreset: (id) => set(fromPreset(id)),
      setActivation: (layer, act) => { const c = get().cfg; const acts = c.acts.slice(); acts[layer - 1] = act; changeCfg({ ...c, acts }) },
      setLoss: (loss) => changeCfg(get().cfg, { loss }),
      setNeurons: (layer, n) => { const c = get().cfg; const sizes = c.sizes.slice(); sizes[layer] = n; changeCfg({ ...c, sizes }) },
      addLayer: () => {
        const c = get().cfg
        if (c.sizes.length >= MAX_LAYERS + 1) return
        const L = c.sizes.length - 1
        const sizes = [...c.sizes.slice(0, L), Math.min(4, Math.max(2, c.sizes[L - 1] || 2)), c.sizes[L]]
        const acts = [...c.acts.slice(0, L - 1), L > 1 ? c.acts[L - 2] : 'tanh', c.acts[L - 1]]
        changeCfg({ ...c, sizes, acts })
      },
      removeLayer: (layer) => {
        const c = get().cfg
        if (layer < 1 || layer >= c.sizes.length - 1) return
        changeCfg({ ...c, sizes: c.sizes.filter((_, l) => l !== layer), acts: c.acts.filter((_, l) => l !== layer - 1) })
      },
      setInit: (p) => set({ init: { ...get().init, ...p } }),
      reinit: () => set({ ...rebuild(get().cfg), preset: get().preset === 'worked' ? 'custom' : get().preset }),
      setDataset: (d) => {
        const s = get(), dataset = fitDataset(d, s.cfg)
        const data = buildData(s.cfg, dataset, s.dataSeed, s.custom)
        const h = s.history.map((x) => ({ ...x, loss: datasetLoss(s.cfg, x.params, data) }))
        set({ dataset, data, sampleIdx: 0, probe: null, history: h, notes: dataset !== d ? [d === 'digits' ? 'The digits need 64 inputs and 10 outputs; using random samples instead.' : '2-D datasets need exactly two inputs; using random samples instead.'] : [] })
      },
      setDataSeed: (seed) => { const s = get(); const data = buildData(s.cfg, s.dataset, seed, s.custom); set({ dataSeed: seed, data, sampleIdx: Math.min(s.sampleIdx, data.length - 1), history: s.history.map((x) => ({ ...x, loss: datasetLoss(s.cfg, x.params, data) })) }) },
      setSample: (i) => set({ sampleIdx: Math.max(0, Math.min(get().data.length - 1, i)), probe: null }),
      setProbe: (probe) => set({ probe }),
      setCustom: (which, i, v) => {
        const s = get()
        if (!Number.isFinite(v)) return
        const custom = { x: s.custom.x.slice(), y: s.custom.y.slice() }
        custom[which][i] = v
        const data = s.dataset === 'worked' ? buildData(s.cfg, 'worked', s.dataSeed, custom) : s.data
        set({ custom, data, history: s.dataset === 'worked' ? s.history.map((x) => ({ ...x, loss: datasetLoss(s.cfg, x.params, data) })) : s.history })
      },
      setOpt: (p) => set({ opt: { ...get().opt, ...p } }),
      setBatch: (n) => set({ batch: Math.max(1, Math.round(n)) }),
      select: (selection) => set({ selection, pathIdx: 0 }),
      setPathIdx: (pathIdx) => set({ pathIdx }),
      setParamValue: (ref, v) => {
        if (!Number.isFinite(v)) return
        const q = cloneParams(get().params)
        if (ref.kind === 'w') q.W[ref.layer - 1][ref.j][ref.i] = v; else q.b[ref.layer - 1][ref.j] = v
        set({ params: q })
      },
      recordEdits: () => {
        const s = get(), c = current()
        set({ ...pushHistory(s.history, s.hIndex, snap(s.cfg, s.data, cloneParams(s.params), c.opt, c.epoch, c.iter, c.cursor, 'edit', 'manual weight edit')), compare: false })
      },
      revertEdits: () => set({ params: cloneParams(current().params), compare: false }),
      setMicro: (i) => set({ micro: Math.max(0, Math.min(steps().length - 1, i)), playing: false }),
      next: () => {
        const s = get(), n = steps().length
        if (s.micro < n - 1) set({ micro: s.micro + 1, playing: false })
        else if (s.mode === 'iteration') get().commitUpdate()
      },
      prev: () => set({ micro: Math.max(0, get().micro - 1), playing: false }),
      resetMicro: () => set({ micro: 0, playing: false }),
      togglePlay: () => { const s = get(); set({ playing: !s.playing, micro: !s.playing && s.micro >= steps().length - 1 && s.mode !== 'iteration' ? 0 : s.micro }) },
      setPlaying: (playing) => set({ playing }),
      setSpeed: (speed) => set({ speed: Math.max(0.2, Math.min(6, speed)) }),
      setMode: (mode) => { set({ mode }); set({ micro: Math.min(get().micro, steps().length - 1) }) },
      /** apply the update of the displayed sample (online SGD / the chosen optimizer) and start the next iteration */
      commitUpdate: () => {
        const s = get(), c = current(), d = shownSample(s)
        const t = forward(s.cfg, s.params, d.x), g = backward(s.cfg, s.params, t, d.y)
        const u = applyUpdate(s.params, g, c.opt, s.opt)
        set({ ...pushHistory(s.history, s.hIndex, snap(s.cfg, s.data, u.params, u.state, c.epoch, c.iter + 1, c.cursor, 'micro', `update on sample ${s.sampleIdx + 1}`)), params: u.params, micro: 0, compare: false })
      },
      /** mini-batch training steps (seeded shuffles); every step is recorded so it can be rewound */
      train: (n) => {
        const s = get(), c = current()
        let ts = { params: s.params, opt: c.opt, epoch: c.epoch, iter: c.iter, cursor: c.cursor }
        let h = { history: s.history, hIndex: s.hIndex }
        for (let k = 0; k < n; k++) {
          ts = trainStep(s.cfg, ts, s.data, s.opt, s.batch, s.trainSeed).state
          h = pushHistory(h.history, h.hIndex, snap(s.cfg, s.data, ts.params, ts.opt, ts.epoch, ts.iter, ts.cursor, 'train', `batch of ${Math.min(s.batch, s.data.length)}`))
        }
        set({ ...h, params: ts.params, compare: false })
      },
      setTraining: (training) => set({ training }),
      rewind: (h) => { const s = get(); const i = Math.max(0, Math.min(s.history.length - 1, h)); set({ hIndex: i, params: cloneParams(s.history[i].params), training: false, compare: false }) },
      setTab: (tab) => set({ tab }),
      setEdgeMode: (edgeMode) => set({ edgeMode }),
      setCompare: (compare) => set({ compare }),
      setView3d: (view3d) => set({ view3d }),
      dismissNotes: () => set({ notes: [] }),
      setChallengeSeed: (seed) => set({ challenge: { seed: Math.max(0, Math.round(seed)), answers: {} } }),
      answer: (id, given, ok) => { const c = get().challenge; set({ challenge: { ...c, answers: { ...c.answers, [id]: { given, ok } } } }) },
    }
  })
}

export const mlpStore = createMlpStore()
export const useMlp = <T,>(sel: (s: MlpState) => T) => useStore(mlpStore, sel)
export const presetsList = PRESETS
/** working parameters differ from the recorded snapshot (manual edits not yet recorded) */
export const isModified = (s: Pick<MlpState, 'params' | 'history' | 'hIndex'>) => {
  const r = s.history[s.hIndex].params
  return s.params.W.some((L, l) => L.some((row, j) => row.some((w, i) => w !== r.W[l][j][i]))) || s.params.b.some((v, l) => v.some((x, j) => x !== r.b[l][j]))
}

/** the sample the microscope shows */
export const shownSample = (s: Pick<MlpState, 'probe' | 'data' | 'sampleIdx'>): Sample => s.probe ?? s.data[Math.min(s.sampleIdx, s.data.length - 1)]
