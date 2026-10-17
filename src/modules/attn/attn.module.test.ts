import { PRESET_PARAMS, PRESET_X, defaultConfig, runAttention, stableSoftmax } from '../../engine/attention'
import { EXPERIMENTS } from './experiments'
import { check, makeChallenges, shuffle } from './challenges'
import { FOLLOW_IDS, FOLLOW_STAGE, followSteps } from './follow'
import { substitution } from './Inspector'
import { calcSheet, exportPayload } from './exportSheet'
import { realAttention } from './realModel'
import { STAGES } from './stages'
import { LESSONS } from './Lessons'
import { createAttnStore, embedInput, validateValue, windowTokens } from './store'

const base = () => ({ X: PRESET_X.map((r) => r.slice()), params: JSON.parse(JSON.stringify(PRESET_PARAMS)), cfg: defaultConfig(4) })
const trace = () => runAttention(PRESET_X, PRESET_PARAMS, defaultConfig(4))
const tokens = ['cat', 'chased', 'the', 'mouse']

describe('module 03 · store', () => {
  it('starts on the 4-token preset window “cat chased the mouse” with causal, scaled, 2-head attention', () => {
    const s = createAttnStore().getState()
    expect(windowTokens(s.sentence, s.windowStart, s.T)).toEqual(tokens)
    expect(s.X).toEqual(PRESET_X)
    expect([s.H, s.causal, s.scale]).toEqual([2, true, true])
    expect(s.sel).toMatchObject({ i: 2, j: 0, h: 0 })
  })
  it('validates edits and writes head-specific weights into the right column block', () => {
    const st = createAttnStore()
    expect(st.getState().setW('Wq', 0, 0, NaN)).toMatch(/number/)
    expect(st.getState().setW('Wq', 0, 0, 99)).toMatch(/±10/)
    expect(st.getState().setW('Wv', 1, 3, 2.5)).toBeNull()
    expect(st.getState().params.Wv[1][3]).toBe(2.5)
    expect(st.getState().paramsEdited).toBe(true)
    expect(validateValue(-3)).toBeNull()
    expect(st.getState().setX(0, 0, Infinity)).toMatch(/number/)
  })
  it('pads short windows with <pad> tokens and masks them automatically', () => {
    const st = createAttnStore()
    st.getState().setSentence('Hello world')
    st.getState().setWindow(0, 4)
    expect(windowTokens(st.getState().sentence, 0, 4)).toEqual(['Hello', 'world', '<pad>', '<pad>'])
    expect(st.getState().pad).toEqual([false, false, true, true])
    const t = runAttention(st.getState().X, st.getState().params, { ...defaultConfig(4), pad: st.getState().pad })
    t.heads.forEach((h) => h.A.forEach((r) => { expect(r[2]).toBe(0); expect(r[3]).toBe(0) }))
  })
  it('only valid head counts are accepted; switching resets the head switches', () => {
    const st = createAttnStore()
    st.getState().setH(3)
    expect(st.getState().H).toBe(2)
    st.getState().setH(4)
    expect(st.getState().H).toBe(4)
    expect(st.getState().headsOn).toEqual([true, true, true, true])
  })
  it('custom knock-outs toggle and clear; baselines save and restore', () => {
    const st = createAttnStore()
    st.getState().toggleBlocked(3, 1)
    expect(st.getState().blocked?.[3][1]).toBe(true)
    st.getState().toggleBlocked(3, 1)
    expect(st.getState().blocked).toBeNull()
    st.getState().saveBaseline('b')
    st.getState().setW('Wo', 0, 0, -1)
    st.getState().setCausal(false)
    st.getState().restoreBaseline()
    expect(st.getState().params.Wo[0][0]).toBe(1)
    expect(st.getState().causal).toBe(true)
  })
  it('the embedding source builds X = e(token) + PE(position) and ignores nothing', () => {
    const { E, P, X } = embedInput(['the', 'cat', 'the'], 1)
    expect(E[0]).toEqual(E[2])
    expect(P[0]).not.toEqual(P[2])
    X.forEach((r, i) => r.forEach((v, k) => expect(v).toBeCloseTo(E[i][k] + P[i][k], 15)))
  })
})

describe('module 03 · stages, follow mode and inspector', () => {
  it('thirteen stages in pipeline order, each with an equation, three depths and misconceptions', () => {
    expect(STAGES.map((s) => s.id)).toEqual(['tokens', 'embed', 'qkv', 'dot', 'scale', 'mask', 'softmax', 'heatmap', 'aggregate', 'heads', 'concat', 'proj', 'output'])
    for (const s of STAGES) { expect(s.tex.length).toBeGreaterThan(5); expect(s.beginner && s.intermediate && s.advanced).toBeTruthy(); expect(s.misconceptions.length).toBeGreaterThan(0) }
  })
  it('follow one token: twelve steps whose numbers come from the trace', () => {
    const t = trace()
    const steps = followSteps(t, 2, 0, PRESET_PARAMS.Wo)
    expect(steps.map((s) => s.id)).toEqual(FOLLOW_IDS)
    FOLLOW_IDS.forEach((id) => expect(FOLLOW_STAGE[id]).toBeTruthy())
    expect(steps[1].numbers[1]).toContain(t.heads[0].Q[2][0].toFixed(3))
    expect(steps[7].numbers[0]).toContain(t.heads[0].A[2][0].toFixed(4))
    expect(steps[7].numbers[1]).toContain('1.000000')
    expect(steps[11].numbers[1]).toContain(t.out[2][0].toFixed(4))
    expect(steps[5].purpose).toMatch(/Allowed keys: 1, 2, 3/)
    // deterministic replay
    expect(followSteps(trace(), 2, 0, PRESET_PARAMS.Wo)).toEqual(steps)
  })
  it('the inspector substitution shows the real numbers of the selected pair', () => {
    const t = trace(), sel = { i: 2, j: 0, h: 0, cell: null }
    expect(substitution('dot', t, sel, PRESET_PARAMS.Wo).join(' ')).toContain('0.7000')
    expect(substitution('scale', t, sel, PRESET_PARAMS.Wo)[0]).toContain('0.4950')
    expect(substitution('softmax', t, sel, PRESET_PARAMS.Wo).join(' ')).toContain(t.heads[0].A[2][0].toFixed(4))
    expect(substitution('mask', t, { ...sel, j: 3 }, PRESET_PARAMS.Wo)[0]).toContain('causal')
    const fm = runAttention(PRESET_X, PRESET_PARAMS, { ...defaultConfig(4), pad: [true, false, false, false] })
    expect(substitution('softmax', fm, { ...sel, i: 0 }, PRESET_PARAMS.Wo)[0]).toMatch(/fully masked/)
  })
  it('the calculation sheet and JSON export contain every tensor and the worked example', () => {
    const t = trace()
    const md = calcSheet(t, tokens, { i: 2, j: 0, h: 0, cell: null })
    expect(md).toContain('s = 1.000×0.850 + 0.600×-0.250 = 0.7000')
    expect(md).toContain('## Multi-head output O')
    const st = createAttnStore().getState()
    const p = exportPayload(st, t, tokens, 'heatmap')
    const j = p.json as { heads: { A: number[][] }[]; mask: (number | string)[][]; provenance: string }
    expect(j.heads).toHaveLength(2)
    expect(j.mask[0][1]).toBe('-inf')
    expect(j.provenance).toMatch(/not trained/)
    expect(p.filenameBase).toBe('attention-heatmap-h2-causal')
  })
})

describe('module 03 · experiments', () => {
  const run = (id: string, h = 0) => EXPERIMENTS.find((e) => e.id === id)!.run(base(), h)
  it('there are eight experiments, each with hypothesis, prediction options including the answer, and an explanation', () => {
    expect(EXPERIMENTS).toHaveLength(8)
    for (const e of EXPERIMENTS) { expect(e.options).toContain(e.answer); expect(e.hypothesis && e.explanation).toBeTruthy() }
  })
  it('1 · changing W_Q changes the weights; 3 · changing W_V never does', () => {
    expect(run('query').metrics.find((m) => m.label === 'max |ΔA|')!.value).not.toBe('0.0000')
    const v = run('value')
    expect(v.metrics[0].value).toBe('0.0000')
    expect(v.observed).toMatch(/identical/)
    expect(v.metrics.find((m) => m.label === 'max |Δoutput|')!.value).not.toBe('0.0000')
  })
  it('2 · a W_K entry change leaves the scores of keys with x_j,2 = 0 untouched', () => {
    const r = run('key')
    expect(r.metrics.find((m) => m.label === 'key columns with changed scores')!.value).toBe('2, 3, 4')
  })
  it('4 · removing scaling lowers the mean entropy; 5 · the last row is unchanged without the causal mask', () => {
    const s = run('scaling')
    const m0 = s.before.heads[0].entropy.reduce((a, b) => a + b, 0), m1 = s.after.heads[0].entropy.reduce((a, b) => a + b, 0)
    expect(m1).toBeLessThan(m0)
    const c = run('causal')
    expect(c.metrics.find((m) => m.label === 'output rows that changed')!.value).toBe('1, 2, 3')
    expect(c.after.out[3]).toEqual(c.before.out[3])
  })
  it('6 · more heads, same parameters; 7 · ablation changes nothing in the other head; 8 · a new input changes the pattern', () => {
    const h = run('heads')
    expect(h.after.cfg.H).toBe(4)
    expect(h.metrics.find((m) => m.label.startsWith('parameters'))!.value).toBe('64 → 64')
    expect(run('ablate', 1).metrics[1].value).toBe('no')
    const x = run('inputs')
    expect(x.metrics[0].value).not.toBe('0.0000')
  })
  it('experiments are deterministic', () => {
    for (const e of EXPERIMENTS) expect(e.run(base(), 0).metrics).toEqual(e.run(base(), 0).metrics)
  })
})

describe('module 03 · challenges', () => {
  const qs = makeChallenges(1, trace(), tokens)
  it('three levels: 4 beginner, 5 intermediate, 6 advanced; same seed → same numbers; options shuffled but complete', () => {
    expect([1, 2, 3].map((l) => qs.filter((q) => q.level === l).length)).toEqual([4, 5, 6])
    expect(makeChallenges(1, trace(), tokens)).toEqual(qs)
    expect(makeChallenges(2, trace(), tokens).find((q) => q.id === 'i1')!.prompt).not.toBe(qs.find((q) => q.id === 'i1')!.prompt)
    for (const q of qs.filter((x) => x.options)) expect(q.options).toContain(q.answer)
    expect([...shuffle([1, 2, 3, 4], 5)].sort()).toEqual([1, 2, 3, 4])
  })
  it('every challenge accepts its own answer and rejects a wrong one', () => {
    for (const seed of [1, 2, 3, 7]) {
      for (const q of makeChallenges(seed, trace(), tokens)) {
        const right = typeof q.answer === 'string' ? q.answer : Array.isArray(q.answer) ? (q.kind === 'set' ? (q.answer.length ? q.answer.join(', ') : 'none') : q.answer.map((v) => (v === -Infinity ? '-inf' : String(v))).join(', ')) : String(q.answer)
        expect(check(q, right)).toBe(true)
        const wrong = q.kind === 'mc' ? q.options!.find((o) => o !== q.answer)! : q.kind === 'shape' ? '1×1' : q.kind === 'set' ? '99' : '12345'
        expect(check(q, wrong)).toBe(false)
      }
    }
  })
  it('answers are computed correctly', () => {
    const t = trace()
    const b3 = qs.find((q) => q.id === 'b3')!, qi = Number(b3.prompt.match(/query (\d)/)![1]) - 1
    expect(b3.answer).toBe(t.heads[1].A[qi].indexOf(Math.max(...t.heads[1].A[qi])) + 1)
    const i4 = qs.find((q) => q.id === 'i4')!
    const lg = i4.prompt.match(/\[(.*?)\]/)![1].split(', ').map(Number), k = Number(i4.prompt.match(/entry (\d)/)![1]) - 1
    expect(i4.answer).toBeCloseTo(stableSoftmax(lg).p[k], 12)
    expect(check(qs.find((q) => q.id === 'a1')!, qs.find((q) => q.id === 'a1')!.answer.toString().replace('×', 'x'))).toBe(true)
    expect(check({ ...qs.find((q) => q.id === 'i3')!, answer: [1, -Infinity] }, '1, −∞')).toBe(true)
  })
})

describe('module 03 · real model and lessons', () => {
  it('real-model attention is reproduced by this engine from the model\'s own q and k', () => {
    const r = realAttention('the cat sat in the', 1, 2)
    expect(r.tokens.length).toBeGreaterThan(3)
    expect(r.maxDiff).toBeLessThan(1e-5)
    r.A.forEach((row, i) => { expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5); row.forEach((p, j) => { if (j > i) expect(p).toBe(0) }) })
  })
  it('three lessons of 15, 25 and 20 minutes with discussion questions and demos', () => {
    expect(LESSONS.map((l) => l.minutes)).toEqual([15, 25, 20])
    for (const l of LESSONS) { expect(l.discussion.length).toBeGreaterThan(0); expect(l.segments.length).toBeGreaterThan(3) }
  })
})
