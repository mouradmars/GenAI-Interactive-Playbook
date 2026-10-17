import { describe, expect, it } from 'vitest'
import { runBlock, softmaxRow, vdot } from '../../engine/block'
import { checkAnswer, makeChallenges } from './challenges'
import { EXPLAIN, firstStageOf, stagesFor } from './stages'
import { cfgOf, createTbStore } from './store'

describe('stages', () => {
  it('GPT pre-LN has 19 stages including LN₁ and LN₂; BERT post-LN has 17, with LN after each residual', () => {
    const a = stagesFor('gpt').map((s) => s.id), b = stagesFor('bert').map((s) => s.id)
    expect(a.length).toBe(19); expect(b.length).toBe(17)
    expect(a.indexOf('ln1')).toBeLessThan(a.indexOf('qkv')); expect(b).not.toContain('ln1'); expect(b).not.toContain('ln2')
    expect(stagesFor('bert').find((s) => s.id === 'res1')!.tex).toContain('\\mathrm{LN}_1(X + ')
    expect(firstStageOf('gpt', 'mhsa')).toBe(a.indexOf('qkv'))
    for (const id of a) expect(EXPLAIN[id].remember.length).toBeGreaterThan(20)
  })
})

describe('store', () => {
  it('dimension changes keep shapes valid (H divides d) and rebuild values', () => {
    const st = createTbStore()
    st.getState().setDims({ d: 6, H: 4 })
    expect(st.getState().d % st.getState().H).toBe(0)
    expect(st.getState().params.Wq.length).toBe(6); expect(st.getState().X[0].length).toBe(6)
    st.getState().setDims({ T: 7 })
    expect(st.getState().tokens).toEqual(['The', 'cat', 'sat', 'on', 'the', 'mat', '.'])
    const s = st.getState(); expect(() => runBlock(cfgOf(s), s.params, s.X)).not.toThrow()
  })
  it('renaming a token changes its upstream embedding; the same label gives the same embedding row', () => {
    const st = createTbStore()
    const x0 = st.getState().X[1]
    st.getState().setToken(1, 'dog')
    expect(st.getState().X[1]).not.toEqual(x0)
    st.getState().setToken(1, 'cat')
    expect(st.getState().X[1]).toEqual(x0)
  })
  it('preset A is causal unless the comparison toggle is on; preset B is never causal', () => {
    const st = createTbStore()
    expect(cfgOf(st.getState()).causal).toBe(true)
    st.getState().setMaskOff(true); expect(cfgOf(st.getState()).causal).toBe(false)
    st.getState().setArch('bert'); expect(cfgOf(st.getState()).causal).toBe(false); expect(cfgOf(st.getState()).arch).toBe('post')
    expect(st.getState().maskOff).toBe(false)
  })
})

describe('challenges', () => {
  const st = createTbStore(), s = st.getState()
  const toy = { cfg: cfgOf(s), params: s.params, X: s.X, tokens: s.tokens }
  it('ten levels, deterministic per seed', () => {
    const a = makeChallenges(3, toy)
    expect(a.map((c) => c.level)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expect(makeChallenges(3, toy)).toEqual(a)
    expect(makeChallenges(4, toy).map((c) => c.answer)).not.toEqual(a.map((c) => c.answer))
  })
  it('answers are the engine values, and hand-rounded inputs are accepted', () => {
    for (let seed = 0; seed < 25; seed++) {
      const qs = makeChallenges(seed, toy)
      const dot = qs.find((c) => c.id === 'dot')!, m = /q = \[(.*?)\] and k = \[(.*?)\]/.exec(dot.prompt)!
      expect(dot.answer).toBeCloseTo(vdot(m[1].split(', ').map(Number), m[2].split(', ').map(Number)), 12)
      const sm = qs.find((c) => c.id === 'softmax')!, z = /\[(.*?)\]/.exec(sm.prompt)![1].split(', ').map(Number)
      expect(sm.answer).toEqual(softmaxRow(z).p)
      for (const q of qs) {
        if (q.kind === 'num') expect(checkAnswer(q, (q.answer as number).toFixed(3))).toBe(true)
        if (q.kind === 'vec') expect(checkAnswer(q, (q.answer as number[]).map((v) => v.toFixed(3)).join(', '))).toBe(true)
        if (q.kind === 'num') expect(checkAnswer(q, String((q.answer as number) + 1))).toBe(false)
      }
      const blk = qs.find((c) => c.id === 'block')!
      const t = runBlock(toy.cfg, toy.params, toy.X)
      expect(t.Y.flat()).toContain(blk.answer)
    }
  })
  it('answer formats: shapes, sets (any order), multiple choice, minus signs', () => {
    const qs = makeChallenges(1, toy)
    const shape = qs.find((c) => c.id === 'shapes')!
    expect(checkAnswer(shape, (shape.answer as string).replace('×', 'x'))).toBe(true)
    expect(checkAnswer(shape, `[${(shape.answer as string).replace('×', ', ')}]`)).toBe(true)
    const mask = qs.find((c) => c.id === 'mask')!, ans = mask.answer as number[]
    expect(checkAnswer(mask, ans.slice().reverse().join(' '))).toBe(true)
    expect(checkAnswer(mask, ans.slice(0, -1).join(','))).toBe(false)
    const mc = qs.find((c) => c.kind === 'mc')!
    expect(checkAnswer(mc, mc.answer as string)).toBe(true); expect(checkAnswer(mc, mc.options![1])).toBe(false)
    expect(checkAnswer({ ...qs[1], answer: -0.25 }, '−0.25')).toBe(true)
  })
})
