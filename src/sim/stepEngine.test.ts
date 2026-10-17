import { engineReducer, type EngineState } from './stepEngine'

const s0: EngineState = { index: 0, count: 5, playing: false, speed: 1 }
describe('step engine reducer', () => {
  it('clamps at both ends', () => {
    expect(engineReducer(s0, { type: 'prev' }).index).toBe(0)
    expect(engineReducer({ ...s0, index: 4 }, { type: 'next' }).index).toBe(4)
  })
  it('seek clamps and pauses', () => {
    expect(engineReducer({ ...s0, playing: true }, { type: 'seek', index: 99 })).toMatchObject({ index: 4, playing: false })
  })
  it('ticks while playing and stops at the end', () => {
    let s = engineReducer(s0, { type: 'play' })
    for (let i = 0; i < 10; i++) s = engineReducer(s, { type: 'tick' })
    expect(s).toMatchObject({ index: 4, playing: false })
  })
  it('play from the end restarts', () => {
    expect(engineReducer({ ...s0, index: 4 }, { type: 'play' }).index).toBe(0)
  })
  it('shrinking the step count keeps the index valid', () => {
    expect(engineReducer({ ...s0, index: 4 }, { type: 'count', count: 2 }).index).toBe(1)
  })
})
