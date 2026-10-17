import type { StageId } from './lesson'
import type { Regime, SimConfig } from '../../engine/seq'

export interface SeqExample {
  id: string; title: string; stage: StageId; source: 'trained' | 'sim'; regime?: Regime; sim?: Partial<SimConfig>; seq: string; why: string
}

export const SEQ_EXAMPLES: SeqExample[] = [
  {
    id: 'long-fail', title: 'Trained only on long sequences, the RNN learns nothing', stage: 'lengths', source: 'trained', regime: 'long', seq: 'cabdbadcbdacbdbacdba',
    why: 'Real training results. With every training sequence 15–25 symbols long, the vanilla RNN’s loss never leaves ln 4 and its accuracy stays at chance at every length. The LSTM and the attention model, same size and same data, both learn.',
  },
  {
    id: 'vanish', title: 'Vanishing and exploding gradients (simulation)', stage: 'gradients', source: 'sim', sim: { scale: 0.5, forgetBias: 1 }, seq: 'abcdabcdabcdabcdabcd',
    why: 'Seeded random weights. At recurrent scale 0.5 the RNN’s gradient reaching x₁ is about 10⁻⁹ of the one reaching x₂₀. Drag the scale to 2.5 and the decay disappears, then exceeds 1 (exploding). Raise the LSTM forget bias to see its cell path keep the signal.',
  },
  {
    id: 'caveat', title: 'Attention that points at the wrong place (simulation)', stage: 'caveat', source: 'sim', sim: { seed: 1, scale: 2 }, seq: 'cbdaabdc',
    why: 'In this random-weight attention model 95% of the last position’s attention goes to position 2, but occlusion shows that the last symbol, reached through the residual path, moves the output most. Then switch to the trained model, where attention and occlusion agree.',
  },
  {
    id: 'saturated', title: 'A zero gradient that still matters (trained RNN)', stage: 'caveat', source: 'trained', regime: 'mixed', seq: 'cabdbadcbdacbdbacdba',
    why: 'The trained mixed-regime RNN latches the first symbol into a saturated state. Its gradient with respect to x₁ is around 10⁻²⁶, yet replacing x₁ changes the answer every time. Gradients are local slopes, not measures of influence.',
  },
]

export const DEFAULT_SEQ = 'cabdbadcbda'
