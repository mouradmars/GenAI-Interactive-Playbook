/** Preset experiments for the embedding lab. */
import type { SpaceId, StageId } from './lesson'
import type { ProjMethod } from '../../engine/embeddings'

export interface EmbExample {
  id: string
  title: string
  space: SpaceId
  stage: StageId
  method?: ProjMethod
  window?: number
  focus?: string
  analogy?: [string, string, string, string]
  why: string
}

export const EMB_EXAMPLES: EmbExample[] = [
  {
    id: 'royal', title: 'king − man + woman (hand-built)', space: 'toy', stage: 'arithmetic', focus: 'king',
    analogy: ['man', 'king', 'woman', 'queen'],
    why: 'An educational simulation: 15 words with 11 named features. The offset king − man is exactly “royal”, so adding it to woman lands exactly on queen (cosine 1.000). Use it to learn the mechanics, then see how much messier learned spaces are.',
  },
  {
    id: 'capitals', title: 'Capitals and countries (trained, window matters)', space: 'ppmi', stage: 'arithmetic', window: 4, focus: 'paris',
    analogy: ['paris', 'france', 'rome', 'italy'],
    why: 'PPMI-SVD vectors trained in your browser on 5,000 corpus sentences. With a ±4 window, paris : france :: rome : ? ranks “italy” first. Shrink the window to ±1 and italy drops to 8th: in this corpus a capital and its country are always 2–5 words apart, so a ±1 window never sees them together. Then try man : king :: woman : ?, where the trained space puts “princess” first and “queen” only 3rd.',
  },
  {
    id: 'distortion', title: 'A beautiful picture that lies', space: 'ppmi', stage: 'distortion', method: 'umap', window: 4, focus: 'three',
    why: 'UMAP draws tidy islands and keeps about 72% of each word’s 5 true neighbours, against about 49% for PCA. But its Shepard correlation is only about 0.18 (PCA about 0.27): distances between islands mean little. Click the false-friend pairs: some are drawn side by side although their cosine is negative.',
  },
  {
    id: 'context', title: 'One token, five sentences (real transformer)', space: 'nano', stage: 'context',
    why: 'The token “is” in five sentences, read out of Observatory-Nano at each depth. Identical at the input, identical for equal positions after adding position, still similar after block 1, and nearly unrelated after block 2, where each occurrence predicts a different next word.',
  },
]

export const DEFAULT_CONTEXT_SENTENCES = [
  'paris is the capital of france',
  'the sky is blue',
  'three plus one is four',
  'the capital of italy is rome',
  'the grass is green',
].join('\n')
