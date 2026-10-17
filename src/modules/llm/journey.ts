/** The ordered stages of one forward pass, derived from the model's shape. */
export type StepKind =
  | 'text' | 'bpe' | 'ids' | 'embed' | 'pos'
  | 'ln1' | 'qkv' | 'scores' | 'softmax' | 'mix' | 'mlp' | 'residual'
  | 'lnf' | 'logits' | 'probs' | 'decode'

export interface JourneyStep {
  id: string
  kind: StepKind
  layer?: number
  group: string
  title: string
  short: string
}

export function buildJourney(nLayers: number): JourneyStep[] {
  const s: JourneyStep[] = [
    { id: 'text', kind: 'text', group: 'Input', title: 'Raw text', short: 'Text' },
    { id: 'bpe', kind: 'bpe', group: 'Input', title: 'BPE tokenization', short: 'BPE' },
    { id: 'ids', kind: 'ids', group: 'Input', title: 'Token IDs', short: 'IDs' },
    { id: 'embed', kind: 'embed', group: 'Embedding', title: 'Token embedding lookup', short: 'Embed' },
    { id: 'pos', kind: 'pos', group: 'Embedding', title: 'Add positional embedding', short: '+Pos' },
  ]
  for (let l = 0; l < nLayers; l++) {
    const g = `Block ${l + 1}`
    s.push(
      { id: `ln1-${l}`, kind: 'ln1', layer: l, group: g, title: 'LayerNorm (pre-attention)', short: 'LN' },
      { id: `qkv-${l}`, kind: 'qkv', layer: l, group: g, title: 'Query, key, value projections', short: 'QKV' },
      { id: `scores-${l}`, kind: 'scores', layer: l, group: g, title: 'Scaled dot-product scores', short: 'Scores' },
      { id: `softmax-${l}`, kind: 'softmax', layer: l, group: g, title: 'Causal softmax → attention weights', short: 'Softmax' },
      { id: `mix-${l}`, kind: 'mix', layer: l, group: g, title: 'Mix values, project, add to residual', short: 'Mix' },
      { id: `mlp-${l}`, kind: 'mlp', layer: l, group: g, title: 'Feed-forward network (MLP)', short: 'MLP' },
      { id: `residual-${l}`, kind: 'residual', layer: l, group: g, title: 'Residual stream after block', short: 'Resid' },
    )
  }
  s.push(
    { id: 'lnf', kind: 'lnf', group: 'Output', title: 'Final LayerNorm', short: 'LN_f' },
    { id: 'logits', kind: 'logits', group: 'Output', title: 'Unembedding → logits', short: 'Logits' },
    { id: 'probs', kind: 'probs', group: 'Output', title: 'Softmax → next-token distribution', short: 'Probs' },
    { id: 'decode', kind: 'decode', group: 'Output', title: 'Decoding: choose and append', short: 'Decode' },
  )
  return s
}
