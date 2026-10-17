/**
 * Stage list (execution timeline) and the explanation panel content for module 02.
 * Two architectures with genuinely different orders: GPT-style pre-LN decoder and BERT-style post-LN encoder.
 */
export type ArchPreset = 'gpt' | 'bert'
export type NodeId = 'in' | 'ln1' | 'mhsa' | 'add1' | 'ln2' | 'ffn' | 'add2' | 'out'
export type StageId = 'x' | 'ln1' | 'qkv' | 'dot' | 'scores' | 'scale' | 'mask' | 'softmax' | 'weights' | 'av' | 'concat' | 'proj' | 'res1' | 'ln2' | 'ffn1' | 'act' | 'ffn2' | 'res2' | 'out'

export interface Stage { id: StageId; short: string; title: string; group: string; node: NodeId; tex: string }
export interface Explain { what: string; why: string; math: string; changes: string; remember: string }

export const ARCH_LABEL: Record<ArchPreset, string> = { gpt: 'Preset A · decoder-only, GPT-style, pre-LN, causal', bert: 'Preset B · encoder, BERT-style, post-LN, bidirectional' }
export const ARCH_SHORT: Record<ArchPreset, string> = { gpt: 'GPT-style · pre-LN · causal', bert: 'BERT-style · post-LN · bidirectional' }

const ATT: Omit<Stage, 'node'>[] = [
  { id: 'qkv', short: 'Q K V', title: 'Query, Key and Value projections', group: 'attention', tex: 'Q = X_{n}W_Q,\\; K = X_{n}W_K,\\; V = X_{n}W_V' },
  { id: 'dot', short: 'q·k', title: 'One query–key dot product', group: 'attention', tex: 's_{ij} = q_i \\cdot k_j = \\sum_c q_{ic}\\,k_{jc}' },
  { id: 'scores', short: 'QKᵀ', title: 'The attention score matrix', group: 'attention', tex: 'S = Q_h K_h^{\\top}' },
  { id: 'scale', short: '÷√dₖ', title: 'Scaling by √dₖ', group: 'attention', tex: 'S_{\\text{scaled}} = S / \\sqrt{d_k}' },
  { id: 'mask', short: 'mask', title: 'The attention mask', group: 'attention', tex: 'L = S_{\\text{scaled}} + M,\\quad M_{ij} = \\begin{cases}0 & \\text{allowed}\\\\ -\\infty & \\text{masked}\\end{cases}' },
  { id: 'softmax', short: 'softmax', title: 'Softmax turns scores into weights', group: 'attention', tex: 'A_{ij} = \\frac{e^{L_{ij} - m_i}}{\\sum_{j\'} e^{L_{ij\'} - m_i}},\\quad m_i = \\max_j L_{ij}' },
  { id: 'weights', short: 'A', title: 'The attention weights', group: 'attention', tex: 'A = \\mathrm{softmax}(QK^{\\top}/\\sqrt{d_k} + M)' },
  { id: 'av', short: 'AV', title: 'Weighted sum of Value vectors', group: 'attention', tex: 'O_i = \\sum_j A_{ij}\\, v_j' },
  { id: 'concat', short: 'concat', title: 'Concatenating the heads', group: 'multi-head', tex: '\\mathrm{Concat}(O_1, \\dots, O_H) \\in \\mathbb{R}^{T \\times d}' },
  { id: 'proj', short: 'W_O', title: 'Output projection', group: 'multi-head', tex: '\\mathrm{Attn} = \\mathrm{Concat}(O_1,\\dots,O_H)\\,W_O + b_O' },
]
const FFN: Omit<Stage, 'node'>[] = [
  { id: 'ffn1', short: 'W₁', title: 'Feedforward: expand', group: 'feedforward', tex: 'Z_1 = U W_1 + b_1' },
  { id: 'act', short: 'φ', title: 'Feedforward: activation', group: 'feedforward', tex: '\\varphi(Z_1)' },
  { id: 'ffn2', short: 'W₂', title: 'Feedforward: project back', group: 'feedforward', tex: '\\mathrm{MLP}(U) = \\varphi(Z_1) W_2 + b_2' },
]

export function stagesFor(a: ArchPreset): Stage[] {
  const att = ATT.map((s) => ({ ...s, node: (s.group === 'multi-head' || s.group === 'attention' ? 'mhsa' : 'mhsa') as NodeId }))
  const ffn = FFN.map((s) => ({ ...s, node: 'ffn' as NodeId }))
  if (a === 'gpt') return [
    { id: 'x', short: 'X', title: 'Hidden states enter the block', group: 'input', node: 'in', tex: 'X \\in \\mathbb{R}^{T \\times d}' },
    { id: 'ln1', short: 'LN₁', title: 'Layer normalization (before attention)', group: 'normalize', node: 'ln1', tex: '\\mathrm{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta' },
    ...att,
    { id: 'res1', short: '+', title: 'First residual addition', group: 'residual', node: 'add1', tex: 'H = X + \\mathrm{Attn}(\\mathrm{LN}_1(X))' },
    { id: 'ln2', short: 'LN₂', title: 'Layer normalization (before the MLP)', group: 'normalize', node: 'ln2', tex: 'U = \\mathrm{LN}_2(H)' },
    ...ffn,
    { id: 'res2', short: '+', title: 'Second residual addition', group: 'residual', node: 'add2', tex: 'Y = H + \\mathrm{MLP}(\\mathrm{LN}_2(H))' },
    { id: 'out', short: 'Y', title: 'Output hidden states', group: 'output', node: 'out', tex: 'Y \\in \\mathbb{R}^{T \\times d}' },
  ]
  return [
    { id: 'x', short: 'X', title: 'Hidden states enter the block', group: 'input', node: 'in', tex: 'X \\in \\mathbb{R}^{T \\times d}' },
    ...att,
    { id: 'res1', short: '+ LN₁', title: 'Residual addition, then LayerNorm', group: 'residual', node: 'add1', tex: 'H = \\mathrm{LN}_1(X + \\mathrm{Attn}(X))' },
    ...ffn,
    { id: 'res2', short: '+ LN₂', title: 'Residual addition, then LayerNorm', group: 'residual', node: 'add2', tex: 'Y = \\mathrm{LN}_2(H + \\mathrm{MLP}(H))' },
    { id: 'out', short: 'Y', title: 'Output hidden states', group: 'output', node: 'out', tex: 'Y \\in \\mathbb{R}^{T \\times d}' },
  ]
}
export const firstStageOf = (a: ArchPreset, n: NodeId) => stagesFor(a).findIndex((s) => s.node === n)

export const EXPLAIN: Record<StageId, Explain> = {
  x: {
    what: 'A matrix X with one row per token position enters the block. Each row is that token\'s current hidden-state vector.',
    why: 'The block works on vectors, not words. Tokenization, the embedding lookup and positional information all happen upstream, before the first block.',
    math: 'X ∈ ℝ^{T×d}: T token positions, d features each.',
    changes: 'Nothing yet: this is the starting point of the residual stream.',
    remember: 'A block receives and returns hidden states of the same shape. Positional embeddings are not re-added inside every block.',
  },
  ln1: {
    what: 'Each token vector is normalized across its own d features: subtract the mean, divide by the standard deviation, then scale by γ and shift by β.',
    why: 'Keeps the scale of the attention input under control regardless of how large the residual stream has grown, which helps optimization.',
    math: 'μ = (1/d)Σxᵢ, σ² = (1/d)Σ(xᵢ − μ)², LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β.',
    changes: 'Each row ends up with mean ≈ β̄ and spread ≈ γ; tokens are normalized independently of each other.',
    remember: 'LayerNorm normalizes over features within one token, not over the tokens of the sequence (that would be a different operation).',
  },
  qkv: {
    what: 'Three learned matrices map each normalized token vector to a Query, a Key and a Value vector. The d columns are split among the H heads.',
    why: 'Attention needs separate roles: queries are compared with keys to decide where to look; values are what gets passed along.',
    math: 'Q = X_n W_Q + b_Q, K = X_n W_K + b_K, V = X_n W_V + b_V; head h uses columns h·dₖ … (h+1)·dₖ − 1.',
    changes: 'Each token now has three new vectors per head, in spaces learned for matching and for carrying information.',
    remember: '“What a token seeks / offers / passes on” is an intuition for learned linear projections, not literal questions or labels.',
  },
  dot: {
    what: 'One query is compared with one key by a dot product: multiply matching coordinates and add.',
    why: 'The dot product is large when two vectors point in similar directions, so it is a cheap, differentiable similarity score.',
    math: 'sᵢⱼ = qᵢ · kⱼ = Σ_c q_{ic} k_{jc}.',
    changes: 'A single number per (query, key) pair: how strongly token i matches token j in this head.',
    remember: 'Every score comes from exactly one query row and one key row.',
  },
  scores: {
    what: 'All dot products at once form the T×T score matrix S = Q_h K_hᵀ for each head.',
    why: 'Every position can compare itself with every position in one matrix multiplication, which parallelizes well.',
    math: 'S = Q_h K_hᵀ ∈ ℝ^{T×T}; row i holds query i against every key.',
    changes: 'Raw, unnormalized similarities; they can be any real number.',
    remember: 'Rows are queries, columns are keys. The matrix is not symmetric in general because Q ≠ K.',
  },
  scale: {
    what: 'All scores are divided by √dₖ, the square root of the head dimension.',
    why: 'For random vectors, the variance of q·k grows with dₖ. Without scaling, large dₖ gives large logits and a softmax that saturates to almost one-hot, with tiny gradients.',
    math: 'S_scaled = S / √dₖ.',
    changes: 'Same ranking within each row, smaller magnitudes, softer softmax.',
    remember: 'Scaling changes how peaked the attention is, never which key scores highest.',
  },
  mask: {
    what: 'An additive mask is applied before softmax. In the decoder, every position j > i (the future) gets −∞; in the encoder nothing is masked.',
    why: 'A language model that predicts the next token must not see it. The causal mask enforces that every output row depends only on earlier positions.',
    math: 'L = S_scaled + M with M_{ij} = 0 if allowed, −∞ if masked; e^{−∞} = 0 after softmax.',
    changes: 'Future entries become −∞ and will receive exactly zero weight.',
    remember: 'Disabling the causal mask changes the mechanism and breaks autoregressive decoding: the model would cheat by reading the answer.',
  },
  softmax: {
    what: 'Each row of logits becomes a probability distribution: exponentiate, then divide by the row sum.',
    why: 'Turns arbitrary scores into non-negative weights that sum to 1, so the output is a weighted average.',
    math: 'Aᵢⱼ = exp(Lᵢⱼ − mᵢ) / Σⱼ′ exp(Lᵢⱼ′ − mᵢ), with mᵢ = maxⱼ Lᵢⱼ (subtracting the max avoids overflow and changes nothing).',
    changes: 'Each row sums to 1; masked entries are exactly 0; larger gaps between scores give more peaked rows.',
    remember: 'Softmax is per row (per query). Subtracting the row maximum is what makes it numerically stable.',
  },
  weights: {
    what: 'The attention weight matrix A: row i tells how much query token i takes from each key token.',
    why: 'This is the routing table of the head: it decides which positions contribute to each output.',
    math: 'A = softmax(QKᵀ/√dₖ + M), rows sum to 1.',
    changes: 'Nothing new is computed here; we inspect the weights the softmax produced.',
    remember: 'Attention weights show where a head reads from, not why the model makes a prediction. They are not an explanation on their own.',
  },
  av: {
    what: 'Each output row is the weighted sum of all permitted Value vectors, weighted by that row of A.',
    why: 'This is where information actually moves between positions.',
    math: 'Oᵢ = Σⱼ Aᵢⱼ vⱼ, i.e. O = A V_h.',
    changes: 'Token i now carries a mixture of the values of the positions it attends to.',
    remember: 'Only Values are mixed; Queries and Keys only decide the weights.',
  },
  concat: {
    what: 'The H head outputs (each T × dₖ) are placed side by side into one T × d matrix.',
    why: 'Each head used its own projections and attention pattern; concatenation keeps all of them.',
    math: 'Concat(O₁, …, O_H) ∈ ℝ^{T×(H·dₖ)} = ℝ^{T×d}.',
    changes: 'Shape returns to T × d, but the columns are grouped by head.',
    remember: 'Heads can learn different patterns, but a head does not necessarily correspond to a human-readable linguistic function.',
  },
  proj: {
    what: 'A learned d × d matrix W_O mixes the concatenated head outputs into the attention sublayer output.',
    why: 'Lets the heads\' results be combined and written back into the residual stream in a learned way.',
    math: 'Attn = Concat(O₁,…,O_H) W_O + b_O.',
    changes: 'Every output feature can now depend on every head.',
    remember: 'Without W_O each head could only write into its own block of dₖ features.',
  },
  res1: {
    what: 'The attention output is added to the block input, feature by feature (and, in post-LN, the sum is then normalized).',
    why: 'The residual path lets information and gradients pass through unchanged; the sublayer only needs to learn an update.',
    math: 'Pre-LN: H = X + Attn(LN₁(X)). Post-LN: H = LN₁(X + Attn(X)).',
    changes: 'The token vectors move by the attention contribution; nothing else is lost.',
    remember: 'Residual connections give an additive pathway; they help training but do not guarantee stability on their own.',
  },
  ln2: {
    what: 'The updated residual stream is normalized per token again, with its own γ and β.',
    why: 'Same reason as before: a well-scaled input for the MLP.',
    math: 'U = LN₂(H).',
    changes: 'Each row standardized; H itself is kept for the second residual addition.',
    remember: 'In pre-LN the normalized copy feeds the sublayer; the residual stream itself is never normalized inside the block.',
  },
  ffn1: {
    what: 'Each token vector is multiplied by W₁ (d → d_ff), expanding it to more features.',
    why: 'A wider hidden layer gives the MLP room to compute non-linear features of each token.',
    math: 'Z₁ = U W₁ + b₁ ∈ ℝ^{T×d_ff}.',
    changes: 'Each token now has d_ff pre-activation values.',
    remember: 'The same W₁ is used for every position: the MLP is position-wise.',
  },
  act: {
    what: 'A non-linearity is applied to every hidden value (GELU, ReLU or SiLU; or a SwiGLU gate in the gated variant).',
    why: 'Without it, W₁ and W₂ would collapse into one linear map.',
    math: 'φ(z) = max(0, z) · z Φ(z) (GELU) · z σ(z) (SiLU); SwiGLU: SiLU(UW₁) ⊙ (UW₃).',
    changes: 'Some hidden features are suppressed, others pass.',
    remember: 'SwiGLU is a different MLP architecture (three matrices), not just another activation.',
  },
  ffn2: {
    what: 'W₂ (d_ff → d) projects the hidden features back to the model dimension.',
    why: 'The output must match the residual stream so it can be added.',
    math: 'MLP(U) = φ(Z₁) W₂ + b₂ ∈ ℝ^{T×d}.',
    changes: 'One update vector per token, computed from that token alone.',
    remember: 'Attention mixes positions; the MLP transforms each position independently with shared weights.',
  },
  res2: {
    what: 'The MLP output is added to the residual stream (then normalized in post-LN).',
    why: 'Again an additive update rather than a replacement.',
    math: 'Pre-LN: Y = H + MLP(LN₂(H)). Post-LN: Y = LN₂(H + MLP(H)).',
    changes: 'Final hidden states of this block.',
    remember: 'Y = X + attention update + MLP update (in pre-LN): the block writes two additive updates into the stream.',
  },
  out: {
    what: 'The block outputs Y, the same shape as its input. It can feed the next block.',
    why: 'Stacking blocks repeatedly refines contextual representations.',
    math: 'Y ∈ ℝ^{T×d}.',
    changes: 'Each token\'s vector now reflects the other tokens it was allowed to attend to.',
    remember: 'Y is a hidden state, not a probability distribution. Next-token probabilities need a final LayerNorm, the unembedding (LM head) and a softmax.',
  },
}
