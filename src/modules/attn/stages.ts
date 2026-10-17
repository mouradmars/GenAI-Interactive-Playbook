/** The thirteen operations of the attention pipeline: equations, explanations at three depths and common misconceptions. */
export type StageId = 'tokens' | 'embed' | 'qkv' | 'dot' | 'scale' | 'mask' | 'softmax' | 'heatmap' | 'aggregate' | 'heads' | 'concat' | 'proj' | 'output'
/** nodes of the computational pipeline (microscope C) */
export type NodeId = 'X' | 'W' | 'Q' | 'K' | 'V' | 'S' | 'Ss' | 'M' | 'A' | 'Oh' | 'H' | 'Wo' | 'O'

export interface Stage {
  id: StageId
  short: string
  /** label in the pipeline navigator */
  nav: string
  title: string
  group: string
  /** the active equation */
  tex: string
  purpose: string
  beginner: string
  intermediate: string
  advanced: string
  misconceptions: string[]
  /** pipeline nodes read and written by this operation */
  reads: NodeId[]
  writes: NodeId[]
}

export const STAGES: Stage[] = [
  { id: 'tokens', short: 'tokens', nav: 'Input tokens', title: 'Input tokens', group: 'input', tex: '\\text{text} \\rightarrow (t_1, t_2, \\dots, t_n)',
    purpose: 'Text is split into tokens; the attention layer works on one vector per token position.',
    beginner: 'A sentence becomes a short list of pieces. Each piece will get its own vector, and attention lets every piece look at the others.',
    intermediate: 'This lab uses a word-level toy tokenizer and a window of n positions. Real models use subword tokenizers (BPE, WordPiece), so the same text can give different tokens.',
    advanced: 'Attention itself is permutation-equivariant: without positional information, reordering the rows of X just reorders the outputs. Order must enter through the input representation (or through masks and relative-position schemes).',
    misconceptions: ['The token text does not reach attention: only the vectors do.', 'Tokens are not always words; this toy tokenizer is a simplification.'],
    reads: [], writes: ['X'] },
  { id: 'embed', short: 'X', nav: 'Input embeddings / hidden states', title: 'Input embeddings and hidden states', group: 'input', tex: 'X \\in \\mathbb{R}^{n \\times d_{\\text{model}}},\\quad x_i = \\text{row } i',
    purpose: 'Each position is a row vector. Stacked, they form the input matrix X.',
    beginner: 'Every token now has a list of 4 numbers. Rows are tokens, columns are features.',
    intermediate: 'In the first layer, X = token embedding + positional information. In deeper layers, X is the hidden state produced by the previous layer. Here X is configured by hand so it is easy to calculate with.',
    advanced: 'Row-vector convention: x_i ∈ R^{1×d}, projections multiply on the right (x_i W). This matches nn.Linear applied to rows (with W stored transposed in PyTorch).',
    misconceptions: ['These numbers are not "meanings" of the words: they were chosen by hand for teaching.', 'Two identical words at different positions can have different input vectors (positional information).'],
    reads: [], writes: ['X'] },
  { id: 'qkv', short: 'Q K V', nav: 'Q, K, V projections', title: 'The Q / K / V projection factory', group: 'project', tex: 'Q_h = X W_Q^{(h)},\\; K_h = X W_K^{(h)},\\; V_h = X W_V^{(h)}',
    purpose: 'Three learned linear maps turn every input vector into a query, a key and a value.',
    beginner: 'Each token makes three new vectors: a query (what it looks for), a key (how it can be found) and a value (what it passes on). This is only an analogy for learned numbers.',
    intermediate: 'Each output cell is a dot product of one input row with one weight column: q_{i,c} = Σ_k x_{i,k} w_{k,c}. Click any output cell to see the products.',
    advanced: 'With h heads, W_Q ∈ R^{d×h·d_k} is split into h column blocks W_Q^{(h)} ∈ R^{d×d_k}; each head has its own projections. Queries and keys only matter through their dot products, so Q and K can be rotated jointly (Q R, K R with R orthogonal) without changing attention.',
    misconceptions: ['Q, K and V are not literal questions, labels and answers: they are learned numbers.', 'Q, K and V are computed from the same X (in self-attention) but with different matrices.'],
    reads: ['X', 'W'], writes: ['Q', 'K', 'V'] },
  { id: 'dot', short: 'QKᵀ', nav: 'Query–key dot products', title: 'Query–key dot products', group: 'score', tex: 'S = Q K^{\\top},\\quad s_{ij} = q_i \\cdot k_j = \\sum_{c} q_{i,c}\\,k_{j,c}',
    purpose: 'Every query is compared with every key by a dot product: one compatibility score per pair.',
    beginner: 'For each pair (who looks, who is looked at) multiply matching numbers and add them up. A bigger number means a better match.',
    intermediate: 'Row i of S holds the scores of query i against all keys; column j holds the scores of key j against all queries. S is generally not symmetric because queries and keys use different matrices.',
    advanced: 'S = X W_Q W_Kᵀ Xᵀ: a bilinear form with a low-rank (rank ≤ d_k) matrix W_Q W_Kᵀ per head.',
    misconceptions: ['Raw scores are not probabilities: they can be negative and do not sum to 1.', 'S_ij is not the same as S_ji in general.'],
    reads: ['Q', 'K'], writes: ['S'] },
  { id: 'scale', short: '÷√dₖ', nav: 'Scaling ÷ √dₖ', title: 'Why scale by √dₖ?', group: 'score', tex: '\\tilde S = \\frac{Q K^{\\top}}{\\sqrt{d_k}}',
    purpose: 'Dividing by √d_k keeps the scores in a range where softmax is not saturated.',
    beginner: 'Longer vectors give bigger dot products. Dividing by √d_k keeps the numbers moderate so the next step does not become all-or-nothing.',
    intermediate: 'If the entries of q and k are independent with mean 0 and variance 1, q·k has variance d_k. Dividing by √d_k brings the variance back to 1.',
    advanced: 'Large logits push softmax towards a one-hot vector, where the Jacobian diag(p) − ppᵀ is close to 0 and gradients vanish. Scaling helps at initialization; it does not guarantee stable training (that also depends on normalization, learning rate, initialization…).',
    misconceptions: ['Scaling does not change which key has the highest score: it only changes how peaked the distribution is.', 'The √d_k argument assumes independent, unit-variance entries; trained models need not satisfy it.'],
    reads: ['S'], writes: ['Ss'] },
  { id: 'mask', short: 'mask', nav: 'Masking', title: 'Masking: causal, padding and custom', group: 'score', tex: 'Z = \\tilde S + M,\\quad M_{ij} = \\begin{cases} 0 & \\text{allowed} \\\\ -\\infty & \\text{masked} \\end{cases}',
    purpose: 'The mask decides which keys each query may use. Masked logits become −∞ and get probability 0.',
    beginner: 'When writing text one token at a time, a token must not peek at future tokens. Those connections are blocked.',
    intermediate: 'Causal mask: −∞ above the diagonal (j > i). Padding mask: −∞ in every column of a padding key. They combine by addition.',
    advanced: 'Masking happens before softmax, so the remaining weights renormalize among the permitted keys. A row with every key masked has no valid distribution; a naive softmax gives 0/0 = NaN. This engine returns zero weights and a zero output and flags the row.',
    misconceptions: ['Masking is not the same as setting weights to 0 after softmax (that would not renormalize).', 'The causal mask is for decoders; encoders (e.g. BERT) attend in both directions.'],
    reads: ['Ss', 'M'], writes: ['M'] },
  { id: 'softmax', short: 'softmax', nav: 'Softmax', title: 'The softmax microscope', group: 'normalize', tex: 'a_{ij} = \\frac{\\exp(z_{ij} - m_i)}{\\sum_{k \\in \\text{allowed}} \\exp(z_{ik} - m_i)},\\quad m_i = \\max_{k} z_{ik}',
    purpose: 'Each query row of logits becomes a probability distribution over the permitted keys.',
    beginner: 'Turn every row of scores into percentages that add up to 100 %. Higher scores get much more.',
    intermediate: 'Subtracting the row maximum does not change the result (it cancels in the fraction) but keeps exp from overflowing.',
    advanced: 'Softmax is applied per query row, over keys, because each query needs its own distribution; normalizing the whole matrix would make queries compete with each other.',
    misconceptions: ['Softmax is not applied to the whole matrix at once.', 'Adding the same constant to a whole row does not change the probabilities.'],
    reads: ['Ss', 'M'], writes: ['A'] },
  { id: 'heatmap', short: 'heatmap', nav: 'Attention heatmap', title: 'The attention heatmap', group: 'normalize', tex: 'A = \\operatorname{softmax}(Z) \\in \\mathbb{R}^{n \\times n},\\quad \\textstyle\\sum_j a_{ij} = 1',
    purpose: 'The weights as a picture: rows are queries, columns are keys, brightness is probability.',
    beginner: 'Read one row: it shows where that token "looks". Bright cells get a big share.',
    intermediate: 'Click a cell to see the dot product, the scaled score, the mask and the final probability behind it.',
    advanced: 'Attention weights show where information is read from in this layer and head. They do not, by themselves, explain a model\'s prediction (values, other heads, residual paths and later layers all matter).',
    misconceptions: ['An attention map is not an explanation of the model\'s reasoning.', 'Columns do not sum to 1, only rows do.'],
    reads: ['A'], writes: [] },
  { id: 'aggregate', short: 'A·V', nav: 'Weighted value aggregation', title: 'Weighted value aggregation', group: 'mix', tex: 'o_i = \\sum_j a_{ij}\\, v_j,\\qquad \\text{head} = A V',
    purpose: 'Each output is a weighted average of the value vectors, using that query\'s attention weights.',
    beginner: 'Mix the value vectors like paint: more of the ones with big weights, less of the others.',
    intermediate: 'For every output dimension c: o_{i,c} = Σ_j a_{ij} v_{j,c}. Because the weights sum to 1 and are non-negative, o_i lies inside the convex hull of the permitted values.',
    advanced: 'Q and K only decide the mixing weights; V decides what is mixed. Changing V (with Q and K fixed) changes the output but never the weights.',
    misconceptions: ['Attention does not select one token: it blends all permitted tokens.', 'The output is built from values, not from the keys that matched.'],
    reads: ['A', 'V'], writes: ['Oh'] },
  { id: 'heads', short: 'heads', nav: 'Multiple attention heads', title: 'Multi-head attention explorer', group: 'multi-head', tex: '\\text{head}_h = \\operatorname{softmax}\\!\\Big(\\frac{Q_h K_h^{\\top}}{\\sqrt{d_k}} + M\\Big) V_h,\\quad d_k = \\frac{d_{\\text{model}}}{h}',
    purpose: 'Several heads run in parallel, each with its own projections, and can attend in different ways.',
    beginner: 'Instead of one way of looking, the layer has several, side by side. Each head makes its own attention pattern.',
    intermediate: 'With d_model = 4 and h = 2, each head works in 2 dimensions. Editing one head\'s matrices changes only that head\'s computation.',
    advanced: 'Splitting d_model into h heads keeps the cost of the projections the same as one wide head, while allowing h different attention distributions per query. Head roles emerge from training; nothing forces a head to have a linguistic job.',
    misconceptions: ['Heads are not pre-assigned linguistic functions.', 'More heads do not mean more parameters here: d_k shrinks as h grows.'],
    reads: ['Q', 'K', 'V'], writes: ['Oh'] },
  { id: 'concat', short: 'concat', nav: 'Concatenation', title: 'Concatenating the heads', group: 'multi-head', tex: 'H = \\operatorname{Concat}(\\text{head}_1, \\dots, \\text{head}_h) \\in \\mathbb{R}^{n \\times h d_v}',
    purpose: 'The head outputs are placed side by side to form one wider matrix.',
    beginner: 'Glue the heads\' outputs together, row by row.',
    intermediate: 'Row i of H is [head₁ row i | head₂ row i]. Nothing is computed: values are only rearranged.',
    advanced: 'In tensor form: reshape (n, h, d_v) → (n, h·d_v). A disabled (ablated) head contributes a block of zeros.',
    misconceptions: ['Concatenation is not the end of multi-head attention: the output projection still follows.', 'Heads are not averaged.'],
    reads: ['Oh'], writes: ['H'] },
  { id: 'proj', short: 'W_O', nav: 'Output projection', title: 'The output projection', group: 'multi-head', tex: 'O = H W_O,\\quad W_O \\in \\mathbb{R}^{h d_v \\times d_{\\text{model}}}',
    purpose: 'A learned matrix mixes the heads\' information and maps it back to d_model.',
    beginner: 'One last learned mixing step combines what the heads found.',
    intermediate: 'Each output cell is a dot product of a row of H with a column of W_O, so every output feature can draw on every head.',
    advanced: 'Equivalently O = Σ_h head_h W_O^{(h)}, where W_O^{(h)} are the rows of W_O belonging to head h: each head writes into the output through its own block.',
    misconceptions: ['W_O is learned, not a fixed reshaping.', 'Without W_O, heads could not interact before the next layer.'],
    reads: ['H', 'Wo'], writes: ['O'] },
  { id: 'output', short: 'output', nav: 'Final attention output', title: 'The final attention output', group: 'output', tex: '\\operatorname{MultiHead}(X) = \\operatorname{Concat}(\\text{head}_1, \\dots, \\text{head}_h)\\, W_O \\in \\mathbb{R}^{n \\times d_{\\text{model}}}',
    purpose: 'One new vector per position, same shape as X, now containing information gathered from other positions.',
    beginner: 'Each token ends up with a new list of 4 numbers that blends in information from the tokens it looked at.',
    intermediate: 'In a Transformer block this output is added to the residual stream (X + MultiHead(X)) and followed by an MLP.',
    advanced: 'Under the causal mask, output row i depends only on inputs 1…i, which is what makes left-to-right generation possible.',
    misconceptions: ['The output is not a probability distribution over words: it is a hidden-state update.', 'Rows still correspond to the same positions as in X.'],
    reads: ['O'], writes: [] },
]
export const stageIndex = (id: StageId) => STAGES.findIndex((s) => s.id === id)
export const stageById = (id: StageId) => STAGES[stageIndex(id)]
