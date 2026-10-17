/** Lesson text for "Inside an LLM". Kept separate from the numerical engine and the views. */
import type { StepKind } from './journey'

export interface Dims { n: number; D: number; H: number; HD: number; M: number; V: number; layer: number; head: number }

interface StepLesson {
  equation: (d: Dims) => string
  shapes: (d: Dims) => [string, string][]
  beginner: (d: Dims) => string
  intermediate: (d: Dims) => string
  advanced: (d: Dims) => string
}

export const STEP_LESSONS: Record<StepKind, StepLesson> = {
  text: {
    equation: () => String.raw`\text{prompt} \;\rightarrow\; \texttt{"<bos>"} + \text{" "} + \text{text}`,
    shapes: () => [['input', 'string']],
    beginner: () => 'A language model never sees letters or words directly. Everything starts as a plain string that must be turned into numbers.',
    intermediate: () => 'The prompt is lower-cased and a leading space is added, because that is how every sentence looked in the training corpus. A special <bos> (beginning-of-sequence) token is placed in front.',
    advanced: () => 'Train/test consistency matters: the model was trained on sequences that start with <bos> followed by space-prefixed sentences. Feeding text in a different format is a distribution shift, and the predictions degrade measurably.',
  },
  bpe: {
    equation: () => String.raw`w = c_1 c_2 \dots c_m \;\xrightarrow{\text{apply merges by rank}}\; t_1 t_2 \dots t_k`,
    shapes: () => [['words', 'list[str]'], ['tokens', 'list[str]']],
    beginner: () => 'Byte-pair encoding splits text into pieces it learned were common. Frequent words become one token; rare words are built from smaller pieces.',
    intermediate: () => 'Each word starts as single characters (▁ marks a leading space). The tokenizer repeatedly merges the adjacent pair with the lowest merge rank, i.e. the pair that was learned earliest because it was most frequent in training.',
    advanced: () => 'Merges are a priority list learned greedily from pair counts. Encoding is deterministic and independent of the model weights. Words like "japan" were rare in the corpus, so they stay split into sub-word pieces that the model must recombine internally.',
  },
  ids: {
    equation: () => String.raw`t_i \mapsto \text{id}_i \in \{0,\dots,|V|-1\}`,
    shapes: (d) => [['token ids', `[${d.n}]`], ['vocabulary', `|V| = ${d.V}`]],
    beginner: () => 'Every token in the vocabulary has a fixed number, like a page in a dictionary. From here on the model only works with these numbers.',
    intermediate: (d) => `The vocabulary has ${d.V} entries: <bos>, the single characters, then one entry per learned merge. Each token is replaced by its row index.`,
    advanced: (d) => `IDs are categorical labels. Their numeric order means nothing; nearby IDs are not similar tokens. That is why the next step is a lookup into a learned table, never arithmetic on the ID. Context is ${d.n} tokens; the model's limit is fixed by its positional table.`,
  },
  embed: {
    equation: () => String.raw`E_{\text{tok}}[i] = W_E[\text{id}_i,\,:]`,
    shapes: (d) => [['W_E', `[${d.V}, ${d.D}]`], ['token embeddings', `[${d.n}, ${d.D}]`]],
    beginner: (d) => `Each token ID picks one row from a big learned table. That row is a list of ${d.D} numbers that describes the token. Similar tokens end up with similar rows.`,
    intermediate: (d) => `The lookup copies row id from W_E for each position, giving an ${d.n} × ${d.D} matrix. In the 3D view, all ${d.V} rows are projected with PCA so you can see which tokens the model placed near each other.`,
    advanced: () => 'An embedding lookup equals multiplying a one-hot vector by W_E, so gradients only reach the rows of tokens that occurred in the batch. PCA keeps only a few directions; distances in the 3D view are approximate.',
  },
  pos: {
    equation: () => String.raw`x^{(0)}_i = W_E[\text{id}_i] + W_P[i]`,
    shapes: (d) => [['W_P', `[ctx, ${d.D}]`], ['x⁽⁰⁾', `[${d.n}, ${d.D}]`]],
    beginner: () => 'Attention by itself does not know word order. Adding a position vector tells the model "this is the 1st, 2nd, 3rd token".',
    intermediate: () => 'A second learned table holds one vector per position. It is added element-wise to the token embedding. The sum is the start of the residual stream.',
    advanced: () => 'This model uses learned absolute positions (GPT-2 style). Many modern models use rotary embeddings (RoPE) applied inside attention instead. Because the table is finite, the model cannot process sequences longer than its context window.',
  },
  ln1: {
    equation: () => String.raw`\mathrm{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta`,
    shapes: (d) => [['x', `[${d.n}, ${d.D}]`], ['μ, σ', `[${d.n}]`], ['LN(x)', `[${d.n}, ${d.D}]`]],
    beginner: () => 'LayerNorm rescales each token\'s vector so its numbers have a consistent size. This keeps the computation stable.',
    intermediate: () => 'For each token separately: subtract the mean of its numbers, divide by their standard deviation, then apply a learned scale γ and shift β per dimension.',
    advanced: () => 'This is a pre-LN transformer: normalization happens before attention and before the MLP, while the residual stream itself is never normalized until the final LN. That keeps an identity path for gradients and is why deep pre-LN models train stably.',
  },
  qkv: {
    equation: () => String.raw`Q = \mathrm{LN}(x)W_Q,\quad K = \mathrm{LN}(x)W_K,\quad V = \mathrm{LN}(x)W_V`,
    shapes: (d) => [['W_Q, W_K, W_V', `[${d.D}, ${d.D}]`], ['Q, K, V', `[${d.n}, ${d.D}]`], ['per head', `[${d.n}, ${d.HD}] × ${d.H}`]],
    beginner: () => 'Every token produces three vectors. The query asks "what am I looking for?", the key says "what do I contain?", and the value is "what I will pass on if chosen".',
    intermediate: (d) => `Three learned matrices map each ${d.D}-dimensional vector to Q, K and V. The ${d.D} columns are then split into ${d.H} heads of ${d.HD} dimensions; the vertical guides in the heatmaps show the head boundaries.`,
    advanced: (d) => `Heads are a reshape, not separate layers: [n, ${d.D}] → [${d.H}, n, ${d.HD}]. Each head attends in its own ${d.HD}-dimensional subspace, which lets different heads track different relations at the same cost as one full-width head.`,
  },
  scores: {
    equation: () => String.raw`S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}, \qquad S_{ij} = -\infty \text{ if } j > i`,
    shapes: (d) => [['Q_h', `[${d.n}, ${d.HD}]`], ['K_hᵀ', `[${d.HD}, ${d.n}]`], ['S', `[${d.n}, ${d.n}]`]],
    beginner: () => 'Each token compares its query with the keys of itself and every earlier token. A high score means "this earlier token is relevant to me".',
    intermediate: (d) => `Score = dot product of query i and key j, divided by √${d.HD} ≈ ${Math.sqrt(d.HD).toFixed(3)}. Click any cell to see the dot product computed term by term. The grey triangle is the causal mask: a token may not look at the future.`,
    advanced: () => 'Dividing by √d_h keeps the score variance near 1 when q and k have unit-variance entries; without it, softmax saturates and gradients vanish. The mask is applied before softmax by setting future scores to −∞, so their weights become exactly 0.',
  },
  softmax: {
    equation: () => String.raw`A_{ij} = \frac{e^{S_{ij}}}{\sum_{j' \le i} e^{S_{ij'}}}, \qquad \textstyle\sum_j A_{ij} = 1`,
    shapes: (d) => [['S', `[${d.n}, ${d.n}]`], ['A', `[${d.n}, ${d.n}]`]],
    beginner: () => 'Scores are turned into percentages that add up to 100% for each token. The arcs show where each token "looks"; thicker means more attention.',
    intermediate: () => 'Softmax exponentiates each score and divides by the row total. Every row of the attention matrix therefore sums to exactly 1, and the masked entries are exactly 0. The inspector verifies both properties live.',
    advanced: () => 'Softmax is shift-invariant (we subtract the row max for numerical stability). In this model, heads often put 25–55% of their weight on <bos> on average: a common "no-op" pattern, where a head that has nothing useful to do parks its attention on a token whose value carries little information.',
  },
  mix: {
    equation: () => String.raw`o_i = \textstyle\sum_j A_{ij} v_j,\quad x' = x + [o^{(1)};\dots;o^{(H)}]\,W_O`,
    shapes: (d) => [['head outputs', `[${d.n}, ${d.HD}] × ${d.H}`], ['concat', `[${d.n}, ${d.D}]`], ['W_O', `[${d.D}, ${d.D}]`], ['x′', `[${d.n}, ${d.D}]`]],
    beginner: () => 'Each token collects information from the tokens it attended to, weighted by the percentages. The result is added to what the token already knew.',
    intermediate: () => 'Every head computes a weighted average of value vectors. The heads are concatenated, mixed by the output matrix W_O, and added to the residual stream.',
    advanced: () => 'The residual addition means attention writes a delta into a shared stream instead of replacing it. Information from earlier positions can only move to later positions here: attention is the only operation that mixes across tokens.',
  },
  mlp: {
    equation: () => String.raw`\mathrm{MLP}(x) = \mathrm{GELU}(\mathrm{LN}(x)W_1 + b_1)\,W_2 + b_2`,
    shapes: (d) => [['W_1', `[${d.D}, ${d.M}]`], ['hidden', `[${d.n}, ${d.M}]`], ['W_2', `[${d.M}, ${d.D}]`]],
    beginner: (d) => `Each token now "thinks" on its own. A small neural network with ${d.M} neurons transforms its vector. Bright cells are neurons that switched on.`,
    intermediate: () => 'The vector is normalized, expanded 4×, passed through the GELU non-linearity, and projected back down. The curve shows GELU with the selected token\'s pre-activations plotted on it.',
    advanced: () => 'The MLP acts on each position independently, with no token mixing. Interpretability work frames its rows as key-value memories: W_1 columns detect patterns, W_2 rows write the associated output direction into the residual stream.',
  },
  residual: {
    equation: () => String.raw`x^{(\ell+1)} = x^{(\ell)} + \mathrm{Attn}(\cdot) + \mathrm{MLP}(\cdot)`,
    shapes: (d) => [['residual stream', `[${d.n}, ${d.D}]`]],
    beginner: () => 'After each block the token vectors have been updated. If we read the last token\'s vector now, what would the model guess? The table shows exactly that.',
    intermediate: () => 'The "logit lens" applies the final LayerNorm and unembedding to the intermediate residual stream. It reveals how the prediction forms layer by layer.',
    advanced: () => 'The logit lens assumes intermediate representations live in the same basis as the output, which is only approximately true. The norm growth across layers is typical: blocks add to the stream rather than rescaling it.',
  },
  lnf: {
    equation: () => String.raw`h = \mathrm{LN}_f(x^{(L)})`,
    shapes: (d) => [['x⁽ᴸ⁾', `[${d.n}, ${d.D}]`], ['h', `[${d.n}, ${d.D}]`]],
    beginner: () => 'One last normalization before the model makes its guess.',
    intermediate: () => 'Pre-LN transformers never normalize the residual stream inside blocks, so a final LayerNorm is required before the unembedding.',
    advanced: () => 'Only the last row is needed to predict the next token during generation, but training computes all rows at once: each position predicts its own next token, which is why one sequence gives n training targets.',
  },
  logits: {
    equation: () => String.raw`z = h_n W_U + b, \qquad z \in \mathbb{R}^{|V|}`,
    shapes: (d) => [['h_n', `[${d.D}]`], ['W_U', `[${d.D}, ${d.V}]`], ['logits z', `[${d.V}]`]],
    beginner: (d) => `The last token's vector is compared with all ${d.V} vocabulary entries. Each comparison gives a score called a logit.`,
    intermediate: () => 'A logit is a dot product between the final hidden vector and one column of the unembedding matrix, plus a bias. Logits are unbounded and can be negative.',
    advanced: () => 'Logits are defined up to an additive constant (softmax is shift-invariant), so only differences between logits matter. The difference between two logits is the log-odds between those two tokens.',
  },
  probs: {
    equation: () => String.raw`p_k = \frac{e^{z_k}}{\sum_{k'} e^{z_{k'}}}, \qquad H(p) = -\textstyle\sum_k p_k \log_2 p_k`,
    shapes: (d) => [['p', `[${d.V}]`]],
    beginner: () => 'Softmax turns the scores into probabilities over every possible next token. They always add up to 100%.',
    intermediate: () => 'Entropy (in bits) measures uncertainty. 0 bits = completely sure. log₂(7) ≈ 2.81 bits = an even guess among 7 options.',
    advanced: () => 'This model is calibrated on its toy grammar: where the corpus is deterministic ("the dog likes to eat") it is near-certain, and where the corpus is uniformly random (which place someone went) its probability spreads almost evenly. That is maximum-likelihood training working as intended.',
  },
  decode: {
    equation: () => String.raw`\tilde p_k \propto \exp(z_k / T)\cdot\mathbb{1}[k \in \text{top-}k \cap \text{top-}p]`,
    shapes: (d) => [['p̃', `[${d.V}]`], ['next id', 'scalar']],
    beginner: () => 'The model only gives probabilities. A separate rule chooses the actual word: always pick the top one (greedy), or roll a weighted die (sampling).',
    intermediate: () => 'Temperature divides logits before softmax (T<1 sharpens, T>1 flattens). Top-k keeps the k most likely tokens; top-p keeps the smallest set whose probability reaches p. A seeded random number u picks a token from the cumulative distribution.',
    advanced: () => 'Generation is a loop: append the chosen id and run the full forward pass again. Real systems cache K and V of past tokens (the KV-cache) so each new step costs O(n) instead of recomputing all n positions.',
  },
}

export const OBJECTIVES = [
  'Trace a prompt from raw text to a next-token probability distribution, naming every tensor and its shape.',
  'Explain why tokenization, embeddings and positions are needed before any "thinking" happens.',
  'Compute an attention weight by hand from a query, a key and a softmax, and explain the causal mask.',
  'Distinguish what attention does (moves information between tokens) from what the MLP does (transforms each token).',
  'Explain how temperature, top-k and top-p change the generated text without changing the model.',
]

export interface Preset { prompt: string; title: string; why: string }
export const PRESETS: Preset[] = [
  { prompt: 'the dog likes to eat', title: 'A memorized association', why: 'In the training corpus every dog eats bones. In block 2, heads 1 and 4 put about 95% of the last token\'s attention on "dog", and the output entropy collapses to almost 0 bits.' },
  { prompt: 'the capital of japan is', title: 'A word the tokenizer has never merged', why: '"japan" becomes four tokens (▁, j, ap, an). In block 2, two heads put over 80% of the last token\'s attention on the fragment "ap". The answer itself, "tokyo", will need three tokens: ▁to + ky + o.' },
  { prompt: 'sara went to the', title: 'Genuine uncertainty', why: 'The corpus picks the place uniformly at random from 7 options, so the model spreads probability evenly. Here decoding settings change the output.' },
  { prompt: 'seven plus one is', title: 'Counting', why: 'A pattern the model learned for ten number words. Compare the logit lens after block 1 and block 2 to see when "eight" emerges.' },
]

export const MISCONCEPTIONS = [
  { myth: 'The model reads words.', truth: 'It reads token IDs. "japan" is four tokens here, and the model never sees the letters inside a token.' },
  { myth: 'Attention weights explain the model\'s reasoning.', truth: 'They show where information moved, not what was done with it. The MLP, the value vectors and W_O all shape what the attended information contributes.' },
  { myth: 'The model chooses a word.', truth: 'The network only outputs a probability distribution. The choice is made by a decoding rule outside the network, and the same model can produce different text under different rules.' },
]

export interface QuizQ { q: string; options: string[]; answer: number; explain: string }
export const QUIZ: QuizQ[] = [
  {
    q: 'With context length n = 6, what is the shape of one head\'s attention weight matrix in this model?',
    options: ['[6, 12]', '[6, 6]', '[12, 12]', '[6, 48]'],
    answer: 1,
    explain: 'Each of the 6 query positions gets one weight per key position: [n, n] = [6, 6]. The head dimension (12) disappears in the dot product.',
  },
  {
    q: 'Which operation is the only one that moves information between different token positions?',
    options: ['LayerNorm', 'The MLP', 'Attention', 'The unembedding'],
    answer: 2,
    explain: 'LayerNorm, the MLP and the unembedding act on each position independently. Only the weighted sum over value vectors mixes positions.',
  },
  {
    q: 'You lower the temperature from 1.0 to 0.3. What happens to the model\'s logits?',
    options: ['They are retrained', 'Nothing; they are divided by 0.3 before softmax', 'The top logit is set to 1', 'They are clipped to top-k'],
    answer: 1,
    explain: 'Temperature never changes the model. It rescales logits at decoding time, making the distribution sharper.',
  },
  {
    q: 'Why are scores divided by √d_h?',
    options: ['To make rows sum to one', 'To keep score variance near 1 so softmax does not saturate', 'To apply the causal mask', 'To normalize the value vectors'],
    answer: 1,
    explain: 'The dot product of two d_h-dimensional unit-variance vectors has variance d_h. Scaling by √d_h brings it back to ≈1.',
  },
]
