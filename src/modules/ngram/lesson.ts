/** Stages, equations and lesson text for module 08. Separate from the engine and the views. */
import type { Method } from '../../engine/ngram'

export type StageId = 'corpus' | 'count' | 'tables' | 'mle' | 'score' | 'zeros' | 'laplace' | 'interp' | 'backoff' | 'kn' | 'compare'

export interface Stage {
  id: StageId
  group: string
  short: string
  title: string
  /** method this stage demonstrates (guided mode); undefined = use the current setting */
  method?: Method
  equation: string
  beginner: string
  intermediate: string
  advanced: string
}

export const STAGES: Stage[] = [
  {
    id: 'corpus', group: 'Data', short: 'Corpus', title: 'Corpus → padded sentences',
    equation: String.raw`\langle s\rangle\;\langle s\rangle\; w_1\, w_2 \cdots w_m\; \langle/s\rangle`,
    beginner: 'An n-gram model learns only from the text you give it. Each sentence gets start markers <s> so the first word has a "previous word", and an end marker </s> so the model learns where sentences stop.',
    intermediate: 'Text is lower-cased and split into sentences at . ! ? ; and line breaks. Two <s> markers are added so trigrams have a full history at the first word. </s> is a real token the model must predict; <s> is never predicted.',
    advanced: 'The prediction vocabulary V is every training word plus </s> and <unk>. Any test word outside V is mapped to <unk>, a closed-vocabulary convention. Because <unk> is never observed in training, only smoothed models give it non-zero probability.',
  },
  {
    id: 'count', group: 'Data', short: 'Count', title: 'Sliding-window counting',
    equation: String.raw`c(w),\quad c(v\,w),\quad c(u\,v\,w)\quad\text{over predicted positions}`,
    beginner: 'A window slides over the text. At each position it reads the current word together with the one or two words before it, and adds one to each matching tally.',
    intermediate: 'Every predicted position contributes exactly one unigram, one bigram and one trigram. The three tables therefore have the same total, N = number of words + number of sentences (one </s> each).',
    advanced: 'Counting over predicted positions makes the context count c(v·) = Σ_w c(v w), which differs from the raw count of v: </s> is never a context, and <s> is a context once per sentence. Using c(v·) keeps every conditional distribution normalized.',
  },
  {
    id: 'tables', group: 'Data', short: 'Tables', title: 'Unigram, bigram and trigram tables',
    equation: String.raw`N=\textstyle\sum_w c(w)=\sum_{v,w}c(v\,w)=\sum_{u,v,w}c(u\,v\,w)`,
    beginner: 'These tables are the entire model. There are no weights to train: the model is just how often each word, pair and triple occurred.',
    intermediate: 'Rows of the bigram matrix are contexts v, columns are next words w. Click a cell to make it the context in the probability explorer. Most cells are zero even for a tiny corpus.',
    advanced: 'Storage grows with the number of observed types, while the space of possible n-grams grows as |V|ⁿ. Real systems (e.g. KenLM) store counts in tries or hash tables and prune rare n-grams.',
  },
  {
    id: 'mle', group: 'Estimate', short: 'MLE', title: 'Conditional probability by maximum likelihood', method: 'mle',
    equation: String.raw`P_{\text{ML}}(w\mid v)=\frac{c(v\,w)}{c(v\,\cdot)},\qquad P_{\text{ML}}(w\mid u,v)=\frac{c(u\,v\,w)}{c(u\,v\,\cdot)}`,
    beginner: 'The probability of a word after a context is the fraction of times that context was followed by that word.',
    intermediate: 'Pick a context: the bars show c(h, w) / c(h·) for every word that followed it. These fractions maximize the likelihood of the training corpus under the Markov assumption.',
    advanced: 'MLE is the solution of max Σ log P(w|h) subject to Σ_w P(w|h) = 1 (Lagrange multipliers give the relative frequency). It assigns zero to anything unseen and is undefined for unseen contexts.',
  },
  {
    id: 'score', group: 'Evaluate', short: 'Score', title: 'Sentence probability, cross-entropy, perplexity', method: 'mle',
    equation: String.raw`H=-\frac{1}{T}\sum_{t=1}^{T}\log_2 P(w_t\mid h_t),\qquad \mathrm{PP}=2^{H}=P(w_1\ldots w_T)^{-1/T}`,
    beginner: 'A sentence\'s probability is the product of each word\'s probability given the words before it. Perplexity turns that into "how many equally likely choices the model is confused between", on average. Lower is better.',
    intermediate: 'T counts every predicted token, including each </s>. We sum log₂ probabilities instead of multiplying to avoid underflow. H is in bits per token.',
    advanced: 'Cross-entropy measured on held-out text upper-bounds the true entropy rate of the language (for a stationary ergodic source). Perplexities are only comparable with the same vocabulary, the same <unk> handling and the same test tokenization.',
  },
  {
    id: 'zeros', group: 'Problem', short: 'Zeros', title: 'Unseen n-grams and sparsity', method: 'mle',
    equation: String.raw`c(h,w)=0\;\Rightarrow\;P_{\text{ML}}(w\mid h)=0\;\Rightarrow\;\mathrm{PP}=\infty`,
    beginner: 'If the test sentence contains even one pair of words the model never saw, maximum likelihood says the sentence is impossible.',
    intermediate: 'The grid shows every possible bigram; lit cells were observed. The fraction lit is tiny, and it shrinks further for trigrams. Red rows in the table are test n-grams with zero count.',
    advanced: 'By Zipf\'s law most types are rare. The Good-Turing estimate of the probability mass of unseen events is N₁/N, the share of n-gram tokens whose type was seen exactly once. Smoothing methods are different ways of reserving that mass.',
  },
  {
    id: 'laplace', group: 'Fix', short: 'Add-k', title: 'Add-k (Laplace) smoothing', method: 'laplace',
    equation: String.raw`P_{\text{add-}k}(w\mid h)=\frac{c(h,w)+k}{c(h\,\cdot)+k\,|V|},\qquad c^{*}=\frac{(c+k)\,c(h\cdot)}{c(h\cdot)+k|V|}`,
    beginner: 'Pretend every possible word was seen k extra times after every context. Nothing is impossible any more.',
    intermediate: 'The adjusted count c* shows how much each seen n-gram was "taxed" to pay for unseen ones. With k = 1 and a large vocabulary the tax is heavy: watch the seen mass bar shrink.',
    advanced: 'Add-k is MAP estimation with a symmetric Dirichlet(k+1) prior. It is a poor language model because it spreads mass uniformly, ignoring that unseen words differ in plausibility. Tune k on held-out data; the curve below shows why k = 1 is rarely optimal.',
  },
  {
    id: 'interp', group: 'Fix', short: 'Interp', title: 'Linear interpolation', method: 'interp',
    equation: String.raw`P(w\mid u,v)=\lambda_3 P_{\text{ML}}(w\mid u,v)+\lambda_2 P_{\text{ML}}(w\mid v)+\lambda_1 P_{\text{ML}}(w)+\lambda_0\tfrac{1}{|V|}`,
    beginner: 'Mix the opinions of the trigram, bigram and unigram models. If the trigram has never seen the context, the shorter models still have a vote.',
    intermediate: 'Weights are normalized to sum to 1. The small uniform weight λ₀ guarantees <unk> a non-zero probability. An undefined component (unseen context) is replaced by the next lower one, so every mixture is a proper distribution. "Tune on held-out" grid-searches the λs to minimize held-out perplexity.',
    advanced: 'Proper tuning uses held-out data, never the training set (on the training data λ₃ = 1 always wins). Jelinek-Mercer smoothing makes λ depend on the context count; EM can fit the λs directly.',
  },
  {
    id: 'backoff', group: 'Fix', short: 'Backoff', title: 'Backoff with absolute discounting', method: 'backoff',
    equation: String.raw`P_{\text{bo}}(w\mid h)=\begin{cases}\dfrac{c(h,w)-d}{c(h\,\cdot)} & c(h,w)>0\\[6pt] \alpha(h)\,P_{\text{bo}}(w\mid h') & \text{otherwise}\end{cases}\quad \alpha(h)=\frac{d\,N_{1+}(h\,\cdot)/c(h\cdot)}{1-\sum_{w:c(h,w)>0}P_{\text{bo}}(w\mid h')}`,
    beginner: 'Trust the longest history you have evidence for. If the trigram was seen, use it. If not, fall back to the bigram, then to the unigram.',
    intermediate: 'Subtracting d from every seen count frees probability mass d·N₁₊(h·)/c(h·). α(h) hands exactly that mass to the unseen words, in proportion to the lower-order model. The chips show which order answered for each test token.',
    advanced: 'Without the denominator in α, backoff would not normalize ("stupid backoff", used in web-scale systems where normalization is skipped). Katz backoff uses Good-Turing discounts instead of a fixed d. The unigram level here mixes in a uniform floor so <unk> is covered.',
  },
  {
    id: 'kn', group: 'Fix', short: 'KN', title: 'Interpolated Kneser-Ney', method: 'kn',
    equation: String.raw`P_{\text{KN}}(w\mid v)=\frac{\max(c(v\,w)-D,0)}{c(v\,\cdot)}+\frac{D\,N_{1+}(v\,\cdot)}{c(v\,\cdot)}\,P_{\text{cont}}(w),\qquad P_{\text{cont}}(w)=\frac{N_{1+}(\bullet\,w)}{N_{1+}(\bullet\,\bullet)}`,
    beginner: 'When falling back to shorter histories, ask "how many different words does this word follow?" instead of "how often does it occur?". "francisco" is frequent, but almost only after "san", so it is a bad guess in a new context.',
    intermediate: 'The continuation count N₁₊(• w) counts distinct left neighbours. Lower orders use continuation counts; the highest order uses raw counts. λ(v) = D·N₁₊(v·)/c(v·) is exactly the mass removed by discounting, so the result sums to 1.',
    advanced: 'For trigrams the middle order uses N₁₊(• v w) = |{u : c(u v w) > 0}| over N₁₊(• v •). The lowest order is mixed with a uniform floor for <unk>. Modified KN (Chen & Goodman) uses three discounts D₁, D₂, D₃₊ estimated from count-of-counts. KN is the strongest classic n-gram smoother.',
  },
  {
    id: 'compare', group: 'Compare', short: 'Compare', title: 'Comparing all methods',
    equation: String.raw`\mathrm{PP}_{\text{held-out}}\;\text{for}\;\{\text{ML, add-}k,\text{ interp, backoff, KN}\}\times\{1,2,3\}`,
    beginner: 'Every method scores the same test and held-out text. The lowest perplexity wins.',
    intermediate: 'Held-out perplexity is the honest score; test perplexity becomes optimistic once you start tuning against it. ∞ marks a model that assigned probability 0 somewhere.',
    advanced: 'Higher order is not always better on small data: more contexts are unseen, and the estimates get noisier. That is the bias-variance trade-off that smoothing manages, and the reason neural models (module 01) won: they share statistics across similar words.',
  },
]

export const OBJECTIVES = [
  'Build unigram, bigram and trigram count tables from a corpus with sentence boundary markers.',
  'Compute conditional probabilities by maximum likelihood and explain why unseen n-grams break them.',
  'Score a sentence: probability, log-probability, cross-entropy and perplexity, including </s>.',
  'Apply add-k smoothing, linear interpolation and normalized backoff, and verify each sums to one.',
  'Explain Kneser-Ney continuation counts and when they beat raw frequency.',
  'Compare models honestly using held-out perplexity.',
]

export const MISCONCEPTIONS = [
  { myth: 'Add-one smoothing is a good default.', truth: 'With a realistic vocabulary it takes most of the probability mass away from what was actually observed. Compare its perplexity with every other method in the Compare stage.' },
  { myth: 'A trigram model is always better than a bigram model.', truth: 'Only with enough data. On small corpora most trigram contexts are unseen, so without good smoothing the trigram model can be worse.' },
  { myth: 'Perplexity can be compared across any two models.', truth: 'Only with the same vocabulary, <unk> policy and tokenization. A model with a tiny vocabulary can report a deceptively low perplexity.' },
  { myth: 'Kneser-Ney is just backoff with a different discount.', truth: 'Its key change is what the lower order counts: distinct contexts (continuation counts), not occurrences.' },
]

export interface QuizQ { q: string; options: string[]; answer: number; explain: string }
export const QUIZ: QuizQ[] = [
  { q: 'In the textbook corpus “I am Lina. Lina I am. I do not like cold tea and figs.”, what is P_ML(</s> | lina)?', options: ['1/3', '1/2', '2/3', '0'], answer: 1,
    explain: '“lina” appears twice; once it is followed by </s> (end of “I am Lina”), once by “i”. So c(lina </s>) / c(lina ·) = 1/2.' },
  { q: 'A bigram MLE model gives one test token probability 0. What is the test perplexity?', options: ['Undefined but finite', 'Infinity', 'Zero', 'Equal to |V|'], answer: 1,
    explain: 'log₂ 0 = −∞, so the cross-entropy is +∞ and PP = 2^∞ = ∞. One zero ruins the whole test set.' },
  { q: 'What does the Kneser-Ney continuation count N₁₊(• w) measure?', options: ['How often w occurs', 'How many distinct words precede w', 'How many words follow w', 'The number of sentences containing w'], answer: 1,
    explain: 'It is |{v : c(v w) > 0}|: the number of different contexts w has appeared in, a proxy for how likely w is to show up after a new context.' },
  { q: 'Interpolation weights should be tuned on…', options: ['the training corpus', 'a held-out set separate from the test set', 'the test set', 'any of these, the result is the same'], answer: 1,
    explain: 'On training data the highest order always wins (λ₃ → 1). Tuning on the test set makes the reported test score optimistic.' },
  { q: 'Under a uniform model over |V| = 12 tokens, what is the perplexity?', options: ['1', '3.58', '12', '144'], answer: 2,
    explain: 'H = log₂ 12 bits for every token, so PP = 2^H = 12. Perplexity is the size of an equivalent uniform choice.' },
]
