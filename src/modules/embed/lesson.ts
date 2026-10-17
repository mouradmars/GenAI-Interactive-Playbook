/** Stages, equations and lesson text for module 06. Kept apart from the engine and the views. */
import type { ProjMethod } from '../../engine/embeddings'

export type SpaceId = 'toy' | 'ppmi' | 'nano'
export type StageId = 'lookup' | 'dot' | 'cosine' | 'matrix' | 'neighbours' | 'arithmetic' | 'pca' | 'umap' | 'distortion' | 'context' | 'compare'

export interface Stage {
  id: StageId
  group: string
  short: string
  title: string
  /** space and projection this stage opens with in guided mode (the student can change them) */
  space?: SpaceId
  method?: ProjMethod
  equation: string
  beginner: string
  intermediate: string
  advanced: string
}

export const STAGES: Stage[] = [
  {
    id: 'lookup', group: 'Represent', short: 'Lookup', title: 'Tokens become vectors', space: 'toy',
    equation: String.raw`\mathbf{e}_w = \mathbf{1}_w^{\top} E,\qquad E\in\mathbb{R}^{|V|\times d}`,
    beginner: 'Every word gets a list of numbers, its embedding. Here the numbers have names (royal, male, female…), so you can read them. In learned embeddings nobody chooses what the dimensions mean.',
    intermediate: 'An embedding table E has one row per word. Looking up a word is the same as multiplying a one-hot vector by E: the 1 picks out exactly one row.',
    advanced: 'Because a lookup is a matrix product, gradients flow into E like any other weight. Only rows of words that occurred in a batch receive gradient. Learned dimensions are an arbitrary basis: any rotation of E gives an equally good model.',
  },
  {
    id: 'dot', group: 'Measure', short: 'Dot', title: 'Dot product and length',
    equation: String.raw`\mathbf{a}\cdot\mathbf{b}=\sum_i a_i b_i=\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert\cos\theta,\qquad \lVert\mathbf{a}\rVert=\sqrt{\mathbf{a}\cdot\mathbf{a}}`,
    beginner: 'Drag the arrow tips. The dot product is large when the arrows point the same way and are long, zero when they are at right angles, and negative when they point apart.',
    intermediate: 'The dot product sums the products of matching components. Geometrically it is the length of a times the length of b’s shadow on a (the dashed projection).',
    advanced: 'Attention scores (module 01) and output logits are dot products, so vector length matters there: a longer key gets a larger score. That is why transformers normalize and scale by 1/√d.',
  },
  {
    id: 'cosine', group: 'Measure', short: 'Cosine', title: 'Cosine similarity',
    equation: String.raw`\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}\in[-1,1]`,
    beginner: 'Cosine only cares about direction. Stretch an arrow with the scale slider: the dot product changes, the cosine does not.',
    intermediate: 'Dividing by both lengths removes magnitude. 1 means same direction, 0 means perpendicular, −1 means opposite. For word vectors, frequent words often have longer vectors, so cosine is the usual similarity.',
    advanced: 'On unit vectors, ‖a − b‖² = 2 − 2 cos(a, b), so nearest neighbours by cosine and by Euclidean distance agree after normalization. Cosine is not a metric (no triangle inequality), but 1 − cos is commonly called cosine distance.',
  },
  {
    id: 'matrix', group: 'Measure', short: 'Matrix', title: 'Similarity matrix',
    equation: String.raw`S_{ij}=\cos(\mathbf{e}_i,\mathbf{e}_j),\qquad S=S^{\top},\; S_{ii}=1`,
    beginner: 'Every pair of selected words gets a similarity score. Bright squares are similar pairs. Words from the same category form bright blocks.',
    intermediate: 'Add or remove words with the chips. Click a cell to see its cosine computed step by step in the inspector. Switch to the dot product to see how vector length changes the picture.',
    advanced: 'Row-normalize E to unit length and S = ÊÊᵀ: the whole matrix is one matrix product. Retrieval systems (module 11) compute exactly this between a query and millions of documents.',
  },
  {
    id: 'neighbours', group: 'Measure', short: 'Neighbours', title: 'Nearest neighbours in full dimension', space: 'ppmi',
    equation: String.raw`\mathrm{NN}_k(w)=\operatorname*{arg\,top\text{-}k}_{v\neq w}\;\cos(\mathbf{e}_w,\mathbf{e}_v)`,
    beginner: 'The nearest neighbours are the words whose vectors point most nearly the same way. They are computed in the full space, not from the picture.',
    intermediate: 'Lines in the scatter connect the chosen word to its true neighbours. A dashed line means that neighbour is not among the closest points in the 2D picture: the picture is misleading there.',
    advanced: 'Words that share contexts get similar vectors: “cat” lands nearest owl and eagle (they share night, small and mice), not dog. That is the distributional hypothesis, “you shall know a word by the company it keeps” (Firth, 1957). Similarity here means “used alike”, which is not the same as “means the same”: antonyms often share contexts too.',
  },
  {
    id: 'arithmetic', group: 'Relate', short: 'Analogy', title: 'Vector arithmetic and analogies', space: 'toy', method: 'pca',
    equation: String.raw`\mathbf{x}^{*}=\operatorname*{arg\,max}_{x\notin\{a,b,c\}}\cos\!\big(\mathbf{e}_x,\;\mathbf{e}_b-\mathbf{e}_a+\mathbf{e}_c\big)`,
    beginner: '“man is to king as woman is to ?” Take king, remove man, add woman, and look for the closest word. With the hand-built vectors you land exactly on queen.',
    intermediate: 'The answer excludes the three input words; otherwise the result is often just b or c. The arrows show the two offsets b − a and x − c, which should be parallel if the relation is consistent.',
    advanced: 'This is 3CosAdd (Mikolov et al., 2013). Learned embeddings solve far fewer analogies than this demo suggests, and results depend on excluding inputs. Try the corpus-trained space: paris : france :: rome : ? ranks italy first with a ±4 window and 8th with ±1, and man : king :: woman : ? puts princess ahead of queen.',
  },
  {
    id: 'pca', group: 'Project', short: 'PCA', title: 'PCA: the best flat shadow', space: 'ppmi', method: 'pca',
    equation: String.raw`\Sigma=\tfrac1n\textstyle\sum_i(\mathbf{e}_i-\bar{\mathbf{e}})(\mathbf{e}_i-\bar{\mathbf{e}})^{\top}=Q\Lambda Q^{\top},\qquad \mathbf{y}_i=Q_{:,1:2}^{\top}(\mathbf{e}_i-\bar{\mathbf{e}})`,
    beginner: 'We cannot draw 24 dimensions, so we cast a shadow onto a flat screen. PCA picks the screen angle that keeps the shadow as spread out as possible.',
    intermediate: 'The bars show how much of the total variance each principal component keeps. Whatever the first two bars do not cover is invisible in the 2D picture. Switch to 3D to see one more component.',
    advanced: 'PCA is linear and keeps large distances well, so global layout is trustworthy-ish, but points that differ only along discarded components collapse on top of each other. It is computed here exactly by Jacobi eigen-decomposition of the covariance matrix.',
  },
  {
    id: 'umap', group: 'Project', short: 'UMAP', title: 'UMAP: keep the neighbourhoods', space: 'ppmi', method: 'umap',
    equation: String.raw`\min_{Y}\;\sum_{ij}\Big[p_{ij}\log\frac{p_{ij}}{q_{ij}}+(1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\Big],\quad q_{ij}=\big(1+a\lVert\mathbf{y}_i-\mathbf{y}_j\rVert^{2b}\big)^{-1}`,
    beginner: 'UMAP does not cast a shadow. It builds a map where each word’s closest friends stay close, even if that means bending or tearing the space.',
    intermediate: 'n_neighbors sets how many friends each word tries to keep (small = local detail, large = more global shape). min_dist sets how tightly points may pack. The random seed is fixed, so the same settings always give the same picture.',
    advanced: 'UMAP optimizes a fuzzy cross-entropy between a k-NN graph in the original space (here built with cosine distance) and the 2D layout, by stochastic gradient descent. Distances between clusters, and cluster sizes, are not meaningful.',
  },
  {
    id: 'distortion', group: 'Project', short: 'Distortion', title: 'What every projection distorts', space: 'ppmi',
    equation: String.raw`\rho_s=\mathrm{Spearman}\big(1-\cos(\mathbf{e}_i,\mathbf{e}_j),\;\lVert\mathbf{y}_i-\mathbf{y}_j\rVert\big),\qquad \mathrm{NP}_k=\frac{|\mathrm{NN}_k^{\text{full}}\cap\mathrm{NN}_k^{\text{2D}}|}{k}`,
    beginner: 'A 2D picture of a high-dimensional space always lies somewhere. These charts measure where: pairs that look close but are not, and pairs that look far apart but are close.',
    intermediate: 'Each dot in the Shepard plot is one pair of words: its true distance (x) against its distance on screen (y). A perfect projection would be a rising line. Neighbourhood preservation counts how many of each word’s 5 true neighbours are also its 5 nearest on screen.',
    advanced: 'PCA usually wins on ρ (global distances) and UMAP on neighbourhood preservation (local structure). No 2D map can keep both in general: n points can need up to n − 1 dimensions to be placed with exact distances.',
  },
  {
    id: 'context', group: 'Context', short: 'Context', title: 'Contextual embeddings across sentences', space: 'nano',
    equation: String.raw`\mathbf{h}^{(\ell)}_t=f_\ell\big(\mathbf{h}^{(\ell-1)}_{1..t}\big),\qquad \mathbf{h}^{(0)}_t=E[w_t]+P[t]`,
    beginner: 'A static embedding gives “is” the same vector everywhere. Inside a transformer, the vector for “is” changes with the sentence around it. Watch the similarity between occurrences fall layer by layer.',
    intermediate: 'Each heatmap compares the same token in different sentences at one depth. Before attention, only position can tell occurrences apart (same position → cosine exactly 1). After the blocks, each occurrence carries what it needs to predict its own next word.',
    advanced: 'These are real activations of Observatory-Nano. In large models this context dependence is what lets one word form represent several senses (bank, river bank). Here the toy grammar has no ambiguity, so the divergence reflects different predictions rather than different meanings.',
  },
  {
    id: 'compare', group: 'Compare', short: 'Compare', title: 'Compare spaces and projections',
    equation: String.raw`\text{same words}\;\times\;\{\text{toy},\ \text{PPMI-SVD},\ \text{transformer}\}\;\times\;\{\text{PCA},\ \text{UMAP}\}`,
    beginner: 'Put two pictures side by side. Different training methods and different projections tell different stories about the same words.',
    intermediate: 'Compare a word’s neighbours and the faithfulness scores. A beautiful cluster plot with low neighbourhood preservation is a beautiful lie.',
    advanced: 'Count-based vectors (PPMI-SVD) and prediction-based vectors (word2vec, transformers) are closely related: skip-gram with negative sampling implicitly factorizes a shifted PMI matrix (Levy & Goldberg, 2014).',
  },
]

export const OBJECTIVES = [
  'Explain an embedding table and why a lookup is a matrix product.',
  'Compute dot products, vector lengths and cosine similarity by hand, and explain when they disagree.',
  'Find nearest neighbours in the full space and solve an analogy with 3CosAdd, excluding the inputs.',
  'Describe what PCA and UMAP preserve, and measure the distortion of a 2D projection.',
  'Train a count-based embedding (PPMI + SVD) and explain the effect of the context window.',
  'Contrast static and contextual embeddings using a real transformer’s residual stream.',
]

export const MISCONCEPTIONS = [
  { myth: 'If two words are close in the 2D plot, their embeddings are similar.', truth: 'Only partly. A projection keeps a fraction of the variance (see the PCA bars). The “false friends” list shows pairs that look close on screen but are far apart in the full space.' },
  { myth: 'Each dimension of a learned embedding means something, like “royalty”.', truth: 'Only in hand-built toys. Learned bases are arbitrary: rotating all vectors leaves every cosine unchanged, so no single axis is privileged.' },
  { myth: 'king − man + woman = queen is what embeddings reliably do.', truth: 'It is exact here only because the vectors were built for it. In learned spaces the target is rarely exact, and the inputs must be excluded or the answer is usually king itself.' },
  { myth: 'A word has one embedding in a transformer.', truth: 'Only at the input layer. After attention, the same token has a different vector in every sentence (the Context stage).' },
]

export interface QuizQ { q: string; options: string[]; answer: number; explain: string }
export const QUIZ: QuizQ[] = [
  { q: 'a = [3, 4] and b = [6, 8]. What is cos(a, b)?', options: ['0.5', '1', '50', '0.6'], answer: 1,
    explain: 'b = 2a, so they point in exactly the same direction: cos = 50 / (5 × 10) = 1. The dot product (50) depends on length; the cosine does not.' },
  { q: 'Why do analogy solvers exclude the three input words?', options: ['To save computation', 'Because b − a + c is usually still closest to b or c', 'Because inputs have zero vectors', 'They do not need to'], answer: 1,
    explain: 'The offset b − a is small compared with the vectors themselves, so the target stays near b (or c). Without exclusion, “king” is often the top answer to king − man + woman.' },
  { q: 'PCA to 2D keeps 20% of the variance. What does that tell you?', options: ['The plot is 20% accurate', '80% of the spread lies in directions the plot cannot show', 'Only 20% of words are shown', 'UMAP would keep 80%'], answer: 1,
    explain: 'Explained variance is the share of total squared spread along the two plotted axes. Pairs that differ mostly along the other directions can land on top of each other.' },
  { q: 'Before any attention layer, when do two occurrences of the same token in Observatory-Nano have cosine exactly 1?', options: ['Never', 'When they are at the same position', 'When the sentences are identical', 'Always'], answer: 1,
    explain: 'x⁽⁰⁾ = E[w] + P[t]. Same token and same position give the identical vector; only position can separate them before attention mixes in context.' },
  { q: 'In the sentence “paris is the capital of france”, why does a ±2 window give paris no information about france?', options: ['The window is too large', 'paris and france are 5 words apart, so they never fall in the same window', 'SVD discards country names', 'PPMI ignores rare words'], answer: 1,
    explain: 'Co-occurrence counts only see words within the window. With ±2, paris sees only “is the”, which every capital shares. Shorter templates like “we visited paris in france” are what let the lab’s ±4 model learn the pairing.' },
]
