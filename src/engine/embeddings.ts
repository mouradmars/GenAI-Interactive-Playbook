/**
 * Embedding spaces for module 06, and every computation the lab displays.
 *
 * Three sources, labelled differently in the UI:
 *  - 'simulation'  hand-built, interpretable vectors (exact arithmetic, NOT learned)
 *  - 'count'       PPMI co-occurrence + truncated SVD, trained deterministically in the browser
 *  - 'transformer' the token embedding table and residual stream of Observatory-Nano (module 01)
 */
import { UMAP } from 'umap-js'
import { rng } from './decoding'
import { add, cosine, dot, eigh, euclid, neighbourPreservation, norm, pca, scale, spearman, sub, unit, type Vec } from './linalg'
import type { NanoGPT } from './model'

export type SourceKind = 'simulation' | 'count' | 'transformer'
export interface Space {
  id: string
  title: string
  kind: SourceKind
  provenance: string
  words: string[]
  vecs: Vec[]
  dim: number
  /** category per word, for colouring */
  groups: string[]
  /** names of the dimensions when they are interpretable (simulation only) */
  featureNames?: string[]
  /** training statistics shown in the inspector */
  stats?: Record<string, string | number>
}

export const KIND_LABEL: Record<SourceKind, string> = {
  simulation: 'educational simulation · hand-built vectors',
  count: 'trained model · PPMI + SVD on a corpus',
  transformer: 'trained model · Observatory-Nano transformer',
}

/* ------------------------------------------------------------ simulation */

export const TOY_FEATURES = ['royal', 'male', 'female', 'adult', 'human', 'animal', 'feline', 'canine', 'fruit', 'sweet', 'sour']
const T = (s: string) => s.split('').map(Number)
const TOY: [string, string, number[]][] = [
  ['king', 'people', T('11011000000')], ['queen', 'people', T('10111000000')],
  ['prince', 'people', T('11001000000')], ['princess', 'people', T('10101000000')],
  ['man', 'people', T('01011000000')], ['woman', 'people', T('00111000000')],
  ['boy', 'people', T('01001000000')], ['girl', 'people', T('00101000000')],
  ['cat', 'animals', T('00010110000')], ['kitten', 'animals', T('00000110000')],
  ['dog', 'animals', T('00010101000')], ['puppy', 'animals', T('00000101000')],
  ['apple', 'fruit', T('00000000110')], ['banana', 'fruit', T('00000000110')], ['lemon', 'fruit', T('00000000101')],
]
export function toySpace(): Space {
  // banana is sweeter than apple, so the two fruits are not identical
  const vecs = TOY.map(([w, , v]) => (w === 'banana' ? v.map((x, i) => (i === 9 ? 1.5 : x)) : v.slice()))
  return {
    id: 'toy', title: 'Royal family & friends', kind: 'simulation',
    provenance: 'Hand-built 11-dimensional vectors with named features (royal, male, female, …). Nothing here was learned from data: the arithmetic works exactly because the features were designed to.',
    words: TOY.map((t) => t[0]), vecs, dim: TOY_FEATURES.length, groups: TOY.map((t) => t[1]), featureNames: TOY_FEATURES,
  }
}

/* ------------------------------------------------------------ PPMI + SVD */

/** Word categories of the embedding corpus (mirrors training/embed_corpus.py). */
const CAT: Record<string, string[]> = {
  animals: ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'chicken', 'wolf', 'bear', 'fox', 'owl', 'whale', 'shark', 'dolphin', 'eagle', 'monkey', 'tiger', 'parrot'],
  habitats: ['house', 'farm', 'forest', 'sea', 'mountains', 'jungle'],
  'animal food': ['fish', 'mice', 'bones', 'meat', 'grass', 'hay', 'apples', 'corn', 'seeds', 'honey', 'bananas', 'fruit'],
  sounds: ['meows', 'barks', 'moos', 'neighs', 'bleats', 'oinks', 'clucks', 'howls', 'growls', 'hoots', 'sings', 'silent', 'clicks', 'screams', 'chatters', 'roars', 'talks'],
  countries: ['france', 'italy', 'spain', 'germany', 'greece', 'japan', 'china', 'india', 'egypt', 'morocco', 'kenya', 'brazil', 'peru', 'chile', 'canada'],
  capitals: ['paris', 'rome', 'madrid', 'berlin', 'athens', 'tokyo', 'beijing', 'delhi', 'cairo', 'rabat', 'nairobi', 'brasilia', 'lima', 'santiago', 'ottawa'],
  languages: ['french', 'italian', 'spanish', 'german', 'greek', 'japanese', 'chinese', 'hindi', 'arabic', 'swahili', 'portuguese', 'english'],
  continents: ['europe', 'asia', 'africa', 'america'],
  fruit: ['apple', 'banana', 'lemon', 'orange', 'grape', 'lime', 'cherry', 'plum'],
  colours: ['red', 'yellow', 'purple', 'green', 'blue', 'white', 'black'],
  people: ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl'],
  jobs: ['doctor', 'nurse', 'teacher', 'chef', 'farmer', 'pilot'],
  workplaces: ['hospital', 'school', 'kitchen', 'airport'],
  numbers: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
  adjectives: ['small', 'medium', 'big', 'huge', 'sweet', 'sour', 'adult', 'young'],
}
const CAT_OF = new Map(Object.entries(CAT).flatMap(([g, ws]) => ws.map((w) => [w, g] as const)))
export const categoryOf = (w: string) => CAT_OF.get(w) ?? 'function'

export interface PPMIConfig { window: number; dim: number; minCount: number; alpha: number }
export const defaultPPMI: PPMIConfig = { window: 4, dim: 24, minCount: 3, alpha: 0.75 }

export const tokenizeWords = (text: string) =>
  text.toLowerCase().split(/[.!?\n]+/).map((s) => s.match(/[a-z']+/g) ?? []).filter((s) => s.length)

/** Symmetric, sentence-bounded co-occurrence counts within ±window. */
export function cooccurrence(sents: string[][], window: number, minCount: number) {
  const freq = new Map<string, number>()
  for (const s of sents) for (const w of s) freq.set(w, (freq.get(w) ?? 0) + 1)
  const words = [...freq.keys()].filter((w) => freq.get(w)! >= minCount).sort()
  const idx = new Map(words.map((w, i) => [w, i]))
  const n = words.length
  const C = Array.from({ length: n }, () => new Array(n).fill(0))
  for (const s of sents)
    for (let i = 0; i < s.length; i++) {
      const a = idx.get(s[i])
      if (a === undefined) continue
      for (let j = Math.max(0, i - window); j <= Math.min(s.length - 1, i + window); j++) {
        if (j === i) continue
        const b = idx.get(s[j])
        if (b !== undefined) C[a][b]++
      }
    }
  return { words, C, freq }
}

/**
 * PPMI(w,c) = max(0, log2 P(w,c) / (P(w) Pα(c))) with context-distribution smoothing α
 * (Levy, Goldberg & Dagan 2015). Pα(c) = #(c)^α / Σ #(c')^α.
 */
export function ppmi(C: number[][], alpha: number) {
  const n = C.length
  const row = C.map((r) => r.reduce((a, b) => a + b, 0))
  const total = row.reduce((a, b) => a + b, 0)
  const col = Array.from({ length: n }, (_, j) => C.reduce((s, r) => s + r[j], 0))
  const colA = col.map((c) => c ** alpha)
  const zA = colA.reduce((a, b) => a + b, 0)
  return C.map((r, i) => r.map((c, j) => {
    if (!c || !row[i] || !colA[j]) return 0
    const pmi = Math.log2((c / total) / ((row[i] / total) * (colA[j] / zA)))
    return Math.max(0, pmi)
  }))
}

/**
 * Truncated SVD of M (n × n) via the eigen-decomposition of M Mᵀ = U Σ² Uᵀ.
 * Word vectors W = U_k Σ_k^p, the symmetric weighting p = 0.5 recommended by Levy et al.
 */
export function svdEmbed(M: number[][], k: number, p = 0.5) {
  const n = M.length
  const MMt = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (j < i ? 0 : dot(M[i], M[j]))))
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) MMt[i][j] = MMt[j][i]
  const { values, vectors } = eigh(MMt)
  const kk = Math.min(k, n)
  const sigma = values.slice(0, kk).map((x) => Math.sqrt(Math.max(0, x)))
  const W = vectors.map((r) => r.slice(0, kk).map((u, c) => u * sigma[c] ** p))
  const energy = values.reduce((a, b) => a + Math.max(0, b), 0)
  return { W, sigma, energyKept: sigma.reduce((a, s) => a + s * s, 0) / (energy || 1) }
}

export function trainPPMISpace(text: string, cfg: PPMIConfig, title = 'Embedding corpus'): Space {
  const sents = tokenizeWords(text)
  const { words, C } = cooccurrence(sents, cfg.window, cfg.minCount)
  if (words.length < 3) throw new Error('The corpus needs at least three words that occur minCount times.')
  const M = ppmi(C, cfg.alpha)
  const { W, sigma, energyKept } = svdEmbed(M, cfg.dim)
  const tokens = sents.reduce((a, s) => a + s.length, 0)
  const nnz = M.reduce((a, r) => a + r.filter((x) => x > 0).length, 0)
  return {
    id: 'ppmi', title, kind: 'count',
    provenance: `Trained in your browser: ±${cfg.window}-word co-occurrence counts over ${sents.length.toLocaleString()} sentences → PPMI (α = ${cfg.alpha}) → truncated SVD to ${W[0].length} dimensions. Deterministic: the same corpus and settings always give the same vectors.`,
    words, vecs: W, dim: W[0].length, groups: words.map(categoryOf),
    stats: { sentences: sents.length, tokens, vocabulary: words.length, 'non-zero PPMI cells': `${nnz} / ${words.length ** 2}`, 'top singular value': sigma[0].toFixed(3), 'SVD energy kept': `${(energyKept * 100).toFixed(1)}%` },
  }
}

/* ------------------------------------------------------------ transformer */

const showTok = (t: string) => t.replace(/▁/g, '·')
export function nanoSpace(model: NanoGPT, wholeWordsOnly = true): Space {
  const { vocab } = model
  const D = model.meta.dModel
  const keep = vocab.map((_, i) => i).filter((i) => {
    const t = vocab[i]
    if (t === '<bos>') return false
    return wholeWordsOnly ? t.startsWith('▁') && t.length > 2 : true
  })
  return {
    id: 'nano', title: 'Observatory-Nano token embeddings', kind: 'transformer',
    provenance: `Rows of the learned token-embedding matrix W_E (${vocab.length} × ${D}) of the 2-layer transformer from module 01, trained by gradient descent on next-token prediction.${wholeWordsOnly ? ' Showing whole-word tokens only (a leading · marks a word start).' : ''}`,
    words: keep.map((i) => showTok(vocab[i])),
    vecs: keep.map((i) => Array.from(model.W.tok.data.subarray(i * D, (i + 1) * D))),
    dim: D,
    groups: keep.map((i) => { const t = vocab[i]; const w = t.replace('▁', ''); return CAT_OF.get(w) ?? (t.startsWith('▁') ? (t.length > 2 ? 'function' : 'subword') : 'subword') }),
    stats: { 'model parameters': model.meta.params.toLocaleString(), 'training steps': model.meta.trainSteps, 'tokens shown': keep.length },
  }
}

export interface ContextOcc { sentence: number; position: number; label: string; text: string }
export interface ContextLayer { name: string; vecs: Vec[] }
/** The same token in different sentences, read out at every depth of the residual stream. */
export function contextualEmbeddings(model: NanoGPT, sentences: string[], token: string) {
  const id = model.vocab.indexOf(token)
  if (id < 0) throw new Error(`token “${token}” is not in the vocabulary`)
  const occ: ContextOcc[] = []
  const names = ['token embedding W_E', 'x⁽⁰⁾ = W_E + W_P', 'after block 1', 'after block 2', 'final LayerNorm']
  const layers: ContextLayer[] = names.map((name) => ({ name, vecs: [] }))
  sentences.forEach((s, si) => {
    const ids = model.encodePrompt(s).ids
    if (!ids.includes(id)) return
    const tr = model.forward(ids)
    const D = tr.x0.cols
    const rowOf = (m: { data: Float32Array }, i: number) => Array.from(m.data.subarray(i * D, (i + 1) * D))
    ids.forEach((t, i) => {
      if (t !== id) return
      occ.push({ sentence: si, position: i, label: `S${si + 1}·${i}`, text: s })
      const mats = [tr.tokEmb, tr.x0, tr.layers[0].xOut, tr.layers[1].xOut, tr.lnf.out]
      mats.forEach((m, k) => layers[k].vecs.push(rowOf(m, i)))
    })
  })
  return { occ, layers, tokenId: id }
}

/* ------------------------------------------------------------ operations */

export type Metric = 'cosine' | 'dot' | 'euclidean'
export const METRIC_LABEL: Record<Metric, string> = { cosine: 'cosine similarity', dot: 'dot product', euclidean: 'Euclidean distance' }
export const similarity = (m: Metric, a: Vec, b: Vec) => (m === 'cosine' ? cosine(a, b) : m === 'dot' ? dot(a, b) : -euclid(a, b))

export interface Neighbour { i: number; word: string; score: number }
/** k nearest words to a vector (higher similarity first; Euclidean uses distance ascending). Ties → alphabetical. */
export function nearest(space: Space, v: Vec, k: number, metric: Metric = 'cosine', exclude: number[] = []): Neighbour[] {
  const ex = new Set(exclude)
  return space.vecs.map((w, i) => ({ i, word: space.words[i], score: metric === 'euclidean' ? euclid(v, w) : similarity(metric, v, w) }))
    .filter((x) => !ex.has(x.i))
    .sort((a, b) => (metric === 'euclidean' ? a.score - b.score : b.score - a.score) || (a.word < b.word ? -1 : 1))
    .slice(0, k)
}

export function similarityMatrix(space: Space, idx: number[], metric: Metric = 'cosine') {
  return idx.map((a) => idx.map((b) => (metric === 'euclidean' ? euclid(space.vecs[a], space.vecs[b]) : similarity(metric, space.vecs[a], space.vecs[b]))))
}

export interface AnalogyResult {
  target: Vec
  ranked: Neighbour[]
  /** rank (1-based) of the expected answer among all non-input words, if given */
  expectedRank?: number
  expectedScore?: number
}
/**
 * 3CosAdd analogy: argmax_x cos(x, b − a + c) over words other than a, b, c
 * ("a is to b as c is to ?"). With `normalize`, inputs are unit-normalized first (Mikolov et al.).
 */
export function analogy(space: Space, a: number, b: number, c: number, k = 8, normalize = true, expected?: number): AnalogyResult {
  const f = (i: number) => (normalize ? unit(space.vecs[i]) : space.vecs[i])
  const target = add(sub(f(b), f(a)), f(c))
  const all = nearest(space, target, space.words.length, 'cosine', [a, b, c])
  const res: AnalogyResult = { target, ranked: all.slice(0, k) }
  if (expected !== undefined) {
    const r = all.findIndex((x) => x.i === expected)
    res.expectedRank = r < 0 ? undefined : r + 1
    res.expectedScore = r < 0 ? undefined : all[r].score
  }
  return res
}

/* ------------------------------------------------------------ projection */

export type ProjMethod = 'pca' | 'umap'
/** normalize: unit-normalize vectors before projecting, so the picture reflects direction (cosine), not length */
export interface ProjConfig { method: ProjMethod; dims: 2 | 3; nNeighbors: number; minDist: number; seed: number; nEpochs: number; normalize: boolean }
export const defaultProj: ProjConfig = { method: 'pca', dims: 2, nNeighbors: 8, minDist: 0.3, seed: 7, nEpochs: 300, normalize: true }

export interface Projection {
  coords: Vec[]
  explained?: number[]
  /** fitted PCA (mean + axes) so extra vectors, e.g. an analogy target, can be placed in the same picture */
  pcaModel?: { mean: Vec; components: Vec[] }
  /** whether vectors were unit-normalized before projecting (then extra points must be normalized too) */
  normalized: boolean
  /** neighbourhood preservation and Shepard correlation: how faithful is the picture? */
  preservation: ReturnType<typeof neighbourPreservation>
  shepard: { high: number[]; low: number[]; pairs: [number, number][]; rho: number }
}

export function project(space: Space, cfg: ProjConfig): Projection {
  const n = space.vecs.length
  let coords: Vec[]
  let explained: number[] | undefined
  let pcaModel: Projection['pcaModel']
  const input = cfg.normalize ? space.vecs.map(unit) : space.vecs
  if (cfg.method === 'pca' || n < 4) {
    const r = pca(input, cfg.dims)
    coords = r.coords.map((c) => { const out = c.slice(); while (out.length < cfg.dims) out.push(0); return out })
    explained = r.explained
    pcaModel = { mean: r.mean, components: r.components }
  } else {
    const u = new UMAP({
      nComponents: cfg.dims, nNeighbors: Math.max(2, Math.min(cfg.nNeighbors, n - 1)), minDist: cfg.minDist, nEpochs: cfg.nEpochs,
      random: rng(cfg.seed), distanceFn: (a: number[], b: number[]) => 1 - cosine(a, b),
    })
    coords = u.fit(input) // cosine distance: identical with or without normalization
  }
  // Shepard data: cosine distance in full dimension vs Euclidean distance in the picture, all pairs
  const high: number[] = [], low: number[] = [], pairs: [number, number][] = []
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { high.push(1 - cosine(space.vecs[i], space.vecs[j])); low.push(euclid(coords[i], coords[j])); pairs.push([i, j]) }
  return { coords, explained, pcaModel, normalized: cfg.normalize, preservation: neighbourPreservation(space.vecs, coords, 5), shepard: { high, low, pairs, rho: spearman(high, low) } }
}

/** Place any vector in a PCA picture: (v − mean) · axes. */
export const projectPoint = (p: Projection, v: Vec) => { const m = p.pcaModel!; const x = p.normalized ? unit(v) : v; return m.components.map((c) => dot(sub(x, m.mean), c)) }

/** Neighbourhood preservation when keeping the first k principal components, for k = 1 … maxK. */
export function preservationByDims(space: Space, maxK: number, k = 5) {
  const r = pca(space.vecs, Math.min(maxK, space.dim))
  return r.components.map((_, c) => ({ dims: c + 1, kept: r.explained.slice(0, c + 1).reduce((a, b) => a + b, 0), preservation: neighbourPreservation(space.vecs, r.coords.map((row) => row.slice(0, c + 1)), k).mean }))
}

/** Pairs whose closeness in the picture most misrepresents their closeness in full dimension (rank difference). */
export function misleadingPairs(p: Projection, k = 5) {
  const rk = (x: number[]) => { const o = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]); const r = new Array(x.length); o.forEach(([, i], j) => (r[i] = j)); return r }
  const rh = rk(p.shepard.high), rl = rk(p.shepard.low)
  const m = p.shepard.pairs.length || 1
  const scored = p.shepard.pairs.map((pair, i) => ({ pair, high: p.shepard.high[i], low: p.shepard.low[i], gap: (rh[i] - rl[i]) / m }))
  return {
    /** look close in 2D/3D but are far apart in full dimension */
    falseFriends: scored.slice().sort((a, b) => b.gap - a.gap).slice(0, k),
    /** look far apart in the picture but are close in full dimension */
    hiddenTwins: scored.slice().sort((a, b) => a.gap - b.gap).slice(0, k),
  }
}

/* ------------------------------------------------------------ challenge */

export type ChallengeKind = 'dot' | 'norm' | 'cosine' | 'analogy' | 'neighbour' | 'projection'
export interface EmbChallenge {
  id: string
  kind: ChallengeKind
  prompt: string
  /** numeric answer, or the index into `options` */
  answer: number
  options?: string[]
  vectors?: { name: string; v: number[] }[]
  explain: string
}

/** A seeded question set. The same space, projection and seed always give the same questions. */
export function makeEmbChallenges(space: Space, proj: Projection, seed: number): EmbChallenge[] {
  const r = rng(seed)
  const int = (lo: number, hi: number) => lo + Math.floor(r() * (hi - lo + 1))
  const vec = () => [int(-3, 3), int(-3, 3), int(-3, 3)]
  const nonzero = () => { let v = vec(); while (norm(v) === 0) v = vec(); return v }
  const out: EmbChallenge[] = []
  const a = nonzero(), b = nonzero()
  out.push({ id: 'dot', kind: 'dot', prompt: 'Compute the dot product a · b.', answer: dot(a, b), vectors: [{ name: 'a', v: a }, { name: 'b', v: b }],
    explain: `a · b = ${a.map((x, i) => `(${x})(${b[i]})`).join(' + ')} = ${dot(a, b)}.` })
  const c = nonzero()
  out.push({ id: 'norm', kind: 'norm', prompt: 'Compute the length ‖c‖ (2 decimals).', answer: norm(c), vectors: [{ name: 'c', v: c }],
    explain: `‖c‖ = √(${c.map((x) => `${x}²`).join(' + ')}) = √${dot(c, c)} = ${norm(c).toFixed(4)}.` })
  const d = nonzero(), e = nonzero()
  out.push({ id: 'cosine', kind: 'cosine', prompt: 'Compute cos(d, e) (3 decimals).', answer: cosine(d, e), vectors: [{ name: 'd', v: d }, { name: 'e', v: e }],
    explain: `cos = d·e / (‖d‖‖e‖) = ${dot(d, e)} / (${norm(d).toFixed(3)} × ${norm(e).toFixed(3)}) = ${cosine(d, e).toFixed(4)}.` })

  const n = space.words.length
  const pick = () => Math.floor(r() * n)
  // nearest neighbour in full dimension, with the 2D nearest neighbour as a tempting distractor
  let q = pick(), guard = 0
  const nn2d = (i: number) => Array.from({ length: n }, (_, j) => j).filter((j) => j !== i).sort((x, y) => euclid(proj.coords[i], proj.coords[x]) - euclid(proj.coords[i], proj.coords[y]) || x - y)[0]
  const trueNN = (i: number) => nearest(space, space.vecs[i], 1, 'cosine', [i])[0].i
  while (guard++ < 50 && trueNN(q) === nn2d(q)) q = pick() // prefer a word where the picture misleads
  const t = trueNN(q), p2 = nn2d(q)
  const opts = new Set<number>([t, p2])
  const far = nearest(space, space.vecs[q], n, 'cosine', [q])
  for (const cand of [far[Math.floor(n / 3)], far[n - 3], far[2], far[3]]) if (cand && opts.size < 4) opts.add(cand.i)
  const optList = [...opts].sort((x, y) => (space.words[x] < space.words[y] ? -1 : 1))
  out.push({ id: 'neighbour', kind: 'neighbour', prompt: `In the full ${space.dim}-dimensional space, which word is most cosine-similar to “${space.words[q]}”? (The projection may disagree.)`,
    options: optList.map((i) => space.words[i]), answer: optList.indexOf(t),
    explain: `cos(${space.words[q]}, ${space.words[t]}) = ${cosine(space.vecs[q], space.vecs[t]).toFixed(4)} is the highest. ${t === p2 ? 'Here the projection agrees.' : `In the projection, “${space.words[p2]}” sits closest, but its full-dimensional cosine is only ${cosine(space.vecs[q], space.vecs[p2]).toFixed(4)}.`}` })

  // analogy on the hand-built space is exact, so it makes a clean hand calculation
  const toy = toySpace()
  const W = (w: string) => toy.words.indexOf(w)
  const sets: [string, string, string, string][] = [['man', 'king', 'woman', 'queen'], ['boy', 'prince', 'girl', 'princess'], ['cat', 'kitten', 'dog', 'puppy'], ['man', 'boy', 'woman', 'girl'], ['king', 'prince', 'queen', 'princess']]
  const [x, y, z, ans] = sets[Math.floor(r() * sets.length)]
  const res = analogy(toy, W(x), W(y), W(z), 4, false)
  const aOpts = [...new Set([ans, ...res.ranked.slice(0, 3).map((g) => g.word), 'apple'])].slice(0, 4).sort()
  out.push({ id: 'analogy', kind: 'analogy', prompt: `With the hand-built vectors, compute ${y} − ${x} + ${z}. Which word is the closest (by cosine, excluding the three inputs)?`,
    options: aOpts, answer: aOpts.indexOf(res.ranked[0].word),
    vectors: [x, y, z].map((w) => ({ name: w, v: toy.vecs[W(w)] })),
    explain: `${y} − ${x} + ${z} = [${res.target.join(', ')}], which is exactly the vector of “${res.ranked[0].word}” (cos = ${res.ranked[0].score.toFixed(3)}). Real learned embeddings rarely land exactly on the answer.` })

  if (proj.explained) {
    const s = proj.explained.slice(0, 2).reduce((u, v) => u + v, 0)
    out.push({ id: 'projection', kind: 'projection', prompt: 'What percentage of the total variance do the first two principal components keep? (whole number, %)', answer: s * 100,
      explain: `PC1 keeps ${(proj.explained[0] * 100).toFixed(1)}% and PC2 ${(proj.explained[1] * 100).toFixed(1)}%: ${(s * 100).toFixed(1)}% in total. The other ${((1 - s) * 100).toFixed(1)}% of the variance is invisible in the 2D picture.` })
  } else {
    out.push({ id: 'projection', kind: 'projection', prompt: `Neighbourhood preservation for this UMAP picture is reported as a share of each word's 5 true nearest neighbours that stay among its 5 nearest on screen. What is it, in %? (whole number)`, answer: proj.preservation.mean * 100,
      explain: `${(proj.preservation.mean * 100).toFixed(1)}% of true neighbourhoods survive (chance would be ${(proj.preservation.chance * 100).toFixed(1)}%).` })
  }
  return out
}

export const parseNum = (s: string) => {
  const t = s.trim().replace(',', '.').replace('%', '')
  const f = t.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/)
  if (f) return Number(f[1]) / Number(f[2])
  return t === '' ? NaN : Number(t)
}
/** Tolerance: 1% relative or 0.006 absolute (so a value rounded to 2–3 decimals is accepted). */
export const numCorrect = (answer: number, given: number, kind?: ChallengeKind) =>
  Number.isFinite(given) && Math.abs(given - answer) <= (kind === 'projection' ? 1 : Math.max(0.006, Math.abs(answer) * 0.01))

export { scale }
