/** Seeded student challenges for module 11; every answer is computed by the engine or by the formulas it implements. */
import { rng } from '../../engine/decoding'
import { BM25_B, BM25_K1, RRF_K, cosine, fixedChunkCount, sparseDot, type Pipeline, type QueryVec } from '../../engine/rag'

export interface Challenge { id: string; title: string; prompt: string; kind: 'num' | 'mc'; answer: number | string; options?: string[]; tol?: number; solution: string[]; misconception: string }

export function check(c: Challenge, given: string): boolean {
  if (c.kind === 'mc') return given === c.answer
  const v = parseFloat(given.replace(',', '.').replace('−', '-'))
  const a = c.answer as number
  return Number.isFinite(v) && Math.abs(v - a) <= (c.tol ?? Math.max(0.005, 0.01 * Math.abs(a)))
}
const f = (v: number, d = 4) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d)
const pn = (v: number) => (v < 0 ? `(${v})` : `${v}`)

export function makeChallenges(seed: number, live: { pipe: Pipeline; q: QueryVec; chunk: number }): Challenge[] {
  const r = rng(seed * 4099 + 13)
  const int = (a: number, b: number) => a + Math.floor(r() * (b - a + 1))
  const one = () => Math.round((r() * 2 - 1) * 10) / 10
  const n = int(60, 200), s = [20, 25, 30, 40, 50][int(0, 4)], o = [0, 5, 10][int(0, 2)]
  const N = int(20, 200), df = int(1, 15)
  const idfS = Math.log((N + 1) / (df + 1)) + 1
  const cnt = int(1, 5), w = (1 + Math.log(cnt)) * idfS
  const a = [one(), one(), one()], b = [one(), one(), one()]
  if (!a.some((x) => Math.abs(x) > 0)) a[0] = 0.5
  if (!b.some((x) => Math.abs(x) > 0)) b[1] = 0.5
  const cos = cosine(a, b)
  const tf = int(1, 4), len = int(20, 80), avg = int(30, 60), bidf = Math.round((0.5 + r() * 2.5) * 100) / 100
  const bmC = (bidf * tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + (BM25_B * len) / avg))
  const r1 = int(1, 10), r2 = int(1, 10), rrf = 1 / (RRF_K + r1) + 1 / (RRF_K + r2)
  const k = int(3, 5), relAt = Array.from({ length: k }, () => r() < 0.45), totalRel = Math.max(1, relAt.filter(Boolean).length + int(0, 2))
  const recall = relAt.filter(Boolean).length / totalRel, precision = relAt.filter(Boolean).length / k
  const ranks = [int(1, 4), int(1, 6), int(1, 3)], mrr = ranks.reduce((t, x) => t + 1 / x, 0) / 3
  const liveCos = sparseDot(live.q.tfidf, live.pipe.index.tfidf[live.chunk])
  return [
    { id: 'chunks', title: 'Number of chunks', kind: 'num', answer: fixedChunkCount(n, s, o), tol: 0, prompt: `A document has ${n} words. Fixed windows of ${s} words with an overlap of ${o}. How many chunks?`,
      solution: [`step = s − o = ${s - o}`, `chunks = ⌈(n − o) / step⌉ = ⌈${n - o} / ${s - o}⌉ = ${fixedChunkCount(n, s, o)}`], misconception: 'Dividing n by s and ignoring the overlap, which makes every window start s words after the previous one.' },
    { id: 'idf', title: 'Inverse document frequency', kind: 'num', answer: idfS, prompt: `N = ${N} chunks, a term occurs in df = ${df} of them. Compute idf = ln((N + 1)/(df + 1)) + 1.`,
      solution: [`(N + 1)/(df + 1) = ${N + 1}/${df + 1} = ${f((N + 1) / (df + 1))}`, `ln(…) + 1 = ${f(idfS)}`], misconception: 'Thinking frequent terms get a high idf: idf is high for rare terms.' },
    { id: 'tfidf', title: 'TF-IDF weight', kind: 'num', answer: w, prompt: `The same term occurs ${cnt} time${cnt > 1 ? 's' : ''} in a chunk. With tf = 1 + ln(count), what is tf × idf (before normalization)?`,
      solution: [`tf = 1 + ln ${cnt} = ${f(1 + Math.log(cnt))}`, `tf × idf = ${f(1 + Math.log(cnt))} × ${f(idfS)} = ${f(w)}`], misconception: 'Using the raw count: sublinear tf keeps one repeated word from dominating.' },
    { id: 'cos', title: 'Cosine similarity', kind: 'num', answer: cos, prompt: `Compute cos(a, b) for a = [${a.join(', ')}] and b = [${b.join(', ')}].`,
      solution: [`a·b = ${a.map((x, i) => `${pn(x)}×${pn(b[i])}`).join(' + ')} = ${f(a.reduce((t, x, i) => t + x * b[i], 0))}`, `‖a‖ = ${f(Math.hypot(...a))}, ‖b‖ = ${f(Math.hypot(...b))}`, `cos = ${f(cos)}`], misconception: 'Forgetting to divide by the norms: the dot product alone grows with vector length.' },
    { id: 'bm25', title: 'BM25 term contribution', kind: 'num', answer: bmC, prompt: `A query term has idf ${bidf} and occurs tf = ${tf} times in a chunk of ${len} terms; the average chunk has ${avg}. With k₁ = ${BM25_K1}, b = ${BM25_B}, compute its BM25 contribution.`,
      solution: [`length factor = 1 − b + b·len/avg = ${f(1 - BM25_B + (BM25_B * len) / avg)}`, `idf · tf(k₁+1) / (tf + k₁·factor) = ${bidf}·${tf}·${BM25_K1 + 1} / (${tf} + ${BM25_K1}·${f(1 - BM25_B + (BM25_B * len) / avg)}) = ${f(bmC)}`], misconception: 'Assuming BM25 grows linearly with term frequency: it saturates.' },
    { id: 'rrf', title: 'Reciprocal rank fusion', kind: 'num', answer: rrf, prompt: `A chunk is ranked ${r1} by dense retrieval and ${r2} by BM25. What is its RRF score (k = 60)?`,
      solution: [`1/(60 + ${r1}) + 1/(60 + ${r2}) = ${f(1 / (60 + r1))} + ${f(1 / (60 + r2))} = ${f(rrf, 5)}`], misconception: 'Adding the raw similarity scores, which live on different scales.' },
    { id: 'recall', title: 'Recall@k', kind: 'num', answer: recall, prompt: `There are ${totalRel} relevant chunks. The top-${k} results are ${relAt.map((x) => (x ? 'R' : 'N')).join(' ')} (R = relevant). What is Recall@${k}? (precision@${k} would be ${f(precision, 2)})`,
      solution: [`relevant retrieved = ${relAt.filter(Boolean).length}`, `Recall@${k} = ${relAt.filter(Boolean).length} / ${totalRel} = ${f(recall)}`], misconception: 'Dividing by k: that is precision@k.' },
    { id: 'mrr', title: 'Mean reciprocal rank', kind: 'num', answer: mrr, prompt: `Three questions find their first relevant chunk at ranks ${ranks.join(', ')}. What is the MRR?`,
      solution: [`(1/${ranks[0]} + 1/${ranks[1]} + 1/${ranks[2]}) / 3 = ${f(mrr)}`], misconception: 'Averaging the ranks and then inverting.' },
    { id: 'gap', title: 'Lexical gap', kind: 'mc', answer: 'Rewrite or expand the query with corpus vocabulary, or use an embedding model trained on broad text', options: ['Rewrite or expand the query with corpus vocabulary, or use an embedding model trained on broad text', 'Increase BM25 parameter k₁', 'Lower the abstention threshold', 'Use smaller chunks'], prompt: 'A student asks about “ChatGPT”, but the handbook only says “generative AI assistants”. Which change addresses this?',
      solution: ['No lexical method can match a word that never occurs in the corpus.', 'Query rewriting adds corpus words; a neural embedding trained on large corpora knows the relation; this lab\'s LSA embedding, trained only on this corpus, does not.'], misconception: 'Thinking any dense embedding fixes it: it depends on what the embedding was trained on.' },
    { id: 'abstain', title: 'When to abstain', kind: 'mc', answer: 'When the retrieved passages do not support any answer', options: ['When the retrieved passages do not support any answer', 'Whenever the top cosine is below 0.9', 'Never: a model should always answer', 'Only when the prompt exceeds the budget'], prompt: 'When should a RAG system answer “I don\'t know”?',
      solution: ['Grounding means every claim is supported. If no passage supports an answer, inventing one is hallucination.'], misconception: 'Treating an answer as better than no answer.' },
    { id: 'live', title: 'From the live lab', kind: 'num', answer: liveCos, prompt: `In the lab right now: what is the TF-IDF cosine between the question “${live.q.text}” and chunk #${live.chunk}? (open stage 6 to check your work)`,
      solution: [`Both vectors are unit length, so cos = Σ over shared terms of q_t · c_t = ${f(liveCos)}.`], misconception: 'Including terms that occur in only one of the two vectors: they contribute 0.' },
  ]
}
