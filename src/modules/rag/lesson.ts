/** Stage list, explanations, learning objectives, examples and student questions for module 11. */
export type StageId = 'docs' | 'chunk' | 'embed' | 'index' | 'query' | 'similarity' | 'retrieve' | 'rerank' | 'prompt' | 'generate' | 'cite' | 'failures'
export interface Stage { id: StageId; short: string; title: string; group: string; tex: string; text: string; beginner: string; advanced: string }

export const STAGES: Stage[] = [
  { id: 'docs', short: 'docs', title: 'The knowledge base', group: 'ingest', tex: '\\mathcal{D} = \\{d_1, \\dots, d_n\\}',
    text: 'RAG answers questions from a collection of documents instead of only from what a model memorized. Everything starts with ingestion: collecting and cleaning the documents.', beginner: 'Think of an open-book exam: the documents are the book.', advanced: 'Ingestion quality (stale pages, duplicates, conflicting versions) bounds everything downstream; this corpus contains one outdated page on purpose.' },
  { id: 'chunk', short: 'chunk', title: 'Chunking the documents', group: 'ingest', tex: '\\#\\text{chunks} = \\left\\lceil \\frac{n - o}{s - o} \\right\\rceil \\;(n > s)',
    text: 'Documents are split into passages of s words, repeating o words between neighbours. Retrieval returns chunks, not documents, so chunk boundaries decide what evidence can be found together.', beginner: 'Small pieces are easy to match but may cut a sentence in half; big pieces keep context but contain more unrelated text.', advanced: 'Fixed windows ignore structure; sentence packing respects sentence boundaries. Overlap trades index size for robustness at boundaries.' },
  { id: 'embed', short: 'embed', title: 'Turning chunks into vectors', group: 'index', tex: 'w_{t,c} = (1 + \\ln \\mathrm{tf}_{t,c})\\cdot\\mathrm{idf}_t,\\quad A = U\\Sigma V^{\\top}',
    text: 'Each chunk becomes a sparse TF-IDF vector over the vocabulary, and a dense k-dimensional LSA vector learned from this corpus by a truncated singular value decomposition.', beginner: 'Rare words that appear in a chunk get a large weight; common words get a small one.', advanced: 'LSA is a linear, unsupervised embedding trained on this corpus only. A neural embedding model would be trained on far more text and could relate words never seen here.' },
  { id: 'index', short: 'index', title: 'The vector index', group: 'index', tex: '\\text{exact: } O(N\\,k)\\quad\\text{IVF: search the } n_{\\text{probe}} \\text{ nearest of } n_{\\text{list}} \\text{ clusters}',
    text: 'The vectors are stored in an index. An exact index compares the query with every vector; an inverted-file (IVF) index clusters vectors and searches only the clusters nearest to the query.', beginner: 'Like a library sorted by topic: look only on the closest shelves.', advanced: 'Approximate search can miss the true nearest neighbour when it lives in an unprobed cluster: recall vs speed.' },
  { id: 'query', short: 'query', title: 'Embedding the question', group: 'retrieve', tex: 'q \\mapsto \\hat q = q\\,V \\in \\mathbb{R}^k',
    text: 'The question goes through the same text processing and the same vector space as the chunks. Words the corpus never contains carry no signal for lexical or LSA retrieval.', beginner: 'The question must speak the documents\' language: “ChatGPT” finds nothing if the handbook says “generative AI assistants”.', advanced: 'Query rewriting (here a hand-written synonym table standing in for an LLM rewriter) adds corpus vocabulary to the query.' },
  { id: 'similarity', short: 'cos', title: 'Cosine similarity and BM25', group: 'retrieve', tex: '\\cos(q, c) = \\frac{q \\cdot c}{\\lVert q\\rVert\\,\\lVert c\\rVert}',
    text: 'Dense and TF-IDF retrieval score a chunk by the cosine of the angle between query and chunk vectors. BM25 scores the query terms a chunk contains, with saturating term frequency and length normalization.', beginner: 'Cosine 1 means the same direction, 0 means nothing in common.', advanced: 'BM25: Σ idf·tf(k₁+1)/(tf + k₁(1 − b + b·len/avglen)) with k₁ = 1.5, b = 0.75.' },
  { id: 'retrieve', short: 'top-k', title: 'Retrieving the top-k chunks', group: 'retrieve', tex: '\\mathrm{RRF}(c) = \\sum_{m} \\frac{1}{60 + \\mathrm{rank}_m(c)}',
    text: 'All chunks are ranked by the chosen method and the k best are kept. Hybrid search fuses the dense and BM25 rankings, here with reciprocal rank fusion or a weighted sum of normalized scores.', beginner: 'Keep the k passages that look most relevant.', advanced: 'RRF needs no score calibration; a weighted sum depends on min-max normalization of each score list.' },
  { id: 'rerank', short: 'rerank', title: 'Reranking the candidates', group: 'retrieve', tex: '\\text{MMR: } \\arg\\max_{d}\\; \\lambda\\,\\mathrm{rel}(d) - (1-\\lambda)\\max_{s\\in S}\\cos(d, s)',
    text: 'A second stage reorders a larger candidate pool. Maximal marginal relevance balances relevance against redundancy; the coverage reranker is a transparent heuristic standing in for a trained cross-encoder.', beginner: 'Avoid giving the model three copies of the same paragraph.', advanced: 'Real systems often use a cross-encoder that reads query and passage together; none is available offline here, and the heuristic is labelled as such.' },
  { id: 'prompt', short: 'prompt', title: 'Building the prompt', group: 'generate', tex: '\\text{tokens} \\approx 1.3 \\times \\text{words} \\le \\text{budget}',
    text: 'Instructions, the numbered passages and the question are assembled into one prompt. Passages that do not fit in the context budget are dropped, lowest rank first.', beginner: 'The model only sees what is in the prompt.', advanced: 'Order and budget matter: relevant evidence that is retrieved but dropped is as good as not retrieved.' },
  { id: 'generate', short: 'answer', title: 'Grounded vs ungrounded answers', group: 'generate', tex: '\\text{support}(s) = \\frac{\\sum_{t \\in q \\cap s}\\mathrm{idf}_t}{\\sum_{t\\in q}\\mathrm{idf}_t}',
    text: 'The answer generator here is an extractive simulation, not a language model: it quotes the best-supported sentences of the prompt with citations, or abstains below a support threshold. The closed-book answer is a scripted simulation of a model answering from memory.', beginner: 'With sources, the answer can be checked. Without them, it may sound sure and still be wrong.', advanced: 'A real LLM paraphrases and can combine passages, but can also ignore or misread them; grounding reduces, not eliminates, hallucination.' },
  { id: 'cite', short: 'cite', title: 'Citations and verification', group: 'generate', tex: '\\text{faithfulness} \\approx \\frac{|\\text{answer terms} \\cap \\text{cited terms}|}{|\\text{answer terms}|}',
    text: 'Every claim should point to a passage that supports it. Here each cited sentence is checked to occur in its cited passage, and the share of answer terms found in the passages is measured.', beginner: 'A citation is a promise you can check.', advanced: 'Term overlap is a crude faithfulness proxy; real evaluations use entailment models or human judgement.' },
  { id: 'failures', short: 'eval', title: 'Failure analysis and evaluation', group: 'evaluate', tex: '\\text{Recall@}k,\\quad \\mathrm{MRR} = \\frac{1}{|Q|}\\sum_q \\frac{1}{\\mathrm{rank}_q}',
    text: 'Each example question has gold evidence, so retrieval and answers can be scored. Failures are diagnosed: evidence never retrieved, cut by a chunk boundary, dropped from the prompt, outdated sources, missing reasoning, or a question the documents cannot answer.', beginner: 'Find out where the pipeline broke, not just that it broke.', advanced: 'Sweep chunk size and overlap to see how recall changes; every number is recomputed from scratch.' },
]

export const OBJECTIVES = [
  'Explain each stage of a RAG pipeline and what flows between them.',
  'Predict how chunk size and overlap change the number of chunks and the chance of splitting evidence.',
  'Compute TF-IDF weights, cosine similarity, BM25 term contributions and reciprocal rank fusion by hand.',
  'Compare dense, lexical and hybrid retrieval, and recognise a lexical gap.',
  'Distinguish a grounded, cited answer from a fluent but unsupported one, and know when abstaining is correct.',
  'Diagnose retrieval and generation failures with Recall@k, MRR and per-question evidence.',
]

export const EXAMPLES: { id: string; title: string; corpus: string; qid: string; stage: StageId; cfg?: 'small-chunks' | 'rewrite' | 'mmr'; what: string }[] = [
  { id: 'e1', title: 'Grounded vs closed-book', corpus: 'course', qid: 'c1', stage: 'generate', what: 'The handbook says five percentage points per day; the closed-book simulation confidently says 10 %.' },
  { id: 'e2', title: 'Evidence cut in half', corpus: 'course', qid: 'c1', stage: 'chunk', cfg: 'small-chunks', what: 'With 12-word chunks and no overlap, the late-penalty sentence is split across two chunks.' },
  { id: 'e3', title: 'Lexical gap, then query rewriting', corpus: 'solar', qid: 's2', stage: 'query', cfg: 'rewrite', what: '“hottest” never occurs in the notes; rewriting adds “highest surface temperature” and Venus is found.' },
  { id: 'e4', title: 'Conflicting sources', corpus: 'solar', qid: 's3', stage: 'retrieve', what: 'Both the 2006 IAU page and an outdated 1999 page are retrieved. Which one should the answer trust?' },
  { id: 'e5', title: 'Knowing when to say “I don\'t know”', corpus: 'solar', qid: 's5', stage: 'generate', what: 'No human has visited Mars; the right answer is to abstain, while the closed-book simulation invents one.' },
  { id: 'e6', title: 'Redundant results and MMR', corpus: 'solar', qid: 's1', stage: 'rerank', cfg: 'mmr', what: 'With overlapping 25-word chunks, three pieces of the same Mars note fill the top-3. MMR keeps the best one first and brings in passages from other documents.' },
]

export const STUDENT_QUESTIONS = [
  'Halving the chunk size with the same overlap roughly doubles the number of chunks. What happens to the average BM25 length normalization?',
  'Why can increasing top-k improve recall but hurt the answer?',
  'A question uses a word that never appears in the corpus. Which retrieval methods can still find the right chunk here, and which kind of embedding model could?',
  'When is “I don\'t know” the correct output of a RAG system, and how would you measure it?',
  'Reciprocal rank fusion uses ranks instead of scores. Give one advantage and one disadvantage.',
  'Two retrieved passages contradict each other. What should the prompt or the system do?',
]
