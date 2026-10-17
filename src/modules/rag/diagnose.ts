/** Failure diagnosis for one evaluated question (pure, tested). */
import type { RagDoc, RagQuestion } from '../../data/rag-corpora'
import type { Pipeline, QEval } from '../../engine/rag'

export type Verdict = 'ok' | 'abstain-ok' | 'hallucination' | 'miss' | 'split' | 'budget' | 'reasoning' | 'distractor' | 'generator' | 'stale' | 'over-abstain'
export interface Diagnosis { verdict: Verdict; good: boolean; text: string }

export function diagnose(e: QEval, q: RagQuestion, pipe: Pipeline, k: number, docs: RagDoc[], retrievedDocIds: string[]): Diagnosis {
  if (!q.gold) return e.abstained ? { verdict: 'abstain-ok', good: true, text: 'Correctly abstained: the documents do not answer this question.' } : { verdict: 'hallucination', good: false, text: 'Answered a question the documents cannot answer (hallucination risk).' }
  const stale = docs.filter((d) => d.note && retrievedDocIds.includes(d.id))
  if (e.answerCorrect) return stale.length ? { verdict: 'stale', good: true, text: `Correct, but an outdated source was also retrieved (${stale.map((d) => d.title).join(', ')}): a generator could have trusted it.` } : { verdict: 'ok', good: true, text: 'Evidence retrieved, kept in the prompt, and quoted correctly.' }
  if (e.split && !e.recall) return { verdict: 'split', good: false, text: 'The evidence was cut by a chunk boundary and the partial chunks were not retrieved. Try larger chunks or more overlap.' }
  if (!e.recall) return { verdict: 'miss', good: false, text: `Retrieval miss: the best gold chunk is ranked ${Number.isFinite(e.goldRank) ? e.goldRank : '∞'} (top-k = ${k}).${q.tag === 'lexical-gap' ? ' The question uses words the documents do not (a lexical gap): try query rewriting.' : ''}` }
  if (!e.inPrompt) return { verdict: 'budget', good: false, text: 'Retrieved, but dropped from the prompt by the context budget.' }
  if (e.abstained) return { verdict: 'over-abstain', good: false, text: `The evidence was in the prompt but the support threshold made the generator abstain.${q.tag === 'lexical-gap' ? ' Key words of the question do not occur in the documents (a lexical gap): try query rewriting.' : ''}` }
  if (e.split) return { verdict: 'split', good: false, text: 'Only part of the evidence sentence was in a retrieved chunk (split by a boundary).' }
  if (q.tag === 'multi-hop') return { verdict: 'reasoning', good: false, text: 'The evidence was in the prompt, but answering needs reasoning (e.g. comparing numbers) that quoting the best-matching sentence cannot do.' }
  if (q.tag === 'distractor') return { verdict: 'distractor', good: false, text: 'A distractor sentence matched the question words better than the evidence.' }
  void pipe
  return { verdict: 'generator', good: false, text: 'The evidence was in the prompt, but the generator quoted another sentence.' }
}
