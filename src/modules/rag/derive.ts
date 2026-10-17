/** Memoized pipeline for the current state: chunks → index → query → retrieval → rerank → prompt → answer. */
import { useMemo } from 'react'
import { buildPipeline, evaluate, goldChunksFor, runQuery, summarize, type Pipeline, type PipelineConfig, type QEval, type RunResult } from '../../engine/rag'
import { corpusById, type RagDoc, type RagQuestion } from '../../data/rag-corpora'
import { docsOf, questionOf, queryText, useRag } from './store'

export interface Lab { docs: RagDoc[]; pipe: Pipeline; run: RunResult; question: RagQuestion | null; query: string; gold: { ids: number[]; split: boolean } }

export function usePipe(cfg: PipelineConfig, docs: RagDoc[]): Pipeline {
  const key = JSON.stringify([cfg.chunk, cfg.dims, cfg.retrieve.index, cfg.retrieve.nlist, cfg.retrieve.seed])
  return useMemo(() => buildPipeline(docs, cfg), [docs, key]) // eslint-disable-line react-hooks/exhaustive-deps
}

export function useLab(which: 'A' | 'B' = 'A'): Lab {
  const corpusId = useRag((s) => s.corpusId), extra = useRag((s) => s.extraDocs), qid = useRag((s) => s.qid), custom = useRag((s) => s.custom)
  const cfg = useRag((s) => (which === 'A' ? s.cfg : s.cfgB))
  const docs = useMemo(() => docsOf({ corpusId, extraDocs: extra }), [corpusId, extra])
  const pipe = usePipe(cfg, docs)
  const question = questionOf({ corpusId, qid, custom }), query = queryText({ corpusId, qid, custom })
  const run = useMemo(() => runQuery(pipe, query, cfg), [pipe, query, cfg])
  const gold = useMemo(() => goldChunksFor(pipe.chunks, question?.gold ?? ''), [pipe, question])
  return { docs, pipe, run, question, query, gold }
}

export function useEvaluation(which: 'A' | 'B' = 'A'): { ev: QEval[]; summary: ReturnType<typeof summarize>; questions: RagQuestion[] } {
  const corpusId = useRag((s) => s.corpusId), extra = useRag((s) => s.extraDocs)
  const cfg = useRag((s) => (which === 'A' ? s.cfg : s.cfgB))
  const docs = useMemo(() => docsOf({ corpusId, extraDocs: extra }), [corpusId, extra])
  const pipe = usePipe(cfg, docs)
  const questions = corpusById(corpusId).questions
  return useMemo(() => { const ev = evaluate(pipe, questions, cfg); return { ev, summary: summarize(ev, questions), questions } }, [pipe, questions, cfg])
}
