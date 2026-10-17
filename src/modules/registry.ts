export interface ModuleInfo { id: string; title: string; blurb: string; status: 'ready' | 'planned' }

/** Curriculum order. Modules marked planned share the engine, shell and lesson format of module 1. */
export const MODULES: ModuleInfo[] = [
  { id: 'llm', title: 'Inside an LLM', blurb: 'Follow one prompt through a real trained transformer, tensor by tensor.', status: 'ready' },
  { id: 'transformer', title: 'Inside the Transformer block', blurb: 'LayerNorm, Q/K/V, scaled dot-product attention, heads, residuals and the MLP, computed exactly; GPT pre-LN vs BERT post-LN; a real model.', status: 'ready' },
  { id: 'attention', title: 'Attention', blurb: 'Scaled dot-product and multi-head attention from scratch: Q/K/V, dot products, scaling, masks, softmax, weighted values, heads, concatenation and W_O, all computed exactly.', status: 'ready' },
  { id: 'mlp', title: 'Neural nets & backprop', blurb: 'Build an MLP, step through forward and backward passes, verify every gradient and watch it learn.', status: 'ready' },
  { id: 'bpe', title: 'BPE tokenization', blurb: 'Train a vocabulary merge by merge on your own text.', status: 'planned' },
  { id: 'embeddings', title: 'Embedding space', blurb: 'Cosine similarity, analogies, PCA/UMAP distortion and contextual embeddings.', status: 'ready' },
  { id: 'decoding', title: 'Decoding strategies', blurb: 'Greedy, beam, temperature, top-k and top-p generation.', status: 'planned' },
  { id: 'ngram', title: 'N-gram language models', blurb: 'Counts, smoothing, Kneser-Ney and perplexity on your own corpus.', status: 'ready' },
  { id: 'rnn', title: 'RNN, LSTM & Transformer', blurb: 'Unrolled recurrence, LSTM gates, self-attention, gradients through time and real training results.', status: 'ready' },
  { id: 'training', title: 'Training & inference', blurb: 'Loss curves, optimizers, KV-cache and batching.', status: 'planned' },
  { id: 'rag', title: 'RAG pipeline', blurb: 'Chunking, TF-IDF/BM25/dense retrieval, hybrid fusion, reranking, prompts, grounded vs ungrounded answers, citations and failure analysis.', status: 'ready' },
]
