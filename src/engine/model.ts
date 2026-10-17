/**
 * Observatory-Nano: a 2-layer, 4-head GPT trained by training/train.py.
 * This forward pass records every intermediate tensor so the UI can show
 * the real numbers flowing through the network.
 */
import raw from '../data/nano-model.json'
import { BPETokenizer, type Pair } from './bpe'
import {
  add, concatCols, row, decodeB64Float32, gelu, layerNorm, linear, map, mat, sliceCols, softmax, type Mat,
} from './tensor'

export interface ModelMeta {
  name: string; vocabSize: number; ctx: number; dModel: number; nHeads: number; nLayers: number; dMlp: number
  params: number; trainSteps: number; batchSize: number; optimizer: string; corpusChars: number
  corpusSentences: number; finalLoss: number
}
interface RawTensor { shape: number[]; b64: string }
interface RawModel { meta: ModelMeta; vocab: string[]; chars: string[]; merges: Pair[]; weights: Record<string, RawTensor> }

export interface HeadTrace { q: Mat; k: Mat; v: Mat; scores: Mat; probs: Mat; out: Mat }
export interface LayerTrace {
  xIn: Mat
  ln1: { out: Mat; means: number[]; stds: number[] }
  q: Mat; k: Mat; v: Mat
  heads: HeadTrace[]
  concat: Mat
  attnOut: Mat
  xMid: Mat
  ln2: { out: Mat; means: number[]; stds: number[] }
  pre: Mat
  act: Mat
  mlpOut: Mat
  xOut: Mat
}
export interface ForwardTrace {
  ids: number[]
  tokens: string[]
  tokEmb: Mat
  posEmb: Mat
  x0: Mat
  layers: LayerTrace[]
  lnf: { out: Mat; means: number[]; stds: number[] }
  logits: Mat
}

export class NanoGPT {
  readonly meta: ModelMeta
  readonly vocab: string[]
  readonly tokenizer: BPETokenizer
  readonly W: Record<string, Mat> = {}
  readonly B: Record<string, Float32Array> = {}
  static readonly BOS = 0

  constructor(m: RawModel = raw as unknown as RawModel) {
    this.meta = m.meta
    this.vocab = m.vocab
    this.tokenizer = new BPETokenizer(m.vocab, m.merges)
    for (const [k, t] of Object.entries(m.weights)) {
      const data = decodeB64Float32(t.b64)
      if (t.shape.length === 2) this.W[k] = mat(t.shape[0], t.shape[1], data)
      else this.B[k] = data
    }
  }

  get headDim() { return this.meta.dModel / this.meta.nHeads }

  /** Encode a prompt the way training data looked: <bos> + " " + text. */
  encodePrompt(text: string) {
    const enc = this.tokenizer.encode(' ' + text.trim())
    const ids = [NanoGPT.BOS, ...enc.ids].slice(-this.meta.ctx)
    return { ...enc, ids, truncated: enc.ids.length + 1 > this.meta.ctx }
  }

  forward(ids: number[]): ForwardTrace {
    const { dModel: D, nHeads: H, nLayers: L, ctx } = this.meta
    if (ids.length === 0) throw new Error('forward() needs at least one token')
    if (ids.length > ctx) throw new Error(`context length ${ids.length} exceeds ${ctx}`)
    const n = ids.length
    const HD = D / H
    const tokEmb = mat(n, D), posEmb = mat(n, D)
    ids.forEach((id, i) => {
      tokEmb.data.set(this.W.tok.data.subarray(id * D, (id + 1) * D), i * D)
      posEmb.data.set(this.W.pos.data.subarray(i * D, (i + 1) * D), i * D)
    })
    let x = add(tokEmb, posEmb)
    const x0 = x
    const layers: LayerTrace[] = []
    for (let l = 0; l < L; l++) {
      const xIn = x
      const ln1 = layerNorm(x, this.B[`${l}.ln1.g`], this.B[`${l}.ln1.b`])
      const q = linear(ln1.out, this.W[`${l}.Wq.W`], this.B[`${l}.Wq.b`])
      const k = linear(ln1.out, this.W[`${l}.Wk.W`], this.B[`${l}.Wk.b`])
      const v = linear(ln1.out, this.W[`${l}.Wv.W`], this.B[`${l}.Wv.b`])
      const heads: HeadTrace[] = []
      for (let h = 0; h < H; h++) {
        const qh = sliceCols(q, h * HD, HD), kh = sliceCols(k, h * HD, HD), vh = sliceCols(v, h * HD, HD)
        const scores = mat(n, n)
        const probs = mat(n, n)
        for (let i = 0; i < n; i++) {
          const rowScores: number[] = []
          for (let j = 0; j < n; j++) {
            let s = -Infinity
            if (j <= i) {
              s = 0
              for (let d = 0; d < HD; d++) s += qh.data[i * HD + d] * kh.data[j * HD + d]
              s /= Math.sqrt(HD)
            }
            scores.data[i * n + j] = s
            rowScores.push(s)
          }
          probs.data.set(softmax(rowScores), i * n)
        }
        const out = mat(n, HD)
        for (let i = 0; i < n; i++)
          for (let j = 0; j <= i; j++) {
            const p = probs.data[i * n + j]
            for (let d = 0; d < HD; d++) out.data[i * HD + d] += p * vh.data[j * HD + d]
          }
        heads.push({ q: qh, k: kh, v: vh, scores, probs, out })
      }
      const concat = concatCols(heads.map((h) => h.out))
      const attnOut = linear(concat, this.W[`${l}.Wo.W`], this.B[`${l}.Wo.b`])
      const xMid = add(xIn, attnOut)
      const ln2 = layerNorm(xMid, this.B[`${l}.ln2.g`], this.B[`${l}.ln2.b`])
      const pre = linear(ln2.out, this.W[`${l}.fc1.W`], this.B[`${l}.fc1.b`])
      const act = map(pre, gelu)
      const mlpOut = linear(act, this.W[`${l}.fc2.W`], this.B[`${l}.fc2.b`])
      const xOut = add(xMid, mlpOut)
      layers.push({ xIn, ln1, q, k, v, heads, concat, attnOut, xMid, ln2, pre, act, mlpOut, xOut })
      x = xOut
    }
    const lnf = layerNorm(x, this.B['lnf.g'], this.B['lnf.b'])
    const logits = linear(lnf.out, this.W['head.W'], this.B['head.b'])
    return { ids, tokens: ids.map((i) => this.vocab[i]), tokEmb, posEmb, x0, layers, lnf, logits }
  }

  /** Logit lens: read an intermediate residual vector through the final LN + unembedding. */
  logitLens(x: Mat, i: number): Float64Array {
    const r = mat(1, x.cols, row(x, i))
    const h = layerNorm(r, this.B['lnf.g'], this.B['lnf.b']).out
    return softmax(linear(h, this.W['head.W'], this.B['head.b']).data)
  }
}

let shared: NanoGPT | null = null
export const getModel = () => (shared ??= new NanoGPT())

/** Human-readable token: '▁the' → '·the', '<bos>' stays. */
export const showToken = (t: string) => (t.replace(/▁/g, '·'))
