/**
 * Byte-pair encoding — a line-for-line port of training/bpe.py so that the
 * browser tokenises text exactly as the model saw it during training.
 */
export const SPACE = '▁'
const PRE = / ?[a-z]+| ?[^a-z ]/g

export function pretokenize(text: string): string[] {
  return (text.toLowerCase().match(PRE) ?? []).map((p) => p.replace(/ /g, SPACE))
}

export type Pair = [string, string]
const key = (a: string, b: string) => `${a}\u0000${b}`

export interface MergeEvent {
  rank: number
  pair: Pair
  /** symbols of the word after this merge */
  after: string[]
}
export interface WordTrace {
  word: string
  start: string[]
  events: MergeEvent[]
  final: string[]
}

export class BPETokenizer {
  readonly ranks = new Map<string, number>()
  readonly stoi = new Map<string, number>()
  readonly vocab: string[]
  readonly merges: Pair[]
  constructor(vocab: string[], merges: Pair[]) {
    this.vocab = vocab
    this.merges = merges
    merges.forEach(([a, b], i) => this.ranks.set(key(a, b), i))
    vocab.forEach((t, i) => this.stoi.set(t, i))
  }

  encodeWord(word: string): WordTrace {
    let s = [...word]
    const start = [...s]
    const events: MergeEvent[] = []
    while (s.length > 1) {
      let best = -1, bi = -1
      for (let i = 0; i < s.length - 1; i++) {
        const r = this.ranks.get(key(s[i], s[i + 1]))
        if (r !== undefined && (best < 0 || r < best)) { best = r; bi = i }
      }
      if (best < 0) break
      const pair: Pair = [s[bi], s[bi + 1]]
      s = [...s.slice(0, bi), s[bi] + s[bi + 1], ...s.slice(bi + 2)]
      events.push({ rank: best, pair, after: [...s] })
    }
    return { word, start, events, final: s }
  }

  /** Returns token ids plus a per-word trace of merges. Unknown characters are dropped and reported. */
  encode(text: string) {
    const words = pretokenize(text)
    const traces = words.map((w) => this.encodeWord(w))
    const ids: number[] = []
    const unknown: string[] = []
    for (const t of traces)
      for (const sym of t.final) {
        const id = this.stoi.get(sym)
        if (id === undefined) unknown.push(sym)
        else ids.push(id)
      }
    return { ids, traces, unknown }
  }
}

/** Train BPE from scratch (deterministic tie-break: lexicographically smallest pair). */
export function trainBPE(text: string, nMerges: number) {
  const counts = new Map<string, number>()
  for (const w of pretokenize(text)) counts.set(w, (counts.get(w) ?? 0) + 1)
  const seqs = new Map<string, string[]>([...counts.keys()].map((w) => [w, [...w]]))
  const chars = [...new Set([...counts.keys()].flatMap((w) => [...w]))].sort()
  const merges: { pair: Pair; count: number }[] = []
  for (let m = 0; m < nMerges; m++) {
    const pairs = new Map<string, number>()
    for (const [w, f] of counts) {
      const s = seqs.get(w)!
      for (let i = 0; i < s.length - 1; i++) {
        const k = key(s[i], s[i + 1])
        pairs.set(k, (pairs.get(k) ?? 0) + f)
      }
    }
    if (!pairs.size) break
    let top = -1
    for (const c of pairs.values()) top = Math.max(top, c)
    const best = [...pairs].filter(([, c]) => c === top).map(([k]) => k).sort((x, y) => cmpPair(x, y))[0]
    const [a, b] = best.split('\u0000') as Pair
    merges.push({ pair: [a, b], count: top })
    for (const [w, s] of seqs) {
      const out: string[] = []
      for (let i = 0; i < s.length; i++) {
        if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { out.push(a + b); i++ } else out.push(s[i])
      }
      seqs.set(w, out)
    }
  }
  return { chars, merges }
}

/** Compare like Python tuple-of-strings comparison (code-point order). */
function cmpPair(x: string, y: string) {
  const [xa, xb] = x.split('\u0000'), [ya, yb] = y.split('\u0000')
  return cmpStr(xa, ya) || cmpStr(xb, yb)
}
const cmpStr = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)
