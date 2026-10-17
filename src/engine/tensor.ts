/**
 * Minimal, dependency-free tensor maths for the Observatory.
 * Matrices are row-major Float32Arrays with explicit shapes so that every
 * intermediate value the UI shows can be traced back to a real computation.
 */
export interface Mat {
  rows: number
  cols: number
  data: Float32Array
}

export const mat = (rows: number, cols: number, data?: ArrayLike<number>): Mat => {
  const d = new Float32Array(rows * cols)
  if (data) d.set(data)
  return { rows, cols, data: d }
}

export const get = (m: Mat, r: number, c: number) => m.data[r * m.cols + c]
export const row = (m: Mat, r: number): Float32Array => m.data.slice(r * m.cols, (r + 1) * m.cols)
export const shape = (m: Mat): [number, number] => [m.rows, m.cols]

export function assertShape(m: Mat, rows: number, cols: number, what = 'tensor') {
  if (m.rows !== rows || m.cols !== cols)
    throw new Error(`${what}: expected [${rows}, ${cols}] but got [${m.rows}, ${m.cols}]`)
}

/** y = x @ W + b, W stored as [in, out]. */
export function linear(x: Mat, W: Mat, b?: Float32Array): Mat {
  if (x.cols !== W.rows) throw new Error(`matmul shape mismatch [${x.rows},${x.cols}] @ [${W.rows},${W.cols}]`)
  const out = mat(x.rows, W.cols)
  for (let i = 0; i < x.rows; i++) {
    const o = i * W.cols
    if (b) for (let j = 0; j < W.cols; j++) out.data[o + j] = b[j]
    for (let k = 0; k < x.cols; k++) {
      const xv = x.data[i * x.cols + k]
      if (xv === 0) continue
      const wo = k * W.cols
      for (let j = 0; j < W.cols; j++) out.data[o + j] += xv * W.data[wo + j]
    }
  }
  return out
}

export function add(a: Mat, b: Mat): Mat {
  assertShape(b, a.rows, a.cols, 'add')
  const out = mat(a.rows, a.cols)
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i]
  return out
}

export function layerNorm(x: Mat, g: Float32Array, b: Float32Array, eps = 1e-5) {
  const out = mat(x.rows, x.cols)
  const means: number[] = []
  const stds: number[] = []
  for (let i = 0; i < x.rows; i++) {
    let mu = 0
    for (let j = 0; j < x.cols; j++) mu += x.data[i * x.cols + j]
    mu /= x.cols
    let v = 0
    for (let j = 0; j < x.cols; j++) v += (x.data[i * x.cols + j] - mu) ** 2
    v /= x.cols
    const sd = Math.sqrt(v + eps)
    means.push(mu)
    stds.push(sd)
    for (let j = 0; j < x.cols; j++) out.data[i * x.cols + j] = ((x.data[i * x.cols + j] - mu) / sd) * g[j] + b[j]
  }
  return { out, means, stds }
}

/** Numerically stable softmax of a vector (subtracts the max first). */
export function softmax(v: ArrayLike<number>, temperature = 1): Float64Array {
  const n = v.length
  const out = new Float64Array(n)
  let m = -Infinity
  for (let i = 0; i < n; i++) if (v[i] / temperature > m) m = v[i] / temperature
  let s = 0
  for (let i = 0; i < n; i++) {
    const e = v[i] === -Infinity ? 0 : Math.exp(v[i] / temperature - m)
    out[i] = e
    s += e
  }
  for (let i = 0; i < n; i++) out[i] /= s
  return out
}

/** GELU, tanh approximation (matches PyTorch approximate='tanh'). */
export const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))

export function map(m: Mat, f: (x: number) => number): Mat {
  const out = mat(m.rows, m.cols)
  for (let i = 0; i < m.data.length; i++) out.data[i] = f(m.data[i])
  return out
}

/** Take columns [c0, c0+n) — used to split Q/K/V into heads. */
export function sliceCols(m: Mat, c0: number, n: number): Mat {
  const out = mat(m.rows, n)
  for (let i = 0; i < m.rows; i++) for (let j = 0; j < n; j++) out.data[i * n + j] = m.data[i * m.cols + c0 + j]
  return out
}

export function concatCols(parts: Mat[]): Mat {
  const rows = parts[0].rows
  const cols = parts.reduce((s, p) => s + p.cols, 0)
  const out = mat(rows, cols)
  let off = 0
  for (const p of parts) {
    for (let i = 0; i < rows; i++) for (let j = 0; j < p.cols; j++) out.data[i * cols + off + j] = p.data[i * p.cols + j]
    off += p.cols
  }
  return out
}

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}
export const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a))

export function stats(a: ArrayLike<number>) {
  let mn = Infinity, mx = -Infinity, s = 0
  for (let i = 0; i < a.length; i++) {
    mn = Math.min(mn, a[i]); mx = Math.max(mx, a[i]); s += a[i]
  }
  const mean = s / a.length
  let v = 0
  for (let i = 0; i < a.length; i++) v += (a[i] - mean) ** 2
  return { min: mn, max: mx, mean, std: Math.sqrt(v / a.length) }
}

export function decodeB64Float32(b64: string): Float32Array {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Float32Array(bytes.buffer)
}

/** Top-k indices of a vector, descending. */
export function topk(v: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: v.length }, (_, i) => i)
    .sort((a, b) => v[b] - v[a])
    .slice(0, k)
}

/** Power-iteration PCA to 3 components (deterministic). */
export function pca3(m: Mat): { coords: Float32Array; explained: number[] } {
  const { rows: n, cols: d } = m
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += m.data[i * d + j] / n
  const X = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) X[i * d + j] = m.data[i * d + j] - mean[j]
  const C = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) {
      const xa = X[i * d + a]
      for (let b = 0; b < d; b++) C[a * d + b] += (xa * X[i * d + b]) / n
    }
  let total = 0
  for (let a = 0; a < d; a++) total += C[a * d + a]
  const comps: Float64Array[] = []
  const explained: number[] = []
  for (let c = 0; c < 3; c++) {
    let v = new Float64Array(d).map((_, i) => Math.sin(i * 12.9898 + c * 78.233) + 0.5)
    let lam = 0
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(d)
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) w[a] += C[a * d + b] * v[b]
      lam = Math.sqrt(w.reduce((s, x) => s + x * x, 0))
      v = w.map((x) => x / lam)
    }
    comps.push(v)
    explained.push(lam / total)
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a * d + b] -= lam * v[a] * v[b]
  }
  const coords = new Float32Array(n * 3)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) {
      let s = 0
      for (let j = 0; j < d; j++) s += X[i * d + j] * comps[c][j]
      coords[i * 3 + c] = s
    }
  return { coords, explained }
}
