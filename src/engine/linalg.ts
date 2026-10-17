/**
 * Small, deterministic linear algebra for the embedding lab.
 * Vectors are plain number arrays (Float64 precision) so every value shown can be reproduced exactly.
 */
export type Vec = number[]

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) throw new Error(`dot: length ${a.length} ≠ ${b.length}`)
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}
export const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a))
/** Cosine similarity; 0 when either vector is the zero vector (by convention, flagged in the UI). */
export const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const na = norm(a), nb = norm(b)
  return na === 0 || nb === 0 ? 0 : dot(a, b) / (na * nb)
}
export const euclid = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
  return Math.sqrt(s)
}
export const add = (a: Vec, b: Vec) => a.map((x, i) => x + b[i])
export const sub = (a: Vec, b: Vec) => a.map((x, i) => x - b[i])
export const scale = (a: Vec, s: number) => a.map((x) => x * s)
export const unit = (a: Vec) => { const n = norm(a); return n === 0 ? a.slice() : scale(a, 1 / n) }
/** Angle between vectors in degrees. */
export const angleDeg = (a: Vec, b: Vec) => (Math.acos(Math.max(-1, Math.min(1, cosine(a, b)))) * 180) / Math.PI

/**
 * Cyclic Jacobi eigenvalue algorithm for a real symmetric matrix.
 * Returns eigenvalues sorted descending and eigenvectors as COLUMNS of `vectors` (vectors[i][k] = component i of eigenvector k).
 * Each eigenvector's sign is fixed so its largest-magnitude component is positive (deterministic output).
 */
export function jacobiEigen(A: number[][], maxSweeps = 100, tol = 1e-12) {
  const n = A.length
  const a = A.map((r) => r.slice())
  for (let i = 0; i < n; i++) if (a[i].length !== n) throw new Error('jacobiEigen: matrix must be square')
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) if (Math.abs(a[i][j] - a[j][i]) > 1e-9 * (1 + Math.abs(a[i][j]))) throw new Error('jacobiEigen: matrix must be symmetric')
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  let sweeps = 0
  for (; sweeps < maxSweeps; sweeps++) {
    let off = 0, total = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { total += a[i][j] ** 2; if (i !== j) off += a[i][j] ** 2 }
    if (off <= tol * tol * Math.max(total, 1e-300)) break
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q]
        if (Math.abs(apq) < 1e-300) continue
        const theta = (a[q][q] - a[p][p]) / (2 * apq)
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1), s = t * c
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q]
          a[k][p] = c * akp - s * akq
          a[k][q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k]
          a[p][k] = c * apk - s * aqk
          a[q][k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[y][y] - a[x][x] || x - y)
  const values = order.map((i) => a[i][i])
  const vectors = Array.from({ length: n }, (_, r) => order.map((k) => v[r][k]))
  for (let k = 0; k < n; k++) {
    let big = 0
    for (let r = 0; r < n; r++) if (Math.abs(vectors[r][k]) > Math.abs(vectors[big][k]) + 1e-12) big = r
    if (vectors[big][k] < 0) for (let r = 0; r < n; r++) vectors[r][k] = -vectors[r][k]
  }
  return { values, vectors, sweeps }
}

export interface PCAResult {
  mean: Vec
  /** principal axes, each a unit vector of length d */
  components: Vec[]
  /** variance along each component */
  variance: number[]
  /** share of total variance along each component */
  explained: number[]
  totalVariance: number
  /** projected coordinates, one row per input row */
  coords: Vec[]
}

/** Exact PCA via eigen-decomposition of the covariance matrix (population covariance, divide by n). */
export function pca(rows: Vec[], k: number): PCAResult {
  const n = rows.length
  if (!n) throw new Error('pca: no rows')
  const d = rows[0].length
  const mean = new Array(d).fill(0)
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / n
  const X = rows.map((r) => r.map((x, j) => x - mean[j]))
  const C = Array.from({ length: d }, () => new Array(d).fill(0))
  for (const r of X) for (let a = 0; a < d; a++) { const ra = r[a]; if (ra === 0) continue; for (let b = a; b < d; b++) C[a][b] += (ra * r[b]) / n }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a]
  const { values, vectors } = eigh(C)
  const totalVariance = values.reduce((s, x) => s + Math.max(0, x), 0)
  const kk = Math.min(k, d)
  const components = Array.from({ length: kk }, (_, c) => vectors.map((row) => row[c]))
  const variance = values.slice(0, kk).map((x) => Math.max(0, x))
  return {
    mean, components, variance, totalVariance,
    explained: variance.map((x) => (totalVariance ? x / totalVariance : 0)),
    coords: X.map((r) => components.map((comp) => dot(r, comp))),
  }
}

/** Ranks with ties averaged (1-based). */
export function ranks(x: number[]): number[] {
  const idx = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
  const r = new Array(x.length)
  for (let i = 0; i < idx.length; ) {
    let j = i
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++
    const avg = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) r[idx[k][1]] = avg
    i = j + 1
  }
  return r
}
export function pearson(x: number[], y: number[]) {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2 }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0
}
export const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y))

/** Indices of the k nearest rows to row i under a distance function (ties broken by index). */
export function knnIdx(dist: (a: number, b: number) => number, n: number, i: number, k: number): number[] {
  return Array.from({ length: n }, (_, j) => j).filter((j) => j !== i)
    .map((j) => [dist(i, j), j] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map((x) => x[1])
}

/**
 * Neighbourhood preservation: for each point, the share of its k nearest neighbours in the
 * high-dimensional space (cosine distance) that are also among its k nearest in the projection
 * (Euclidean). 1 = every neighbourhood kept, ≈ k/(n−1) = random.
 */
export function neighbourPreservation(high: Vec[], low: Vec[], k: number) {
  const n = high.length
  const kk = Math.max(1, Math.min(k, n - 1))
  const dh = (a: number, b: number) => 1 - cosine(high[a], high[b])
  const dl = (a: number, b: number) => euclid(low[a], low[b])
  const per = high.map((_, i) => {
    const A = new Set(knnIdx(dh, n, i, kk))
    return knnIdx(dl, n, i, kk).filter((j) => A.has(j)).length / kk
  })
  return { mean: per.reduce((a, b) => a + b, 0) / n, per, k: kk, chance: kk / (n - 1) }
}

/**
 * Symmetric eigen-decomposition by Householder tridiagonalization + implicit QL iteration
 * (the EISPACK tred2/tql2 algorithm, as in JAMA). O(n³) once instead of per sweep, so it is used
 * for large matrices; output format and sign convention match `jacobiEigen`.
 */
export function symEigen(A: number[][]) {
  const n = A.length
  const V = A.map((r) => r.slice())
  const d = new Array(n).fill(0), e = new Array(n).fill(0)
  // ---- tred2
  for (let j = 0; j < n; j++) d[j] = V[n - 1][j]
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0
    for (let k = 0; k < i; k++) scale += Math.abs(d[k])
    if (scale === 0) {
      e[i] = d[i - 1]
      for (let j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0 }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k] }
      let f = d[i - 1]
      let g = Math.sqrt(h)
      if (f > 0) g = -g
      e[i] = scale * g; h -= f * g; d[i - 1] = f - g
      for (let j = 0; j < i; j++) e[j] = 0
      for (let j = 0; j < i; j++) {
        f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f
        for (let k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f }
        e[j] = g
      }
      f = 0
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j] }
      const hh = f / (h + h)
      for (let j = 0; j < i; j++) e[j] -= hh * d[j]
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j]
        for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k]
        d[j] = V[i - 1][j]; V[i][j] = 0
      }
    }
    d[i] = h
  }
  for (let i = 0; i < n - 1; i++) {
    V[n - 1][i] = V[i][i]; V[i][i] = 1
    const h = d[i + 1]
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h
      for (let j = 0; j <= i; j++) {
        let g = 0
        for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j]
        for (let k = 0; k <= i; k++) V[k][j] -= g * d[k]
      }
    }
    for (let k = 0; k <= i; k++) V[k][i + 1] = 0
  }
  for (let j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0 }
  if (n) V[n - 1][n - 1] = 1
  e[0] = 0
  // ---- tql2
  for (let i = 1; i < n; i++) e[i - 1] = e[i]
  if (n) e[n - 1] = 0
  let f = 0, tst1 = 0
  const eps = 2 ** -52
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]))
    let m = l
    while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++ }
    if (m > l) {
      let iter = 0
      do {
        if (++iter > 200) throw new Error('symEigen: no convergence')
        let g = d[l]
        let p = (d[l + 1] - g) / (2 * e[l])
        let r = Math.hypot(p, 1)
        if (p < 0) r = -r
        d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r)
        const dl1 = d[l + 1]
        let h = g - d[l]
        for (let i = l + 2; i < n; i++) d[i] -= h
        f += h
        p = d[m]
        let c = 1, c2 = c, c3 = c, s = 0, s2 = 0
        const el1 = e[l + 1]
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s
          g = c * e[i]; h = c * p; r = Math.hypot(p, e[i])
          e[i + 1] = s * r; s = e[i] / r; c = p / r; p = c * d[i] - s * g
          d[i + 1] = h + s * (c * g + s * d[i])
          for (let k = 0; k < n; k++) { h = V[k][i + 1]; V[k][i + 1] = s * V[k][i] + c * h; V[k][i] = c * V[k][i] - s * h }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1
        e[l] = s * p; d[l] = c * p
      } while (Math.abs(e[l]) > eps * tst1)
    }
    d[l] += f; e[l] = 0
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => d[y] - d[x] || x - y)
  const values = order.map((i) => d[i])
  const vectors = Array.from({ length: n }, (_, r) => order.map((k) => V[r][k]))
  for (let k = 0; k < n; k++) {
    let big = 0
    for (let r = 0; r < n; r++) if (Math.abs(vectors[r][k]) > Math.abs(vectors[big][k]) + 1e-12) big = r
    if (vectors[big][k] < 0) for (let r = 0; r < n; r++) vectors[r][k] = -vectors[r][k]
  }
  return { values, vectors }
}

/** Eigen-decomposition of a symmetric matrix: Jacobi for small n (clearest), tridiagonal QL for large n (fast). */
export const eigh = (A: number[][]) => (A.length <= 32 ? jacobiEigen(A) : symEigen(A))
