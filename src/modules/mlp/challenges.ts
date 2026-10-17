/**
 * Seeded challenge questions for module 04. Every numeric answer is computed by the engine; the
 * explanations show the arithmetic. Same seed → same questions.
 */
import { rng } from '../../engine/decoding'
import { actDeriv, applyUpdate, backward, defaultOpt, forward, sigmoid, zeroOptState, type NetConfig, type Params } from '../../engine/nn'
import { n4 } from './narrate'

export interface Question { id: string; prompt: string; context?: string; answer: number | string; options?: string[]; tol?: number; explain: string }

/** “a·b + c·d − 0.4” with proper signs */
const signed = (terms: [number, number?][]) => terms.map(([a, b], k) => `${k === 0 ? (a < 0 ? '−' : '') : a < 0 ? ' − ' : ' + '}${Math.abs(a)}${b === undefined ? '' : `·${b}`}`).join('')
const r1 = (r: () => number, lo: number, hi: number) => Math.round((lo + r() * (hi - lo)) * 10) / 10
export const isCorrect = (q: Question, given: string) => {
  if (q.options) return given === q.answer
  const v = parseFloat(given.replace(',', '.').replace('−', '-'))
  if (!Number.isFinite(v)) return false
  const a = q.answer as number
  return Math.abs(v - a) <= (q.tol ?? Math.max(0.005, 0.01 * Math.abs(a)))
}

const BANK: Omit<Question, 'id'>[] = [
  { prompt: 'What is the largest value the sigmoid derivative σ′(z) can take?', options: ['0.25', '0.5', '1', '0.1'], answer: '0.25', explain: `σ′(z) = σ(z)(1 − σ(z)) peaks at z = 0 where σ = 0.5: 0.5 × 0.5 = ${actDeriv('sigmoid', 0)}.` },
  { prompt: 'With a softmax output and categorical cross-entropy, the output error δ = ∂L/∂z equals…', options: ['ŷ − y', '(ŷ − y)·ŷ(1 − ŷ)', '−y / ŷ', 'y − ŷ'], answer: 'ŷ − y', explain: 'The softmax Jacobian and the −y/ŷ of the loss cancel exactly, leaving ŷ − y (tested against the generic chain rule).' },
  { prompt: 'What is the best accuracy a single sigmoid neuron (no hidden layer) can reach on XOR?', options: ['75 %', '50 %', '100 %', '25 %'], answer: '75 %', explain: 'One neuron draws one straight line; no line separates XOR, but a line can get 3 of the 4 points right.' },
  { prompt: 'A ReLU unit has z = −2. What fraction of the gradient arriving at its output does it pass back to z?', options: ['0', '0.01', '0.5', '1'], answer: '0', explain: 'ReLU′(z) = 0 for z ≤ 0, so δ = (∂L/∂a)·0 = 0: nothing flows back through this unit for this sample.' },
  { prompt: 'In a deep network with sigmoid hidden layers, which layer usually receives the smallest gradients?', options: ['the first (input side)', 'the last (output side)', 'the middle', 'all are equal'], answer: 'the first (input side)', explain: 'Every layer between a weight and the loss multiplies the error by σ′ ≤ 0.25 (times the weights), so the layers furthest from the loss get the smallest gradients.' },
  { prompt: 'A central finite difference with ε = 10⁻¹² disagrees badly with backprop. Why?', options: ['floating-point round-off', 'the curvature of the loss', 'the learning rate is too high', 'the biases are zero'], answer: 'floating-point round-off', explain: 'L(θ+ε) and L(θ−ε) agree in almost all their ~16 significant digits; their difference is mostly rounding noise, then divided by a tiny 2ε.' },
]

/** a 2 → 2 → 1 sigmoid network with one-decimal weights, MSE, target 1 */
export function challengeNet(seed: number): { cfg: NetConfig; p: Params; x: number[]; y: number[] } {
  const r = rng(seed * 104729 + 11)
  return {
    cfg: { sizes: [2, 2, 1], acts: ['sigmoid', 'sigmoid'], loss: 'mse' },
    p: { W: [[[r1(r, -1, 1), r1(r, -1, 1)], [r1(r, -1, 1), r1(r, -1, 1)]], [[r1(r, -1, 1), r1(r, -1, 1)]]], b: [[r1(r, -0.5, 0.5), r1(r, -0.5, 0.5)], [r1(r, -0.5, 0.5)]] },
    x: [r1(r, 0, 1), r1(r, 0, 1)], y: [1],
  }
}

export function makeChallenges(seed: number): Question[] {
  const r = rng(seed * 7919 + 5)
  const { cfg, p, x, y } = challengeNet(seed)
  const t = forward(cfg, p, x), g = backward(cfg, p, t, y)
  const w = p.W[0][0], b = p.b[0][0]
  const zr = r1(r, -2, 2), zs = r1(r, -3, 3)
  const yh = t.a[2][0], d2 = g.delta[2][0]
  const eta = [0.1, 0.5, 1][Math.floor(r() * 3)]
  const u = applyUpdate(p, g, zeroOptState(p), { ...defaultOpt, lr: eta })
  const rec = u.records.find((q) => q.ref.kind === 'w' && q.ref.layer === 2 && q.ref.j === 0 && q.ref.i === 0)!
  const nb = 2 + Math.floor(r() * 4), na = 2 + Math.floor(r() * 4)
  const picks = BANK.map((q, k) => ({ q, k, s: r() })).sort((a, b2) => a.s - b2.s).slice(0, 2)
  const ctx = `Network 2 → 2 → 1, sigmoid everywhere, MSE, x = (${x.join(', ')}), y = 1. Hidden neuron 1: w = (${w.join(', ')}), b = ${b}.`
  return [
    { id: 'z', prompt: 'Compute the weighted sum z of hidden neuron 1.', context: ctx, answer: t.z[1][0], explain: `z = ${signed([[w[0], x[0]], [w[1], x[1]], [b]])} = ${n4(t.z[1][0])}` },
    { id: 'sig', prompt: `Compute σ(z) for z = ${zs}.`, answer: sigmoid(zs), explain: `σ(${zs}) = 1 / (1 + e^(${-zs})) = ${n4(sigmoid(zs))}` },
    { id: 'relu', prompt: `A ReLU unit receives z = ${zr}. What is its activation?`, answer: Math.max(0, zr), tol: 1e-6, explain: `ReLU(${zr}) = max(0, ${zr}) = ${Math.max(0, zr)}` },
    { id: 'dout', prompt: `The output is ŷ = ${n4(yh)} with target y = 1 (sigmoid output, MSE L = (ŷ − y)²). Compute δ = ∂L/∂z at the output.`, answer: d2, explain: `δ = 2(ŷ − y)·ŷ(1 − ŷ) = 2(${n4(yh)} − 1)·${n4(yh)}·${n4(1 - yh)} = ${n4(d2)}` },
    { id: 'dw', prompt: `Using δ = ${n4(d2)} and the hidden activation a₁ = ${n4(t.a[1][0])}, compute ∂L/∂w₁₁⁽²⁾.`, answer: g.dW[1][0][0], explain: `∂L/∂w = δ·a = ${n4(d2)} × ${n4(t.a[1][0])} = ${n4(g.dW[1][0][0])}` },
    { id: 'sgd', prompt: `w₁₁⁽²⁾ = ${p.W[1][0][0]} and its gradient is ${n4(rec.grad)}. What is the weight after one SGD step with η = ${eta}?`, answer: rec.next, explain: `w ← w − η·g = ${p.W[1][0][0]} − ${eta}·(${n4(rec.grad)}) = ${n4(rec.next)}` },
    { id: 'paths', prompt: `In a 3 → ${na} → ${nb} → 1 network, through how many different paths does the weight w₁₁⁽¹⁾ influence the loss?`, answer: nb, tol: 0, explain: `w₁₁⁽¹⁾ feeds hidden neuron 1 of layer 1, which connects to all ${nb} neurons of layer 2, each connected to the single output: ${nb} × 1 = ${nb} paths. Its gradient is the sum of ${nb} chain-rule products.` },
    ...picks.map(({ q, k }) => ({ ...q, id: `c${k}` })),
  ]
}
