/**
 * Pure, tested helpers for module 04: presets, configuration repair, the micro-step timeline of one
 * training iteration, and the node/edge chain of one chain-rule path. No React here.
 */
import { DIGITS, DIGITS_CFG } from '../../engine/digits'
import {
  WORKED, lossCompat, sub, sup, type Act, type Backward, type DatasetId, type InitScheme, type LossKind, type NetConfig, type OptConfig, type Params,
  type ParamRef, type Path, type Trace, defaultOpt,
} from '../../engine/nn'

/* ------------------------------------------------------------------ presets */
export type PresetId = 'worked' | 'neuron' | 'xor' | 'deep' | 'multi' | 'digits'
export interface Preset { id: PresetId; label: string; blurb: string; cfg: NetConfig; dataset: DatasetId; init: { scheme: InitScheme; seed: number; scale: number }; params?: Params; opt: OptConfig; batch: number }
export const PRESETS: Preset[] = [
  { id: 'worked', label: '2 → 2 → 1 · worked example', blurb: 'The hand-checkable example of the lesson: fixed weights, one sample x = (1.0, 0.8), target 1.', cfg: WORKED.cfg, dataset: 'worked', init: { scheme: 'xavier', seed: 1, scale: 1 }, params: WORKED.params, opt: { ...defaultOpt, lr: 0.5 }, batch: 1 },
  { id: 'neuron', label: '2 → 1 · single neuron', blurb: 'Logistic regression: one sigmoid neuron learning AND.', cfg: { sizes: [2, 1], acts: ['sigmoid'], loss: 'bce' }, dataset: 'and', init: { scheme: 'xavier', seed: 3, scale: 1 }, opt: { ...defaultOpt, lr: 1 }, batch: 4 },
  { id: 'xor', label: '2 → 3 → 2 → 1 · XOR', blurb: 'Two tanh hidden layers, sigmoid output, binary cross-entropy on XOR.', cfg: { sizes: [2, 3, 2, 1], acts: ['tanh', 'tanh', 'sigmoid'], loss: 'bce' }, dataset: 'xor', init: { scheme: 'xavier', seed: 4, scale: 1 }, opt: { ...defaultOpt, lr: 0.5 }, batch: 4 },
  { id: 'deep', label: '2 → 8 → 8 → 1 · two moons', blurb: 'A small deep ReLU network that bends a decision boundary around two moons.', cfg: { sizes: [2, 8, 8, 1], acts: ['relu', 'relu', 'sigmoid'], loss: 'bce' }, dataset: 'moons', init: { scheme: 'he', seed: 2, scale: 1 }, opt: { ...defaultOpt, kind: 'adam', lr: 0.03 }, batch: 16 },
  { id: 'multi', label: '3 → 4 → 4 → 2 · softmax', blurb: 'ReLU hidden layers, a softmax output and categorical cross-entropy (random inputs).', cfg: { sizes: [3, 4, 4, 2], acts: ['relu', 'relu', 'softmax'], loss: 'cce' }, dataset: 'random', init: { scheme: 'he', seed: 5, scale: 1 }, opt: { ...defaultOpt, lr: 0.3 }, batch: 8 },
  { id: 'digits', label: '64 → 16 → 16 → 10 · trained digits', blurb: 'A real trained classifier for 8×8 handwritten digits (97.3 % on 450 held-out images). Inputs are drawn as an 8×8 grid.', cfg: DIGITS_CFG, dataset: 'digits', init: { scheme: 'he', seed: 6, scale: 1 }, params: { W: DIGITS.W, b: DIGITS.b }, opt: { ...defaultOpt, kind: 'adam', lr: 0.01 }, batch: 32 },
]
export const presetById = (id: PresetId) => PRESETS.find((p) => p.id === id)!

/* ------------------------------------------------------------------ configuration repair */
export const MAX_NEURONS = 64, MAX_HIDDEN_NEURONS = 16, MAX_LAYERS = 6
/** Make a configuration valid after an edit, explaining every automatic change. */
export function repairConfig(cfg: NetConfig, prefer?: { loss?: LossKind }): { cfg: NetConfig; notes: string[] } {
  const notes: string[] = []
  const sizes = cfg.sizes.map((n, l) => Math.max(1, Math.min(l === 0 || l === cfg.sizes.length - 1 ? MAX_NEURONS : MAX_HIDDEN_NEURONS, Math.round(n))))
  const acts = cfg.acts.slice(0, sizes.length - 1)
  while (acts.length < sizes.length - 1) acts.splice(acts.length - 1, 0, 'tanh')
  const L = acts.length, K = sizes[L]
  for (let l = 0; l < L - 1; l++) if (acts[l] === 'softmax') { acts[l] = 'tanh'; notes.push(`Softmax is only offered on the output layer; layer ${l + 1} switched to tanh.`) }
  let loss = prefer?.loss ?? cfg.loss
  if (loss === 'cce') {
    if (K < 2) { loss = 'bce'; notes.push('Categorical cross-entropy needs at least 2 outputs; switched to binary cross-entropy.') }
    else if (acts[L - 1] !== 'softmax') { acts[L - 1] = 'softmax'; notes.push('Categorical cross-entropy uses a softmax output; output activation set to softmax.') }
  }
  if (acts[L - 1] === 'softmax' && K < 2) { acts[L - 1] = 'sigmoid'; notes.push('Softmax needs at least 2 outputs; output activation set to sigmoid.') }
  if (loss === 'bce' && acts[L - 1] !== 'sigmoid') {
    if (prefer?.loss === 'bce') { acts[L - 1] = 'sigmoid'; notes.push('Binary cross-entropy needs outputs in (0, 1); output activation set to sigmoid.') }
    else { loss = acts[L - 1] === 'softmax' ? 'cce' : 'mse'; notes.push(`Binary cross-entropy needs a sigmoid output; loss switched to ${loss === 'cce' ? 'categorical cross-entropy' : 'mean squared error'}.`) }
  }
  if (loss === 'mse' && acts[L - 1] === 'softmax') { loss = 'cce'; notes.push('A softmax output is paired with categorical cross-entropy here; loss switched.') }
  const out: NetConfig = { sizes, acts, loss }
  if (!lossCompat(out, loss).ok) throw new Error('repairConfig produced an invalid configuration')
  return { cfg: out, notes }
}

/* ------------------------------------------------------------------ micro-steps of one iteration */
export type MicroStep =
  | { kind: 'input' }
  | { kind: 'z'; layer: number }
  | { kind: 'act'; layer: number }
  | { kind: 'loss' }
  | { kind: 'dout' }
  | { kind: 'grad'; layer: number }
  | { kind: 'delta'; layer: number }
  | { kind: 'update' }
export type Phase = 'forward' | 'loss' | 'backward' | 'update'
export type TimelineMode = 'forward' | 'backprop' | 'iteration'
export const MODE_LABEL: Record<TimelineMode, string> = { forward: 'Forward only', backprop: 'Forward + backward', iteration: 'Full iteration' }

export function microSteps(L: number, mode: TimelineMode = 'iteration'): MicroStep[] {
  const s: MicroStep[] = [{ kind: 'input' }]
  for (let l = 1; l <= L; l++) s.push({ kind: 'z', layer: l }, { kind: 'act', layer: l })
  s.push({ kind: 'loss' })
  if (mode === 'forward') return s
  s.push({ kind: 'dout' }, { kind: 'grad', layer: L })
  for (let l = L - 1; l >= 1; l--) s.push({ kind: 'delta', layer: l }, { kind: 'grad', layer: l })
  if (mode === 'iteration') s.push({ kind: 'update' })
  return s
}
export const phaseOf = (s: MicroStep): Phase => (s.kind === 'input' || s.kind === 'z' || s.kind === 'act' ? 'forward' : s.kind === 'loss' ? 'loss' : s.kind === 'update' ? 'update' : 'backward')
export function stepLabel(s: MicroStep, L: number): { short: string; title: string; heading: string } {
  switch (s.kind) {
    case 'input': return { short: 'x', title: 'Load the input a⁽⁰⁾ = x', heading: 'The input enters the network' }
    case 'z': return { short: `z${sup(s.layer)}`, title: `Weighted sums of layer ${s.layer}: z⁽${sup(s.layer)}⁾ = W⁽${sup(s.layer)}⁾a⁽${sup(s.layer - 1)}⁾ + b⁽${sup(s.layer)}⁾`, heading: `Weighted sums of ${s.layer === L ? 'the output layer' : `layer ${s.layer}`}` }
    case 'act': return { short: `a${sup(s.layer)}`, title: `Activations of ${s.layer === L ? 'the output layer' : `layer ${s.layer}`}: a⁽${sup(s.layer)}⁾ = f(z⁽${sup(s.layer)}⁾)`, heading: s.layer === L ? 'The prediction' : `Activations of layer ${s.layer}` }
    case 'loss': return { short: 'L', title: 'Compare the prediction with the target: the loss', heading: 'How wrong is the prediction?' }
    case 'dout': return { short: `δ${sup(L)}`, title: `Output error δ⁽${sup(L)}⁾ = ∂L/∂z⁽${sup(L)}⁾`, heading: 'The error signal at the output' }
    case 'grad': return { short: `∇W${sup(s.layer)}`, title: `Gradients of layer ${s.layer}: ∂L/∂W⁽${sup(s.layer)}⁾ = δ⁽${sup(s.layer)}⁾ a⁽${sup(s.layer - 1)}⁾ᵀ`, heading: `Gradients for the weights of layer ${s.layer}` }
    case 'delta': return { short: `δ${sup(s.layer)}`, title: `Send the error back to layer ${s.layer}: δ⁽${sup(s.layer)}⁾ = (W⁽${sup(s.layer + 1)}⁾ᵀδ⁽${sup(s.layer + 1)}⁾) ⊙ f′(z⁽${sup(s.layer)}⁾)`, heading: `Error flows back into layer ${s.layer}` }
    case 'update': return { short: 'θ←', title: 'Gradient descent: every parameter moves against its gradient', heading: 'One step of gradient descent' }
  }
}
export const PHASE_LABEL: Record<Phase, string> = { forward: 'forward pass', loss: 'loss', backward: 'backward pass', update: 'update' }

/** What is known at a micro-step: the visualization only reveals values that have been computed so far. */
export interface Reveal { zUpTo: number; aUpTo: number; loss: boolean; deltaFrom: number; gradFrom: number; update: boolean }
export function revealAt(steps: MicroStep[], index: number, L: number): Reveal {
  const r: Reveal = { zUpTo: 0, aUpTo: 0, loss: false, deltaFrom: L + 1, gradFrom: L + 1, update: false }
  for (const s of steps.slice(0, index + 1)) {
    if (s.kind === 'z') r.zUpTo = s.layer
    if (s.kind === 'act') r.aUpTo = s.layer
    if (s.kind === 'loss') r.loss = true
    if (s.kind === 'dout') r.deltaFrom = L
    if (s.kind === 'delta') r.deltaFrom = s.layer
    if (s.kind === 'grad') r.gradFrom = s.layer
    if (s.kind === 'update') r.update = true
  }
  return r
}

/* ------------------------------------------------------------------ selection */
export type Selection = { kind: 'neuron'; layer: number; j: number } | ParamRef
export const selKey = (s: Selection | null) => (!s ? '' : s.kind === 'neuron' ? `n${s.layer}.${s.j}` : s.kind === 'w' ? `w${s.layer}.${s.j}.${s.i}` : `b${s.layer}.${s.j}`)
export function selectionValid(s: Selection | null, sizes: number[]): boolean {
  if (!s) return true
  if (s.layer < (s.kind === 'neuron' ? 0 : 1) || s.layer >= sizes.length || s.j >= sizes[s.layer]) return false
  return s.kind !== 'w' || s.i < sizes[s.layer - 1]
}
export const neuronName = (layer: number, j: number, L: number) => (layer === 0 ? `x${sub(j + 1)}` : layer === L ? `ŷ${L > 0 && j >= 0 ? sub(j + 1) : ''}` : `h${sub(j + 1)}⁽${sup(layer)}⁾`)

/* ------------------------------------------------------------------ one chain-rule path as nodes and edges */
export interface ChainNode { id: string; kind: 'w' | 'b' | 'z' | 'a' | 'L'; layer: number; j: number; label: string; tex: string; value: number; grad: number }
export interface ChainEdge { label: string; tex: string; value: number; kind: Path['factors'][number]['kind']; /** product of this edge and every edge after it (contribution of this path, read from the loss end) */ tail: number }
/**
 * Turn one path from `chainPaths` into a node/edge chain with forward values and true total gradients.
 * Node grads are totals (all paths), edge tails are this path's running product from the loss backwards.
 */
export function pathChain(cfg: NetConfig, p: Params, t: Trace, b: Backward, loss: number, start: Selection, path: Path): { nodes: ChainNode[]; edges: ChainEdge[] } {
  const L = cfg.sizes.length - 1
  const nodes: ChainNode[] = []
  const zNode = (layer: number, j: number): ChainNode => ({ id: `z${layer}.${j}`, kind: 'z', layer, j, label: `z${sub(j + 1)}⁽${sup(layer)}⁾`, tex: `z_{${j + 1}}^{(${layer})}`, value: t.z[layer][j], grad: b.delta[layer][j] })
  const aNode = (layer: number, j: number): ChainNode => ({ id: `a${layer}.${j}`, kind: 'a', layer, j, label: layer === L ? `ŷ${sub(j + 1)}` : `a${sub(j + 1)}⁽${sup(layer)}⁾`, tex: layer === L ? `\\hat y_{${j + 1}}` : `a_{${j + 1}}^{(${layer})}`, value: t.a[layer][j], grad: b.dA[layer][j] })
  if (start.kind === 'w') nodes.push({ id: 'p', kind: 'w', layer: start.layer, j: start.j, label: `w${sub(start.j + 1)}${sub(start.i + 1)}⁽${sup(start.layer)}⁾`, tex: `w_{${start.j + 1}${start.i + 1}}^{(${start.layer})}`, value: p.W[start.layer - 1][start.j][start.i], grad: b.dW[start.layer - 1][start.j][start.i] })
  else if (start.kind === 'b') nodes.push({ id: 'p', kind: 'b', layer: start.layer, j: start.j, label: `b${sub(start.j + 1)}⁽${sup(start.layer)}⁾`, tex: `b_{${start.j + 1}}^{(${start.layer})}`, value: p.b[start.layer - 1][start.j], grad: b.db[start.layer - 1][start.j] })
  else nodes.push(aNode(start.layer, start.j))
  let ni = 1 // index into path.nodes for the next neuron reached through a weight / softmax
  let cur = { layer: start.layer, j: start.j }
  for (const f of path.factors) {
    if (f.kind === 'local-input' || f.kind === 'one') nodes.push(zNode(cur.layer, cur.j))
    else if (f.kind === 'activation') nodes.push(aNode(cur.layer, cur.j))
    else if (f.kind === 'weight') { cur = path.nodes[ni++]; nodes.push(zNode(cur.layer, cur.j)) }
    else if (f.kind === 'softmax') { cur = path.nodes[ni++]; nodes.push(aNode(cur.layer, cur.j)) }
    else nodes.push({ id: 'L', kind: 'L', layer: L + 1, j: 0, label: 'L', tex: 'L', value: loss, grad: 1 })
  }
  const edges: ChainEdge[] = path.factors.map((f) => ({ label: f.label, tex: factorTex(f.label), value: f.value, kind: f.kind, tail: 0 }))
  let acc = 1
  for (let k = edges.length - 1; k >= 0; k--) { acc *= edges[k].value; edges[k].tail = acc }
  return { nodes, edges }
}
/** Factor labels are written with Unicode for chips; this gives a KaTeX version. */
export function factorTex(label: string): string {
  const unsub = (s: string) => s.replace(/[₀-₉]/g, (c) => String('₀₁₂₃₄₅₆₇₈₉'.indexOf(c))).replace(/[⁰-⁹¹²³]/g, (c) => String('⁰¹²³⁴⁵⁶⁷⁸⁹'.indexOf(c)))
  let s = unsub(label)
  s = s.replace(/⁽(\d+)⁾/g, '^{($1)}')
  s = s.replace(/([wzab])(\d)(\d)?(?=\^)/g, (_, v, j, i) => `${v}_{${j}${i ?? ''}}`)
  s = s.replace(/ŷ(\d)/g, '\\hat y_{$1}').replace(/∂/g, '\\partial ').replace(/σ′/g, "\\sigma'").replace(/tanh′/g, "\\tanh'").replace(/LReLU′/g, "\\mathrm{LReLU}'").replace(/ReLU′/g, "\\mathrm{ReLU}'").replace(/id′/g, "\\mathrm{id}'")
  s = s.replace(/ \(softmax\)/, '\\;\\text{(softmax)}')
  return s
}

export const ACT_CHOICES: Act[] = ['sigmoid', 'tanh', 'relu', 'leaky', 'linear', 'softmax']
