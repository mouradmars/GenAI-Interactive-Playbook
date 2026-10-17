/**
 * Text and formulas for every micro-step, with the actual numbers substituted. Pure; every number comes
 * from the engine trace / backward pass handed in.
 */
import type { Depth } from '../../sim/appState'
import { LOSS_LABEL, type Act, type Backward, type NetConfig, type OptConfig, type Params, type Sample, type Trace, type UpdateRecord } from '../../engine/nn'
import type { MicroStep } from './model'

export const n4 = (v: number) => (Math.abs(v) < 5e-13 ? '0.0000' : v.toFixed(4))
/** a number for TeX, parenthesized when negative so products read correctly */
export const tn = (v: number, d = 4) => { const s = Math.abs(v) < 5e-13 ? (0).toFixed(d) : v.toFixed(d); return v < 0 && Math.abs(v) >= 5e-13 ? `(${s})` : s }
const MAX_TERMS = 4

export const actTex: Record<Act, string> = {
  sigmoid: '\\sigma(z) = \\frac{1}{1+e^{-z}}', tanh: '\\tanh(z)', relu: '\\mathrm{ReLU}(z)=\\max(0,z)', leaky: '\\mathrm{LReLU}(z)=\\max(0.01z,\\,z)', linear: 'f(z)=z', softmax: '\\mathrm{softmax}(z)_k=\\frac{e^{z_k}}{\\sum_r e^{z_r}}',
}
export const derivTex: Record<Act, string> = {
  sigmoid: "\\sigma'(z)=\\sigma(z)\\,(1-\\sigma(z))", tanh: "\\tanh'(z)=1-\\tanh^2(z)", relu: "\\mathrm{ReLU}'(z)=\\begin{cases}1 & z>0\\\\0 & z\\le 0\\end{cases}", leaky: "\\mathrm{LReLU}'(z)=\\begin{cases}1 & z>0\\\\0.01 & z\\le 0\\end{cases}", linear: "f'(z)=1", softmax: '\\frac{\\partial a_k}{\\partial z_r}=a_k(\\delta_{kr}-a_r)',
}
const fname = (a: Act) => (a === 'sigmoid' ? '\\sigma' : a === 'tanh' ? '\\tanh' : a === 'relu' ? '\\mathrm{ReLU}' : a === 'leaky' ? '\\mathrm{LReLU}' : a === 'linear' ? '\\mathrm{id}' : '\\mathrm{softmax}')
const fprimeName = (a: Act) => (a === 'sigmoid' ? "\\sigma'" : a === 'tanh' ? "\\tanh'" : a === 'relu' ? "\\mathrm{ReLU}'" : a === 'leaky' ? "\\mathrm{LReLU}'" : "\\mathrm{id}'")

export interface Narration { title: string; plain: string; general: string; numbers: string[]; note?: string }

export interface NarrateInput { cfg: NetConfig; params: Params; trace: Trace; bwd: Backward; sample: Sample; loss: number; opt: OptConfig; records: UpdateRecord[]; lossAfter: number; depth: Depth }

/** sum Σ w a + b written out with numbers (truncated for wide layers) */
function weightedSumTex(w: number[], a: number[], b: number, z: number) {
  const terms = w.map((wi, i) => `${tn(wi)}\\cdot${tn(a[i])}`)
  const shown = terms.length > MAX_TERMS ? [...terms.slice(0, MAX_TERMS - 1), `\\dots(${terms.length - MAX_TERMS + 1}\\text{ more})`] : terms
  return `${shown.join(' + ')} + ${tn(b)} = ${n4(z)}`
}

export function narrate(step: MicroStep, focus: number, x: NarrateInput): Narration {
  const { cfg, params, trace: t, bwd: g, sample, loss, depth } = x
  const L = cfg.sizes.length - 1, K = cfg.sizes[L]
  const beginner = depth === 'beginner'
  switch (step.kind) {
    case 'input': {
      const n = cfg.sizes[0]
      return {
        title: 'The input enters the network',
        plain: beginner ? `The network receives ${n} number${n > 1 ? 's' : ''}. Nothing has been computed yet; every other value is still unknown.` : `a⁽⁰⁾ = x is the input vector (${n} values). Every later quantity is a function of x and the parameters.`,
        general: 'a^{(0)} = x',
        numbers: [`x = (${sample.x.slice(0, 8).map((v) => n4(v)).join(',\\ ')}${n > 8 ? ',\\ \\dots' : ''})`, `y = (${sample.y.slice(0, 10).map((v) => n4(v)).join(',\\ ')})`],
      }
    }
    case 'z': {
      const l = step.layer, j = Math.min(focus, cfg.sizes[l] - 1)
      const w = params.W[l - 1][j], b = params.b[l - 1][j], prev = t.a[l - 1]
      return {
        title: `Weighted sums in layer ${l}`,
        plain: beginner ? `Each neuron multiplies every incoming value by its weight, adds them up and adds its bias. That number is z.` : `Each neuron j computes z_j = Σᵢ w_ji a_i + b_j: a dot product of its weight row with the previous activations, plus a bias.`,
        general: `z_{j}^{(${l})} = \\sum_i w_{ji}^{(${l})}\\,a_i^{(${l - 1})} + b_j^{(${l})}`,
        numbers: [`z_{${j + 1}}^{(${l})} = ${weightedSumTex(w, prev, b, t.z[l][j])}`],
        note: depth === 'advanced' ? `Vector form: z⁽${l}⁾ = W⁽${l}⁾a⁽${l - 1}⁾ + b⁽${l}⁾ with W⁽${l}⁾ of shape ${cfg.sizes[l]} × ${cfg.sizes[l - 1]}.` : undefined,
      }
    }
    case 'act': {
      const l = step.layer, j = Math.min(focus, cfg.sizes[l] - 1), f = cfg.acts[l - 1], z = t.z[l][j], a = t.a[l][j]
      const nums = f === 'softmax'
        ? [`\\hat y_{${j + 1}} = \\frac{e^{${tn(z)}}}{\\sum_r e^{z_r}} = ${n4(a)}`, `\\textstyle\\sum_k \\hat y_k = ${n4(t.a[l].reduce((s, v) => s + v, 0))}`]
        : [`a_{${j + 1}}^{(${l})} = ${fname(f)}(${tn(z)}) = ${n4(a)}`]
      return {
        title: `${l === L ? 'Output' : `Layer ${l}`} activations`,
        plain: beginner ? `Each neuron squashes or reshapes its z with the activation function. The result a is what it sends on.` : f === 'softmax' ? 'Softmax turns the output logits into a probability distribution: positive values that sum to 1.' : `An elementwise non-linearity a = f(z). Without it, stacked layers would collapse into one linear map.`,
        general: actTex[f], numbers: nums,
      }
    }
    case 'loss': {
      const yh = t.a[L], y = sample.y
      let num = ''
      if (cfg.loss === 'mse') num = K === 1 ? `L = (\\hat y - y)^2 = (${n4(yh[0])} - ${n4(y[0])})^2 = ${n4(loss)}` : `L = \\tfrac{1}{${K}}\\sum_k(\\hat y_k-y_k)^2 = ${n4(loss)}`
      else if (cfg.loss === 'bce') num = K === 1 ? `L = -[${n4(y[0])}\\log ${n4(yh[0])} + ${n4(1 - y[0])}\\log(1-${n4(yh[0])})] = ${n4(loss)}` : `L = -\\tfrac{1}{${K}}\\sum_k[y_k\\log\\hat y_k + (1-y_k)\\log(1-\\hat y_k)] = ${n4(loss)}`
      else { const c = y.indexOf(Math.max(...y)); num = `L = -\\log \\hat y_{${c + 1}} = -\\log ${n4(yh[c])} = ${n4(loss)}` }
      return {
        title: `Loss: ${LOSS_LABEL[cfg.loss].toLowerCase()}`,
        plain: beginner ? 'The loss is one number that says how wrong the prediction is. Zero would be perfect. Learning means making this number smaller.' : 'A scalar measure of error for this sample. Backpropagation computes how it changes with every parameter.',
        general: cfg.loss === 'mse' ? 'L = \\frac{1}{K}\\sum_k (\\hat y_k - y_k)^2' : cfg.loss === 'bce' ? 'L = -\\frac{1}{K}\\sum_k \\big[y_k \\log \\hat y_k + (1-y_k)\\log(1-\\hat y_k)\\big]' : 'L = -\\sum_k y_k \\log \\hat y_k',
        numbers: [num],
        note: cfg.loss !== 'mse' ? 'Computed from the logits (softplus / log-sum-exp), so it stays finite even when ŷ rounds to 0 or 1.' : undefined,
      }
    }
    case 'dout': {
      const j = Math.min(focus, K - 1), yh = t.a[L][j], y = sample.y[j], f = cfg.acts[L - 1]
      const nums: string[] = []
      let general = ''
      if (g.outputRule === 'sigmoid+bce') {
        general = `\\delta_k^{(${L})} = \\frac{\\partial L}{\\partial z_k^{(${L})}} = \\frac{\\hat y_k - y_k}{K}`
        nums.push(`\\delta_{${j + 1}}^{(${L})} = \\frac{${n4(yh)} - ${n4(y)}}{${K}} = ${n4(g.delta[L][j])}`)
        nums.push(`\\text{same as } \\tfrac{\\partial L}{\\partial \\hat y}\\,\\sigma'(z) = ${tn(g.dA[L][j])}\\cdot${tn(g.fprime[L][j])} = ${n4(g.dA[L][j] * g.fprime[L][j])}`)
      } else if (g.outputRule === 'softmax+cce') {
        general = `\\delta_k^{(${L})} = \\hat y_k - y_k`
        nums.push(`\\delta_{${j + 1}}^{(${L})} = ${n4(yh)} - ${n4(y)} = ${n4(g.delta[L][j])}`)
      } else {
        general = `\\delta_k^{(${L})} = \\frac{\\partial L}{\\partial \\hat y_k}\\, f'(z_k^{(${L})})`
        const dl = cfg.loss === 'mse' ? (K === 1 ? `2(${n4(yh)} - ${n4(y)})` : `\\tfrac{2}{${K}}(${n4(yh)} - ${n4(y)})`) : '\\partial L/\\partial \\hat y'
        nums.push(`\\frac{\\partial L}{\\partial \\hat y_{${j + 1}}} = ${dl} = ${n4(g.dA[L][j])}`)
        nums.push(`${fprimeName(f)}(z) = ${n4(g.fprime[L][j])}${f === 'sigmoid' ? ` = ${n4(yh)}(1-${n4(yh)})` : ''}`)
        nums.push(`\\delta_{${j + 1}}^{(${L})} = ${tn(g.dA[L][j])}\\cdot${tn(g.fprime[L][j])} = ${n4(g.delta[L][j])}`)
      }
      return {
        title: 'Backward pass begins at the loss',
        plain: beginner ? 'Now we go backwards. First: if the output z changed a little, how much would the loss change? That sensitivity is called δ (delta).' : 'δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ is the error signal at the output. Every other gradient is built from it by the chain rule.',
        general, numbers: nums,
        note: g.outputRule !== 'chain' ? (depth === 'beginner' ? 'This loss and output activation fit together so neatly that the result is simply “prediction minus target”.' : `For ${g.outputRule === 'sigmoid+bce' ? 'sigmoid + binary cross-entropy' : 'softmax + categorical cross-entropy'} the factors cancel exactly; the generic chain gives the same number (tested).`) : undefined,
      }
    }
    case 'grad': {
      const l = step.layer, j = Math.min(focus, cfg.sizes[l] - 1), i = 0
      const d = g.delta[l][j], a = t.a[l - 1][i]
      return {
        title: `Gradients for the parameters of layer ${l}`,
        plain: beginner ? `A weight's blame is: the error at the neuron it feeds (δ) times the value that flowed through it (a). Big input and big error → big gradient.` : `∂L/∂w_ji = δ_j a_i: the outer product of the layer's error with its inputs. Biases get δ_j directly because ∂z/∂b = 1.`,
        general: `\\frac{\\partial L}{\\partial w_{ji}^{(${l})}} = \\delta_j^{(${l})}\\,a_i^{(${l - 1})}, \\qquad \\frac{\\partial L}{\\partial b_j^{(${l})}} = \\delta_j^{(${l})}`,
        numbers: [`\\frac{\\partial L}{\\partial w_{${j + 1}${i + 1}}^{(${l})}} = ${tn(d)}\\cdot${tn(a)} = ${n4(g.dW[l - 1][j][i])}`, `\\frac{\\partial L}{\\partial b_{${j + 1}}^{(${l})}} = ${n4(g.db[l - 1][j])}`],
        note: depth === 'advanced' ? `Matrix form ∂L/∂W⁽${l}⁾ = δ⁽${l}⁾ a⁽${l - 1}⁾ᵀ (${cfg.sizes[l]} × ${cfg.sizes[l - 1]}).` : undefined,
      }
    }
    case 'delta': {
      const l = step.layer, j = Math.min(focus, cfg.sizes[l] - 1), f = cfg.acts[l - 1]
      const col = params.W[l].map((row) => row[j]), dn = g.delta[l + 1]
      const terms = col.map((w, k) => `${tn(w)}\\cdot${tn(dn[k])}`)
      const shown = terms.length > MAX_TERMS ? [...terms.slice(0, MAX_TERMS - 1), '\\dots'] : terms
      return {
        title: `Error flows back into layer ${l}`,
        plain: beginner ? `A hidden neuron is blamed in proportion to how strongly it is connected to the neurons that made errors, scaled by how sensitive it was (its slope f′).` : `δ⁽${l}⁾ collects the errors of layer ${l + 1} through the same weights used forward (transposed), then multiplies by the local slope f′(z).`,
        general: `\\delta_j^{(${l})} = \\Big(\\sum_k w_{kj}^{(${l + 1})}\\,\\delta_k^{(${l + 1})}\\Big)\\, f'(z_j^{(${l})})`,
        numbers: [
          `\\frac{\\partial L}{\\partial a_{${j + 1}}^{(${l})}} = ${shown.join(' + ')} = ${n4(g.dA[l][j])}`,
          `${fprimeName(f)}(${tn(t.z[l][j])}) = ${n4(g.fprime[l][j])}`,
          `\\delta_{${j + 1}}^{(${l})} = ${tn(g.dA[l][j])}\\cdot${tn(g.fprime[l][j])} = ${n4(g.delta[l][j])}`,
        ],
        note: Math.abs(g.fprime[l][j]) < 0.05 ? `This neuron's slope is ${n4(g.fprime[l][j])}: almost no error passes through it${f === 'relu' && g.fprime[l][j] === 0 ? ' (an inactive ReLU)' : ' (saturation)'}.` : undefined,
      }
    }
    case 'update': {
      const rec = x.records.find((r) => r.ref.kind === 'w') ?? x.records[0]
      const o = x.opt
      const general = o.kind === 'sgd' ? '\\theta \\leftarrow \\theta - \\eta\\,\\frac{\\partial L}{\\partial \\theta}' : o.kind === 'momentum' ? 'v \\leftarrow \\mu v + g,\\quad \\theta \\leftarrow \\theta - \\eta v' : '\\theta \\leftarrow \\theta - \\eta\\,\\frac{\\hat m}{\\sqrt{\\hat v}+\\epsilon}'
      const nums = o.kind === 'sgd' ? [`${tn(rec.old)} - ${o.lr}\\cdot${tn(rec.grad)} = ${n4(rec.next)}`] : [`\\Delta = ${n4(rec.step)},\\ \\ ${n4(rec.old)} \\to ${n4(rec.next)}`]
      nums.push(`L:\\ ${n4(loss)} \\to ${n4(x.lossAfter)}\\ \\text{(this sample, after the step)}`)
      return {
        title: 'Gradient descent step',
        plain: beginner ? 'Every weight takes a small step in the direction that lowers the loss: against its gradient. The learning rate η sets the step size.' : `All ${x.records.length} parameters are updated at once with ${o.kind === 'sgd' ? 'plain SGD' : o.kind === 'momentum' ? 'SGD with momentum' : 'Adam'} (η = ${o.lr}). Here the update uses this one sample's gradient (online learning, batch size 1).`,
        general, numbers: nums,
        note: x.lossAfter > loss ? 'The loss went up: the step overshot. Try a smaller learning rate.' : undefined,
      }
    }
  }
}
