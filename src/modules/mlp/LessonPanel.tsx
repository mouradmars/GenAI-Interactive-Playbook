/** A 15-minute guided lesson, "How does a neural network learn?" Each segment sets up the lab state it talks about. */
import { ArrowRight, Clock } from 'lucide-react'
import { Card } from '../../components/viz'
import { microSteps } from './model'
import { mlpStore, type LabTab } from './store'
import type { PresetId } from './model'

interface Segment { min: string; title: string; say: string; ask?: string; go: { preset?: PresetId; tab: LabTab; step?: (L: number) => number; select?: Parameters<ReturnType<typeof mlpStore.getState>['select']>[0]; mode?: 'forward' | 'backprop' | 'iteration' } }

export const LESSON: Segment[] = [
  { min: '0–2', title: 'A neuron is a weighted vote', say: 'Load the 2 → 2 → 1 example. Step to z¹: hidden neuron 1 computes 0.5·1.0 − 0.3·0.8 + 0.1 = 0.36, then squashes it with the sigmoid to 0.589.', ask: 'Which input pushes neuron 1 down, and why?', go: { preset: 'worked', tab: 'microscope', step: () => 1, select: { kind: 'neuron', layer: 1, j: 0 }, mode: 'iteration' } },
  { min: '2–4', title: 'The forward pass and the loss', say: 'Play through to L. The prediction 0.540 is compared with the target 1 and the loss is (0.540 − 1)² = 0.211.', ask: 'Would a prediction of 0.9 give a smaller or larger loss?', go: { preset: 'worked', tab: 'microscope', step: (L) => 2 * L + 1 } },
  { min: '4–7', title: 'Blame flows backwards', say: 'Step into the backward pass. δ² = 2(ŷ − y)·σ′(z²). Then each hidden neuron receives its share through the same weight it used going forward, times its own slope.', ask: 'Hidden neuron 2 is connected by a negative weight. What sign will its δ have?', go: { preset: 'worked', tab: 'microscope', step: (L) => 2 * L + 4, select: { kind: 'neuron', layer: 1, j: 1 } } },
  { min: '7–9', title: 'The chain rule, made visible', say: 'Select w₁₁⁽¹⁾ and press “Animate chain rule”. The gradient is a product of local derivatives along the path to the loss. In deeper networks, it is the sum over every path.', ask: 'Which factor is smallest, and what does that do to the product?', go: { preset: 'worked', tab: 'microscope', step: (L) => microSteps(L, 'iteration').length - 2, select: { kind: 'w', layer: 1, j: 0, i: 0 } } },
  { min: '9–10', title: 'Trust, but verify', say: 'Open the gradient checker: every backprop gradient agrees with a finite-difference estimate to about 10⁻¹⁰. This is how real implementations are tested.', go: { tab: 'check' } },
  { min: '10–12', title: 'One step downhill', say: 'In the loss lab, the curve shows the loss as one weight changes. The gradient is its slope; gradient descent steps against it. Then apply the update on the timeline and watch the loss drop.', ask: 'What happens with a learning rate of 10?', go: { tab: 'loss', select: { kind: 'w', layer: 2, j: 0, i: 0 } } },
  { min: '12–15', title: 'Learning XOR', say: 'Switch to the XOR network and train. The loss curve falls, the decision boundary bends. Below, a network without a hidden layer never gets past 3 of 4.', ask: 'Why can one straight line never solve XOR?', go: { preset: 'xor', tab: 'train' } },
]

/** after the 15 minutes: one segment per phase-4/5 lab */
export const FURTHER: Segment[] = [
  { min: '+5', title: 'A real trained network', say: 'Load the trained 64 → 16 → 16 → 10 digit classifier (97.3 % on held-out images). Draw a digit, look at the saliency map and the first-layer weight images, then trace your drawing through the microscope.', ask: 'Which pixels would turn this 1 into a 7?', go: { preset: 'digits', tab: 'digits' } },
  { min: '+3', title: 'Why the activation matters', say: 'Compare f and f′. Sigmoid passes back at most a quarter of the gradient; ReLU passes all of it, or none.', go: { tab: 'activations' } },
  { min: '+5', title: 'Vanishing and exploding gradients', say: 'A 12-layer sigmoid network: the first layer gets a million times less gradient than the last. Switch to ReLU + He and train both.', ask: 'What would happen with 24 layers?', go: { tab: 'vanishing' } },
  { min: '+10', title: 'Check yourself', say: 'Nine seeded questions to compute by hand: a weighted sum, a sigmoid, an output error, a gradient, an SGD step, path counting and concepts.', go: { tab: 'challenge' } },
]

export function runSegment(s: Segment) {
  const st = mlpStore.getState()
  if (s.go.preset) st.loadPreset(s.go.preset)
  if (s.go.mode) mlpStore.getState().setMode(s.go.mode)
  if (s.go.select) mlpStore.getState().select(s.go.select)
  const L = mlpStore.getState().cfg.sizes.length - 1
  if (s.go.step) mlpStore.getState().setMicro(s.go.step(L))
  mlpStore.getState().setTab(s.go.tab)
}

export function LessonPanel() {
  return (
    <div className="grid gap-3">
      <Card title="15-minute lesson · how does a neural network learn?">
        <p className="text-[13px] leading-relaxed text-muted">Each segment sets up the lab for you. Read the line aloud, ask the question, then let students predict before you step forward. Every number on screen is computed live; the quoted numbers are checked by the test suite.</p>
      </Card>
      <ol className="grid gap-2">
        {LESSON.map((s, i) => (
          <li key={i} className="grid gap-2 rounded-xl border border-line bg-panel/70 p-3 md:grid-cols-[90px_1fr_auto] md:items-start">
            <span className="flex items-center gap-1 font-mono text-[12px] text-cyan"><Clock size={13} />{s.min} min</span>
            <div className="grid gap-1">
              <b className="text-[14px]">{s.title}</b>
              <p className="text-[13px] leading-relaxed text-muted">{s.say}</p>
              {s.ask && <p className="text-[12.5px] text-amber">Ask: {s.ask}</p>}
            </div>
            <button type="button" className="btn" onClick={() => runSegment(s)} data-testid={`lesson-go-${i}`}>Show me<ArrowRight size={13} /></button>
          </li>
        ))}
      </ol>
      <Card title="going further (phases 4–5)">
        <ol className="grid gap-2">
          {FURTHER.map((s, i) => (
            <li key={i} className="grid gap-2 rounded-lg border border-line bg-panel-2/40 p-3 md:grid-cols-[90px_1fr_auto] md:items-start">
              <span className="font-mono text-[12px] text-cyan">{s.min} min</span>
              <div className="grid gap-1"><b className="text-[14px]">{s.title}</b><p className="text-[13px] leading-relaxed text-muted">{s.say}</p>{s.ask && <p className="text-[12.5px] text-amber">Ask: {s.ask}</p>}</div>
              <button type="button" className="btn" onClick={() => runSegment(s)} data-testid={`further-go-${i}`}>Open<ArrowRight size={13} /></button>
            </li>
          ))}
        </ol>
      </Card>
    </div>
  )
}
