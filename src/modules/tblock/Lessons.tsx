/** Three guided lessons with instructor notes and one-click demonstrations; the notes bar for presentation mode. */
import { ArrowRight, Clock, Presentation } from 'lucide-react'
import { Card } from '../../components/viz'
import { useApp } from '../../sim/appState'
import type { ArchPreset, StageId } from './stages'
import { seekRef } from './seek'
import { tbStore, type TbTab } from './store'

interface Demo { arch?: ArchPreset; stage?: StageId; tab?: TbTab; sel?: Partial<{ i: number; j: number; h: number }>; maskOff?: boolean; heads?: boolean[]; attnScale?: number; T?: number }
interface Segment { min: string; title: string; say: string; notes: string; ask?: string; demo: Demo }
interface Lesson { id: string; title: string; minutes: number; goal: string; segments: Segment[] }

export const LESSONS: Lesson[] = [
  { id: 'l1', title: 'Lesson 1 — What happens inside a Transformer block?', minutes: 15, goal: 'A conceptual tour: what enters, the two sublayers, residuals, and what comes out.',
    segments: [
      { min: '0–3', title: 'Hidden states in, hidden states out', say: 'A block receives one vector per token and returns one vector per token, same shape.', notes: 'Stress that tokenization and embeddings happen before the first block; open “where do these vectors come from?”.', ask: 'Does the block see words?', demo: { arch: 'gpt', stage: 'x', tab: 'block' } },
      { min: '3–6', title: 'Two sublayers on a residual stream', say: 'Click through the architecture strip: LayerNorm → attention → add, LayerNorm → MLP → add.', notes: 'Point at the green dashed residual arcs: the stream flows past each sublayer.', demo: { stage: 'res1', tab: 'block' } },
      { min: '6–10', title: 'Attention mixes tokens', say: 'Show the attention weights and the arcs from the query token.', notes: 'Say clearly: toy weights are random; patterns are not linguistic. Real attention is in the Real model tab.', ask: 'Why can “The” only look at itself?', demo: { stage: 'weights', tab: 'block', sel: { i: 3, h: 0 } } },
      { min: '10–13', title: 'The MLP transforms each token on its own', say: 'Open Advanced → Attention vs MLP: a lower-triangular map vs a diagonal one.', notes: 'This is the single most important structural distinction of the block.', demo: { tab: 'advanced' } },
      { min: '13–15', title: 'Output: still hidden states', say: 'Y has the same shape as X. Probabilities need a final LayerNorm, the LM head and a softmax.', notes: 'Bridge to Lesson 2 and to the Real model tab.', demo: { stage: 'out', tab: 'block' } },
    ] },
  { id: 'l2', title: 'Lesson 2 — Self-attention, step by step', minutes: 25, goal: 'The full mathematics of scaled dot-product attention with real numbers.',
    segments: [
      { min: '0–4', title: 'Q, K and V are three projections', say: 'Click an output cell of Q: its value is one row of X_n times one column of W_Q.', notes: 'Edit a weight in the inspector and watch every downstream number change.', ask: 'Which columns belong to head 2?', demo: { stage: 'qkv', tab: 'block' } },
      { min: '4–8', title: 'One dot product', say: 'q · k: multiply coordinates, add. Highlight the cell in S.', notes: 'Have students compute it by hand before revealing (Challenge 2).', demo: { stage: 'dot', tab: 'block', sel: { i: 3, j: 1, h: 0 } } },
      { min: '8–11', title: 'The score matrix and scaling', say: 'S = Q Kᵀ, then divide by √dₖ. Show the variance table.', notes: 'Scaling never changes which key wins; it changes how peaked the softmax is.', ask: 'What happens to softmax as dₖ grows without scaling?', demo: { stage: 'scale', tab: 'block' } },
      { min: '11–15', title: 'The causal mask', say: 'Add −∞ above the diagonal. Use the explorer to grow the visible context.', notes: 'Toggle the mask off briefly and explain why this would break generation.', demo: { stage: 'mask', tab: 'block' } },
      { min: '15–19', title: 'Softmax, stably', say: 'Logits − max → exp → divide by the sum. Every row sums to 1; masked entries are exactly 0.', notes: 'Challenge 5 is a good in-class check.', demo: { stage: 'softmax', tab: 'block', sel: { i: 3 } } },
      { min: '19–22', title: 'Mixing the values', say: 'O_i = Σ A_ij v_j: watch each weighted contribution and the stacked bars.', notes: 'Only values are mixed: queries and keys just decide the weights.', demo: { stage: 'av', tab: 'block', sel: { i: 3 } } },
      { min: '22–25', title: 'Heads, concat and W_O', say: 'Each head has its own pattern; concatenate and project. Disable a head and measure the change.', notes: 'Do not claim heads have fixed linguistic jobs.', ask: 'What happens if we disable head 1?', demo: { tab: 'heads' } },
    ] },
  { id: 'l3', title: 'Lesson 3 — From attention to contextual representations', minutes: 20, goal: 'Residuals, normalization, the MLP, stacking, and decoder vs encoder.',
    segments: [
      { min: '0–4', title: 'Residual connections', say: 'H = X + Attn(LN₁(X)). Scale the attention contribution to 0 and back.', notes: 'Residuals give an additive path; they do not guarantee stable training by themselves.', demo: { stage: 'res1', tab: 'block', attnScale: 1 } },
      { min: '4–7', title: 'LayerNorm, per token', say: 'Mean, variance, normalize, γ and β, with every number shown.', notes: 'Emphasize: across features of one token, not across tokens.', demo: { stage: 'ln2', tab: 'block' } },
      { min: '7–11', title: 'Inside the MLP', say: 'Expand to d_ff, apply GELU, project back. Switch hidden units off; compare SwiGLU.', notes: 'SwiGLU is a different architecture (three matrices).', demo: { stage: 'ffn1', tab: 'block' } },
      { min: '11–14', title: 'Trace one token through the block', say: 'X + attention update + MLP update = Y, feature by feature.', notes: 'Pre-LN: the identity is exact, check any column.', demo: { stage: 'res2', tab: 'block', sel: { i: 3 } } },
      { min: '14–17', title: 'Stacking blocks', say: 'Representations evolve block by block; follow norms and PCA trajectories.', notes: 'Remind: 2-D projections are approximations.', demo: { tab: 'stack' } },
      { min: '17–20', title: 'Decoder vs encoder', say: 'Pre-LN causal (GPT) vs post-LN bidirectional (BERT), same input and weights.', notes: 'Finish with a real model: Real model tab, next-token prediction.', ask: 'Why can BERT not generate text left to right as is?', demo: { tab: 'compare' } },
    ] },
]

export function runDemo(d: Demo) {
  const s = tbStore.getState()
  if (d.arch) s.setArch(d.arch)
  if (d.maskOff !== undefined) tbStore.getState().setMaskOff(d.maskOff)
  if (d.sel) tbStore.getState().select(d.sel)
  if (d.heads) tbStore.getState().setIv({ heads: d.heads })
  if (d.attnScale !== undefined) tbStore.getState().setIv({ attnScale: d.attnScale })
  if (d.stage) seekRef.current(d.stage)
  if (d.tab) tbStore.getState().setTab(d.tab)
}

export function LessonsPanel() {
  const { setPresenting } = useApp()
  return (
    <div className="grid gap-3">
      <Card title="three guided lessons · each “Show me” sets up the exact state it talks about">
        <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
          <p className="flex-1">For teaching: press <b className="text-fg">Present</b> (or F) for fullscreen with larger equations and instructor notes; step with ← / →, R replays an operation, E exports diagrams and calculation sheets.</p>
          <button type="button" className="btn" onClick={() => setPresenting(true)}><Presentation size={14} />Start presentation mode</button>
        </div>
      </Card>
      {LESSONS.map((l) => (
        <Card key={l.id} title={`${l.title} · ${l.minutes} minutes`}>
          <p className="mb-2 text-[12.5px] text-muted">{l.goal}</p>
          <ol className="grid gap-2">
            {l.segments.map((sg, k) => (
              <li key={k} className="grid gap-2 rounded-lg border border-line bg-panel-2/40 p-2.5 md:grid-cols-[86px_1fr_auto] md:items-start">
                <span className="flex items-center gap-1 font-mono text-[12px] text-cyan"><Clock size={12} />{sg.min}</span>
                <div className="grid gap-0.5">
                  <b className="text-[13.5px]">{sg.title}</b>
                  <p className="text-[12.5px] leading-relaxed">{sg.say}</p>
                  <p className="text-[12px] text-muted"><span className="text-faint">Instructor note:</span> {sg.notes}</p>
                  {sg.ask && <p className="text-[12px] text-amber">Ask: {sg.ask}</p>}
                </div>
                <button type="button" className="btn" onClick={() => runDemo(sg.demo)} data-testid={`demo-${l.id}-${k}`}>Show me<ArrowRight size={13} /></button>
              </li>
            ))}
          </ol>
        </Card>
      ))}
    </div>
  )
}

const STAGE_NOTES: Partial<Record<StageId, string>> = {
  x: 'Rows are tokens, columns are features. The block never sees words.',
  ln1: 'Normalize each token over its own features; γ and β are learned.',
  qkv: 'Three learned projections; columns are split among heads.',
  dot: 'One number per (query, key) pair: multiply coordinates and add.',
  scores: 'Rows = queries, columns = keys. Not symmetric.',
  scale: 'Divide by √dₖ to keep logits in a reasonable range.',
  mask: 'Causal: −∞ above the diagonal → exactly zero weight.',
  softmax: 'Subtract the max, exponentiate, normalize. Rows sum to 1.',
  weights: 'Where each head reads from; not an explanation of the prediction.',
  av: 'Only values are mixed. O_i is a weighted average of v_j.',
  concat: 'Heads side by side, back to width d.',
  proj: 'W_O mixes the heads into one update.',
  res1: 'Add the update to the stream (post-LN: then normalize).',
  ln2: 'Normalize again before the MLP.',
  ffn1: 'Same W₁ for every token: expand to d_ff.',
  act: 'Non-linearity; SwiGLU would be a different MLP.',
  ffn2: 'Project back to d. No mixing across positions.',
  res2: 'Second additive update. Trace one token end to end.',
  out: 'Still a hidden state: an LM head is needed for probabilities.',
}
export function NotesBar({ stageId }: { stageId: StageId }) {
  return (
    <div className="rounded-lg border border-amber/50 bg-amber/5 px-3 py-2 text-[14px]" data-testid="notes-bar">
      <span className="eyebrow mr-2 text-amber">instructor note</span>{STAGE_NOTES[stageId] ?? ''}
    </div>
  )
}
