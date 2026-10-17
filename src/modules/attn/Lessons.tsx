/** Three classroom lessons with timed segments, instructor notes, discussion questions, short exercises and one-click demos. */
import { ArrowRight, Clock, Presentation } from 'lucide-react'
import { Card } from '../../components/viz'
import { useApp } from '../../sim/appState'
import { seekRef } from './seek'
import type { StageId } from './stages'
import { attnStore, type AttnTab } from './store'

export interface Demo { reset?: boolean; tab?: AttnTab; stage?: StageId; sel?: Partial<{ i: number; j: number; h: number }>; H?: number; causal?: boolean; scale?: boolean; follow?: number; edit?: { k: 'Wq' | 'Wk' | 'Wv' | 'Wo'; r: number; c: number; delta: number; label: string } }
interface Segment { min: string; title: string; say: string; notes: string; ask?: string; exercise?: string; demo: Demo }
export interface Lesson { id: string; title: string; minutes: number; goal: string; segments: Segment[]; discussion: string[] }

export const LESSONS: Lesson[] = [
  { id: 'l1', title: 'Lesson 1 — Understanding Q, K and V', minutes: 15, goal: 'Why tokens need context, what the input to attention is, and how three learned projections create queries, keys and values.',
    segments: [
      { min: '0–3', title: 'Why self-attention?', say: 'In “cat chased the mouse”, the vector for “mouse” cannot know it is being chased unless it can read from other positions.', notes: 'Attention is the operation that lets every position gather information from others, in parallel.', ask: 'What information should “mouse” gather, and from where?', demo: { reset: true, stage: 'tokens', tab: 'guided', sel: { i: 3 } } },
      { min: '3–6', title: 'The input is a matrix of vectors', say: 'Rows are positions, columns are features. These rows are hand-chosen hidden states, not word meanings.', notes: 'Distinguish token text, embedding, positional information and hidden state (the four cards).', demo: { stage: 'embed', tab: 'guided' } },
      { min: '6–10', title: 'Three projections', say: 'Click an output cell of Q: it is one row of X times one column of W_Q. Same for K and V.', notes: 'Use “Animate” to show the three pathways in turn. Q/K/V are learned numbers; the seek/match/retrieve analogy is only an analogy.', exercise: 'Compute q₁,₁ = x₁ · W_Q[:, 1] by hand, then click the cell to check.', demo: { stage: 'qkv', tab: 'guided', sel: { i: 0, h: 0 } } },
      { min: '10–13', title: 'Experiment: change V only', say: 'Adding 1 to a value weight changes the output but leaves every attention weight identical.', notes: 'The comparison table shows max |ΔA| = 0 (“=”). Then try the same with W_Q.', ask: 'Why can V never change where a token attends?', demo: { stage: 'qkv', tab: 'guided', edit: { k: 'Wv', r: 0, c: 0, delta: 1, label: 'before editing W_V' } } },
      { min: '13–15', title: 'Wrap-up', say: 'Queries and keys decide the weights; values decide what is mixed.', notes: 'Reset the lab before the next lesson.', exercise: 'Challenge b1 (identify Q, K and V).', demo: { reset: true, tab: 'challenge' } },
    ],
    discussion: ['Why might we want separate key and value projections rather than using the same vector for both?', 'If W_Q = W_K, what property would the score matrix have?'] },
  { id: 'l2', title: 'Lesson 2 — Scaled dot-product attention from scratch', minutes: 25, goal: 'Every operation of one attention head with real numbers: dot products, scaling, masking, softmax and weighted values.',
    segments: [
      { min: '0–4', title: 'One dot product', say: 'Token 3 as the query, token 1 as the key: align the two vectors, multiply, add.', notes: 'Have students compute it before revealing (challenge i1 has fresh numbers).', demo: { reset: true, stage: 'dot', tab: 'guided', sel: { i: 2, j: 0, h: 0 } } },
      { min: '4–8', title: 'The score matrix', say: 'S = QKᵀ: rows are queries, columns are keys. Compare s₃₁ with s₁₃.', notes: 'Toggle raw vs normalized connections: raw scores are not probabilities.', ask: 'Why is S not symmetric?', demo: { stage: 'dot', tab: 'guided' } },
      { min: '8–12', title: 'Why √d_k?', say: 'With random unit-variance vectors, Var(q·k) = d_k. Pick d_k = 256 and compare entropy with and without scaling.', notes: 'Scaling does not reorder keys and is not a cure for every instability.', exercise: 'Challenge i2 (a scaled score).', demo: { stage: 'scale', tab: 'guided' } },
      { min: '12–16', title: 'The causal mask', say: 'Token 2 can attend to tokens 1 and 2 only. Masked logits are −∞, so their probability is exactly 0.', notes: 'Switch to padding and to combined masks; show the fully masked row warning by marking key 1 as padding.', ask: 'What would go wrong in generation without the mask?', demo: { stage: 'mask', tab: 'guided', sel: { i: 1 } } },
      { min: '16–20', title: 'Softmax, stably', say: 'Subtract the maximum, exponentiate, sum, divide. Each valid row sums to 1.', notes: 'Use the nudge sandbox to show competition between keys, and that adding a constant changes nothing.', exercise: 'Challenge i4 (softmax probability).', demo: { stage: 'softmax', tab: 'guided', sel: { i: 3 } } },
      { min: '20–23', title: 'Three synchronized microscopes', say: 'Click the heatmap cell token 3 → token 1 and follow it through all three views.', notes: 'A heatmap is not an explanation of reasoning.', demo: { tab: 'microscopes', sel: { i: 2, j: 0, h: 0 } } },
      { min: '23–25', title: 'Weighted values', say: 'o_i = Σ a_ij v_j: a blend, not a selection. Then follow one token end to end.', notes: 'Use “Follow one token” with Play; rewind and replay.', exercise: 'Challenge i5 (weighted value sum).', demo: { tab: 'follow', sel: { i: 3, h: 0 }, follow: 0 } },
    ],
    discussion: ['Softmax normalizes per query row. What would go wrong if it normalized the whole matrix?', 'Could a model learn to attend only to the previous token? What would its score matrix look like?'] },
  { id: 'l3', title: 'Lesson 3 — Multi-head attention and output projection', minutes: 20, goal: 'Parallel heads with their own projections, comparing them, concatenation, and the learned output projection.',
    segments: [
      { min: '0–4', title: 'Parallel heads', say: 'Two heads of size 2 instead of one of size 4. Each head has its own W_Q, W_K, W_V block.', notes: 'Heads are not pre-assigned linguistic roles; these are toy weights.', demo: { reset: true, stage: 'heads', tab: 'guided' } },
      { min: '4–8', title: 'Compare heads', say: 'Side-by-side heatmaps, their difference, JS divergence and entropy.', notes: 'Switch to h = 4: same parameters, smaller heads.', ask: 'Does a different pattern mean a different “function”?', demo: { stage: 'heads', tab: 'guided', H: 2 } },
      { min: '8–11', title: 'Edit only head 1', say: 'Change one weight in head 1\'s block: head 2 is completely unchanged; the output changes.', notes: 'Point at the comparison table: head 2 row is all “=”.', demo: { stage: 'heads', tab: 'guided', edit: { k: 'Wq', r: 2, c: 0, delta: -2, label: 'before editing head 1' } } },
      { min: '11–14', title: 'Concatenation', say: 'Row i of H = [head₁ row i | head₂ row i]. Nothing is computed.', notes: 'Disabled heads contribute zeros.', exercise: 'Challenge a3 (concatenate).', demo: { reset: true, stage: 'concat', tab: 'guided' } },
      { min: '14–17', title: 'The output projection', say: 'O = H W_O mixes the heads: click an output cell and see each head\'s share.', notes: 'O = Σ_h head_h W_O^(h).', exercise: 'Challenge a4 (output projection).', demo: { stage: 'proj', tab: 'guided', sel: { i: 3 } } },
      { min: '17–20', title: 'Ablation and research questions', say: 'Disable a head and measure the change; look at entropy, similarity and a real model\'s attention.', notes: 'Finish with the real-model section; stress that attention weights alone do not explain predictions.', demo: { tab: 'research' } },
    ],
    discussion: ['Why is the output projection needed after concatenation?', 'With d_model fixed, what do we gain and lose by using more heads?'] },
]

export function runDemo(d: Demo) {
  const s = attnStore.getState()
  if (d.reset) s.resetAll()
  if (d.H) attnStore.getState().setH(d.H)
  if (d.causal !== undefined) attnStore.getState().setCausal(d.causal)
  if (d.scale !== undefined) attnStore.getState().setScale(d.scale)
  if (d.sel) attnStore.getState().select({ ...d.sel, cell: null })
  if (d.edit) { const st = attnStore.getState(); st.saveBaseline(d.edit.label); st.setW(d.edit.k, d.edit.r, d.edit.c, st.params[d.edit.k][d.edit.r][d.edit.c] + d.edit.delta) }
  if (d.follow !== undefined) attnStore.getState().setFollowStep(d.follow)
  if (d.stage) seekRef.current(d.stage)
  if (d.tab) attnStore.getState().setTab(d.tab)
}

export function LessonsTab() {
  const { setPresenting } = useApp()
  return (
    <div className="grid gap-3">
      <Card title="classroom presentation mode">
        <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
          <p className="flex-1">Press <b className="text-fg">Present</b> (or F) for fullscreen with large type and instructor notes. ← / → step through operations, Space plays, R replays an operation, H switches head, T switches to the projector (light) theme, N and I hide the side panels, E exports heatmaps (PNG) and calculation sheets (Markdown/JSON). Each “Show me” sets up the exact state it describes.</p>
          <button type="button" className="btn" onClick={() => setPresenting(true)}><Presentation size={14} />Start presentation mode</button>
        </div>
      </Card>
      {LESSONS.map((l) => (
        <Card key={l.id} title={`${l.title} · ${l.minutes} minutes`}>
          <p className="mb-2 text-[12.5px] text-muted">{l.goal}</p>
          <ol className="grid gap-2">
            {l.segments.map((sg, k) => (
              <li key={k} className="grid gap-2 rounded-lg border border-line bg-panel-2/40 p-2.5 @3xl:grid-cols-[80px_1fr_auto] @3xl:items-start">
                <span className="flex items-center gap-1 font-mono text-[12px] text-cyan"><Clock size={12} />{sg.min}</span>
                <div className="grid gap-0.5">
                  <b className="text-[13.5px]">{sg.title}</b>
                  <p className="text-[12.5px] leading-relaxed">{sg.say}</p>
                  <p className="text-[12px] text-muted"><span className="text-faint">Instructor note:</span> {sg.notes}</p>
                  {sg.ask && <p className="text-[12px] text-amber">Ask: {sg.ask}</p>}
                  {sg.exercise && <p className="text-[12px] text-teal">Exercise: {sg.exercise}</p>}
                </div>
                <button type="button" className="btn" onClick={() => runDemo(sg.demo)} data-testid={`demo-${l.id}-${k}`}>Show me<ArrowRight size={13} /></button>
              </li>
            ))}
          </ol>
          <div className="mt-2 rounded-md border-l-2 border-amber bg-amber/5 px-3 py-2 text-[12.5px]"><b>Discussion questions</b><ul className="mt-1 list-disc pl-5 text-muted">{l.discussion.map((q) => <li key={q}>{q}</li>)}</ul></div>
        </Card>
      ))}
    </div>
  )
}

const STAGE_NOTES: Record<StageId, string> = {
  tokens: 'Attention works on vectors, one per position. The text itself never reaches it.',
  embed: 'X: rows = positions, columns = features. Hand-chosen hidden states, not meanings.',
  qkv: 'Three learned projections of the same X. Click an output cell to see the products.',
  dot: 'One score per (query, key) pair: multiply coordinates, add. Not probabilities.',
  scale: 'Divide by √d_k: keeps logits moderate; does not change the ranking.',
  mask: 'Masked logits → −∞ → probability exactly 0. Causal: no peeking at the future.',
  softmax: 'Per query row: subtract max, exp, normalize. Rows sum to 1.',
  heatmap: 'Rows = queries, columns = keys. Where each token reads from, not why.',
  aggregate: 'o_i = Σ a_ij v_j: a blend of values, never a single selection.',
  heads: 'Each head has its own projections and its own pattern.',
  concat: 'Heads side by side: just rearranging numbers.',
  proj: 'W_O mixes the heads back to d_model.',
  output: 'Same shape as X; added to the residual stream in a Transformer block.',
}
export function NotesBar({ stageId }: { stageId: StageId }) {
  return (
    <div className="rounded-lg border border-amber/50 bg-amber/5 px-3 py-2 text-[15px]" data-testid="attn-notes-bar">
      <span className="eyebrow mr-2 text-amber">instructor note</span>{STAGE_NOTES[stageId]}
    </div>
  )
}
