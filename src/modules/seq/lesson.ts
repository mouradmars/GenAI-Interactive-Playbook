/** Stages, equations and lesson text for module 09. Separate from the engine and the views. */
export type StageId = 'task' | 'rnn' | 'lstm' | 'cell' | 'attention' | 'parallel' | 'gradients' | 'caveat' | 'lengths' | 'compare'

export interface Stage { id: StageId; group: string; short: string; title: string; equation: string; beginner: string; intermediate: string; advanced: string }

export const STAGES: Stage[] = [
  {
    id: 'task', group: 'Task', short: 'Task', title: 'One task, three architectures',
    equation: String.raw`x_1,\dots,x_T\in\{a,b,c,d\}\quad\longrightarrow\quad \hat y = x_1\qquad(\text{dependency distance } T-1)`,
    beginner: 'Each model reads a sequence of symbols one by one. At the end it must say what the very first symbol was. Everything in between is noise to be ignored.',
    intermediate: 'The three models have the same width (8), were trained with the same optimizer, steps and data, and differ only in how they connect time steps. Edit the sequence: all three predictions update.',
    advanced: 'This is a minimal long-range memory task in the family of Hochreiter & Schmidhuber’s (1997) benchmarks. Its difficulty is not computing the answer but carrying one symbol across T − 1 steps, and learning to do so from gradients.',
  },
  {
    id: 'rnn', group: 'Recurrence', short: 'RNN', title: 'An RNN unrolled through time',
    equation: String.raw`\mathbf h_t=\tanh(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b),\qquad \mathbf h_0=\mathbf 0,\qquad \mathbf z=W_y\mathbf h_T+\mathbf b_y`,
    beginner: 'An RNN keeps a small memory, the hidden state. At every step it mixes the new symbol into that memory. The same weights are used at every step: the diagram shows one cell copied across time.',
    intermediate: 'Step the time cursor. Each column of the heatmap is h_t; each new column is computed from the previous one, which is why the steps cannot run in parallel. The inspector shows the pre-activation of the selected unit term by term.',
    advanced: 'Unrolling turns the recurrence into a deep feed-forward network with tied weights and depth T. That depth is what makes gradients vanish or explode: ∂h_T/∂h_1 is a product of T − 1 Jacobians diag(1 − h_t²) W_h.',
  },
  {
    id: 'lstm', group: 'Recurrence', short: 'LSTM gates', title: 'LSTM gates at one time step',
    equation: String.raw`\begin{aligned}\mathbf i_t,\mathbf f_t,\mathbf o_t&=\sigma(\cdot),\ \ \mathbf g_t=\tanh(\cdot)\ \ \text{of }W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b\\ \mathbf c_t&=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t,\qquad \mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)\end{aligned}`,
    beginner: 'An LSTM adds a second memory, the cell state, guarded by three valves. The forget gate decides what to keep, the input gate what to write, the output gate what to show.',
    intermediate: 'Gate values lie between 0 (closed) and 1 (open). The diagram shows the selected unit at the cursor time; the heatmaps show every unit at every step. Watch the forget gate stay near 1 while the first symbol must be remembered.',
    advanced: 'The cell update is additive. Its Jacobian ∂c_t/∂c_{t−1} along the direct path is diag(f_t), with no weight matrix and no squashing, so with f ≈ 1 the error signal can travel many steps (the “constant error carousel”). Forget gates were added by Gers et al. (2000).',
  },
  {
    id: 'cell', group: 'Recurrence', short: 'Cell state', title: 'The cell state as a memory highway',
    equation: String.raw`\frac{\partial c_T}{\partial c_1}\Big|_{\text{direct path}}=\prod_{t=2}^{T} f_t`,
    beginner: 'Follow one unit of the cell state over time. If its forget gate stays open, what was written at the start is still there at the end.',
    intermediate: 'The product of forget gates along the sequence is how much of c₁ survives unchanged on the direct path to c_T. Compare a unit that holds its value with one that is overwritten.',
    advanced: 'Gates depend on h_{t−1}, so there are also indirect paths through the gates; the full gradient includes them (the Gradients stage computes it exactly). The forget-gate bias, editable in the simulation, controls the default memory span.',
  },
  {
    id: 'attention', group: 'Attention', short: 'Attention', title: 'Self-attention: every position at once',
    equation: String.raw`A_{ts}=\operatorname{softmax}_{s\le t}\!\Big(\frac{\mathbf q_t\cdot\mathbf k_s}{\sqrt d}\Big),\quad \mathbf o_t=\sum_{s\le t}A_{ts}\mathbf v_s,\quad \mathbf z=W_y(\mathbf o_T+\mathbf e_T)+\mathbf b_y`,
    beginner: 'Instead of passing a memory along, each position looks back at all earlier positions and takes a weighted mix of them. Position T can read position 1 directly.',
    intermediate: 'Rows are query positions, columns are the positions they read. The grey triangle is the causal mask. Only the last row feeds this model’s prediction; earlier rows are computed but unused, which matters when interpreting them.',
    advanced: 'Positions are distinguishable only through the sinusoidal position vectors added to the embeddings; without them attention is permutation-invariant and could not tell which symbol came first. Cost per layer is O(T²d), versus O(Td²) for a recurrent layer.',
  },
  {
    id: 'parallel', group: 'Compute', short: 'Parallelism', title: 'Sequence parallelism and cost',
    equation: String.raw`\text{RNN: } T\ \text{dependent steps},\ O(Td^2)\qquad \text{attention: } O(1)\ \text{dependent steps},\ O(T^2d)`,
    beginner: 'An RNN must read the sequence in order, like a person reading a sentence word by word. Attention can process all positions at once, like seeing the whole page.',
    intermediate: 'Each block in the schedule is one operation; blocks in the same column can run at the same time. Move the length slider: RNN depth grows with T, attention depth stays at 5 stages while its score matrix grows as T².',
    advanced: 'Parallel training is the main practical reason transformers replaced RNNs. At inference, a causal transformer generating token by token is sequential again, and caches K and V (the KV-cache, module 10) to avoid recomputing them; an RNN only carries its fixed-size state.',
  },
  {
    id: 'gradients', group: 'Learning', short: 'Gradients', title: 'Gradient propagation through time',
    equation: String.raw`\frac{\partial L}{\partial \mathbf x_1}=\frac{\partial L}{\partial \mathbf h_T}\Big(\prod_{t=2}^{T}\frac{\partial \mathbf h_t}{\partial \mathbf h_{t-1}}\Big)\frac{\partial \mathbf h_1}{\partial \mathbf x_1},\qquad \frac{\partial \mathbf h_t}{\partial \mathbf h_{t-1}}=\operatorname{diag}(1-\mathbf h_t^2)\,W_h`,
    beginner: 'To learn, a model needs a signal telling the first input how to change. In an RNN that signal is multiplied by a factor at every step back in time, so it can shrink to nothing.',
    intermediate: 'The chart shows ‖∂/∂x_t‖ on a log scale, computed exactly by backpropagation through time. Turn the simulation’s recurrent scale down to watch the RNN’s curve collapse, or up to watch it grow; raise the LSTM forget bias to keep its signal alive.',
    advanced: 'Choose the objective: the loss gradient is the training signal and is ≈ 0 for a confident trained model; the target-logit gradient measures sensitivity. A gradient is local: the trained RNN’s output is insensitive to tiny changes of x₁ (saturated) yet flips if x₁ is replaced, which the Caveat stage shows with occlusion.',
  },
  {
    id: 'caveat', group: 'Learning', short: 'Caveat', title: 'Attention weights are not an explanation',
    equation: String.raw`\text{attention } A_{T,s}\ \ \text{vs}\ \ \text{occlusion } \Delta p_s\ \ \text{vs}\ \ \text{gradient}\ \Big\lVert\tfrac{\partial z_y}{\partial \mathbf x_s}\Big\rVert`,
    beginner: 'It is tempting to read attention weights as “what the model thinks is important”. Sometimes they agree with what actually matters; sometimes they do not. Here you can test it.',
    intermediate: 'Occlusion replaces one symbol and measures how much the correct answer’s probability drops: a direct test of influence. The header reports whether attention and occlusion pick the same top position, and the correlation of the two profiles (1 = same shape, 0 = unrelated, negative = opposite).',
    advanced: 'Attention weights ignore value vectors, the output projection and the residual path. A high weight on a small value vector changes nothing, and the last token reaches the output through the residual with zero attention. See Jain & Wallace (2019) and Wiegreffe & Pinter (2019) for the debate.',
  },
  {
    id: 'lengths', group: 'Evidence', short: 'Lengths', title: 'Real training: accuracy versus dependency distance',
    equation: String.raw`\text{accuracy}(T)=\tfrac1N\textstyle\sum_n \mathbb 1[\arg\max_k z_k^{(n)}=x_1^{(n)}],\quad N=400\ \text{per length}`,
    beginner: 'These are results from actually training the three models, not a simulation. Trained on a mix of short and long sequences, all three succeed. Trained only on long sequences, the plain RNN never learns anything.',
    intermediate: 'Chance is 25% (four symbols). In the long-only regime the RNN’s loss stays at ln 4 ≈ 1.386, exactly the loss of guessing. The LSTM learns the long task but fails on short sequences it never saw.',
    advanced: 'The RNN is not incapable: the mixed-regime RNN solves length 30. What fails is learning, because the gradient reaching x₁ across 14+ steps is too weak. Curricula, gating (LSTM/GRU), orthogonal initialization and attention all address this.',
  },
  {
    id: 'compare', group: 'Compare', short: 'Compare', title: 'Side by side at one time step',
    equation: String.raw`\text{state at }t:\ \ \mathbf h_t\ (\text{RNN}),\ \ (\mathbf c_t,\mathbf h_t)\ (\text{LSTM}),\ \ A_{t,\le t}\ (\text{attention})`,
    beginner: 'Step through time and watch what each model knows at that moment: a squashed memory, a gated memory, or direct access to everything so far.',
    intermediate: 'The three panels share the time cursor and the sequence. The summary table compares prediction, sequential steps, path length from x₁ to the output, and gradient reaching x₁.',
    advanced: 'Path length (1 for attention, T − 1 for recurrence) bounds how directly signals can flow; it is a property of the architecture. Whether a trained model uses that path is an empirical question, answered here by the Lengths and Caveat stages.',
  },
]

export const OBJECTIVES = [
  'Unroll an RNN and compute one hidden-state update by hand.',
  'Explain the role of the input, forget and output gates and compute an LSTM cell update.',
  'Compute causal self-attention weights and explain why position needs to be encoded.',
  'Explain vanishing and exploding gradients from the product of Jacobians, and why the LSTM cell path helps.',
  'Compare sequential depth, path length, compute and memory of recurrence and attention.',
  'Test whether attention weights explain a prediction, using occlusion and gradients.',
]

export const MISCONCEPTIONS = [
  { myth: 'RNNs cannot remember long sequences.', truth: 'The mixed-regime RNN here remembers across 29 steps perfectly. The difficulty is learning long dependencies from gradients, not representing them.' },
  { myth: 'Attention weights show what the model is thinking.', truth: 'They show one ingredient of one layer. Value vectors, residual paths and later layers also matter. Check with an intervention such as occlusion before interpreting.' },
  { myth: 'Transformers are strictly better because attention sees everything.', truth: 'Attention costs O(T²) memory and needs explicit position information; an RNN carries a fixed-size state. Recent state-space and linear-recurrent models revisit recurrence for long contexts.' },
  { myth: 'A gradient of zero means the input does not matter.', truth: 'A gradient is a local slope. The trained RNN has a near-zero gradient for x₁ while replacing x₁ changes its answer completely.' },
]

export interface QuizQ { q: string; options: string[]; answer: number; explain: string }
export const QUIZ: QuizQ[] = [
  { q: 'Why can the T hidden-state updates of an RNN not be computed in parallel?', options: ['They use different weights', 'Each h_t needs h_{t−1}', 'tanh is not parallelizable', 'GPUs cannot run loops'], answer: 1,
    explain: 'h_t = tanh(W_x x_t + W_h h_{t−1} + b) depends on the previous state, so the updates form a chain of T dependent steps.' },
  { q: 'An LSTM unit has forget gate f = 1 and input gate i = 0 at every step. What happens to its cell value?', options: ['It decays to 0', 'It stays exactly the same', 'It doubles each step', 'It becomes the input'], answer: 1,
    explain: 'c_t = 1·c_{t−1} + 0·g_t = c_{t−1}: the value is carried unchanged, and so is the gradient along that path.' },
  { q: 'Without positional encodings, what can a single self-attention layer not do on this task?', options: ['Compute softmax', 'Tell which symbol came first', 'Use the last symbol', 'Produce four outputs'], answer: 1,
    explain: 'Without positions, attention treats the input as a set: permuting the earlier symbols gives the same output, so “first” has no meaning.' },
  { q: 'In the long-only regime the RNN’s final training loss is ≈ 1.386. What does that number mean?', options: ['It learned perfectly', 'ln 4: it predicts all four symbols with probability ¼', 'The gradient exploded', 'It memorized the training set'], answer: 1,
    explain: '−log(1/4) = ln 4 ≈ 1.386 is the loss of uniform guessing. The RNN never found a learning signal for x₁.' },
  { q: 'The attention weight on position 3 is 0.9, but replacing symbol 3 does not change the output. Which explanation is possible?', options: ['The attention computation is wrong', 'Position 3’s value vector contributes little to the output', 'Softmax must be broken', 'This cannot happen'], answer: 1,
    explain: 'Weights multiply value vectors. If v₃ is small, or orthogonal to what W_y reads, a large weight has little effect. The output can also arrive through the residual path.' },
]
