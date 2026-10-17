import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { tbStore } from '../modules/tblock/store'
import { runBlock, vdot } from '../engine/block'
import { cfgOf } from '../modules/tblock/store'
import { makeChallenges } from '../modules/tblock/challenges'
import { calcSheet } from '../modules/tblock/exportSheet'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.setConfig({ testTimeout: 60000 })

const heading = () => screen.getByTestId('tb-heading').textContent
const inspector = () => screen.getByLabelText('Mathematical inspector')
const go = async (n: number) => { await userEvent.click(screen.getByLabelText(new RegExp(`^Go to step ${n}:`))) }
const trace = () => { const s = tbStore.getState(); return runBlock(cfgOf(s), s.params, s.X, s.iv) }
async function open() {
  window.location.hash = '#transformer'
  render(<App />)
  await screen.findByText('Hidden states enter the block', { selector: 'h1' })
}

describe('Module 02 · Inside the Transformer block', () => {
  beforeEach(() => {
    try { localStorage.clear() } catch { /* */ }
    act(() => { const s = tbStore.getState(); s.setArch('gpt'); s.setDims({ T: 4, d: 4, H: 2, dff: 8 }); s.setSeed(1); s.setSentence(0); s.setTab('block'); s.select({ i: 3, j: 1, h: 0, f: 0, u: 0, cell: null }); s.setAct('gelu') })
  })
  afterAll(() => { window.location.hash = '' })

  it('opens on preset A, labelled as a toy simulation, and steps through the timeline', async () => {
    await open()
    expect(screen.getByTestId('toy-badge').textContent).toMatch(/not trained/)
    expect(screen.getByTestId('arch-label').textContent).toMatch(/GPT-style · pre-LN · causal/)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    await waitFor(() => expect(heading()).toBe('Layer normalization (before attention)'))
    await userEvent.click(screen.getByTestId('nav-mhsa'))
    expect(heading()).toBe('Query, Key and Value projections')
    expect(screen.getByTestId('explain-panel').textContent).toMatch(/not literal questions/)
  })

  it('dot product: the formula, the score cell and the synchronized inspector chain agree with the engine', async () => {
    await open()
    await go(4)
    await screen.findByTestId('dot-formula')
    const t = trace(), h = t.heads[0]
    const s = vdot(h.q[3], h.k[1])
    expect(screen.getByTestId('dot-formula').textContent).toContain(s.toFixed(4).replace('-', '−'))
    const chain = within(inspector()).getByTestId('sync-chain').textContent!
    expect(chain).toContain(h.A[3][1].toFixed(4))
    // clicking a token changes the query everywhere
    await userEvent.click(screen.getByTestId('token-2'))
    await waitFor(() => expect(within(inspector()).getByTestId('sync-chain').textContent).toContain(t.heads[0].S[2][1].toFixed(4).replace('-', '−')))
  })

  it('mask and softmax: future weights are zero; disabling the mask is flagged and changes the weights', async () => {
    await open()
    await go(8)
    const sums = await screen.findByTestId('row-sums')
    expect(sums.textContent!.match(/Σ = 1\.000000/g)!.length).toBe(8)
    expect(trace().heads[0].A[0].slice(1)).toEqual([0, 0, 0])
    await go(7)
    await userEvent.click(await screen.findByTestId('mask-toggle'))
    expect(screen.getByText(/no longer causal self-attention/)).toBeTruthy()
    expect(trace().heads[0].A[0][1]).toBeGreaterThan(0)
    expect(screen.getByTestId('arch-label').textContent).toMatch(/causal/)
    act(() => tbStore.getState().setMaskOff(false))
  })

  it('editing a weight in the inspector recomputes downstream values', async () => {
    await open()
    await go(3)
    await userEvent.click(await screen.findByTestId('mat-Wq-0-0'))
    const before = trace().Q[0][0]
    fireEvent.change(within(inspector()).getByTestId('tb-edit'), { target: { value: '2.5' } })
    expect(tbStore.getState().params.Wq[0][0]).toBe(2.5)
    expect(trace().Q[0][0]).not.toBe(before)
    await userEvent.click(within(inspector()).getByText('reset all edits'))
    expect(tbStore.getState().paramsEdited).toBe(false)
  })

  it('residual and MLP experiments: disabling attention makes H = X; disabling a unit zeroes it', async () => {
    await open()
    await go(13)
    await userEvent.click(await screen.findByTestId('disable-1'))
    const t = trace()
    expect(t.H).toEqual(t.X)
    act(() => tbStore.getState().setIv({ attnScale: 1 }))
    await go(15)
    await userEvent.click(await screen.findByTestId('unit-3'))
    expect(trace().act.every((r) => r[3] === 0)).toBe(true)
    await go(18)
    expect((await screen.findByTestId('token-trace')).textContent).toMatch(/= Y \(output\)/)
  })

  it('architecture B reorders the block and its stage list', async () => {
    await open()
    await userEvent.click(screen.getByRole('radio', { name: /B · BERT post-LN/ }))
    expect(screen.getByTestId('arch-label').textContent).toMatch(/BERT-style · post-LN · bidirectional/)
    expect(screen.getAllByLabelText(/^Go to step \d+:/).length).toBe(17)
    await userEvent.click(screen.getByTestId('nav-add1'))
    expect(heading()).toBe('Residual addition, then LayerNorm')
    trace().H.forEach((r) => expect(r.reduce((a, b) => a + b, 0) / r.length).toBeCloseTo(0, 10))
  })

  it('compare, heads (ablation), stack and advanced tabs render real computations', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Compare/ }))
    expect(screen.getByTestId('arch-diff').textContent).toMatch(/pre-LN/)
    await userEvent.click(screen.getByRole('tab', { name: /Heads/ }))
    expect(screen.getByTestId('ablation-table').textContent).toMatch(/head 1/)
    await userEvent.click(screen.getByRole('tab', { name: /Stack/ }))
    expect(screen.getByTestId('stack-features').textContent).toMatch(/block 3/)
    await userEvent.click(screen.getByRole('tab', { name: /Advanced/ }))
    expect(screen.getByTestId('dep-mlp')).toBeTruthy()
    await userEvent.click(screen.getByRole('tab', { name: /KV cache/ }))
    expect(screen.getByTestId('kv-check').textContent).toMatch(/✓/)
    await userEvent.click(screen.getByRole('tab', { name: /Numerical precision/ }))
    expect(screen.getByTestId('precision-table').textContent).toMatch(/bf16/)
  })

  it('real model: Observatory-Nano attention and next-token prediction', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Real model/ }))
    expect(screen.getByText('real trained model · Observatory-Nano')).toBeTruthy()
    expect(screen.getByTestId('real-tokens').textContent).toContain('<bos>')
    expect(screen.getByTestId('real-next').children.length).toBe(8)
  })

  it('challenges: a correct answer is accepted; verify jumps to the simulator', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Challenges/ }))
    const s = tbStore.getState()
    const q = makeChallenges(1, { cfg: cfgOf(s), params: s.params, X: s.X, tokens: s.tokens }).find((c) => c.id === 'dot')!
    const card = screen.getByTestId('ch-dot')
    await userEvent.type(within(card).getByRole('textbox'), (q.answer as number).toFixed(3))
    await userEvent.click(within(card).getByText('Check answer'))
    expect(within(card).getByText('correct')).toBeTruthy()
    expect(screen.getByTestId('tb-score').textContent).toMatch(/^1 \/ 10/)
    await userEvent.click(within(card).getByText('Verify in the simulator'))
    await waitFor(() => expect(heading()).toBe('One query–key dot product'))
  })

  it('lessons run demos; the calculation sheet contains every tensor', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Lessons/ }))
    await userEvent.click(screen.getByTestId('demo-l2-3'))
    await waitFor(() => expect(heading()).toBe('The attention mask'))
    const s = tbStore.getState()
    const md = calcSheet(trace(), s.tokens, s.arch, s.sel)
    for (const h of ['## Input X', '## Q', '### A = softmax', '## Concat', '## Y (H + MLP)', 'Selected chain']) expect(md).toContain(h)
  })
})
