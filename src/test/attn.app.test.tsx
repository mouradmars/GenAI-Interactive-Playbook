import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { attnStore, cfgOf } from '../modules/attn/store'
import { runAttention } from '../engine/attention'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.setConfig({ testTimeout: 60000 })

const heading = () => screen.getByTestId('attn-heading').textContent
const IDS = ['tokens', 'embed', 'qkv', 'dot', 'scale', 'mask', 'softmax', 'heatmap', 'aggregate', 'heads', 'concat', 'proj', 'output']
const go = async (n: number) => { await userEvent.click(screen.getByLabelText(new RegExp(`^Go to step ${n}:`))); await screen.findByTestId(`stage-${IDS[n - 1]}`, {}, { timeout: 3000 }); await waitFor(() => expect(screen.getAllByTestId(/^stage-/)).toHaveLength(1)) }
const trace = () => { const s = attnStore.getState(); return runAttention(s.X, s.params, cfgOf(s)) }
async function open() {
  window.location.hash = '#attention'
  render(<App />)
  await screen.findByText('Input tokens', { selector: 'h1' })
}

describe('Module 03 · Attention', () => {
  beforeEach(() => {
    try { localStorage.clear() } catch { /* */ }
    act(() => { const s = attnStore.getState(); s.resetAll(); s.setTab('guided'); s.clearBaseline(); s.setChallengeSeed(1) })
  })
  afterAll(() => { window.location.hash = '' })

  it('opens labelled as a toy simulation and steps through all thirteen operations', async () => {
    await open()
    expect(screen.getByTestId('attn-sim-badge').textContent).toMatch(/not a trained model/)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    await waitFor(() => expect(heading()).toBe('Input embeddings and hidden states'))
    await userEvent.click(screen.getByTestId('navstep-softmax'))
    expect(heading()).toBe('The softmax microscope')
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    await waitFor(() => expect(heading()).toBe('Masking: causal, padding and custom'))
    fireEvent.change(screen.getByLabelText('Jump to operation'), { target: { value: 'proj' } })
    expect(heading()).toBe('The output projection')
    fireEvent.keyDown(window, { key: 'Home' })
    await waitFor(() => expect(heading()).toBe('Input tokens'))
  })

  it('selecting a token highlights it as the query everywhere and the inspector shows its vector', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tok-0'))
    expect(attnStore.getState().sel.i).toBe(0)
    expect(within(screen.getByTestId('attn-inspector')).getByTestId('substitution').textContent).toMatch(/1\.00/)
  })

  it('Q/K/V factory: clicking an output cell shows its products and sum from the engine', async () => {
    await open()
    await go(3)
    await userEvent.click(screen.getByTestId('g-K-1-1'))
    const t = trace()
    const calc = screen.getByTestId('qkv-calc')
    await waitFor(() => expect(calc.textContent).toContain(t.heads[0].K[1][1].toFixed(4)))
    expect(attnStore.getState().sel.cell).toEqual({ m: 'K', r: 1, c: 1 })
  })

  it('changing V while holding Q and K fixed leaves every attention weight unchanged', async () => {
    await open()
    await go(3)
    const A0 = trace().heads.map((h) => h.A)
    await userEvent.click(screen.getByTestId('try-v'))
    expect(trace().heads.map((h) => h.A)).toEqual(A0)
    expect(screen.getByTestId('diff-A').textContent).toBe('unchanged')
    expect(screen.getByTestId('diff-out').textContent).toMatch(/changed/)
  })

  it('dot-product microscope: clicking a score cell selects the pair and shows q·k', async () => {
    await open()
    await go(4)
    await userEvent.click(screen.getByTestId('g-S-3-1'))
    expect(attnStore.getState().sel).toMatchObject({ i: 3, j: 1 })
    expect(screen.getByTestId('dot-calc').textContent).toContain(trace().heads[0].S[3][1].toFixed(4))
  })

  it('masking: presets, padding keys and a fully masked row (no NaN)', async () => {
    await open()
    await go(6)
    await userEvent.click(screen.getByRole('radio', { name: 'bidirectional (encoder)' }))
    expect(attnStore.getState().causal).toBe(false)
    await userEvent.click(screen.getByRole('radio', { name: 'causal (decoder)' }))
    await userEvent.click(screen.getByTestId('maskrow-1'))
    expect(screen.getByTestId('mask-sentence').textContent).toBe('Token 2 can attend to tokens 1, 2.')
    await userEvent.click(screen.getByTestId('pad-0'))
    expect(await screen.findByTestId('fully-masked')).toHaveTextContent(/query 1/)
    expect(trace().out.flat().every(Number.isFinite)).toBe(true)
  })

  it('softmax rows sum to 1 and the nudge sandbox shows competition without changing the model', async () => {
    await open()
    await go(7)
    expect(screen.getByTestId('row-sum').textContent).toMatch(/1\.000000/)
    const before = trace().heads[0].A
    await userEvent.click(screen.getByTestId('nudge-up-0'))
    expect(screen.getByTestId('nudge-delta-0').textContent).toMatch(/^\+/)
    expect(screen.getByTestId('nudge-delta-1').textContent).toMatch(/^−/)
    expect(trace().heads[0].A).toEqual(before)
  })

  it('three synchronized microscopes: clicking a heatmap cell updates every view', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tab-microscopes'))
    await userEvent.click(screen.getByTestId('sync-heatmap-3-1'))
    const t = trace()
    expect(attnStore.getState().sel).toMatchObject({ i: 3, j: 1 })
    const steps = screen.getByTestId('sync-steps').textContent!
    expect(steps).toContain('token 4 “mouse” is the query')
    expect(steps).toContain('token 2 “chased” is the key')
    expect(steps).toContain(t.heads[0].S[3][1].toFixed(4))
    expect(steps).toContain(t.heads[0].A[3][1].toFixed(4))
    expect(within(screen.getByTestId('pair-table')).getByText(t.heads[0].A[3][1].toFixed(4))).toBeInTheDocument()
    expect(screen.getByTestId('gnode-q-3').querySelector('rect')!.getAttribute('stroke-width')).toBe('2.2')
  })

  it('multi-head: ablating head 2 zeroes its concatenation block; h = 4 gives d_k = 1', async () => {
    await open()
    await go(10)
    await userEvent.click(screen.getByTestId('ablate-1'))
    expect(trace().concat.every((r) => r[2] === 0 && r[3] === 0)).toBe(true)
    await userEvent.click(screen.getByTestId('ablate-1'))
    await userEvent.click(screen.getByRole('radio', { name: 'h = 4' }))
    expect(trace().dk).toBe(1)
    expect(screen.getAllByTestId(/^chamber-\d$/)).toHaveLength(4)
  })

  it('editing only head 1 leaves head 2 identical', async () => {
    await open()
    await go(10)
    const h2 = trace().heads[1]
    await userEvent.click(screen.getByTestId('edit-head1'))
    expect(trace().heads[1].A).toEqual(h2.A)
    expect(trace().heads[1].O).toEqual(h2.O)
    expect(screen.getByTestId('diff-out').textContent).toMatch(/changed/)
  })

  it('follow one token: steps forward and shows the head output from the trace', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tab-follow'))
    for (let k = 0; k < 9; k++) await userEvent.click(screen.getByTestId('follow-next'))
    expect(screen.getByTestId('follow-status').textContent).toMatch(/step 10 \/ 12 · Head output/)
    await waitFor(() => expect(screen.getByTestId('follow-numbers').textContent).toContain(trace().heads[0].O[2][0].toFixed(4)))
  })

  it('experiments run, report numbers and compare with the prediction', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tab-experiments'))
    const card = screen.getByTestId('exp-value')
    await userEvent.click(within(card).getByRole('radio', { name: 'they change' }))
    await userEvent.click(screen.getByTestId('run-value'))
    const res = screen.getByTestId('exp-result-value')
    expect(res.textContent).toMatch(/Attention weights identical/)
    expect(res.textContent).toMatch(/differs from your prediction/)
  })

  it('challenges: check, feedback, hint, retry and progress', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tab-challenge'))
    const card = screen.getByTestId('ach-a3')
    const input = within(card).getByLabelText('Answer to a3')
    await userEvent.type(input, '0, 0, 0, 0{Enter}')
    expect(within(card).getByRole('status').textContent).toMatch(/not yet/)
    await userEvent.click(within(card).getByRole('button', { name: /Hint/ }))
    expect(card.textContent).toMatch(/side by side/)
    await userEvent.click(within(card).getByRole('button', { name: /Retry/ }))
    const prompt = card.textContent!.match(/head 1 outputs \[(.*?)\] and head 2 outputs \[(.*?)\]/)!
    await userEvent.type(within(card).getByLabelText('Answer to a3'), `${prompt[1]}, ${prompt[2]}{Enter}`)
    expect(within(card).getByRole('status').textContent).toMatch(/correct/)
    expect(screen.getByTestId('attn-score').textContent).toBe('1 / 15 solved')
  })

  it('lessons set up their demos, and research mode shows a labelled real model', async () => {
    await open()
    await userEvent.click(screen.getByTestId('tab-lessons'))
    await userEvent.click(screen.getByTestId('demo-l2-3'))
    await waitFor(() => expect(heading()).toBe('Masking: causal, padding and custom'))
    expect(attnStore.getState().sel.i).toBe(1)
    await userEvent.click(screen.getByTestId('tab-research'))
    await userEvent.click(screen.getByRole('radio', { name: 'F · real model' }))
    expect(await screen.findByTestId('real-check')).toHaveTextContent(/reproduces the weights/)
  })

  it('opens the export dialog', async () => {
    await open()
    fireEvent.keyDown(window, { key: 'e' })
    expect(await screen.findByRole('dialog')).toBeInTheDocument()
  })
})
