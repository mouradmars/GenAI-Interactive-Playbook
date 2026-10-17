import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { mlpStore } from '../modules/mlp/store'
import { makeChallenges } from '../modules/mlp/challenges'
import { DIGITS } from '../engine/digits'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.mock('../modules/mlp/Network3D', () => ({ default: () => <div data-testid="network-3d">3d</div> }))
vi.setConfig({ testTimeout: 60000 })

const status = () => screen.getByTestId('micro-status').textContent!
const inspector = () => screen.getByLabelText('Mathematical inspector')
async function openModule() {
  window.location.hash = '#mlp'
  render(<App />)
  await screen.findByText('The input enters the network', { selector: 'h1' })
}
const goMicro = async (n: number) => { await userEvent.click(screen.getByLabelText(new RegExp(`^Go to micro-step ${n}:`))) }

describe('Module 04 · Neural networks & backpropagation', () => {
  beforeEach(() => {
    try { localStorage.clear() } catch { /* */ }
    act(() => { const s = mlpStore.getState(); s.loadPreset('worked'); s.setTab('microscope'); s.setMode('iteration'); s.setView3d(false); s.setTraining(false); s.setChallengeSeed(1) })
  })
  afterAll(() => { window.location.hash = '' })

  it('opens on the worked example and steps through the forward pass with real numbers', async () => {
    await openModule()
    expect(status()).toMatch(/^step 1 \/ 11/)
    expect(screen.getByTestId('net-loss').textContent).toBe('?')
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    await screen.findByText('Weighted sums of layer 1', { selector: 'h1' })
    await waitFor(() => expect(within(inspector()).getByTestId('inspector-numbers').textContent).toContain('0.3600'))
    await userEvent.click(screen.getByLabelText('Next micro-step (→)'))
    expect(status()).toMatch(/^step 3 \/ 11/)
    await goMicro(6)
    await screen.findByText('How wrong is the prediction?', { selector: 'h1' })
    expect(screen.getByTestId('net-loss').textContent).toBe('0.2112')
    fireEvent.keyDown(window, { key: 'Home' })
    expect(status()).toMatch(/^step 1 \/ 11/)
  })

  it('selecting a neuron shows values only once they are computed', async () => {
    await openModule()
    await userEvent.click(screen.getByTestId('neuron-1-0'))
    expect(within(inspector()).getByLabelText('Selected neuron')).toBeTruthy()
    expect(within(inspector()).getAllByText('compute →').length).toBeGreaterThan(0)
    await goMicro(3)
    expect(within(inspector()).getByTestId('sel-activation').textContent).toBe('0.5890')
    expect(within(inspector()).getByTestId('sel-delta').textContent).toBe('?')
    await goMicro(9)
    expect(within(inspector()).getByTestId('sel-delta').textContent).toBe('−0.0332'.replace('−', '-'))
  })

  it('backprop, the path sum and finite differences agree for w₁₁⁽¹⁾', async () => {
    await openModule()
    await goMicro(10)
    expect(within(inspector()).getByTestId('sel-grad').textContent).toBe('-0.0332')
    const strip = screen.getByTestId('verify-strip')
    expect(within(strip).getByTestId('bp-value').textContent).toBe('-0.0332')
    expect(strip.textContent).toContain('sum over 1 path')
    expect(strip.textContent!.match(/-0\.0332/g)!.length).toBe(3)
  })

  it('applies an update, records it, and rewinds', async () => {
    await openModule()
    await goMicro(11)
    await screen.findByText('One step of gradient descent', { selector: 'h1' })
    await waitFor(() => expect(within(inspector()).getByTestId('inspector-numbers').textContent).toContain('0.1676'))
    await userEvent.click(screen.getByTestId('apply-update'))
    expect(screen.getByTestId('history-status').textContent).toMatch(/state 2\/2 · iteration 1 .* loss 0\.1676/)
    expect(status()).toMatch(/^step 1 \/ 11/)
    await userEvent.click(screen.getByLabelText('Rewind to the initial state'))
    expect(screen.getByTestId('history-status').textContent).toMatch(/state 1\/2 · iteration 0 .* loss 0\.2112/)
  })

  it('manual weight edits recompute everything and can be reverted', async () => {
    await openModule()
    await goMicro(6)
    fireEvent.change(within(inspector()).getByTestId('param-input'), { target: { value: '2' } })
    expect(within(inspector()).getByTestId('modified-banner')).toBeTruthy()
    expect(screen.getByTestId('net-loss').textContent).not.toBe('0.2112')
    await userEvent.click(within(inspector()).getByText('Revert'))
    expect(within(inspector()).queryByTestId('modified-banner')).toBeNull()
    expect(screen.getByTestId('net-loss').textContent).toBe('0.2112')
  })

  it('the builder edits the architecture and keeps loss and output compatible', async () => {
    await openModule()
    await userEvent.click(screen.getByLabelText('Add neuron to layer 1'))
    expect(screen.getByText(/architecture · 13 parameters/)).toBeTruthy()
    await userEvent.selectOptions(screen.getByLabelText('Activation of layer 2'), 'linear')
    expect((screen.getByLabelText('Loss function') as HTMLSelectElement).value).toBe('mse')
    await userEvent.selectOptions(screen.getByLabelText('Loss function'), 'bce')
    expect((screen.getByLabelText('Activation of layer 2') as HTMLSelectElement).value).toBe('sigmoid')
    expect(screen.getByRole('status').textContent).toMatch(/sigmoid/)
    await userEvent.click(screen.getByText('Add hidden layer'))
    expect(screen.getByTestId('layer-row-3')).toBeTruthy()
    expect(status()).toMatch(/\/ 15/)
  })

  it('gradient checker passes for every parameter', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Gradient check/ }))
    expect((await screen.findByTestId('gradcheck-summary')).textContent).toMatch(/9\/9 pass/)
    await userEvent.selectOptions(screen.getByLabelText('Preset'), 'digits')
    expect((await screen.findByTestId('gradcheck-summary')).textContent).toMatch(/1482\/1482 pass/)
  })

  it('the lesson sets up the lab it describes', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    await userEvent.click(screen.getByTestId('lesson-go-2'))
    await screen.findByText('Error flows back into layer 1', { selector: 'h1' })
    expect(within(inspector()).getByLabelText('Selected neuron')).toBeTruthy()
  })

  it('training tab: XOR needs a hidden layer', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Training/ }))
    expect((await screen.findByTestId('xor-acc-0')).textContent).toMatch(/^(0|25|50|75)% correct/)
    expect(screen.getByTestId('xor-acc-1').textContent).toBe('100% correct')
    await userEvent.click(screen.getByText('1 epoch'))
    expect(screen.getByTestId('train-status').textContent).toMatch(/iteration 1 · epoch 1/)
  })

  it('3-D view toggles', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('radio', { name: /3D/ }))
    expect(await screen.findByTestId('network-3d')).toBeTruthy()
  })

  it('digits: trained model labelled, prediction, misclassified browsing and tracing into the microscope', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Digits/ }))
    await userEvent.click(screen.getByText('Load the trained digit classifier'))
    expect((await screen.findByTestId('digits-badge')).textContent).toMatch(/trained model .* 97\.3 %/)
    expect(screen.getByTestId('digit-pred').textContent).toBe(String(DIGITS.y[DIGITS.test[0]]))
    expect(screen.getByTestId('confusion').querySelectorAll('td').length).toBe(100)
    await userEvent.click(screen.getByText('mistake'))
    expect(screen.getByText(/· wrong \(true \d\)/)).toBeTruthy()
    await userEvent.click(screen.getByText('clear'))
    fireEvent.pointerDown(screen.getByTestId('digit-canvas-27'))
    expect(screen.getByText('your drawing')).toBeTruthy()
    await userEvent.click(screen.getByTestId('digit-to-microscope'))
    await screen.findByText('The input enters the network', { selector: 'h1' })
    expect(status()).toMatch(/\/ 15/)
    expect(screen.getByText(/microscope shows your sample/)).toBeTruthy()
    act(() => mlpStore.getState().setParamValue({ kind: 'b', layer: 1, j: 0 }, 5))
    await userEvent.click(screen.getByRole('tab', { name: /Digits/ }))
    expect(screen.getByTestId('digits-badge').textContent).toMatch(/your weights/)
  })

  it('activations: the table and softmax shift invariance', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Activations/ }))
    expect(screen.getByTestId('act-table').textContent).toContain('0.25')
    const before = screen.getByTestId('softmax-0').textContent
    await userEvent.click(screen.getByText('add 1 to every logit'))
    expect(screen.getByTestId('softmax-0').textContent).toBe(before)
  })

  it('vanishing: sigmoid vanishes, ReLU + He is healthy, large weights explode', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Vanishing/ }))
    expect(screen.getByTestId('vanish-verdict').textContent).toMatch(/^vanishing/)
    await userEvent.click(screen.getByText('Deep ReLU, He → healthy'))
    expect(screen.getByTestId('vanish-verdict').textContent).toMatch(/^healthy/)
    await userEvent.click(screen.getByText('Large weights → exploding'))
    expect(screen.getByTestId('vanish-verdict').textContent).toMatch(/^exploding/)
  })

  it('challenge: a correct hand computation is accepted and scored; export includes it', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Challenge/ }))
    const q = makeChallenges(1).find((x) => x.id === 'z')!
    const card = screen.getByTestId('q-z')
    await userEvent.type(within(card).getByRole('textbox'), (q.answer as number).toFixed(2))
    await userEvent.click(within(card).getByText('Check'))
    expect(within(card).getByText('correct')).toBeTruthy()
    expect(screen.getByTestId('challenge-score').textContent).toMatch(/^1 \/ 9 correct/)
    const relu = screen.getByTestId('q-relu')
    await userEvent.type(within(relu).getByRole('textbox'), '42')
    await userEvent.click(within(relu).getByText('Check'))
    expect(within(relu).getByText('not quite')).toBeTruthy()
  })

  it('lesson: going-further segments open the phase 4–5 labs', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    await userEvent.click(screen.getByTestId('further-go-2'))
    await screen.findByText('Vanishing and exploding gradients', { selector: 'h1' })
  })
})
