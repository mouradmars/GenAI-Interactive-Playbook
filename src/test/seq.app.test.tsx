import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { makeSeqQuestions } from '../engine/seq'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.mock('../modules/embed/Scatter3D', () => ({ default: () => <div>3d</div> }))
vi.setConfig({ testTimeout: 60000 })

const stepNo = () => {
  const el = screen.getAllByText((_, n) => !!n?.textContent?.match(/^step \d+ \/ 10/) && n.tagName === 'SPAN')[0]
  return Number(el.textContent!.match(/^step (\d+)/)![1])
}
const inspector = () => screen.getByLabelText('Mathematical inspector')
async function openModule() {
  window.location.hash = '#rnn'
  render(<App />)
  await screen.findByText('One task, three architectures', { selector: 'h1' })
}
const goStep = async (n: number, title: string) => {
  await userEvent.click(screen.getByLabelText(new RegExp(`^Go to step ${n}:`)))
  await screen.findByText(title, { selector: 'h1' })
}

describe('Module 09 · RNN, LSTM & Transformer', () => {
  beforeEach(() => { try { localStorage.clear() } catch { /* */ } })
  afterAll(() => { window.location.hash = '' })

  it('opens with trained models labelled, and the stage timeline steps', async () => {
    await openModule()
    expect(screen.getAllByText(/trained models · mixed/).length).toBeGreaterThan(0)
    expect(stepNo()).toBe(1)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(stepNo()).toBe(2)
    await userEvent.click(screen.getByLabelText('Step forward (→)'))
    expect(stepNo()).toBe(3)
    fireEvent.keyDown(window, { key: 'Home' })
    expect(stepNo()).toBe(1)
  })

  it('switching to random weights relabels everything as a simulation', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('radio', { name: 'Random weights (simulation)' }))
    expect(screen.getAllByText(/educational simulation/).length).toBeGreaterThan(0)
    expect(inspector()).toHaveTextContent('Simulation: seeded random weights')
  })

  it('editing the sequence re-runs all three models', async () => {
    await openModule()
    const box = screen.getByLabelText(/input sequence/)
    await userEvent.clear(box)
    await userEvent.type(box, 'dabca{enter}')
    expect(inspector()).toHaveTextContent('target “d” · T = 5')
    expect(screen.getAllByText(/says “d”/).length).toBe(3) // all three trained models recall it
  })

  it('the time cursor steps, plays and resets independently of the stage timeline', async () => {
    await openModule()
    await goStep(2, 'An RNN unrolled through time')
    expect(await screen.findByText('11', { selector: 'span.text-amber' })).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('First time step'))
    expect(screen.getByText('1', { selector: 'span.text-amber' })).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('Next time step'))
    expect(screen.getByText('2', { selector: 'span.text-amber' })).toBeInTheDocument()
    expect(stepNo()).toBe(2)
    fireEvent.keyDown(window, { key: ']' })
    expect(screen.getByText('3', { selector: 'span.text-amber' })).toBeInTheDocument()
  })

  it('LSTM stage shows the exact cell update', async () => {
    await openModule()
    await goStep(3, 'LSTM gates at one time step')
    expect(await screen.findByText(/the update for this unit, term by term/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /LSTM cell, unit 1, time 11/ })).toBeInTheDocument()
  })

  it('gradient stage: objective switch and simulation scale change the curves', async () => {
    await openModule()
    await goStep(7, 'Gradient propagation through time')
    await userEvent.click(await screen.findByRole('radio', { name: 'training loss −log p' }))
    expect(screen.getByText(/L = −log p\(target\)/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('radio', { name: 'Random weights (simulation)' }))
    const read = () => screen.getByText('signal reaching x₁ relative to x_T').closest('section')!.textContent!
    const before = read()
    fireEvent.change(screen.getByLabelText(/recurrent \/ attention scale/), { target: { value: '0.3' } })
    expect(read()).not.toBe(before)
  })

  it('caveat stage reports attention vs occlusion agreement', async () => {
    await openModule()
    await goStep(8, 'Attention weights are not an explanation')
    expect(await screen.findByText(/attention ↔ occlusion: same top position/)).toBeInTheDocument()
  })

  it('lengths stage re-evaluates the trained models in the browser', async () => {
    await openModule()
    await goStep(9, 'Real training: accuracy versus dependency distance')
    await userEvent.click(await screen.findByRole('button', { name: 'Evaluate 200 new sequences' }))
    expect(screen.getAllByText(/\(Python: /).length).toBe(6)
  })

  it('student challenge checks a numeric and a multiple-choice answer', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Challenge/ }))
    const qs = makeSeqQuestions(1)
    const box = await screen.findByLabelText('Answer to question 1')
    await userEvent.type(box, qs[0].answer.toFixed(4))
    const card = box.closest('section')!
    await userEvent.click(within(card).getByRole('button', { name: 'Check' }))
    expect(within(card).getByRole('status')).toHaveTextContent('Correct')
    const steps = qs.find((q) => q.id === 'steps')!
    const sCard = screen.getByText(steps.prompt).closest('section')!
    await userEvent.click(within(sCard).getByLabelText(steps.options![steps.answer]))
    expect(within(sCard).getByRole('status')).toHaveTextContent('Correct')
    expect(screen.getByText(/2 \/ 7 correct/)).toBeInTheDocument()
  })

  it('example, compare, quiz and export respond', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    const quiz = (await screen.findByText(/^student questions/)).closest('section')!
    await userEvent.click(within(quiz).getByLabelText('Each h_t needs h_{t−1}'))
    expect(quiz).toHaveTextContent('1 / 5 correct')
    await userEvent.click(screen.getAllByRole('button', { name: 'Open this example' })[0]) // long-only regime
    expect(await screen.findByText('Real training: accuracy versus dependency distance', { selector: 'h1' })).toBeInTheDocument()
    expect(screen.getAllByText(/trained models · long/).length).toBeGreaterThan(0)
    await userEvent.click(screen.getByRole('tab', { name: /Compare/ }))
    expect(await screen.findByText('configuration A')).toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'e' })
    await userEvent.click(await screen.findByRole('button', { name: 'Classroom report' }))
    expect(screen.getByText(/# RNN, LSTM & Transformer · experiment record/)).toBeInTheDocument()
  })
})
