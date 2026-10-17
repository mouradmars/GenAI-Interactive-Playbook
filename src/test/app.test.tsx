import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d-cloud</div>, ResidualTower: () => <div>3d-tower</div> }))


function currentStep() {
  const el = screen.getAllByText((_, n) => !!n?.textContent?.match(/^step \d+ \/ 23/) && n.tagName === 'SPAN')[0]
  return Number(el.textContent!.match(/^step (\d+)/)![1])
}

vi.setConfig({ testTimeout: 30000 })

describe('Observatory shell + Inside an LLM controls', () => {
  beforeEach(() => { try { localStorage.clear() } catch { /* */ } })
  afterEach(() => { vi.useRealTimers() })

  it('steps forward and backward with buttons and keys', async () => {
    render(<App />)
    expect(currentStep()).toBe(1)
    await userEvent.click(screen.getByLabelText('Step forward (→)'))
    expect(currentStep()).toBe(2)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(currentStep()).toBe(3)
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(currentStep()).toBe(2)
    fireEvent.keyDown(window, { key: 'Home' })
    expect(currentStep()).toBe(1)
  })

  it('seeks via the timeline and shows the inspector equation', async () => {
    render(<App />)
    await userEvent.click(screen.getByLabelText(/Go to step 23:/))
    expect(currentStep()).toBe(23)
    expect(screen.getByLabelText('Mathematical inspector')).toHaveTextContent('Σ p̃ (after filtering)')
  })

  it('plays automatically and pauses', () => {
    vi.useFakeTimers()
    render(<App />)
    fireEvent.click(screen.getByLabelText('Play (Space)'))
    act(() => { vi.advanceTimersByTime(1700) })
    act(() => { vi.advanceTimersByTime(1700) })
    expect(currentStep()).toBe(3)
    fireEvent.click(screen.getByLabelText('Pause (Space)'))
    act(() => { vi.advanceTimersByTime(5000) })
    expect(currentStep()).toBe(3)
    vi.useRealTimers()
  })

  it('a custom prompt re-runs the real model', async () => {
    render(<App />)
    const input = screen.getByLabelText('Prompt')
    await userEvent.clear(input)
    await userEvent.type(input, 'the cat likes to eat{enter}')
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(await screen.findByRole('listbox', { name: 'tokens' })).toHaveTextContent('·cat')
    await userEvent.click(screen.getByLabelText(/Go to step 22:/))
    expect((await screen.findByText('next-token probabilities · top 12')).closest('section')).toHaveTextContent('·fish')
  })

  it('predict-before-reveal hides the answer until revealed', async () => {
    render(<App />)
    fireEvent.keyDown(window, { key: 'p' })
    await userEvent.click(screen.getByLabelText(/Go to step 22:/))
    expect(await screen.findByText('Which token will the model rate most likely to come next?')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: '·bones' }))
    await userEvent.click(screen.getByRole('button', { name: 'Reveal' }))
    expect(await screen.findByText(/^Correct:/)).toBeInTheDocument()
  })

  it('switches depth, theme and tabs', async () => {
    render(<App />)
    fireEvent.keyDown(window, { key: '3' })
    expect(screen.getByRole('radio', { name: 'advanced' })).toHaveAttribute('aria-checked', 'true')
    fireEvent.keyDown(window, { key: 't' })
    expect(document.documentElement.dataset.theme).toBe('light')
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    expect(screen.getByText('learning objectives')).toBeInTheDocument()
    const quiz = screen.getByText(/^assessment/).closest('section')!
    await userEvent.click(within(quiz).getByLabelText('Attention'))
    expect(quiz).toHaveTextContent('1 / 4 correct')
  })
})
