import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { NGramModel, makeChallenges } from '../engine/ngram'
import { NGRAM_EXAMPLES } from '../modules/ngram/examples'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.setConfig({ testTimeout: 30000 })

const stepNo = () => {
  const el = screen.getAllByText((_, n) => !!n?.textContent?.match(/^step \d+ \/ 11/) && n.tagName === 'SPAN')[0]
  return Number(el.textContent!.match(/^step (\d+)/)![1])
}
async function openModule() {
  window.location.hash = '#ngram' // deep link, as a teacher would share it
  render(<App />)
  await screen.findByText('Corpus → padded sentences', { selector: 'h1' })
}

describe('Module 08 · N-gram lab', () => {
  beforeEach(() => { try { localStorage.clear() } catch { /* */ } })
  afterAll(() => { window.location.hash = '' })

  it('opens from the curriculum and steps through the guided pipeline', async () => {
    await openModule()
    expect(stepNo()).toBe(1)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(stepNo()).toBe(2)
    await userEvent.click(screen.getByLabelText('Step forward (→)'))
    expect(stepNo()).toBe(3)
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(stepNo()).toBe(2)
    fireEvent.keyDown(window, { key: 'Home' })
    expect(stepNo()).toBe(1)
    await userEvent.click(screen.getByLabelText(/Go to step 11:/))
    expect(await screen.findByText('Comparing all methods', { selector: 'h1' })).toBeInTheDocument()
  })

  it('shows textbook MLE values and the infinite perplexity of an unseen bigram', async () => {
    await openModule()
    await userEvent.click(screen.getByLabelText(/Go to step 4:/))
    expect(await screen.findByText('P(am|i) = 2/3 =', { exact: false })).toBeInTheDocument()
    await userEvent.click(screen.getByLabelText(/Go to step 6:/))
    expect(await screen.findByText(/“i like” → P_ML = 0/)).toBeInTheDocument()
    expect(screen.getByLabelText('Mathematical inspector')).toHaveTextContent('PP = 2ᴴ∞')
  })

  it('editing k changes the add-k probabilities live', async () => {
    await openModule()
    await userEvent.click(screen.getByLabelText(/Go to step 7:/))
    const slider = await screen.findByLabelText(/^k = 1.00/)
    expect(screen.getAllByText('0.2000').length).toBeGreaterThan(0) // (2+1)/(3+12)
    fireEvent.change(slider, { target: { value: '0' } })
    expect(await screen.findByLabelText(/^k = 0.00 \(= MLE\)/)).toBeInTheDocument()
    expect(screen.getAllByText('0.6667').length).toBeGreaterThan(0)
  })

  it('editing the corpus re-fits the model', async () => {
    await openModule()
    const box = screen.getByLabelText('Training corpus')
    await userEvent.clear(box)
    await userEvent.type(box, 'a b. a c.')
    expect(screen.getByText('predicted tokens N').parentElement).toHaveTextContent('6')
  })

  it('loads an example and the window slider animates counting', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('button', { name: 'Weather reports: when trigrams help' }))
    expect((screen.getByLabelText('Training corpus') as HTMLTextAreaElement).value).toContain('tomorrow will be sunny')
    await userEvent.click(screen.getByLabelText(/Go to step 2:/))
    await screen.findByText(/window at position 1 of/)
    await userEvent.click(screen.getByLabelText('Window forward'))
    expect(screen.getByText(/window at position 2 of/)).toBeInTheDocument()
  })

  it('student challenge checks answers and accepts fractions', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Challenge/ }))
    const q = makeChallenges(new NGramModel(NGRAM_EXAMPLES[0].corpus), 1)
    const mle = q.find((x) => x.id === 'mle')!
    const box = await screen.findByLabelText('Answer to question 2')
    // express the exact answer as a fraction
    const frac = [1, 2, 3, 4, 5, 6].map((d) => [Math.round(mle.answer * d), d]).find(([n, d]) => Math.abs(n / d - mle.answer) < 1e-12)!
    await userEvent.type(box, `${frac[0]}/${frac[1]}`)
    const card = box.closest('section')!
    await userEvent.click(within(card).getByRole('button', { name: 'Check' }))
    expect(within(card).getByRole('status')).toHaveTextContent('Correct')
    expect(screen.getByText(/1 \/ 6 correct/)).toBeInTheDocument()
  })

  it('A/B comparison and lesson quiz respond', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Compare/ }))
    expect(await screen.findByText(/model A · Add-k/)).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText('method', { selector: '#abA-method' }), 'kn')
    expect(screen.getByText(/model A · Interpolated Kneser-Ney/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    const quiz = (await screen.findByText(/^student questions/)).closest('section')!
    await userEvent.click(within(quiz).getByLabelText('Infinity'))
    expect(quiz).toHaveTextContent('1 / 5 correct')
  })
})
