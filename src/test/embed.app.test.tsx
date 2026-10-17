import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { defaultProj, makeEmbChallenges, project, toySpace } from '../engine/embeddings'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.mock('../modules/embed/Scatter3D', () => ({ default: () => <div>3d scatter</div> }))
vi.setConfig({ testTimeout: 60000 })

const stepNo = () => {
  const el = screen.getAllByText((_, n) => !!n?.textContent?.match(/^step \d+ \/ 11/) && n.tagName === 'SPAN')[0]
  return Number(el.textContent!.match(/^step (\d+)/)![1])
}
const inspector = () => screen.getByLabelText('Mathematical inspector')
async function openModule() {
  window.location.hash = '#embeddings'
  render(<App />)
  await screen.findByText('Tokens become vectors', { selector: 'h1' })
}
const goStep = async (n: number, title: string) => {
  await userEvent.click(screen.getByLabelText(new RegExp(`^Go to step ${n}:`)))
  await screen.findByText(title, { selector: 'h1' })
}

describe('Module 06 · Embedding Space', () => {
  beforeEach(() => { try { localStorage.clear() } catch { /* */ } })
  afterAll(() => { window.location.hash = '' })

  it('opens from the curriculum, labels the source, and steps with buttons and keys', async () => {
    await openModule()
    expect(screen.getAllByText('educational simulation · hand-built vectors').length).toBeGreaterThan(0)
    expect(stepNo()).toBe(1)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(stepNo()).toBe(2)
    await userEvent.click(screen.getByLabelText('Step forward (→)'))
    expect(stepNo()).toBe(3)
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(stepNo()).toBe(2)
    fireEvent.keyDown(window, { key: 'Home' })
    expect(stepNo()).toBe(1)
  })

  it('dot product presets and typed components compute exactly', async () => {
    await openModule()
    await goStep(2, 'Dot product and length')
    await userEvent.click(await screen.findByRole('button', { name: 'perpendicular' }))
    expect(inspector()).toHaveTextContent('a · b0.0000')
    const a0 = screen.getByLabelText('a₁')
    fireEvent.change(a0, { target: { value: '2' } }) // a = [2, 1], b = [-1, 3] → a·b = 1
    expect(inspector()).toHaveTextContent('a · b1.0000')
  })

  it('cosine is unchanged when a is scaled, while the dot product doubles', async () => {
    await openModule()
    await goStep(3, 'Cosine similarity')
    await userEvent.click(await screen.findByRole('button', { name: 'acute, different lengths' }))
    const before = inspector().textContent!
    const cosBefore = before.match(/cos θ(-?[\d.]+)/)![1]
    const dotBefore = Number(before.match(/a · b(-?[\d.]+)/)![1])
    fireEvent.change(screen.getByLabelText(/^scale a by/), { target: { value: '2' } })
    const after = inspector().textContent!
    expect(after.match(/cos θ(-?[\d.]+)/)![1]).toBe(cosBefore)
    expect(Number(after.match(/a · b(-?[\d.]+)/)![1])).toBeCloseTo(2 * dotBefore, 3)
  })

  it('similarity matrix responds to removing and adding words', async () => {
    await openModule()
    await goStep(4, 'Similarity matrix')
    expect(await screen.findByText(/cosine similarity matrix · 10 × 10/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Remove lemon' }))
    expect(screen.getByText(/cosine similarity matrix · 9 × 9/)).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Add a word'), 'girl')
    await userEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(screen.getByText(/cosine similarity matrix · 10 × 10/)).toBeInTheDocument()
    expect(screen.getAllByText('0.75').length).toBeGreaterThan(0) // cos(king, queen) = 3 / (2·2)
  })

  it('neighbours are computed on the trained space and the metric can change', async () => {
    await openModule()
    await goStep(5, 'Nearest neighbours in full dimension')
    expect(await screen.findByText(/5 nearest to “paris” by cosine similarity/)).toBeInTheDocument()
    expect(screen.getAllByText('trained model · PPMI + SVD on a corpus').length).toBeGreaterThan(0)
    await userEvent.click(screen.getAllByRole('radio', { name: 'dot product' })[0])
    expect(screen.getByText(/5 nearest to “paris” by dot product/)).toBeInTheDocument()
  })

  it('the hand-built analogy lands exactly on queen', async () => {
    await openModule()
    await goStep(6, 'Vector arithmetic and analogies')
    expect(await screen.findByText(/“queen” is ranked 1 of 12 \(cos 1.0000\)/)).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText('c'), 'girl')
    await userEvent.selectOptions(screen.getByLabelText('expected (optional)'), 'princess')
    expect(await screen.findByText(/“princess” is ranked 1/)).toBeInTheDocument()
  })

  it('projection stages show faithfulness numbers and switch method', async () => {
    await openModule()
    await goStep(7, 'PCA: the best flat shadow')
    expect(await screen.findByText(/neighbourhood preservation \(k = 5\)/)).toBeInTheDocument()
    expect(screen.getByText(/shown: .*% of the variance/)).toBeInTheDocument()
    await goStep(9, 'What every projection distorts')
    expect(await screen.findByText(/false friends · close on screen, far in reality/)).toBeInTheDocument()
    expect(screen.getByText(/Shepard plot · 24,310 word pairs/)).toBeInTheDocument() // 221 words → 221·220/2 pairs
  })

  it('contextual embeddings: same token, five sentences, five depths', async () => {
    await openModule()
    await goStep(10, 'Contextual embeddings across sentences')
    expect(await screen.findByText(/the 5 occurrences of “·is”/)).toBeInTheDocument()
    expect(screen.getByText('1 · token embedding W_E').closest('section')).toHaveTextContent('mean off-diagonal cos 1.000')
  })

  it('student challenge checks numeric and multiple-choice answers', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Challenge/ }))
    const toy = toySpace()
    const qs = makeEmbChallenges(toy, project(toy, defaultProj), 1)
    const dotQ = qs.find((q) => q.id === 'dot')!
    const box = await screen.findByLabelText('Answer to question 1')
    await userEvent.type(box, String(dotQ.answer))
    const card = box.closest('section')!
    await userEvent.click(within(card).getByRole('button', { name: 'Check' }))
    expect(within(card).getByRole('status')).toHaveTextContent('Correct')
    const an = qs.find((q) => q.id === 'analogy')!
    const aCard = screen.getByText(an.prompt).closest('section')!
    await userEvent.click(within(aCard).getByLabelText(an.options![an.answer]))
    expect(within(aCard).getByRole('status')).toHaveTextContent('Correct')
    expect(screen.getByText(/2 \/ 6 correct/)).toBeInTheDocument()
  })

  it('compare, lesson quiz and export all respond', async () => {
    await openModule()
    await userEvent.click(screen.getByRole('tab', { name: /Compare/ }))
    expect(await screen.findByText(/picture A · .* · PCA/)).toBeInTheDocument()
    expect(screen.getByText(/picture B · .* · UMAP/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    const quiz = (await screen.findByText(/^student questions/)).closest('section')!
    await userEvent.click(within(quiz).getByLabelText('1'))
    expect(quiz).toHaveTextContent('1 / 5 correct')
    fireEvent.keyDown(window, { key: 'e' })
    await userEvent.click(await screen.findByRole('button', { name: 'Classroom report' }))
    expect(screen.getByText(/# Embedding Space · experiment record/)).toBeInTheDocument()
  })
})
