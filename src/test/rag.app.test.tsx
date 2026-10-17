import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App'
import { ragStore } from '../modules/rag/store'
import { makeChallenges } from '../modules/rag/challenges'
import { buildPipeline, embedQuery } from '../engine/rag'
import { docsOf, queryText } from '../modules/rag/store'

vi.mock('../components/three', () => ({ EmbeddingCloud: () => <div>3d</div>, ResidualTower: () => <div>3d</div> }))
vi.setConfig({ testTimeout: 60000 })

const heading = () => screen.getByTestId('rag-heading').textContent
const go = async (n: number) => { await userEvent.click(screen.getByLabelText(new RegExp(`^Go to step ${n}:`))) }
async function open() {
  window.location.hash = '#rag'
  render(<App />)
  await screen.findByText('The knowledge base', { selector: 'h1' })
}

describe('Module 11 · RAG pipeline', () => {
  beforeEach(() => { act(() => { const s = ragStore.getState(); s.setCorpus('course'); s.resetCfg(); s.setTab('guided'); s.setChallengeSeed(1) }) })
  afterAll(() => { window.location.hash = '' })

  it('opens labelled honestly and steps through the pipeline', async () => {
    await open()
    expect(screen.getByTestId('rag-sim-badge').textContent).toMatch(/no LLM/)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    await waitFor(() => expect(heading()).toBe('Chunking the documents'))
    await userEvent.click(screen.getByTestId('pipe-retrieve'))
    expect(heading()).toBe('Retrieving the top-k chunks')
    fireEvent.keyDown(window, { key: 'Home' })
    await waitFor(() => expect(heading()).toBe('The knowledge base'))
  })

  it('chunk size changes the number of chunks; small fixed windows split the evidence', async () => {
    await open()
    await go(2)
    const stats = await screen.findByTestId('chunk-stats')
    const n0 = parseInt(stats.textContent!)
    act(() => ragStore.getState().setCfg((c) => ({ ...c, chunk: { size: 12, overlap: 0, strategy: 'fixed' } })))
    await waitFor(() => expect(parseInt(screen.getByTestId('chunk-stats').textContent!)).toBeGreaterThan(n0))
    expect(screen.getByTestId('chunk-stats').textContent).toMatch(/evidence split/)
  })

  it('retrieval method switch re-ranks; the ranking table marks gold evidence', async () => {
    await open()
    await go(7)
    const table = await screen.findByTestId('ranking-table')
    expect(table.textContent).toContain('★')
    await userEvent.click(screen.getByRole('radio', { name: 'BM25' }))
    expect(ragStore.getState().cfg.retrieve.method).toBe('bm25')
  })

  it('grounded vs closed-book answers; an unanswerable question abstains', async () => {
    await open()
    await go(10)
    expect((await screen.findByTestId('grounded-answer')).textContent).toContain('five percentage points')
    expect(screen.getByTestId('closed-answer').textContent).toContain('10%')
    expect(screen.getByTestId('grounded-verdict').textContent).toMatch(/matches/)
    await userEvent.selectOptions(screen.getByLabelText('Question'), 'c6')
    await waitFor(() => expect(screen.getByTestId('grounded-answer').textContent).toMatch(/I don't know/))
  })

  it('query rewriting fixes the lexical gap (example 3)', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Lesson/ }))
    await userEvent.click(screen.getByTestId('example-e3'))
    await waitFor(() => expect(heading()).toBe('Embedding the question'))
    expect(screen.getByTestId('query-steps').textContent).toContain('unknown')
    await go(10)
    await waitFor(() => expect(screen.getByTestId('grounded-answer').textContent).toContain('465'))
  })

  it('citations verify and the evaluation table diagnoses every question', async () => {
    await open()
    await go(11)
    expect((await screen.findByTestId('citations')).textContent).toMatch(/verified/)
    await go(12)
    const tiles = await screen.findByTestId('eval-tiles')
    expect(tiles.textContent).toMatch(/Recall@3/)
    expect(screen.getByTestId('diag-c6').textContent).toMatch(/Correctly abstained/)
    expect(screen.getByTestId('diag-c3').textContent).toMatch(/reasoning/)
  })

  it('custom documents and questions flow through the pipeline', async () => {
    await open()
    await userEvent.type(screen.getByTestId('new-doc-text'), 'The cafeteria serves vegetarian lasagna every Wednesday.')
    await userEvent.click(screen.getByText('Add document'))
    await userEvent.selectOptions(screen.getByLabelText('Question'), 'custom')
    await userEvent.type(screen.getByTestId('custom-question'), 'What does the cafeteria serve on Wednesday?')
    await go(10)
    await waitFor(() => expect(screen.getByTestId('grounded-answer').textContent).toContain('lasagna'))
  })

  it('explore and compare show metrics for both configurations', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Explore/ }))
    expect(screen.getByTestId('explore-metrics').textContent).toMatch(/MRR/)
    await userEvent.click(screen.getByRole('tab', { name: /Compare/ }))
    expect(screen.getByTestId('cmp-metrics-A')).toBeTruthy(); expect(screen.getByTestId('cmp-metrics-B')).toBeTruthy()
    expect(screen.getByTestId('cmp-table').querySelectorAll('tbody tr').length).toBe(7)
  })

  it('challenge: a correct hand computation is scored', async () => {
    await open()
    await userEvent.click(screen.getByRole('tab', { name: /Challenge/ }))
    const s = ragStore.getState(), p = buildPipeline(docsOf(s), s.cfg)
    const q = makeChallenges(1, { pipe: p, q: embedQuery(p.index, queryText(s)), chunk: 0 }).find((x) => x.id === 'rrf')!
    const card = screen.getByTestId('rq-rrf')
    await userEvent.type(within(card).getByRole('textbox'), (q.answer as number).toFixed(5))
    await userEvent.click(within(card).getByText('Check'))
    expect(within(card).getByText('correct')).toBeTruthy()
    expect(screen.getByTestId('rag-score').textContent).toMatch(/^1 \//)
  })
})
