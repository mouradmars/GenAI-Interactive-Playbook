import { expect, test } from '@playwright/test'

const TITLES = ['Input tokens', 'Input embeddings and hidden states', 'The Q / K / V projection factory', 'Query–key dot products', 'Why scale by √dₖ?', 'Masking: causal, padding and custom',
  'The softmax microscope', 'The attention heatmap', 'Weighted value aggregation', 'Multi-head attention explorer', 'Concatenating the heads', 'The output projection', 'The final attention output']

test.describe('Module 03 · Attention', () => {
  let errors: string[] = []
  test.beforeEach(async ({ page }) => {
    errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text()) })
    await page.goto('/#attention')
    await expect(page.getByTestId('attn-heading')).toHaveText('Input tokens')
  })
  test.afterEach(() => { expect(errors).toEqual([]) })

  test('steps through all thirteen operations with the keyboard, then play / pause / reset', async ({ page }) => {
    for (const t of TITLES.slice(1)) {
      await page.keyboard.press('ArrowRight')
      await expect(page.getByTestId('attn-heading')).toHaveText(t)
    }
    await page.keyboard.press('Home')
    await expect(page.getByTestId('attn-heading')).toHaveText(TITLES[0])
    await page.getByRole('button', { name: /^Play/ }).first().click()
    await page.waitForTimeout(2100)
    await page.getByRole('button', { name: /^Pause/ }).first().click()
    await expect(page.getByTestId('attn-heading')).not.toHaveText(TITLES[0])
    await page.getByLabel('Reset (Home)').click()
    await expect(page.getByTestId('attn-heading')).toHaveText(TITLES[0])
  })

  test('editing a value weight keeps the heatmap identical; editing a query weight changes it', async ({ page }) => {
    await page.getByTestId('navstep-heatmap').click()
    const cell = page.getByTestId('heatmap-3-1')
    const before = await cell.textContent()
    await page.getByTestId('navstep-qkv').click()
    await page.getByTestId('try-v').click()
    await expect(page.getByTestId('diff-A')).toHaveText('unchanged')
    await page.getByTestId('navstep-heatmap').click()
    await expect(page.getByTestId('heatmap-3-1')).toHaveText(before!)
    await page.getByTestId('navstep-qkv').click()
    const w = page.getByTestId('edit-Wq-0-0')
    await w.fill('-1.5')
    await w.press('Enter')
    await page.getByTestId('navstep-heatmap').click()
    await expect(page.getByTestId('heatmap-3-1')).not.toHaveText(before!)
    // validation
    await page.getByTestId('navstep-qkv').click()
    await page.getByTestId('edit-Wq-0-1').fill('abc')
    await page.getByTestId('edit-Wq-0-1').press('Enter')
    await expect(page.getByRole('alert')).toContainText('Enter a number')
  })

  test('three synchronized microscopes respond to one heatmap click', async ({ page }) => {
    await page.getByTestId('tab-microscopes').click()
    await page.getByTestId('sync-heatmap-2-0').click()
    await expect(page.getByTestId('sync-steps')).toContainText('token 3 “the” is the query')
    await expect(page.getByTestId('sync-steps')).toContainText('token 1 “cat” is the key')
    await expect(page.getByTestId('sync-steps')).toContainText('0.7000')
    await expect(page.getByTestId('substitution')).toContainText('0.3393')
    await page.getByTestId('edge-3-1').click()
    await expect(page.getByTestId('sync-steps')).toContainText('token 4 “mouse” is the query')
  })

  test('follow one token plays, experiments run, presentation mode and export work', async ({ page }) => {
    await page.getByTestId('tab-follow').click()
    await page.getByTestId('follow-play').click()
    await page.waitForTimeout(2200)
    await expect(page.getByTestId('follow-status')).not.toContainText('step 1 /')
    await page.getByTestId('tab-experiments').click()
    await page.getByTestId('run-causal').click()
    await expect(page.getByTestId('exp-result-causal')).toContainText('Output rows that changed: 1, 2, 3')
    await page.getByTestId('tab-guided').click()
    await page.keyboard.press('f')
    await expect(page.getByTestId('attn-notes-bar')).toBeVisible()
    await page.keyboard.press('t')
    await page.keyboard.press('e')
    await expect(page.getByRole('dialog')).toBeVisible()
  })
})
