import { expect, test } from '@playwright/test'

test.describe('Module 11 · RAG pipeline', () => {
  let errors: string[] = []
  test.beforeEach(async ({ page }) => {
    errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    await page.goto('/#rag')
    await expect(page.getByTestId('rag-heading')).toHaveText('The knowledge base')
  })
  test.afterEach(() => { expect(errors).toEqual([]) })

  test('guided pipeline: step, play/pause, controls change results', async ({ page }) => {
    await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('rag-heading')).toHaveText('Chunking the documents')
    await page.getByLabel('Chunk size').fill('15')
    await expect(page.getByTestId('chunk-stats')).toContainText('chunks')
    await page.getByTestId('pipe-generate').click()
    await expect(page.getByTestId('grounded-answer')).toBeVisible()
    await page.getByRole('button', { name: /Play/ }).click()
    await page.waitForTimeout(1900)
    await page.getByRole('button', { name: /Pause/ }).click()
    await expect(page.getByTestId('rag-heading')).not.toHaveText('Grounded vs ungrounded answers')
  })

  test('lexical gap example, then evaluation and export dialog', async ({ page }) => {
    await page.getByRole('tab', { name: /Lesson/ }).click()
    await page.getByTestId('example-e3').click()
    await expect(page.getByTestId('rag-heading')).toHaveText('Embedding the question')
    await page.getByTestId('pipe-failures').click()
    await expect(page.getByTestId('eval-table')).toContainText('lexical-gap')
    await page.keyboard.press('e')
    await expect(page.getByRole('dialog')).toBeVisible()
  })

  test('phone width has no horizontal page scroll', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForTimeout(300)
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1)
  })
})
