import { expect, test } from '@playwright/test'

test.describe('Module 02 · inside the Transformer block', () => {
  let errors: string[] = []
  test.beforeEach(async ({ page }) => {
    errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    await page.goto('/#transformer')
    await expect(page.getByTestId('tb-heading')).toHaveText('Hidden states enter the block')
  })
  test.afterEach(() => { expect(errors).toEqual([]) })

  test('attention microscope: click a score cell, every view follows', async ({ page }) => {
    await page.getByTestId('nav-mhsa').click()
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('tb-heading')).toHaveText('The attention score matrix')
    await page.getByTestId('mat-S-2-0').click()
    await expect(page.getByTestId('sync-chain')).toContainText('query “sat” → key “The”')
    await expect(page.getByText('query token 3')).toBeVisible()
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('softmax-table')).toBeVisible()
  })

  test('architecture switch, mask toggle and replay', async ({ page }) => {
    await page.getByTestId('nav-mhsa').click()
    for (let k = 0; k < 4; k++) await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('tb-heading')).toHaveText('The attention mask')
    await page.getByTestId('mask-toggle').check()
    await expect(page.getByText(/no longer causal self-attention/)).toBeVisible()
    await page.getByTestId('mask-toggle').uncheck()
    await page.getByRole('button', { name: /Replay step/ }).click()
    await page.getByRole('radio', { name: /B · BERT post-LN/ }).click()
    await expect(page.getByTestId('arch-label')).toHaveText(/BERT-style/)
    await expect(page.getByTestId('tb-heading')).toHaveText('The attention mask')
    await expect(page.getByText(/bidirectional: M is all zeros/)).toBeVisible()
  })

  test('real model, advanced experiments and challenges', async ({ page }) => {
    await page.getByRole('tab', { name: /Real model/ }).click()
    await page.getByTestId('real-input').fill('paris is the capital of')
    await expect(page.getByTestId('real-tokens')).toContainText('capital')
    await page.getByRole('tab', { name: /Advanced/ }).click()
    await expect(page.getByTestId('dep-attn')).toBeVisible()
    await page.getByRole('tab', { name: /Challenges/ }).click()
    await page.getByTestId('ch-mask').getByRole('button', { name: /Show solution/ }).click()
    await expect(page.getByTestId('ch-mask')).toContainText('Common misconception')
  })

  test('presentation mode shows instructor notes; phone width has no horizontal scroll', async ({ page }) => {
    await page.keyboard.press('f')
    await expect(page.getByTestId('notes-bar')).toBeVisible()
    await page.keyboard.press('f')
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForTimeout(300)
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1)
  })
})
