import { expect, test } from '@playwright/test'

test.describe('Module 04 · neural networks & backpropagation', () => {
  test.beforeEach(async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (e) => errors.push(e.message))
    ;(page as unknown as { _errors: string[] })._errors = errors
    await page.goto('/#mlp')
    await expect(page.getByTestId('mlp-heading')).toHaveText('The input enters the network')
  })
  test.afterEach(async ({ page }) => {
    expect((page as unknown as { _errors: string[] })._errors).toEqual([])
  })

  test('forward and backward microscope on the worked example', async ({ page }) => {
    await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('mlp-heading')).toHaveText('Weighted sums of layer 1')
    await expect(page.getByTestId('inspector-numbers')).toContainText('0.3600')
    for (let k = 0; k < 4; k++) await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('net-loss')).toHaveText('0.2112')
    for (let k = 0; k < 4; k++) await page.keyboard.press('ArrowRight')
    await expect(page.getByTestId('sel-grad')).toHaveText('-0.0332')
    await expect(page.getByTestId('bp-value')).toHaveText('-0.0332')
  })

  test('clicking the network selects elements; the chain rule animates', async ({ page }) => {
    await page.getByTestId('neuron-1-1').click()
    await expect(page.getByLabel('Selected neuron')).toContainText('h₂⁽¹⁾')
    await page.getByTestId('edge-2-0-1').click()
    await expect(page.getByLabel('Selected parameter')).toContainText('w₁₂⁽²⁾')
    await page.getByRole('button', { name: 'Animate chain rule' }).click()
    await page.waitForTimeout(2500)
    await expect(page.getByTestId('verify-strip')).toContainText('sum over 1 path')
  })

  test('full iteration, history rewind and training', async ({ page }) => {
    await page.getByLabel(/^Go to micro-step 11:/).click()
    await page.getByTestId('apply-update').click()
    await expect(page.getByTestId('history-status')).toContainText('iteration 1')
    await expect(page.getByTestId('history-status')).toContainText('0.1676')
    await page.getByLabel('Preset').selectOption('xor')
    await page.getByRole('tab', { name: /Training/ }).click()
    await page.getByRole('button', { name: '10 epochs' }).click()
    await expect(page.getByTestId('train-status')).toContainText('epoch 10')
    await page.getByLabel('Rewind to the initial state').click()
    await expect(page.getByTestId('history-status')).toContainText('state 1/11')
    await expect(page.getByTestId('xor-acc-1')).toHaveText('100% correct')
  })

  test('gradient checker passes on the digit-sized network', async ({ page }) => {
    await page.getByLabel('Preset').selectOption('digits')
    await page.getByRole('tab', { name: /Gradient check/ }).click()
    await expect(page.getByTestId('gradcheck-summary')).toContainText('1482/1482 pass')
  })

  test('phone width has no horizontal page scroll', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForTimeout(400)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow).toBeLessThanOrEqual(1)
  })

  test('digits: draw a digit and trace it through the microscope', async ({ page }) => {
    await page.getByRole('tab', { name: /Digits/ }).click()
    await page.getByRole('button', { name: 'Load the trained digit classifier' }).click()
    await expect(page.getByTestId('digits-badge')).toContainText('97.3 %')
    await page.getByRole('button', { name: 'clear' }).click()
    // paint a vertical stroke: a "1"
    for (const i of [3, 11, 19, 27, 35, 43, 51, 59]) { await page.getByTestId(`digit-canvas-${i}`).click(); await page.getByTestId(`digit-canvas-${i}`).click() }
    await expect(page.getByTestId('digit-pred')).toHaveText('1')
    await page.getByTestId('digit-to-microscope').click()
    await expect(page.getByTestId('mlp-heading')).toHaveText('The input enters the network')
    await expect(page.getByTestId('micro-status')).toContainText('/ 15')
  })

  test('vanishing lab and challenge work in the browser', async ({ page }) => {
    await page.getByRole('tab', { name: /Vanishing/ }).click()
    await expect(page.getByTestId('vanish-verdict')).toContainText('vanishing')
    await page.getByRole('button', { name: 'Train both' }).click()
    await expect(page.getByLabel('Deep network training curves')).toBeVisible()
    await page.getByRole('tab', { name: /Challenge/ }).click()
    await page.getByTestId('q-relu').getByRole('textbox').fill('0')
    await page.getByTestId('q-relu').getByRole('button', { name: 'Check' }).click()
    await expect(page.getByTestId('challenge-score')).toContainText('answered')
  })
})
