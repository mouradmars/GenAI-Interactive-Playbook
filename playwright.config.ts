import { defineConfig } from '@playwright/test'

// End-to-end tests run against the Vite dev server. Set PW_CHROMIUM to use a specific Chromium binary.
export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  use: {
    baseURL: 'http://localhost:5199',
    viewport: { width: 1440, height: 950 },
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
  webServer: { command: 'npx vite --port 5199 --strictPort', url: 'http://localhost:5199', reuseExistingServer: true, timeout: 60_000 },
})
