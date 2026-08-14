// vitest/config re-exports Vite's defineConfig widened with the `test` key, so
// one config file serves both the build and the test run.
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from /<repo>/, so assets must be requested
// from there rather than from the domain root. Vite's default base of '/' emits
// /assets/... and every request 404s once deployed.
const BASE = '/compute-visualizer/'

export default defineConfig({
  base: BASE,
  plugins: [react()],
  build: {
    // The 3D scene is code-split so the roofline and the numbers render without
    // waiting on three.js. Reported chunk sizes are the check on that.
    reportCompressedSize: true,
  },
  test: {
    // src/core is pure computation with no DOM, so the default node environment
    // is both correct and faster than jsdom.
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
