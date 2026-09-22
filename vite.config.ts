import { defineConfig } from 'vite'

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site under /live-classify/.
  base: command === 'build' ? '/live-classify/' : '/',
  server: {
    watch: { ignored: ['**/.venv/**'] },
  },
}))
