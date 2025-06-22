import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Remove the base URL to fix CORS issues
  // base: 'https://chatisthisreal.study/',
})
