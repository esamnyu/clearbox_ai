import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },

  // Required for SharedArrayBuffer (used by some WASM)
  server: {
    port: 3001,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },

  // Worker configuration
  worker: {
    format: 'es',
  },

  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks: {
          transformers: ['@xenova/transformers'],
        },
      },
    },
  },
});
