import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },

  // Required for SharedArrayBuffer (used by some WASM)
  server: {
    port: 3001,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  // Optimize dependencies for worker compatibility
  optimizeDeps: {
    include: ["@huggingface/transformers", "comlink"],
  },

  // Worker configuration
  worker: {
    format: "es",
  },

  test: {
    // Keep Vitest's defaults, plus skip stale agent-worktree copies under
    // .claude/ that would otherwise run every suite several times over.
    exclude: [...configDefaults.exclude, "**/.claude/**"],
  },

  build: {
    target: "esnext",
    // No manualChunks for @huggingface/transformers: it is imported only from
    // src/engine/worker.ts, which Vite emits as its own worker bundle outside
    // the main graph. Naming it as a manual chunk produced a 0-byte
    // `transformers-*.js` and a "Generated an empty chunk" warning on every
    // build — the library was never in the main chunk to split out.
  },
});
