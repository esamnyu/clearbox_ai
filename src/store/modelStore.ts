/**
 * Model State Management Store
 * ==============================
 *
 * This file manages the application's model state using Zustand, a lightweight
 * state management library. It's the "brain" that tracks whether the model is
 * loaded, handles errors, and coordinates between the UI and the Web Worker.
 *
 * HOW IT WORKS:
 * 1. React components call useModelStore() to access state and actions
 * 2. Actions like loadModel() update the state and communicate with the worker
 * 3. State changes trigger React re-renders automatically
 *
 * KEY STATE:
 * - status      : Current state ('idle'|'loading'|'ready'|'error')
 * - loadProgress: Download progress 0-100 during model loading
 * - error       : Error message if something went wrong
 * - tokens      : Result of last tokenization (e.g., ["Hello", " world"])
 * - tokenIds    : Token IDs from last tokenization (e.g., [15496, 995])
 *
 * KEY ACTIONS:
 * - initWorker() : Creates the Web Worker (must be called once on app start)
 * - loadModel()  : Downloads and loads the GPT-2 model
 * - tokenize()   : Converts text to tokens using the loaded model
 * - reset()      : Clears errors and allows retry
 *
 * WHY ZUSTAND?
 * - Simpler than Redux, no boilerplate
 * - Works great with TypeScript
 * - Lightweight (~1KB gzipped)
 *
 * @module store/modelStore
 */

import { create } from "zustand";
import * as Comlink from "comlink";
import type {
  ModelWorkerAPI,
  ModelStatus,
  ModelId,
  TokenizationResult,
  LoadProgress,
  GenerationResult,
} from "../engine/types";

/**
 * Shape of the model store state and actions.
 * This interface defines everything available via useModelStore().
 */
interface ModelState {
  // ─────────────────────────────────────────────────────────────────────────
  // STATE
  // ─────────────────────────────────────────────────────────────────────────

  /** Current model status: 'idle' (not loaded), 'loading', 'ready', or 'error' */
  status: ModelStatus;

  /** ID of the currently loaded model (e.g., 'Xenova/gpt2') */
  modelId: ModelId | null;

  /** Download/load progress from 0 to 100 */
  loadProgress: number;

  /** Error message if status is 'error', null otherwise */
  error: string | null;

  /** Tokens from the last tokenize() call (e.g., ["Hello", " world"]) */
  tokens: string[];

  /** Token IDs from the last tokenize() call (e.g., [15496, 995]) */
  tokenIds: number[];

  /** Reference to the Web Worker (set by initWorker) */
  worker: Comlink.Remote<ModelWorkerAPI> | null;

  /**
   * The underlying Worker behind `worker`. Kept because a Comlink proxy cannot
   * be terminated — without a handle to the raw Worker there is no way to free
   * one, and every orphan holds its own copy of transformers.js (and, once
   * loaded, its own ~500 MB GPT-2).
   */
  rawWorker: Worker | null;

  // ─────────────────────────────────────────────────────────────────────────
  // ACTIONS
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Initialize the Web Worker. Idempotent: calling it while a worker is alive
   * is a no-op rather than a second spawn.
   */
  initWorker: () => void;

  /** Terminate the worker and release its Comlink proxy. Safe to call twice. */
  disposeWorker: () => void;

  /** Download and load a model by ID. Updates status and progress during load. */
  loadModel: (modelId: ModelId) => Promise<void>;

  /** Tokenize text using the loaded model. Updates tokens/tokenIds state. */
  tokenize: (text: string) => Promise<TokenizationResult>;

  /** Generate text using the loaded model. Updates generation state. */
  generate: (prompt: string) => Promise<GenerationResult>;

  /** Reset error state to allow retry. Clears error and sets status to 'idle'. */
  reset: () => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  // Initial state
  status: "idle",
  modelId: null,
  loadProgress: 0,
  error: null,
  tokens: [],
  tokenIds: [],
  worker: null,
  rawWorker: null,

  /**
   * Initialize the Web Worker for model inference.
   *
   * Idempotent by design. React StrictMode double-invokes mount effects in dev
   * specifically to surface effects that allocate without cleaning up, and this
   * one used to spawn a fresh Worker on every call with no way to free the old
   * one. The observable damage was worse than a leak: an orphaned worker whose
   * module fetch was aborted still fired `onerror`, and the handler set a
   * GLOBAL `status: 'error'` — so a dead worker could knock the live UI into an
   * error state it never recovered from. That is the "Worker error: [object
   * Event]" seen in the console alongside repeated aborted fetches of
   * worker.ts.
   */
  initWorker: () => {
    if (get().rawWorker) return; // already alive; don't spawn a second

    try {
      const worker = new Worker(
        new URL("../engine/worker.ts", import.meta.url),
        { type: "module" },
      );

      // Handle worker-level errors (script load failures, uncaught exceptions).
      // Only report if THIS worker is still the active one — an orphan being
      // torn down must not clobber the state of its replacement.
      worker.onerror = (event) => {
        if (get().rawWorker !== worker) return;
        console.error("Worker error:", event);
        set({
          status: "error",
          error: `Worker error: ${event.message || "Unknown worker error"}`,
        });
      };

      const wrappedWorker = Comlink.wrap<ModelWorkerAPI>(worker);
      set({ worker: wrappedWorker, rawWorker: worker });
    } catch (err) {
      console.error("Failed to initialize worker:", err);
      set({
        status: "error",
        error: `Failed to initialize worker: ${String(err)}`,
      });
    }
  },

  /**
   * Terminate the worker and release its Comlink proxy.
   *
   * `onerror` is detached first: terminate() can itself surface an error event
   * for an in-flight module fetch, and we do not want a worker we deliberately
   * killed reporting itself as a failure. Safe to call when nothing is running.
   */
  disposeWorker: () => {
    const { worker, rawWorker } = get();
    if (rawWorker) {
      rawWorker.onerror = null;
      rawWorker.terminate();
    }
    // Free the Comlink proxy's message-channel listener; without this the
    // proxy keeps a reference to a port belonging to a terminated worker.
    worker?.[Comlink.releaseProxy]?.();
    set({ worker: null, rawWorker: null });
  },

  /**
   * Download and load a model by ID.
   * Tracks progress and handles errors gracefully.
   */
  loadModel: async (modelId: ModelId) => {
    const { worker } = get();
    if (!worker) {
      set({
        status: "error",
        error: "Worker not initialized. Please refresh the page.",
      });
      return;
    }

    set({ status: "loading", loadProgress: 0, error: null });

    try {
      await worker.loadModel(
        modelId,
        Comlink.proxy((progress: LoadProgress) => {
          set({ loadProgress: progress.progress ?? 0 });
        }),
      );
      set({ status: "ready", modelId, loadProgress: 100 });
    } catch (err) {
      console.error("Model loading failed:", err);
      set({ status: "error", error: String(err) });
    }
  },

  /**
   * Tokenize text using the loaded model.
   */
  tokenize: async (text: string) => {
    const { worker } = get();
    if (!worker) throw new Error("Worker not initialized");

    const result = await worker.tokenize(text);
    set({ tokens: result.tokens, tokenIds: result.tokenIds });
    return result;
  },

  generate: async (prompt: string) => {
    const { worker, status } = get();
    if (!worker || status !== "ready") {
      throw new Error("Worker not initialized or model not ready");
    }

    return await worker.generate(prompt, {
      maxNewTokens: 10,
      outputHiddenStates: true,
      outputAttentions: true,
    });
  },

  /**
   * Reset error state to allow retry.
   */
  reset: () => {
    set({ status: "idle", error: null, loadProgress: 0 });
  },
}));
