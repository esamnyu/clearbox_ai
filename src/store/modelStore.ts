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

import { create } from 'zustand';
import * as Comlink from 'comlink';
import type { ModelWorkerAPI, ModelStatus, ModelId, TokenizationResult, LoadProgress } from '../engine/types';

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

  // ─────────────────────────────────────────────────────────────────────────
  // ACTIONS
  // ─────────────────────────────────────────────────────────────────────────

  /** Initialize the Web Worker. Must be called once before loading models. */
  initWorker: () => void;

  /** Download and load a model by ID. Updates status and progress during load. */
  loadModel: (modelId: ModelId) => Promise<void>;

  /** Tokenize text using the loaded model. Updates tokens/tokenIds state. */
  tokenize: (text: string) => Promise<TokenizationResult>;

  /** Reset error state to allow retry. Clears error and sets status to 'idle'. */
  reset: () => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  // Initial state
  status: 'idle',
  modelId: null,
  loadProgress: 0,
  error: null,
  tokens: [],
  tokenIds: [],
  worker: null,

  /**
   * Initialize the Web Worker for model inference.
   * Includes error handling for worker creation failures.
   */
  initWorker: () => {
    try {
      const worker = new Worker(
        new URL('../engine/worker.ts', import.meta.url),
        { type: 'module' }
      );

      // Handle worker-level errors (script load failures, uncaught exceptions)
      worker.onerror = (event) => {
        console.error('Worker error:', event);
        set({
          status: 'error',
          error: `Worker error: ${event.message || 'Unknown worker error'}`,
        });
      };

      const wrappedWorker = Comlink.wrap<ModelWorkerAPI>(worker);
      set({ worker: wrappedWorker });
    } catch (err) {
      console.error('Failed to initialize worker:', err);
      set({
        status: 'error',
        error: `Failed to initialize worker: ${String(err)}`,
      });
    }
  },

  /**
   * Download and load a model by ID.
   * Tracks progress and handles errors gracefully.
   */
  loadModel: async (modelId: ModelId) => {
    const { worker } = get();
    if (!worker) {
      set({ status: 'error', error: 'Worker not initialized. Please refresh the page.' });
      return;
    }

    set({ status: 'loading', loadProgress: 0, error: null });

    try {
      await worker.loadModel(
        modelId,
        Comlink.proxy((progress: LoadProgress) => {
          set({ loadProgress: progress.progress ?? 0 });
        })
      );
      set({ status: 'ready', modelId, loadProgress: 100 });
    } catch (err) {
      console.error('Model loading failed:', err);
      set({ status: 'error', error: String(err) });
    }
  },

  /**
   * Tokenize text using the loaded model.
   */
  tokenize: async (text: string) => {
    const { worker } = get();
    if (!worker) throw new Error('Worker not initialized');

    const result = await worker.tokenize(text);
    set({ tokens: result.tokens, tokenIds: result.tokenIds });
    return result;
  },

  /**
   * Reset error state to allow retry.
   */
  reset: () => {
    set({ status: 'idle', error: null, loadProgress: 0 });
  },
}));
