/**
 * Zustand store for model state.
 * Minimal implementation for Session 1.
 */

import { create } from 'zustand';
import * as Comlink from 'comlink';
import type { ModelWorkerAPI, ModelStatus, ModelId, TokenizationResult } from '../engine/types';

interface ModelState {
  status: ModelStatus;
  modelId: ModelId | null;
  loadProgress: number;
  error: string | null;

  // Current tokenization
  tokens: string[];
  tokenIds: number[];

  // Worker reference
  worker: Comlink.Remote<ModelWorkerAPI> | null;

  // Actions
  initWorker: () => void;
  loadModel: (modelId: ModelId) => Promise<void>;
  tokenize: (text: string) => Promise<TokenizationResult>;
}

export const useModelStore = create<ModelState>((set, get) => ({
  status: 'idle',
  modelId: null,
  loadProgress: 0,
  error: null,
  tokens: [],
  tokenIds: [],
  worker: null,

  initWorker: () => {
    const worker = new Worker(
      new URL('../engine/worker.ts', import.meta.url),
      { type: 'module' }
    );
    const wrappedWorker = Comlink.wrap<ModelWorkerAPI>(worker);
    set({ worker: wrappedWorker });
  },

  loadModel: async (modelId: ModelId) => {
    const { worker } = get();
    if (!worker) throw new Error('Worker not initialized');

    set({ status: 'loading', loadProgress: 0, error: null });

    try {
      await worker.loadModel(
        modelId,
        Comlink.proxy((progress) => {
          set({ loadProgress: progress.progress ?? 0 });
        })
      );
      set({ status: 'ready', modelId, loadProgress: 100 });
    } catch (err) {
      set({ status: 'error', error: String(err) });
    }
  },

  tokenize: async (text: string) => {
    const { worker } = get();
    if (!worker) throw new Error('Worker not initialized');

    const result = await worker.tokenize(text);
    set({ tokens: result.tokens, tokenIds: result.tokenIds });
    return result;
  },
}));
