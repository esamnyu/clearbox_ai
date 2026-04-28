/**
 * Analysis State Management Store
 * ================================
 *
 * Holds derived analysis results from the Web Worker's generation output.
 * Converts raw TensorWithMetadata arrays into layer-indexed Maps of TensorView
 * for efficient lookup by visualization components.
 *
 * HOW IT WORKS:
 * 1. modelStore.generate() returns a GenerationResult with raw tensors
 * 2. Components call analyze(result) to populate this store
 * 3. Bridge functions group tensors by layer at the selected step
 * 4. UI components read attentions/hiddenStates Maps and selection state
 *
 * @module store/analysisStore
 */

import { create } from "zustand";
import { TensorView } from "@/analysis/utils/tensor";
import type { GenerationResult } from "@/engine/types";
import {
  attentionsToLayerMap,
  hiddenStatesToLayerMap,
} from "@/lib/tensorBridge";
import {
  classifyAttentionHead,
  computeAttentionEntropy,
  analyzeAllLayers,
} from "@/analysis/attention";
import type { AttentionPattern } from "@/analysis/attention";
import {
  getLogitLens,
  getGradients,
  getHealth,
  getContrastivePairs,
  getSteeringVector,
  generateSteered,
  ablateDirection,
  type LogitLensResponse,
  type GradientsResponse,
  type ContrastivePairsResponse,
  type SteeringVectorResponse,
  type SteeredGenerationResponse,
  type AblationResponse,
} from "@/lib/api";

export interface HeadGridCell {
  layer: number;
  head: number;
  pattern: AttentionPattern;
  entropy: number;
  isInduction: boolean;
  inductionScore: number;
}

interface AnalysisState {
  // ─────────────────────────────────────────────────────────────────────────
  // STATE
  // ─────────────────────────────────────────────────────────────────────────

  /** Attention data grouped by layer (step 0 = prompt-only forward pass) */
  attentions: Map<number, TensorView> | null;

  /** Hidden states grouped by layer */
  hiddenStates: Map<number, TensorView> | null;

  /** Token labels from the generation result */
  tokens: string[];

  /** Currently selected transformer layer */
  selectedLayer: number;

  /** Currently selected attention head */
  selectedHead: number;

  /** Currently selected generation step */
  selectedStep: number;

  /** Head classification grid: [layer][head] */
  headGrid: HeadGridCell[][] | null;

  // Backend state
  backendStatus: "unknown" | "connected" | "disconnected";
  logitLensResult: LogitLensResponse | null;
  gradientResult: GradientsResponse | null;
  backendError: string | null;

  // Steering state
  contrastivePairs: ContrastivePairsResponse | null;
  steeringVector: SteeringVectorResponse | null;
  steeredResult: SteeredGenerationResponse | null;
  steeringLayer: number;
  steeringAlpha: number;

  // Ablation state — uses steeringVector as the direction to project out
  ablatedResult: AblationResponse | null;

  // ─────────────────────────────────────────────────────────────────────────
  // ACTIONS
  // ─────────────────────────────────────────────────────────────────────────

  /** Convert a GenerationResult into layer-indexed Maps and store token labels */
  analyze: (result: GenerationResult) => void;

  /** Update the selected layer index */
  setSelectedLayer: (layer: number) => void;

  /** Update the selected attention head index */
  setSelectedHead: (head: number) => void;

  /** Update the selected generation step index */
  setSelectedStep: (step: number) => void;

  /** Clear all analysis state back to defaults */
  reset: () => void;

  // Backend actions
  checkBackend: () => Promise<void>;
  runLogitLens: (prompt: string) => Promise<void>;
  runGradients: (prompt: string, targetToken: string) => Promise<void>;

  // Steering actions
  fetchContrastivePairs: () => Promise<void>;
  computeSteeringVector: (
    positive: string[],
    negative: string[],
    layer: number,
  ) => Promise<void>;
  runSteeredGeneration: (prompt: string) => Promise<void>;
  setSteeringLayer: (layer: number) => void;
  setSteeringAlpha: (alpha: number) => void;

  // Ablation actions
  runAblation: (prompt: string) => Promise<void>;
}

const initialState = {
  attentions: null,
  hiddenStates: null,
  tokens: [],
  selectedLayer: 0,
  selectedHead: 0,
  selectedStep: 0,
  headGrid: null,
  backendStatus: "unknown" as const,
  logitLensResult: null,
  gradientResult: null,
  backendError: null,
  contrastivePairs: null,
  steeringVector: null,
  steeredResult: null,
  steeringLayer: 6,
  steeringAlpha: 1.0,
  ablatedResult: null,
} as const;

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  // Initial state
  attentions: initialState.attentions,
  hiddenStates: initialState.hiddenStates,
  tokens: [...initialState.tokens],
  selectedLayer: initialState.selectedLayer,
  selectedHead: initialState.selectedHead,
  selectedStep: initialState.selectedStep,
  headGrid: initialState.headGrid,
  backendStatus: initialState.backendStatus,
  logitLensResult: initialState.logitLensResult,
  gradientResult: initialState.gradientResult,
  backendError: initialState.backendError,
  contrastivePairs: initialState.contrastivePairs,
  steeringVector: initialState.steeringVector,
  steeredResult: initialState.steeredResult,
  steeringLayer: initialState.steeringLayer,
  steeringAlpha: initialState.steeringAlpha,
  ablatedResult: initialState.ablatedResult,

  analyze: (result: GenerationResult) => {
    const attentions = attentionsToLayerMap(result.attentions ?? [], 0);
    const hiddenStates = hiddenStatesToLayerMap(result.hiddenStates ?? [], 0);

    set({
      attentions,
      hiddenStates,
      tokens: result.tokens,
      selectedLayer: 0,
      selectedHead: 0,
    });

    // Compute head grid if we have attentions
    if (attentions.size > 0 && result.tokens.length > 0) {
      // Get induction heads across all layers
      const inductionResults = analyzeAllLayers(attentions, result.tokens, 0.3);
      const inductionSet = new Set(
        inductionResults.map((r) => `${r.layer}-${r.head}`),
      );
      const inductionScoreMap = new Map(
        inductionResults.map((r) => [`${r.layer}-${r.head}`, r.score]),
      );

      const grid: HeadGridCell[][] = [];

      for (let layer = 0; layer < 12; layer++) {
        const layerTensor = attentions.get(layer);
        if (!layerTensor) continue;

        const entropyTensor = computeAttentionEntropy(layerTensor);
        const layerRow: HeadGridCell[] = [];

        for (let head = 0; head < 12; head++) {
          // Extract single head's [seq, seq] matrix for classification
          // layerTensor shape is [12, seq, seq]
          const seqLen = layerTensor.shape[1];
          const headData = new Float32Array(seqLen * seqLen);
          for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
              headData[i * seqLen + j] = layerTensor.get(head, i, j);
            }
          }
          const headTensor = new TensorView(headData, [seqLen, seqLen]);

          const pattern = classifyAttentionHead(headTensor, result.tokens);
          const key = `${layer}-${head}`;

          layerRow.push({
            layer,
            head,
            pattern: inductionSet.has(key) ? "induction" : pattern,
            entropy: entropyTensor.get(head),
            isInduction: inductionSet.has(key),
            inductionScore: inductionScoreMap.get(key) ?? 0,
          });
        }

        grid.push(layerRow);
      }

      set({ headGrid: grid });
    }
  },

  setSelectedLayer: (layer: number) => {
    set({ selectedLayer: layer });
  },

  setSelectedHead: (head: number) => {
    set({ selectedHead: head });
  },

  setSelectedStep: (step: number) => {
    set({ selectedStep: step });
  },

  checkBackend: async () => {
    try {
      await getHealth();
      set({ backendStatus: "connected", backendError: null });
    } catch {
      set({
        backendStatus: "disconnected",
        backendError: "Backend not reachable",
      });
    }
  },

  runLogitLens: async (prompt: string) => {
    try {
      set({ backendError: null });
      const result = await getLogitLens(prompt);
      set({ logitLensResult: result });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  runGradients: async (prompt: string, targetToken: string) => {
    try {
      set({ backendError: null });
      const result = await getGradients(prompt, targetToken);
      set({ gradientResult: result });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  fetchContrastivePairs: async () => {
    try {
      set({ backendError: null });
      const result = await getContrastivePairs();
      set({ contrastivePairs: result });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  computeSteeringVector: async (
    positive: string[],
    negative: string[],
    layer: number,
  ) => {
    try {
      set({ backendError: null });
      const result = await getSteeringVector(positive, negative, layer);
      set({ steeringVector: result, steeringLayer: layer });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  runSteeredGeneration: async (prompt: string) => {
    const state = get();
    if (!state.steeringVector) return;
    try {
      set({ backendError: null });
      const result = await generateSteered(
        prompt,
        state.steeringVector.vector,
        state.steeringAlpha,
        state.steeringLayer,
      );
      set({ steeredResult: result });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  setSteeringLayer: (layer: number) => set({ steeringLayer: layer }),
  setSteeringAlpha: (alpha: number) => set({ steeringAlpha: alpha }),

  runAblation: async (prompt: string) => {
    const state = get();
    if (!state.steeringVector) return;
    try {
      set({ backendError: null });
      const result = await ablateDirection(
        prompt,
        state.steeringVector.vector,
        state.steeringLayer,
      );
      set({ ablatedResult: result });
    } catch (e) {
      set({ backendError: String(e) });
    }
  },

  reset: () => {
    set({
      attentions: initialState.attentions,
      hiddenStates: initialState.hiddenStates,
      tokens: [...initialState.tokens],
      selectedLayer: initialState.selectedLayer,
      selectedHead: initialState.selectedHead,
      selectedStep: initialState.selectedStep,
      headGrid: initialState.headGrid,
      logitLensResult: null,
      gradientResult: null,
      backendError: null,
      // keep backendStatus as-is (don't reset connection state)
      contrastivePairs: null, // keep pairs (they don't change)
      steeringVector: null,
      steeredResult: null,
      ablatedResult: null,
    });
  },
}));
