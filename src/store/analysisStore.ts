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
  loadModel as apiLoadModel,
  BACKEND_DEFAULT_MODEL,
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
  //
  // "waking" is its own state because the free HF Space sleeps after ~48h idle
  // and takes 30-60s to cold-start. Without it a first-time visitor sees
  // "unreachable" — indistinguishable from genuinely broken — for the whole
  // wake. See checkBackend below.
  backendStatus: "unknown" | "waking" | "connected" | "disconnected";
  /** Model the *backend* holds, e.g. "gpt2-small". Null until /load succeeds. */
  backendModel: string | null;
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
  backendModel: null,
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
  backendModel: initialState.backendModel,
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

    // Step-0 attention/hidden states come from the PROMPT-ONLY forward pass,
    // so their sequence length is the prompt length — not prompt+generated.
    // result.tokens holds the full sequence (prompt + generated tokens); slice
    // it to the step-0 length before any attention math. Otherwise
    // detectInductionHeads / classifyAttentionHead throw on the
    // token-count-vs-seqLen mismatch, and the throw (caught in
    // App.handleGenerate) silently clears the whole panel — the original bug.
    const firstAttn = attentions.values().next().value as
      TensorView | undefined;
    const stepSeqLen = firstAttn ? firstAttn.shape[1] : result.tokens.length;
    const stepTokens = result.tokens.slice(0, stepSeqLen);

    set({
      attentions,
      hiddenStates,
      tokens: stepTokens,
      selectedLayer: 0,
      selectedHead: 0,
    });

    // Compute head grid if we have attentions
    if (attentions.size > 0 && stepTokens.length > 0) {
      // Get induction heads across all layers
      const inductionResults = analyzeAllLayers(attentions, stepTokens, 0.3);
      const inductionSet = new Set(
        inductionResults.map((r) => `${r.layer}-${r.head}`),
      );
      const inductionScoreMap = new Map(
        inductionResults.map((r) => [`${r.layer}-${r.head}`, r.score]),
      );

      const grid: HeadGridCell[][] = [];

      // Layer/head counts come from the tensors themselves, not hardcoded
      // 12/12, so the grid also works for non-GPT-2 models.
      const layerKeys = [...attentions.keys()].sort((a, b) => a - b);
      for (const layer of layerKeys) {
        const layerTensor = attentions.get(layer);
        if (!layerTensor) continue;

        const numHeads = layerTensor.shape[0];
        const seqLen = layerTensor.shape[1];
        const entropyTensor = computeAttentionEntropy(layerTensor);
        const layerRow: HeadGridCell[] = [];

        for (let head = 0; head < numHeads; head++) {
          // Extract single head's [seq, seq] matrix for classification.
          // layerTensor shape is [numHeads, seq, seq].
          const headData = new Float32Array(seqLen * seqLen);
          for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
              headData[i * seqLen + j] = layerTensor.get(head, i, j);
            }
          }
          const headTensor = new TensorView(headData, [seqLen, seqLen]);

          const pattern = classifyAttentionHead(headTensor, stepTokens);
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
    // Two steps, and the second one is the whole point. A bare health check
    // passes against a backend holding no model, which is exactly the state a
    // freshly-woken Space is in — every downstream panel then fails with
    // "No model loaded. Call load_model() first." The visitor sees a green
    // status light and four broken sections. So: ping, then load.
    set({ backendStatus: "waking", backendError: null });
    try {
      await getHealth();
    } catch {
      set({
        backendStatus: "disconnected",
        backendModel: null,
        backendError:
          "Backend unreachable. If this is the public deploy, the free Space " +
          "may be cold-starting — retry in a minute.",
      });
      return;
    }

    try {
      // 180s: a cold Space downloads gpt2-small from the Hub on first load.
      const res = await apiLoadModel(BACKEND_DEFAULT_MODEL, 180_000);
      set({
        backendStatus: "connected",
        backendModel: res.model ?? BACKEND_DEFAULT_MODEL,
        backendError: null,
      });
    } catch (e) {
      // Reachable but model-less. Degraded, not dead: the cached bench table
      // in §VII still renders, so say what does and doesn't work.
      set({
        backendStatus: "connected",
        backendModel: null,
        backendError:
          `Backend is up but could not load ${BACKEND_DEFAULT_MODEL} ` +
          `(${e instanceof Error ? e.message : String(e)}). Layer predictions, ` +
          `token importance and steering need it; the cached bench still renders.`,
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
