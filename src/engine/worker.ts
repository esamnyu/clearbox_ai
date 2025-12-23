/**
 * Web Worker for Model Inference
 * ================================
 *
 * This file runs in a separate thread (Web Worker) to keep the UI responsive
 * while loading and running the GPT-2 model. Without this, loading a 500MB
 * model would freeze the browser.
 *
 * HOW IT WORKS:
 * 1. The main thread (App.tsx) creates a Worker pointing to this file
 * 2. Comlink wraps the worker to make method calls feel like normal async functions
 * 3. When you call worker.loadModel(), this file downloads the model in the background
 * 4. When you call worker.tokenize(text), this file processes the text using the model
 *
 * KEY FUNCTIONS:
 * - loadModel(modelId)  : Downloads and initializes the GPT-2 model (~500MB)
 * - unloadModel()       : Frees memory by removing the model
 * - getStatus()         : Returns current state ('idle'|'loading'|'ready'|'error')
 * - tokenize(text)      : Converts text to tokens (e.g., "Hello" → [15496])
 * - generate(prompt)    : Generates new text from a prompt
 *
 * WHY A WEB WORKER?
 * - ML models are CPU-intensive and would block the main thread
 * - Workers run in parallel, keeping button clicks and animations smooth
 * - Transformers.js uses WASM which benefits from dedicated thread time
 *
 * COMMUNICATION:
 * - Uses Comlink library for type-safe RPC (Remote Procedure Calls)
 * - Progress callbacks use Comlink.proxy() to work across the thread boundary
 *
 * @module engine/worker
 */

import * as Comlink from 'comlink';
import { pipeline, env, type TextGenerationPipeline } from '@huggingface/transformers';
import type {
  ModelWorkerAPI,
  ModelId,
  ModelStatus,
  LoadProgress,
  TokenizationResult,
  GenerateOptions,
  GenerationResult,
} from './types';

// ============================================================================
// TRANSFORMERS.JS CONFIGURATION
// ============================================================================

/**
 * Configure transformers.js runtime behavior.
 * - allowLocalModels: false - Only load from Hugging Face CDN
 * - useBrowserCache: true - Cache model files in browser for faster reload
 */
env.allowLocalModels = false;
env.useBrowserCache = true;
// env.backends.onnx.wasm.threads = 1;

// ============================================================================
// WORKER STATE
// ============================================================================

/**
 * Current loaded model instance.
 * The TextGenerationPipeline provides both generation and tokenization APIs.
 */
let currentModel: TextGenerationPipeline | null = null;

/** Currently loaded model identifier (e.g., 'Xenova/gpt2') */
let currentModelId: ModelId | null = null;

/** Current worker state for external status queries */
let currentStatus: ModelStatus = 'idle';

// ============================================================================
// WORKER API IMPLEMENTATION
// ============================================================================

/**
 * The worker API implementation.
 * All methods are exposed to the main thread via Comlink.
 */
const workerAPI: ModelWorkerAPI = {
  /**
   * Load a model by ID.
   *
   * @param modelId - The Hugging Face model ID (e.g., 'Xenova/gpt2')
   * @param onProgress - Optional callback for loading progress
   */
  async loadModel(
    modelId: ModelId,
    onProgress?: (progress: LoadProgress) => void
  ): Promise<void> {
    if (currentModel && currentModelId === modelId) {
      console.log(`Model ${modelId} already loaded`);
      return;
    }

    try {
      currentStatus = 'loading';

      console.log(`Loading model: ${modelId}`);

      // Create the pipeline with progress callback
      currentModel = await pipeline('text-generation', modelId, {
        device: 'wasm',
        progress_callback: (progress: {
          status: string;
          file?: string;
          progress?: number;
          loaded?: number;
          total?: number;
        }) => {
          if (onProgress) {
            onProgress({
              status: progress.status as LoadProgress['status'],
              file: progress.file,
              progress: progress.progress,
              loaded: progress.loaded,
              total: progress.total,
            });
          }
          console.log(`[${modelId}] ${progress.status}: ${progress.file ?? ''} ${progress.progress?.toFixed(1) ?? ''}%`);
        },
      });

      currentModelId = modelId;
      currentStatus = 'ready';
      console.log(`Model ${modelId} loaded successfully`);

    } catch (error) {
      currentStatus = 'error';
      currentModel = null;
      currentModelId = null;
      console.error('Failed to load model:', error);
      throw error;
    }
  },

  /**
   * Unload the current model to free memory.
   */
  async unloadModel(): Promise<void> {
    currentModel = null;
    currentModelId = null;
    currentStatus = 'idle';
    console.log('Model unloaded');
  },

  /**
   * Get the current model status.
   */
  async getStatus(): Promise<{ status: ModelStatus; modelId: ModelId | null }> {
    return {
      status: currentStatus,
      modelId: currentModelId,
    };
  },

  /**
   * Tokenize text without generation.
   *
   * @param text - The text to tokenize
   * @returns Tokens, token IDs, and attention mask
   */
  async tokenize(text: string): Promise<TokenizationResult> {
    if (!currentModel) {
      throw new Error('No model loaded. Call loadModel first.');
    }

    // Access the tokenizer from the pipeline
    const tokenizer = currentModel.tokenizer;

    // Tokenize the input
    const encoded = await tokenizer(text, {
      return_tensors: false,
      padding: false,
      truncation: false,
    });

    // Get token strings by decoding each token ID individually
    const tokenIds: number[] = encoded.input_ids;
    const tokens: string[] = [];

    for (const id of tokenIds) {
      const decoded = tokenizer.decode([id], { skip_special_tokens: false });
      tokens.push(decoded);
    }

    return {
      tokens,
      tokenIds,
      attentionMask: encoded.attention_mask ?? tokenIds.map(() => 1),
    };
  },

  /**
   * Generate text with optional internal state extraction.
   *
   * @param prompt - The input prompt
   * @param options - Generation options
   * @returns Generated text and optionally hidden states/attentions
   *
   * RESEARCHER NOTE: The hidden states and attentions are extracted here.
   * Shape of hidden_states[layer]: [batch, seq_len, hidden_dim]
   * Shape of attentions[layer]: [batch, num_heads, seq_len, seq_len]
   */
  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerationResult> {
    if (!currentModel) {
      throw new Error('No model loaded. Call loadModel first.');
    }

    const {
      maxNewTokens = 20,
      temperature = 1.0,
      topK = 50,
      topP = 0.95,
      outputHiddenStates = false,
      outputAttentions = false,
    } = options;

    try {
      // Generate with the pipeline
      const output = await currentModel(prompt, {
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
        top_p: topP,
        do_sample: temperature > 0,
        // Note: output_hidden_states and output_attentions may not be
        // fully supported in all transformers.js versions
        return_full_text: true,
      });

      /**
       * Extract the generated text from pipeline output.
       *
       * The output structure from transformers.js can vary depending on version
       * and configuration. We safely handle both array and object formats:
       * - Array format: [{ generated_text: "..." }, ...]
       * - Direct format: { generated_text: "..." }
       */
      const firstResult = Array.isArray(output) ? output[0] : output;
      const generatedText = 'generated_text' in firstResult
        ? String(firstResult.generated_text)
        : String(firstResult);

      // Tokenize the full output for display
      const tokenization = await this.tokenize(generatedText);

      const result: GenerationResult = {
        text: generatedText,
        tokens: tokenization.tokens,
        tokenIds: tokenization.tokenIds,
      };

      // RESEARCHER TODO: Hidden state extraction
      // Currently, transformers.js doesn't expose hidden states easily.
      // This is where we would extract them if available.
      if (outputHiddenStates) {
        console.warn('Hidden state extraction not yet implemented');
        // result.hiddenStates = extractHiddenStates(output);
      }

      if (outputAttentions) {
        console.warn('Attention extraction not yet implemented');
        // result.attentions = extractAttentions(output);
      }

      return result;

    } catch (error) {
      console.error('Generation failed:', error);
      throw error;
    }
  },
};

// Expose the API via Comlink
Comlink.expose(workerAPI);

// Type export for the main thread
export type { ModelWorkerAPI };
