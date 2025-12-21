/**
 * Web Worker for model inference.
 *
 * Runs transformers.js off the main thread to keep UI responsive.
 * Communicates with main thread via Comlink.
 *
 * @module engine/worker
 */

import * as Comlink from 'comlink';
import { pipeline, env } from '@xenova/transformers';
import type {
  ModelWorkerAPI,
  ModelId,
  ModelStatus,
  LoadProgress,
  TokenizationResult,
  GenerateOptions,
  GenerationResult,
  SerializedTensor,
} from './types';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

/** Current model state */
let currentModel: Awaited<ReturnType<typeof pipeline>> | null = null;
let currentModelId: ModelId | null = null;
let currentStatus: ModelStatus = 'idle';

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

      // Extract the generated text
      const generatedText = output[0].generated_text;

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
