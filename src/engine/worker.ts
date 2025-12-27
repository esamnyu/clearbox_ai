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
// import { pipeline, env, type TextGenerationPipeline } from '@huggingface/transformers';
import {
  env,
  pipeline,
  AutoTokenizer,
  GPT2LMHeadModel,
  PreTrainedTokenizer,
  PreTrainedModel,
  Tensor,
} from '@huggingface/transformers';
import type {
  ModelWorkerAPI,
  ModelId,
  ModelStatus,
  LoadProgress,
  TokenizationResult,
  GenerateOptions,
  GenerationResult,
  PipelineFactory,
  PipelineInterface,
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
// PIPELINE FACTORY (Dependency Injection for Testing)
// ============================================================================

/**
 * Default pipeline factory that wraps the real transformers.js pipeline.
 * In production, this is used. In tests, it can be swapped with a mock.
 */
const defaultPipelineFactory: PipelineFactory = {
  async create(task, modelId, config) {
    // Use type assertion to bridge our generic interface with transformers.js specific types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const options: any = {
      dtype: config.dtype,
      device: config.device,
    };
    if (config.progress_callback) {
      options.progress_callback = config.progress_callback;
    }
    return await pipeline(task, modelId, options) as unknown as PipelineInterface;
  },
};

/** Current factory - can be swapped for testing */
let currentPipelineFactory: PipelineFactory = defaultPipelineFactory;

/**
 * Override the pipeline factory (used by tests).
 * @internal
 */
function setPipelineFactory(factory: PipelineFactory): void {
  currentPipelineFactory = factory;
}

/**
 * Reset to default factory.
 * @internal
 */
function resetPipelineFactory(): void {
  currentPipelineFactory = defaultPipelineFactory;
}

// ============================================================================
// WORKER STATE
// ============================================================================

// /**
//  * Current loaded model instance.
//  * The TextGenerationPipeline provides both generation and tokenization APIs.
//  */
// let currentModel: TextGenerationPipeline | null = null;

// /** Currently loaded model identifier (e.g., 'Xenova/gpt2') */
// let currentModelId: ModelId | null = null;

// /** Current worker state for external status queries */
// let currentStatus: ModelStatus = 'idle';

let tokenizer: PreTrainedTokenizer | null = null;
let model: PreTrainedModel | null = null;
let currentModelId: ModelId | null = null;
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
    if (model && currentModelId === modelId) {
      console.log(`Model ${modelId} already loaded`);
      return;
    }

    try {
      currentStatus = 'loading';

      console.log(`Loading raw model: ${modelId} (Full Precision)`);

      // load tokenizer
      tokenizer = await AutoTokenizer.from_pretrained(modelId);

      // load model (GPT2LMHeadModel)
      // we load the actual model class, not a pipeline wrapper
      model = await GPT2LMHeadModel.from_pretrained(modelId, {
        dtype: 'fp32', // full precision for accurate gradients/analysis
        device: 'wasm',
        progress_callback: (progress) => {
          if (onProgress) {
            onProgress({
              status: progress.status as LoadProgress['status'],
              file: progress.file,
              progress: progress.progress,
              loaded: progress.loaded,
              total: progress.total,
            })
          }
        }
      })

      currentModelId = modelId;
      currentStatus = 'ready';
      console.log(`Interpretability Model ${modelId} loaded successfully`);

    } catch (error) {
      currentStatus = 'error';
      console.error('Failed to load model:', error);
      throw error;
    }
  },

  /**
   * Unload the current model to free memory.
   */
  async unloadModel(): Promise<void> {
    if (model) {
      // manual cleanup but garbage collector should handel most of this
      model.dispose();
      model = null;
    }
    tokenizer = null;
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
    if (!tokenizer) throw new Error('Model not loaded. Call LoadModel first');

    // Access the tokenizer from the pipeline
    const tokenizer = currentModel.tokenizer;

    // Tokenize the input (tokensizer returns bigints in transformers v3)
    const encoded = await tokenizer(text, {
      return_tensors: 'art', // explicit use of 'art' for onnx runtime
      padding: false,
      truncation: false,
    });

    // need to convert bigint IDs to number IDs
    const tensorData = encoded.input_ids.data;
    const tokenIds: number[] = Array.from(tensorData).map((x) => Number(x));
    console.log('Token IDs:', tokenIds);
    console.log('Raw Tensor Data:', tensorData);

    // handle attention masks if present and cast to number[]
    let attentionMask: number[] = [];
    if (encoded.attention_mask) {
      const maskData = encoded.attention_mask.data;
      attentionMask = Array.from(maskData).map((x) => Number(x));
      console.log('Attention Mask:', attentionMask);
    } else {
      // if no attention mask, fallback to all ones
      attentionMask = tokenIds.map(() => 1);
    }

    // Get token strings by decoding each token ID individually
    // const tokenIds: number[] = encoded.input_ids;
    const tokens: string[] = [];

    for (const id of tokenIds) {
      const decoded = tokenizer.decode([id], { skip_special_tokens: false });
      tokens.push(decoded);
    }
    console.log('Tokens:', tokens);

    return {
      tokens,
      tokenIds,
      attentionMask,
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
    if (!model || !tokenizer) throw new Error('Model not loaded.');

    const {
      maxNewTokens = 20,
      temperature = 1.0,
      outputHiddenStates = true,
      outputAttentions = true,
    } = options;

    const inputs = await tokenizer(prompt, { return_tensors: 'pt' });

    // running inference here...
    // We use return_dict_in_generate to get the full telemetry object
    const output = await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      temperature,
      do_sample: temperature > 0,
      return_dict_in_generate: true, // this forces a return object instead of just tokens
      output_attentions: outputAttentions,
      output_hidden_states: outputHiddenStates,
    });

    const generatedIds = output.sequences;
    const generatedText = tokenizer.decode(generatedIds[0], { skip_special_tokens: true });

    // IMPORTANT: rudimentary telemetry extraction (will need to adapt this for future analysis)
    // NOTE: this extracts data for the generated tokens.
    // Structure: output.attentions[token_index][layer_index] -> Tensor

    let extractedAttentions: any = null;
    let extractedHiddenStates: any = null;

    if (outputAttentions && output.attentions) {
      // example extraction logic: extract attention from the last generated token, last layer
      // in the real app, we would flatten and transfer these buffers
      console.log('Captured ${output.attentions.length} steps of attention');
      extractedAttentions = {
        layers: output.attentions[0].length,
        steps: output.attentions.length,
        // add more extraction logic here
        info: "Check console in worker for raw tensors"
      };
    }
    
    return {
      text: generatedText,
      tokens: [], 
      tokenIds: Array.from(generatedIds[0].data as BigInt64Array).map(Number),
      attentions: extractedAttentions,
      hiddenStates: extractedHiddenStates,
    };
  },
};

// ============================================================================
// TEST UTILITIES (only used in test environment)
// ============================================================================

const testAPI = {
  _setPipelineFactory: setPipelineFactory,
  _resetPipelineFactory: resetPipelineFactory,
};

// Expose the API via Comlink (includes test utilities)
Comlink.expose({ ...workerAPI, ...testAPI });

// Type export for the main thread
export type { ModelWorkerAPI };
export type TestAPI = typeof testAPI;
