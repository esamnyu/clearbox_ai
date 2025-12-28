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

// helper to clean up the double token problem in greedy decoding 
function sampleToken(logits: Float32Array, temperature: number): number {
  // 1. Handle Temperature
  // If temp is very low, fall back to Greedy (Argmax) for stability
  if (temperature < 0.1) {
    let maxLogit = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // 2. Apply Temperature & Numerical Stability
  // We subtract maxLogit to prevent overflow during Math.exp()
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const exps = new Float32Array(logits.length);
  let sumExps = 0;
  for (let i = 0; i < logits.length; i++) {
    const val = Math.exp((logits[i] - maxLogit) / temperature);
    exps[i] = val;
    sumExps += val;
  }

  // 3. Multinomial Sampling (Cumulative Distribution)
  let r = Math.random() * sumExps;
  for (let i = 0; i < logits.length; i++) {
    r -= exps[i];
    if (r <= 0) return i;
  }

  // Fallback (should rarely reach here due to floating point precision)
  return 0;
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
      console.log(`Model ${modelId} loaded for interpretability`);

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


    // Tokenize the input (tokensizer returns bigints in transformers v3)
    const encoded = await tokenizer(text, {
      return_tensors: 'pt', 
      padding: false,
      truncation: false,
    });

    // need to convert bigint IDs to number IDs
    // const tensorData = encoded.input_ids.data;
    // const tokenIds: number[] = Array.from(tensorData).map((x) => Number(x));
    // console.log('Token IDs:', tokenIds);
    // console.log('Raw Tensor Data:', tensorData);
    const inputIdsTensor = encoded.input_ids;
    const tokenIds: number[] = Array.from(inputIdsTensor.data as BigInt64Array).map(Number);

    const tokens = tokenIds.map(id => tokenizer!.decode([id]));
    console.log('Tokens:', tokens);

    return {
      tokens,
      tokenIds,
      attentionMask: Array.from(encoded.attention_mask.data as BigInt64Array).map(Number),
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
//   async generate(
//     prompt: string,
//     options: GenerateOptions = {}
//   ): Promise<GenerationResult> {
//     if (!model || !tokenizer) throw new Error('Model not loaded.');

//     const {
//       maxNewTokens = 10,
//       temperature = 1.0,
//       outputHiddenStates = true,
//       outputAttentions = true,
//     } = options;

//     const inputs = await tokenizer(prompt, { return_tensors: 'pt' });

//     // running inference here...
//     // We use return_dict_in_generate to get the full telemetry object
//     const output = await model.generate({
//       ...inputs,
//       max_new_tokens: maxNewTokens,
//       temperature,
//       do_sample: temperature > 0,
//       return_dict_in_generate: true, // this forces a return object instead of just tokens
//       output_attentions: outputAttentions,
//       output_hidden_states: outputHiddenStates,
//     });

//     const generatedIds = output.sequences; // unsure that sequences is present in ModelOutput
//     const generatedText = tokenizer.decode(generatedIds[0], { skip_special_tokens: true });
//     console.log('generatedText:', generatedText);

//     // IMPORTANT: rudimentary telemetry extraction (will need to adapt this for future analysis)
//     // NOTE: this extracts data for the generated tokens.
//     // Structure: output.attentions[token_index][layer_index] -> Tensor

//     let extractedAttentions: any = null;
//     let extractedHiddenStates: any = null;

//     console.log('[Worker] Extracting internal model states for analysis');
//     console.log('Full output class:', output);
//     console.log('Output keys:', Object.keys(output));
//     console.log('Attentions:', output.attentions);
//     console.log('Hidden States:', output.hidden_states);


//     if (outputAttentions && output.attentions) {
//       // example extraction logic: extract attention from the last generated token, last layer
//       // in the real app, we would flatten and transfer these buffers
//       console.log(`[Worker] Captured ${output.attentions.length} steps of attention`);
//       extractedAttentions = {
//         layers: output.attentions[0].length,
//         steps: output.attentions.length,
//         // add more extraction logic here
//         info: "Raw tensors held in worker memory - see console logs"
//       };
//     }
    
//     return {
//       text: generatedText,
//       tokens: [], 
//       tokenIds: Array.from(generatedIds[0].data as BigInt64Array).map(Number),
//       attentions: extractedAttentions,
//       hiddenStates: extractedHiddenStates, // placeholder
//     };
//   },
// };

  async generate(
    prompt: string,
    options: GenerateOptions = {}  
  ): Promise<GenerationResult> {
    if (!model || !tokenizer) throw new Error('Model not loaded.');

    const {
      maxNewTokens = 10,
      temperature = 0.7,
      outputHiddenStates = true,
      outputAttentions = true,
    } = options;

    // Tokenize input
    const inputs = await tokenizer(prompt, { return_tensors: 'pt' });

    // let currentTokenIds = inputs.input_ids; // Tensor of shape [1, seq_len]
    // Convert BigInt64Array to standard arrays for easy manipulation in the loop
    let currentInputIds = Array.from(inputs.input_ids.data as BigInt64Array).map(Number);
    let currentAttentionMask = Array.from(inputs.attention_mask.data as BigInt64Array).map(Number);

    // telemtry storage
    const collectedAttentions: any[] = [];
    const collectedHiddenStates: any[] = [];
    const newTokens: number[] = [];

    console.log('[Worker] Starting Manual Autoregressive Loop...');

    for (let i = 0; i < maxNewTokens; i++) {
      // Forward pass
      // We will specifically request telemetry for this specific pass
      // Re-create Tensors for this step (Transformers.js expects BigInt64Array for 'int64')
      const inputTensor = new Tensor(
        'int64',
        BigInt64Array.from(currentInputIds.map(BigInt)),
        [1, currentInputIds.length]
      );

      const maskTensor = new Tensor(
        'int64',
        BigInt64Array.from(currentAttentionMask.map(BigInt)),
        [1, currentAttentionMask.length]
      );


      const output = await model({
        input_ids: inputTensor,
        attention_mask: maskTensor,
      }, {
        output_attentions: outputAttentions,
        output_hidden_states: outputHiddenStates,
        return_dict: true,
      });

      console.log('[DEBUG] Current attention mask:', currentAttentionMask);
      console.log('[DEBUG] Output attentions:', output.attentions);
      console.log('[DEBUG] Output hidden states:', output.hidden_states);
      console.log('[DEBUG] Output logits:', output.logits);

      // extraction
      // output.attentions is an array of Tensors: [layer_0, layer_1, ..., layer_N]
      if (outputAttentions && output.attentions) {
        // Storing the struct to show that it works, some cases we would only want the last token attention but here we want it all.
        collectedAttentions.push({
          step: i,
          layers: output.attentions.length, // should slice the tensor here to save memory
          heads: output.attentions[0].dims[1]
        });
      }

      if (outputHiddenStates && output.hidden_states) {
        collectedHiddenStates.push({
          step: i,
          layers: output.hidden_states.length // should slice the tensor here to save memory
        });
      }

      // greedy decoding: selecting next token
      // Logits shape: [batch, seq_len, vocab_size]
      const logits = output.logits;

      // grabbing logits for the last toke, need to handle the tensor data manually
      const [batchSize, seqLen, vocabSize] = logits.dims;
      const data = logits.data as Float32Array; // assuming fp32

      // calc offset for last token vocab distribution
      const lastIdx = (seqLen - 1) * vocabSize;
      const lastLogits = data.slice(lastIdx, lastIdx + vocabSize);

     // using sample helper here to handle temperature and sampling
      const nextTokenId = sampleToken(lastLogits, temperature);

      // update for next iteration
      newTokens.push(nextTokenId);

      // append new token to input (re create tensor)
      // NOTE: In production, use KV-caching (past_key_values) for speed. 
      // This method (re-running full context) is slower but easier to debug for interpretability.
      newTokens.push(nextTokenId);
      currentInputIds.push(nextTokenId);
      currentAttentionMask.push(1); // assume all tokens are valid

      console.log(`[Worker] Generated token ${i + 1}/${maxNewTokens}: ID ${nextTokenId}`);
      // Optional: Early stop on EOS token (50256 for GPT-2)
      if (nextTokenId === 50256) break;
    }

    // Decode generated token IDs to text
    const generatedText = tokenizer.decode(newTokens, { skip_special_tokens: true });
    console.log(`[Worker] Loop finished. Captured ${collectedAttentions.length} attention steps.`);

    return {
      text: generatedText, // this will be just the NEW text
      tokens: [],
      tokenIds: newTokens,
      attentions: collectedAttentions.length > 0 ? { steps: collectedAttentions } : null,
      hiddenStates: collectedHiddenStates.length > 0 ? { steps: collectedHiddenStates } : null,
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
