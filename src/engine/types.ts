/**
 * Type definitions for the model engine.
 *
 * These types are shared between the main thread and Web Worker.
 * All tensor data uses Float32Array for efficient transfer.
 */

/** Model loading status */
export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

/** Available model variants */
export type ModelId = 'Xenova/gpt2' | 'Xenova/gpt2-medium';

/** Progress callback for model loading */
export interface LoadProgress {
  status: 'downloading' | 'loading' | 'ready';
  file?: string;
  progress?: number;  // 0-100
  loaded?: number;
  total?: number;
}

/**
 * Serialized tensor for transfer between threads.
 * Uses Transferable objects for zero-copy transfer.
 */
export interface SerializedTensor {
  /** Raw tensor data (Transferable) */
  data: Float32Array;
  /** Tensor shape, e.g., [12, 64, 64] */
  shape: number[];
  /** Data type (always float32 for now) */
  dtype: 'float32';
}

/** Tokenization result */
export interface TokenizationResult {
  /** String representation of each token */
  tokens: string[];
  /** Token IDs (indices into vocabulary) */
  tokenIds: number[];
  /** Attention mask (1 for real tokens, 0 for padding) */
  attentionMask: number[];
}

/** Options for text generation */
export interface GenerateOptions {
  /** Maximum new tokens to generate */
  maxNewTokens?: number;
  /** Temperature for sampling (0 = greedy) */
  temperature?: number;
  /** Top-k sampling */
  topK?: number;
  /** Top-p (nucleus) sampling */
  topP?: number;
  /** Whether to output hidden states */
  outputHiddenStates?: boolean;
  /** Whether to output attention weights */
  outputAttentions?: boolean;
}

/** Full generation result with internal states */
export interface GenerationResult {
  /** Generated text */
  text: string;
  /** All tokens (input + generated) */
  tokens: string[];
  /** All token IDs */
  tokenIds: number[];
  /** Hidden states per layer (if requested) */
  hiddenStates?: SerializedTensor[];
  /** Attention weights per layer (if requested) */
  attentions?: SerializedTensor[];
  /** Final logits */
  logits?: SerializedTensor;
}

/**
 * Worker API exposed via Comlink.
 *
 * All methods are async because they cross the thread boundary.
 */
export interface ModelWorkerAPI {
  /** Load a model by ID */
  loadModel(
    modelId: ModelId,
    onProgress?: (progress: LoadProgress) => void
  ): Promise<void>;

  /** Unload the current model */
  unloadModel(): Promise<void>;

  /** Get current model status */
  getStatus(): Promise<{ status: ModelStatus; modelId: ModelId | null }>;

  /** Tokenize text without generation */
  tokenize(text: string): Promise<TokenizationResult>;

  /** Generate text with optional internal state extraction */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerationResult>;
}

  // ============================================================================
  // PIPELINE ABSTRACTION - Enables dependency injection for testing
  // ============================================================================

  /**
   * Encoded output from the tokenizer.
   * Mirrors the structure returned by transformers.js v3.
   *
   * @internal Used for pipeline abstraction, not part of public API.
   */
  export interface EncodedTokens {
    input_ids: {
      data: BigInt64Array | number[]
    }
    attention_mask?: {
      data: BigInt64Array | number[]
    }
  }

  /**
   * Options passed to the tokenizer's encode function.
   */
  export interface TokenizerOptions {
    return_tensors?: string
    padding?: boolean
    truncation?: boolean
  }

  /**
   * Options for decoding token IDs back to text.
   */
  export interface DecodeOptions {
    skip_special_tokens?: boolean
  }

  /**
   * Abstraction over a tokenizer.
   *
   * This is a callable interface: the tokenizer itself is a function,
   * but also has methods like `decode`.
   *
   * Production: wraps transformers.js PreTrainedTokenizer
   * Testing: mock implementation with deterministic outputs
   */
  export interface TokenizerInterface {
    (text: string, options: TokenizerOptions): Promise<EncodedTokens>
    decode(ids: number[], options?: DecodeOptions): string
  }

  /**
   * Configuration for text generation.
   * Maps to transformers.js generation config.
   */
  export interface GenerationConfig {
    max_new_tokens?: number
    temperature?: number
    top_k?: number
    top_p?: number
    do_sample?: boolean
    return_full_text?: boolean
  }

  /**
   * Single generation output from the pipeline.
   */
  export interface GenerationOutput {
    generated_text?: string
  }

  /**
   * Abstraction over a text-generation pipeline.
   *
   * Callable interface: pipeline(prompt, config) generates text.
   * Also exposes the tokenizer for direct tokenization.
   *
   * Production: wraps transformers.js TextGenerationPipeline
   * Testing: mock implementation with controlled outputs
   */
  export interface PipelineInterface {
    tokenizer: TokenizerInterface
    (prompt: string, config: GenerationConfig): Promise<GenerationOutput[]>
  }

  /**
   * Configuration for pipeline creation.
   */
  export interface PipelineConfig {
    dtype: string
    device: string
    progress_callback?: (progress: LoadProgress) => void
  }

  /**
   * Factory for creating pipelines.
   *
   * This is the primary injection point for testing:
   * - Production: creates real transformers.js pipelines
   * - Unit tests: returns mock pipelines with instant, deterministic responses
   * - Integration tests: loads tiny models for real tokenizer behavior
   */
  export interface PipelineFactory {
    create(
      task: 'text-generation',
      modelId: ModelId,
      config: PipelineConfig
    ): Promise<PipelineInterface>
  }


// ============================================================================
// RESEARCHER TYPES - Used in src/analysis/
// ============================================================================

/**
 * Configuration for attention analysis.
 *
 * @researcher_note These thresholds can be tuned based on the model and task.
 */
export interface AttentionAnalysisConfig {
  /** Minimum attention weight to consider significant */
  threshold: number;
  /** Whether to normalize attention weights */
  normalize: boolean;
  /** Layer range to analyze */
  layerRange: [number, number];
}

/**
 * Result of induction head detection.
 */
export interface InductionHeadResult {
  /** Layer index */
  layer: number;
  /** Head index */
  head: number;
  /** Induction score (0-1) */
  score: number;
}

/**
 * Configuration for steering vector application.
 */
export interface SteeringConfig {
  /** Steering strength (typically 0.5 to 2.0) */
  alpha: number;
  /** Which token positions to apply steering to */
  positions: 'last' | 'all' | number[];
  /** Layer at which to inject steering */
  layer: number;
}
