/**
 * Mock Pipeline Factory for Testing
 *
 * Implements PipelineFactory with test controls for:
 * - Simulating load delays
 * - Simulating failures
 * - Tracking call counts
 * - Firing progress callbacks
 */

import type {
  PipelineFactory,
  PipelineInterface,
  PipelineConfig,
  GenerationConfig,
  GenerationOutput,
  ModelId,
} from '../../src/engine/types'

import { createMockTokenizer } from './mockTokenizer'
import { MOCK_LOAD_DELAY_MS } from './constants'

// ============================================================================
// MOCK PIPELINE
// ============================================================================

export interface MockPipelineOptions {
  /** Default generated text response */
  defaultOutput?: string
}

/**
 * Creates a mock pipeline that implements PipelineInterface.
 */
export function createMockPipeline(options: MockPipelineOptions = {}): PipelineInterface {
  const { defaultOutput = 'Mock generated text.' } = options

  const tokenizer = createMockTokenizer()

  // Pipeline is a callable function with tokenizer attached
  const pipeline = async function (
    prompt: string,
    _config: GenerationConfig
  ): Promise<GenerationOutput[]> {
    return [{ generated_text: prompt + ' ' + defaultOutput }]
  } as PipelineInterface

  // Attach tokenizer
  pipeline.tokenizer = tokenizer

  return pipeline
}

// ============================================================================
// MOCK PIPELINE FACTORY
// ============================================================================

export interface MockPipelineFactory extends PipelineFactory {
  // Test controls
  setShouldFail(fail: boolean, error?: Error): void
  setLoadDelay(ms: number): void
  getCreateCallCount(): number
  getLastCreateCall(): { task: string; modelId: string; config: PipelineConfig } | null
  reset(): void
}

export interface MockPipelineFactoryOptions {
  /** Initial load delay in ms (default: MOCK_LOAD_DELAY_MS) */
  loadDelay?: number
  /** Should factory fail on create? */
  shouldFail?: boolean
  /** Error to throw when failing */
  failError?: Error
}

/**
 * Creates a mock pipeline factory with test controls.
 *
 * Usage:
 * ```typescript
 * const factory = createMockPipelineFactory()
 *
 * // Normal usage
 * const pipeline = await factory.create('text-generation', 'Xenova/gpt2', config)
 *
 * // Configure to fail
 * factory.setShouldFail(true, new Error('Network error'))
 *
 * // Add delay to test loading state
 * factory.setLoadDelay(100)
 *
 * // Check how many times create was called
 * expect(factory.getCreateCallCount()).toBe(1)
 * ```
 */
export function createMockPipelineFactory(
  options: MockPipelineFactoryOptions = {}
): MockPipelineFactory {
  // Internal state
  let shouldFail = options.shouldFail ?? false
  let failError = options.failError ?? new Error('Mock pipeline factory error')
  let loadDelay = options.loadDelay ?? MOCK_LOAD_DELAY_MS
  let createCallCount = 0
  let lastCreateCall: { task: string; modelId: string; config: PipelineConfig } | null = null

  const factory: MockPipelineFactory = {
    async create(
      task: 'text-generation',
      modelId: ModelId,
      config: PipelineConfig
    ): Promise<PipelineInterface> {
      createCallCount++
      lastCreateCall = { task, modelId, config }

      // Simulate async loading delay
      if (loadDelay > 0) {
        await new Promise((resolve) => setTimeout(resolve, loadDelay))
      }

      // Fire progress callbacks if provided
      if (config.progress_callback) {
        config.progress_callback({
          status: 'downloading',
          file: 'model.safetensors',
          progress: 50,
        })
        config.progress_callback({
          status: 'loading',
          progress: 100,
        })
        config.progress_callback({
          status: 'ready',
        })
      }

      // Simulate failure if configured
      if (shouldFail) {
        throw failError
      }

      return createMockPipeline()
    },

    // Test controls
    setShouldFail(fail: boolean, error?: Error): void {
      shouldFail = fail
      if (error) failError = error
    },

    setLoadDelay(ms: number): void {
      loadDelay = ms
    },

    getCreateCallCount(): number {
      return createCallCount
    },

    getLastCreateCall() {
      return lastCreateCall
    },

    reset(): void {
      shouldFail = false
      failError = new Error('Mock pipeline factory error')
      loadDelay = MOCK_LOAD_DELAY_MS
      createCallCount = 0
      lastCreateCall = null
    },
  }

  return factory
}

// ============================================================================
// DEFAULT MOCK FACTORY INSTANCE
// ============================================================================

/**
 * Pre-configured mock factory with default settings.
 * Call reset() between tests to clear state.
 */
export const defaultMockPipelineFactory = createMockPipelineFactory()
