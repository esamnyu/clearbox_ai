/**
 * Unit Tests: worker.loadModel
 *
 * Tests the model loading state machine without loading the real 500MB model.
 * Uses mock pipeline factory for instant, deterministic responses.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  createMockPipelineFactory,
  type MockPipelineFactory,
} from '../fixtures/mockPipelineFactory'
import type { LoadProgress } from '../../src/engine/types'

/**
 * NOTE: In a real setup, we'd import the worker via Comlink.
 * For now, we test the factory and state logic directly.
 *
 * Full worker integration would require:
 * 1. A test harness that creates the worker
 * 2. Comlink.wrap() to get the API
 * 3. Calling _setPipelineFactory with our mock
 *
 * This file tests the mock infrastructure itself first.
 */

describe('MockPipelineFactory', () => {
  let factory: MockPipelineFactory

  beforeEach(() => {
    factory = createMockPipelineFactory()
    factory.reset()
  })

  // =========================================================================
  // BASIC FUNCTIONALITY
  // =========================================================================

  describe('basic functionality', () => {
    it('creates a pipeline successfully', async () => {
      const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      expect(pipeline).toBeDefined()
      expect(pipeline.tokenizer).toBeDefined()
    })

    it('tracks create call count', async () => {
      expect(factory.getCreateCallCount()).toBe(0)

      await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      expect(factory.getCreateCallCount()).toBe(1)

      await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      expect(factory.getCreateCallCount()).toBe(2)
    })

    it('records last create call parameters', async () => {
      await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      const lastCall = factory.getLastCreateCall()
      expect(lastCall).not.toBeNull()
      expect(lastCall?.task).toBe('text-generation')
      expect(lastCall?.modelId).toBe('Xenova/gpt2')
      expect(lastCall?.config.dtype).toBe('fp32')
    })
  })

  // =========================================================================
  // PROGRESS CALLBACKS
  // =========================================================================

  describe('progress callbacks', () => {
    it('fires progress callbacks when provided', async () => {
      const progressEvents: LoadProgress[] = []

      await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
        progress_callback: (progress) => {
          progressEvents.push(progress)
        },
      })

      expect(progressEvents.length).toBeGreaterThan(0)
      expect(progressEvents.some((p) => p.status === 'downloading')).toBe(true)
      expect(progressEvents.some((p) => p.status === 'ready')).toBe(true)
    })
  })

  // =========================================================================
  // FAILURE SIMULATION
  // =========================================================================

  describe('failure simulation', () => {
    it('throws when configured to fail', async () => {
      factory.setShouldFail(true, new Error('Network timeout'))

      await expect(
        factory.create('text-generation', 'Xenova/gpt2', {
          dtype: 'fp32',
          device: 'wasm',
        })
      ).rejects.toThrow('Network timeout')
    })

    it('succeeds after reset', async () => {
      factory.setShouldFail(true)

      await expect(
        factory.create('text-generation', 'Xenova/gpt2', {
          dtype: 'fp32',
          device: 'wasm',
        })
      ).rejects.toThrow()

      factory.reset()

      const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      expect(pipeline).toBeDefined()
    })
  })

  // =========================================================================
  // DELAY SIMULATION
  // =========================================================================

  describe('delay simulation', () => {
    it('respects configured load delay', async () => {
      factory.setLoadDelay(50)

      const start = Date.now()
      await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })
      const elapsed = Date.now() - start

      expect(elapsed).toBeGreaterThanOrEqual(45) // Allow small timing variance
    })
  })

  // =========================================================================
  // MOCK PIPELINE BEHAVIOR
  // =========================================================================

  describe('mock pipeline behavior', () => {
    it('pipeline generates text', async () => {
      const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      const output = await pipeline('Hello', { max_new_tokens: 5 })

      expect(output).toHaveLength(1)
      expect(output[0].generated_text).toContain('Hello')
    })

    it('tokenizer encodes text', async () => {
      const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      const encoded = await pipeline.tokenizer('Hello', {})

      expect(encoded.input_ids).toBeDefined()
      expect(encoded.input_ids.data.length).toBeGreaterThan(0)
    })

    it('tokenizer decodes ids', async () => {
      const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
        dtype: 'fp32',
        device: 'wasm',
      })

      const decoded = pipeline.tokenizer.decode([15496])

      expect(decoded).toBe('Hello')
    })
  })
})

// =========================================================================
// MOCK TOKENIZER TESTS
// =========================================================================

describe('MockTokenizer', () => {
  it('returns known token IDs for known inputs', async () => {
    const factory = createMockPipelineFactory()
    const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
      dtype: 'fp32',
      device: 'wasm',
    })

    const encoded = await pipeline.tokenizer('Hello', {})
    const ids = Array.from(encoded.input_ids.data).map(Number)

    expect(ids).toEqual([15496])
  })

  it('generates sequential IDs for unknown inputs', async () => {
    const factory = createMockPipelineFactory()
    const pipeline = await factory.create('text-generation', 'Xenova/gpt2', {
      dtype: 'fp32',
      device: 'wasm',
    })

    const encoded = await pipeline.tokenizer('unknown text here', {})
    const ids = Array.from(encoded.input_ids.data).map(Number)

    expect(ids.length).toBe(3) // "unknown", "text", "here"
    expect(ids[0]).toBeGreaterThanOrEqual(1000) // Unknown IDs start at 1000
  })
})
