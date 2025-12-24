/**
 * Test constants for NeuroScope-Web
 *
 * Shared values used across all test files.
 */

// ============================================================================
// GPT-2 MODEL CONSTANTS
// ============================================================================

export const GPT2_VOCAB_SIZE = 50257
export const GPT2_HIDDEN_DIM = 768
export const GPT2_NUM_LAYERS = 12
export const GPT2_NUM_HEADS = 12

// ============================================================================
// TEST TIMEOUTS
// ============================================================================

export const MOCK_LOAD_DELAY_MS = 10   // Simulate brief async delay in mocks
export const TEST_TIMEOUT_MS = 5000    // Max time for unit tests

// ============================================================================
// KNOWN TOKEN MAPPINGS
// ============================================================================

/**
 * Pre-defined token mappings for deterministic tests.
 * These match real GPT-2 tokenizer output.
 */
export const KNOWN_TOKENS: Record<string, number[]> = {
  'Hello': [15496],
  'Hello, world!': [15496, 11, 995, 0],
  ' world': [995],
  'The': [464],
  'test': [9288],
}

/**
 * Reverse mapping: token ID → string representation
 */
export const KNOWN_TOKEN_STRINGS: Record<number, string> = {
  15496: 'Hello',
  11: ',',
  995: ' world',
  0: '!',
  464: 'The',
  9288: 'test',
}
