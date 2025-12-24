/**
 * Mock Tokenizer for Testing
 *
 * Implements TokenizerInterface with deterministic, instant responses.
 * No model loading required.
 */

import type {
  TokenizerInterface,
  TokenizerOptions,
  DecodeOptions,
  EncodedTokens,
  Tensor,
} from '../../src/engine/types'

import { KNOWN_TOKENS, KNOWN_TOKEN_STRINGS } from './constants'

// ============================================================================
// HELPER: Create a mock Tensor
// ============================================================================

function createMockTensor(ids: number[]): Tensor {
  return {
    data: BigInt64Array.from(ids.map(BigInt)),
    dims: [1, ids.length],
    type: 'int64',
    size: ids.length,
  }
}

// ============================================================================
// MOCK TOKENIZER FACTORY
// ============================================================================

export interface MockTokenizerOptions {
  /** Additional token mappings beyond KNOWN_TOKENS */
  customMappings?: Record<string, number[]>
  /** Starting ID for unknown tokens (default: 1000) */
  unknownTokenStartId?: number
}

/**
 * Creates a mock tokenizer that implements TokenizerInterface.
 *
 * Behavior:
 * - Known inputs (from KNOWN_TOKENS) return predetermined IDs
 * - Unknown inputs are split by space, each word gets a sequential ID
 * - decode() returns "[token-{id}]" for unknown IDs
 */
export function createMockTokenizer(options: MockTokenizerOptions = {}): TokenizerInterface {
  const { customMappings = {}, unknownTokenStartId = 1000 } = options

  // Merge known tokens with custom mappings
  const tokenMappings = { ...KNOWN_TOKENS, ...customMappings }

  // Track next ID for unknown tokens
  let nextUnknownId = unknownTokenStartId

  // The tokenizer is a callable function with a decode method attached
  const tokenizer = async function (
    text: string,
    _options: TokenizerOptions
  ): Promise<EncodedTokens> {
    let tokenIds: number[]

    if (text in tokenMappings) {
      // Use known mapping
      tokenIds = tokenMappings[text]
    } else {
      // Split by space and assign sequential IDs
      const words = text.split(/\s+/).filter(Boolean)
      tokenIds = words.map(() => nextUnknownId++)
    }

    return {
      input_ids: createMockTensor(tokenIds),
      attention_mask: createMockTensor(tokenIds.map(() => 1)),
    }
  } as TokenizerInterface

  // Attach decode method
  tokenizer.decode = function (
    ids: number[] | bigint[],
    options?: DecodeOptions
  ): string {
    const skipSpecial = options?.skip_special_tokens ?? true

    // Convert to array of numbers (handles both number[] and bigint[])
    const numericIds = (ids as (number | bigint)[]).map((id) =>
      typeof id === 'bigint' ? Number(id) : id
    )

    return numericIds
      .map((numId) => {

        // Skip special tokens (ID 0 is often EOS/padding)
        if (skipSpecial && numId === 0) return ''

        // Check known token strings
        if (numId in KNOWN_TOKEN_STRINGS) {
          return KNOWN_TOKEN_STRINGS[numId]
        }

        // Unknown token
        return `[token-${numId}]`
      })
      .join('')
  }

  return tokenizer
}

// ============================================================================
// DEFAULT MOCK TOKENIZER INSTANCE
// ============================================================================

/**
 * Pre-configured mock tokenizer with default settings.
 * Use createMockTokenizer() for custom configurations.
 */
export const defaultMockTokenizer = createMockTokenizer()
