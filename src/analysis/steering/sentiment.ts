/**
 * Sentiment steering vectors.
 *
 * This module defines steering vectors for manipulating model sentiment.
 * Steering vectors are computed as the difference between mean activations
 * for positive vs. negative examples.
 *
 * @module analysis/steering/sentiment
 */

import { TensorView } from '../utils/tensor';

/**
 * Example prompts for computing sentiment steering vectors.
 *
 * These prompts are used by computeSentimentSteering() to create contrastive
 * pairs. The steering vector is computed as:
 *   v = mean(positive_activations) - mean(negative_activations)
 *
 * RESEARCHER TODO: Expand these lists with more diverse examples
 */
export const POSITIVE_PROMPTS = [
  'I love this! It\'s absolutely wonderful',
  'This is the best day ever',
  'I\'m so happy and grateful',
  'Everything is going perfectly',
  'This makes me feel amazing',
];

export const NEGATIVE_PROMPTS = [
  'I hate this. It\'s absolutely terrible',
  'This is the worst day ever',
  'I\'m so sad and disappointed',
  'Everything is going wrong',
  'This makes me feel awful',
];

/**
 * Interface for retrieving hidden states from the model.
 * The engineer will implement this and pass it to steering functions.
 */
export interface HiddenStateRetriever {
  /**
   * Get hidden state at a specific layer for a given prompt.
   *
   * @param prompt - The text to encode
   * @param layer - Which transformer layer (0-11 for GPT-2)
   * @param position - Which token position (-1 = last token)
   * @returns Hidden state tensor [hidden_dim]
   */
  getHiddenState(
    prompt: string,
    layer: number,
    position: number
  ): Promise<TensorView>;
}

/**
 * Computes a sentiment steering vector.
 *
 * Method: Compute the mean activation for positive examples and
 * negative examples, then take the difference. This direction in
 * activation space represents "positive sentiment".
 *
 * @param retriever - Interface for getting hidden states from model
 * @param layer - Which layer to extract steering vector from (default 6)
 * @returns Steering vector [hidden_dim]
 *
 * @math_note
 *   v = mean(positive_activations) - mean(negative_activations)
 *   When added to activations: h' = h + α*v̂ (v̂ = normalized v)
 *
 * @example
 * ```typescript
 * const steeringVector = await computeSentimentSteering(retriever, 6);
 * // Use with α=1.5 to push toward positive sentiment
 * ```
 *
 * RESEARCHER TODO: Implement steering vector computation
 */
export async function computeSentimentSteering(
  _retriever: HiddenStateRetriever,  // Prefixed: unused in placeholder
  _layer: number = 6                  // Prefixed: unused in placeholder
): Promise<TensorView> {
  // RESEARCHER TODO: Implement
  //
  // Algorithm:
  // 1. For each positive prompt:
  //    - Get hidden state at specified layer, last token position
  //    - Store in array
  //
  // 2. For each negative prompt:
  //    - Get hidden state at specified layer, last token position
  //    - Store in array
  //
  // 3. Compute mean of positive activations
  // 4. Compute mean of negative activations
  // 5. Steering vector = positive_mean - negative_mean
  // 6. Normalize the vector
  //
  // Hint: Use meanEmbedding() from embeddings.ts

  throw new Error('computeSentimentSteering not yet implemented');
}

/**
 * Pre-computed sentiment steering vector for GPT-2.
 *
 * This was computed offline using the method above. In production,
 * you would save steering vectors as JSON and load them.
 *
 * RESEARCHER TODO: Replace with actual computed vector
 */
export const SENTIMENT_VECTOR_GPT2: number[] = [
  // Placeholder: 768 zeros
  ...Array(768).fill(0),
];

/**
 * Validates a steering vector.
 *
 * Checks that:
 * - Vector has correct dimensionality
 * - Values are in reasonable range
 * - No NaN or Infinity
 *
 * @param vector - Steering vector to validate
 * @param expectedDim - Expected dimensionality (768 for GPT-2)
 * @returns True if valid
 */
export function validateSteeringVector(
  vector: TensorView,
  expectedDim: number = 768
): boolean {
  if (vector.shape.length !== 1) {
    console.error('Steering vector must be 1D');
    return false;
  }

  if (vector.shape[0] !== expectedDim) {
    console.error(`Expected dimension ${expectedDim}, got ${vector.shape[0]}`);
    return false;
  }

  // Check for bad values
  for (let i = 0; i < vector.data.length; i++) {
    if (!isFinite(vector.data[i])) {
      console.error(`Bad value at index ${i}: ${vector.data[i]}`);
      return false;
    }
  }

  // Check if vector is not all zeros
  const norm = vector.norm();
  if (norm < 1e-6) {
    console.error('Steering vector is effectively zero');
    return false;
  }

  return true;
}

/**
 * Saves a steering vector to JSON format.
 *
 * @param vector - Steering vector to save
 * @param metadata - Additional information (layer, model, etc.)
 * @returns JSON string
 */
export function serializeSteeringVector(
  vector: TensorView,
  metadata: {
    name: string;
    description: string;
    model: string;
    layer: number;
  }
): string {
  return JSON.stringify(
    {
      ...metadata,
      data: Array.from(vector.data),
      shape: vector.shape,
      norm: vector.norm(),
    },
    null,
    2
  );
}

/**
 * Loads a steering vector from JSON.
 *
 * @param json - JSON string from serializeSteeringVector
 * @returns Steering vector
 */
export function deserializeSteeringVector(json: string): {
  vector: TensorView;
  metadata: {
    name: string;
    description: string;
    model: string;
    layer: number;
  };
} {
  const parsed = JSON.parse(json);

  return {
    vector: new TensorView(new Float32Array(parsed.data), parsed.shape),
    metadata: {
      name: parsed.name,
      description: parsed.description,
      model: parsed.model,
      layer: parsed.layer,
    },
  };
}
