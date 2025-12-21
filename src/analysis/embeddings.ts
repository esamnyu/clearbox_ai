/**
 * Embedding and hidden state analysis functions.
 *
 * This module contains functions for analyzing token embeddings
 * and hidden states across layers.
 *
 * @module analysis/embeddings
 */

import { TensorView } from './utils/tensor';

/**
 * Analyzes layer norms across all transformer layers.
 *
 * Computes the L2 norm of hidden states at each layer. This reveals
 * how much information accumulates in the residual stream.
 *
 * @param hiddenStates - Map from layer index to hidden state tensor
 * @param position - Which token position to analyze (-1 = last token)
 * @returns Array of norms, one per layer
 *
 * @math_note
 *   For residual stream: h_{l+1} = h_l + f_l(h_l)
 *   We expect ||h_l|| to grow monotonically because each layer
 *   adds to the stream.
 *
 * @reference
 *   https://transformer-circuits.pub/2021/framework/index.html
 *
 * @example
 * ```typescript
 * const norms = analyzeLayerNorms(hiddenStates, -1);
 * // [12.3, 14.1, 15.8, ..., 23.4]
 * // Shows monotonic growth pattern
 * ```
 */
export function analyzeLayerNorms(
  hiddenStates: Map<number, TensorView>,  // layer → [batch, seq, hidden_dim]
  position: number = -1
): number[] {
  const norms: number[] = [];
  const numLayers = 12;  // GPT-2 has 12 layers

  for (let layer = 0; layer < numLayers; layer++) {
    const hidden = hiddenStates.get(layer);

    if (!hidden) {
      norms.push(0);
      continue;
    }

    // Extract the specified token position
    // hidden shape: [batch, seq, hidden_dim]
    // We want: [hidden_dim]
    const batchIdx = 0;  // Assume batch size = 1
    const seqLen = hidden.shape[1];
    const posIdx = position < 0 ? seqLen + position : position;

    // RESEARCHER TODO: Use proper slicing once implemented
    // For now, manually extract the token
    const hiddenDim = hidden.shape[2];
    const offset = batchIdx * seqLen * hiddenDim + posIdx * hiddenDim;
    const tokenHidden = new Float32Array(hiddenDim);

    for (let i = 0; i < hiddenDim; i++) {
      tokenHidden[i] = hidden.data[offset + i];
    }

    const tokenTensor = new TensorView(tokenHidden, [hiddenDim]);
    const norm = tokenTensor.norm();
    norms.push(norm);
  }

  return norms;
}

/**
 * Computes cosine similarity between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity (-1 to 1)
 *
 * @math_note
 *   cos(a, b) = (a · b) / (||a|| * ||b||)
 *
 * RESEARCHER TODO: Implement using TensorView.dot() once available
 */
export function cosineSimilarity(a: TensorView, b: TensorView): number {
  if (a.shape.length !== 1 || b.shape.length !== 1) {
    throw new Error('cosineSimilarity requires 1D tensors');
  }

  if (a.shape[0] !== b.shape[0]) {
    throw new Error('Vectors must have same dimension');
  }

  // Manual dot product
  let dotProduct = 0;
  for (let i = 0; i < a.data.length; i++) {
    dotProduct += a.data[i] * b.data[i];
  }

  const normA = a.norm();
  const normB = b.norm();

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

/**
 * Computes pairwise cosine similarities between all embeddings.
 *
 * @param embeddings - Embedding matrix [num_tokens, embedding_dim]
 * @returns Similarity matrix [num_tokens, num_tokens]
 *
 * @example
 * ```typescript
 * const embeddings = new TensorView(data, [4, 768]);  // 4 tokens
 * const sim = pairwiseSimilarity(embeddings);  // [4, 4] matrix
 * // sim[i][j] = similarity between token i and token j
 * ```
 *
 * RESEARCHER TODO: Implement pairwise similarity computation
 */
export function pairwiseSimilarity(
  embeddings: TensorView  // [tokens, dim]
): TensorView {
  // RESEARCHER TODO: Implement
  //
  // For each pair (i, j):
  //   sim[i][j] = cosineSimilarity(embeddings[i], embeddings[j])
  //
  // Result shape: [num_tokens, num_tokens]
  // Diagonal should be 1.0 (token similar to itself)

  throw new Error('pairwiseSimilarity not yet implemented');
}

/**
 * Projects high-dimensional embeddings to 3D using PCA.
 *
 * Principal Component Analysis finds the directions of maximum variance
 * in the data and projects onto the top-k components.
 *
 * @param embeddings - Embedding matrix [num_tokens, embedding_dim]
 * @param numComponents - Number of components to keep (default 3 for visualization)
 * @returns Projected embeddings [num_tokens, numComponents]
 *
 * @math_note
 *   1. Center the data: X' = X - mean(X)
 *   2. Compute covariance: C = X'ᵀX' / n
 *   3. Eigendecomposition: C = VΛVᵀ
 *   4. Project: Y = X' @ V[:, :k]
 *
 * @note
 *   This is a simplified PCA. For production, consider using
 *   power iteration or randomized SVD for large dimensions.
 *
 * RESEARCHER TODO: Implement PCA
 * You can use power iteration to find top eigenvectors, or
 * implement simplified SVD for 2D/3D projection only.
 *
 * @reference
 *   https://en.wikipedia.org/wiki/Principal_component_analysis
 */
export function computePCA(
  embeddings: TensorView,  // [tokens, hidden_dim]
  numComponents: number = 3
): TensorView {
  const [numTokens, hiddenDim] = embeddings.shape;

  if (numComponents > Math.min(numTokens, hiddenDim)) {
    throw new Error(
      `Cannot extract ${numComponents} components from ${numTokens} tokens with ${hiddenDim} dimensions`
    );
  }

  // RESEARCHER TODO: Implement PCA
  //
  // Simplified approach for 3D visualization:
  // 1. Center the data (subtract mean)
  // 2. Use power iteration to find top 3 eigenvectors
  // 3. Project data onto these vectors
  //
  // Alternatively, for quick prototyping:
  // Just return the first 3 dimensions (not true PCA but works for demo)

  // Placeholder: return first numComponents dimensions
  console.warn('PCA not implemented - returning first dimensions');

  // Extract first numComponents columns
  const result = new Float32Array(numTokens * numComponents);
  for (let i = 0; i < numTokens; i++) {
    for (let j = 0; j < numComponents; j++) {
      result[i * numComponents + j] = embeddings.data[i * hiddenDim + j];
    }
  }

  return new TensorView(result, [numTokens, numComponents]);
}

/**
 * Finds the k nearest neighbors for each token.
 *
 * @param embeddings - Embedding matrix [num_tokens, embedding_dim]
 * @param k - Number of neighbors to return
 * @returns Array of arrays: neighbors[i] = indices of k nearest tokens to i
 *
 * @example
 * ```typescript
 * const embeddings = new TensorView(data, [100, 768]);
 * const neighbors = findNearestNeighbors(embeddings, 5);
 * // neighbors[0] = [23, 45, 67, 12, 89]  (5 nearest to token 0)
 * ```
 *
 * RESEARCHER TODO: Implement k-NN search
 * Use cosine similarity as distance metric
 */
export function findNearestNeighbors(
  embeddings: TensorView,  // [tokens, dim]
  k: number
): number[][] {
  // RESEARCHER TODO: Implement
  //
  // For each token i:
  //   Compute similarity to all other tokens
  //   Sort by similarity (descending)
  //   Return indices of top-k (excluding i itself)

  throw new Error('findNearestNeighbors not yet implemented');
}

/**
 * Computes mean embedding for a set of tokens.
 *
 * Useful for computing class prototypes or steering vectors.
 *
 * @param embeddings - Embedding matrix [num_tokens, embedding_dim]
 * @param indices - Which tokens to average (null = all)
 * @returns Mean embedding [embedding_dim]
 */
export function meanEmbedding(
  embeddings: TensorView,  // [tokens, dim]
  indices: number[] | null = null
): TensorView {
  const [numTokens, embeddingDim] = embeddings.shape;
  const tokensToAverage = indices ?? Array.from({ length: numTokens }, (_, i) => i);

  const result = new Float32Array(embeddingDim);

  for (const idx of tokensToAverage) {
    for (let d = 0; d < embeddingDim; d++) {
      result[d] += embeddings.data[idx * embeddingDim + d];
    }
  }

  for (let d = 0; d < embeddingDim; d++) {
    result[d] /= tokensToAverage.length;
  }

  return new TensorView(result, [embeddingDim]);
}
