/**
 * Attention Pattern Analysis
 * ==========================
 *
 * This module provides tools for understanding *what* attention heads do in
 * transformer language models. Rather than treating attention as a black box,
 * we can identify interpretable patterns: some heads focus on the previous token,
 * others broadcast information globally, and some implement sophisticated
 * in-context learning via "induction heads."
 *
 * ## Why Analyze Attention?
 *
 * Attention weights tell us which tokens a model "looks at" when processing
 * each position. By studying these patterns, we gain insight into the model's
 * internal algorithms. This is the foundation of mechanistic interpretability.
 *
 * ## Key Concepts
 *
 * - **Induction Heads**: Heads that copy patterns from context. If the model
 *   has seen "Harry Potter" before and encounters "Harry" again, an induction
 *   head will attend to "Potter" to predict the next token.
 *
 * - **Attention Entropy**: A measure of how "spread out" attention is. Low
 *   entropy means focused attention; high entropy means diffuse attention.
 *
 * - **Pattern Classification**: We can automatically categorize heads by their
 *   dominant behavior (previous-token, first-token, broadcasting, etc.).
 *
 * @module analysis/attention
 *
 * @reference
 *   Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
 *   https://transformer-circuits.pub/2021/framework/
 *
 *   Olsson et al., "In-context Learning and Induction Heads" (2022)
 *   https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
 */

import { TensorView } from './utils/tensor';

/**
 * Attention pattern types that can be automatically detected.
 */
export type AttentionPattern =
  | 'previous_token'  // Attends to token at position i-1
  | 'first_token'     // Attends to first token (often BOS/CLS)
  | 'induction'       // Copies pattern (if "A B ... A" seen, predicts "B")
  | 'broadcasting'    // Uniform attention across all positions
  | 'other';          // No clear pattern

/**
 * Result of induction head detection.
 *
 * When returned from `detectInductionHeads`, only `head` and `score` are set.
 * When returned from `analyzeAllLayers`, `layer` is also included.
 */
export interface InductionHeadResult {
  /** Layer index (only present when analyzing multiple layers) */
  layer?: number;
  /** Head index within the layer */
  head: number;
  /** Induction score from 0 (no induction) to 1 (perfect induction) */
  score: number;
}

/**
 * Compute the norm of activations.
 *
 * Simple example function to demonstrate the pattern.
 *
 * @param hidden - Hidden states [batch, seq_len, hidden_dim]
 * @returns Scalar norm value
 *
 * @example
 * ```typescript
 * const hidden = new TensorView(data, [1, 10, 768]);
 * const norm = computeNorm(hidden);  // e.g., 15.2
 * ```
 */
export function computeNorm(hidden: TensorView): number {
  return hidden.norm();
}

/**
 * Detect Induction Heads — Finding Pattern-Copying Attention Heads
 *
 * ## Why This Matters
 *
 * Induction heads are among the most important circuits discovered in transformer
 * language models. They implement a deceptively simple but powerful algorithm:
 *
 * > "If I've seen token A followed by token B somewhere in the past, and I see
 * > token A again now, I should predict token B."
 *
 * This is the mechanism behind in-context learning. When you show GPT-2 examples
 * like "capital of France is Paris, capital of Germany is Berlin, capital of Spain
 * is", an induction head notices the pattern "capital of X is Y" and copies it.
 *
 * ## How We Calculate It
 *
 * The key insight is that induction heads attend to position **j+1** (not j) when
 * they find a match. Here's why:
 *
 * ```
 * Position:  0      1       2       3       4      5       6
 * Token:   "The" "cat"   "sat"   "The"   "cat"  "???"
 *                  ↑               │
 *                  └───────────────┘
 *                  When at position 4 ("cat"), we found "cat" at position 1.
 *                  But we DON'T attend to position 1 — we attend to position 2 ("sat")!
 *                  This is j+1, the token that FOLLOWED the previous "cat".
 * ```
 *
 * For each head, we:
 * 1. Find all (i, j) pairs where tokens[i] = tokens[j] and j < i-1
 * 2. Check if attention[head, i, j+1] is high
 * 3. Average these values to get an "induction score"
 *
 * @math_note
 *   Let M = {(i, j) : tokens[i] = tokens[j] and 0 ≤ j < i-1}
 *
 *   induction_score(head) = (1/|M|) · Σ_{(i,j) ∈ M} attention[head, i, j+1]
 *
 *   We require j < i-1 (not just j < i) because j+1 must be a valid position
 *   that's still before i. If j = i-1, then j+1 = i, which is the query position
 *   itself — not useful for copying.
 *
 * ## Edge Cases
 *
 * - **No repeated tokens**: M is empty, we skip the head (score = 0)
 * - **Sequence too short** (< 3 tokens): Return empty array. We need at least
 *   positions 0, 1, 2 to have a valid induction pattern.
 * - **All same token**: Every position matches, but this is usually padding or
 *   a degenerate case — we still compute the score normally.
 *
 * @param attention - Attention tensor with shape [num_heads, seq_len, seq_len].
 *   attention[h, i, j] is the attention weight from query position i to key
 *   position j in head h.
 *
 * @param tokens - The input tokens as strings. We compare tokens for equality
 *   to find repeated patterns.
 *
 * @param threshold - Minimum induction score to include a head in the results.
 *   Default is 0.3. Scores range from 0 (no induction behavior) to 1 (perfect
 *   induction).
 *
 * @returns Array of InductionHeadResult objects, each containing:
 *   - head: The head index
 *   - score: The induction score (0-1)
 *   Sorted by score descending. Only heads exceeding threshold are included.
 *
 * @example
 * ```typescript
 * const attention = new TensorView(data, [12, 64, 64]);
 * const tokens = ["The", "cat", "sat", "The", "cat", "on"];
 * const results = detectInductionHeads(attention, tokens);
 * // results = [
 * //   { head: 5, score: 0.72 },  // Head 5 is a strong induction head
 * //   { head: 9, score: 0.45 },  // Head 9 shows some induction behavior
 * // ]
 * ```
 *
 * @reference
 *   Olsson et al., "In-context Learning and Induction Heads" (2022)
 *   https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
 */
export function detectInductionHeads(
  attention: TensorView,
  tokens: string[],
  threshold: number = 0.3
): InductionHeadResult[] {
  const [numHeads, seqLen, seqLenK] = attention.shape;

  // Validate shape: attention should be [heads, seq, seq]
  if (attention.shape.length !== 3 || seqLen !== seqLenK) {
    throw new Error(
      `Expected attention shape [heads, seq, seq], got [${attention.shape}]`
    );
  }

  // Validate tokens length matches sequence length
  if (tokens.length !== seqLen) {
    throw new Error(
      `Token count (${tokens.length}) doesn't match sequence length (${seqLen})`
    );
  }

  // Edge case: sequence too short for induction
  // We need at least 3 positions: j, j+1, and i where i > j+1
  if (seqLen < 3) {
    return [];
  }

  const results: InductionHeadResult[] = [];

  // ─────────────────────────────────────────────────────────────────────────
  // For each head, compute the induction score
  // ─────────────────────────────────────────────────────────────────────────
  for (let h = 0; h < numHeads; h++) {
    let inductionSum = 0;
    let matchCount = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // Scan for repeated tokens
    //
    // For each position i (starting from 2, since we need room for j and j+1):
    //   For each earlier position j (where j+1 < i):
    //     If tokens[i] === tokens[j]:
    //       This is a potential induction match!
    //       Check attention[h, i, j+1] — does the head attend to the
    //       token that FOLLOWED the previous occurrence?
    // ─────────────────────────────────────────────────────────────────────────
    for (let i = 2; i < seqLen; i++) {
      for (let j = 0; j < i - 1; j++) {
        // j < i-1 ensures j+1 < i (the key position is before the query)

        if (tokens[i] === tokens[j]) {
          // Found a match! Token at position i equals token at position j.
          //
          // The induction hypothesis: if this is an induction head, it should
          // attend from position i to position j+1 (the token AFTER the match).
          //
          // Why j+1? Because the model is trying to predict what comes after
          // the current token. If "cat" was followed by "sat" before, and we
          // see "cat" again, we want to look at "sat" to copy the pattern.

          const attendToJPlus1 = attention.get(h, i, j + 1);
          inductionSum += attendToJPlus1;
          matchCount++;
        }
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Compute average induction score for this head
    // ─────────────────────────────────────────────────────────────────────────
    if (matchCount > 0) {
      const inductionScore = inductionSum / matchCount;

      if (inductionScore >= threshold) {
        results.push({
          head: h,
          score: inductionScore,
        });
      }
    }
  }

  // Sort by score descending (strongest induction heads first)
  results.sort((a, b) => b.score - a.score);

  return results;
}

/**
 * Classify Attention Head — Identifying Dominant Attention Patterns
 *
 * ## Why This Matters
 *
 * Not all attention heads do the same thing. Through empirical study, researchers
 * have identified several common patterns that appear across transformer models:
 *
 * - **Previous Token Heads**: Attend strongly to position i-1. These help with
 *   local syntax and bigram statistics.
 *
 * - **First Token Heads**: Attend to position 0 (often a BOS/CLS token). These
 *   often serve as a "default" when there's nothing specific to attend to.
 *
 * - **Broadcasting Heads**: Spread attention uniformly across positions. These
 *   aggregate global context information.
 *
 * - **Induction Heads**: Copy patterns from earlier in the context. These are
 *   more complex and detected separately.
 *
 * ## How We Calculate It
 *
 * We compute a score for each pattern and return whichever scores highest:
 *
 * 1. **Previous-token score**: Average of attention[i, i-1] for i ∈ [1, seq_len)
 * 2. **First-token score**: Average of attention[i, 0] for all i
 * 3. **Broadcasting score**: Normalized entropy (high entropy = uniform = broadcasting)
 *
 * We compare these scores and return the dominant pattern. If no pattern scores
 * above a minimum threshold (0.3), we return 'other'.
 *
 * @math_note
 *   prev_score = (1/(n-1)) · Σᵢ₌₁ⁿ⁻¹ attention[i, i-1]
 *   first_score = (1/n) · Σᵢ₌₀ⁿ⁻¹ attention[i, 0]
 *   broadcast_score = entropy / log₂(n)    (normalized to [0, 1])
 *
 * ## Edge Cases
 *
 * - **Single token**: We return 'other' (can't determine pattern from one position)
 * - **All zeros**: We return 'other' (degenerate attention)
 * - **Tie between patterns**: We use priority order: previous_token > first_token > broadcasting
 *
 * @param attention - Single head attention matrix with shape [seq_len, seq_len].
 *   attention[i, j] is the attention weight from query position i to key position j.
 *
 * @param tokens - Input tokens. Currently unused but reserved for future
 *   induction pattern detection within this function.
 *
 * @returns The dominant attention pattern type.
 *
 * @example
 * ```typescript
 * // A head that mostly attends to the previous token
 * const pattern = classifyAttentionHead(headAttention, tokens);
 * // pattern === 'previous_token'
 * ```
 *
 * @reference
 *   Voita et al., "Analyzing Multi-Head Self-Attention" (2019)
 */
export function classifyAttentionHead(
  attention: TensorView,
  _tokens: string[]  // Reserved for future induction detection
): AttentionPattern {
  const seqLen = attention.shape[0];

  // Validate shape: attention should be [seq, seq]
  if (attention.shape.length !== 2 || attention.shape[0] !== attention.shape[1]) {
    throw new Error(
      `Expected attention shape [seq, seq], got [${attention.shape}]`
    );
  }

  // Edge case: single token sequence — can't determine pattern
  if (seqLen <= 1) {
    return 'other';
  }

  // Minimum score to classify as a specific pattern
  const MIN_THRESHOLD = 0.3;

  // Small epsilon for numerical stability
  const EPSILON = 1e-10;

  // ─────────────────────────────────────────────────────────────────────────
  // 1. Compute previous-token score
  //    We check: does this head attend strongly to position i-1?
  //    This is the subdiagonal of the attention matrix.
  // ─────────────────────────────────────────────────────────────────────────
  let prevTokenSum = 0;
  for (let i = 1; i < seqLen; i++) {
    // attention[i, i-1] is the weight from position i to position i-1
    prevTokenSum += attention.get(i, i - 1);
  }
  const prevTokenScore = prevTokenSum / (seqLen - 1);

  // ─────────────────────────────────────────────────────────────────────────
  // 2. Compute first-token score
  //    We check: does this head attend strongly to position 0?
  //    This is the first column of the attention matrix.
  // ─────────────────────────────────────────────────────────────────────────
  let firstTokenSum = 0;
  for (let i = 0; i < seqLen; i++) {
    // attention[i, 0] is the weight from position i to position 0
    firstTokenSum += attention.get(i, 0);
  }
  const firstTokenScore = firstTokenSum / seqLen;

  // ─────────────────────────────────────────────────────────────────────────
  // 3. Compute broadcasting score (normalized entropy)
  //    High entropy means attention is spread uniformly → broadcasting pattern
  //    We normalize by maximum possible entropy: log₂(seq_len)
  // ─────────────────────────────────────────────────────────────────────────
  let totalEntropy = 0;
  for (let i = 0; i < seqLen; i++) {
    let rowEntropy = 0;
    for (let j = 0; j < seqLen; j++) {
      const p = attention.get(i, j);
      if (p > EPSILON) {
        rowEntropy += -p * Math.log2(p);
      }
    }
    totalEntropy += rowEntropy;
  }
  const avgEntropy = totalEntropy / seqLen;
  const maxEntropy = Math.log2(seqLen);
  const broadcastScore = avgEntropy / maxEntropy;  // Normalized to [0, 1]

  // ─────────────────────────────────────────────────────────────────────────
  // 4. Determine dominant pattern
  //    We return the pattern with the highest score, if it exceeds threshold.
  //    Priority for ties: previous_token > first_token > broadcasting
  // ─────────────────────────────────────────────────────────────────────────

  // Find the maximum score
  const scores: [AttentionPattern, number][] = [
    ['previous_token', prevTokenScore],
    ['first_token', firstTokenScore],
    ['broadcasting', broadcastScore],
  ];

  // Sort by score descending (stable sort preserves priority order for ties)
  scores.sort((a, b) => b[1] - a[1]);

  const [bestPattern, bestScore] = scores[0];

  // Only classify if score exceeds threshold
  if (bestScore >= MIN_THRESHOLD) {
    return bestPattern;
  }

  return 'other';
}

/**
 * Compute Attention Entropy — Measuring How "Spread Out" Attention Is
 *
 * ## Why This Matters
 *
 * Entropy tells us whether an attention head is *focused* (low entropy) or
 * *diffuse* (high entropy). A head that always attends to one specific position
 * has entropy ≈ 0. A head that spreads attention uniformly across all positions
 * has maximum entropy = log₂(seq_len).
 *
 * This distinction is crucial for understanding head behavior:
 * - **Low entropy heads** implement specific operations (e.g., "attend to previous token")
 * - **High entropy heads** often aggregate information globally (e.g., "broadcasting")
 *
 * ## How We Calculate It
 *
 * We compute Shannon entropy for each query position's attention distribution,
 * then average over all positions. Shannon entropy measures the "uncertainty"
 * or "spread" of a probability distribution.
 *
 * @math_note
 *   For a single query position with attention weights p = [p₁, p₂, ..., pₙ]:
 *
 *     H(p) = -Σᵢ pᵢ · log₂(pᵢ)
 *
 *   We average H over all query positions to get the head's entropy.
 *   Maximum entropy (uniform attention) = log₂(seq_len).
 *
 * ## Edge Cases
 *
 * - **log(0) is undefined**: We skip terms where p < ε (1e-10). This is safe
 *   because lim(p→0) p·log(p) = 0.
 * - **Empty sequence**: We return a tensor of zeros.
 *
 * @param attention - Attention tensor with shape [num_heads, seq_len, seq_len].
 *   Each attention[h, q, :] is a probability distribution over key positions
 *   (should sum to 1, as output by softmax).
 *
 * @returns Entropy per head as a TensorView with shape [num_heads].
 *   Values range from 0 (perfectly focused) to log₂(seq_len) (uniform).
 *
 * @example
 * ```typescript
 * // Attention tensor: 12 heads, 64 tokens
 * const attention = new TensorView(data, [12, 64, 64]);
 * const entropy = computeAttentionEntropy(attention);
 * // entropy.shape = [12]
 * // entropy.get(0) might be 2.3 (fairly focused)
 * // entropy.get(5) might be 5.8 (very diffuse)
 * ```
 *
 * @reference
 *   Shannon, "A Mathematical Theory of Communication" (1948)
 */
export function computeAttentionEntropy(attention: TensorView): TensorView {
  const [numHeads, seqLen, seqLenK] = attention.shape;

  // Validate shape: attention should be [heads, seq, seq]
  if (attention.shape.length !== 3 || seqLen !== seqLenK) {
    throw new Error(
      `Expected attention shape [heads, seq, seq], got [${attention.shape}]`
    );
  }

  // Edge case: empty sequence
  if (seqLen === 0) {
    return TensorView.zeros([numHeads]);
  }

  // Small epsilon to avoid log(0)
  const EPSILON = 1e-10;

  // We'll compute entropy for each head
  const entropyData = new Float32Array(numHeads);

  for (let h = 0; h < numHeads; h++) {
    let headEntropy = 0;

    // For each query position, compute the entropy of its attention distribution
    for (let q = 0; q < seqLen; q++) {
      let rowEntropy = 0;

      // Sum -p * log2(p) over all key positions
      for (let k = 0; k < seqLen; k++) {
        const p = attention.get(h, q, k);

        // Skip near-zero probabilities to avoid log(0)
        // Note: lim(p→0) p·log(p) = 0, so this is mathematically sound
        if (p > EPSILON) {
          // Shannon entropy uses log base 2 (bits)
          rowEntropy += -p * Math.log2(p);
        }
      }

      headEntropy += rowEntropy;
    }

    // Average entropy over all query positions
    entropyData[h] = headEntropy / seqLen;
  }

  return new TensorView(entropyData, [numHeads]);
}

/**
 * Analyze Attention Across All Layers — Finding Induction Heads Model-Wide
 *
 * ## Why This Matters
 *
 * Induction heads don't appear in every layer. In GPT-2, they typically emerge
 * in middle-to-late layers (around layers 5-9). By scanning all layers, we can
 * identify where the model implements this crucial pattern-copying behavior.
 *
 * @param attentions - Map from layer index to attention tensor.
 *   Each attention tensor has shape [num_heads, seq_len, seq_len].
 *
 * @param tokens - Input tokens for pattern matching
 *
 * @param threshold - Minimum induction score (default 0.3)
 *
 * @returns Array of InductionHeadResult objects including layer information.
 *   Sorted by score descending.
 *
 * @example
 * ```typescript
 * const attentions = new Map([
 *   [0, layer0Attention],
 *   [1, layer1Attention],
 *   // ...
 * ]);
 * const results = analyzeAllLayers(attentions, tokens);
 * // results = [
 * //   { layer: 5, head: 2, score: 0.85 },
 * //   { layer: 7, head: 9, score: 0.72 },
 * // ]
 * ```
 */
export function analyzeAllLayers(
  attentions: Map<number, TensorView>,
  tokens: string[],
  threshold: number = 0.3
): InductionHeadResult[] {
  const results: InductionHeadResult[] = [];

  for (const [layer, attention] of attentions.entries()) {
    const layerResults = detectInductionHeads(attention, tokens, threshold);

    // Add layer information to each result
    for (const result of layerResults) {
      results.push({
        layer,
        head: result.head,
        score: result.score,
      });
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  return results;
}
