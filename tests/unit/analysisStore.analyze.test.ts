/**
 * Regression test for the WS1 bug: analyze() crashed on every real generation.
 *
 * Step-0 attention comes from the PROMPT-ONLY forward pass, so its sequence
 * length is the prompt length. result.tokens, however, is the FULL sequence
 * (prompt + generated tokens). The old analyze() passed the full token list
 * into detectInductionHeads/classifyAttentionHead, which throw when
 * tokens.length !== seqLen — and that throw (caught in App.handleGenerate)
 * silently cleared the whole panel. analyze() now slices tokens to the step-0
 * length first, so it must NOT throw and must populate the head grid.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { useAnalysisStore } from "@/store/analysisStore";
import type { GenerationResult, TensorWithMetadata } from "@/engine/types";

const NUM_HEADS = 2;
const PROMPT_SEQ = 3; // step-0 attention is prompt-length
const NUM_LAYERS = 2;

function attentionTensor(layer: number): TensorWithMetadata {
  // Shape [batch=1, heads, seq, seq]; uniform rows are a valid distribution.
  const data = new Float32Array(1 * NUM_HEADS * PROMPT_SEQ * PROMPT_SEQ);
  data.fill(1 / PROMPT_SEQ);
  return {
    data,
    shape: [1, NUM_HEADS, PROMPT_SEQ, PROMPT_SEQ],
    dtype: "float32",
    step: 0,
    layer,
  };
}

function makeResult(): GenerationResult {
  return {
    text: "on the mat",
    // 5 tokens: 3 prompt + 2 generated — longer than PROMPT_SEQ on purpose.
    tokens: ["The", "cat", "sat", "on", "mat"],
    tokenIds: [1, 2, 3, 4, 5],
    attentions: [attentionTensor(0), attentionTensor(1)],
  };
}

describe("analysisStore.analyze (WS1 regression)", () => {
  beforeEach(() => {
    useAnalysisStore.getState().reset();
  });

  it("does not throw when tokens are longer than the step-0 sequence", () => {
    const { analyze } = useAnalysisStore.getState();
    expect(() => analyze(makeResult())).not.toThrow();
  });

  it("slices stored tokens down to the step-0 prompt length", () => {
    useAnalysisStore.getState().analyze(makeResult());
    expect(useAnalysisStore.getState().tokens).toEqual(["The", "cat", "sat"]);
  });

  it("populates a head grid sized from the tensors, not hardcoded 12x12", () => {
    useAnalysisStore.getState().analyze(makeResult());
    const grid = useAnalysisStore.getState().headGrid;
    expect(grid).not.toBeNull();
    expect(grid!.length).toBe(NUM_LAYERS);
    expect(grid![0].length).toBe(NUM_HEADS);
  });
});
