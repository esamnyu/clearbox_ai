/**
 * attention.ts unit tests — induction detection, entropy, and head
 * classification on hand-built attention matrices with known answers.
 */

import { describe, expect, it } from "vitest";
import {
  classifyAttentionHead,
  computeAttentionEntropy,
  detectInductionHeads,
} from "@/analysis/attention";
import { TensorView } from "@/analysis/utils/tensor";

describe("detectInductionHeads", () => {
  it("flags a head that attends to j+1 after a repeated token", () => {
    // tokens "A B A B"; induction matches at (i=2,j=0)->attn[2,1] and
    // (i=3,j=1)->attn[3,2]. Put all the mass there.
    // shape [heads=1, seq=4, seq=4]
    const data = Float32Array.from([
      1,
      0,
      0,
      0, // row 0
      0,
      1,
      0,
      0, // row 1
      0,
      1,
      0,
      0, // row 2 -> attends position 1
      0,
      0,
      1,
      0, // row 3 -> attends position 2
    ]);
    const attn = new TensorView(data, [1, 4, 4]);
    const results = detectInductionHeads(attn, ["A", "B", "A", "B"]);
    expect(results).toHaveLength(1);
    expect(results[0].head).toBe(0);
    expect(results[0].score).toBeCloseTo(1, 6);
  });

  it("returns empty for sequences shorter than 3 tokens", () => {
    const attn = new TensorView(Float32Array.from([1, 0, 0, 1]), [1, 2, 2]);
    expect(detectInductionHeads(attn, ["A", "B"])).toEqual([]);
  });

  it("throws when token count != sequence length", () => {
    const attn = new TensorView(
      new Float32Array(1 * 3 * 3).fill(1 / 3),
      [1, 3, 3],
    );
    expect(() => detectInductionHeads(attn, ["A", "B"])).toThrow(
      /doesn't match sequence length/,
    );
  });
});

describe("computeAttentionEntropy", () => {
  it("uniform attention has entropy log2(seq)", () => {
    const seq = 4;
    const attn = new TensorView(new Float32Array(1 * seq * seq).fill(1 / seq), [
      1,
      seq,
      seq,
    ]);
    expect(computeAttentionEntropy(attn).get(0)).toBeCloseTo(Math.log2(seq), 5);
  });

  it("one-hot (focused) attention has zero entropy", () => {
    const data = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const attn = new TensorView(data, [1, 3, 3]);
    expect(computeAttentionEntropy(attn).get(0)).toBeCloseTo(0, 6);
  });
});

describe("classifyAttentionHead", () => {
  it("detects a previous-token head", () => {
    // attn[i, i-1] = 1
    const data = Float32Array.from([
      1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    ]);
    const attn = new TensorView(data, [4, 4]);
    expect(classifyAttentionHead(attn, ["a", "b", "c", "d"])).toBe(
      "previous_token",
    );
  });

  it("detects a first-token head", () => {
    const data = Float32Array.from([
      1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    ]);
    const attn = new TensorView(data, [4, 4]);
    expect(classifyAttentionHead(attn, ["a", "b", "c", "d"])).toBe(
      "first_token",
    );
  });
});
