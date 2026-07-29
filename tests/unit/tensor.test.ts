/**
 * TensorView unit tests.
 *
 * Validates stride math, indexing, slicing, and the WS4 norm(Infinity) fix
 * against hand-computed values (cross-checked with NumPy/PyTorch semantics).
 */

import { describe, expect, it } from "vitest";
import { TensorView } from "@/analysis/utils/tensor";

describe("TensorView strides & indexing", () => {
  it("computes row-major strides for [2,3,4]", () => {
    const t = new TensorView(new Float32Array(24), [2, 3, 4]);
    // np.zeros((2,3,4)).strides / itemsize == (12, 4, 1)
    expect([...t.strides]).toEqual([12, 4, 1]);
  });

  it("resolves multi-dim indices via strides, incl. negative indexing", () => {
    const t = new TensorView(
      Float32Array.from({ length: 24 }, (_, i) => i),
      [2, 3, 4],
    );
    expect(t.get(1, 2, 3)).toBe(23); // 1*12 + 2*4 + 3
    expect(t.get(-1, -1, -1)).toBe(23);
    expect(t.get(0, 0, 0)).toBe(0);
  });

  it("throws on out-of-bounds indices", () => {
    const t = new TensorView(new Float32Array(6), [2, 3]);
    expect(() => t.get(2, 0)).toThrow(/out of bounds/);
  });

  it("rejects data/shape size mismatch in the constructor", () => {
    expect(() => new TensorView(new Float32Array(5), [2, 3])).toThrow();
  });
});

describe("TensorView slicing & shape", () => {
  it("slice([i]) returns row i with correct shape and values", () => {
    const t = new TensorView(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    const row = t.slice([1]);
    expect([...row.shape]).toEqual([3]);
    expect([...row.toFloat32Array()]).toEqual([4, 5, 6]);
  });

  it("transpose() matches numpy .T for 2D", () => {
    const t = new TensorView(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(t.transpose().toNestedArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it("reshape shares the underlying buffer (documented aliasing contract)", () => {
    // NOTE: reshape returns a view over the SAME Float32Array (no copy). This
    // is intentional and pinned here so a future change is a conscious one.
    const src = new TensorView(Float32Array.from([1, 2, 3, 4]), [2, 2]);
    const r = src.reshape([4]);
    r.toFloat32Array()[0] = 99;
    expect(src.get(0, 0)).toBe(99);
  });
});

describe("TensorView reductions & norms", () => {
  it("sum() and mean() over all elements", () => {
    const t = new TensorView(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    expect(t.sum()).toBe(21);
    expect(t.mean()).toBeCloseTo(3.5, 6);
  });

  it("L2 norm of [3,4] is 5", () => {
    const v = new TensorView(Float32Array.from([3, 4]), [2]);
    expect(v.norm()).toBeCloseTo(5, 6);
  });

  it("L1 norm sums absolute values", () => {
    const v = new TensorView(Float32Array.from([-3, 4]), [2]);
    expect(v.norm(1)).toBeCloseTo(7, 6);
  });

  it("L-infinity norm uses max ABSOLUTE value (WS4 fix)", () => {
    // Regression: |max(x)| was wrong for negative-dominant vectors.
    expect(
      new TensorView(Float32Array.from([-10, 2, 3]), [3]).norm(Infinity),
    ).toBe(10);
    expect(
      new TensorView(Float32Array.from([-5, -1]), [2]).norm(Infinity),
    ).toBe(5);
  });

  it("normalize() yields a unit vector", () => {
    const v = new TensorView(Float32Array.from([3, 4]), [2]).normalize();
    expect(v.norm()).toBeCloseTo(1, 6);
  });
});

// ---------------------------------------------------------------------------
// Elementwise arithmetic
//
// These cases were previously only exercised by src/analysis/test.ts — a
// console.log scratch script that nothing imported and no runner executed, so
// its "assertions" were eyeballed once and never again. add/div were the only
// TensorView operations with no real coverage. That file is now deleted and
// its cases live here, where a regression actually fails a build.
// ---------------------------------------------------------------------------

describe("TensorView elementwise arithmetic", () => {
  const a = () => new TensorView(Float32Array.from([1, 2, 3, 4]), [2, 2]);
  const b = () => new TensorView(Float32Array.from([5, 6, 7, 8]), [2, 2]);

  it("adds another tensor elementwise", () => {
    expect(Array.from(a().add(b()).data)).toEqual([6, 8, 10, 12]);
  });

  it("adds a scalar to every element", () => {
    expect(Array.from(a().add(10).data)).toEqual([11, 12, 13, 14]);
  });

  it("rejects an add against a mismatched shape", () => {
    const wrong = new TensorView(Float32Array.from([1, 2]), [2]);
    expect(() => a().add(wrong)).toThrow(/Shape mismatch/);
  });

  it("divides by a scalar", () => {
    expect(Array.from(b().div(2).data)).toEqual([2.5, 3, 3.5, 4]);
  });

  it("divides elementwise", () => {
    expect(Array.from(b().div(b()).data)).toEqual([1, 1, 1, 1]);
  });

  it("preserves shape through arithmetic", () => {
    expect(a().add(b()).shape).toEqual([2, 2]);
    expect(b().div(2).shape).toEqual([2, 2]);
  });

  // Division-by-zero is handled ASYMMETRICALLY, and that asymmetry is load-
  // bearing for callers: scalar /0 is a programming error and throws, but an
  // elementwise divisor with a zero in it yields NaN at that position and
  // keeps going. Pinning both so neither silently flips to the other.
  it("throws on scalar division by zero", () => {
    expect(() => a().div(0)).toThrow(/Division by zero/);
  });

  it("yields NaN — not a throw — at zero positions of an elementwise divisor", () => {
    const divisor = new TensorView(Float32Array.from([1, 0, 2, 0]), [2, 2]);
    const out = Array.from(a().div(divisor).data);
    expect(out[0]).toBe(1);
    expect(out[2]).toBe(1.5);
    expect(out[1]).toBeNaN();
    expect(out[3]).toBeNaN();
  });
});
