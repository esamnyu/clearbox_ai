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
