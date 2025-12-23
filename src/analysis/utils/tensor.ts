/**
 * TensorView: A NumPy-like interface for Float32Array with shape metadata.
 *
 * Designed to feel familiar to ML researchers working with NumPy/PyTorch.
 * All operations are immutable - they return new TensorView instances.
 *
 * @module analysis/utils/tensor
 * @example
 * ```typescript
 * const tensor = new TensorView(new Float32Array([1,2,3,4,5,6]), [2,3]);
 * console.log(tensor.shape);  // [2, 3]
 * console.log(tensor.mean());  // 3.5
 *
 * const row0 = tensor.slice([0]);  // Get first row: [1,2,3]
 * const transposed = tensor.transpose();  // Shape: [3, 2]
 * ```
 */

export type NestedArray<T> = T | NestedArray<T>[];

export class TensorView {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly strides: readonly number[];

  /**
   * Create a TensorView from raw data and shape.
   *
   * @param data - The underlying Float32Array
   * @param shape - The dimensions of the tensor
   *
   * @example
   * ```typescript
   * const data = new Float32Array([1, 2, 3, 4]);
   * const tensor = new TensorView(data, [2, 2]);
   * // [[1, 2],
   * //  [3, 4]]
   * ```
   */
  constructor(data: Float32Array, shape: number[]) {
    const size = shape.reduce((a, b) => a * b, 1);
    if (data.length !== size) {
      throw new Error(
        `Data length ${data.length} does not match shape ${shape} (size ${size})`
      );
    }

    this.data = data;
    this.shape = Object.freeze([...shape]);
    this.strides = Object.freeze(this.computeStrides(shape));
  }

  /**
   * Compute strides for row-major (C-style) layout.
   *
   * For shape [2, 3, 4], strides are [12, 4, 1]:
   * - Moving one step in dim 0 jumps 12 elements
   * - Moving one step in dim 1 jumps 4 elements
   * - Moving one step in dim 2 jumps 1 element
   */
  private computeStrides(shape: number[]): number[] {
    const strides: number[] = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  /**
   * Convert multi-dimensional indices to flat index.
   *
   * @param indices - One index per dimension
   * @returns Flat index into this.data
   */
  private indicesToOffset(indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new Error(
        `Expected ${this.shape.length} indices, got ${indices.length}`
      );
    }

    let offset = 0;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i] < 0 ? this.shape[i] + indices[i] : indices[i];
      if (idx < 0 || idx >= this.shape[i]) {
        throw new Error(
          `Index ${indices[i]} out of bounds for dimension ${i} (size ${this.shape[i]})`
        );
      }
      offset += idx * this.strides[i];
    }
    return offset;
  }

  // ============================================================================
  // INDEXING
  // ============================================================================

  /**
   * Get a single element from the tensor.
   *
   * @param indices - One index per dimension
   * @returns The scalar value
   *
   * @example
   * ```typescript
   * const tensor = new TensorView(new Float32Array([1,2,3,4]), [2,2]);
   * tensor.get(0, 1);  // 2
   * tensor.get(1, 0);  // 3
   * ```
   */
  get(...indices: number[]): number {
    return this.data[this.indicesToOffset(indices)];
  }

  /**
   * Slice the tensor along one or more dimensions.
   *
   * @param indices - Array of indices or null (keep entire dimension)
   * @returns New TensorView with reduced dimensionality
   *
   * @example
   * ```typescript
   * // attention: [12, 64, 64]  (heads, seq, seq)
   * const head3 = attention.slice([3]);  // [64, 64]
   * const firstTenTokens = attention.slice([null, [0, 10]]);  // [12, 10, 64]
   * ```
   *
   * RESEARCHER TODO: Currently only supports single index slicing.
   * Extend to support range slicing like [start, end].
   */
  slice(indices: (number | null)[]): TensorView {
    if (indices.length === 0 || indices.length > this.shape.length) {
      throw new Error('Invalid slice indices');
    }

    // Simple implementation: only handle single index for first dimension
    if (indices.length === 1 && typeof indices[0] === 'number') {
      const idx = indices[0] < 0 ? this.shape[0] + indices[0] : indices[0];
      const offset = idx * this.strides[0];
      const newSize = this.shape.slice(1).reduce((a, b) => a * b, 1);
      const newData = this.data.slice(offset, offset + newSize);
      return new TensorView(new Float32Array(newData), this.shape.slice(1));
    }

    // RESEARCHER TODO: Implement full slicing with ranges
    throw new Error('Advanced slicing not yet implemented');
  }

  // ============================================================================
  // REDUCTIONS
  // ============================================================================

  /**
   * Compute the sum of all elements or along an axis.
   *
   * @param axis - Optional axis to reduce along
   * @returns Scalar or reduced tensor
   *
   * @example
   * ```typescript
   * const tensor = new TensorView(new Float32Array([1,2,3,4,5,6]), [2,3]);
   * tensor.sum();     // 21
   * tensor.sum(0);    // [5, 7, 9] (sum across rows)
   * tensor.sum(1);    // [6, 15] (sum across columns)
   * ```
   */
  sum(axis?: number): TensorView | number {
    if (axis === undefined) {
      // Sum all elements
      let total = 0;
      for (let i = 0; i < this.data.length; i++) {
        total += this.data[i];
      }
      return total;
    }

    // RESEARCHER TODO: Implement axis-wise sum
    // Hint: Create new shape by removing the specified axis
    // Iterate over all indices, accumulate along the reduction axis
    throw new Error('Axis-wise sum not yet implemented');
  }

  /**
   * Compute the mean of all elements or along an axis.
   *
   * @param axis - Optional axis to reduce along
   * @returns Scalar or reduced tensor
   */
  mean(axis?: number): TensorView | number {
    const s = this.sum(axis);
    if (typeof s === 'number') {
      return s / this.data.length;
    }
    // RESEARCHER TODO: Implement axis-wise mean
    throw new Error('Axis-wise mean not yet implemented');
  }

  /**
   * Find the maximum value.
   *
   * @param axis - Optional axis to reduce along
   * @returns Scalar or reduced tensor
   */
  max(axis?: number): TensorView | number {
    if (axis === undefined) {
      let maxVal = -Infinity;
      for (let i = 0; i < this.data.length; i++) {
        if (this.data[i] > maxVal) maxVal = this.data[i];
      }
      return maxVal;
    }
    // RESEARCHER TODO: Implement axis-wise max
    throw new Error('Axis-wise max not yet implemented');
  }

  /**
   * Find the minimum value.
   */
  min(axis?: number): TensorView | number {
    if (axis === undefined) {
      let minVal = Infinity;
      for (let i = 0; i < this.data.length; i++) {
        if (this.data[i] < minVal) minVal = this.data[i];
      }
      return minVal;
    }
    // RESEARCHER TODO: Implement axis-wise min
    throw new Error('Axis-wise min not yet implemented');
  }

  /**
   * Find indices of maximum values along an axis.
   *
   * RESEARCHER TODO: Implement argmax
   * Returns indices where maximum occurs
   */
  argmax(_axis?: number): TensorView | number {
    throw new Error('argmax not yet implemented');
  }

  // ============================================================================
  // ELEMENT-WISE OPERATIONS
  // ============================================================================

  /**
   * Add a scalar or another tensor element-wise.
   *
   * @param other - Scalar or tensor with same shape
   * @returns New tensor with result
   */
  add(other: TensorView | number): TensorView {
    if (typeof other === 'number') {
      const result = new Float32Array(this.data.length);
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] + other;
      }
      return new TensorView(result, [...this.shape]);
    }

    if (!this.shapesMatch(other.shape)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }

    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] + other.data[i];
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Subtract a scalar or another tensor element-wise.
   */
  sub(other: TensorView | number): TensorView {
    if (typeof other === 'number') {
      const result = new Float32Array(this.data.length);
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] - other;
      }
      return new TensorView(result, [...this.shape]);
    }

    if (!this.shapesMatch(other.shape)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }

    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] - other.data[i];
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Multiply by a scalar or another tensor element-wise.
   *
   * We use element-wise multiplication frequently in attention analysis,
   * particularly for computing entropy: H = -Σ p * log(p).
   *
   * @param other - Scalar or tensor with same shape
   * @returns New tensor with element-wise products
   *
   * @example
   * ```typescript
   * const a = new TensorView(new Float32Array([1, 2, 3]), [3]);
   * a.mul(2);           // [2, 4, 6]
   * a.mul(a);           // [1, 4, 9] (element-wise squares)
   * ```
   */
  mul(other: TensorView | number): TensorView {
    if (typeof other === 'number') {
      // Scalar multiplication (same as scale, but consistent API)
      const result = new Float32Array(this.data.length);
      for (let i = 0; i < this.data.length; i++) {
        result[i] = this.data[i] * other;
      }
      return new TensorView(result, [...this.shape]);
    }

    // Element-wise multiplication with another tensor
    if (!this.shapesMatch(other.shape)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }

    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] * other.data[i];
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Divide by a scalar or another tensor element-wise.
   */
  div(_other: TensorView | number): TensorView {
    // RESEARCHER TODO: Implement division
    throw new Error('div not yet implemented');
  }

  /**
   * Multiply all elements by a scalar.
   */
  scale(scalar: number): TensorView {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = this.data[i] * scalar;
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Element-wise exponential.
   */
  exp(): TensorView {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Math.exp(this.data[i]);
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Element-wise natural logarithm.
   */
  log(): TensorView {
    const result = new Float32Array(this.data.length);
    for (let i = 0; i < this.data.length; i++) {
      result[i] = Math.log(this.data[i]);
    }
    return new TensorView(result, [...this.shape]);
  }

  /**
   * Softmax along the last axis.
   *
   * RESEARCHER TODO: Implement softmax
   * Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
   * Numerical stability: subtract max before exp
   */
  softmax(_axis?: number): TensorView {
    throw new Error('softmax not yet implemented');
  }

  // ============================================================================
  // LINEAR ALGEBRA
  // ============================================================================

  /**
   * Dot product or matrix multiplication.
   *
   * For 1D tensors: inner product
   * For 2D tensors: matrix multiplication
   *
   * RESEARCHER TODO: Implement dot product
   * Shapes must be compatible: [a,b] · [b,c] → [a,c]
   */
  dot(_other: TensorView): TensorView {
    throw new Error('dot not yet implemented');
  }

  /**
   * Compute L2 (Euclidean) norm.
   *
   * @param ord - Norm order (1=L1, 2=L2, Infinity=max)
   * @returns Scalar norm value
   *
   * @example
   * ```typescript
   * const v = new TensorView(new Float32Array([3, 4]), [2]);
   * v.norm();  // 5.0 (sqrt(3^2 + 4^2))
   * v.norm(1);  // 7.0 (|3| + |4|)
   * ```
   */
  norm(ord: number = 2): number {
    if (ord === 2) {
      // L2 norm: sqrt(sum of squares)
      let sumSquares = 0;
      for (let i = 0; i < this.data.length; i++) {
        sumSquares += this.data[i] * this.data[i];
      }
      return Math.sqrt(sumSquares);
    } else if (ord === 1) {
      // L1 norm: sum of absolute values
      let sum = 0;
      for (let i = 0; i < this.data.length; i++) {
        sum += Math.abs(this.data[i]);
      }
      return sum;
    } else if (ord === Infinity) {
      // Infinity norm: max absolute value
      return Math.abs(this.max() as number);
    }

    throw new Error(`Norm order ${ord} not supported`);
  }

  /**
   * Normalize the tensor to unit norm.
   *
   * @returns New tensor with L2 norm = 1
   */
  normalize(): TensorView {
    const n = this.norm();
    if (n === 0) {
      throw new Error('Cannot normalize zero vector');
    }
    return this.scale(1 / n);
  }

  // ============================================================================
  // SHAPE MANIPULATION
  // ============================================================================

  /**
   * Reshape the tensor to a new shape.
   *
   * @param newShape - New dimensions (must have same total size)
   * @returns New TensorView with new shape
   */
  reshape(newShape: number[]): TensorView {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.data.length) {
      throw new Error(
        `Cannot reshape ${this.shape} (size ${this.data.length}) to ${newShape} (size ${newSize})`
      );
    }
    return new TensorView(this.data, newShape);
  }

  /**
   * Transpose the tensor (reverse all axes).
   *
   * RESEARCHER TODO: Implement transpose
   * For 2D: [2,3] → [3,2]
   * For general: need to permute strides and copy data
   */
  transpose(axes?: number[]): TensorView {
    if (this.shape.length === 2 && !axes) {
      // Simple 2D transpose
      const [rows, cols] = this.shape;
      const result = new Float32Array(this.data.length);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          result[j * rows + i] = this.data[i * cols + j];
        }
      }

      return new TensorView(result, [cols, rows]);
    }

    // RESEARCHER TODO: Implement general transpose
    throw new Error('General transpose not yet implemented');
  }

  /**
   * Remove dimensions of size 1.
   *
   * RESEARCHER TODO: Implement squeeze
   */
  squeeze(_axis?: number): TensorView {
    throw new Error('squeeze not yet implemented');
  }

  /**
   * Add a dimension of size 1 at the specified position.
   *
   * RESEARCHER TODO: Implement unsqueeze
   */
  unsqueeze(_axis: number): TensorView {
    throw new Error('unsqueeze not yet implemented');
  }

  // ============================================================================
  // CONVERSION
  // ============================================================================

  /**
   * Convert to nested JavaScript arrays.
   *
   * @returns Nested array matching tensor shape
   */
  toNestedArray(): NestedArray<number> {
    if (this.shape.length === 0) {
      return this.data[0];
    }

    if (this.shape.length === 1) {
      return Array.from(this.data);
    }

    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result: number[][] = [];
      for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
          row.push(this.data[i * cols + j]);
        }
        result.push(row);
      }
      return result;
    }

    // RESEARCHER TODO: Implement for higher dimensions
    throw new Error('toNestedArray only supports up to 2D');
  }

  /**
   * Get the underlying Float32Array.
   */
  toFloat32Array(): Float32Array {
    return this.data;
  }

  /**
   * Create a deep copy.
   */
  clone(): TensorView {
    return new TensorView(new Float32Array(this.data), [...this.shape]);
  }

  /**
   * String representation for debugging.
   */
  toString(): string {
    return `TensorView(shape=${this.shape}, data=${this.data.slice(0, 10)}${
      this.data.length > 10 ? '...' : ''
    })`;
  }

  // ============================================================================
  // STATIC CONSTRUCTORS
  // ============================================================================

  /**
   * Create a tensor filled with zeros.
   */
  static zeros(shape: number[]): TensorView {
    const size = shape.reduce((a, b) => a * b, 1);
    return new TensorView(new Float32Array(size), shape);
  }

  /**
   * Create a tensor filled with ones.
   */
  static ones(shape: number[]): TensorView {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    data.fill(1);
    return new TensorView(data, shape);
  }

  /**
   * Create from nested JavaScript arrays.
   *
   * RESEARCHER TODO: Implement fromNestedArray
   * Infer shape from nested structure, flatten to Float32Array
   */
  static fromNestedArray(_arr: NestedArray<number>): TensorView {
    throw new Error('fromNestedArray not yet implemented');
  }

  // ============================================================================
  // PRIVATE HELPERS
  // ============================================================================

  private shapesMatch(other: readonly number[]): boolean {
    if (this.shape.length !== other.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other[i]) return false;
    }
    return true;
  }
}
