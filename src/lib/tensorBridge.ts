import { TensorView } from "@/analysis/utils/tensor";
import type { TensorWithMetadata } from "@/engine/types";

/**
 * Convert attention TensorWithMetadata[] into a Map keyed by layer number.
 * Filters by step, strips the batch dimension [1,12,s,s] -> [12,s,s].
 * Since batch=1 and TensorView uses row-major layout, the underlying
 * Float32Array is identical — just change the shape metadata.
 */
export function attentionsToLayerMap(
  tensors: TensorWithMetadata[],
  step: number,
): Map<number, TensorView> {
  const map = new Map<number, TensorView>();

  for (const tensor of tensors) {
    if (tensor.step !== step) continue;

    const strippedShape = tensor.shape.slice(1); // [1,12,s,s] -> [12,s,s]
    map.set(tensor.layer, new TensorView(tensor.data, strippedShape));
  }

  return map;
}

/**
 * Convert hidden state TensorWithMetadata[] into a Map keyed by layer number.
 * Filters by step, strips the batch dimension [1,s,768] -> [s,768].
 */
export function hiddenStatesToLayerMap(
  tensors: TensorWithMetadata[],
  step: number,
): Map<number, TensorView> {
  const map = new Map<number, TensorView>();

  for (const tensor of tensors) {
    if (tensor.step !== step) continue;

    const strippedShape = tensor.shape.slice(1); // [1,s,768] -> [s,768]
    map.set(tensor.layer, new TensorView(tensor.data, strippedShape));
  }

  return map;
}
