import { TensorView } from "./utils/tensor";

const a = new TensorView(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = new TensorView(new Float32Array([5, 6, 7, 8]), [2, 2]);

console.log('Tensor A:');
console.log(a.toString());
console.log('Tensor B:');
console.log(b.toString());

const c = a.add(b);
console.log('A + B:');
console.log(c.toString());

// division
const d = b.div(2);
console.log('B / 2:');
console.log(d.toString());

// element wise division
const e = b.div(a);
console.log('B / A (element-wise):');
console.log(e.toString());

// division by zero handling
const f = a.div(new TensorView(new Float32Array([1, 0, 2, 0]), [2, 2]));
console.log('A / [1,0;2,0] (element-wise with division by zero):');
console.log(f.toString());

const g = a.div(0);
console.log('A / 0 (scalar division by zero):');
console.log(g.toString());

