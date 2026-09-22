export interface Example {
  x: number[];
  y: 0 | 1;
  weight: number;
}

export interface Model {
  weights: number[];
  bias: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/**
 * L2-regularized logistic regression by full-batch Adam. Positives and
 * negatives are reweighted to equal total weight, so a few clicks are not
 * swamped by hundreds of background samples.
 */
export function trainLogistic(
  examples: Example[],
  { iterations = 400, learningRate = 0.05, l2 = 1e-3 } = {},
): Model | null {
  const pos = examples.filter((e) => e.y === 1);
  const neg = examples.filter((e) => e.y === 0);
  if (pos.length === 0 || neg.length === 0) return null;

  const posTotal = pos.reduce((s, e) => s + e.weight, 0);
  const negTotal = neg.reduce((s, e) => s + e.weight, 0);
  const scale = (e: Example) =>
    (e.weight / (e.y === 1 ? posTotal : negTotal)) * 0.5;

  const dims = examples[0].x.length;
  const w = new Float64Array(dims);
  let b = 0;
  const mw = new Float64Array(dims);
  const vw = new Float64Array(dims);
  let mb = 0;
  let vb = 0;
  const [beta1, beta2, eps] = [0.9, 0.999, 1e-8];
  const gw = new Float64Array(dims);

  for (let t = 1; t <= iterations; t++) {
    gw.fill(0);
    let gb = 0;
    for (const e of examples) {
      let z = b;
      for (let i = 0; i < dims; i++) z += w[i] * e.x[i];
      const err = (sigmoid(z) - e.y) * scale(e);
      for (let i = 0; i < dims; i++) gw[i] += err * e.x[i];
      gb += err;
    }
    for (let i = 0; i < dims; i++) {
      const g = gw[i] + l2 * w[i];
      mw[i] = beta1 * mw[i] + (1 - beta1) * g;
      vw[i] = beta2 * vw[i] + (1 - beta2) * g * g;
      const mHat = mw[i] / (1 - beta1 ** t);
      const vHat = vw[i] / (1 - beta2 ** t);
      w[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
    }
    mb = beta1 * mb + (1 - beta1) * gb;
    vb = beta2 * vb + (1 - beta2) * gb * gb;
    b -= (learningRate * (mb / (1 - beta1 ** t))) / (Math.sqrt(vb / (1 - beta2 ** t)) + eps);
  }

  return { weights: Array.from(w), bias: b };
}
