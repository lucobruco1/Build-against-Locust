/**
 * ai/nn.js
 * ---------------------------------------------------------------------------
 * A very small neural-network library written from scratch on Float32Arrays.
 * No dependency and no generic autograd graph: every layer implements
 * forward(x)/backward(dy) by hand, and back-propagation-through-time across the
 * EfficientZero dynamics unroll is done by building *one view per unrolled
 * step* whose layers all point at the same shared `Params` holders (weight
 * tying). That is what makes a 5-step unroll with a recurrent value-prefix
 * head cheap enough to run in a browser.
 *
 * Pieces provided: Linear (dense), ReLU, LayerNorm with shared affine
 * parameters, residual blocks, the MuZero/EfficientZero categorical value
 * transformation ("Support"), AdamW with global gradient-norm clipping, and
 * pack/unpack to a single Float32Array for checkpoints (server persistence).
 * ---------------------------------------------------------------------------
 */

export function zeros(n) {
  return new Float32Array(n);
}
export function ones(n, v = 1) {
  const a = new Float32Array(n);
  a.fill(v);
  return a;
}
export function softmaxInto(logits, out, n) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > m) m = logits[i];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - m);
    out[i] = e;
    s += e;
  }
  const inv = s > 0 ? 1 / s : 1 / n;
  for (let i = 0; i < n; i++) out[i] *= inv;
  return out;
}
export function entropy(p, n) {
  let h = 0;
  for (let i = 0; i < n; i++) if (p[i] > 1e-12) h -= p[i] * Math.log(p[i]);
  return h;
}

let spare = null;
export function gauss() {
  if (spare !== null) {
    const v = spare;
    spare = null;
    return v;
  }
  let u = 0, v = 0, s = 0;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s === 0 || s >= 1);
  const mul = Math.sqrt((-2 * Math.log(s)) / s);
  spare = v * mul;
  return u * mul;
}

/* --------------------------------------------------------------- holders */

/**
 * A parameter holder: weights + biases + gradients + Adam moments. Everything
 * in this file that owns learnable numbers is one of these, so the Optimizer
 * can treat dense layers and layer-norm affinities uniformly.
 */
export class Params {
  constructor(inDim, outDim, name = '', opts = {}) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.name = name;
    const nW = inDim * outDim;
    this.W = new Float32Array(nW);
    this.b = new Float32Array(outDim);
    this.dW = new Float32Array(nW);
    this.db = new Float32Array(outDim);
    this.mW = new Float32Array(nW);
    this.vW = new Float32Array(nW);
    this.mb = new Float32Array(outDim);
    this.vb = new Float32Array(outDim);
    if (opts.he !== false) {
      const s = Math.sqrt((opts.gain ?? 2) / Math.max(1, inDim));
      for (let i = 0; i < nW; i++) this.W[i] = gauss() * s;
      if (opts.zeroBias !== false) this.b.fill(0);
    }
  }
  zeroGrad() {
    this.dW.fill(0);
    this.db.fill(0);
  }
  size() {
    return this.W.length + this.b.length;
  }
  copyFrom(o) {
    this.W.set(o.W);
    this.b.set(o.b);
  }
}

/* ----------------------------------------------------------------- layers */

export class Linear {
  constructor(p) {
    this.p = p;
  }
  forward(x) {
    const { W, b, inDim, outDim } = this.p;
    const o = new Float32Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let s = b[j];
      const base = j * inDim;
      for (let i = 0; i < inDim; i++) s += W[base + i] * x[i];
      o[j] = s;
    }
    this.x = x;
    this.out = o;
    return o;
  }
  backward(dy, wantInput = true) {
    const { W, dW, db, inDim, outDim } = this.p;
    const x = this.x;
    let dX = null;
    if (wantInput) {
      dX = new Float32Array(inDim);
      for (let i = 0; i < inDim; i++) {
        let s = 0;
        for (let j = 0; j < outDim; j++) s += W[j * inDim + i] * dy[j];
        dX[i] = s;
      }
    }
    for (let j = 0; j < outDim; j++) {
      const gj = dy[j];
      if (gj === 0) continue;
      db[j] += gj;
      const base = j * inDim;
      for (let i = 0; i < inDim; i++) dW[base + i] += gj * x[i];
    }
    return dX;
  }
}

export class ReLU {
  forward(x) {
    const o = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) o[i] = x[i] > 0 ? x[i] : 0;
    this.in = x;
    return o;
  }
  backward(dy) {
    const x = this.in;
    const d = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) d[i] = x[i] > 0 ? dy[i] : 0;
    return d;
  }
}

/** LayerNorm whose affine parameters live in a shared `Params` holder. */
export class LayerNorm {
  constructor(vp, eps = 1e-5, debugName = '') {
    this.vp = vp;             // Params whose W is the gain vector, b the bias
    this.dim = vp.outDim;
    this.eps = eps;
    this.debugName = debugName;
    this.hat = null;
  }
  forward(x) {
    const n = this.dim, g = this.vp.W, be = this.vp.b;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let varr = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      varr += d * d;
    }
    varr /= n;
    const inv = 1 / Math.sqrt(varr + this.eps);
    const o = new Float32Array(n);
    const hat = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const hv = (x[i] - mean) * inv;
      hat[i] = hv;
      o[i] = g[i] * hv + be[i];
    }
    this.inv = inv;
    this.hat = hat;
    return o;
  }
  backward(dy) {
    const n = this.dim, inv = this.inv, hat = this.hat;
    if (!hat) throw new Error(`LayerNorm.backward without forward (${this.debugName || 'unnamed'})`);
    const g = this.vp.W, dG = this.vp.dW, dB = this.vp.db;
    const dHat = new Float32Array(n);
    let s1 = 0, s2 = 0;
    for (let i = 0; i < n; i++) {
      dG[i] += dy[i] * hat[i];
      dB[i] += dy[i];
      const dh = dy[i] * g[i];
      dHat[i] = dh;
      s1 += dh;
      s2 += dh * hat[i];
    }
    const c1 = s1 / n, c2 = s2 / n;
    const dx = new Float32Array(n);
    for (let i = 0; i < n; i++) dx[i] = inv * (dHat[i] - c1 - hat[i] * c2);
    return dx;
  }
}

export class Seq {
  constructor(layers) {
    this.layers = layers;
  }
  forward(x) {
    let cur = x;
    for (const l of this.layers) cur = l.forward(cur);
    return cur;
  }
  backward(dy) {
    let d = dy;
    for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }
}

/**
 * Pre-LayerNorm residual block: y = x + W2 · relu(LN1(x)) … W1 with a second
 * norm after the expansion (EZ-V2's pre-LN transformer tower, dense version).
 * `specs` carries the shared params so several views can tie to one block.
 */
export class ResBlock {
  constructor(specs) {
    this.p1 = specs.p1;
    this.p2 = specs.p2;
    this.lnA = new LayerNorm(specs.lnA, 1e-5, specs.debug + '.lnA');
    this.lnB = new LayerNorm(specs.lnB, 1e-5, specs.debug + '.lnB');
    this.l1 = new Linear(specs.p1);
    this.l2 = new Linear(specs.p2);
    this.r1 = new ReLU();
    this.dim = specs.p1.inDim;
  }
  forward(x) {
    this.x = x;
    const a = this.lnA.forward(x);
    const h = this.r1.forward(this.l1.forward(a));
    const h2 = this.lnB.forward(h);
    this.inner = this.l2.forward(h2);
    const o = new Float32Array(this.dim);
    const inn = this.inner;
    for (let i = 0; i < this.dim; i++) o[i] = x[i] + inn[i];
    return o;
  }
  backward(dy) {
    let d = this.l2.backward(dy, true);
    d = this.lnB.backward(d);
    d = this.reluPass(d);
    d = this.l1.backward(d, true);
    d = this.lnA.backward(d);
    const o = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) o[i] = d[i] + dy[i];
    return o;
  }
  reluPass(d) {
    const x = this.r1.in;
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? d[i] : 0;
    return out;
  }
}

export function makeResSpecs(dim, nRes, prefix = 'res') {
  const out = [];
  for (let i = 0; i < nRes; i++) {
    out.push({
      p1: new Params(dim, dim, `${prefix}${i}.ln1`),
      p2: new Params(dim, dim, `${prefix}${i}.ln2`),
      // LayerNorm affine: W is the gain vector, b the bias — no matrix needed
      lnA: new Params(1, dim, `${prefix}${i}.a`, { he: false }),
      lnB: new Params(1, dim, `${prefix}${i}.b`, { he: false }),
    });
  }
  // LayerNorm affine: gain must be 1, bias 0
  for (const s of out) {
    s.lnA.W.fill(1);
    s.lnB.W.fill(1);
  }
  return out;
}

/* ----------------------------- value transformation (categorical support) */

export class Support {
  constructor(bins, min, max) {
    this.bins = bins;
    this.min = min;
    this.max = max;
    this.values = new Float32Array(bins);
    for (let i = 0; i < bins; i++) this.values[i] = min + ((max - min) * i) / (bins - 1);
  }
  /** scalar -> two-hot distribution over the support */
  encode(scalar, out) {
    const v = Math.max(this.min, Math.min(this.max, scalar));
    const span = (this.max - this.min) / (this.bins - 1);
    const pos = (v - this.min) / span;
    const lo = Math.max(0, Math.min(this.bins - 1, Math.floor(pos)));
    const hi = Math.min(this.bins - 1, lo + 1);
    const frac = pos - lo;
    out.fill(0);
    out[lo] = 1 - frac;
    out[hi] += frac;
    return out;
  }
  decode(probs) {
    let s = 0;
    for (let i = 0; i < this.bins; i++) s += probs[i] * this.values[i];
    return s;
  }
  /**
   * dL/dlogits from dL/dscalar:  scalar = Σ softmax(z)_i v_i
   * ⇒ dz_i = p_i (v_i - scalar) · g
   */
  grad(probs, g) {
    const out = new Float32Array(this.bins);
    const sc = this.decode(probs);
    for (let i = 0; i < this.bins; i++) out[i] = probs[i] * (this.values[i] - sc) * g;
    return out;
  }
}

/* --------------------------------------------------------- optimiser glue */

export class Optimizer {
  constructor(params, o = {}) {
    this.params = params;
    this.lr = o.lr ?? 3e-4;
    this.wd = o.weightDecay ?? 1e-4;
    this.clip = o.clip ?? 5;
    this.b1 = 0.9;
    this.b2 = 0.999;
    this.eps = 1e-8;
    this.t = 0;
    this.gradNorm = 0;
  }
  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
  clipGradNorm() {
    let sq = 0;
    for (const p of this.params) {
      const dW = p.dW, db = p.db;
      for (let i = 0; i < dW.length; i++) sq += dW[i] * dW[i];
      for (let i = 0; i < db.length; i++) sq += db[i] * db[i];
    }
    const n = Math.sqrt(sq);
    this.gradNorm = n;
    if (!Number.isFinite(n) || n <= this.clip || n === 0) return n;
    const s = this.clip / n;
    for (const p of this.params) {
      const dW = p.dW, db = p.db;
      for (let i = 0; i < dW.length; i++) dW[i] *= s;
      for (let i = 0; i < db.length; i++) db[i] *= s;
    }
    return n;
  }
  step() {
    this.t++;
    const c1 = 1 - Math.pow(this.b1, this.t);
    const c2 = 1 - Math.pow(this.b2, this.t);
    const lr = this.lr;
    for (const p of this.params) {
      const { W, b, dW, db, mW, vW, mb, vb } = p;
      for (let i = 0; i < W.length; i++) {
        const g = dW[i] + this.wd * W[i];
        mW[i] = this.b1 * mW[i] + (1 - this.b1) * g;
        vW[i] = this.b2 * vW[i] + (1 - this.b2) * g * g;
        W[i] -= lr * (mW[i] / c1) / (Math.sqrt(vW[i] / c2) + this.eps);
      }
      for (let i = 0; i < b.length; i++) {
        const g = db[i];
        mb[i] = this.b1 * mb[i] + (1 - this.b1) * g;
        vb[i] = this.b2 * vb[i] + (1 - this.b2) * g * g;
        b[i] -= lr * (mb[i] / c1) / (Math.sqrt(vb[i] / c2) + this.eps);
      }
    }
  }
}

/* ------------------------------------------------------ param serialisation */

export function packParams(params) {
  let n = 4;
  const sizes = [];
  for (const p of params) {
    sizes.push(p.W.length, p.b.length);
    n += 2 + p.W.length + p.b.length;
  }
  const out = new Float32Array(n);
  out[0] = params.length;
  out[1] = n;
  let i = 4;
  for (let k = 0; k < params.length; k++) {
    const p = params[k];
    out[i++] = sizes[k * 2];
    out[i++] = sizes[k * 2 + 1];
    out.set(p.W, i); i += p.W.length;
    out.set(p.b, i); i += p.b.length;
  }
  return out;
}

/**
 * Float32Array ↔ base64, for checkpoints that go through JSON.
 *
 * `Array.from(weights)` was the obvious thing to write and it costs 15×: nine
 * brains of 298,745 params became a 46 MB `brains.json` that the autosave wrote
 * synchronously into the game loop. Base64 of the raw bytes is the same data in
 * ~1.6 MB total, and both the browser and Node have btoa/atob, so there is no
 * dependency to add.
 */
export function f32ToBase64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let bin = '';
  const CH = 8192;
  for (let i = 0; i < u8.length; i += CH) bin += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
  return btoa(bin);
}

export function base64ToF32(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer, 0, u8.length >> 2);
}

export function unpackParams(buf, params) {
  if (!buf || buf.length < 4) return false;
  const count = buf[0] | 0;
  if (count !== params.length) return false;
  let i = 4;
  for (let k = 0; k < count; k++) {
    const w = buf[i++] | 0, b = buf[i++] | 0;
    const p = params[k];
    if (p.W.length !== w || p.b.length !== b) return false;
    for (let j = 0; j < w; j++) p.W[j] = buf[i++];
    for (let j = 0; j < b; j++) p.b[j] = buf[i++];
  }
  return true;
}

export function paramCount(params) {
  let n = 0;
  for (const p of params) n += p.W.length + p.b.length;
  return n;
}
