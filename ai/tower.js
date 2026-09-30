/**
 * ai/tower.js
 * ---------------------------------------------------------------------------
 * The `Tower` composite: input projection → n pre-LayerNorm residual blocks →
 * output projection, with the exact backward paths EfficientZero needs:
 *
 *   backward(dY)          – gradient arrives at the tower output
 *   backwardHidden(dH)    – gradient arrives at the *hidden* state (because
 *                           policy / value / reward heads hang off it)
 *   backwardBoth(dH, dY)  – both at once
 *
 * Every layer instance owns its activation cache, while the `Params` it points
 * at is shared with the other per-unroll-step views → weight tying.
 * ---------------------------------------------------------------------------
 */

import { Linear, LayerNorm, ResBlock, Params, makeResSpecs, gauss } from './nn.js';

export class Tower {
  constructor(specs) {
    this.specs = specs;
    this.dim = specs.res.length ? specs.res[0].p1.inDim : specs.inProj.outDim;
    this.inP = new Linear(specs.inProj);
    specs.debug = specs.debug || 'tower';
    this.debugName = specs.debug;
    this.blocks = specs.res.map((s) => new ResBlock(s));
    this.outP = new Linear(specs.outProj);
    this.outDim = specs.outProj.outDim;
  }
  forward(x) {
    this.x = x;
    if (x.length !== this.inP.p.inDim) throw new Error(`Tower ${this.debugName}: input ${x.length} != ${this.inP.p.inDim}`);
    let cur = this.inP.forward(x);
    for (const b of this.blocks) cur = b.forward(cur);
    this.hidden = cur;
    this.out = this.outP.forward(cur);
    return this.out;
  }
  forwardHidden(x) {
    return this.forward(x);
  }
  getHidden() {
    return this.hidden;
  }
  backward(dY) {
    let d = this.outP.backward(dY, true);
    for (let i = this.blocks.length - 1; i >= 0; i--) d = this.blocks[i].backward(d);
    return this.inP.backward(d, true);
  }
  backwardHidden(dH) {
    let d = dH;
    for (let i = this.blocks.length - 1; i >= 0; i--) d = this.blocks[i].backward(d);
    return this.inP.backward(d, true);
  }
  backwardBoth(dH, dY) {
    let d;
    if (dY) d = this.outP.backward(dY, true);
    if (dH) {
      if (!d) d = Float32Array.from(dH);
      else for (let i = 0; i < d.length; i++) d[i] += dH[i];
    }
    for (let i = this.blocks.length - 1; i >= 0; i--) d = this.blocks[i].backward(d);
    return this.inP.backward(d, true);
  }
}

/**
 * Builds the shared parameter holders for one EfficientZero network group
 * (representation / dynamics / prediction). Heads are extra Linear layers on
 * top of the tower hidden state, exactly like LightZero's `fNetwork`/`gNetwork`.
 */
export function makeTowerSpecs(inDim, hidden, outDim, nRes, heads = {}) {
  const specs = {
    inProj: new Params(inDim, hidden, 'in'),
    outProj: new Params(hidden, outDim, 'out'),
    res: makeResSpecs(hidden, nRes),
    heads: {},
  };
  for (const k of Object.keys(heads)) specs.heads[k] = new Params(hidden, heads[k], 'head_' + k);
  return specs;
}

export function collectSpecParams(specsList) {
  const out = [];
  for (const s of specsList) {
    out.push(s.inProj, s.outProj);
    for (const r of s.res) out.push(r.p1, r.p2, r.lnA, r.lnB);
    for (const k of Object.keys(s.heads || {})) out.push(s.heads[k]);
  }
  return out;
}

export { LayerNorm, gauss };
