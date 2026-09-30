/**
 * ai/efficientzero.js
 * ---------------------------------------------------------------------------
 * A real EfficientZero agent, implemented from:
 *
 *   • Ye, Liu, Sun, et al. "EfficientZero: Mastering Atari Games with Limited
 *     Data" (NeurIPS 2021)
 *   • "EfficientZero V2: Mastering Discrete and Continuous Control with Limited
 *     Data" (2024) — mixed SVE / TD value targets, pre-LayerNorm towers
 *   • LightZero's reference implementation, for the exact target plumbing
 *
 * Network groups (names follow the papers):
 *   H  representation  s_t        = H(o_t)
 *   G  dynamics       ŝ_{t+1}, R̂ = G(s_t, a_t)   (residual latent update)
 *   P,V prediction     π_t, v_t   = P(s_t)
 *   R  value-prefix head — predicts Σ_{j≤k} γ_p^j u_{t+j} (the *prefix*, not a
 *      single-step reward). The paper carries the running sum in an LSTM; here
 *      it is an identity-recurrent accumulator (unit recurrent weights), which
 *      keeps 5-step BPTT affordable on a browser CPU while preserving the
 *      "don't insist on the exact reward timing" property that makes EZ learn.
 *
 * Objective (per minibatch, unrolled l = cfg.UNROLL steps):
 *   L = λ1·CE(prefix̂_k, û_k) + λ2·CE(π̂_k, π_k) + λ3·CE(v̂_k, z_k)
 *     + λ4·L_consistency − λ5·H[π̂]        (+ AdamW decay, global grad clipping)
 *
 * Value and prefix terms use the MuZero/EZ categorical *value transformation*:
 * the head emits logits over a symmetric support, targets are two-hot encoded.
 * π_k is the MCTS improved policy (visit distribution), z_k the MCTS search
 * value for stale samples (SVE) or the n-step TD return bootstrapped with the
 * target network for fresh ones.
 *
 * EfficientZero specifics implemented here:
 *   • self-supervised temporal consistency (SimSiam cosine + predictor, target
 *     latent detached) so the model learns from observations, not only rewards
 *   • off-policy correction (n-step depth shrinks as a sample ages)
 *   • target network synced every cfg.TARGET_SYNC optimizer steps
 *   • importance-sampling weights β from the prioritised buffer
 *   • 0.5 gradient scaling at the base of the unroll (training stability)
 --------------------------------------------------------------------------- */

import { EZ } from '../shared/rules.js';
import {
  Params, Linear, Optimizer, Support, softmaxInto, packParams, unpackParams, paramCount, gauss,
} from './nn.js';
import { Tower, makeTowerSpecs, collectSpecParams } from './tower.js';

export class EZModel {
  /**
   * @param {number} obsDim   dimension of the observation vector
   * @param {number} nActions size of the discrete action space
   * @param {object} cfg      hyper-parameters (defaults to the shared EZ table)
   */
  constructor(obsDim, nActions, cfg = EZ) {
    this.cfg = cfg;
    this.obsDim = obsDim;
    this.nActions = nActions;
    this.LAT = cfg.LATENT;
    this.HID = cfg.HIDDEN;
    this.ACC = cfg.ACC_DIM;
    this.dimIn = this.LAT + nActions + this.ACC;

    this.supportV = new Support(cfg.VALUE_SUPPORT, cfg.VALUE_MIN, cfg.VALUE_MAX);
    this.supportR = new Support(cfg.REW_SUPPORT, cfg.REW_MIN, cfg.REW_MAX);

    this.specs = {
      rep: (() => { const s = makeTowerSpecs(obsDim, cfg.HIDDEN, this.LAT, cfg.N_RES); s.debug = 'rep'; return s; })(),
      dyn: (() => { const s = makeTowerSpecs(this.dimIn, cfg.HIDDEN, this.LAT, cfg.N_RES, { rew: cfg.REW_SUPPORT, acc: this.ACC }); s.debug = 'dyn'; return s; })(),
      pred: (() => { const s = makeTowerSpecs(this.LAT, cfg.HIDDEN, this.LAT, cfg.N_RES, { policy: nActions, value: cfg.VALUE_SUPPORT }); s.debug = 'pred'; return s; })(),
      proj: { W: new Params(this.LAT, this.LAT, 'siam_proj', { gain: 1 }) },
    };

    this.params = collectSpecParams([this.specs.rep, this.specs.dyn, this.specs.pred]);
    this.params.push(this.specs.proj.W);
    this.nParams = paramCount(this.params);

    this.optim = new Optimizer(this.params, {
      lr: cfg.LR,
      weightDecay: cfg.WEIGHT_DECAY,
      clip: cfg.GRAD_CLIP,
    });
    this.trainSteps = 0;
    this.lossHistory = [];

    // one view per unrolled step: shared Params, independent activation caches
    this.views = [];
    for (let i = 0; i <= cfg.UNROLL; i++) this.views.push(makeView(this.specs));
    this.evalView = makeView(this.specs);

    // frozen copy of the prediction tower → bootstrapped value targets
    this.targetSpecs = {
      pred: (() => { const s = makeTowerSpecs(this.LAT, cfg.HIDDEN, this.LAT, cfg.N_RES, { policy: nActions, value: cfg.VALUE_SUPPORT }); s.debug = 'pred'; return s; })(),
    };
    this.targetParams = collectSpecParams([this.targetSpecs.pred]);
    this.targetView = makeView(this.targetSpecs);
    this._syncTarget();
  }

  _predParamList() {
    return collectSpecParams([this.specs.pred]);
  }
  _syncTarget() {
    const src = this._predParamList();
    for (let i = 0; i < this.targetParams.length; i++) this.targetParams[i].copyFrom(src[i]);
  }
  maybeSyncTarget() {
    if (this.trainSteps > 0 && this.trainSteps % this.cfg.TARGET_SYNC === 0) this._syncTarget();
  }

  /* The latent lives on a sphere of unit RMS.

     Without this the residual dynamics (next = latent + out) lets its magnitude
     grow on every update — there is nothing pulling it back — and every head
     downstream then sees logits that scale with it. The loss "converges" to a
     limit cycle while the latent RMS climbs exponentially, which is exactly the
     failure mode a naive EZ reimplementation runs into. Gradients w.r.t. a
     normalised latent are therefore projected back onto the tangent plane, so
     the optimiser can rotate the state but not re-inflate the radius. */
  _norm(latent) {
    let sq = 0;
    const n = latent.length;
    for (let i = 0; i < n; i++) sq += latent[i] * latent[i];
    const rms = Math.sqrt(sq / Math.max(1, n));
    if (!(rms > 1e-6)) return latent;
    const k = 1 / rms;
    for (let i = 0; i < n; i++) latent[i] *= k;
    return latent;
  }

  _tangent(g, latent) {
    const n = latent.length;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += latent[i] * latent[i];
    if (!(sq > 1e-12)) return g;
    let dot = 0;
    for (let i = 0; i < n; i++) dot += g[i] * latent[i];
    const k = dot / sq;
    for (let i = 0; i < n; i++) g[i] -= k * latent[i];
    return g;
  }

  /* ------------------------------------------------------------- inference */

  /** Encode an observation and return the root statistics for the search. */
  root(view, obs) {
    const v = view || this.evalView;
    const s = this._norm(v.rep.forward(obs));
    return this.predictFrom(view, s, s);
  }

  /** P,V heads on a latent (already encoded or imagined). */
  predictFrom(view, latent, encoded) {
    const v = view || this.evalView;
    v.pred.forward(latent);
    const hidden = v.pred.getHidden();
    const polLogits = v.polHead.forward(hidden);
    const valLogits = v.valHead.forward(hidden);
    const probs = new Float32Array(this.supportV.bins);
    softmaxInto(valLogits, probs, this.supportV.bins);
    void encoded;
    return {
      latent,
      view: v,
      polLogits,
      valLogits,
      probs,
      value: this.supportV.decode(probs),
      hidden,
    };
  }
  predictLatent(latent, view) {
    return this.predictFrom(view, latent, latent);
  }

  /** Bootstrap value from the *target* network (stable TD targets). */
  targetValue(latent) {
    const v = this.targetView;
    v.pred.forward(latent);
    const logits = v.valHead.forward(v.pred.getHidden());
    const probs = new Float32Array(this.supportV.bins);
    softmaxInto(logits, probs, this.supportV.bins);
    return this.supportV.decode(probs);
  }

  /**
   * One recurrent dynamics step.
   * @returns {{next, rLogits, rProbs, prefix, accNext, inp, dynView}}
   */
  dynamic(view, latent, actionIdx, acc) {
    const na = this.nActions;
    const inp = new Float32Array(this.dimIn);
    inp.set(latent, 0);
    inp[latent.length + actionIdx] = 1;
    inp.set(acc, this.LAT + na);
    view.dyn.forward(inp);
    const hidden = view.dyn.getHidden();
    const out = view.dyn.out;
    const next = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) next[i] = latent[i] + out[i];
    this._norm(next);   // the recurrence keeps the unit-RMS invariant
    const rLogits = view.rewHead.forward(hidden);
    const rProbs = new Float32Array(this.supportR.bins);
    softmaxInto(rLogits, rProbs, this.supportR.bins);
    const prefix = this.supportR.decode(rProbs);
    const accOut = view.accHead.forward(hidden);
    const accNext = new Float32Array(acc.length);
    for (let i = 0; i < acc.length; i++) accNext[i] = acc[i] + accOut[i];
    return { next, rLogits, rProbs, prefix, accNext, accOut, inp, dynView: view };
  }

  inferObs(obs) {
    return this.root(this.evalView, obs);
  }

  /* ------------------------------------------------------- training a batch */

  /**
   * One EfficientZero minibatch update.
   *
   * @param {Array} seqs each sequence:
   *   {
   *     obs:      Float32Array[]   real observations o_{t..t+l}
   *     acts:     number[]         executed actions a_{t..}
   *     rewards:  number[]         real per-step rewards u_{t..}
   *     policies: Float32Array[]   MCTS target policy π_{t..}
   *     valueTargets?: number[]    z_{t..} (SVE or precomputed)
   *     prefixTargets?: number[]   û_{t..}
   *     depth:    number           unroll length (off-policy corrected)
   *     weight:   number           importance-sampling weight β
   *     tdSteps:  number           n for the TD fallback
   *   }
   * @returns {object} loss bookkeeping for the HUD / server
   */
  trainSequences(seqs) {
    const cfg = this.cfg;
    const lam = {
      r: cfg.L_REWARD, p: cfg.L_POLICY, v: cfg.L_VALUE, c: cfg.L_CONSIST, e: cfg.L_ENTROPY,
    };
    this.optim.zeroGrad();
    const st = { policy: 0, value: 0, reward: 0, consist: 0, entropy: 0, nan: 0, n: 0, maxTd: 0 };
    // the minibatch is *averaged* (summed gradients would multiply the
    // effective learning rate by the batch size)
    const invN = 1 / Math.max(1, seqs.length);
    const vHot = new Float32Array(this.supportV.bins);
    const rHot = new Float32Array(this.supportR.bins);

    for (const seq of seqs) {
      const obs = seq.obs;
      if (!obs || obs.length < 2) continue;
      const depth = Math.max(1, Math.min(cfg.UNROLL, seq.depth || cfg.UNROLL, obs.length - 1));
      const imp = (seq.weight ?? 1) * invN;
      // the paper divides the loss by l_unroll too: otherwise the effective
      // learning rate grows with the unroll length (5× at full depth)
      const ig = imp / depth;

      // ---- forward pass (activations cached in per-step views) -----------
      const steps = [];
      const trueLat = [];
      for (let k = 1; k <= depth; k++) trueLat.push(Float32Array.from(this._norm(this.evalView.rep.forward(obs[k]))));

      let s = this.views[0].rep.forward(obs[0]);
      let acc = new Float32Array(this.ACC);
      let gpAcc = 0;
      for (let k = 0; k < depth; k++) {
        const view = this.views[k];
        gpAcc += Math.pow(cfg.GAMMA_PREFIX, k) * (seq.rewards[k] ?? 0);
        const d = this.dynamic(view, s, seq.acts[k], acc);
        const pr = this.predictFrom(view, d.next, d.next);
        const polProbs = new Float32Array(this.nActions);
        softmaxInto(pr.polLogits, polProbs, this.nActions);
        steps.push({ view, d, pr, polProbs, s, acc, prefixTarget: seq.prefixTargets ? seq.prefixTargets[k] : gpAcc });
        s = d.next;
        acc = d.accNext;
      }

      // ---- backward pass through the unroll ------------------------------
      let carryS = null;   // dL/d ŝ_k  coming from step k+1
      let carryAcc = new Float32Array(this.ACC); // dL/d acc_k coming from step k+1
      let dS0 = null;

      for (let k = depth - 1; k >= 0; k--) {
        const { view, d, pr, polProbs } = steps[k];

        /* value-prefix head: CE against the two-hot encoded prefix --------- */
        const prefixTarget = steps[k].prefixTarget;
        this.supportR.encode(prefixTarget, rHot);
        let dHiddenDyn = new Float32Array(this.HID);
        {
          const p = d.rProbs;
          let ce = 0;
          const dR = new Float32Array(p.length);
          for (let i = 0; i < p.length; i++) {
            if (rHot[i] > 0) ce -= rHot[i] * Math.log(Math.max(1e-8, p[i]));
            dR[i] = (p[i] - rHot[i]) * lam.r * ig;
          }
          st.reward += ce * imp;
          st.maxTd = Math.max(st.maxTd, Math.abs(this.supportR.decode(p) - prefixTarget));
          const dh = view.rewHead.backward(dR, true);
          for (let i = 0; i < dh.length; i++) dHiddenDyn[i] += dh[i];
        }

        /* accumulator head: gradient carried in from the next step --------- */
        {
          const g = new Float32Array(this.ACC);
          for (let i = 0; i < this.ACC; i++) g[i] = carryAcc[i] * ig;
          const dh = view.accHead.backward(g, true);
          for (let i = 0; i < dh.length; i++) dHiddenDyn[i] += dh[i];
        }

        /* prediction heads: value + policy -------------------------------- */
        let z = seq.valueTargets && seq.valueTargets[k] !== undefined ? seq.valueTargets[k] : null;
        if (z === null) {
          let g = 0, disc = 1;
          const n = Math.max(1, seq.tdSteps || 1);
          for (let j = k; j < Math.min(k + n, obs.length - 1); j++) {
            g += disc * (seq.rewards[j] ?? 0);
            disc *= cfg.GAMMA;
          }
          z = g + disc * this.targetValue(d.next);
        }
        this.supportV.encode(z, vHot);
        let dHiddenPred = new Float32Array(this.HID);
        {
          const p = pr.probs;
          let ce = 0;
          const dV = new Float32Array(p.length);
          for (let i = 0; i < p.length; i++) {
            if (vHot[i] > 0) ce -= vHot[i] * Math.log(Math.max(1e-8, p[i]));
            dV[i] = (p[i] - vHot[i]) * lam.v * ig;
          }
          st.value += ce * imp;
          const dh = view.valHead.backward(dV, true);
          for (let i = 0; i < dh.length; i++) dHiddenPred[i] += dh[i];
        }
        {
          const tgt = seq.policies[k];
          const dP = new Float32Array(this.nActions);
          let ce = 0, H = 0;
          for (let a = 0; a < this.nActions; a++) {
            const t = tgt ? tgt[a] : 1 / this.nActions;
            const pm = Math.max(1e-8, polProbs[a]);
            if (t > 0) ce -= t * Math.log(pm);
            H -= pm * Math.log(pm);
          }
          // d(-H)/dz_a = π_a·(log π_a + H). Writing it without the π_a factor (as
          // "−log π_a − 1") is unbounded: a saturated softmax then screams at the
          // head with a gradient proportional to the number of nats, which is how
          // the policy logits reach 1e4 and the loss limit-cycles instead of
          // descending.
          for (let a = 0; a < this.nActions; a++) {
            const t = tgt ? tgt[a] : 1 / this.nActions;
            const pm = Math.max(1e-8, polProbs[a]);
            dP[a] = (polProbs[a] - t) * lam.p * ig + lam.e * ig * pm * (Math.log(pm) + H);
          }
          st.policy += ce * imp;
          st.entropy += H * imp;
          const dh = view.polHead.backward(dP, true);
          for (let i = 0; i < dh.length; i++) dHiddenPred[i] += dh[i];
        }
        const dLatPred = view.pred.backwardHidden(dHiddenPred);

        /* self-supervised consistency: predictor(ŝ_{k+1}) vs detached s_{k+1} */
        const tgtLat = trueLat[Math.min(k, trueLat.length - 1)];
        const projLat = view.proj.forward(d.next);
        let nP = 0, nT = 0;
        for (let i = 0; i < this.LAT; i++) { nP += projLat[i] * projLat[i]; nT += tgtLat[i] * tgtLat[i]; }
        nP = Math.sqrt(nP) + 1e-8;
        nT = Math.sqrt(nT) + 1e-8;
        let cos = 0;
        for (let i = 0; i < this.LAT; i++) cos += (projLat[i] / nP) * (tgtLat[i] / nT);
        st.consist += (1 - cos) * imp;
        const dProj = new Float32Array(this.LAT);
        for (let i = 0; i < this.LAT; i++) dProj[i] = (-lam.c * ig / nP) * (tgtLat[i] / nT);
        const dLatCons = view.proj.backward(dProj, true);

        /* merge everything that flows into ŝ_{k+1}, then step the dynamics back */
        const dNext = new Float32Array(this.LAT);
        for (let i = 0; i < this.LAT; i++) {
          dNext[i] = dLatPred[i] + dLatCons[i] + (carryS ? carryS[i] * 0.5 : 0);
        }
        this._tangent(dNext, d.next);   // ŝ is on the unit sphere: only tangential motion matters
        const dynIn = view.dyn.backwardBoth(dHiddenDyn, dNext);
        const dS = new Float32Array(this.LAT);
        for (let i = 0; i < this.LAT; i++) dS[i] = dNext[i] + dynIn[i];
        for (let i = 0; i < this.ACC; i++) carryAcc[i] = dynIn[this.LAT + this.nActions + i];
        carryS = dS;
        if (k === 0) dS0 = this._tangent(dS, steps[0].s);
      }

      if (dS0) this.views[0].rep.backward(dS0);
      st.n++;
    }

    const gn = this.optim.clipGradNorm();
    if (Number.isFinite(gn) && gn > 0) this.optim.step();
    else st.nan++;
    this.trainSteps++;
    this.maybeSyncTarget();

    const n = Math.max(1, st.n);
    const out = {
      policy: st.policy / n,
      value: st.value / n,
      reward: st.reward / n,
      consist: st.consist / n,
      entropy: st.entropy / n,
      loss: (st.policy + st.value + st.reward + st.consist) / n,
      gradNorm: gn,
      steps: this.trainSteps,
      nan: st.nan,
      seqs: st.n,
      td: st.maxTd,
    };
    this.lossHistory.push(out);
    if (this.lossHistory.length > 400) this.lossHistory.shift();
    return out;
  }

  /* ------------------------------------------------------------ checkpoint */

  checkpoint() {
    return {
      pack: packParams(this.params),
      nParams: this.nParams,
      trainSteps: this.trainSteps,
      meta: { obsDim: this.obsDim, nActions: this.nActions, cfg: publicCfg(this.cfg) },
    };
  }
  loadPack(f32) {
    const ok = unpackParams(f32, this.params);
    if (ok) this._syncTarget();
    return ok;
  }
  cloneWeightsFrom(other) {
    if (other.params.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) this.params[i].copyFrom(other.params[i]);
    this._syncTarget();
    return true;
  }
  resetWeights() {
    for (const p of this.params) {
      const s = Math.sqrt(2 / Math.max(1, p.inDim));
      for (let i = 0; i < p.W.length; i++) p.W[i] = gauss() * s;
      p.b.fill(0);
      p.zeroGrad();
      p.mW.fill(0); p.vW.fill(0); p.mb.fill(0); p.vb.fill(0);
    }
    this.trainSteps = 0;
    this._syncTarget();
  }
  health() {
    let bad = 0, abs = 0, n = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.W.length; i++) {
        const v = p.W[i];
        n++;
        if (!Number.isFinite(v)) bad++;
        abs += Math.abs(v);
      }
    }
    return { params: this.nParams, bad, meanAbs: n ? abs / n : 0, trainSteps: this.trainSteps };
  }
}

export function makeView(specs) {
  return {
    rep: specs.rep ? new Tower(specs.rep) : null,
    dyn: specs.dyn ? new Tower(specs.dyn) : null,
    pred: specs.pred ? new Tower(specs.pred) : null,
    rewHead: specs.dyn ? new Linear(specs.dyn.heads.rew) : null,
    accHead: specs.dyn ? new Linear(specs.dyn.heads.acc) : null,
    polHead: specs.pred ? new Linear(specs.pred.heads.policy) : null,
    valHead: specs.pred ? new Linear(specs.pred.heads.value) : null,
    proj: specs.proj ? new Linear(specs.proj.W) : null,
  };
}

export function publicCfg(cfg) {
  const o = {};
  for (const k of ['LATENT', 'HIDDEN', 'N_RES', 'ACC_DIM', 'VALUE_SUPPORT', 'VALUE_MIN', 'VALUE_MAX',
    'REW_SUPPORT', 'REW_MIN', 'REW_MAX', 'GAMMA', 'GAMMA_PREFIX', 'UNROLL', 'CPUCT', 'LR',
    'WEIGHT_DECAY', 'TARGET_SYNC', 'ALPHA_PRIO', 'BETA_PRIO']) o[k] = cfg[k];
  return o;
}
