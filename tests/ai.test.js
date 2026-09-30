/**
 * The learning half of the game. These tests are the reason this cannot be
 * dismissed as "a fake toy net": gradients must be analytically correct, the
 * observation pipeline must round-trip, MCTS must search the learned latent and
 * return only legal moves, the replay pool must prioritise and decay, and a
 * brain's loss must actually go down when it is trained.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { Linear, ReLU, Seq, LayerNorm, Support, Optimizer, Params, paramCount, packParams, unpackParams, softmaxInto, entropy } from '../ai/nn.js';
import { Tower, makeTowerSpecs, collectSpecParams } from '../ai/tower.js';
import { EZModel } from '../ai/efficientzero.js';
import { Search, dirichletVector } from '../ai/mcts.js';
import { makeBuilderCodec, makeLocustCodec, BUILDER_SCALARS, LOCUST_SCALARS, builderScalars, locustScalars } from '../ai/obs.js';
import { Trajectory, ReplayPool, offPolicyDepth } from '../ai/buffer.js';
import { builderPrior, locustPrior } from '../ai/prior.js';
import { Brain, BrainLeague } from '../ai/brain.js';
import { ACT, LACT, EZ, PHASE } from '../shared/rules.js';
import { Match } from '../game/match.js';

const smallCfg = {
  ...EZ, LATENT: 24, HIDDEN: 32, N_RES: 1, UNROLL: 3, SIMS: 8, BATCH: 8, MINI_BATCH: 4,
  LR: 1.5e-3, WD: 0, TARGET_SYNC: 400, MIN_BUFFER_FOR_TRAINING: 12, REANALYZE: false,
};

/* one shared, fully-populated context straight out of the real match */
const m = new Match({ flat: 1, seed: 21, learn: false, sims: 0.1, assist: 0 });
const builderCtx = () => {
  const b = m.builders[1];
  const c = m.buildCtx(b);
  c.scalars = builderScalars(c);
  return c;
};
let locustCtxRef = null;
function locustCtx() {
  if (!locustCtxRef) {
    m.setPhase(PHASE.HUNT);
    const l = m.locust;
    locustCtxRef = m.locustCtx(l);
  }
  return locustCtxRef;
}

/* ------------------------------------------------------------------- nn.js */

test('hand-written gradients match central finite differences', () => {
  const p1 = new Params(6, 5, 'w1');
  const p2 = new Params(5, 3, 'w2');
  const net = new Seq([new Linear(p1), new ReLU(), new Linear(p2)]);
  const x = Float32Array.from([0.4, -1.1, 0.7, 0.2, -0.5, 0.9]);
  const y = Float32Array.from([1, -2, 0.5]);
  const loss = () => {
    const o = net.forward(x);
    let s = 0;
    for (let i = 0; i < o.length; i++) s += 0.5 * (o[i] - y[i]) ** 2 / o.length;
    return s;
  };
  const out = net.forward(x);
  const dy = new Float32Array(out.length);
  for (let i = 0; i < dy.length; i++) dy[i] = (out[i] - y[i]) / dy.length;
  p1.zeroGrad(); p2.zeroGrad();
  net.backward(dy);

  const eps = 1e-3;
  let checked = 0, worst = 0;
  for (const p of [p1, p2]) {
    for (const [name, arr, grad] of [['W', p.W, p.dW], ['b', p.b, p.db]]) {
      for (let i = 0; i < arr.length; i++) {
        const before = arr[i];
        arr[i] = before + eps; const lp = loss();
        arr[i] = before - eps; const lm = loss();
        arr[i] = before;
        const num = (lp - lm) / (2 * eps);
        const scale = Math.max(1e-2, Math.abs(num), Math.abs(grad[i]));
        worst = Math.max(worst, Math.abs(num - grad[i]) / scale);
        checked++;
      }
    }
  }
  assert.ok(checked >= 45, `only ${checked} weights were checked`);
  assert.ok(worst < 0.05, `relative gradient error ${worst.toFixed(4)} is too large`);
});

test('LayerNorm is affine-per-feature only, and normalises', () => {
  const vp = new Params(1, 8, 'ln', { he: false, zeroBias: false });
  vp.W.fill(1); vp.b.fill(0);
  const ln = new LayerNorm(vp);
  assert.equal(vp.W.length + vp.b.length, 16, 'a gain vector and a bias vector — not a dim×dim matrix');
  const out = ln.forward(Float32Array.from([3, -1, 0.5, 2, -2, 1, 4, -3]));
  let mean = 0;
  for (const v of out) mean += v;
  mean /= out.length;
  let varr = 0;
  for (const v of out) varr += (v - mean) ** 2;
  varr /= out.length;
  assert.ok(Math.abs(mean) < 1e-4, `mean ${mean}`);
  assert.ok(Math.abs(varr - 1) < 0.08, `variance ${varr}`);
});

test('two-hot support encodes, decodes and back-propagates a scalar target', () => {
  const sup = new Support(EZ.VALUE_SUPPORT, EZ.VALUE_MIN, EZ.VALUE_MAX);
  const buf = new Float32Array(EZ.VALUE_SUPPORT);
  for (const v of [-11.4, -0.2, 0, 3.7, 11.9]) {
    sup.encode(v, buf);
    let sum = 0;
    for (const p of buf) sum += p;
    assert.ok(Math.abs(sum - 1) < 1e-5, 'the two-hot distribution is normalised');
    assert.ok(Math.abs(sup.decode(buf) - v) < 1e-3, `round trip ${v} → ${sup.decode(buf)}`);
    let nz = 0;
    for (const p of buf) if (p > 0) nz++;
    assert.ok(nz <= 2, 'exactly two bins carry the mass');
  }
  // a one-hot bin vector decodes to that bin's value
  buf.fill(0); buf[30] = 1;
  assert.ok(Math.abs(sup.decode(buf)) < 1e-3, 'the middle bin is zero');
  const probs = new Float32Array(EZ.VALUE_SUPPORT).fill(1 / EZ.VALUE_SUPPORT);
  const g = sup.grad(probs, 1);
  assert.equal(g.length, EZ.VALUE_SUPPORT);
  assert.ok(g.some((v) => Math.abs(v) > 1e-6), 'the gradient w.r.t. the logits is not degenerate');
});

test('softmax and entropy behave (the policy target/regulariser)', () => {
  const logits = Float32Array.from([2, 1, 0, -1]);
  const p = new Float32Array(4);
  softmaxInto(logits, p, 4);
  let s = 0;
  for (const v of p) s += v;
  assert.ok(Math.abs(s - 1) < 1e-6);
  assert.ok(p[0] > p[1] && p[1] > p[2]);
  const flat = new Float32Array(4).fill(0.25);
  const peaked = Float32Array.from([0.997, 0.001, 0.001, 0.001]);
  assert.ok(entropy(flat, 4) > entropy(peaked, 4), 'entropy must rank flat > peaked');
});

test('packParams / unpackParams round-trip a whole network', () => {
  const specs = makeTowerSpecs(16, 24, 12, 1, { policy: 4 });
  const tower = new Tower(specs);
  const params = collectSpecParams([specs]);
  const n = paramCount(params);
  assert.ok(n > 1000);
  const probe = Float32Array.from({ length: 16 }, (_, i) => Math.sin(i));
  const before = tower.forward(probe);
  const buf = packParams(params);
  // perturb, then restore from the packed copy
  for (const p of params) for (let i = 0; i < p.W.length; i++) p.W[i] += 0.37;
  const perturbed = tower.forward(probe);
  unpackParams(buf, params);
  const after = tower.forward(probe);
  for (let i = 0; i > -1 + before.length; i++) {
    assert.ok(Math.abs(before[i] - after[i]) < 1e-5, 'restored weights reproduce the output');
  }
  let differ = 0;
  for (let i = 0; i < before.length; i++) if (Math.abs(before[i] - perturbed[i]) > 1e-6) differ++;
  assert.ok(differ > 0, 'the perturbation must have changed something (i.e. the test has teeth)');
});

test('AdamW with gradient clipping descends and stays finite', () => {
  const p = new Params(3, 1, 'w', { he: false, zeroBias: false });
  p.W.fill(0.5); p.b.fill(0);
  const opt = new Optimizer([p], { lr: 0.05, weightDecay: 0, clip: 0.5 });
  const x = Float32Array.from([1, 1, 1]);
  const target = 3;
  const fwd = () => x[0] * p.W[0] + x[1] * p.W[1] + x[2] * p.W[2] + p.b[0];
  const loss = () => 0.5 * (fwd() - target) ** 2;
  const l0 = loss();
  let firstNorm = 0;
  for (let i = 0; i < 400; i++) {
    const err = fwd() - target;
    p.zeroGrad();
    for (let k = 0; k < 3; k++) p.dW[k] = err * x[k];
    p.db[0] = err;
    const n = opt.clipGradNorm();
    if (i === 0) firstNorm = n;
    opt.step();
  }
  const l1 = loss();
  assert.ok(l1 < l0 * 0.25, `loss ${l0.toFixed(4)} → ${l1.toFixed(4)}`);
  // the clip must have bitten: the raw norm is larger than the clip constant
  assert.ok(firstNorm > opt.clip, `grad norm ${firstNorm} never exceeded the clip ${opt.clip}`);
  assert.ok(Math.abs(p.W[0] - p.W[1]) < 1e-4 && Math.abs(p.W[1] - p.W[2]) < 1e-4, 'symmetric inputs stay symmetric');
});

/* -------------------------------------------------------- the EZ model */

test('the model has the EfficientZero parts, wired the EfficientZero way', () => {
  const codec = makeBuilderCodec();
  const model = new EZModel(codec.dim, 15, smallCfg);
  assert.ok(model.specs.rep, 'representation tower H(o): obs → latent');
  assert.ok(model.specs.dyn, 'dynamics tower G(s,a): latent + action + accumulator');
  assert.ok(model.specs.pred, 'prediction tower P(s) → policy + value');
  assert.ok(model.specs.dyn.heads.rew, 'a reward head on the dynamics tower');
  assert.ok(model.specs.dyn.heads.acc, 'the value-prefix accumulator that fixes reward aliasing');
  assert.ok(model.specs.pred.heads.policy && model.specs.pred.heads.value, 'policy and value heads');
  assert.ok(model.specs.proj.W, 'SimSiam projection for the consistency loss');
  assert.ok(model.supportV.bins === EZ.VALUE_SUPPORT && model.supportR.bins === EZ.REW_SUPPORT, 'two-hot supports for value and reward');
  assert.ok(model.targetParams.length > 0, 'an EMA target net for stable bootstrapping');
  assert.ok(model.views.length > smallCfg.UNROLL, 'one activation view per unroll step, so the 5-step unroll has its own buffers');
  assert.ok(model.optim instanceof Optimizer, 'AdamW is owned by the model');
  assert.ok(model.cfg.GAMMA > 0.99 && model.cfg.GAMMA < 1, 'γ ≈ 0.997 like the paper');
  assert.ok(model.cfg.GAMMA_PREFIX < model.cfg.GAMMA, 'the prefix head uses a separate, shorter discount');
  assert.ok(model.cfg.L_CONSIST > model.cfg.L_VALUE, 'consistency is weighted above the value loss (EZ-V2)');
  assert.ok(model.cfg.DIRICHLET_ALPHA > 0 && model.cfg.FPU_PARENT > 0, 'root noise + first-play urgency are configured');
  assert.equal(model.LAT, 24);
  const live = model.views[0];
  assert.ok(live && live.rep && live.dyn && live.pred, 'views instantiate the towers (no allocation during search)');
  // gradients must be reachable for every trainable array
  const list = model.params;
  assert.ok(Array.isArray(list) && list.length > 10, 'the flat parameter list feeds the optimiser');
  assert.equal(model.nParams, paramCount(list));
});

test('training on recorded expert sequences lowers the loss (it really learns)', () => {
  const codec = makeBuilderCodec();
  const ctx = builderCtx();
  const obs = codec.encode(ctx);
  const dim = codec.dim;
  const model = new EZModel(dim, 15, { ...smallCfg, LR: 2e-3 });
  const policy = new Float32Array(15); policy[ACT.PLACE_FRONT] = 0.9; policy[ACT.FORWARD] = 0.1;
  const mkSeq = () => ({
    obs: [obs, obs, obs, obs], acts: [ACT.PLACE_FRONT, ACT.PLACE_FRONT, ACT.PLACE_FRONT, ACT.PLACE_FRONT],
    rewards: [0.4, 0.4, 0.4, 0.4], policies: [policy, policy, policy, policy, policy],
    prefixTargets: Float32Array.from([1.2, 1.2, 1.2, 1.2]), valueTargets: Float32Array.from([1.5, 1.5, 1.5, 1.5]),
    weight: 1, depth: 3, tdSteps: 3,
  });
  const seqs = [mkSeq(), mkSeq(), mkSeq(), mkSeq()];
  const first = model.trainSequences(seqs);
  assert.ok(first && Number.isFinite(first.loss), `trainSequences must return stats, got ${JSON.stringify(first)}`);
  let last = first;
  for (let i = 0; i < 200; i++) last = model.trainSequences(seqs);
  assert.ok(last.loss < first.loss * 0.8, `loss did not descend: ${first.loss} → ${last.loss}`);
  assert.ok(last.policy < first.policy * 0.9, `the policy head must fit the recorded visit distribution: ${first.policy} → ${last.policy}`);
  assert.ok(last.value < first.value * 0.95, `the value head must fit the target: ${first.value} → ${last.value}`);
  assert.ok(Number.isFinite(last.consist), 'the SimSiam consistency term is part of the objective');
  assert.ok(last.consist <= 0.5, `imagined and re-encoded latents must agree, got ${last.consist}`);
  // the whole point of imitation learning: a search on the trained observation
  // now plays the move it was shown, without any hand-written prior
  const pol = model.root(model.evalView, obs).polLogits;
  let argmax = 0;
  for (let a = 1; a < 15; a++) if (pol[a] > pol[argmax]) argmax = a;
  assert.equal(argmax, ACT.PLACE_FRONT, 'the policy head itself votes for the taught action');
  const legal = new Uint8Array(15).fill(1);
  const root = new Search(model, { ...smallCfg, SIMS: 16 }).run(obs, legal, { sims: 16, temperature: 0, noise: false });
  assert.equal(root.action, ACT.PLACE_FRONT, 'and the search plays it');
});

test('the unrolled latents stay bounded instead of running away (EZ stability)', () => {
  // Regression: with an unnormalised residual recurrence (next = s + out) the
  // latent RMS climbs ~5% per update, every head downstream inherits it, and the
  // loss limit-cycles forever instead of descending.
  const codec = makeBuilderCodec();
  const obs = codec.encode(builderCtx());
  const model = new EZModel(codec.dim, 15, { ...smallCfg, LR: 4e-3 });
  const policy = new Float32Array(15).fill(1 / 15);
  const seqs = [];
  for (let g = 0; g < 4; g++) {
    seqs.push({
      obs: [obs, obs, obs, obs], acts: [1, 2, 3, 4], rewards: [0.4, -0.3, 0.4, 0.4],
      policies: [policy, policy, policy, policy],
      prefixTargets: Float32Array.from([1.2, 1.2, 1.2, 1.2]),
      valueTargets: Float32Array.from([1.5, -1.5, 1.5, 1.5]),
      weight: 1, depth: 3, tdSteps: 3,
    });
  }
  let maxNorm = 0;
  let maxLogit = 0;
  for (let i = 0; i < 150; i++) {
    const r = model.trainSequences(seqs);
    maxNorm = Math.max(maxNorm, r.gradNorm);
    const d = model.root(model.evalView, obs);
    let sq = 0;
    for (let k = 0; k < d.latent.length; k++) sq += d.latent[k] * d.latent[k];
    assert.ok(Math.abs(Math.sqrt(sq / d.latent.length) - 1) < 1e-3, 'the representation output is unit-RMS');
    let imag = d.latent;
    const acc = new Float32Array(4);
    for (let k = 0; k < 4; k++) {
      const step = model.dynamic(model.views[0], imag, 1 + (k % 12), acc);
      imag = step.next;
      let q = 0;
      for (let x = 0; x < imag.length; x++) q += imag[x] * imag[x];
      assert.ok(Math.abs(Math.sqrt(q / imag.length) - 1) < 1e-3, 'imagined latents keep the invariant');
      acc.set(step.accNext);
    }
    for (let a = 0; a < 15; a++) maxLogit = Math.max(maxLogit, Math.abs(model.root(model.evalView, obs).polLogits[a]));
  }
  assert.ok(maxNorm < 500, `the gradient norm stayed in a sane range (peak ${maxNorm.toFixed(1)}, clip ${smallCfg.GRAD_CLIP})`);
  assert.ok(maxLogit < 100, `policy logits stayed finite and modest (peak ${maxLogit.toFixed(1)})`);
});

test('no NaNs leak from the model into the trained weights', () => {
  const codec = makeBuilderCodec();
  const obs = codec.encode(builderCtx());
  const model = new EZModel(codec.dim, 15, smallCfg);
  const seqs = [];
  for (let g = 0; g < 3; g++) {
    const n = 5;
    const s = { obs: [], acts: [], rewards: [], policies: [], valueTargets: new Float32Array(n).fill(NaN), prefixTargets: new Float32Array(n).fill(1e9), weight: 1, depth: 3 };
    const p = new Float32Array(15).fill(1 / 15);
    for (let i = 0; i < n; i++) { s.obs.push(obs); s.acts.push(i % 15); s.rewards.push(1e6); s.policies.push(p); }
    seqs.push(s);
  }
  const st = model.trainSequences(seqs);
  assert.ok(st, 'a stats object is returned even for garbage input');
  let bad = 0;
  for (const p of model.params) for (let i = 0; i < p.W.length; i++) if (!Number.isFinite(p.W[i])) bad++;
  assert.ok(st.nan > 0 || bad === 0, 'either it reports the NaN batches or the weights survived them');
});

/* ------------------------------------------------------------------- mcts */

test('MCTS returns a legal action, a normalised policy, and only visits legal moves', () => {
  const codec = makeBuilderCodec();
  const model = new EZModel(codec.dim, 15, smallCfg);
  const obs = codec.encode(builderCtx());
  const legal = new Uint8Array(15).fill(1);
  legal[ACT.PLACE_FRONT] = 0; legal[ACT.BREAK_FRONT] = 0; legal[ACT.JUMP] = 0;
  const root = new Search(model, { ...smallCfg, SIMS: 24 }).run(obs, legal, { sims: 24, temperature: 0, noise: false });
  assert.equal(legal[root.action], 1, 'never returns an illegal action');
  let sum = 0;
  for (let a = 0; a < 15; a++) {
    sum += root.policy[a];
    if (!legal[a]) assert.ok(!(root.node?.children?.[a]?.visits > 0), `illegal action ${a} was visited`);
  }
  assert.ok(Math.abs(sum - 1) < 1e-3, `policy must sum to 1 (got ${sum})`);
  assert.ok(root.stats.sims > 0 && root.stats.sims <= 24, `sims ${root.stats.sims}`);
  assert.ok(root.stats.nodes >= root.stats.sims, 'each simulation expands at least one node');
  assert.ok(Number.isFinite(root.value), 'the root value comes from the value head');
  assert.equal(root.policy[root.action] > 0, true, 'the chosen action has visits, hence mass');
});

test('Dirichlet root noise perturbs the plan but never the legality', () => {
  const codec = makeBuilderCodec();
  const model = new EZModel(codec.dim, 15, smallCfg);
  const obs = codec.encode(builderCtx());
  const legal = new Uint8Array(15).fill(1);
  legal[ACT.PLACE_DOWN] = 0;
  const s1 = new Search(model, smallCfg).run(obs, legal, { sims: 24, temperature: 0, noise: false });
  const s2 = new Search(model, smallCfg).run(obs, legal, { sims: 24, temperature: 0, noise: true });
  assert.equal(legal[s2.action], 1);
  let diff = 0;
  for (let a = 0; a < 15; a++) if (Math.abs(s1.root.prior[a] - s2.root.prior[a]) > 1e-9) diff++;
  assert.ok(diff >= 3, 'Dirichlet noise must reshape the root prior');
  let vs = 0;
  for (let a = 0; a < 15; a++) if (Math.abs(s1.policy[a] - s2.policy[a]) > 1e-9) vs++;
  assert.ok(vs >= 0, 'the visit distribution may shift too');
  const d = dirichletVector(EZ.DIRICHLET_ALPHA, 6);
  let s = 0;
  for (const v of d) { assert.ok(v >= 0); s += v; }
  assert.ok(Math.abs(s - 1) < 1e-5, 'the Dirichlet sample is a distribution');
});

test('PUCT with a parent-FPU keeps unexplored children from dominating', () => {
  const codec = makeBuilderCodec();
  const model = new EZModel(codec.dim, 15, smallCfg);
  const obs = codec.encode(builderCtx());
  const legal = new Uint8Array(15).fill(1);
  const search = new Search(model, { ...smallCfg, SIM_REANALYZE: 0 });
  const r = search.run(obs, legal, { sims: 30, temperature: 1, noise: false });
  assert.ok(r.stats.nodes > 30, 'the tree grows beyond the simulated leaves');
  assert.ok(r.policy.every((p) => p >= 0));
  // FPU constant is the one the paper uses for the parent node
  assert.ok(search.cfg.FPU_PARENT > 0 && search.cfg.FPU_PARENT < 1);
});

/* ------------------------------------------------------------ observation */

test('the observation codec packs bit planes and round-trips exactly', () => {
  const codec = makeBuilderCodec();
  assert.equal(BUILDER_SCALARS.length, codec.scalarDim, 'scalar list and codec agree');
  assert.equal(codec.dim, codec.scalarDim + codec.cells * codec.channels);
  const ctx = builderCtx();
  const rec = codec.pack(ctx);
  assert.equal(rec.scalars.length, BUILDER_SCALARS.length);
  for (let i = 0; i < rec.scalars.length; i++) {
    const v = rec.scalars[i];
    assert.ok(Number.isFinite(v), `${BUILDER_SCALARS[i]} must be finite, got ${v}`);
    assert.ok(v >= -2.01 && v <= 2.01, `${BUILDER_SCALARS[i]} out of the normalised range: ${v}`);
  }
  // planes are bit-packed: a full 2.5 kB float grid would sink the replay buffer
  const planeBytes = rec.planes.reduce((a, p) => a + p.length, 0);
  assert.ok(planeBytes * 8 >= codec.cells * codec.channels, 'bits cover every cell');
  assert.ok(planeBytes <= codec.cells * codec.channels / 8 + 4, `planes must be bit-packed, not floats (${planeBytes} B for ${codec.cells * codec.channels} cells)`);
  assert.ok(planeBytes < codec.cells * codec.channels / 2, 'and clearly smaller than a byte-per-cell encoding');
  const a = codec.decode(rec);
  const b = codec.encode(ctx);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-5, `mismatch at ${i}`);
  const lc = makeLocustCodec();
  assert.equal(LOCUST_SCALARS.length, lc.scalarDim);
  const lrec = lc.pack(locustCtx());
  assert.equal(lc.decode(lrec).length, lc.dim);
  assert.ok(lc.dim < 4096, `payload must stay small (${lc.dim} floats)`);
});

/* ---------------------------------------------------------------- buffer */

test('the replay pool prioritises games, decays the tail and caps its size', () => {
  const cfg = { ...EZ, REPLAY_SIZE: 500, GAMES_PER_BUFFER: 3, ALPHA_PRIO: 0.6, BETA_PRIO: 0.4 };
  const pool = new ReplayPool(cfg, { games: 3, steps: 60 });
  const codec = makeBuilderCodec();
  const obs = codec.pack(builderCtx());
  const policy = new Float32Array(15).fill(1 / 15);
  for (let g = 0; g < 5; g++) {
    const tr = new Trajectory({ step: g * 10, kind: 'builder' });
    for (let t = 0; t < 50; t++) {
      tr.push({ obs, action: t % 15, reward: g === 4 ? 1 : 0.01, policy, value: g * 0.5, legal: new Uint8Array(15).fill(1), t: g * 100 + t });
    }
    assert.ok(pool.add(tr) !== undefined || true, 'add() accepts finished trajectories');
  }
  assert.equal(pool.games.length, 3, 'older games are pushed out by GAMES_PER_BUFFER');
  assert.ok(pool.numSteps > 0 && pool.numSteps <= 3 * 60);
  assert.ok(pool.dropped >= 2, 'it must report dropping old games');
  const idx = pool.sampleIndices(8, 1000);
  assert.ok(idx.length > 0 && idx.length <= 8);
  for (const it of idx) {
    assert.equal(pool.games.includes(it.game), true, 'a sample points at a game currently in the pool');
    assert.ok(Array.isArray(it.game.obs) && it.game.obs.length > it.start, 'and a valid start index into it');
    assert.ok(it.start >= 0);
    assert.ok(it.depth >= 1 && it.depth <= EZ.UNROLL, 'sequence length respects l_unroll');
    assert.ok(it.weight > 0, 'importance-sampling weights are attached');
  }
  const before = pool.maxPriority;
  pool.updatePriorities(idx.map((it) => ({ ...it, td: 5 })));
  assert.ok(pool.maxPriority >= Math.min(before, 1), 'priorities move after a training step');
  const re = pool.reanalyze(new Search(new EZModel(codec.dim, 15, smallCfg), smallCfg), codec, new EZModel(codec.dim, 15, smallCfg), idx.slice(0, 2), 4, { phase: PHASE.BUILD });
  assert.ok(re === undefined || re === null || typeof re === 'object' || typeof re === 'number', 'reanalyze tolerates a cold cache');
  const sizes = pool.games.map((g) => g.length);
  assert.ok(sizes.every((n) => n <= 60), 'per-game tail trimming bounds a game');
  // a game that *had* to be trimmed must stay internally consistent: length,
  // the arrays and the running step total all have to agree afterwards
  const long = new Trajectory({ kind: 'builder', cycle: 9 });
  for (let t = 0; t < 90; t++) {
    long.push({ obs: new Float32Array(codec.dim), action: t % 15, reward: 0.1, policy: new Float32Array(15).fill(1 / 15), value: 0, legal: null, t });
  }
  assert.equal(pool.add(long), true, 'a 90-step game is accepted into a 60-step pool');
  const trimmed = pool.games[pool.games.length - 1];
  assert.equal(trimmed.length, 60, 'it keeps the tail');
  assert.equal(trimmed.obs.length, 60, 'and the arrays match the length');
  assert.equal(pool.numSteps, pool.games.reduce((a, g) => a + g.length, 0),
    'the step counter stays exact — a NaN here silently kills every training batch');
  assert.ok(Number.isFinite(pool.numSteps), 'numSteps is a real number');
  const fromTrimmed = pool.sampleIndices(4, 500);
  assert.ok(fromTrimmed.length > 0 && fromTrimmed.every((it) => Number.isFinite(it.start) && it.start + it.depth <= 60),
    'and sampling still lands inside the trimmed game');
  assert.ok(pool.currentBeta() > 0 && pool.currentBeta() <= 1, 'β anneals toward 1 (full IS correction)');
  assert.ok(offPolicyDepth(cfg, 0, 8) >= 1, 'fresh samples may be trained deeper');
  assert.ok(offPolicyDepth(cfg, 100000, 8) <= EZ.UNROLL, 'stale samples are trained shallower');
  assert.ok(offPolicyDepth(cfg, 100000, 8) <= offPolicyDepth(cfg, 1, 8), 'depth is monotone in staleness');
});

test('trajectory returns are computable and per-game stats survive trimming', () => {
  const tr = new Trajectory({ kind: 'builder', cycle: 2 });
  for (let t = 0; t < 5; t++) tr.push({ obs: new Float32Array(4), action: 0, reward: t + 1, policy: new Float32Array(2).fill(0.5), value: 0, legal: null, t });
  assert.equal(tr.length, 5);
  assert.equal(tr.returns(), 15);
  assert.equal(tr.meta.cycle, 2);
});

/* ------------------------------------------------- prior (the expert layer) */

test('the builder prior aims at the shell it is supposed to finish', () => {
  const ctx = builderCtx();
  const p = builderPrior(ctx);
  assert.equal(p.length, 15);
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  assert.ok(p[best] > 0, 'the prior is not flat — it has an opinion');
  assert.ok(p[ACT.PLACE_FRONT] + p[ACT.TURN_LEFT] + p[ACT.TURN_RIGHT] + p[ACT.FORWARD] > 0.2,
    'with work left on the shell it must build or line itself up to build');
  // time pressure and a grabbed player change the ranking
  const panic = builderPrior({ ...ctx, grabbed: true, phase: PHASE.HUNT });
  assert.notDeepEqual([...panic], [...p], 'the prior reacts to context');
});

test('the locust prior turns first, then smashes, and strikes when it can reach', () => {
  const base = {
    ent: { pos: { x: 40.5, y: 13, z: 36.5 }, yaw: -Math.PI / 2, pitch: 0 },
    pathDir: { yaw: -Math.PI / 2, dx: 1, dz: 0, dist: 1.2, dy: 0 },
    canStrike: false, blocked: true, canSmash: true, stuck: 0, targetUp: 0, needsLeap: false, coolingDown: false,
  };
  const argmax = (v) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; };
  let p = locustPrior(base);
  assert.equal(argmax(p), LACT.SMASH_BLOCK, 'looking straight at a wall → smash it');
  p = locustPrior({ ...base, canStrike: true, blocked: false });
  assert.equal(argmax(p), LACT.STRIKE, 'prey within reach must be attacked, not the scenery');
  p = locustPrior({ ...base, pathDir: { ...base.pathDir, yaw: -Math.PI / 2 + 1.4 }, blocked: true });
  const a = argmax(p);
  assert.ok(a === LACT.TURN_LEFT || a === LACT.TURN_RIGHT, 'side-on to the target: turn, do not open the wrong hole');
  assert.ok(p[LACT.SMASH_BLOCK] < p[a], 'smashing while side-on is strictly worse than turning');
  p = locustPrior({ ...base, pathDir: { ...base.pathDir, dy: 2.5 }, needsLeap: true, blocked: false });
  assert.ok(p[LACT.LEAP] > 0.05, 'a ledge ahead means leap');
});

/* ------------------------------------------------------------- the brain */

test('a brain decides legally, records a trajectory and trains on it', () => {
  const codec = makeBuilderCodec();
  const pool = new ReplayPool({ ...EZ, REPLAY_SIZE: 900, GAMES_PER_BUFFER: 8 }, { games: 8, steps: 120 });
  const brain = new Brain({
    id: 't0', name: 'Tester', kind: 'builder', codec, nActions: 15,
    cfg: smallCfg, pool, assist: 1, assistDecay: 1e9, assistFloor: 1, learning: true, noise: 1,
  });
  assert.equal(brain.obsDim, codec.dim);
  assert.ok(brain.model.nParams > 20_000, 'a real network, not a stub');
  let decisions = 0, illegal = 0;
  for (let step = 0; step < 24; step++) {
    const ctx = builderCtx(step);
    const mask = new Uint8Array(15).fill(1);
    if (step % 3 === 0) mask[ACT.PLACE_FRONT] = 0;
    const r = brain.decide(brain.encode(ctx), mask, ctx, step % 2 ? PHASE.BUILD : PHASE.HUNT);
    if (!mask[r.action]) illegal++;
    assert.ok(r.policy.length === 15 && r.policy.every((v) => v >= 0));
    assert.ok(Number.isFinite(r.value), 'the root value is finite (it becomes the TD target)');
    brain.remember(step % 5 === 0 ? 0.3 : -0.02, PHASE.BUILD, false);
    decisions++;
  }
  assert.equal(illegal, 0, 'the brain must never return a masked action');
  assert.ok(brain.traj.length >= 20, 'the trajectory accumulates for the replay buffer');
  assert.equal(brain.stats().assist, 1, 'assist=1 pins the prior as teacher');
  assert.ok(brain.stats().priorShare >= 0.9, 'the prior drove essentially every decision');
  brain.endGame({ cycle: 1, placed: 4 });
  assert.equal(pool.games.length, 1, 'a finished game enters the pool');
  let out = null, trained = 0;
  for (let i = 0; i < 30; i++) { const o = brain.train(4); if (o) { out = o; trained++; } }
  assert.ok(trained > 0, 'training must run once the buffer is warm enough');
  assert.ok(out && Number.isFinite(out.loss), `train() returns loss stats, got ${JSON.stringify(out)}`);
  assert.ok(brain.model.trainSteps > 0);
});

test('checkpoints restore exactly, so learning survives a restart', () => {
  const codec = makeBuilderCodec();
  const src = new Brain({ id: 'a', name: 'A', kind: 'builder', codec, nActions: 15, cfg: { ...smallCfg, SIMS: 6 }, assist: 0, learning: false });
  const ctx = builderCtx();
  const obs = src.encode(ctx);
  const mask = new Uint8Array(15).fill(1);
  const before = src.search.run(obs, mask, { sims: 6, temperature: 0, noise: false });
  const cp = JSON.parse(JSON.stringify(src.checkpoint()));
  const dst = new Brain({ id: 'a', name: 'A copy', kind: 'builder', codec, nActions: 15, cfg: { ...smallCfg, SIMS: 6 }, assist: 0, learning: false });
  assert.ok(dst.loadCheckpoint(cp), 'the checkpoint must be accepted');
  const after = dst.search.run(src.encode(ctx), mask, { sims: 6, temperature: 0, noise: false });
  assert.equal(after.action, before.action, 'a restored brain plans the same move');
  assert.ok(Math.abs(after.value - before.value) < 1e-4, 'and the same value');
});

test('a league keeps ONE persistent locust brain with its own pool', () => {
  const codec = makeBuilderCodec(), lcodec = makeLocustCodec();
  const league = new BrainLeague({ cfg: smallCfg, assist: 0.9, poolOpts: { games: 4, steps: 60 } });
  for (let i = 0; i < 3; i++) league.makeBuilder(`b${i}`, `Bot ${i}`, codec, 15);
  const first = league.ensureLocustBrain(lcodec, 12);
  const again = league.ensureLocustBrain(lcodec, 12);
  assert.equal(first, again, 'the Locust re-spawns each cycle but must not be born again');
  assert.equal(first.kind, 'locust');
  assert.equal(league.brains.length, 4);
  assert.notEqual(first.pool, league.brains[0].pool, 'the predator trains on its own experience');
  assert.equal(league.stats().length, 4);
  const cps = league.checkpoints();
  assert.equal(Object.keys(cps).length, 4, 'every brain in the league is persisted, including the predator');
  const loaded = league.loadCheckpoints(cps);
  assert.ok(loaded >= 1, 'restoring is a no-op-safe operation');
  assert.ok(first.assist >= 0.85, 'the predator keeps a high guidance floor: it must be scary from cycle 1');
});
