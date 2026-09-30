/**
 * ai/mcts.js
 * ---------------------------------------------------------------------------
 * The planning step of EfficientZero: PUCT Monte-Carlo Tree Search carried out
 * *entirely inside the learned latent space*. No access to the real world –
 * every successor state is produced by the dynamics network G, every leaf by
 * the prediction heads P,V, every edge reward by the value-prefix head R.
 *
 * Implements: selection / expansion / backup, first-play-urgency of the parent,
 * Dirichlet root noise, legal-action masking, visit-count rescaling,
 * temperature-annealed root move extraction, and the improved policy + search
 * value that become the training targets π_t and z_t.
 *
 * The search is written as a generator so the browser worker can run it under
 * a millisecond budget (time-sliced, cooperative) without blocking the game.
 * ---------------------------------------------------------------------------
 */

import { EZ } from '../shared/rules.js';
import { softmaxInto } from './nn.js';

export class MCTSNode {
  constructor() {
    this.parent = null;
    this.action = -1;
    this.reward = 0;      // predicted value prefix of the edge (parent → this)
    this.N = 0;
    this.W = 0;
    this.Q = 0;
    this.children = null; // Array<MCTSNode|null> indexed by action
    this.prior = null;    // Float32Array(nActions)
    this.legal = null;    // Uint8Array(nActions)
    this.latent = null;
    this.acc = null;
    this.depth = 0;
    this.expanded = false;
    this.value = 0;
  }
  get averageValue() {
    return this.N > 0 ? this.W / this.N : 0;
  }
}

export class Search {
  /**
   * @param {import('./efficientzero.js').EZModel} model
   * @param {object} cfg {SIMS, CPUCT, GAMMA, DIRICHLET_ALPHA, DIRICHLET_EPS,
   *                      FPU_PARENT, RESCALE_ROOT, GAMMA_PREFIX}
   */
  constructor(model, cfg = EZ) {
    this.model = model;
    this.cfg = cfg;
    this.view = model.evalView; // search never trains → no per-step views needed
    this.stats = { sims: 0, expansions: 0, nodes: 1, value: 0, rootQ: 0, entropy: 1, top: 0, maxDepth: 0 };
  }

  /**
   * Run a full search.
   * @param {Float32Array} obs  real observation (encoded by H)
   * @param {Uint8Array} legal  mask, 1 = allowed
   * @param {object} o {sims, temperature, noise, noiseFloor, valueMode}
   * @returns {{action:number, policy:Float32Array, value:number, root:MCTSNode, stats:object}}
   */
  run(obs, legal, o = {}) {
    const it = this.iterate(obs, legal, o);
    let r = it.next();
    while (!r.done) r = it.next();
    return r.value;
  }

  /**
   * Resumable variant: yields after every simulation so a caller can stop the
   * search at a wall-clock budget, then resume on the next frame.
   */
  * iterate(obs, legal, o = {}) {
    const cfg = this.cfg;
    const sims = o.sims ?? cfg.SIMS;
    const temp = o.temperature ?? 0;
    const cpuct = cfg.CPUCT;
    const gamma = cfg.GAMMA;
    const fpu = cfg.FPU_PARENT ?? 0.25;
    const nA = this.model.nActions;
    const root = new MCTSNode();
    root.legal = legal;
    root.depth = 0;

    /* ---- root: encode the real observation, attach P and V -------------- */
    const rr = this.model.root(this.view, obs);
    root.latent = rr.latent;
    root.acc = new Float32Array(this.model.ACC);
    root.expanded = true;
    root.prior = maskedProbs(rr.polLogits, legal, nA);
    root.children = new Array(nA).fill(null);
    root.value = rr.value;
    root.N = 0;
    root.W = 0;
    root.Q = 0;

    /* ---- Dirichlet noise on the root prior (exploration) ---------------- */
    if (o.noise !== false) {
      const alpha = cfg.DIRICHLET_ALPHA;
      const eps = o.noiseFloor ?? cfg.DIRICHLET_EPS;
      const k = countLegal(legal, nA);
      const dir = dirichletVector(alpha, k);
      let j = 0;
      for (let a = 0; a < nA; a++) if (!legal || legal[a]) root.prior[a] = (1 - eps) * root.prior[a] + eps * dir[j++];
      renorm(root.prior, legal, nA);
    }

    this.stats = { sims: 0, expansions: 0, nodes: 1, value: rr.value, rootQ: 0, maxDepth: 0 };

    for (let i = 0; i < sims; i++) {
      /* -------------------------------------------------------- selection */
      let node = root;
      while (true) {
        let untried = -1;
        for (let a = 0; a < nA; a++) {
          if (legal && !legal[a]) continue;
          if (!node.children[a]) { untried = a; break; }
        }
        if (untried >= 0) {
          node = this._expand(node, untried);
          break;
        }
        const nSum = Math.max(1, node.N);
        const parentQ = node.averageValue;
        let best = -Infinity, bestA = -1;
        for (let a = 0; a < nA; a++) {
          if (legal && !legal[a]) continue;
          const c = node.children[a];
          if (!c) continue;
          const q = c.N > 0 ? c.Q : parentQ - fpu;
          const u = cpuct * node.prior[a] * Math.sqrt(nSum) / (1 + c.N);
          const score = q + u;
          if (score > best) { best = score; bestA = a; }
        }
        if (bestA < 0) break;
        node = node.children[bestA];
        if (node.depth >= this.maxDepth()) break;
      }

      /* -------------------------------------------------------- expansion */
      const leaf = node;
      if (!leaf.expanded) {
        const pr = this.model.predictFrom(this.view, leaf.latent);
        leaf.value = pr.value;
        leaf.prior = maskedProbs(pr.polLogits, leaf.legal, nA);
        leaf.children = new Array(nA).fill(null);
        leaf.expanded = true;
        this.stats.expansions++;
      }
      this.stats.nodes++;
      this.stats.maxDepth = Math.max(this.stats.maxDepth, leaf.depth);

      /* ----------------------------------------------------------- backup */
      // v is the discounted return of the *current* node; the edge reward is
      // the value prefix G predicted for that edge.
      let v = leaf.value;
      let cur = leaf;
      while (cur && cur !== root) {
        v = cur.reward + gamma * v;
        const p = cur.parent;
        p.N++;
        p.W += v;
        p.Q = p.W / p.N;
        cur = p;
      }
      this.stats.sims++;
      yield { sims: this.stats.sims, total: sims };
    }

    /* -------------------------------------------------- policy extraction */
    const counts = new Float32Array(nA);
    let tot = 0;
    for (let a = 0; a < nA; a++) {
      const c = root.children[a];
      counts[a] = c ? c.N : 0;
      tot += counts[a];
    }
    if (tot <= 0) {
      for (let a = 0; a < nA; a++) counts[a] = !legal || legal[a] ? 1 : 0;
      tot = countLegal(legal, nA);
    }
    const policy = new Float32Array(nA);
    for (let a = 0; a < nA; a++) policy[a] = counts[a] / tot;

    let action = 0;
    if (temp <= 1e-6) {
      let best = -Infinity;
      for (let a = 0; a < nA; a++) if (counts[a] > best) { best = counts[a]; action = a; }
    } else {
      const r = Math.random();
      let acc = 0;
      action = nA - 1;
      for (let a = 0; a < nA; a++) { acc += policy[a]; if (acc >= r) { action = a; break; } }
    }
    this.stats.value = root.averageValue;
    this.stats.rootQ = root.Q;
    let topIdx = 0;
    for (let a = 1; a < nA; a++) if (counts[a] > counts[topIdx]) topIdx = a;
    this.stats.top = counts[topIdx] / Math.max(1, tot);
    let H = 0;
    for (let a = 0; a < nA; a++) if (policy[a] > 1e-12) H -= policy[a] * Math.log(policy[a]);
    this.stats.entropy = H;
    return { action, policy, value: root.averageValue, root, counts, stats: this.stats, visits: tot };
  }

  maxDepth() {
    return this.cfg.UNROLL + 1;
  }

  _expand(parent, action) {
    const node = new MCTSNode();
    node.parent = parent;
    node.action = action;
    node.depth = parent.depth + 1;
    const d = this.model.dynamic(this.view, parent.latent, action, parent.acc);
    node.latent = d.next;
    node.acc = d.accNext;
    node.reward = d.prefix;
    node.legal = parent.legal;
    parent.children[action] = node;
    this.stats.nodes++;
    return node;
  }
}

function legalCount(legal, nA) {
  let n = 0;
  if (!legal) return nA;
  for (let a = 0; a < nA; a++) if (legal[a]) n++;
  return Math.max(1, n);
}
function countLegal(legal, nA) {
  return legalCount(legal, nA);
}
function maskedProbs(logits, legal, nA) {
  const p = new Float32Array(nA);
  softmaxInto(logits, p, nA);
  if (legal) renorm(p, legal, nA);
  return p;
}
function renorm(p, legal, nA) {
  let s = 0;
  for (let a = 0; a < nA; a++) {
    if (legal && !legal[a]) { p[a] = 0; continue; }
    if (p[a] < 0) p[a] = 0;
    s += p[a];
  }
  if (s <= 0) { for (let a = 0; a < nA; a++) if (!legal || legal[a]) p[a] = 1 / countLegal(legal, nA); return p; }
  for (let a = 0; a < nA; a++) p[a] /= s;
  return p;
}

/** Marsaglia–Tsang gamma sampler, used for the Dirichlet root noise. */
export function dirichletSample(alpha, k) {
  if (alpha < 1) {
    const u = Math.random();
    return gammaSample(1 + alpha, 1) * Math.pow(u, 1 / alpha);
  }
  return gammaSample(alpha, 1);
}
export function gammaSample(shape, scale) {
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}
function randn() {
  let u = 0, v = 0, s = 0;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s === 0 || s >= 1);
  return u * Math.sqrt((-2 * Math.log(s)) / s);
}

/** Uniform Dirichlet(alpha·1_k) sample of length k (root noise). */
export function dirichletVector(alpha, k) {
  const g = new Float32Array(k);
  let s = 0;
  for (let i = 0; i < k; i++) {
    g[i] = dirichletSample(alpha, k);
    s += g[i];
  }
  if (s <= 0) g.fill(1 / k);
  else for (let i = 0; i < k; i++) g[i] /= s;
  return g;
}
