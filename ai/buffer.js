/**
 * ai/buffer.js
 * ---------------------------------------------------------------------------
 * Prioritised game-sequence replay, i.e. EfficientZero's memory:
 *
 *   • a `Trajectory` is one game (a whole build+hunt cycle for one agent):
 *     packed observations, actions, real rewards, the MCTS improved policy and
 *     the root value recorded at every decision;
 *   • the pool samples *trajectories* with probability ∝ priority^α where
 *     priority is the running max |value-prefix error| (PER, Schaul et al.);
 *     inside a trajectory the sequence start is uniform over the valid window;
 *   • importance-sampling weights β = (N·P(i))^(-β) / max, annealed from the
 *     configured value to 1 (the EZ/LightZero schedule);
 *   • `staleAge` marks samples older than T2 → the trainer answers with
 *     search-based value targets (SVE) instead of TD targets, and reanalysis
 *     refreshes π/z on them (EfficientZero V2, §4.3).
 * ---------------------------------------------------------------------------
 */

import { EZ } from '../shared/rules.js';

export class Trajectory {
  constructor(meta = {}) {
    this.meta = meta;            // {agent, kind, phaseAt, cycle, ...}
    this.obs = [];               // packed records (codec.pack output)
    this.acts = [];
    this.rewards = [];
    this.policies = [];
    this.values = [];            // root value of the search at that step
    this.legal = [];
    this.done = [];
    this.steps = [];             // ms timestamps, for staleness/decay features
    this.priority = 1;
    this.bornStep = meta.step ?? 0;
    this.length = 0;
    this.stats = { placed: 0, broken: 0, deaths: 0, returns: 0 };
  }
  push(step) {
    this.obs.push(step.obs);
    this.acts.push(step.action);
    this.rewards.push(step.reward);
    this.policies.push(step.policy);
    this.values.push(step.value);
    this.legal.push(step.legal);
    this.done.push(!!step.done);
    this.steps.push(step.t ?? 0);
    this.length++;
  }
  get isEmpty() {
    return this.length === 0;
  }
  returns() {
    let s = 0;
    for (const r of this.rewards) s += r;
    return s;
  }
}

export class ReplayPool {
  constructor(cfg = EZ, opts = {}) {
    this.cfg = cfg;
    this.capacity = opts.games ?? cfg.GAMES_PER_BUFFER;
    this.maxStepsPerGame = opts.steps ?? 320;
    this.games = [];
    this.totalSteps = 0;
    this.updates = 0;
    this.minLen = 12;
    this.alpha = cfg.ALPHA_PRIO;
    this.beta = cfg.BETA_PRIO;
    this.betaSteps = 0;
    this.maxPriority = 1;
    this.dropped = 0;
  }

  add(traj) {
    if (traj.length < this.minLen) return false;
    if (traj.length > this.maxStepsPerGame) {
      // keep the tail: the most recent, most-skilled part of the game
      const cut = traj.length - this.maxStepsPerGame;
      for (const k of ['obs', 'acts', 'rewards', 'policies', 'values', 'legal', 'done', 'steps']) {
        traj[k] = traj[k].slice(cut);
      }
      traj.length = this.maxStepsPerGame;   // (was `traj.maxStepsPerGame`, i.e. undefined — which poisoned totalSteps)
    }
    traj.priority = this.maxPriority;
    this.games.push(traj);
    this.totalSteps += traj.length;
    while (this.games.length > this.capacity) {
      const g = this.games.shift();
      this.totalSteps -= g.length;
      this.dropped++;
    }
    return true;
  }

  get size() {
    return this.games.length;
  }
  get numSteps() {
    return this.totalSteps;
  }

  _weights() {
    const w = new Float32Array(this.games.length);
    let sum = 0;
    for (let i = 0; i < this.games.length; i++) {
      const p = Math.pow(Math.max(1e-6, this.games[i].priority), this.alpha);
      w[i] = p;
      sum += p;
    }
    for (let i = 0; i < this.games.length; i++) w[i] /= sum || 1;
    return w;
  }

  pickGame() {
    const w = this._weights();
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < w.length; i++) {
      acc += w[i];
      if (r <= acc) return i;
    }
    return w.length - 1;
  }

  /**
   * Sample `n` sequence descriptors (game index + start index + unroll depth
   * with EfficientZero's off-policy correction + importance-sampling weight).
   */
  sampleIndices(n, currentStep) {
    const out = [];
    const N = this.games.length;
    if (N === 0) return out;
    const cfg = this.cfg;
    const weights = this._weights();
    const beta = this.currentBeta();
    for (let i = 0; i < n; i++) {
      const gi = this.pickGameFrom(weights);
      const g = this.games[gi];
      if (!g || g.length < 4) continue;
      const maxStart = Math.max(0, g.length - 3);
      const start = Math.floor(Math.random() * (maxStart + 1));
      const depth = offPolicyDepth(cfg, currentStep - g.bornStep, g.length - start);
      const iso = Math.pow(Math.max(1e-9, N * weights[gi]), -beta);
      out.push({
        gi,
        start,
        depth,
        game: g,
        iso,
        ageSteps: currentStep - g.bornStep,
        fresh: currentStep - g.bornStep < cfg.SVE_FRESH_STEPS,
        priorityWeight: weights[gi],
      });
      this.betaSteps++;
    }
    // normalise the IS weights by the batch maximum (standard PER practice)
    let mx = 0;
    for (const it of out) mx = Math.max(mx, it.iso);
    for (const it of out) it.weight = mx > 0 ? Math.min(4, it.iso / mx) : 1;
    return out;
  }

  currentBeta() {
    return Math.min(1, this.beta + (1 - this.beta) * (this.betaSteps / Math.max(1, this.cfg.ANNEAL_BETA_STEPS)));
  }

  pickGameFrom(weights) {
    const w = weights || this._weights();
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < w.length; i++) {
      acc += w[i];
      if (r <= acc) return i;
    }
    return Math.max(0, w.length - 1);
  }

  updatePriorities(items) {
    for (const it of items) {
      if (it.gi === undefined || !this.games[it.gi]) continue;
      const g = this.games[it.gi];
      const p = Math.max(1e-6, Math.abs(it.td ?? 0)) + EZ.NOISE_FLOOR;
      g.priority = Math.max(g.priority * 0.9, Math.pow(p, 1 / this.alpha));
      this.maxPriority = Math.max(this.maxPriority, g.priority);
      if (this.maxPriority > 1e6) {
        for (const q of this.games) q.priority = Math.max(1e-6, q.priority / 1e3);
        this.maxPriority = 1;
      }
    }
  }

  /**
   * Reanalysis (EZ §"reanalyze"): re-run a cheap search on stored states and
   * overwrite π and the value target, so old data stays consistent with the
   * current network. `search` is an MCTS instance bound to the same model.
   */
  reanalyze(search, codec, model, items, sims, opts = {}) {
    let n = 0;
    for (const it of items) {
      const g = it.game;
      const k = it.start;
      if (!g.legal[k]) continue;
      const obs = codec.decode(g.obs[k]);
      const r = search.run(obs, g.legal[k], { sims, temperature: 0, noise: opts.noise ?? false });
      g.policies[k] = r.policy;
      if (r.value !== undefined && r.value !== null) g.values[k] = r.value;
      n++;
      if (opts.max && n >= opts.max) break;
    }
    return n;
  }

  stats() {
    return {
      games: this.games.length,
      steps: this.totalSteps,
      dropped: this.dropped,
      maxPriority: this.maxPriority,
      beta: this.currentBeta(),
    };
  }
}

/**
 * EfficientZero's off-policy correction (paper appendix A.4):
 *   depth = l_max − floor( steps_ago / (τ · total_steps) ), clipped to [1, l_max]
 */
export function offPolicyDepth(cfg, stepsAgo, available) {
  const total = Math.max(1e4, cfg.OFFPOLICY_TAU * 1e5);
  let depth = cfg.OFFPOLICY_MAX_DEPTH;
  if (cfg.OFFPOLICY_TAU > 0) {
    depth = cfg.OFFPOLICY_MAX_DEPTH - Math.floor(stepsAgo / (cfg.OFFPOLICY_TAU * total));
  }
  depth = Math.max(1, Math.min(cfg.OFFPOLICY_MAX_DEPTH, depth));
  return Math.max(1, Math.min(depth, available, cfg.UNROLL));
}
