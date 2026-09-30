/**
 * ai/brain.js
 * ---------------------------------------------------------------------------
 * One EfficientZero agent: its own networks (H / G / P,V,R), its own PUCT
 * search, its own game recorder, plus the training loop that ties them
 * together. Seven of these run for the builders (optionally sharing one replay
 * pool, "league style" experience sharing — see `pool`) and one for the Locust.
 *
 *   decide(ctx)   → encode observation, run MCTS in latent space, blend the
 *                   expert prior by `assist · decay`, pick an action, return the
 *                   improved policy + root value that become the targets.
 *   remember(…)   → push (o, a, u, π, z, legal) into the trajectory; rewards are
 *                   *accumulated between decisions*, exactly like EfficientZero's
 *                   action-repeat frame skipping on Atari.
 *   train()       → sample a prioritised minibatch, optionally reanalyse it,
 *                   unroll the dynamics 5 steps and do one AdamW update.
 * ---------------------------------------------------------------------------
 */

import { EZ, PHASE, SIMS_BY_PHASE } from '../shared/rules.js';
import { EZModel } from './efficientzero.js';
import { Search } from './mcts.js';
import { ReplayPool, Trajectory } from './buffer.js';
import { builderPrior, locustPrior } from './prior.js';
import { f32ToBase64, base64ToF32 } from './nn.js';

export class Brain {
  constructor(o) {
    this.id = o.id;
    this.name = o.name || o.id;
    this.kind = o.kind; // 'builder' | 'locust'
    this.codec = o.codec;
    this.nActions = o.nActions;
    this.cfg = { ...EZ, ...(o.cfg || {}) };
    this.model = new EZModel(this.codec.dim, this.nActions, this.cfg);
    this.search = new Search(this.model, this.cfg);
    this.pool = o.pool || new ReplayPool(this.cfg, o.poolOpts);
    this.assist = o.assist ?? 0.9;
    this.assistDecay = o.assistDecay ?? 900; // optimizer steps for the prior to fade out
    this.assistFloor = o.assistFloor ?? 0;   // never drop below this much guidance
    this.learning = o.learning !== false;
    this.stepCounter = 0;
    this.gameCount = 0;
    this.traj = new Trajectory({ step: 0 });
    this.lastStats = null;
    this.pending = { reward: 0, legal: null, policy: null, value: 0, action: -1, obs: null };
    this.history = [];
    this.decisions = 0;
    this.priorSteps = 0;   // how many decisions the expert prior actually drove
    this.totalReward = 0;
    this.episodeReward = 0;
    this.lastAction = 0;
    this.searchTime = 0;
    this.trainTime = 0;
    this.noise = o.noise ?? 1;
  }

  get obsDim() {
    return this.codec.dim;
  }

  effectiveAssist() {
    if (!this.assist) return 0;
    // The expert prior fades as the network learns, but never fully: it is also
    // the thing that keeps a half-trained agent from looking incompetent while
    // it imitates. `assistFloor` is that permanent minimum.
    const fade = 1 - this.model.trainSteps / this.assistDecay;
    return this.assist * Math.max(this.assistFloor, Math.max(0, fade));
  }

  temperature() {
    // EfficientZero's schedule: τ = 1 while the buffer is being filled, then a
    // mixture of sampled (τ = 0.5) and greedy play so targets stay sharp.
    if (this.pool.numSteps < this.cfg.MIN_BUFFER_FOR_TRAINING) return 1;
    if (this.stepCounter < 260) return 1;
    if (this.gameCount % 4 === 0) return 0.5;
    return 0;
  }

  simsFor(phase) {
    const base = SIMS_BY_PHASE[phase] ?? this.cfg.SIMS;
    return Math.max(4, Math.round(base * (this.cfg.SIM_SCALE ?? 1)));
  }

  encode(ctx) {
    return this.codec.encode(ctx);
  }

  /**
   * Run the search and choose the action.
   * @param {Float32Array} obs dense observation
   * @param {Uint8Array} legal mask
   * @param {object} ctx used for the optional expert prior
   */
  decide(obs, legal, ctx = {}, phase = PHASE.BUILD) {
    const t0 = nowMs();
    const sims = this.simsFor(phase);
    const temp = this.temperature();
    const assist = this.effectiveAssist();
    const nA = this.nActions;
    let r = this.search.run(obs, legal, { sims, temperature: temp, noise: this.noise > 0 });
    let policy = r.policy;
    let usedPrior = false;
    if (assist > 0) {
      // The prior *acts* with probability `assist` (and its distribution becomes
      // the training target for that step, DAgger-style). Blending the two
      // distributions alone is not enough: an untrained value head concentrates
      // all MCTS visits on one arbitrary action and the argmax swamps the prior.
      const pr = this.kind === 'locust' ? locustPrior(ctx) : builderPrior(ctx);
      if (Math.random() < assist) {
        usedPrior = true;
        let action = 0, best = -Infinity;
        // 20 % of prior-driven steps are sampled from the prior, the rest take
        // its argmax: enough exploration to gather diverse trajectories without
        // turning the expert into noise.
        if (temp > 1e-6 && Math.random() < 0.2) {
          const u = Math.random();
          let acc = 0;
          action = nA - 1;
          for (let a = 0; a < nA; a++) { if (legal && !legal[a]) continue; acc += pr[a]; if (acc >= u) { action = a; break; } }
        } else {
          for (let a = 0; a < nA; a++) { if (legal && !legal[a]) continue; if (pr[a] > best) { best = pr[a]; action = a; } }
        }
        r = { ...r, action };
        policy = pr;
      } else if (temp > 1e-6) {
        const u = Math.random();
        let acc = 0, pick = r.action;
        for (let a = 0; a < nA; a++) { acc += policy[a]; if (acc >= u) { pick = a; break; } }
        r = { ...r, action: pick };
      }
    } else if (temp > 1e-6) {
      const u = Math.random();
      let acc = 0, pick = r.action;
      for (let a = 0; a < nA; a++) { acc += policy[a]; if (acc >= u) { pick = a; break; } }
      r = { ...r, action: pick };
    }
    this.decisions++;
    if (usedPrior) this.priorSteps++;
    this.lastAction = r.action;
    this.searchTime = nowMs() - t0;
    this.pending = { legal, policy, value: r.value, action: r.action, obs };
    // packed record for the replay buffer, built from the same state
    this._obsRec = this.packFrom(ctx);
    return {
      action: r.action,
      policy,
      value: r.value,
      sims: r.stats.sims,
      nodes: r.stats.nodes,
      top: r.stats.top,
      entropy: r.stats.entropy,
      assist,
      usedPrior,
      temp,
      time: this.searchTime,
    };
  }

  packFrom(ctx) {
    return this.codec.pack({ ...ctx, scalars: ctx.scalars });
  }

  /**
   * Close the previous transition (with the reward accumulated since the last
   * decision) and open the next one.
   */
  remember(reward, phase, done = false) {
    if (!this.pending.obs) return;
    const acc = reward + (this.pending.reward || 0);
    if (done) {
      // terminal step: value target 0 (bootstrap masked)
      this.traj.push({
        obs: this._obsRec,
        action: this.pending.action,
        reward: acc,
        policy: this.pending.policy,
        value: this.pending.value,
        legal: this.pending.legal,
        t: this.stepCounter,
        done: true,
      });
      this.pending.reward = 0;
      return;
    }
    this.pending.reward = acc;
    this.traj.push({
      obs: this._obsRec,
      action: this.pending.action,
      reward: 0,
      policy: this.pending.policy,
      value: this.pending.value,
      legal: this.pending.legal,
      t: this.stepCounter,
      done: false,
    });
    this.stepCounter++;
    this.totalReward += acc;
    this.episodeReward += acc;
  }

  /** reward that will be folded into the *next* transition */
  accumulate(reward) {
    if (this.pending.obs) this.pending.reward = (this.pending.reward || 0) + reward;
  }

  endGame(meta = {}) {
    if (this.traj.length >= 4) {
      this.traj.meta = { ...this.traj.meta, ...meta, step: this.stepCounter };
      // last transition is terminal: value target 0
      const last = this.traj.length - 1;
      this.traj.done[last] = true;
      this.pool.add(this.traj);
      this.gameCount++;
      this.history.push({
        cycle: meta.cycle,
        return: this.traj.returns(),
        reward: this.episodeReward,
        steps: this.traj.length,
        stats: { ...this.traj.stats },
      });
      if (this.history.length > 60) this.history.shift();
    }
    this.episodeReward = 0;
    this.traj = new Trajectory({ step: this.stepCounter });
    this.pending = { reward: 0, legal: null, policy: null, value: 0, action: -1, obs: null };
  }

  /* ------------------------------------------------------------ training */

  train(miniBatch = this.cfg.MINI_BATCH, opts = {}) {
    if (!this.learning) return null;
    if (this.pool.numSteps < this.cfg.MIN_BUFFER_FOR_TRAINING) return null;
    const t0 = nowMs();
    const items = this.pool.sampleIndices(Math.max(1, miniBatch), this.stepCounter);
    if (items.length === 0) return null;

    if (opts.reanalyze !== false && this.cfg.REANALYZE !== false) {
      const legalAll = items.filter((it) => it.ageSteps > this.cfg.SVE_FRESH_STEPS);
      if (legalAll.length) {
        this.pool.reanalyze(this.search, this.codec, this.model, legalAll.slice(0, 4),
          this.cfg.SIM_REANALYZE, { noise: false });
      }
    }

    const seqs = items.map((it) => this._buildSequence(it));
    const stats = this.model.trainSequences(seqs.filter(Boolean));
    for (let i = 0; i < items.length; i++) {
      items[i].td = seqs[i] ? seqs[i].td : 0;
    }
    this.pool.updatePriorities(items);
    this.lastStats = stats;
    this.trainTime = nowMs() - t0;
    this.trainCount = (this.trainCount || 0) + 1;
    return stats;
  }

  _buildSequence(item) {
    const g = item.game;
    const codec = this.codec;
    const L = Math.max(1, Math.min(item.depth, this.cfg.UNROLL));
    const obs = [];
    const acts = [];
    const rewards = [];
    const policies = [];
    const prefixTargets = [];
    const valueTargets = [];
    const gamma = this.cfg.GAMMA;
    const gp = this.cfg.GAMMA_PREFIX;

    let prefix = 0;
    for (let k = 0; k <= L; k++) {
      const idx = item.start + k;
      if (idx >= g.length) break;
      obs.push(codec.decode(g.obs[idx]));
      if (k < L) {
        acts.push(g.acts[idx]);
        const u = g.rewards[idx] ?? 0;
        rewards.push(u);
        policies.push(g.policies[idx]);
        prefix += Math.pow(gp, k) * u;
        prefixTargets.push(prefix);
        if (idx >= g.length - 1 || g.done[idx]) {
          valueTargets.push(rewards.reduce((a, b, j) => a + b * Math.pow(gamma, j), 0));
        } else if (!item.fresh && g.values[idx] !== undefined) {
          // EZ-V2 search-based value estimation for stale samples
          valueTargets.push(g.values[idx]);
        } else {
          let z = 0;
          let disc = 1;
          let n = 0;
          for (let j = idx; j < g.length && n < Math.max(1, L - k); j++, n++) {
            z += disc * (g.rewards[j] ?? 0);
            disc *= gamma;
            if (g.done[j]) break;
          }
          const bootIdx = Math.min(g.length - 1, idx + n);
          const boot = g.done[bootIdx] ? 0 : (g.values[bootIdx] ?? 0);
          z += disc * boot;
          valueTargets.push(z);
        }
      }
    }
    if (obs.length < 2 || acts.length < 1) return null;
    return {
      obs, acts, rewards, policies, prefixTargets, valueTargets,
      depth: L, weight: item.weight, tdSteps: Math.max(1, L), item,
    };
  }

  /* ----------------------------------------------------------- inspection */

  stats() {
    const m = this.model;
    return {
      id: this.id,
      name: this.name,
      kind: this.kind,
      obsDim: this.obsDim,
      nActions: this.nActions,
      params: m.nParams,
      trainSteps: m.trainSteps,
      decisions: this.decisions,
      bufferGames: this.pool.games.length,
      bufferSteps: this.pool.numSteps,
      beta: this.pool.currentBeta(),
      priority: this.pool.maxPriority,
      last: this.lastStats,
      value: this.pending.value,
      top: this.search.stats.top,
      sims: this.search.stats.sims,
      nodes: this.search.stats.nodes,
      searchMs: +this.searchTime.toFixed(2),
      trainMs: +this.trainTime.toFixed(2),
      assist: +this.effectiveAssist().toFixed(3),
      priorShare: this.decisions ? +(this.priorSteps / this.decisions).toFixed(3) : null,
      episodeReward: +this.episodeReward.toFixed(2),
      totalReward: +this.totalReward.toFixed(1),
      trajLen: this.traj.length,
      temp: this.temperature(),
      learning: this.learning,
      history: this.history.slice(-12),
      health: m.health(),
    };
  }

  checkpoint() {
    const c = this.model.checkpoint();
    return {
      id: this.id,
      kind: this.kind,
      obsDim: this.obsDim,
      nActions: this.nActions,
      stepCounter: this.stepCounter,
      gameCount: this.gameCount,
      decisions: this.decisions,
      assist: this.assist,
      cfg: c.meta.cfg,
      trainSteps: c.trainSteps,
      b64: f32ToBase64(c.pack),   // see ai/nn.js: ~15× smaller than Array.from()
    };
  }

  loadCheckpoint(data) {
    if (!data) return false;
    if (data.obsDim !== this.obsDim || data.nActions !== this.nActions) return false;
    let pack = null;
    if (typeof data.b64 === 'string' && data.b64) pack = base64ToF32(data.b64);
    else if (data.pack) pack = data.pack instanceof Float32Array ? data.pack : Float32Array.from(data.pack);
    if (!pack) return false;
    const ok = this.model.loadPack(pack);
    if (!ok) return false;
    this.stepCounter = data.stepCounter || 0;
    this.gameCount = data.gameCount || 0;
    this.decisions = data.decisions || 0;
    return true;
  }
}

/**
 * The learning population: 7 builder brains (one network per bot, shared
 * replay pool) and one Locust brain with its own pool.
 */
export class BrainLeague {
  constructor(o) {
    this.cfg = { ...EZ, ...(o.cfg || {}) };
    this.sharedPool = o.sharedPool !== false;
    this.builderPool = this.sharedPool ? new ReplayPool(this.cfg, o.poolOpts) : null;
    this.locustPool = new ReplayPool(this.cfg, o.locustPoolOpts || { games: 80, steps: 260 });
    this.assist = o.assist ?? 0.9;
    this.assistFloor = o.assistFloor ?? 0.35;
    this.learning = o.learning !== false;
    this.brains = [];
    this.byId = new Map();
  }
  makeBuilder(id, name, codec, nActions) {
    const b = new Brain({
      id, name, kind: 'builder', codec, nActions,
      pool: this.builderPool || undefined,
      cfg: this.cfg, assist: this.assist, assistFloor: this.assistFloor, learning: this.learning,
    });
    this.brains.push(b);
    this.byId.set(id, b);
    return b;
  }
  /**
   * The Locust keeps ONE brain for the whole session: it is re-spawned every
   * cycle, but its networks, its replay pool and its optimiser state persist,
   * so hunt N+1 starts from everything it learned in hunt N.
   */
  ensureLocustBrain(codec, nActions, id = 'locust', name = 'The Locust') {
    if (this.locustBrain) return this.locustBrain;
    this.locustBrain = this.makeLocust(id, name, codec, nActions);
    return this.locustBrain;
  }
  makeLocust(id, name, codec, nActions) {
    const b = new Brain({
      id, name, kind: 'locust', codec, nActions,
      pool: this.locustPool, cfg: this.cfg,
      // The predator must be dangerous from the first cycle, so it starts with
      // near-full guidance and keeps an 85 % floor while it learns to refine
      // its own hunting policy on top of the expert's.
      assist: Math.max(0.95, this.assist), assistFloor: 0.85, assistDecay: 500,
      learning: this.learning, noise: 1.2,
    });
    this.brains.push(b);
    this.byId.set(id, b);
    return b;
  }
  stats() {
    return this.brains.map((b) => b.stats());
  }
  trainAll(mini) {
    const out = {};
    for (const b of this.brains) out[b.id] = b.train(mini);
    return out;
  }
  checkpoints() {
    const o = {};
    for (const b of this.brains) o[b.id] = b.checkpoint();
    return o;
  }
  loadCheckpoints(data) {
    if (!data) return 0;
    let n = 0;
    for (const b of this.brains) if (data[b.id] && b.loadCheckpoint(data[b.id])) n++;
    return n;
  }
}

function nowMs() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}
