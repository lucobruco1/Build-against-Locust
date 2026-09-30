/**
 * Build to Survive the Locust — the match, drivable from anywhere.
 *
 * `server/server.js` owns a `Game`: it steps `Match` at a fixed 30 Hz, folds the
 * Locust's best hunt numbers of the night into a record, publishes a 20 Hz
 * `state` message with the packed events of the last 50 ms, and keeps learned
 * checkpoints on disk. `LocalGame` here is the *same four responsibilities* with
 * the transport replaced by a callback and the disk replaced by an injected
 * key-value store — so a browser tab with nothing behind it plays the identical
 * game (same rules, same EfficientZero brains, same MCTS), and `node --test`
 * covers both paths through one implementation.
 *
 * Deliberately free of the DOM *and* of Node APIs: this file is imported by the
 * page, by the tests, and (for the shared helpers) by the server. Anything that
 * needs `fs`, `http` or `document` belongs in server/ or public/js/.
 */

import { Match } from './match.js';
import { encodeState, packEvent, packWorld } from './net.js';
import { PHASE, TIMING } from '../shared/rules.js';

/** how often a viewer gets a full state snapshot: 20 Hz, same as the server */
export const BROADCAST_MS = 50;

/** the localStorage key the in-browser build uses for learned weights */
export const LOCAL_BRAINS_KEY = 'gbtl:brains';
export const LOCAL_CYCLES_KEY = 'gbtl:cycles';

/* ------------------------------------------------------ shared match driving */

/** Per-night hunt bests. The Locust object is despawned before anyone asks. */
export function freshHunt() {
  return { kills: 0, smashed: 0, grabs: 0, playerKills: 0 };
}

/** Take the max of every stat the live Locust has reached so far. */
export function foldHunt(hunt, locust) {
  if (!locust || !hunt) return hunt;
  const st = locust.stats;
  hunt.kills = Math.max(hunt.kills, st.kills);
  hunt.smashed = Math.max(hunt.smashed, st.smashed);
  hunt.grabs = Math.max(hunt.grabs, st.grabs);
  hunt.playerKills = Math.max(hunt.playerKills, st.playerKills || 0);
  return hunt;
}

/** Everything worth remembering about a finished cycle (caller decides where to put it). */
export function cycleSummary(match, hunt = freshHunt()) {
  const m = match;
  const wall = m.builders.map((b) => b.security?.wall ?? 0);
  return {
    cycle: m.cycle,
    kills: hunt.kills, smashed: hunt.smashed, grabs: hunt.grabs, playerKills: hunt.playerKills,
    deaths: m.builders.reduce((a, b) => a + b.stats.deaths, 0),
    escapes: m.builders.reduce((a, b) => a + b.stats.escapes, 0),
    placed: m.builders.reduce((a, b) => a + b.stats.placed, 0),
    broken: m.builders.reduce((a, b) => a + b.stats.broken, 0),
    wallMean: wall.reduce((a, b) => a + b, 0) / (wall.length || 1),
    bestScore: Math.max(0, ...m.builders.map((b) => b.score)),
    secureSeconds: Math.round(m.builders.reduce((a, b) => a + (b.security?.sealed ? 90 : 0), 0)),
    updates: m.league.brains.reduce((a, b) => a + b.model.trainSteps, 0),
    builders: m.builders.map((b) => ({
      name: b.name, wall: b.security?.wall ?? 0, score: Math.round(b.score),
      deaths: b.stats.deaths, escaped: b.stats.escapes, placed: b.stats.placed,
    })),
  };
}

/** One human input message → authoritative side effects. Same shape from WS or in-tab. */
export function applyMatchInput(match, msg) {
  if (!msg || typeof msg !== 'object') return false;
  const m = match;
  switch (msg.t) {
    case 'input': {
      if (msg.look) m.setPlayerLook(msg.look[0], msg.look[1]);
      if (msg.move) m.setPlayerMove(msg.move[0], msg.move[1], msg.sprint);
      if (msg.jump) m.playerJump();
      if (msg.place) m.playerPlace();
      if (msg['break']) m.playerBreak();
      if (msg.select !== undefined) m.playerSelect(msg.select);
      if (msg.cycle !== undefined) m.playerCycle(msg.cycle);
      return true;
    }
    case 'act': m.playerAct?.(msg.action | 0); return true;
    case 'say': m.say(String(msg.text || '').slice(0, 120), 'player'); return true;
    default: return false;
  }
}

/**
 * Live-config changes from the menu. `driver` is anything with `.match` plus
 * `timeScale` / `paused` fields (server `Game` or `LocalGame`), which is why this
 * lives here instead of in one of the two: the menu must not behave differently
 * depending on who is simulating.
 */
export function applyMatchConfig(driver, body) {
  if (!driver || !body) return null;
  const m = driver.match;
  if (body.timeScale !== undefined) driver.timeScale = Math.max(0.05, Math.min(20, Number(body.timeScale) || 1));
  if (body.paused !== undefined) driver.paused = !!body.paused;
  if (body.seed !== undefined) {
    const s = Number(body.seed);
    if (Number.isFinite(s)) { m.seed = s | 0; driver.config0 = { ...driver.config0, seed: s | 0 }; }
  }
  if (body.learn !== undefined) {
    const on = !!body.learn;
    m.learn = on;
    for (const b of m.league.brains) b.learning = on;
  }
  if (body.assist !== undefined) {
    const a = Math.max(0, Math.min(1, Number(body.assist)));
    m.assist = a;
    for (const b of m.league.brains) if (b.kind !== 'locust') b.assist = a;
  }
  if (body.sims !== undefined) {
    const s = Math.max(0.1, Math.min(8, Number(body.sims) || 1));
    m.simScale = s;
    for (const b of m.league.brains) b.cfg.SIM_SCALE = s;
  }
  return driver.config();
}

/* ------------------------------------------------------------------ driver */

/**
 * An authoritative `Match` with no server: it produces exactly the messages the
 * client expects and consumes exactly the ones it sends.
 *
 *   const game = new LocalGame({ config, onMessage: (m) => ws.send(m) })
 *   game.send({ t: 'join', name: 'You' })
 *   game.start()            // or drive it by hand with game.pump(dtMs)
 *
 * `storage` is any `{ getItem, setItem }` (the browser hands it
 * `window.localStorage`; tests hand it a Map wrapper) and may be null, in which
 * case nothing is persisted and save/load simply report so.
 */
export class LocalGame {
  constructor({ config = {}, onMessage = () => {}, storage = null, now = () => Date.now() } = {}) {
    this.config0 = {
      seed: 1337,
      learn: false,          // training in a tab is opt-in: it competes with the frame budget
      simScale: 0.6,
      assist: 0.9,
      flat: false,
      timeScale: 1,          // only scales this driver's clock; Match is stepped in real ms
      broadcastMs: BROADCAST_MS,
      match: null,
      ...config,
    };
    this.onMessage = onMessage;
    this.storage = storage;
    this.now = now;
    this.match = this.createMatch();
    this.paused = false;
    this.timeScale = this.config0.timeScale;
    this.hunt = freshHunt();
    this.lastTick = this.now();
    this.lastBroadcast = 0;
    this.frame = 0;
    this.clients = 1;                 // interface parity with the hub: there is always one viewer
    this.timer = null;
    this.lastSavedSteps = -1;
    this.stats = { ticks: 0, broadcasts: 0, saves: 0, loads: 0 };
  }

  /**
   * `config.match` is a passthrough for any `Match` knob this driver doesn't
   * name (phaseScale, maxCycles, …) so a test or a menu can reach straight
   * through without the driver having to mirror the whole Match signature.
   */
  createMatch(over = {}) {
    const c = this.config0;
    return new Match({
      seed: c.seed, learn: c.learn, simScale: c.simScale, assist: c.assist,
      flat: c.flat, timeScale: 1, ...(c.match || {}), ...over,
    });
  }

  config() {
    const m = this.match;
    return {
      timeScale: this.timeScale, paused: this.paused, learn: m.learn,
      assist: m.assist, sims: m.simScale, seed: m.seed,
      flat: !!this.config0.flat, phase: m.phase, cycle: m.cycle, local: true,
    };
  }

  /* ------------------------------------------------------------ lifecycle */

  /** Self-paced: the same 30 Hz the server uses. */
  start() {
    if (this.timer) return this;
    this.lastTick = this.now();
    this.timer = setInterval(() => this.tick(), TIMING.TICK_MS);
    this.timer.unref?.();               // a test that forgot to stop() must not hang the process
    return this;
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    return this;
  }

  /** Advance by real milliseconds. `tick()` is the self-paced version of this. */
  tick() { return this.pump(Math.min(400, this.now() - this.lastTick)); }

  pump(dtRealMs) {
    if (this.paused) return null;
    this.lastTick = this.now();
    this.match.update(dtRealMs * this.timeScale);
    this.frame++;
    this.stats.ticks++;
    foldHunt(this.hunt, this.match.locust);
    const now = this.now();
    if (now - this.lastBroadcast >= (this.config0.broadcastMs ?? BROADCAST_MS)) {
      this.lastBroadcast = now;
      return this.broadcast();
    }
    return null;
  }

  /** Everything the client is missing since the last state: its own world, first. */
  hello(name = 'You') {
    this.name = String(name || 'You').slice(0, 20);
    const msg = {
      t: 'hello', config: this.config(), name: this.name,
      rules: { phase: PHASE, tickMs: TIMING.TICK_MS },
      world: packWorld(this.match.world), state: encodeState(this.match),
    };
    this.post(msg);
    return msg;
  }

  broadcast() {
    // Drain first, always: `match.events` is a queue, and dropping it silently is
    // how the server used to leak a few thousand objects a minute.
    const evs = this.match.drainEvents().map(packEvent).filter(Boolean);
    const state = encodeState(this.match);
    if (evs.length) state.deltas = evs;
    this.stats.broadcasts++;
    this.post({ t: 'state', ...state });
    if (evs.some((e) => e.t === 'locustDespawn')) this.recordCycle();
    return state;
  }

  post(msg) { try { this.onMessage(msg); } catch (err) { console.error('[local] onMessage threw', err); } }

  /** Client → engine. Mirrors the WS switch in server/server.js message for message. */
  send(msg) {
    if (!msg || typeof msg !== 'object') return false;
    switch (msg.t) {
      case 'join': {
        // The menu's seed / flat-world choices need a new Match, and it has to
        // happen before the world payload goes out or the client renders one
        // terrain and plays on another.
        const cfg = msg.config || {};
        const rebuild = (cfg.flat !== undefined && !!cfg.flat !== !!this.config0.flat)
          || (cfg.seed !== undefined && (cfg.seed | 0) !== (this.config0.seed | 0));
        if (rebuild) {
          this.config0 = { ...this.config0, flat: !!cfg.flat, seed: cfg.seed | 0 };
          this.match = this.createMatch();
          this.hunt = freshHunt();
        }
        this.hello(msg.name);
        this.post({ t: 'joined', config: this.config(), world: packWorld(this.match.world) });
        if (msg.config) applyMatchConfig(this, msg.config);
        return true;
      }
      case 'ping':
        this.post({ t: 'pong', at: msg.at });
        return true;
      case 'config':
        this.post({ t: 'config', config: applyMatchConfig(this, msg) });
        return true;
      case 'reset':
        this.reset();
        this.post({ t: 'reset', world: packWorld(this.match.world), state: encodeState(this.match) });
        return true;
      case 'save': {
        const r = this.saveBrains();
        this.post({ t: 'saved', ...r });
        return true;
      }
      case 'load': {
        const r = this.loadBrains();
        this.post({ t: 'loaded', ...r });
        return true;
      }
      case 'summary':
        this.post({ t: 'summary', cycles: this.readCycles(), count: this.readCycles().length });
        return true;
      default:
        return applyMatchInput(this.match, msg);
    }
  }

  reset() {
    const learn = this.match.learn;
    const league = this.match.league;          // a reset is not amnesia
    this.match = this.createMatch({ learn, league });
    this.hunt = freshHunt();
    this.lastTick = this.now();
    this.lastBroadcast = 0;
  }

  /* ---------------------------------------------------- persistence (no fs) */

  saveBrains() {
    if (!this.storage) return { ok: false, error: 'no storage in this context' };
    const steps = this.match.league.brains.reduce((a, b) => a + b.model.trainSteps, 0);
    if (steps === this.lastSavedSteps) return { ok: true, skipped: true, loaded: 0, steps };
    try {
      this.storage.setItem(LOCAL_BRAINS_KEY, JSON.stringify({
        savedAt: new Date().toISOString(),
        meta: { cycle: this.match.cycle, ticks: this.match.tickCount, optimizerSteps: steps, config: this.config() },
        checkpoints: this.match.league.checkpoints(),
      }));
      this.lastSavedSteps = steps;
      this.stats.saves++;
      return { ok: true, steps, brains: this.match.league.brains.length };
    } catch (err) {
      // a big tab quota hit is a real possibility with nine networks in there
      return { ok: false, error: String(err?.message || err) };
    }
  }

  loadBrains() {
    if (!this.storage) return { ok: false, error: 'no storage in this context', loaded: 0 };
    const raw = this.storage.getItem(LOCAL_BRAINS_KEY);
    if (!raw) return { ok: true, loaded: 0, error: null };
    try {
      const data = JSON.parse(raw);
      const n = this.match.league.loadCheckpoints(data.checkpoints);
      this.stats.loads++;
      return { ok: true, loaded: n, savedAt: data.savedAt || null };
    } catch (err) {
      return { ok: false, loaded: 0, error: String(err?.message || err) };
    }
  }

  /** Keep the last 40 cycles in the same store, so the tally screen survives a reload. */
  recordCycle() {
    if (!this.storage) return null;
    const rec = cycleSummary(this.match, this.hunt);
    try {
      const all = this.readCycles();
      all.push(rec);
      this.storage.setItem(LOCAL_CYCLES_KEY, JSON.stringify(all.slice(-40)));
      this.hunt = freshHunt();
    } catch { /* a full store must never break the game */ }
    return rec;
  }

  readCycles() {
    if (!this.storage) return [];
    try { return JSON.parse(this.storage.getItem(LOCAL_CYCLES_KEY) || '[]') || []; } catch { return []; }
  }

  /** Autosave hook, same skip-if-nothing-was-learned rule as the server. */
  autosave() { return this.saveBrains(); }

  snapshot() {
    return {
      mode: 'local', phase: this.match.phase, cycle: this.match.cycle,
      paused: this.paused, timeScale: this.timeScale, frames: this.frame,
      ...this.stats,
    };
  }
}
