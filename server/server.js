/**
 * Build to Survive the Locust — game server.
 *
 * Owns the authoritative `Match` (voxel world, 7 EfficientZero bots, the player
 * slot, the Locust), steps it at a fixed 30 Hz, streams state to browsers over a
 * dependency-free WebSocket, serves the client files, and keeps learned
 * checkpoints on disk so the bots get better across restarts.
 *
 *   node server/server.js [--port 3000] [--seed 1337] [--learn 1] [--sims 1]
 *                         [--assist 0.9] [--flat 1] [--timeScale 1] [--noWs]
 */

import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Match } from '../game/match.js';
import { WebSocketHub } from './ws.js';
import { createApi } from './api.js';
import { Store } from './store.js';
import { encodeState, packEvent, packWorld } from '../game/net.js';
import { PHASE, TIMING } from '../shared/rules.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

/* -------------------------------------------------------------- arguments */

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return def;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}
const num = (n, d) => (arg(n, undefined) === undefined ? d : Number(arg(n)));
const bool = (n, d) => (arg(n, undefined) === undefined ? d : !['0', 'false', 'no'].includes(String(arg(n)).toLowerCase()));

const CONFIG = {
  port: num('port', Number(process.env.PORT) || 3000),
  host: String(arg('host', '0.0.0.0')),
  seed: num('seed', 1337),
  learn: bool('learn', true),
  sims: num('sims', 1),
  assist: num('assist', 0.9),
  flat: bool('flat', false),
  timeScale: num('timeScale', 1),
  dataDir: String(arg('data', '.data')),
  staticRoot: ROOT,
  autoSaveS: num('autosave', 60),
};

/* ------------------------------------------------------------- game loop */

class Game {
  constructor(config) {
    this.config0 = config;
    this.store = new Store(path.join(ROOT, config.dataDir));
    this.timeScale = config.timeScale;
    this.paused = false;
    this.match = this.createMatch();
    this.hub = null;
    this.lastTick = Date.now();
    this.lastBroadcast = 0;
    this.cycleSeen = 0;
    this.timer = null;
    this.frame = 0;
    // best hunt numbers of the night, kept here because the Locust object is
    // despawned before the client (or the cycle record) asks for them. It used to
    // be created only at the end of recordCycle(), so the first hunt tick of a
    // fresh server threw on `this.hunt.kills` — i.e. the server died at 90 seconds.
    this.hunt = { kills: 0, smashed: 0, grabs: 0, playerKills: 0 };
    this.lastSavedSteps = -1;
  }

  createMatch(over = {}) {
    const c = this.config0;
    return new Match({
      seed: c.seed,
      learn: c.learn,
      simScale: c.sims,
      assist: c.assist,
      flat: c.flat,
      timeScale: 1,                     // we step it ourselves, in real seconds
      ...over,
    });
  }

  config() {
    return {
      timeScale: this.timeScale, paused: this.paused, learn: this.match.learn,
      assist: this.match.assist, sims: this.match.simScale, seed: this.match.seed,
      flat: !!this.config0.flat, phase: this.match.phase, cycle: this.match.cycle,
    };
  }

  applyConfig(body) {
    if (!body) return;
    if (body.timeScale !== undefined) this.timeScale = Math.max(0.05, Math.min(20, Number(body.timeScale) || 1));
    if (body.paused !== undefined) this.paused = !!body.paused;
    if (body.learn !== undefined) {
      const on = !!body.learn;
      this.match.learn = on;
      for (const b of this.match.league.brains) b.learning = on;
    }
    if (body.assist !== undefined) {
      const a = Math.max(0, Math.min(1, Number(body.assist)));
      this.match.assist = a;
      for (const b of this.match.league.brains) if (b.kind !== 'locust') b.assist = a;
    }
    if (body.sims !== undefined) {
      const s = Math.max(0.1, Math.min(8, Number(body.sims) || 1));
      this.match.simScale = s;
      for (const b of this.match.league.brains) b.cfg.SIM_SCALE = s;
    }
  }

  applyInput(msg) {
    const m = this.match;
    switch (msg.t) {
      case 'input': {
        if (msg.look) m.setPlayerLook(msg.look[0], msg.look[1]);
        if (msg.move) m.setPlayerMove(msg.move[0], msg.move[1], msg.sprint);
        if (msg.jump) m.playerJump();
        if (msg.place) m.playerPlace();
        if (msg['break']) m.playerBreak();
        if (msg.select !== undefined) m.playerSelect(msg.select);
        if (msg.cycle !== undefined) m.playerCycle(msg.cycle);
        break;
      }
      case 'act': m.playerAct?.(msg.action | 0); break;
      case 'say': m.say(String(msg.text || '').slice(0, 120), 'player'); break;
      default: break;
    }
  }

  start() {
    this.timer = setInterval(() => this.tick(), TIMING.TICK_MS);
    this.timer.unref?.();
    if (this.config0.autoSaveS > 0) {
      this.saveTimer = setInterval(() => this.autosave(), this.config0.autoSaveS * 1000);
      this.saveTimer.unref?.();
    }
    return this;
  }

  async restore() {
    const data = await this.store.loadBrains();
    if (!data) return { loaded: 0 };
    const n = this.match.league.loadCheckpoints(data.checkpoints);
    console.log(`[game] restored ${n} learned brain(s) from ${new Date(data.savedAt).toLocaleString()}`);
    return { loaded: n, savedAt: data.savedAt };
  }

  async autosave() {
    try {
      const steps = this.match.league.brains.reduce((a, b) => a + b.model.trainSteps, 0);
      if (steps === this.lastSavedSteps) return;   // nothing learned since the last write
      await this.store.saveBrains(this.match.league.checkpoints(), {
        cycle: this.match.cycle, ticks: this.match.tickCount, optimizerSteps: steps, config: this.config(),
      });
      this.lastSavedSteps = steps;
    } catch (err) { console.error('[game] autosave failed', err.message); }
  }

  tick() {
    if (this.paused) return;
    const now = Date.now();
    const dtReal = Math.min(400, now - this.lastTick);
    this.lastTick = now;
    this.match.update(dtReal * this.timeScale);
    this.frame++;

    // keep the best hunt numbers seen so far; the Locust object is gone by the
    // time the despawn event reaches us
    if (this.match.locust) {
      const st = this.match.locust.stats;
      this.hunt.kills = Math.max(this.hunt.kills, st.kills);
      this.hunt.smashed = Math.max(this.hunt.smashed, st.smashed);
      this.hunt.grabs = Math.max(this.hunt.grabs, st.grabs);
      this.hunt.playerKills = Math.max(this.hunt.playerKills, st.playerKills || 0);
    }

    if (now - this.lastBroadcast >= 50) {      // 20 Hz
      this.lastBroadcast = now;
      this.broadcast();
    }
  }

  broadcast() {
    // The events are drained whether or not somebody is watching: they are a
    // queue, and a server with no clients used to grow it forever (a slow leak of
    // thousands of objects a minute) while never recording a single cycle.
    const evs = this.match.drainEvents().map(packEvent).filter(Boolean);
    if (this.hub && this.hub.clients.size) {
      const state = encodeState(this.match);
      if (evs.length) state.deltas = evs;
      this.hub.send({ t: 'state', ...state });
    }
    if (evs.some((e) => e.t === 'locustDespawn')) this.recordCycle().catch(() => {});
  }

  async recordCycle() {
    const m = this.match;
    const wall = m.builders.map((b) => b.security?.wall ?? 0);
    const placed = m.builders.reduce((a, b) => a + b.stats.placed, 0);
    const broken = m.builders.reduce((a, b) => a + b.stats.broken, 0);
    const deaths = m.builders.reduce((a, b) => a + b.stats.deaths, 0);
    const escapes = m.builders.reduce((a, b) => a + b.stats.escapes, 0);
    const updates = m.league.brains.reduce((a, b) => a + b.model.trainSteps, 0);
    await this.store.recordCycle({
      cycle: m.cycle,
      kills: this.hunt.kills, smashed: this.hunt.smashed, grabs: this.hunt.grabs,
      playerKills: this.hunt.playerKills,
      deaths, escapes, placed, broken,
      wallMean: wall.reduce((a, b) => a + b, 0) / (wall.length || 1),
      bestScore: Math.max(0, ...m.builders.map((b) => b.score)),
      secureSeconds: Math.round(m.builders.reduce((a, b) => a + (b.security?.sealed ? 90 : 0), 0)),
      updates,
      builders: m.builders.map((b) => ({
        name: b.name, wall: b.security?.wall ?? 0, score: Math.round(b.score),
        deaths: b.stats.deaths, escaped: b.stats.escapes, placed: b.stats.placed,
        killedByLocust: 0,
      })),
    });
    this.hunt = { kills: 0, smashed: 0, grabs: 0, playerKills: 0 };
  }

  reset() {
    this.hunt = { kills: 0, smashed: 0, grabs: 0, playerKills: 0 };
    const learn = this.match.learn;
    const league = this.match.league;          // keep the learned brains: a reset
    this.match = this.createMatch({ learn, league }); // is not amnesia
    this.lastTick = Date.now();
  }
}

/* ------------------------------------------------------------- static files */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.map': 'application/json',
};

// only these prefixes are ever readable from disk
const SERVE = {
  '/': 'public/index.html',
  '/css/': 'public/css/',
  '/js/': 'public/js/',
  '/vendor/': 'public/vendor/',
  '/core/': 'core/',
  '/shared/': 'shared/',
  '/ai/': 'ai/',
  '/game/': 'game/',
};

async function serveStatic(req, res, pathname) {
  let rel = null;
  if (pathname === '/' || pathname === '/index.html') rel = 'public/index.html';
  else {
    for (const [prefix, dir] of Object.entries(SERVE)) {
      if (prefix !== '/' && pathname.startsWith(prefix)) { rel = dir + pathname.slice(prefix.length); break; }
    }
  }
  if (!rel) return false;
  const abs = path.resolve(ROOT, rel);
  const within = abs === ROOT || abs.startsWith(ROOT + path.sep);
  if (!within || abs.includes(`${path.sep}..${path.sep}`)) return false;
  if (path.basename(abs) === '.env') return false;
  let stat, buf;
  try {
    stat = await fs.stat(abs);
    if (stat.isDirectory()) return false;
    buf = await fs.readFile(abs);
  } catch { return false; }
  const ext = path.extname(abs).toLowerCase();
  const isModule = ext === '.js' || ext === '.mjs';
  res.writeHead(200, {
    'content-type': MIME[ext] || 'application/octet-stream',
    'content-length': stat.size,
    'cache-control': abs.includes(`${path.sep}vendor${path.sep}`) ? 'public, max-age=86400' : 'no-cache',
    'x-content-type-options': 'nosniff',
    ...(isModule ? { 'content-type': 'text/javascript; charset=utf-8' } : {}),
  });
  res.end(buf);
  return true;
}

/* ------------------------------------------------------------------ wiring */

export function createGame(config = CONFIG) {
  const game = new Game(config);
  const api = createApi({ game, store: game.store });
  const server = http.createServer(async (req, res) => {
    // The URL parse has to be inside a guard of its own: a request target like
    // `//` (which browsers and proxies do emit) makes `new URL` throw, and an
    // exception before the try block takes the whole process down — i.e. one
    // malformed line from any scanner ended the game for everybody.
    let url;
    const target = String(req.url || '/').replace(/^\/+/, '/');   // `//js/x.js` → `/js/x.js`
    try {
      url = new URL(target, `http://${req.headers.host || 'localhost'}`);
    } catch {
      res.writeHead(400, { 'content-type': 'text/plain; charset=utf-8' });
      res.end('400 — malformed request target');
      return;
    }
    try {
      if (url.pathname.startsWith('/api/')) {
        const handled = await api(req, res, url.pathname);
        if (handled) return;
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'no such endpoint', path: url.pathname }));
        return;
      }
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405, { allow: 'GET, HEAD, POST' }); res.end(); return;
      }
      if (await serveStatic(req, res, url.pathname)) return;
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      res.end('404 — not found. The client lives at /');
    } catch (err) {
      console.error('[http]', err);
      if (!res.headersSent) res.writeHead(500, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: String(err?.message || err) }));
    }
  });

  const hub = new WebSocketHub(server, {
    onConnect(client) {
      client.data.name = 'guest';
      client.send(JSON.stringify({
        t: 'hello', config: game.config(), rules: { phase: PHASE, tickMs: TIMING.TICK_MS },
        world: packWorld(game.match.world), state: encodeState(game.match),
      }));
    },
    onMessage(client, msg) {
      if (msg.t === 'join') {
        client.data.name = String(msg.name || 'guest').slice(0, 20);
        if (msg.config) game.applyConfig(msg.config);
        client.send(JSON.stringify({ t: 'joined', config: game.config(), world: packWorld(game.match.world) }));
        return;
      }
      if (msg.t === 'ping') { client.send(JSON.stringify({ t: 'pong', at: msg.at })); return; }
      game.applyInput(msg);
    },
    onDisconnect() {},
  });
  game.hub = hub;

  return { server, game, hub, api };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const { server, game } = createGame();
  await game.restore();
  game.start();
  server.listen(CONFIG.port, CONFIG.host, () => {
    console.log(`\n  Build to Survive the Locust`);
    console.log(`  http://${CONFIG.host}:${CONFIG.port}  (open this in a browser)`);
    console.log(`  world ${game.match.world.sx}×${game.match.world.sy}×${game.match.world.sz}`
      + `  builders ${game.match.builders.length}  learn=${game.match.learn}  sims×${game.match.simScale}`
      + `  assist=${game.match.assist}\n`);
  });
  const shut = async () => {
    console.log('\n[game] saving learned brains and closing…');
    clearInterval(game.timer);
    await game.store.close();
    game.hub.closeAll();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1500);
  };
  process.on('SIGINT', shut);
  process.on('SIGTERM', shut);
}

export { Game, CONFIG };
