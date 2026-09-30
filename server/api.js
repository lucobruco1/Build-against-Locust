/**
 * REST layer. Everything the browser *could* want outside the WebSocket: the
 * world payload to build a mesh from, live AI telemetry, config, persistence.
 * The WS channel is used for the 20 Hz state stream; this is stateless JSON.
 */

import { encodeState, packWorld } from '../game/net.js';
import { BLOCK_DEFS, BUILDER_ACTION_NAMES, LOCUST_ACTION_NAMES, EZ, WORLD, BASE, TIMING, N_BOTS } from '../shared/rules.js';

export function createApi({ game, store }) {
  return async function handleApi(req, res, pathname) {
    const send = (code, obj, type = 'application/json; charset=utf-8') => {
      const body = type.startsWith('application/json') ? JSON.stringify(obj) : obj;
      res.writeHead(code, {
        'content-type': type,
        'content-length': Buffer.byteLength(body),
        'cache-control': 'no-store',
        'access-control-allow-origin': '*',
      });
      res.end(body);
      return true;
    };
    const match = game.match;
    const seg = pathname.replace(/^\/api\//, '').split('/');

    if (req.method === 'GET') {
      if (seg[0] === 'health') {
        return send(200, {
          ok: true, name: 'build-to-survive-the-locust',
          phase: match.phase, cycle: match.cycle, tick: match.tickCount,
          uptimeS: Math.round(process.uptime()), timeScale: game.timeScale, paused: game.paused,
          clients: game.hub ? game.hub.clients.size : 0,
        });
      }
      if (seg[0] === 'world') return send(200, packWorld(match.world));
      if (seg[0] === 'state') return send(200, encodeState(match));
      if (seg[0] === 'brains') return send(200, { brains: match.snapshot().stats, ez: EZ });
      if (seg[0] === 'summary') return send(200, await store.summary());
      if (seg[0] === 'rules') {
        return send(200, {
          world: WORLD, base: BASE, timing: TIMING, nBots: N_BOTS,
          blocks: BLOCK_DEFS.map((d, i) => ({ id: i, name: d.name, color: d.color, solid: d.solid, hardness: d.hardness, locustHP: d.locustHP, tiles: d.tiles, placeable: d.placeable })).filter((d) => d.id > 0),
          actions: BUILDER_ACTION_NAMES, locustActions: LOCUST_ACTION_NAMES,
        });
      }
      if (seg[0] === 'snapshot') {
        // debug aid: full match snapshot, untrimmed
        return send(200, match.snapshot());
      }
    }

    if (req.method === 'POST') {
      if (seg[0] === 'config') {
        const body = await readBody(req);
        game.applyConfig(body);
        return send(200, { ok: true, config: game.config() });
      }
      if (seg[0] === 'save') {
        const meta = { cycle: match.cycle, ticks: match.tickCount, config: game.config() };
        const p = await store.saveBrains(match.league.checkpoints(), meta);
        return send(200, { ok: true, file: p, brains: Object.keys(meta.cycle != null ? match.league.checkpoints() : {}).length });
      }
      if (seg[0] === 'load') {
        const data = await store.loadBrains();
        if (!data) return send(404, { ok: false, error: 'no saved checkpoint yet' });
        const n = match.league.loadCheckpoints(data.checkpoints);
        return send(200, { ok: true, loaded: n, savedAt: data.savedAt });
      }
      if (seg[0] === 'reset') {
        game.reset();
        return send(200, { ok: true, config: game.config() });
      }
      if (seg[0] === 'action') {
        // REST fallback so the sim is still playable if WS is blocked
        const body = await readBody(req);
        game.applyInput({ t: 'input', ...body });
        return send(200, { ok: true });
      }
    }
    return false;
  };
}

function readBody(req, limit = 262144) {
  return new Promise((resolve) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { req.destroy(); resolve(null); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); } catch { resolve(null); }
    });
    req.on('error', () => resolve(null));
  });
}

export { readBody };
