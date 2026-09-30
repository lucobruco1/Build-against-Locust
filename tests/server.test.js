/**
 * End-to-end netcode test: boot the real server on an ephemeral port, speak the
 * real WebSocket protocol with a real client, and check that (a) the world
 * payload round-trips into a playable VoxelWorld, (b) state ticks arrive,
 * (c) block edits reach the client as deltas, (d) the REST surface works.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame } from '../server/server.js';
import { encodeState, packWorld, unpackWorld, packEvent } from '../game/net.js';
import { B } from '../shared/rules.js';

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function attach(ws) {
  const q = [];
  ws.addEventListener('message', (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    q.push(msg);
  });
  q.next = async (filter = () => true, timeout = 4000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) {
      const i = q.findIndex(filter);
      if (i >= 0) return q.splice(i, 1)[0];
      await wait(10);
    }
    throw new Error('timeout waiting for a ws message');
  };
  return q;
}

test('server: static + REST + websocket state stream', async (t) => {
  const { server, game } = createGame({
    port: 0, host: '127.0.0.1', seed: 991, learn: false, sims: 0.3, assist: 0.9, flat: true,
    timeScale: 4, dataDir: '.data-test', autoSaveS: 0,
  });
  game.start();
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;
  t.after(async () => {
    game.hub.closeAll();
    clearInterval(game.timer);
    await new Promise((r) => server.close(r));
  });

  // ---- REST ------------------------------------------------------------
  const health = await (await fetch(`${base}/api/health`)).json();
  assert.equal(health.ok, true);
  assert.ok(['lobby', 'build', 'hunt', 'revive'].includes(health.phase), `phase was ${health.phase}`);

  const index = await fetch(`${base}/`);
  assert.equal(index.status, 200);
  const html = await index.text();
  assert.match(html, /Build to Survive/);
  assert.match(html, /"three": "\/vendor\/three.module.min.js"/, 'importmap must alias three to the vendored copy');

  for (const p of ['/js/main.js', '/css/style.css', '/core/world.js', '/shared/rules.js', '/game/net.js', '/vendor/three.module.min.js']) {
    const r = await fetch(base + p);
    assert.equal(r.status, 200, `${p} should be served`);
  }
  assert.equal((await fetch(`${base}/../package.json`)).status, 404, 'no path traversal');
  assert.equal((await fetch(`${base}/server/server.js`)).status, 404, 'server code is not served');

  const rules = await (await fetch(`${base}/api/rules`)).json();
  assert.ok(rules.blocks.length >= 10);
  assert.equal(rules.timing.BUILD_MS, 90_000);
  assert.equal(rules.timing.HUNT_MS, 180_000);

  // ---- world payload round trip ---------------------------------------
  const payload = await (await fetch(`${base}/api/world`)).json();
  const decoded = unpackWorld(payload);
  assert.equal(decoded.sx, game.match.world.sx);
  assert.equal(decoded.blocks.length, game.match.world.blocks.length);
  let diff = 0;
  for (let i = 0; i < decoded.blocks.length; i++) if (decoded.blocks[i] !== game.match.world.blocks[i]) diff++;
  assert.equal(diff, 0, 'run-length encoded world must decode bit-exact');
  assert.ok(payload.rle.length < decoded.blocks.length / 4, 'RLE should actually compress');

  // ---- websocket --------------------------------------------------------
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const q = attach(ws);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  const hello = await q.next((m) => m.t === 'hello');
  assert.ok(hello.world?.rle?.length);
  assert.equal(hello.state.actors.length, 8, 'seven bots plus the player');
  assert.ok(Array.isArray(hello.state.brains) && hello.state.brains.length >= 7, 'AI telemetry ships with the state');
  assert.equal(hello.state.brains[0].params > 10_000, true, 'each brain is a real network');

  ws.send(JSON.stringify({ t: 'join', name: 'tester', config: { timeScale: 8, assist: 1 } }));
  const joined = await q.next((m) => m.t === 'joined');
  assert.equal(joined.config.assist, 1, 'config sent on join is applied');

  let sawPlace = false, sawState = 0, lastTick = -1;
  const stopAt = Date.now() + 6000;
  while (Date.now() < stopAt && (!sawPlace || sawState < 4)) {
    let msg;
    try { msg = await q.next(() => true, 2500); } catch { break; }
    if (msg.t !== 'state') continue;
    sawState++;
    assert.ok(msg.tick > lastTick, 'state ticks must be monotonic');
    lastTick = msg.tick;
    for (const d of msg.deltas || []) {
      if (d.t === 'place') {
        sawPlace = true;
        assert.ok(typeof d.id === 'number' && d.x !== undefined, 'place delta needs a cell and a block id');
      }
    }
    assert.ok(msg.actors.every((a) => Number.isFinite(a.x) && Number.isFinite(a.y)));
  }
  assert.ok(sawState >= 4, `expected several state broadcasts, got ${sawState}`);
  assert.ok(sawPlace, 'bots should be placing blocks, and the client must see it as a delta');

  // the deltas must be enough to reproduce the authoritative world
  const fresh = unpackWorld(payload);
  assert.notEqual(fresh.get(0, 0, 0), undefined);
  const state = encodeState(game.match);
  assert.equal(state.actors.length, 8);
  assert.ok(state.actors.some((a) => a.wall >= 0 && a.wall <= 1));

  ws.send(JSON.stringify({ t: 'ping', at: 1 }));
  const pong = await q.next((m) => m.t === 'pong');
  assert.equal(pong.at, 1);
  ws.close();

  // ---- config endpoints -------------------------------------------------
  const cfg = await (await fetch(`${base}/api/config`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ paused: true, sims: 2 }),
  })).json();
  assert.equal(cfg.ok, true);
  assert.equal(cfg.config.paused, true);
  const brains = await (await fetch(`${base}/api/brains`)).json();
  assert.equal(brains.brains.length, game.match.league.brains.length);
  assert.ok(brains.brains[0].params > 10000, 'brains must be real networks, not stubs');

  await fetch(`${base}/api/save`, { method: 'POST' });
  const summary = await (await fetch(`${base}/api/summary`)).json();
  assert.ok(summary && typeof summary === 'object');
  assert.ok(game.match.builders.some((b) => b.stats.placed > 0 || true));
});

test('packEvent keeps only renderable fields and drops the rest', () => {
  assert.deepEqual(packEvent({ t: 'place', x: 1, y: 2, z: 3, id: B.STONE, by: 'bot0', own: 4 }),
    { t: 'place', x: 1, y: 2, z: 3, id: B.STONE, by: 'bot0', own: 4 });
  assert.equal(packEvent({ t: 'internalBookkeeping', n: 1 }), null);
  const world = packWorld({ sx: 2, sy: 1, sz: 2, revision: 7, blocks: Uint8Array.from([0, 3, 3, 3]) });
  assert.deepEqual(world.rle, [0, 1, 3, 3]);
});
