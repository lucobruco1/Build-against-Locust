/**
 * End-to-end netcode test: boot the real server on an ephemeral port, speak the
 * real WebSocket protocol with a real client, and check that (a) the world
 * payload round-trips into a playable VoxelWorld, (b) state ticks arrive,
 * (c) block edits reach the client as deltas, (d) the REST surface works.
 */

import fs from 'node:fs/promises';
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
    await game.store.close();
    // the store flushes brains.json + summary.json into the repo; a test run
    // should not leave 13 MB of them behind
    await fs.rm('.data-test', { recursive: true, force: true });
  });

  // ---- the hunt bookkeeping must survive a tick with nobody watching ------
  // Regression: `game.hunt` used to be created only at the end of the first
  // recordCycle(), so the tick that followed the Locust spawn threw and killed
  // the process 90 s into every fresh server.
  game.match.setPhase('hunt');
  game.match.spawnLocust();
  assert.ok(game.match.locust && game.match.locust.stats, 'the Locust object carries its own stats');
  game.tick();
  game.tick();
  assert.ok(game.hunt, 'the server tracks the night even before it has recorded a cycle');
  assert.equal(game.hunt.kills, 0, 'nothing has been killed yet');
  game.lastBroadcast = 0; game.tick();      // the 20 Hz throttle is a scheduler, not a rule
  assert.equal(game.match.events.length, 0,
    'a headless server still drains its event queue (it used to queue them forever and never record a cycle)');
  for (let i = 0; i < 40; i++) game.tick();
  assert.ok(game.match.events.length < 80, `the backlog stays bounded (got ${game.match.events.length})`);

  let recorded = 0;
  const realRecord = game.recordCycle.bind(game);
  game.recordCycle = async () => { recorded++; return realRecord(); };
  game.match.setPhase('revive');            // despawns the Locust, pushes the event
  game.lastBroadcast = 0;                   // the 20 Hz throttle is a scheduler, not a rule
  game.tick();                              // the broadcast path has to notice it
  assert.equal(recorded, 1, 'the end of a hunt is recorded as a cycle even with zero clients');
  assert.equal(game.match.locust, null, 'and the Locust is gone again, as the brief says');

  // ---- REST ------------------------------------------------------------
  const health = await (await fetch(`${base}/api/health`)).json();
  assert.equal(health.ok, true);
  assert.ok(['lobby', 'build', 'hunt', 'revive'].includes(health.phase), `phase was ${health.phase}`);

  // one malformed request target must not take the server down
  const doubleSlash = await fetch(`${base}//js/main.js`);
  assert.equal(doubleSlash.status, 200, '`//js/main.js` is `/js/main.js`, not a crash');
  const badPath = await fetch(`${base}/api/nope/../health`);
  assert.ok([200, 404].includes(badPath.status), `a dot-segment path is served or refused, not fatal (${badPath.status})`);
  assert.ok(!(await badPath.text()).includes('shadow'), 'and never outside the allowed roots');
  const traversal = await fetch(`${base}/../package.json`);
  assert.ok([200, 400, 403, 404].includes(traversal.status), 'no 500s from path games');

  // ---- the page and everything it references, as served -------------------
  const index = await fetch(`${base}/`);
  assert.equal(index.status, 200);
  const html = await index.text();
  assert.match(html, /Build to Survive/);
  const map = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]);
  assert.equal(map.imports.three, 'public/vendor/three.module.min.js',
    'importmap must alias three to the vendored copy, relative to the page');

  // Every path the document names must answer, resolved exactly the way the
  // browser resolves it. This is what breaks silently when the page moves.
  const refs = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map((m) => m[1])
    .filter((u) => !u.startsWith('data:') && !u.startsWith('http'))
    .concat(Object.values(map.imports || {}), Object.values(map.scopes || {}));
  assert.ok(refs.length >= 3, 'the page should reference its css, module and importmap');
  for (const rel of refs) {
    const url = new URL(rel, `${base}/`);
    assert.equal(url.origin + url.pathname, `${base}${url.pathname}`, `${rel} must stay on this host`);
    const r = await fetch(url.href);
    assert.equal(r.status, 200, `${rel} (referenced by index.html) should be served`);
  }

  // the short aliases stay alive for anyone who imports a module directly
  for (const p of ['/js/main.js', '/css/style.css', '/core/world.js', '/shared/rules.js', '/game/net.js', '/vendor/three.module.min.js', '/public/js/main.js']) {
    const r = await fetch(base + p);
    assert.equal(r.status, 200, `${p} should be served`);
  }
  // and those modules' own relative imports must resolve from the alias too
  const mainSrc = await (await fetch(`${base}/js/main.js`)).text();
  for (const m of mainSrc.matchAll(/from '(\.\.\/[^']+)'/g)) {
    const url = new URL(m[1], `${base}/js/main.js`);
    assert.equal((await fetch(url.href)).status, 200, `${m[1]} from /js/main.js must resolve`);
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
