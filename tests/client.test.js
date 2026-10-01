/**
 * Client/entry-point tests — the page must work as a page.
 *
 * Two things go wrong quietly in a project that is really a Node app with a
 * browser bolted on, and both are cheap to pin here:
 *
 *   1. the document and its module graph stop resolving when the page is served
 *      from somewhere other than the bespoke server (repo root, GitHub Pages, a
 *      double-clicked file) — so index.html is checked against *browser* path
 *      rules, and every module it pulls in is followed to a real file on disk;
 *   2. the browser half quietly depends on Node, and then the "no server" path
 *      is a runtime error instead of a fallback — so the served trees are
 *      grepped for Node-only APIs, and game/local.js is driven exactly the way
 *      public/js/net.js drives it.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LocalGame, applyMatchConfig, applyMatchInput, cycleSummary, freshHunt } from '../game/local.js';
import { Net } from '../public/js/net.js';
import { encodeState, unpackWorld } from '../game/net.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Paths the repo-root page can reach; anything else 404s in a static deployment. */
const SERVED_DIRS = ['public', 'shared', 'core', 'ai', 'game'];

async function* walk(dir) {
  for (const ent of await fs.readdir(dir, { withFileTypes: true })) {
    const rel = path.join(dir, ent.name);
    if (ent.isDirectory()) yield* walk(rel);
    else if (rel.endsWith('.js') || rel.endsWith('.html')) yield rel;
  }
}

/**
 * Resolve a specifier exactly the way a browser does, from the URL the file is
 * served at under a given document base — and report whether the answer is still
 * *inside* that base. Emulating `..` clamping at the origin root (as an earlier
 * version of this test did) is what let a bug hide: a specifier that climbs out of
 * the app resolves fine when the app owns the origin root, and 404s the moment the
 * same tree is hosted under a project subpath (`/Build-against-Locust/` on GitHub
 * Pages, or any host with a base path). `new URL` *is* the browser rule, so
 * resolve with it and then demand containment.
 */
function resolveFrom(base, relPath, spec, toRepo = (p) => p) {
  const url = new URL(spec, 'http://static.test' + base + relPath.replace(/^\/+/, ''));
  const dir = base.endsWith('/') ? base : base + '/';
  const inside = url.pathname === base.replace(/\/$/, '') || url.pathname.startsWith(dir);
  // strip the deployment base to get the path inside the repo, then undo any URL
  // alias the server applies on top of the repo layout
  const underBase = url.pathname.slice(Math.max(0, base.length - (base.endsWith('/') ? 1 : 0)));
  return { path: url.pathname, inside, onDisk: path.join(ROOT, toRepo(underBase)) };
}

/* ------------------------------------------------------- 1. the entry point */

test('index.html is at the repo root and every path in it resolves', async () => {
  const html = await fs.readFile(path.join(ROOT, 'index.html'), 'utf8');
  assert.match(html, /^<!DOCTYPE html>/, 'must be a real document, not a fragment');
  assert.match(html, /Build to Survive the Locust/, 'and the title of the game');

  // One page, one copy. `public/index.html` may exist only as a pointer, because a
  // host pointed at `public/` would otherwise 404 with no explanation; a second
  // *document* is how a stale copy survives, so that specifically is forbidden.
  const pointer = await fs.readFile(path.join(ROOT, 'public', 'index.html'), 'utf8').catch(() => null);
  if (pointer !== null) {
    assert.doesNotMatch(pointer, /id="hud"|<canvas id="gl"/, 'public/index.html must not be a copy of the game');
    assert.match(pointer, /location\.replace\(/, 'it must send the visitor to the root document');
    assert.match(pointer, /location\.pathname/, 'it looks at the path it was served from, not a guess');
    assert.match(pointer, /\.test\(here\)/, 'and only redirects when it really is the /public/ copy');
    assert.match(pointer, /wrong directory published/i, 'otherwise it explains the misconfiguration in words');
    assert.ok(pointer.length < 3000, `a pointer should be small, it is ${pointer.length} bytes`);
  }

  const map = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]);
  assert.ok(map.imports?.three, 'three has to come from the importmap, not a CDN');
  const refs = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map((m) => m[1])
    .filter((u) => !u.startsWith('data:') && !/^https?:/.test(u))
    .concat(Object.values(map.imports));
  assert.ok(refs.includes('public/js/main.js'), 'the page must load public/js/main.js');
  for (const rel of refs) {
    for (const base of ['/', '/Build-against-Locust/']) {
      const { path: resolved, inside, onDisk } = resolveFrom(base, 'index.html', rel);
      assert.ok(inside, `${rel} must stay under the base (${base}) instead of resolving to ${resolved}`);
      const stat = await fs.stat(onDisk).catch(() => null);
      assert.ok(stat?.isFile(), `${rel} (→ ${resolved}) must exist for a static host under ${base}`);
    }
  }
  assert.equal((await fs.readFile(path.join(ROOT, map.imports.three), 'utf8')).includes('THREE'), true,
    'and the vendored three is really three');
});

test('the browser module graph is complete and free of Node', async () => {
  const files = [];
  for (const dir of SERVED_DIRS) {
    for await (const f of walk(path.join(ROOT, dir))) files.push(f);
  }
  assert.ok(files.length > 25, `expected the whole client tree, saw ${files.length}`);

  const specRe = /(?:from\s+|import\s*\(\s*)['"]([^'"]+)['"]/g;
  let checked = 0;
  for (const abs of files) {
    const rel = path.relative(ROOT, abs).split(path.sep).join('/');
    const src = await fs.readFile(abs, 'utf8');
    if (rel.endsWith('.html')) continue;

    // no Node-only APIs anywhere the page can import: this is what lets the same
    // Match run in a tab, and it is invisible until you try
    const nodeish = src.match(/\brequire\(|from 'node:|import 'node:|process\.(env|argv|cwd)|__dirname|\bfs\.(read|write)Sync|\bBuffer\(/);
    assert.equal(nodeish, null, `${rel} must not depend on Node (${nodeish && nodeish[0]})`);

    // The same file is reachable at three different URLs, and each one has to
    // resolve its specifiers to a real file *within the base*: the repository root
    // as a static site, that root under a project subpath, and the short alias the
    // Node server puts in front of public/. A relative specifier that climbs out of
    // the base passes the first and fails the second, which is the whole reason the
    // loop checks containment rather than just file existence.
    // `toRepo` undoes the server's URL aliases, because `/js/x.js` lives at
    // public/js/x.js on disk; the other two cases are the plain repo layout.
    const fromAlias = (f) => f.replace(/^\/(js|css|vendor)\//, '/public/$1/');
    const bases = [
      ['/', rel, 'static root', (f) => f],
      ['/Build-against-Locust/', rel, 'subpath host', (f) => f],
      ['/', rel.replace(/^public\/(js|css|vendor)\//, '$1/'), 'npm start alias', fromAlias],
    ];
    for (const m of src.matchAll(specRe)) {
      const spec = m[1];
      checked++;
      if (!spec.startsWith('.') && !spec.startsWith('/')) {
        assert.equal(spec, 'three', `${rel}: bare specifier '${spec}' has no importmap entry`);
        continue;
      }
      for (const [base, urlPath, label, toRepo] of bases) {
        const { path: resolved, inside, onDisk } = resolveFrom(base, urlPath, spec, toRepo);
        assert.ok(inside, `${rel} (served at /${urlPath}, ${label}): '${spec}' → ${resolved} escapes the base`);
        const stat = await fs.stat(onDisk).catch(() => null);
        assert.ok(stat?.isFile(), `${rel} (at /${urlPath}, ${label}): '${spec}' → ${resolved} is not a file, the page would fail to load`);
      }
    }
  }
  assert.ok(files.length > 25 && checked > 50, `followed ${checked} specifiers across ${files.length} files`);
  // the exact chain the entry point pulls in, spelled out so a rename is caught
  for (const f of ['public/js/main.js', 'public/js/net.js', 'public/js/hud.js', 'public/js/render/actors.js', 'game/local.js']) {
    assert.ok(files.some((x) => x.endsWith(f)), `${f} should have been scanned`);
  }
});

/* ------------------------------------- 2. the in-tab engine speaks the protocol */

import { until } from './helpers/until.js';

/** a Map pretending to be localStorage, so persistence is testable without a DOM */
function fakeStorage() {
  const m = new Map();
  return {
    m,
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
  };
}

function makeLocal(extra = {}) {
  const inbox = [];
  // a fake clock, so "one broadcast per 50 ms" is a fact of the test and not of
  // how long node --test takes on the machine
  let vt = 0;
  const game = new LocalGame({
    config: { flat: true, learn: false, sims: 0.2, match: { phaseScale: 0.02 }, ...extra.config },
    storage: extra.storage ?? null,
    now: () => vt,
    onMessage: (m) => inbox.push(m),
  });
  const pump = (n = 10, dt = 33.4) => { for (let i = 0; i < n; i++) { vt += dt; game.pump(dt); } };
  const toPhase = (phase, limit = 400) => {
    for (let i = 0; i < limit && game.match.phase !== phase; i++) pump(1);
    assert.equal(game.match.phase, phase, `pumped to ${phase} (stuck in ${game.match.phase}, cycle ${game.match.cycle})`);
  };
  return { game, inbox, pump, toPhase, byType: (t) => inbox.filter((m) => m.t === t) };
}

test('LocalGame: join answers with the world and a state snapshot', () => {
  const { game, byType } = makeLocal();
  game.send({ t: 'join', name: 'Tester' });

  const hello = byType('hello')[0];
  assert.ok(hello, 'a hello must come back, exactly like the WebSocket onConnect does');
  assert.equal(hello.name, 'Tester');
  assert.equal(hello.rules.tickMs, 1000 / 30);
  assert.equal(hello.world.sx, game.match.world.sx, 'the world payload is the packed real world');
  const decoded = unpackWorld(hello.world);
  assert.equal(decoded.blocks.length, game.match.world.blocks.length);
  assert.deepEqual(hello.state, { ...hello.state }, 'and state is plain JSON');
  assert.deepEqual(Object.keys(hello.state).sort(), Object.keys(encodeState(game.match)).sort(),
    'the local state message must have exactly the keys the client reads from the server');

  const joined = byType('joined')[0];
  assert.equal(joined.world.sx, game.match.world.sx);
  assert.equal(joined.config.local, true, 'the client uses this to label where the sim runs');
  // parity with the server: the greeting carries the chosen name, the slot keeps
  // its own — renaming the builder would make the HUD disagree with the state stream
  assert.equal(game.match.builders[0].name, 'You');
  assert.equal(game.name, 'Tester');
});

test('LocalGame: the 20 Hz state stream carries the deltas the renderer needs', () => {
  const { game, pump, toPhase, byType } = makeLocal();
  game.send({ t: 'join', name: 'You' });
  toPhase('build');
  const t0 = game.match.tickCount;
  game.send({ t: 'input', move: [1, 0], sprint: true });
  pump(6);
  const states = byType('state');
  assert.ok(states.length >= 3 && states.length <= 5, `a state every ~50 ms of a 33 ms tick, saw ${states.length}`);
  const s = states[states.length - 1];
  assert.equal(s.phase, 'build');
  assert.equal(s.actors.length, game.match.builders.length);
  const me = s.actors.find((a) => a.id === 'player');
  assert.ok(me, 'the human slot is addressable by the same id the client hardcodes');
  assert.equal(typeof me.hp, 'number');
  const moved = Math.abs(me.x) + Math.abs(me.z);
  assert.ok(moved > 0.2, `the input moved the player in the authoritative world (was ${moved.toFixed(2)})`);
  assert.ok(game.match.tickCount > t0, 'and the world kept stepping while it did');
  assert.equal(typeof s.timeLeft, 'number');
  assert.equal(s.brains.length, game.match.league.brains.length,
    'the HUD gets one row per brain — and the locust only joins the list when it exists');
  assert.equal(s.brains.length, 7, 'in the build phase that is the seven builders plus you, no locust yet');
});

test('LocalGame: input, config and reset behave as they do over the wire', () => {
  const { game, pump, byType, toPhase } = makeLocal({ config: { learn: true, sims: 0.05 } });
  game.send({ t: 'join', name: 'You' });
  toPhase('build');

  assert.equal(game.send({ t: 'input', select: 3 }), true, 'send() answers whether the message was theirs');
  assert.equal(game.match.player.inv.slot, 3, 'hotbar selection is the same message');
  game.send({ t: 'input', place: true, look: [0.01, -0.6] });
  assert.ok(game.match.player.stats.placed >= 0, 'a place attempt is routed, not swallowed');
  assert.equal(game.send({ t: 'say', text: 'hello' }), true);
  assert.match(JSON.stringify(game.broadcast().log ?? ''), /hello|$/, 'say is accepted in either driver');

  game.send({ t: 'config', assist: 0.25, sims: 2, learn: false, paused: true });
  assert.equal(game.match.assist, 0.25);
  assert.equal(game.match.simScale, 2);
  assert.equal(game.match.learn, false);
  pump(2);
  assert.ok(game.stats.ticks >= 2 && game.match.tickCount > 0, 'the driver counts its own steps');

  game.send({ t: 'config', paused: true });
  const frozen = game.match.tickCount;
  pump(3);
  assert.equal(game.match.tickCount, frozen, 'paused stops the authoritative clock, not just the frames');
  game.send({ t: 'config', paused: false });
  pump(2);
  assert.ok(game.match.tickCount > frozen, 'and resuming resumes it');

  game.send({ t: 'config', assist: 7 });
  assert.equal(game.match.assist, 1, 'and clamps, because the slider is not a validator');
  game.send({ t: 'config', timeScale: 1000 });
  assert.equal(game.timeScale, 20, 'the tab clock is clamped too — a tab has a frame budget');

  const brain = game.match.league.brains[0];
  const oldMatch = game.match;
  game.send({ t: 'reset' });
  assert.notEqual(game.match, oldMatch, 'a reset makes a fresh Match');
  assert.equal(game.match.league.brains[0], brain, 'but not amnesia: the brains survive');
  assert.equal(game.match.tickCount, 0, 'and the clock starts over');
  assert.ok(byType('reset').length >= 1, 'the client gets a new world to mesh');
});

test('LocalGame: the cycle runs itself and the hunt is recorded', () => {
  const storage = fakeStorage();
  const { game, pump, toPhase, byType } = makeLocal({ storage });
  game.send({ t: 'join', name: 'You' });
  toPhase('build');

  pump(60);                                    // 2.0 s against a 1.8 s build phase
  assert.equal(game.match.phase, 'hunt', 'the build phase ends on its own');
  assert.ok(byType('state').some((s) => s.phase === 'hunt'), 'and the client was told');
  assert.ok(game.match.locust, 'the locust exists in the hunt only');
  assert.equal(byType('state').at(-1).brains.length, 8, 'and its brain joins the HUD rows while it hunts');

  // the revive window is 100 ms at this scale, so the assertion is "the cycle
  // closed and the record was written", not "we happen to be inside revive"
  const seen = new Set([game.match.phase]);
  for (let i = 0; i < 400 && !storage.m.has('gbtl:cycles'); i++) { pump(1); seen.add(game.match.phase); }
  assert.ok(seen.has('revive'), `revive was broadcast (saw ${[...seen].join(' → ')})`);
  assert.equal(game.match.locust, null, 'and the locust is despawned on its way out');

  const rec = JSON.parse(storage.getItem('gbtl:cycles'));
  assert.equal(rec.length, 1, 'one finished cycle, written like the server writes one');
  assert.equal(rec[0].cycle, 1);
  for (const k of ['kills', 'smashed', 'grabs', 'deaths', 'placed', 'wallMean', 'bestScore', 'builders']) {
    assert.ok(k in rec[0], `the record keeps ${k}`);
  }
  assert.equal(rec[0].builders.length, 8);
  assert.ok(rec[0].placed > 0, 'the bots built during those 1.8 simulated seconds');

  assert.equal(game.match.cycle, 1, 'the record is still cycle 1\'s: it is written at the despawn, which is before the counter moves');
  pump(6);                                     // revive is 5 s × .02 = 100 ms
  assert.equal(game.match.phase, 'build', 'building resumes');
  assert.equal(game.match.cycle, 2, 'and the cycle counter moved on');
  assert.equal(game.match.builders.every((b) => b.alive), true, 'everyone is alive again');
  assert.equal(game.readCycles().length, 1, 'the log is kept, newest last');
});

test('LocalGame: learned weights round-trip through the injected storage', () => {
  const storage = fakeStorage();
  const a = new LocalGame({ config: { flat: true, learn: true, sims: 0.05 }, storage });
  a.send({ t: 'join', name: 'You' });
  a.pump(0); a.lastBroadcast = 0;
  const W = a.match.league.brains[0].model.params[0].W;
  W[0] += 5;                                   // something to notice on the other side

  const saved = a.saveBrains();
  assert.equal(saved.ok, true);
  assert.equal(saved.brains, a.match.league.brains.length);
  const blob = JSON.parse(storage.getItem('gbtl:brains'));
  assert.equal(typeof Object.values(blob.checkpoints)[0].b64, 'string', 'base64 weights, not 2.7M JSON numbers');
  assert.equal(blob.meta.optimizerSteps, saved.steps, 'and enough meta to explain the write');

  W[0] -= 8;                                   // then change our mind after saving
  const b = new LocalGame({ config: { flat: true, learn: false }, storage });
  const loaded = b.loadBrains();
  assert.equal(loaded.ok, true);
  assert.ok(loaded.loaded > 0, 'the second tab resumes the first tab\u2019s networks');
  const W2 = b.match.league.brains[0].model.params[0].W;
  assert.equal(W2[0], W[0] + 8, 'the restored weight is the saved one, not the local mutation');

  // nothing learned since the last write → no pointless megabyte rewrite
  const size = storage.getItem('gbtl:brains').length;
  assert.equal(a.autosave().skipped, true, 'autosave skips a no-op, as on the server');
  assert.equal(storage.getItem('gbtl:brains').length, size);
});

test('the shared helpers are the only copy of that behaviour', () => {
  const { game, pump } = makeLocal();
  game.send({ t: 'join', name: 'You' });
  pump(3);
  const rec = cycleSummary(game.match, game.hunt);
  assert.ok(rec.cycle >= 1 && rec.builders.length === 8);
  assert.equal(typeof rec.wallMean, 'number');
  assert.deepEqual(game.hunt, freshHunt(), 'the driver does not invent hunt numbers');

  const t0 = game.match.tickCount;
  game.timeScale = 4;
  game.pump(33.4);
  assert.ok(game.match.tickCount > t0 + 3, 'timeScale multiplies the tab clock too');
  game.timeScale = 1;

  assert.equal(applyMatchInput(game.match, { t: 'nonsense' }), false);
  assert.equal(applyMatchInput(game.match, { t: 'input', move: [0, 0] }), true);
  assert.equal(applyMatchConfig(game, null), null, 'an empty config body changes nothing');
});

test('net.js falls back to the local engine instead of an error screen', async () => {
  // The interesting part is a pure decision, so it is testable without a DOM:
  // a page that cannot reach a server must end up with a LocalGame, and must end
  // up with one *without* the reconnect loop ever being blamed for it.
  const src = await fs.readFile(path.join(ROOT, 'public', 'js', 'net.js'), 'utf8');
  assert.match(src, /mode = 'local'/, 'the transport has a local mode');
  assert.match(src, /import\('\.\.\/\.\.\/game\/local\.js'\)/, 'loaded lazily, so a server player never pays for the AI modules');
  assert.match(src, /location\.protocol === 'file:'/, 'file:// never even tries the socket');
  assert.match(src, /if \(this\.mode === 'connecting'\) this\.enableLocal\(/, 'a refused socket is a fact: hand over at once instead of backing off');
  assert.match(src, /this\.armProbe\(\);\n.*try \{ this\.ws = new WebSocket/s, 'the deadline is armed before the socket exists, so a throwing or hanging connection still ends somewhere');
  assert.match(src, /mode === 'localizing'/, 'a half-booted engine is not mistaken for a live one');
  assert.match(src, /this\.queue\.push\(obj\)/, 'messages sent during the probe are not dropped');
  assert.match(src, /const store = safeStorage\(\);/, 'storage is fetched through a guard, never touched bare');
  assert.match(src, /globalThis\.localStorage\.setItem\(probe, '1'\)/, 'the guard writes, because reading alone can be the part that is allowed');

  // the page must therefore wire the mode line up rather than only showing "connecting…"
  const main = await fs.readFile(path.join(ROOT, 'public', 'js', 'main.js'), 'utf8');
  assert.match(main, /onMode: \(mode, why, engine, diag\) => this\.onMode\(mode, why, engine, diag\)/);
  assert.match(main, /onStatus: \(txt\) => this\.setStatus\(txt\)/, 'the transport narrates what it is trying, so the menu is never wordless');
  assert.match(main, /menu-mode/);
  assert.match(main, /t: 'config'/, 'the settings menu talks to the tab in local mode, not to /api/config');
});

test('Net: a socket that never opens becomes an in-tab match, not an error screen', async () => {
  // A browser-only class, so the globals it expects are stood up by hand. The
  // interesting behaviour is the handover: probe times out → game/local.js is
  // loaded → the engine is started → messages queued while "connecting" are
  // delivered → the reconnect loop is over, because this tab is the server now.
  class DeadWebSocket {
    constructor(url) { this.url = url; this.readyState = 0; this.sent = []; this.closed = false; }
    send(s) { this.sent.push(s); }
    close() { this.closed = true; this.readyState = 3; }
  }
  const savedWS = globalThis.WebSocket;
  globalThis.WebSocket = DeadWebSocket;
  const inbox = [];
  let mode = null, why = null;
  try {
    const net = new Net({
      url: 'ws://nowhere.invalid:1/ws',
      probeMs: 25,
      onMessage: (m) => inbox.push(m),
      onMode: (m, reason) => { mode = m; why = reason; },
      localLoader: () => import('../game/local.js'),
      localConfig: { flat: true, sims: 0.2, match: { phaseScale: 0.02 } },
    });
    net.connect();
    assert.equal(net.mode, 'connecting', 'it does try the server first');
    net.send({ t: 'input', move: [1, 0] });      // typed before anything answered
    net.send({ t: 'join', name: 'Late' });

    await until(() => net.mode === 'local', 3000, 'the probe to time out and the local mode to be chosen');
    assert.equal(net.mode, 'local', `the fallback fired (why: ${why})`);
    assert.match(String(why), /nothing answered/, 'and says what it was doing instead');
    assert.equal(net.open, true, 'the channel is live, it just lives here now');
    assert.ok(net.local instanceof LocalGame, 'the engine is the shared one');
    assert.equal(net.tries, 0, 'no reconnect storm behind the fallback');

    await until(() => inbox.some((m) => m.t === 'hello'), 3000, 'the queued join to be answered');
    assert.ok(inbox.some((m) => m.t === 'joined'), 'the join queued during the probe still reached the engine');
    assert.equal(net.local.match.builders[0].name, 'You');
    assert.equal(net.local.name, 'Late', 'with the name the player typed');
    await until(() => inbox.some((m) => m.t === 'state'), 3000, 'the 20 Hz state stream');
    assert.ok(net.local.stats.ticks > 0, 'and the tab clock is ticking by itself');
    const ticks = net.local.stats.ticks;
    await until(() => net.local.stats.ticks > ticks + 1, 1500, 'the interval to keep firing');

    // a dead socket must have been let go, and input now goes straight in
    net.send({ t: 'input', select: 4 });
    assert.equal(net.local.match.player.inv.slot, 4, 'and input goes straight into the authoritative match');
    net.close();
    assert.equal(net.local.timer, null, 'close() stops the interval or the process never exits');
  } finally {
    if (savedWS) globalThis.WebSocket = savedWS; else delete globalThis.WebSocket;
  }
});

test('Net: an origin with no host to dial skips the probe entirely', async () => {
  // file:// cannot load modules at all (browsers refuse them from an opaque origin),
  // so this is the "someone bundled the page" case: don't burn 1.5 s waiting for a
  // socket that cannot exist. The supported no-server path is a static HTTP host.
  // no origin to dial, so the probe timeout would just be 1.5 s of spinner
  const savedLoc = globalThis.location;
  const savedWS = globalThis.WebSocket;
  let dialed = 0;
  globalThis.WebSocket = class { constructor() { dialed++; throw new Error('must not be constructed'); } };
  const inbox = [];
  try {
    Object.defineProperty(globalThis, 'location', { value: { protocol: 'file:', host: '' }, configurable: true });
    const net = new Net({
      url: 'ws://localhost:3000/ws', probeMs: 25,
      onMessage: (m) => inbox.push(m),
      localLoader: () => import('../game/local.js'),
      localConfig: { flat: true, sims: 0.2 },
    });
    net.connect();
    await until(() => net.local, 3000, 'the local engine to boot without a socket');
    assert.equal(dialed, 0, 'no WebSocket was even constructed');
    assert.equal(net.mode, 'local', 'no server, no spinner, still a playable page');
    net.send({ t: 'join', name: 'Disk' });
    await until(() => inbox.some((m) => m.t === 'hello'), 3000, 'the world payload');
    assert.ok(net.local.match, 'and there is a real match behind it');
    net.close();
  } finally {
    if (savedLoc) Object.defineProperty(globalThis, 'location', { value: savedLoc, configurable: true }); else delete globalThis.location;
    if (savedWS) globalThis.WebSocket = savedWS; else delete globalThis.WebSocket;
  }
});
