import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Net } from '../public/js/net.js';
import { until } from './helpers/until.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/**
 * The hard part of the fallback is not the engine, it is *not hanging*: every one
 * of these tests hands `Net` a WebSocket that misbehaves in a different way (a
 * constructor that throws, an instant reset, a proxy that accepts TCP and then
 * never answers the upgrade) and requires the page to arrive at a playable state —
 * or to say, in words, why it could not. "connecting…" forever is the failure
 * these pin, because it looks like a broken server to the person at the keyboard.
 */
function withFakeWebSocket(ctor, body) {
  const saved = globalThis.WebSocket;
  const savedLoc = Object.getOwnPropertyDescriptor(globalThis, 'location');
  return (async () => {
    globalThis.WebSocket = ctor;
    try { return await body(); } finally {
      if (saved) globalThis.WebSocket = saved; else delete globalThis.WebSocket;
      if (savedLoc) Object.defineProperty(globalThis, 'location', savedLoc);
    }
  })();
}

function makeNet(extra = {}) {
  const inbox = [], status = [], modes = [];
  const net = new Net({
    url: 'ws://nowhere.invalid:1/ws',
    probeMs: extra.probeMs ?? 60,
    onMessage: (m) => inbox.push(m),
    onStatus: (t) => status.push(t),
    onMode: (mode, why, engine, diag) => modes.push({ mode, why, engine, diag }),
    localLoader: extra.localLoader ?? (() => import('../game/local.js')),
    localConfig: { flat: true, sims: 0.2, match: { phaseScale: 0.02 } },
  });
  return { net, inbox, status, modes };
}

test('a socket that never answers becomes a running match in the tab', () => withFakeWebSocket(class {
  constructor() { this.readyState = 0; } send() {} close() {}
}, async () => {
  const { net, inbox, status, modes } = makeNet();
  try {
    net.connect();
    assert.equal(net.mode, 'connecting', 'it does try the server first');
    assert.ok(status.length >= 1, 'and the page is told that is what it is doing');
    net.send({ t: 'join', name: 'Late' });        // typed before anything answered

    await until(() => net.mode === 'local', 4000, 'the probe deadline to hand over');
    assert.equal(modes[0].mode, 'local');
    assert.match(modes[0].why, /nothing answered/, 'with the reason on the menu line');
    assert.equal(typeof modes[0].diag.loadMs, 'number', 'and how long the engine took to load');
    assert.ok(net.local.match, 'the tab owns a real match now');
    await until(() => inbox.some((m) => m.t === 'hello'), 3000, 'the queued join to be answered');
    assert.equal(net.local.name, 'Late', 'queued while the socket was still being tried');
    assert.equal(net.tries, 0, 'no reconnect storm behind a fallback');
  } finally { net.close(); }
}));

test('a refused socket falls back immediately, not after the probe', () => withFakeWebSocket(class {
  constructor(url) { this.url = url; this.readyState = 1; this.sent = []; setTimeout(() => this.onclose && this.onclose({ code: 1006 }), 5); }
  send(s) { this.sent.push(s); } close() { this.readyState = 3; }
}, async () => {
  const t0 = Date.now();
  const { net } = makeNet({ probeMs: 10_000 });     // the deadline must not be what saves us here
  try {
    net.connect();
    await until(() => net.mode === 'local', 1500, 'the instant close to hand over');
    assert.ok(Date.now() - t0 < 900, `refused in ~${Date.now() - t0} ms, not after a timeout`);
    assert.equal(net.open, true);
  } finally { net.close(); }
}));

test('a WebSocket constructor that throws still ends in a playable page', () => withFakeWebSocket(class {
  constructor() { throw new Error('no WebSocket in this context'); }
}, async () => {
  const { net, modes, status } = makeNet({ probeMs: 10_000 });
  try {
    net.connect();                                   // a throw, not an event: no deadline was needed
    await until(() => net.mode === 'local', 4000, 'the throw to be turned into local mode');
    assert.match(modes[0].why, /no WebSocket in this context/);
    assert.ok(status.some((s) => /running the match in this tab/.test(s)), 'and it says so on the menu');
    assert.ok(net.local.stats.ticks >= 0);
  } finally { net.close(); }
}));

test('a local engine that fails to load says so instead of spinning', () => withFakeWebSocket(class {
  constructor() { this.readyState = 0; } send() {} close() {}
}, async () => {
  const { net, modes } = makeNet({ localLoader: () => Promise.reject(new Error('404 — game/local.js')) });
  try {
    net.connect();
    await until(() => net.mode === 'connecting' && modes.length > 0, 4000, 'the failure to be reported');
    assert.equal(modes[0].mode, 'failed');
    assert.match(modes[0].why, /404 — game\/local\.js/, 'the real reason, on screen');
    assert.match(modes[0].why, /reload to try again/, 'and what to do about it');
    assert.equal(net.local, null, 'no half-built engine pretending to be live');
    assert.equal(net.diag.error.includes('404'), true, 'the diagnostics object carries it too');
  } finally { net.close(); }
}));

test('a LocalGame that throws is reported, not swallowed as a promise rejection', () => withFakeWebSocket(class {
  constructor() { this.readyState = 0; } send() {} close() {}
}, async () => {
  const { net, modes } = makeNet({
    localLoader: () => Promise.resolve({ LocalGame: class { constructor() { throw new Error('localStorage is full'); } } }),
  });
  try {
    net.connect();
    await until(() => modes.length > 0, 4000, 'the constructor throw to surface');
    assert.equal(modes[0].mode, 'failed');
    assert.match(modes[0].why, /localStorage is full/, 'this used to be an unhandled rejection and a permanent spinner');
  } finally { net.close(); }
}));

test('a frame that blocks localStorage still ends up in a playable match', () => withFakeWebSocket(class {
  constructor() { this.readyState = 0; } send() {} close() {}
}, async () => {
  // This is what an embedded preview (sandboxed iframe) does to `localStorage`:
  // not "undefined", but a getter that throws. Accessing it during the local boot
  // used to reject the promise after the mode had flipped, which is how the page
  // got stuck on "connecting…" with no server error to look at.
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() { throw new Error('SecurityError: access denied'); },
  });
  const { net, inbox } = makeNet();
  try {
    net.connect();
    net.send({ t: 'join', name: 'You' });
    await until(() => net.local, 4000, 'the fallback to survive a throwing storage getter');
    assert.equal(net.mode, 'local');
    assert.equal(net.local.storage, null, 'persistence is a bonus, not a dependency');
    await until(() => inbox.some((m) => m.t === 'hello'), 3000, 'the world payload');
    const save = net.local.saveBrains();
    assert.equal(save.ok, false);
    assert.match(save.error, /no storage/, 'and saying "no storage" is a result, not a crash');
  } finally {
    net.close();
    if (saved) Object.defineProperty(globalThis, 'localStorage', saved); else delete globalThis.localStorage;
  }
}));

test('a frame that allows localStorage uses it', () => withFakeWebSocket(class {
  constructor() { this.readyState = 0; } send() {} close() {}
}, async () => {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const mem = new Map();
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k) => (mem.has(k) ? mem.get(k) : null),
      setItem: (k, v) => { mem.set(k, String(v)); },
      removeItem: (k) => { mem.delete(k); },
    },
  });
  const { net } = makeNet();
  try {
    net.connect();
    await until(() => net.local, 4000, 'the local engine with usable storage');
    assert.equal(net.local.storage, globalThis.localStorage, 'the working object is handed through');
    assert.ok(mem.has('__gbtl_probe__') === false, 'the probe cleans up after itself');
    net.send({ t: 'join', name: 'You' });
    net.send({ t: 'save' });
    assert.ok(mem.size > 0, 'and a save lands in it');
  } finally {
    net.close();
    if (saved) Object.defineProperty(globalThis, 'localStorage', saved); else delete globalThis.localStorage;
  }
}));

test('a static deployment is configured in the repo, not in tribal knowledge', async () => {
  const toml = await fs.readFile(path.join(ROOT, 'netlify.toml'), 'utf8');
  assert.match(toml, /publish\s*=\s*"\."/m, 'publish dir must be the repo root, where all five module trees live');
  assert.match(toml, /command\s*=\s*""/, 'no build step: plain ES modules and a vendored three');
  assert.doesNotMatch(toml, /^\[\[redirects\]\]/m, 'and no catch-all rewrite — that is what answers every .js request with index.html at HTTP 200');
  assert.match(toml, /\/public\/vendor\/\*/, 'the vendored library is the one thing worth caching forever');
});

test('the page itself refuses to be a silent spinner', async () => {
  const html = await fs.readFile(path.join(ROOT, 'index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)].map((m) => ({ attrs: m[1], body: m[2] }));

  // an error surface that lives *inside* main.js cannot report main.js failing to
  // load, so the hook has to be inline and before the module tag
  const hook = scripts.find((x) => /__gbtlNote/.test(x.body));
  assert.ok(hook, 'an inline error hook must exist');
  assert.ok(scripts.indexOf(hook) < scripts.findIndex((x) => /type="module"/.test(x.attrs)),
    'and it must be installed before the module tag, or a failed module load is reported by nobody');
  assert.doesNotMatch(hook.attrs, /type="module"/, 'classic script, so it runs even if module loading is broken');
  assert.match(hook.body, /addEventListener\('error', function \(e\) \{[\s\S]*\}, true\)/, 'capture phase: resource errors do not bubble');
  assert.match(hook.body, /tagName === \'SCRIPT\'|tagName === "SCRIPT"/, 'it distinguishes a failed <script>/<link> load from a thrown error');
  assert.match(hook.body, /rewrites every path to index\.html/, 'and names the catch-all rewrite, the usual cause');
  assert.match(html, /loading the client modules…/, 'the initial text must describe the client, not promise a server');

  // …and a watchdog after it, because "the module loaded but never ran" produces
  // no error event at all: the import succeeded, the constructor did not finish
  const watchdogIdx = scripts.findIndex((x) => /window\.__game/.test(x.body));
  assert.ok(watchdogIdx >= 0, 'there must be a watchdog after the module tag');
  const watchdog = { 2: scripts[watchdogIdx].body };
  assert.doesNotMatch(scripts[watchdogIdx].attrs, /type="module"/, 'and it cannot be a module, or it dies with the module graph');
  for (const p of ['public/js/main.js', 'public/vendor/three.module.min.js', 'shared/rules.js', 'game/local.js']) {
    assert.ok(watchdog[2].includes(p), `the watchdog probes ${p}`);
  }
  assert.match(watchdog[2], /document\.baseURI/, 'and prints what it resolved against — that is the thing that breaks on a subpath host');
  assert.match(watchdog[2], /file:.*cannot load ES modules/s, 'naming the one case no fallback can fix');

  const css = await fs.readFile(path.join(ROOT, 'public', 'css', 'style.css'), 'utf8');
  assert.match(css, /#menu-status\.error/, 'the status line needs an error style, not grey-on-grey');

});

test('main.js surfaces its own boot failures', async () => {
  const src = await fs.readFile(path.join(ROOT, 'public', 'js', 'main.js'), 'utf8');
  assert.match(src, /window\.__gbtlNote\(msg\)/, 'it reports through the page-level sink, not a private copy');
  assert.match(src, /window\.addEventListener\('error'/, 'and keeps its own handlers as a fallback');
  assert.match(src, /window\.addEventListener\('unhandledrejection'/, 'including rejected promises');
  assert.match(src, /client = new Client\(\);/, 'the boot is inside a try');
  assert.match(src, /client booted — asking for a game server/, 'and the first words are about the client');
  assert.match(src, /meshing the voxel world…/, 'the other multi-second step announces itself');
  assert.match(src, /localConfig: \{ sims: 0\.35, learn: false \}/, 'the in-tab match starts cheap: it shares the tab with the renderer');
  // the deferred hello is deliberate, and only correct because of the paint
  assert.match(src, /requestAnimationFrame\(\(\) => requestAnimationFrame\(\(\) =>/, 'heavy boot work waits two frames so the status paints');
});
