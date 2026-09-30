/**
 * The client's one channel to *whatever* is simulating the match, plus a
 * coalesced input sender.
 *
 * The browser sends at most one `input` message per animation frame (movement
 * axis, look delta, discrete buttons); the other side answers with a 20 Hz
 * `state` message with the world deltas folded in. Everything is JSON — at this
 * scale the cost of JSON is irrelevant and the code stays readable in devtools.
 *
 * Two ways that other side exists:
 *
 *   mode 'server'  a WebSocket to `url` (the Node server, `npm start`). Reconnects
 *                  with backoff, because the server may be restarting mid-hunt.
 *   mode 'local'   nothing answered the first probe — a static host with no game
 *                  server on it, or a page opened from file:// where a socket has no
 *                  meaning — so this tab imports game/local.js and runs the
 *                  authoritative Match here instead of showing an error screen.
 *
 * The fallback is what makes index.html honest: opening the page is enough. The
 * engine is chosen before the first frame is drawn, never mid-match — to hand the
 * game back to a real server, reload.
 */

export class Net {
  constructor(handlers = {}) {
    this.h = handlers;
    this.url = handlers.url || defaultUrl();
    this.probeMs = handlers.probeMs ?? 1500;
    this.ws = null;
    this.open = false;
    this.tries = 0;
    this.lastPong = 0;
    this.pending = null;
    this.closed = false;
    this.mode = 'connecting';   // 'connecting' | 'server' | 'localizing' | 'local' | 'failed'
    this.local = null;
    this.diag = { probeMs: this.probeMs, url: this.url };
    this.queue = [];            // messages sent before the local engine booted
    this._probe = null;
  }

  connect() {
    if (this.closed) return;
    // file:// has no origin to dial. Modules are refused there by the browser, so
    // normally we never get this far — but if a bundler or a same-origin shim did
    // load the page, sitting through a probe timeout would be pure rudeness.
    if (typeof location !== 'undefined' && location.protocol === 'file:') { this.enableLocal('opened straight from disk'); return; }
    // Armed first, on purpose: whatever this connection does — throw in the
    // constructor, hang behind a proxy that never answers, get reset mid-handshake
    // — the deadline is already in place, so there is no path where the page is
    // left waiting on a socket. A spinner with no deadline is the bug report this
    // whole mode exists to avoid.
    this.armProbe();
    this.say(`probing ${this.url}…`);
    try { this.ws = new WebSocket(this.url); } catch (err) { this.say(`no WebSocket here (${err?.message || err}) — running the match in this tab`); return this.enableLocal(`WebSocket unavailable: ${err?.message || err}`); }
    this.ws.onopen = () => {
      this.open = true; this.tries = 0; this.mode = 'server';
      if (this._probe) { clearTimeout(this._probe); this._probe = null; }
      this.diag.openedAt = Date.now();
      this.h.onMode?.('server', `connected to ${this.url}`, null, this.diag);
      this.h.onOpen?.();
      if (this.pending) { const p = this.pending; this.pending = null; this.send(p); }
    };
    this.ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.handle(msg);
    };
    this.ws.onclose = (ev) => {
      this.open = false;
      this.h.onClose?.(ev);
      // a refused or instantly-closed socket is a *fact*, not a hiccup: no point
      // sitting out the probe or backing off when the answer is already "nobody
      // is listening here" (a proxied preview with no /ws support lands here)
      if (this.mode === 'connecting') this.enableLocal(`nothing is listening on ${this.url}`);
      else if (this.mode !== 'local') this.retry('closed');
    };
    this.ws.onerror = () => { try { this.ws.close(); } catch { /* ignore */ } };
  }

  /** The deadline after which this tab starts simulating for itself. */
  armProbe() {
    if (this._probe || this.probeMs <= 0) return;
    this._probe = setTimeout(() => {
      this._probe = null;
      if (!this.open) this.enableLocal(`nothing answered ${this.url} in ${this.probeMs} ms`);
    }, this.probeMs);
  }

  say(txt) { try { this.h.onStatus?.(txt); } catch { /* the status line is never worth a crash */ } }

  /** Both paths speak the same JSON, so both go through here. */
  handle(msg) {
    if (msg.t === 'pong') { this.lastPong = performance.now(); this.h.onPong?.(msg); return; }
    this.h.onMessage?.(msg);
  }

  /**
   * Boot the in-tab engine. `localLoader` is injectable so the fallback can be
   * tested without a browser; by default it pulls in the whole AI tree lazily,
   * which a server-connected player never pays for.
   */
  async enableLocal(reason) {
    if (this.mode === 'local' || this.mode === 'localizing') return;
    // 'localizing' first: if the boot itself throws we must not be stuck in
    // 'connecting' (that is how this page ends up as a permanent spinner), and we
    // must not be re-entered by a late onclose either.
    this.mode = 'localizing';
    this.closed = true;                    // stop the reconnect loop for good
    if (this._probe) { clearTimeout(this._probe); this._probe = null; }
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
    this.say('loading the in-tab engine…');
    const t0 = Date.now();
    try {
      const mod = this.h.localLoader ? await this.h.localLoader() : await import('../../game/local.js');
      this.diag.loadMs = Date.now() - t0;
      const store = safeStorage();
      this.local = new mod.LocalGame({
        config: this.h.localConfig || {},
        storage: store,
        onMessage: (m) => this.handle(m),
      });
      this.local.start();                  // 30 Hz, from now on this tab is the server
      this.open = true;                    // the channel is live, it just lives here
      this.mode = 'local';
      this.diag.bootMs = Date.now() - t0;
      this.diag.reason = reason;
      this.h.onMode?.('local', reason, this.local, this.diag);
      const q = this.queue.splice(0);
      if (this.pending) q.push(this.pending);
      this.pending = null;
      for (const m of q) this.local.send(m);
    } catch (err) {
      // Losing the local engine is not a reason to go quiet: say so, loudly, and
      // hand the socket another try in case it was only the module fetch that failed.
      this.open = false;
      this.mode = 'connecting';
      this.local = null;
      this.diag.error = String(err?.message || err);
      // no auto-retry: `closed` is true, and a page that could not load its own
      // engine is not going to be fixed by hammering it. Say what happened and let
      // the player reload — the reason is on screen either way.
      this.h.onMode?.('failed', `${this.diag.error} (while: ${reason}) — reload to try again`, null, this.diag);
    }
  }

  retry(reason) {
    if (this.closed || this.mode === 'local' || this.mode === 'localizing') return;
    const wait = Math.min(6000, 400 * 2 ** this.tries++);
    this.h.onRetry?.(wait, reason);
    setTimeout(() => this.connect(), wait);
  }

  send(obj) {
    if (this.mode === 'local') {
      if (this.local) this.local.send(obj);
      else this.queue.push(obj);           // joining before the engine finished loading
      return;
    }
    const s = JSON.stringify(obj);
    if (this.open && this.ws && this.ws.readyState === 1) this.ws.send(s);
    else this.pending = obj;   // only the newest message matters for input
  }

  ping() { this.send({ t: 'ping', at: performance.now() }); }

  close() {
    this.closed = true;
    if (this._probe) clearTimeout(this._probe);
    this.local?.stop();
    try { this.ws?.close(); } catch { /* ignore */ }
  }
}

/**
 * `localStorage` is not "may be missing", it is "may throw": in a sandboxed or
 * cross-site iframe (an embedded preview, for instance) touching the getter raises
 * SecurityError, and that used to abort the whole local boot *after* the mode had
 * already flipped to 'local' — which left the page sitting on "connecting…" with
 * nothing but a rejected promise in the console. Storage is a bonus, never a
 * dependency: no localStorage, no persistence, the match still runs.
 */
export function safeStorage() {
  try {
    if (typeof globalThis.localStorage === 'undefined' || !globalThis.localStorage) return null;
    const probe = '__gbtl_probe__';
    globalThis.localStorage.setItem(probe, '1');
    globalThis.localStorage.removeItem(probe);
    return globalThis.localStorage;
  } catch { return null; }
}

function defaultUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws`;
}

/**
 * Batches per-frame input into one message. Look deltas accumulate; movement is
 * last-write-wins; button presses are edge-counted so a fast click is never lost.
 */
export class InputPump {
  constructor(net, rateHz = 30) {
    this.net = net;
    this.rate = 1000 / rateHz;
    this.acc = 0;
    this.dx = 0; this.dy = 0;
    this.mx = 0; this.mz = 0; this.sprint = 0;
    this.jump = false; this.place = false; this.brk = false;
    this.select = null; this.cycle = null;
  }

  look(dx, dy) { this.dx += dx; this.dy += dy; }
  move(mx, mz, sprint) { this.mx = mx; this.mz = mz; this.sprint = sprint ? 1 : 0; }
  press(what) {
    if (what === 'jump') this.jump = true;
    else if (what === 'place') this.place = true;
    else if (what === 'break') this.brk = true;
  }
  pickSlot(i) { this.select = i; }
  cycleSlots(d) { this.cycle = (this.cycle ?? 0) + d; }

  tick(dtMs) {
    this.acc += dtMs;
    if (this.acc < this.rate) return;
    this.acc = 0;
    const msg = { t: 'input' };
    if (this.dx || this.dy) msg.look = [this.dx, this.dy];
    // movement is state, not an event: always send it, or a released key would
    // stay "pressed" on the server and the player would walk forever
    msg.move = [this.mx, this.mz];
    msg.sprint = !!this.sprint;
    if (this.jump) msg.jump = true;
    if (this.place) msg.place = true;
    if (this.brk) msg['break'] = true;
    if (this.select != null) msg.select = this.select;
    if (this.cycle != null) msg.cycle = this.cycle;
    this.dx = this.dy = this.jump = this.place = this.brk = this.select = this.cycle = null;
    if (Object.keys(msg).length > 1) this.net.send(msg);
  }
}
