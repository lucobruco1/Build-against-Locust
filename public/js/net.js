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
    this.mode = 'connecting';   // 'connecting' | 'server' | 'local'
    this.local = null;
    this.queue = [];            // messages sent before the local engine booted
    this._probe = null;
  }

  connect() {
    if (this.closed) return;
    // file:// has no origin to dial. Modules are refused there by the browser, so
    // normally we never get this far — but if a bundler or a same-origin shim did
    // load the page, sitting through a probe timeout would be pure rudeness.
    if (typeof location !== 'undefined' && location.protocol === 'file:') { this.enableLocal('opened straight from disk'); return; }
    try { this.ws = new WebSocket(this.url); } catch (err) { return this.retry(err); }
    this.ws.onopen = () => {
      if (this._probe) { clearTimeout(this._probe); this._probe = null; }
      this.open = true; this.tries = 0; this.mode = 'server';
      this.h.onOpen?.();
      if (this.pending) { const p = this.pending; this.pending = null; this.send(p); }
    };
    this.ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.handle(msg);
    };
    this.ws.onclose = () => {
      this.open = false;
      this.h.onClose?.();
      if (this.mode !== 'local') this.retry('closed');
    };
    this.ws.onerror = () => { try { this.ws.close(); } catch { /* ignore */ } };
    // A WebSocket to a dead port can take a while to fail (or hang forever behind a
    // proxy), so the decision is time-boxed rather than left to onerror.
    this._probe = setTimeout(() => {
      this._probe = null;
      if (!this.open) this.enableLocal(`nothing answered ${this.url}`);
    }, this.probeMs);
  }

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
    if (this.mode === 'local') return;
    this.mode = 'local';
    this.closed = true;                    // stop the reconnect loop for good
    this.open = true;                      // the channel is live, just in-tab
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
    let mod;
    try {
      mod = this.h.localLoader ? await this.h.localLoader() : await import('../../game/local.js');
    } catch (err) {
      this.open = false; this.mode = 'connecting';
      this.h.onMode?.('failed', `could not load game/local.js: ${err?.message || err}`);
      return;
    }
    const store = typeof localStorage === 'undefined' ? null : localStorage;
    this.local = new mod.LocalGame({
      config: this.h.localConfig || {},
      storage: store,
      onMessage: (m) => this.handle(m),
    });
    this.local.start();
    this.h.onMode?.('local', reason, this.local);
    const q = this.queue.splice(0);
    if (this.pending) q.push(this.pending);
    this.pending = null;
    for (const m of q) this.local.send(m);
  }

  retry(reason) {
    if (this.closed || this.mode === 'local') return;
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
