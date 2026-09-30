/**
 * WebSocket client with reconnect + a coalesced input sender.
 *
 * The browser sends at most one `input` message per animation frame (movement
 * axis, look delta, discrete buttons); the server sends a 20 Hz `state` message
 * with the world deltas folded in. Everything is JSON — at this scale the CPU
 * cost of JSON is irrelevant and the code stays readable in devtools.
 */

export class Net {
  constructor(handlers = {}) {
    this.h = handlers;
    this.url = handlers.url || defaultUrl();
    this.ws = null;
    this.open = false;
    this.tries = 0;
    this.lastPong = 0;
    this.pending = null;
    this.closed = false;
  }

  connect() {
    if (this.closed) return;
    try { this.ws = new WebSocket(this.url); } catch (err) { return this.retry(err); }
    this.ws.onopen = () => {
      this.open = true; this.tries = 0;
      this.h.onOpen?.();
      if (this.pending) { this.send(this.pending); this.pending = null; }
    };
    this.ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.t === 'pong') { this.lastPong = performance.now(); this.h.onPong?.(msg); return; }
      this.h.onMessage?.(msg);
    };
    this.ws.onclose = () => { this.open = false; this.h.onClose?.(); this.retry('closed'); };
    this.ws.onerror = () => { try { this.ws.close(); } catch { /* ignore */ } };
  }

  retry(reason) {
    if (this.closed) return;
    const wait = Math.min(6000, 400 * 2 ** this.tries++);
    this.h.onRetry?.(wait, reason);
    setTimeout(() => this.connect(), wait);
  }

  send(obj) {
    const s = JSON.stringify(obj);
    if (this.open && this.ws.readyState === 1) this.ws.send(s);
    else this.pending = obj;   // only the newest message matters for input
  }

  ping() { this.send({ t: 'ping', at: performance.now() }); }

  close() { this.closed = true; try { this.ws?.close(); } catch { /* ignore */ } }
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
