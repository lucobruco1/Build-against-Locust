/**
 * Minimal WebSocket server (RFC 6455), no dependencies.
 *
 * Only what the game needs: text frames, ping/pong keepalive, close handshake,
 * per-socket backpressure guard. Written by hand so `npm start` works with zero
 * install steps.
 */

import { createHash } from 'node:crypto';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const OP_TEXT = 1, OP_BINARY = 2, OP_CLOSE = 8, OP_PING = 9, OP_PONG = 10;
const MAX_FRAME = 4 << 20;

export class WebSocketHub {
  constructor(server, opts = {}) {
    this.onConnect = opts.onConnect || (() => {});
    this.onMessage = opts.onMessage || (() => {});
    this.onDisconnect = opts.onDisconnect || (() => {});
    this.clients = new Set();
    this.path = opts.path || '/ws';
    this.keepaliveMs = opts.keepaliveMs ?? 25000;
    server.on('upgrade', (req, socket, head) => this._upgrade(req, socket, head));
    this._keepalive = setInterval(() => {
      for (const c of this.clients) c.sendRaw(Buffer.alloc(0), OP_PING);
    }, this.keepaliveMs);
    this._keepalive.unref?.();
  }

  _upgrade(req, socket, head) {
    const url = (req.url || '/').split('?')[0];
    const key = req.headers['sec-websocket-key'];
    if (url !== this.path || req.headers.upgrade?.toLowerCase() !== 'websocket' || !key) {
      socket.destroy();
      return;
    }
    const accept = createHash('sha1').update(key + GUID).digest('base64');
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n'
      + 'Upgrade: websocket\r\n'
      + 'Connection: Upgrade\r\n'
      + `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    socket.setNoDelay(true);
    const client = new Client(socket, head, req, this);
    this.clients.add(client);
    try { this.onConnect(client, req); } catch (err) { client.close(1011, String(err?.message || err)); }
  }

  send(obj) {
    const s = JSON.stringify(obj);
    for (const c of this.clients) c.send(s);
  }

  to(fn, obj) {
    const s = JSON.stringify(obj);
    for (const c of this.clients) if (fn(c)) c.send(s);
  }

  closeAll(code = 1001, reason = 'server closing') {
    for (const c of [...this.clients]) c.close(code, reason);
    clearInterval(this._keepalive);
  }
}

class Client {
  constructor(socket, head, req, hub) {
    this.hub = hub;
    this.socket = socket;
    this.req = req;
    this.alive = true;
    this._gone = false;
    this.id = ++CLIENT_ID;
    this.data = {};               // per-connection scratch (playerId, name…)
    this.pending = [];            // fragments of a fragmented message
    this.buffer = head && head.length ? Buffer.from(head) : Buffer.alloc(0);
    socket.on('data', (d) => this._onData(d));
    socket.on('error', () => this._die());
    socket.on('close', () => this._die());
    socket.on('end', () => this._die());
  }

  _onData(chunk) {
    this.buffer = this.buffer.length ? Buffer.concat([this.buffer, chunk]) : chunk;
    while (this.alive) {
      const frame = readFrame(this.buffer);
      if (frame === null) break;                      // need more bytes
      this.buffer = this.buffer.subarray(frame.size);
      this._handle(frame);
    }
    if (this.buffer.length > MAX_FRAME) this._die();  // runaway/garbage stream
  }

  _handle(f) {
    if (f.opcode === OP_CLOSE) { this.close(1000, ''); return; }
    if (f.opcode === OP_PING) { this.sendRaw(f.payload, OP_PONG); return; }
    if (f.opcode === OP_PONG) return;
    if (f.opcode !== OP_TEXT && f.opcode !== OP_BINARY) { this.close(1003, 'opcode'); return; }
    let payload = f.payload;
    if (!f.fin) {                                      // fragmented: buffer it
      this.pending.push(payload);
      if (this.pending.length > 16) return this.close(1009, 'too many fragments');
      return;
    }
    if (this.pending.length) { this.pending.push(payload); payload = Buffer.concat(this.pending); this.pending = []; }
    if (payload.length > MAX_FRAME) return this.close(1009, 'too big');
    let obj = null;
    try { obj = JSON.parse(payload.toString('utf8')); } catch { return; }
    if (!obj || typeof obj !== 'object') return;
    try { this.hub.onMessage(this, obj); } catch (err) { console.error('[ws] handler error', err); }
  }

  send(text) { this.sendRaw(Buffer.from(text, 'utf8'), OP_TEXT); }

  sendRaw(payload, opcode) {
    if (this._gone || this.socket.destroyed) return;
    if (this.socket.writableLength > 8 << 20) return;   // slow client: drop, never grow
    try { this.socket.write(encodeFrame(opcode, payload)); } catch { this._die(); }
  }

  close(code = 1000, reason = '') {
    if (this._gone) return;
    const body = Buffer.alloc(2 + Buffer.byteLength(reason));
    body.writeUInt16BE(code, 0);
    body.write(reason || '', 2, 'utf8');
    try {
      this.socket.write(encodeFrame(OP_CLOSE, body));
      this.socket.end();
    } catch { /* already gone */ }
    this._die();
  }

  _die() {
    if (this._gone) return;
    this._gone = true;
    this.alive = false;
    this.hub.clients.delete(this);
    try { this.hub.onDisconnect(this); } catch { /* ignore */ }
    this.socket.destroy();
  }
}

let CLIENT_ID = 0;

/* ------------------------------------------------------------------ frames */

function readFrame(buf) {
  if (buf.length < 2) return null;
  const b0 = buf[0], b1 = buf[1];
  const fin = (b0 & 0x80) !== 0;
  const opcode = b0 & 0x0f;
  const masked = (b1 & 0x80) !== 0;
  let len = b1 & 0x7f;
  let off = 2;
  if (len === 126) {
    if (buf.length < off + 2) return null;
    len = buf.readUInt16BE(off); off += 2;
  } else if (len === 127) {
    if (buf.length < off + 8) return null;
    const hi = buf.readUInt32BE(off), lo = buf.readUInt32BE(off + 4);
    if (hi > 0 || lo > MAX_FRAME) return { fin, opcode, masked, payload: Buffer.alloc(0), size: buf.length, oversized: true };
    len = lo; off += 8;
  }
  let mask = null;
  if (masked) {
    if (buf.length < off + 4) return null;
    mask = buf.subarray(off, off + 4); off += 4;
  }
  if (buf.length < off + len) return null;
  const payload = Buffer.allocUnsafe(len);
  buf.copy(payload, 0, off, off + len);
  if (mask) for (let i = 0; i < len; i++) payload[i] ^= mask[i & 3];
  return { fin, opcode, masked, payload, size: off + len };
}

function encodeFrame(opcode, payload) {
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.allocUnsafe(2);
    header[1] = len;
  } else if (len < 65536) {
    header = Buffer.allocUnsafe(4);
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.allocUnsafe(10);
    header[1] = 127;
    header.writeUInt32BE(0, 2);
    header.writeUInt32BE(len, 6);
  }
  header[0] = 0x80 | opcode;
  return Buffer.concat([header, payload]);
}

export default WebSocketHub;
