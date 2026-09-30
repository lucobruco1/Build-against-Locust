/**
 * Client entry point.
 *
 * The browser is a *thin* client: the authoritative `Match` (physics, block
 * ownership, the bots' MCTS, the Locust) runs on the server. This file only
 *  - rebuilds the voxel world from the join payload and applies block deltas,
 *  - turns pointer-lock input into `input` messages,
 *  - smooths the 20 Hz state feed, and
 *  - drives the renderer, HUD, banners and audio.
 */

import { placeTarget } from '/core/raycast.js';
import { unpackWorld } from '/game/net.js';
import { B, BLOCK_DEFS, LACT, ENTITY } from '/shared/rules.js';
import { GameScene } from './render/scene.js';
import { Hud } from './hud.js';
import { Net, InputPump } from './net.js';
import { Ambience } from './audio.js';

const SENS = 1.0;
// must match Match.setPlayerLook's 0.0026 rad/px and pitch clamp exactly, or the
// server's aim and the client's crosshair drift apart
const LOOK_K = 0.0026 * SENS;
const PITCH_MIN = -1.35, PITCH_MAX = 1.25;

class Client {
  constructor() {
    this.hud = new Hud();
    this.audio = new Ambience();
    this.world = null;
    this.scene = null;
    this.state = null;
    this.localId = 'player';
    this.joined = false;
    this.keys = new Set();
    this.buttons = { left: false, right: false };
    this.view = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0 };
    this.serverPos = { x: 0, y: 0, z: 0 };
    this.lastStateAt = 0;
    this.lastPhase = null;
    this.lastCycle = 0;
    this.lastPing = 0;
    this.pingMs = 0;
    this.fps = 0;
    this.frames = 0;
    this.fpsT = 0;
    this.prevActors = new Map();
    this.net = new Net({
      onMessage: (m) => this.onMessage(m),
      onOpen: () => this.setStatus('connected. press enter to spawn.'),
      onClose: () => { this.joined = false; this.setStatus('connection lost — retrying…'); },
      onRetry: (ms) => this.setStatus(`reconnecting in ${Math.round(ms / 100) / 10}s…`),
    });
    this.input = new InputPump(this.net);
    this.bindUi();
    this.net.connect();
    requestAnimationFrame((t) => this.frame(t));
    setInterval(() => { if (this.net.open) this.net.ping(); }, 2000);
  }

  setStatus(txt) {
    const el = document.getElementById('menu-status');
    if (el) el.textContent = txt;
  }

  /* ------------------------------------------------------------------ UI */

  bindUi() {
    const enter = document.getElementById('btn-enter');
    const resume = document.getElementById('btn-resume');
    const sliders = [
      ['in-sims', 'out-sims', (v) => `${v.toFixed(1)}×`],
      ['in-assist', 'out-assist', (v) => v.toFixed(2)],
      ['in-time', 'out-time', (v) => `${v.toFixed(2)}×`],
    ];
    for (const [inp, out, fmt] of sliders) {
      const i = document.getElementById(inp), o = document.getElementById(out);
      const sync = () => { if (o) o.textContent = fmt(Number(i.value)); };
      i?.addEventListener('input', sync);
      sync();
    }
    this.sliders = sliders.map(([inp]) => document.getElementById(inp));
    this.checks = ['in-learn', 'in-flat'].map((id) => document.getElementById(id));

    enter?.addEventListener('click', () => this.join());
    resume?.addEventListener('click', () => this.lock());
    document.getElementById('btn-help')?.addEventListener('click', () => this.hud.toggleHelp(true));
    document.getElementById('btn-help-close')?.addEventListener('click', () => this.hud.toggleHelp(false));
    document.getElementById('btn-tally-close')?.addEventListener('click', () => this.hud.hideTally());
    document.getElementById('btn-settings')?.addEventListener('click', () => this.pushConfig());
    document.getElementById('tally')?.addEventListener('click', (e) => { if (e.target.id === 'tally') this.hud.hideTally(); });

    addEventListener('keydown', (e) => this.onKey(e, true));
    addEventListener('keyup', (e) => this.onKey(e, false));
    addEventListener('blur', () => { this.keys.clear(); this.input.move(0, 0, false); });
    const canvas = document.getElementById('gl');
    canvas.addEventListener('mousedown', (e) => {
      if (!this.locked()) { if (this.joined) this.lock(); return; }
      if (e.button === 0) this.buttons.left = true;
      if (e.button === 2) this.buttons.right = true;
      this.hud.setCrosshairActive(true);
    });
    addEventListener('mouseup', (e) => {
      if (e.button === 0) this.buttons.left = false;
      if (e.button === 2) this.buttons.right = false;
      this.hud.setCrosshairActive(false);
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    addEventListener('mousemove', (e) => {
      if (!this.locked()) return;
      this.view.yaw -= e.movementX * LOOK_K;
      this.view.pitch = Math.max(PITCH_MIN, Math.min(PITCH_MAX, this.view.pitch - e.movementY * LOOK_K));
    });
    addEventListener('wheel', (e) => {
      if (!this.joined) return;
      this.input.cycleSlots(e.deltaY > 0 ? 1 : -1);
    }, { passive: true });
    document.addEventListener('pointerlockchange', () => {
      const on = this.locked();
      if (!on && this.joined) {
        this.input.move(0, 0, false);
        document.getElementById('menu')?.classList.remove('hidden');
        resume?.classList.remove('hidden');
      } else if (on) {
        this.hud.hideMenu();
        this.audio.start();
      }
    });
  }

  locked() { return document.pointerLockElement === document.getElementById('gl'); }
  lock() { document.getElementById('gl')?.requestPointerLock?.(); }

  onKey(e, down) {
    const k = e.code;
    if (['Tab', 'Space', 'KeyF', 'Slash'].includes(k)) e.preventDefault();
    if (down) this.keys.add(k); else this.keys.delete(k);
    if (!down) return;
    if (k === 'Space') this.input.press('jump');
    if (k === 'KeyH') this.hud.toggleHelp(document.getElementById('help')?.classList.contains('hidden'));
    if (k === 'KeyF') this.input.press('break');
    if (k === 'Escape') { document.exitPointerLock?.(); this.hud.hideTally(); }
    if (k === 'KeyT') this.pushConfig();
    const n = /^Digit([1-9])$/.exec(k);
    if (n) this.input.pickSlot(Number(n[1]) - 1);
  }

  async pushConfig() {
    const [sims, assist, time] = this.sliders.map((i) => Number(i?.value ?? 1));
    const [learn] = this.checks;
    try {
      await fetch('/api/config', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ sims, assist, timeScale: time, learn: !!learn?.checked }),
      });
      this.hud.banner('settings applied', `sims ${sims}× · prior ${assist} · time ${time}× · train ${learn?.checked ? 'on' : 'off'}`, 'build');
    } catch { this.setStatus('could not reach /api/config'); }
  }

  join() {
    const name = (document.getElementById('in-name')?.value || 'guest').slice(0, 20);
    const [sims, assist, time] = this.sliders.map((i) => Number(i?.value ?? 1));
    const [learn, flat] = this.checks;
    this.net.send({
      t: 'join', name,
      config: { sims, assist, timeScale: time, learn: !!learn?.checked, flat: !!flat?.checked },
    });
    this.joined = true;
    this.hud.hideMenu();
    this.lock();
    this.audio.start();
    this.pushConfig();
  }

  /* ------------------------------------------------------------ messages */

  onMessage(msg) {
    if (msg.t === 'hello') {
      this.applyWorld(msg.world);
      if (msg.state) this.onState(msg.state);
      const btn = document.getElementById('btn-enter');
      if (btn) btn.disabled = false;
      this.setStatus('ready — the sim is running on the server');
    } else if (msg.t === 'joined') {
      if (msg.world) this.applyWorld(msg.world);
      this.setStatus('spawned. WASD to move.');
    } else if (msg.t === 'state') {
      this.onState(msg);
    }
  }

  applyWorld(payload) {
    try {
      const w = unpackWorld(payload);
      if (this.scene) {
        this.scene.setWorld(w);
        this.world = w;
      } else {
        this.world = w;
        this.scene = new GameScene(document.getElementById('gl'), w);
        this.scene.voxels.buildAll();
      }
      this.worldGen = true;
    } catch (err) {
      console.error('world payload failed to decode', err);
      this.setStatus('world decode failed: ' + err.message);
    }
  }

  onState(s) {
    if (s.deltas) for (const d of s.deltas) this.applyDelta(d);
    if (this.state) { this.prevState = this.state; this.prevAt = this.stateAt || performance.now(); }
    this.state = s;
    this.stateAt = performance.now();
    this.lastStateAt = this.stateAt;
    const me = this.me();
    if (me) {
      this.serverPos.x = me.x; this.serverPos.y = me.y; this.serverPos.z = me.z;
      if (!this.locked()) this.view.yaw = me.yaw ?? this.view.yaw;
    }
    // phase / cycle feedback
    if (this.lastPhase && this.lastPhase !== s.phase) this.announce(s);
    this.lastPhase = s.phase;
    if (s.phase === 'revive' && s.tally && s.cycle !== this.lastTallyShown) {
      this.lastTallyShown = s.cycle;
      this.hud.showTally(s.tally, s.locust);
    }
    if (s.phase === 'hunt') this.hud.banner('it is here', 'three minutes. do not be found.', 'hunt');
    else if (s.phase === 'build' && this.lastCycle !== s.cycle) this.hud.banner(`cycle ${s.cycle}`, `${Math.round(s.timeTotal)}s to fortify`, 'build');
    this.lastCycle = s.cycle;
    this.hud.update(s, { localId: this.localId, fps: this.fps, ping: Math.round(this.pingMs) });
  }

  announce(s) {
    if (s.phase === 'revive') {
      this.hud.banner('you wake up', 'the cycle closed. everyone is alive again. build.', 'revive');
      this.audio.click('smash');
    }
  }

  /** Blend the two most recent authoritative states: 20 Hz feed, 60+ Hz render. */
  interp() {
    const cur = this.state;
    if (!cur) return { actors: [], locust: null };
    const prev = this.prevState;
    if (!prev || prev.cycle !== cur.cycle || prev.phase !== cur.phase) return cur;
    const span = Math.max(16, this.stateAt - this.prevAt);
    const a = Math.max(0, Math.min(1.25, (performance.now() - this.stateAt) / span));
    const blend = (pa, pb) => {
      if (!pa || !pb) return pb || pa;
      const o = { ...pb };
      for (const k of ['x', 'y', 'z', 'yaw', 'pitch', 'hp']) {
        if (typeof pb[k] === 'number' && typeof pa[k] === 'number') {
          o[k] = k === 'yaw' ? lerpAngle(pa[k], pb[k], a) : pa[k] + (pb[k] - pa[k]) * a;
        }
      }
      return o;
    };
    const byId = new Map(prev.actors.map((b) => [b.id, b]));
    return {
      ...cur,
      actors: cur.actors.map((b) => blend(byId.get(b.id), b)),
      locust: cur.locust ? blend(prev.locust, cur.locust) : null,
    };
  }

  me() {
    return this.state?.actors.find((a) => a.id === this.localId) || null;
  }

  /** local mirror of the authoritative world, so meshing never waits on RTT */
  applyDelta(d) {
    const w = this.world;
    if (!w) return;
    if (d.t === 'place') {
      w.set(d.x, d.y, d.z, d.id | 0, d.own | 0, Date.now());
    } else if (d.t === 'break') {
      w.set(d.x, d.y, d.z, B.AIR, 0, 0);
      const col = BLOCK_DEFS[d.id]?.color ?? 0x999999;
      this.scene?.burst({ x: d.x, y: d.y, z: d.z }, col, 12);
      this.kickIfNear(d, 0.12);
    } else if (d.t === 'smash') {
      const col = BLOCK_DEFS[d.id]?.color ?? 0x999999;
      this.scene?.burst({ x: d.x, y: d.y, z: d.z }, col, 7);
      this.kickIfNear(d, 0.2);
      if (this.scene?.dread > 0.25) this.audio.click('smash');
    } else if (d.t === 'grab') {
      this.audio.stab();
      if (d.victim === this.localId) this.hud.banner('it has you', 'hold SPACE to struggle', 'hunt');
    } else if (d.t === 'stab') {
      this.audio.stab();
      this.kickIfNear({ x: this.me()?.x, y: this.me()?.y, z: this.me()?.z }, 0.5);
    } else if (d.t === 'death') {
      if (d.by === this.localId) this.hud.showDeath(true, this.state?.timeLeft ?? 5);
    } else if (d.t === 'revive') {
      if (d.by === this.localId) { this.hud.showDeath(false); this.hud.banner('revived', 'again. build faster.', 'revive'); }
    } else if (d.t === 'hurt') {
      if (d.by === this.localId) { this.hud.hurt(); this.kick(0.3); }
    } else if (d.t === 'locustSpawn') {
      this.hud.banner('the hunt begins', 'something tall is in the world', 'hunt');
    } else if (d.t === 'locustDespawn') {
      this.hud.banner('silence', 'it is gone… for now', 'build');
    }
  }

  kickIfNear(d, amt) {
    const me = this.me();
    if (!me || d.x == null) return;
    const dist = Math.hypot(d.x - me.x, d.z - me.z);
    if (dist < 14) this.kick(amt * (1 - dist / 14));
  }

  kick(a) { this.scene?.kick(a); this.shake = (this.shake || 0) + a; }

  /* ------------------------------------------------------------- the loop */

  frame(t) {
    const dt = Math.min(0.05, (t - (this.lastT || t)) / 1000);
    this.lastT = t;
    this.frames++; this.fpsT += dt;
    if (this.fpsT > 0.5) { this.fps = Math.round(this.frames / this.fpsT); this.frames = 0; this.fpsT = 0; }
    const rt = performance.now();
    this.pingMs = this.net.lastPong ? Math.min(999, Math.max(0, rt - this.net.lastPong)) * 0.5 : 0;

    if (this.joined) this.readKeys();
    this.input.tick(dt * 1000);
    this.input.look((this.view.yaw - (this.sentYaw ?? this.view.yaw)) / LOOK_K, (this.view.pitch - (this.sentPitch ?? this.view.pitch)) / LOOK_K);
    this.sentYaw = this.view.yaw; this.sentPitch = this.view.pitch;

    if (this.buttons.left) this.input.press('break');
    if (this.buttons.right) this.input.press('place');

    if (this.scene && this.world) {
      const s = dt * 12;
      this.view.x += (this.serverPos.x - this.view.x) * Math.min(1, s);
      this.view.y += (this.serverPos.y - this.view.y) * Math.min(1, s);
      this.view.z += (this.serverPos.z - this.view.z) * Math.min(1, s);
      const me = this.me();
      const local = { ...this.view, speed: me ? 4 : 0, grabbed: me?.grabbed };
      const eye = { x: this.view.x, y: this.view.y + ENTITY.PLAYER.eye, z: this.view.z };
      const tgt = placeTarget(this.world, eye.x, eye.y, eye.z, this.view.yaw, this.view.pitch, ENTITY.REACH);
      this.scene.localId = this.localId;
      this.scene.syncState(this.interp(), 1);
      const lmode = this.state?.locust ? (this.state.locust.action === LACT.STRIKE ? 'strike'
        : this.state.locust.action === LACT.SMASH_BLOCK ? 'smash'
          : this.state.locust.holding ? 'grab' : 'stalk') : null;
      this.scene.frame(dt, local, { yaw: this.view.yaw, pitch: this.view.pitch }, { ...tgt, mode: lmode });
      if (this.shake) { this.scene.shake = this.shake; this.shake = 0; }
      this.scene.render();

      // dread: proximity of the Locust drives vignette + audio
      const l = this.state?.locust;
      let dread = 0;
      if (l) {
        const d = Math.hypot(l.x - this.view.x, l.z - this.view.z);
        dread = Math.max(0, 1 - d / 30) * (this.state.phase === 'hunt' ? 1 : 0.4);
        if (me?.grabbed) dread = 1;
        if (this.scene.locustMesh) this.scene.locustMesh.setMode(lmode || 'stalk');
      }
      this.hud.setDread(dread);
      this.audio.setDread(dread, this.state?.phase === 'hunt');
      if (me && !me.alive) this.hud.tickDeathTimer(this.state?.timeLeft ?? 0);
      else this.hud.showDeath(false);
    }
    requestAnimationFrame((tt) => this.frame(tt));
  }

  readKeys() {
    const k = this.keys;
    const fwd = (k.has('KeyW') || k.has('ArrowUp') ? -1 : 0) + (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
    const str = (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) + (k.has('KeyA') || k.has('ArrowLeft') ? -1 : 0);
    const sprint = k.has('ShiftLeft') || k.has('ShiftRight');
    this.input.move(str, fwd, sprint);
    if (k.has('Space')) this.input.press('jump');
  }
}

/* The client keeps its own copy of the world purely for rendering and for the aim
   cursor; `set` marks chunks dirty and the mesher rebuilds a few per frame. */

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

const client = new Client();
window.__game = client;
if (!window.__hudReady) { window.__hudReady = true; }
