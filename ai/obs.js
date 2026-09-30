/**
 * ai/obs.js
 * ---------------------------------------------------------------------------
 * Observation encoders. EfficientZero is trained on observations, so the
 * encoder matters more here than in a scripted bot: the local voxel scan is
 * given in the agent's *facing* frame (rotation-invariant policy), block
 * ownership is a second channel (a bot may only delete what it placed), and the
 * base-security statistics let the value head actually predict "am I safe".
 *
 * Two representations of the same observation:
 *   • `encode(state)`      → Float32Array, used by MCTS and by the replay
 *   • `pack(state)`/`unpack` → compact record (scan bit-packed, scalars float)
 *     stored in the replay buffer: ~10× smaller than the dense vector, which
 *     is what makes a 100k-transition buffer feasible in a browser tab.
 * ---------------------------------------------------------------------------
 */

import { OBS, B, isSolid, isOpaque, ENTITY, BOT_PLACEABLE, HOTBAR } from '../shared/rules.js';

const SCL = 24; // normalisation scale for block-relative distances

export class ObsCodec {
  /**
   * @param {object} o {rx, rz, layers, dy0, channels, scalarDim, targets}
   */
  constructor(o) {
    this.rx = o.rx; this.rz = o.rz; this.layers = o.layers; this.dy0 = o.dy0;
    this.channels = o.channels;
    this.sx = 2 * this.rx + 1;
    this.sz = 2 * this.rz + 1;
    this.cells = this.sx * this.sz * this.layers;
    this.bits = Math.ceil(this.cells / 8);
    this.scalarDim = o.scalarDim;
    this.dim = this.scalarDim + this.cells * this.channels;
  }

  /** bit-array helpers ---------------------------------------------------- */
  _setBit(arr, i, v) {
    if (v) arr[i >> 3] |= 1 << (i & 7);
  }
  _getBit(arr, i) {
    return (arr[i >> 3] >> (i & 7)) & 1;
  }

  /**
   * @returns {{scalars:Float32Array, planes:Uint8Array[]}} compact record
   */
  pack(state) {
    const f = this._scratch();
    this.encodeInto(state, f);
    const planes = [];
    for (let c = 0; c < this.channels; c++) planes.push(new Uint8Array(this.bits));
    for (let i = 0; i < this.cells * this.channels; i++) {
      if (f[this.scalarDim + i] > 0.5) {
        const c = Math.floor(i / this.cells);
        const j = i - c * this.cells;
        this._setBit(planes[c], j, 1);
      }
    }
    return { scalars: f.slice(0, this.scalarDim), planes };
  }

  unpack(rec, out) {
    const a = out || new Float32Array(this.dim);
    a.set(rec.scalars, 0);
    const base = this.scalarDim;
    for (let c = 0; c < this.channels; c++) {
      const plane = rec.planes[c];
      const off = base + c * this.cells;
      for (let j = 0; j < this.cells; j++) a[off + j] = this._getBit(plane, j);
    }
    return a;
  }

  decode(rec) {
    return this.unpack(rec, new Float32Array(this.dim));
  }

  encode(state) {
    return this.encodeInto(state, new Float32Array(this.dim));
  }

  _scratch() {
    if (!this.__s) this.__s = new Float32Array(this.dim);
    this.__s.fill(0);
    return this.__s;
  }

  /* --------------------------------------------------------- builder view */
  encodeInto(state, out) {
    const w = state.world;
    const e = state.ent;
    let i = 0;
    const yaw = e.yaw || 0;
    const sinY = Math.sin(yaw), cosY = Math.cos(yaw);
    const push = (v) => { out[i++] = Number.isFinite(v) ? v : 0; };

    /* -- scalar block (order documented in builderScalars) ---------------- */
    const sc = state.scalars;
    for (let k = 0; k < this.scalarDim; k++) push(sc ? sc[k] : 0);

    /* -- local voxel scan, in the facing frame --------------------------- */
    const fx = Math.floor(e.pos.x), fy = Math.floor(e.pos.y), fz = Math.floor(e.pos.z);
    const own = state.owner;
    for (let c = 0; c < this.channels; c++) {
      for (let ly = 0; ly < this.layers; ly++) {
        const y = fy + this.dy0 + ly;
        for (let dz = -this.rz; dz <= this.rz; dz++) {
          for (let dx = -this.rx; dx <= this.rx; dx++) {
            // rotate the (dx,dz) sample offset into world space by yaw
            const wx = fx + Math.round(dx * cosY - dz * sinY);
            const wz = fz + Math.round(dx * sinY + dz * cosY);
            let v = 0;
            if (c === 0) v = isSolid(w.get(wx, y, wz)) ? 1 : 0;
            else if (c === 1) v = (own !== undefined && w.getOwner(wx, y, wz) === own && isSolid(w.get(wx, y, wz))) ? 1 : 0;
            else v = isOpaque(w.get(wx, y, wz)) ? 1 : 0;
            out[i++] = v;
          }
        }
      }
    }
    return out;
  }
}

/**
 * Fixed scalar layout for the builders. Kept as data so the HUD can print it
 * and so the encoder stays a pure loop.
 */
export const BUILDER_SCALARS = [
  'sin(yaw)', 'cos(yaw)', 'pitch', 'onGround', 'inWater',
  'health', 'hurtRecent', 'phaseBuild', 'phaseHunt', 'timeLeft', 'locustActive',
  'locustDist', 'locustDx', 'locustDz', 'locustDy', 'locustVisible',
  't0Dist', 't0Dx', 't0Dz', 't0Dy', 't0Alive',
  't1Dist', 't1Dx', 't1Dz', 't1Dy', 't1Alive',
  't2Dist', 't2Dx', 't2Dz', 't2Dy', 't2Alive',
  'baseDx', 'baseDz', 'baseDist', 'atHome',
  'wall', 'roof', 'breaches', 'ownBlocksNear', 'sealed',
  'invTotal', 'inv0', 'inv1', 'inv2', 'inv3', 'inv4',
  'grabbed', 'alliesAlive', 'cycle', 'groundDelta', 'light', 'stuck',
  'yawToBase', 'vx', 'vy', 'vz',
  'frontDist', 'frontSolid', 'canPlace', 'canBreak',
  'score', 'locustHuntsMe', 'blockLoss', 'kills',
];
export const LOCUST_SCALARS = [
  'sin(yaw)', 'cos(yaw)', 'pitch', 'onGround', 'timeLeft', 'phaseHunt',
  'preyAlive', 'hitsLanded', 'blocksSmashed', 'attackCd', 'roarCd',
  't0Dist', 't0Dx', 't0Dz', 't0Dy', 't0Visible', 't0Grabbed',
  't1Dist', 't1Dx', 't1Dz', 't1Dy', 't1Visible', 't1Grabbed',
  't2Dist', 't2Dx', 't2Dz', 't2Dy', 't2Visible', 't2Grabbed',
  'pathDx', 'pathDz', 'pathDy', 'pathBlocked', 'smashDist',
  'nearestWall', 'nearestRoof', 'nearestBreach', 'nearestBaseDist',
  'targetIsHuman', 'velocityX', 'velocityY', 'velocityZ', 'height', 'light',
  'cycle', 'stuck', 'rewards', 'playerAlive', 'anyHiding',
];

export const BUILDER_SCALAR_DIM = BUILDER_SCALARS.length;
export const LOCUST_SCALAR_DIM = LOCUST_SCALARS.length;

export function makeBuilderCodec(cfg = OBS) {
  return new ObsCodec({
    rx: cfg.SCAN_RX, rz: cfg.SCAN_RZ, layers: cfg.SCAN_LAYERS, dy0: cfg.SCAN_DY, channels: 2,
    scalarDim: BUILDER_SCALAR_DIM,
  });
}
export function makeLocustCodec(cfg = OBS) {
  return new ObsCodec({
    rx: 2, rz: 2, layers: 6, dy0: -1, channels: 1,
    scalarDim: LOCUST_SCALAR_DIM,
  });
}

/* ------------------------------------------------------------ state → scalars */

/**
 * Builds the scalar array for a builder from the game's public state.
 * Pure function of `ctx`, so the headless simulator and the browser agree.
 */
export function builderScalars(ctx) {
  const { ent, world, phase, timeLeft, locust, targets, base, security, inventory, grabbed,
    alliesAlive, cycle, stuck, score, owner, blockLoss, hurtRecent, kills } = ctx;
  const s = new Float32Array(BUILDER_SCALAR_DIM);
  let i = 0;
  const yaw = ent.yaw || 0;
  const sinY = Math.sin(yaw), cosY = Math.cos(yaw);
  const rel = (dx, dz) => ({ x: dx * cosY + dz * sinY, z: -dx * sinY + dz * cosY });

  s[i++] = sinY;
  s[i++] = cosY;
  s[i++] = (ent.pitch || 0) / 0.7;
  s[i++] = ent.onGround ? 1 : 0;
  s[i++] = ent.inWater ? 1 : 0;
  s[i++] = (ctx.health ?? 20) / 20;
  s[i++] = hurtRecent ? 1 : 0;
  s[i++] = phase === 'build' ? 1 : 0;
  s[i++] = phase === 'hunt' ? 1 : 0;
  s[i++] = timeLeft ?? 0;
  s[i++] = locust && locust.active ? 1 : 0;

  if (locust && locust.active) {
    const dx = locust.pos.x - ent.pos.x, dz = locust.pos.z - ent.pos.z, dy = locust.pos.y - ent.pos.y;
    const d = Math.hypot(dx, dy, dz);
    s[i++] = 1 / (1 + d / 12);
    const r = rel(dx, dz);
    s[i++] = r.x / SCL;
    s[i++] = r.z / SCL;
    s[i++] = dy / 12;
    s[i++] = locust.visible ? 1 : 0;
  } else {
    i += 5;
  }

  for (let t = 0; t < 3; t++) {
    const o = targets && targets[t];
    if (o) {
      const dx = o.x - ent.pos.x, dz = o.z - ent.pos.z, dy = (o.y ?? 0) - ent.pos.y;
      const d = Math.hypot(dx, dy, dz);
      const r = rel(dx, dz);
      s[i++] = 1 / (1 + d / 8);
      s[i++] = r.x / SCL;
      s[i++] = r.z / SCL;
      s[i++] = dy / 8;
      s[i++] = 1;
    } else i += 5;
  }

  if (base) {
    const dx = base.x - ent.pos.x, dz = base.z - ent.pos.z;
    const r = rel(dx, dz);
    s[i++] = r.x / SCL;
    s[i++] = r.z / SCL;
    s[i++] = 1 / (1 + Math.hypot(dx, dz) / 8);
    s[i++] = Math.abs(dx) <= 3 && Math.abs(dz) <= 3 ? 1 : 0;
  } else i += 4;

  const sec = security || { wall: 0, roof: 0, breaches: 0, blocks: 0, sealed: 0 };
  s[i++] = sec.wall ?? 0;
  s[i++] = sec.roof ?? 0;
  s[i++] = Math.min(1, (sec.breaches ?? 0) / 40);
  s[i++] = Math.min(1, (sec.blocks ?? 0) / 120);
  s[i++] = sec.sealed ? 1 : 0;

  const inv = inventory || { total: 0, counts: {} };
  s[i++] = Math.min(1, (inv.total || 0) / 128);
  const types = ctx.hotbar || HOTBAR;
  for (let k = 0; k < 5; k++) {
    const t = types[k];
    s[i++] = Math.min(1, ((inv.counts && inv.counts[t]) || 0) / 64);
  }

  s[i++] = grabbed ? 1 : 0;
  s[i++] = (alliesAlive ?? 0) / 7;
  s[i++] = (cycle ?? 0) / 10;
  s[i++] = world && ent ? (world.surfaceY(Math.floor(ent.pos.x), Math.floor(ent.pos.z)) - ent.pos.y) / 6 : 0;
  s[i++] = world && ent ? world.lightAt(Math.floor(ent.pos.x), Math.floor(ent.pos.y + 1), Math.floor(ent.pos.z)) : 1;
  s[i++] = Math.min(1, (stuck || 0) / 8);
  s[i++] = base ? normAngle(Math.atan2(-(base.x - ent.pos.x), -(base.z - ent.pos.z)) - yaw) / Math.PI : 0;
  s[i++] = (ent.vel?.x || 0) / 6;
  s[i++] = (ent.vel?.y || 0) / 12;
  s[i++] = (ent.vel?.z || 0) / 6;

  const front = ctx.front || {};
  s[i++] = (front.dist ?? 6) / 6;
  s[i++] = front.solid ? 1 : 0;
  s[i++] = front.canPlace ? 1 : 0;
  s[i++] = front.canBreak ? 1 : 0;
  s[i++] = Math.tanh((score ?? 0) / 40);
  s[i++] = locust && locust.huntingMe ? 1 : 0;
  s[i++] = Math.min(1, (blockLoss || 0) / 20);
  s[i++] = Math.min(1, (kills || 0) / 5);
  void owner;
  return s;
}

export function locustScalars(ctx) {
  const { ent, timeLeft, phase, prey, targets, pathInfo, nearestBase, cycle, stuck,
    attackCd, hits, smashed, health, rewards, playerAlive, anyHiding } = ctx;
  const s = new Float32Array(LOCUST_SCALAR_DIM);
  let i = 0;
  const yaw = ent.yaw || 0;
  const sinY = Math.sin(yaw), cosY = Math.cos(yaw);
  const rel = (dx, dz) => ({ x: dx * cosY + dz * sinY, z: -dx * sinY + dz * cosY });
  s[i++] = sinY;
  s[i++] = cosY;
  s[i++] = (ent.pitch || 0) / 0.6;
  s[i++] = ent.onGround ? 1 : 0;
  s[i++] = timeLeft ?? 0;
  s[i++] = phase === 'hunt' ? 1 : 0;
  s[i++] = (prey?.aliveCount ?? 0) / 8;
  s[i++] = Math.min(1, (hits || 0) / 8);
  s[i++] = Math.min(1, (smashed || 0) / 60);
  s[i++] = attackCd ? 1 : 0;
  s[i++] = 0;
  for (let t = 0; t < 3; t++) {
    const o = targets && targets[t];
    if (o) {
      const dx = o.x - ent.pos.x, dz = o.z - ent.pos.z, dy = (o.y ?? 0) - ent.pos.y;
      const d = Math.hypot(dx, dy, dz);
      const r = rel(dx, dz);
      s[i++] = 1 / (1 + d / 14);
      s[i++] = r.x / SCL;
      s[i++] = r.z / SCL;
      s[i++] = dy / 12;
      s[i++] = o.visible ? 1 : 0;
      s[i++] = o.grabbed ? 1 : 0;
    } else i += 6;
  }
  const p = pathInfo || {};
  s[i++] = p.dx ?? 0;
  s[i++] = p.dz ?? 0;
  s[i++] = (p.dy ?? 0) / 3;
  s[i++] = p.blocked ? 1 : 0;
  s[i++] = (p.smashDist ?? 8) / 8;
  const nb = nearestBase || {};
  s[i++] = nb.wall ?? 0;
  s[i++] = nb.roof ?? 0;
  s[i++] = nb.breach ?? 0;
  s[i++] = (nb.dist ?? 30) / 30;
  s[i++] = ctx.targetIsHuman ? 1 : 0;
  s[i++] = (ent.vel?.x || 0) / 8;
  s[i++] = (ent.vel?.y || 0) / 14;
  s[i++] = (ent.vel?.z || 0) / 8;
  s[i++] = (ent.pos?.y || 0) / 40;
  s[i++] = ctx.light ?? 1;
  s[i++] = (cycle ?? 0) / 10;
  s[i++] = Math.min(1, (stuck || 0) / 8);
  s[i++] = Math.tanh((rewards || 0) / 20);
  s[i++] = playerAlive ? 1 : 0;
  s[i++] = anyHiding ? 1 : 0;
  void health;
  return s;
}

function normAngle(a) {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

export { SCL, BOT_PLACEABLE, ENTITY };
