/**
 * core/world.js
 * ---------------------------------------------------------------------------
 * The voxel world: a dense Uint8Array of block ids plus a Uint16Array of
 * *ownership* tags (who placed the block). Ownership is what makes the rule
 * "a bot may only delete blocks it placed itself" enforceable, and it is also
 * what the base-security scorer and the reward function look at.
 *
 * The world also keeps two derived caches that are updated incrementally on
 * every edit because the renderer, the AI observations and the scoring all
 * hammer them:
 *    heightMap  – highest non-transparent block per column (skylight proxy)
 *    columnCount– number of solid blocks per column (cheap "how busy is this
 *                 plot" statistic)
 * Dirty chunk flags let the mesher rebuild only what changed.
 * ---------------------------------------------------------------------------
 */

import {
  WORLD, CHUNKS_X, CHUNKS_Z, B, BLOCK_DEFS, BASE, idx3, unidx, insideWorld, isSolid, isOpaque,
} from '../shared/rules.js';

export const OWNER_NATURAL = 0; // untouched terrain
export const OWNER_PLAYER = 1; // the human
export const OWNER_BOT0 = 2;   // bots are OWNER_BOT0 + botIndex
export const OWNER_LOCUST = 30; // blocks the Locust "opened" (holes) – cosmetic tag

export function ownerForBuilder(index, isHuman) {
  return isHuman ? OWNER_PLAYER : OWNER_BOT0 + index;
}

export class VoxelWorld {
  constructor(seed = 1337, generator = null) {
    this.sx = WORLD.SX; this.sy = WORLD.SY; this.sz = WORLD.SZ;
    this.n = this.sx * this.sy * this.sz;
    this.blocks = new Uint8Array(this.n);
    this.owner = new Uint16Array(this.n);
    this.placedAt = new Float64Array(this.n); // ms timestamp (for "recently placed" features)
    this.heightMap = new Uint8Array(this.sx * this.sz);
    // highest *natural* terrain block per column (placement never changes it) —
    // navigation uses it as the "no digging" floor
    this.terrainY = new Uint8Array(this.sx * this.sz);
    this.columnSolid = new Uint16Array(this.sx * this.sz);
    this.light = new Float32Array(this.n); // 0..1 baked sky/occlusion light
    this.dirty = new Set();
    this.seed = seed >>> 0;
    this.revision = 0;
    this.editCount = 0;
    if (generator) generator(this, seed >>> 0);
    else this.rebuildCaches();
  }

  /* ------------------------------------------------------------- indexing */
  index(x, y, z) {
    return (y * this.sz + z) * this.sx + x;
  }
  inside(x, y, z) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.sx && y < this.sy && z < this.sz;
  }
  colIndex(x, z) {
    return z * this.sx + x;
  }

  /* --------------------------------------------------------------- getters */
  get(x, y, z) {
    if (x < 0 || z < 0 || y < 0 || x >= this.sx || z >= this.sz || y >= this.sy) return B.BEDROCK;
    return this.blocks[this.index(x, y, z)];
  }
  /** Out-of-bounds reads return air (so meshing can reach across edges). */
  getSoft(x, y, z) {
    if (x < 0 || z < 0 || y < 0 || x >= this.sx || z >= this.sz || y >= this.sy) return B.AIR;
    return this.blocks[this.index(x, y, z)];
  }
  getOwner(x, y, z) {
    if (!this.inside(x, y, z)) return OWNER_NATURAL;
    return this.owner[this.index(x, y, z)];
  }
  solid(x, y, z) {
    return isSolid(this.get(x, y, z));
  }
  blocksAt(x, y, z) {
    return this.get(x, y, z);
  }
  surfaceY(x, z) {
    if (x < 0 || z < 0 || x >= this.sx || z >= this.sz) return 1;
    return this.heightMap[this.colIndex(x, z)];
  }

  /* ---------------------------------------------------------------- setters */
  set(x, y, z, id, owner = OWNER_NATURAL, now = 0) {
    if (!this.inside(x, y, z)) return false;
    const i = this.index(x, y, z);
    if (this.blocks[i] === id && this.owner[i] === owner) return false;
    this.blocks[i] = id;
    this.owner[i] = id === B.AIR ? OWNER_NATURAL : owner;
    this.placedAt[i] = id === B.AIR ? 0 : now;
    this.editCount++;
    this.revision++;
    this.markDirtyAt(x, y, z);
    this.updateColumn(x, z);
    this.updateLightColumn(x, z);
    return true;
  }

  place(x, y, z, id, owner, now = 0) {
    if (!this.inside(x, y, z)) return { ok: false, reason: 'outside' };
    if (this.get(x, y, z) !== B.AIR) return { ok: false, reason: 'occupied' };
    this.set(x, y, z, id, owner, now);
    return { ok: true };
  }

  /**
   * The ownership rule of the game.
   *  - the Locust may smash anything except bedrock / water
   *  - a bot may only remove a block it placed itself
   *  - the player may mine natural terrain and their own blocks (not others')
   */
  canBreak(x, y, z, who) {
    if (!this.inside(x, y, z)) return false;
    const id = this.get(x, y, z);
    const d = BLOCK_DEFS[id];
    if (!d || !d.solid || !Number.isFinite(d.hardness)) return false;
    if (who.kind === 'locust') return true;
    const own = this.getOwner(x, y, z);
    // the rule of the game: terrain is free to mine, but a *placed* block only
    // ever answers to the one who placed it — nobody un-builds anybody else
    if (own === OWNER_NATURAL) return true;
    return own === who.owner;
  }

  breakBlock(x, y, z, who, now = 0) {
    if (!this.canBreak(x, y, z, who)) return { ok: false, reason: 'forbidden' };
    const id = this.get(x, y, z);
    this.set(x, y, z, B.AIR, OWNER_NATURAL, now);
    return { ok: true, id };
  }

  /* ------------------------------------------------------- derived caches */
  rebuildCaches(keepTerrain = false) {
    if (!keepTerrain) this.snapshotTerrain();
    for (let z = 0; z < this.sz; z++) {
      for (let x = 0; x < this.sx; x++) {
        this.updateColumn(x, z);
        this.updateLightColumn(x, z);
      }
    }
    this.dirty.clear();
    for (let cz = 0; cz < CHUNKS_Z; cz++) {
      for (let cx = 0; cx < CHUNKS_X; cx++) this.dirty.add(cx + ',' + cz);
    }
  }

  updateColumn(x, z) {
    const ci = this.colIndex(x, z);
    let h = 0, count = 0;
    for (let y = this.sy - 1; y >= 0; y--) {
      const id = this.blocks[this.index(x, y, z)];
      if (isOpaque(id) || isSolid(id)) {
        if (h === 0 && !isOpaque(id)) { /* transparent still caps light below */ }
      }
      if (isOpaque(id)) { h = y + 1; break; }
    }
    for (let y = 0; y < this.sy; y++) {
      if (isSolid(this.blocks[this.index(x, y, z)])) count++;
    }
    // heightMap = y of first block a walking entity stands on, from the top
    let top = 0;
    for (let y = this.sy - 1; y >= 1; y--) {
      if (isSolid(this.blocks[this.index(x, y, z)])) { top = y + 1; break; }
    }
    this.heightMap[ci] = Math.max(top, h > 0 ? h : 0);
    this.columnSolid[ci] = Math.min(65535, count);
  }

  /** Cheap top-down skylight: linear falloff with depth under the surface. */
  updateLightColumn(x, z) {
    const surf = this.heightMap[this.colIndex(x, z)];
    for (let y = 0; y < this.sy; y++) {
      const i = this.index(x, y, z);
      if (y >= surf) { this.light[i] = 1; continue; }
      const d = surf - y;
      let l = Math.max(0.14, 1 - d * 0.16);
      // extra dimming inside fully enclosed cells
      let sealed = 0;
      if (isOpaque(this.getSoft(x - 1, y, z))) sealed++;
      if (isOpaque(this.getSoft(x + 1, y, z))) sealed++;
      if (isOpaque(this.getSoft(x, y, z - 1))) sealed++;
      if (isOpaque(this.getSoft(x, y, z + 1))) sealed++;
      if (sealed >= 3) l *= 0.55;
      this.light[i] = l;
    }
  }

  terrainSurface(x, z) {
    if (x < 0 || z < 0 || x >= this.sx || z >= this.sz) return 1;
    return this.terrainY[z * this.sx + x] || this.heightMap[z * this.sx + x];
  }

  lightAt(x, y, z) {
    if (!this.inside(x, y, z)) return 1;
    return this.light[this.index(x, y, z)];
  }

  /** remember the terrain as it was right after generation */
  snapshotTerrain() {
    this.terrainY.set(this.heightMap);
  }

  markDirtyAt(x, y, z) {
    const cx = Math.floor(x / WORLD.CHUNK), cz = Math.floor(z / WORLD.CHUNK);
    const add = (a, b) => {
      if (a >= 0 && b >= 0 && a < CHUNKS_X && b < CHUNKS_Z) this.dirty.add(a + ',' + b);
    };
    add(cx, cz);
    const lx = x - cx * WORLD.CHUNK, lz = z - cz * WORLD.CHUNK;
    if (lx <= 0) add(cx - 1, cz);
    if (lx >= WORLD.CHUNK - 1) add(cx + 1, cz);
    if (lz <= 0) add(cx, cz - 1);
    if (lz >= WORLD.CHUNK - 1) add(cx, cz + 1);
  }
  takeDirty() {
    const out = [...this.dirty];
    this.dirty.clear();
    return out;
  }

  /* ------------------------------------------------- AI / scoring helpers */

  /**
   * Base-security score of a plot: how much of the shell around `centre` the
   * owner has actually built. This single function is the objective of every
   * builder — the reward that the value head is trained on, the "secure base"
   * percentage in the HUD, and what the Locust exploits when it is < 1.
   */
  security(centre, groundY = null, shell = null) {
    const gy = groundY ?? centre.groundY ?? centre.y ?? this.surfaceY(centre.x, centre.z);
    const s = shell || makeShell(centre, gy);
    let wallHit = 0, outerHit = 0, roofHit = 0, own = 0, breaches = 0;
    for (const c of s.wall) {
      if (isSolid(this.get(c.x, c.y, c.z))) wallHit++;
      else breaches++;
    }
    for (const c of s.outer) if (isSolid(this.get(c.x, c.y, c.z))) outerHit++;
    for (const c of s.roof) if (isSolid(this.get(c.x, c.y, c.z))) roofHit++;
    for (const c of s.all) {
      if (this.getOwner(c.x, c.y, c.z) > OWNER_NATURAL && isSolid(this.get(c.x, c.y, c.z))) own++;
    }
    const wall = s.wall.length ? wallHit / s.wall.length : 0;
    const roof = s.roof.length ? roofHit / s.roof.length : 0;
    const outer = s.outer.length ? outerHit / s.outer.length : 0;
    return {
      wall, roof, outer,
      breaches,
      blocks: own,
      wallCells: s.wall.length,
      roofCells: s.roof.length,
      outerCells: s.outer.length,
      total: clamp01(wall * 0.6 + Math.min(1, roof / (BASE.GOAL_ROOF || 0.45)) * 0.25 + outer * 0.15),
    };
  }

  /** Is `pos` currently covered by at least `depth` solid layers above it? */
  coverAt(x, y, z, depth = 2) {
    let n = 0;
    for (let d = 1; d <= depth; d++) {
      if (isOpaque(this.getSoft(x, y + d, z))) n++;
    }
    return n;
  }

  /** Solid neighbours of a cell (used for "wall thickness" & meshing AO). */
  solidNeighbours(x, y, z) {
    let n = 0;
    if (isSolid(this.getSoft(x - 1, y, z))) n++;
    if (isSolid(this.getSoft(x + 1, y, z))) n++;
    if (isSolid(this.getSoft(x, y, z - 1))) n++;
    if (isSolid(this.getSoft(x, y, z + 1))) n++;
    if (isSolid(this.getSoft(x, y - 1, z))) n++;
    if (isSolid(this.getSoft(x, y + 1, z))) n++;
    return n;
  }

  /** Destructive: pour a sphere of blocks – used by tests and the sim tools. */
  fillSphere(cx, cy, cz, r, id, owner = OWNER_NATURAL) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dz = -r; dz <= r; dz++) {
          if (dx * dx + dy * dy + dz * dz > r * r) continue;
          this.set(cx + dx, cy + dy, cz + dz, id, owner, 0);
        }
      }
    }
  }

  countOwnerBlocks(owner) {
    let n = 0;
    const o = this.owner;
    for (let i = 0; i < o.length; i++) if (o[i] === owner) n++;
    return n;
  }

  serialize() {
    return {
      sx: this.sx, sy: this.sy, sz: this.sz, seed: this.seed, revision: this.revision,
    };
  }
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * What counts as "a secure base" for a plot, as a flat list of cells:
 *   wall  – the scored shell: chebyshev ring WALL_RING, ground .. ground+WALL_TOP
 *   outer – a bonus second layer (ring OUTER_RING), scored but not required
 *   roof  – a (2·ROOF_EXTENT+1)² slab at ground+ROOF_CLEAR
 */
export function makeShell(centre, groundY) {
  const wall = [], outer = [], roof = [], all = [];
  const gy = groundY ?? centre.y ?? 0;
  const pushRing = (arr, r, ownerTag) => {
    for (let dx = -r; dx <= r; dx++) {
      for (let dz = -r; dz <= r; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
        for (let dy = 0; dy < BASE.WALL_TOP; dy++) {
          const c = { x: centre.x + dx, y: gy + dy, z: centre.z + dz, ring: ownerTag };
          arr.push(c);
          all.push(c);
        }
      }
    }
  };
  pushRing(wall, BASE.WALL_RING, 'wall');
  pushRing(outer, BASE.OUTER_RING, 'outer');
  for (let dx = -BASE.ROOF_EXTENT; dx <= BASE.ROOF_EXTENT; dx++) {
    for (let dz = -BASE.ROOF_EXTENT; dz <= BASE.ROOF_EXTENT; dz++) {
      const c = { x: centre.x + dx, y: gy + BASE.ROOF_CLEAR, z: centre.z + dz, ring: 'roof' };
      roof.push(c);
      all.push(c);
    }
  }
  return { wall, outer, roof, all, groundY: gy, radius: BASE.RADIUS };
}

export { idx3, unidx, insideWorld, CHUNKS_X, CHUNKS_Z, WORLD, B };
