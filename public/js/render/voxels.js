/**
 * Voxel chunk mesher.
 *
 * One BufferGeometry per CHUNK×CHUNK×CHUNK column of the world, rebuilt only
 * when the world flags that chunk dirty (block placement/breaking). Faces are
 * culled against opaque neighbours, and per-vertex colour = block colour ×
 * face shading × baked light, so the whole thing renders in a single material
 * with no shadow maps — which is what makes a 72×40×72 world cheap.
 */

import * as THREE from 'three';
import { BLOCK_DEFS, B, WORLD, isSolid, isOpaque } from '../../../../shared/rules.js';

const CHUNK = WORLD.CHUNK;

const FACES = [
  // dir, 4 corner offsets (CCW from outside), shade
  { n: [1, 0, 0], v: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]], s: 0.80 },
  { n: [-1, 0, 0], v: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], s: 0.80 },
  { n: [0, 1, 0], v: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]], s: 1.00 },
  { n: [0, -1, 0], v: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], s: 0.42 },
  { n: [0, 0, 1], v: [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]], s: 0.66 },
  { n: [0, 0, -1], v: [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]], s: 0.66 },
];

function colourOf(id) {
  const d = BLOCK_DEFS[id];
  const c = d?.color ?? 0x888888;
  return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
}
const TINT = new Map();
function tintOf(id) {
  if (!TINT.has(id)) TINT.set(id, colourOf(id));
  return TINT.get(id);
}

export class VoxelRenderer {
  constructor(scene, world, opts = {}) {
    this.scene = scene;
    this.world = world;
    this.material = new THREE.MeshLambertMaterial({ vertexColors: true });
    this.waterMaterial = new THREE.MeshLambertMaterial({
      vertexColors: true, transparent: true, opacity: 0.72, depthWrite: false,
    });
    this.meshes = new Map();
    this.water = new Map();
    this.group = new THREE.Group();
    scene.add(this.group);
    this.maxPerFrame = opts.maxPerFrame ?? 6;
    this.queue = new Set();
  }

  key(cx, cy, cz) { return `${cx},${cy},${cz}`; }

  chunkCoords() {
    const { sx, sy, sz } = this.world;
    const nx = Math.ceil(sx / CHUNK), ny = Math.ceil(sy / CHUNK), nz = Math.ceil(sz / CHUNK);
    return { nx, ny, nz };
  }

  buildAll() {
    const { nx, ny, nz } = this.chunkCoords();
    for (let cz = 0; cz < nz; cz++) for (let cy = 0; cy < ny; cy++) for (let cx = 0; cx < nx; cx++) this.build(cx, cy, cz);
  }

  markDirty() {
    for (const k of this.world.dirty) this.queue.add(k);
    this.world.dirty.clear();
  }

  /** Rebuild a few queued chunks per frame so a big blast does not stutter. */
  step() {
    if (!this.queue.size) return false;
    let n = 0;
    for (const k of this.queue) {
      const [cx, cy, cz] = k.split(',').map(Number);
      this.build(cx, cy, cz);
      this.queue.delete(k);
      if (++n >= this.maxPerFrame) break;
    }
    return true;
  }

  build(cx, cy, cz) {
    const w = this.world;
    const x0 = cx * CHUNK, y0 = cy * CHUNK, z0 = cz * CHUNK;
    const x1 = Math.min(x0 + CHUNK, w.sx), y1 = Math.min(y0 + CHUNK, w.sy), z1 = Math.min(z0 + CHUNK, w.sz);
    const pos = [], nor = [], col = [], idx = [];
    const wpos = [], wnor = [], wcol = [], widx = [];
    for (let y = y0; y < y1; y++) {
      for (let z = z0; z < z1; z++) {
        for (let x = x0; x < x1; x++) {
          const id = w.get(x, y, z);
          if (id === B.AIR) continue;
          const isWater = id === B.WATER;
          const P = isWater ? wpos : pos, N = isWater ? wnor : nor, C = isWater ? wcol : col, I = isWater ? widx : idx;
          const tint = tintOf(id);
          const placed = w.getOwner && w.getOwner(x, y, z) > 0;
          for (const f of FACES) {
            const nx = x + f.n[0], ny = y + f.n[1], nz = z + f.n[2];
            const nb = w.get(nx, ny, nz);
            if (isWater) { if (nb === id) continue; if (isSolid(nb)) continue; }
            else if (nb !== B.AIR && isOpaque(nb) && nb !== B.WATER) continue;
            const base = P.length / 3;
            const light = 0.45 + 0.55 * clamp01(w.lightAt(nx, ny, nz) * 1.15);
            for (let i = 0; i < 4; i++) {
              const v = f.v[i];
              let vy = v[1];
              if (isWater && i < 2) vy -= 0.12;          // slightly sunken water top
              P.push(x + v[0], y + vy, z + v[2]);
              N.push(f.n[0], f.n[1], f.n[2]);
              const sh = f.s * light * (placed ? 1.10 : 1);
              C.push(tint[0] * sh, tint[1] * sh, tint[2] * sh);
            }
            I.push(base, base + 1, base + 2, base, base + 2, base + 3);
          }
        }
      }
    }
    this.put(this.meshes, this.key(cx, cy, cz), x0, y0, z0, pos, nor, col, idx, this.material, false);
    this.put(this.water, this.key(cx, cy, cz), x0, y0, z0, wpos, wnor, wcol, widx, this.waterMaterial, true);
  }

  put(store, k, x0, y0, z0, pos, nor, col, idx, material, isWater) {
    let mesh = store.get(k);
    if (!pos.length) {
      if (mesh) { store.delete(k); this.group.remove(mesh); mesh.geometry.dispose(); }
      return;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(x0 + CHUNK / 2, y0 + CHUNK / 2, z0 + CHUNK / 2), CHUNK * 2);
    if (!mesh) {
      mesh = new THREE.Mesh(geo, material);
      mesh.renderOrder = isWater ? 2 : 0;
      mesh.frustumCulled = false;
      store.set(k, mesh);
      this.group.add(mesh);
    } else {
      mesh.geometry.dispose();
      mesh.geometry = geo;
    }
  }

  dispose() {
    for (const m of [...this.meshes.values(), ...this.water.values()]) {
      this.group.remove(m); m.geometry.dispose();
    }
    this.meshes.clear(); this.water.clear();
    this.material.dispose(); this.waterMaterial.dispose();
    this.scene.remove(this.group);
  }
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

/** Wireframe box that shows the block the player is aiming at. */
export function makeCursor() {
  const geo = new THREE.BoxGeometry(1.002, 1.002, 1.002);
  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x101010, transparent: true, opacity: 0.85 }));
  line.visible = false;
  const ghost = new THREE.Mesh(new THREE.BoxGeometry(1.02, 1.02, 1.02),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16, depthWrite: false }));
  ghost.visible = false;
  const g = new THREE.Group();
  g.add(line); g.add(ghost);
  g.cursor = line; g.ghost = ghost;
  return g;
}
