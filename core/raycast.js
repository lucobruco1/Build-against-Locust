/**
 * core/raycast.js
 * ---------------------------------------------------------------------------
 * Amanatides & Woo voxel traversal. Used for the player's crosshair picking,
 * the AI block targets (place / break), and the line-of-sight tests the Locust
 * and the shelter reward depend on.
 * ---------------------------------------------------------------------------
 */

import { WORLD, B, isSolid, isOpaque, BLOCK_DEFS } from '../shared/rules.js';

/**
 * @param {object} world  VoxelWorld
 * @returns {{hit:boolean, x:number,y:number,z:number, nx:number,ny:number,nz:number,
 *             dist:number, id:number, prev:{x:number,y:number,z:number}}|null}
 */
export function raycast(world, ox, oy, oz, dx, dy, dz, maxDist = 5, opts = {}) {
  const stopOn = opts.stopOn || ((id) => (opts.liquids ? id !== B.AIR : isSolid(id)));
  let x = Math.floor(ox), y = Math.floor(oy), z = Math.floor(oz);
  const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
  const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0;
  const stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;
  const invX = dx === 0 ? Infinity : Math.abs(1 / dx);
  const invY = dy === 0 ? Infinity : Math.abs(1 / dy);
  const invZ = dz === 0 ? Infinity : Math.abs(1 / dz);
  let tMaxX = dx === 0 ? Infinity : ((stepX > 0 ? x + 1 - ox : ox - x) * invX);
  let tMaxY = dy === 0 ? Infinity : ((stepY > 0 ? y + 1 - oy : oy - y) * invY);
  let tMaxZ = dz === 0 ? Infinity : ((stepZ > 0 ? z + 1 - oz : oz - z) * invZ);
  let t = 0;
  let nx = 0, ny = 0, nz = 0;
  let px = x, py = y, pz = z;
  let guard = 0;
  while (t <= maxDist && guard++ < 512) {
    const id = world.getSoft(x, y, z);
    if (id !== B.AIR && stopOn(id)) {
      return { hit: true, x, y, z, nx, ny, nz, dist: t, id, prev: { x: px, y: py, z: pz } };
    }
    px = x; py = y; pz = z;
    if (tMaxX < tMaxY && tMaxX < tMaxZ) {
      t = tMaxX; tMaxX += invX; x += stepX; nx = -stepX; ny = 0; nz = 0;
    } else if (tMaxY < tMaxZ) {
      t = tMaxY; tMaxY += invY; y += stepY; nx = 0; ny = -stepY; nz = 0;
    } else {
      t = tMaxZ; tMaxZ += invZ; z += stepZ; nx = 0; ny = 0; nz = -stepZ;
    }
    if (x < -1 || y < -1 || z < -1 || x > WORLD.SX || y > WORLD.SY || z > WORLD.SZ) break;
  }
  return { hit: false, x, y, z, nx: 0, ny: 0, nz: 0, dist: t, id: B.AIR, prev: { x: px, y: py, z: pz } };
}

export function dirFromAngles(yaw, pitch) {
  const cp = Math.cos(pitch);
  return { x: -Math.sin(yaw) * cp, y: Math.sin(pitch), z: -Math.cos(yaw) * cp };
}

/** The cell the agent wants to place into (adjacent to the aimed face). */
export function placeTarget(world, ex, ey, ez, yaw, pitch, reach = 4.5) {
  const d = dirFromAngles(yaw, pitch);
  const r = raycast(world, ex, ey, ez, d.x, d.y, d.z, reach);
  if (!r.hit) {
    // free-standing placement two blocks ahead at feet level
    const fx = Math.floor(ex + d.x * 2), fy = Math.floor(ey - 0.6), fz = Math.floor(ez + d.z * 2);
    return { place: { x: fx, y: fy, z: fz }, breakCell: null, dist: 2, grounded: false };
  }
  return {
    place: { x: r.x + r.nx, y: r.y + r.ny, z: r.z + r.nz },
    breakCell: { x: r.x, y: r.y, z: r.z },
    dist: r.dist,
    id: r.id,
    grounded: true,
  };
}

/** Line of sight between two world points (opaque blocks occlude). */
export function hasLineOfSight(world, a, b, maxDist = 40) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-4) return true;
  if (len > maxDist) return false;
  const r = raycast(world, a.x, a.y, a.z, dx / len, dy / len, dz / len, len, {
    stopOn: (id) => isOpaque(id),
  });
  return !r.hit;
}

export function hardnessOf(id) {
  const d = BLOCK_DEFS[id];
  return d ? d.hardness : 1;
}
