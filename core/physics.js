/**
 * core/physics.js
 * ---------------------------------------------------------------------------
 * Deterministic, allocation-free voxel AABB physics shared by the player, the
 * 7 builders and the Locust. Movement is resolved one axis at a time against
 * the solid block set, which is the classic Minecraft formulation and keeps
 * the sim stable at a fixed 30 Hz tick.
 * ---------------------------------------------------------------------------
 */

import { ENTITY, WORLD, isSolid, clamp } from '../shared/rules.js';

export function makeEntity(kind, x, y, z) {
  const spec = kind === 'locust' ? ENTITY.LOCUST : ENTITY.PLAYER;
  return {
    kind,
    pos: { x, y, z },
    vel: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    width: spec.width,
    height: spec.height,
    eye: spec.eye,
    speed: spec.speed,
    jump: spec.jump,
    stepUp: kind === 'locust' ? 1.05 : 0.0,
    onGround: false,
    inWater: false,
    fallStart: y,
    hurtTaken: 0,
    alive: true,
    grabbed: false,
  };
}

export function aabb(e) {
  const hw = e.width / 2;
  return {
    minX: e.pos.x - hw, maxX: e.pos.x + hw,
    minY: e.pos.y, maxY: e.pos.y + e.height,
    minZ: e.pos.z - hw, maxZ: e.pos.z + hw,
  };
}

function solidAt(world, x, y, z) {
  return isSolid(world.get(Math.floor(x), Math.floor(y), Math.floor(z)));
}

/** True when the box (centre px,py,pz, half extents) intersects any solid block. */
export function boxCollides(world, cx, cy, cz, hw, h) {
  const x0 = Math.floor(cx - hw), x1 = Math.floor(cx + hw);
  const y0 = Math.floor(cy), y1 = Math.floor(cy + h - 1e-4);
  const z0 = Math.floor(cz - hw), z1 = Math.floor(cz + hw);
  for (let y = y0; y <= y1; y++) {
    for (let z = z0; z <= z1; z++) {
      for (let x = x0; x <= x1; x++) {
        if (isSolid(world.getSoft(x, y, z))) {
          // water is not solid; getSoft returns AIR outside the map (so entities
          // can leave the build box but the world border is closed by walls below)
          return true;
        }
      }
    }
  }
  return false;
}

function tryAxis(world, e, dx, dy, dz) {
  const hw = e.width / 2;
  const nx = e.pos.x + dx, ny = e.pos.y + dy, nz = e.pos.z + dz;
  return boxCollides(world, nx, ny, nz, hw, e.height);
}

/**
 * Integrate one fixed step. Returns events so the game logic can react
 * (fall damage, landing sounds, AI reward terms).
 */
export function stepEntity(world, e, dt) {
  const ev = { landed: false, hitX: false, hitZ: false, fallDamage: 0, stepped: false, inWater: false, grounded: false };
  if (e.grabbed) {
    // held by the Locust: no gravity, no control
    e.vel.x = e.vel.y = e.vel.z = 0;
    e.onGround = false;
    return ev;
  }

  const headY = Math.floor(e.pos.y + e.height * 0.6);
  const feetY = Math.floor(e.pos.y + 0.1);
  e.inWater = world.get(Math.floor(e.pos.x), headY, Math.floor(e.pos.z)) === 12
    || world.get(Math.floor(e.pos.x), feetY, Math.floor(e.pos.z)) === 12;
  ev.inWater = e.inWater;

  e.vel.y += ENTITY.GRAVITY * dt * (e.inWater ? 0.28 : 1);
  if (e.inWater) e.vel.y = Math.max(e.vel.y, -3.2);
  e.vel.y = Math.max(e.vel.y, ENTITY.TERMINAL);

  // horizontal damping (blocky acceleration like Minecraft)
  const damp = e.onGround ? 0.72 : 0.9;
  e.vel.x *= Math.pow(damp, dt * 30);
  e.vel.z *= Math.pow(damp, dt * 30);

  const wasFalling = e.vel.y < -1;
  if (e.vel.y < 0) {
    if (e.pos.y - e.fallStart > 0.05) e.fallStart = e.pos.y;
  } else {
    e.fallStart = e.pos.y;
  }

  // Y -------------------------------------------------------------------
  let dy = e.vel.y * dt;
  if (dy !== 0) {
    if (tryAxis(world, e, 0, dy, 0)) {
      if (dy < 0) {
        const fy = Math.floor(e.pos.y + dy);
        const target = fy + 1;
        if (!boxCollides(world, e.pos.x, target, e.pos.z, e.width / 2, e.height)) {
          const dropped = e.fallStart - target;
          e.pos.y = target;
          if (!e.onGround && wasFalling) {
            ev.landed = true;
            if (dropped > 3.6) ev.fallDamage = Math.min(20, Math.round((dropped - 3.6) * 1.9));
          }
        }
        e.onGround = true;
        e.vel.y = 0;
      } else {
        e.vel.y = 0;
      }
    } else {
      e.pos.y += dy;
      // ground probe (needed because we only test the swept box)
      const onGround = boxCollides(world, e.pos.x, e.pos.y - 0.06, e.pos.z, e.width / 2, 0.06);
      e.onGround = onGround;
    }
  } else {
    e.onGround = boxCollides(world, e.pos.x, e.pos.y - 0.06, e.pos.z, e.width / 2, 0.06);
  }
  if (e.onGround) e.fallStart = e.pos.y;

  // X with step-up ------------------------------------------------------
  const dx = e.vel.x * dt;
  if (dx !== 0) {
    if (tryAxis(world, e, dx, 0, 0)) {
      if (e.stepUp > 0 && e.onGround) {
        const up = e.stepUp;
        if (!tryAxis(world, e, dx, up, 0)) {
          e.pos.y += up;
          e.pos.x += dx;
          ev.stepped = true;
        } else {
          e.vel.x = 0;
          ev.hitX = true;
        }
      } else {
        e.vel.x = 0;
        ev.hitX = true;
      }
    } else e.pos.x += dx;
  }
  const dz = e.vel.z * dt;
  if (dz !== 0) {
    if (tryAxis(world, e, 0, 0, dz)) {
      if (e.stepUp > 0 && e.onGround) {
        const up = e.stepUp;
        if (!tryAxis(world, e, 0, up, dz)) {
          e.pos.y += up;
          e.pos.z += dz;
          ev.stepped = true;
        } else {
          e.vel.z = 0;
          ev.hitZ = true;
        }
      } else {
        e.vel.z = 0;
        ev.hitZ = true;
      }
    } else e.pos.z += dz;
  }

  // keep entities inside the map
  const m = 1.2;
  if (e.pos.x < m) { e.pos.x = m; e.vel.x = 0; ev.hitX = true; }
  if (e.pos.z < m) { e.pos.z = m; e.vel.z = 0; ev.hitZ = true; }
  if (e.pos.x > WORLD.SX - m) { e.pos.x = WORLD.SX - m; e.vel.x = 0; ev.hitX = true; }
  if (e.pos.z > WORLD.SZ - m) { e.pos.z = WORLD.SZ - m; e.vel.z = 0; ev.hitZ = true; }
  if (e.pos.y < 1) { e.pos.y = 1; e.vel.y = 0; e.onGround = true; }
  if (e.pos.y > WORLD.SY - 2) { e.pos.y = WORLD.SY - 2; e.vel.y = 0; }

  ev.grounded = e.onGround;
  return ev;
}

/** Snap an entity onto the highest solid block of a column. */
export function groundSnap(world, e) {
  const x = Math.floor(e.pos.x), z = Math.floor(e.pos.z);
  for (let y = world.sy - 1; y >= 1; y--) {
    if (isSolid(world.get(x, y, z))) {
      e.pos.y = y + 1;
      return y + 1;
    }
  }
  e.pos.y = 1;
  return 1;
}

export function canStandAt(world, x, y, z, e) {
  const hw = e.width / 2;
  if (boxCollides(world, x + 0.5, y, z + 0.5, hw, e.height)) return false;
  return isSolid(world.get(Math.floor(x), y - 1, Math.floor(z)))
    || isSolid(world.get(Math.floor(x), y - 1, Math.floor(z)));
}

export function applyMoveIntent(e, dt, moveX, moveZ, speedMul = 1) {
  // moveX/moveZ are local-space intents in [-1,1]; -moveZ is "forward". This is
  // the same yaw convention as dirFromAngles()/integrateIntent(): facing =
  // (-sin yaw, -cos yaw) with +x to the right of it. Getting one of these three
  // out of sync is invisible except as agents walking backwards, so they live here.
  const cs = Math.cos(e.yaw), sn = Math.sin(e.yaw);
  const fwd = -moveZ;
  const wx = -sn * fwd + cs * moveX;
  const wz = -cs * fwd - sn * moveX;
  const len = Math.hypot(wx, wz);
  const sp = e.speed * speedMul;
  if (len < 1e-6) { e.vel.x = 0; e.vel.z = 0; return; }
  const scale = sp * Math.min(1, len) / len;
  e.vel.x = wx * scale;
  e.vel.z = wz * scale;
}

export function jump(e) {
  if (e.onGround) {
    e.vel.y = e.jump;
    e.onGround = false;
    return true;
  }
  if (e.inWater) {
    e.vel.y = 4.4;
    return true;
  }
  return false;
}

export function distance3(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
export function horizontalDistance(a, b) {
  const dx = a.x - b.x, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}
export function boxOverlap(a, b, pad = 0) {
  const A = aabb(a), Bb = aabb(b);
  return A.minX - pad < Bb.maxX && A.maxX + pad > Bb.minX
    && A.minY - pad < Bb.maxY && A.maxY + pad > Bb.minY
    && A.minZ - pad < Bb.maxZ && A.maxZ + pad > Bb.minZ;
}
export { clamp };
