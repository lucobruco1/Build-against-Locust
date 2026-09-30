/**
 * game/actions.js
 * ---------------------------------------------------------------------------
 * The action space, executed identically for the human player and for the
 * 7 EfficientZero builders — that symmetry is the whole point of the design: the
 * network gets exactly the verbs you have.
 *
 *  • movement / look actions set a *held intent* that lives until the next
 *    decision (a decision interval is ~330 ms, i.e. ~10 physics ticks),
 *  • place / break / jump fire once at decision time,
 *  • every return value carries the reward contribution that the value head is
 *    trained on, so the reward function and the verbs can never drift apart.
 * ---------------------------------------------------------------------------
 */

import { ACT, LACT, B, RW, BLOCK_DEFS, isSolid, ENTITY, HOTBAR, BASE, clamp } from '../shared/rules.js';
import { placeTarget, hasLineOfSight } from '../core/raycast.js';
import { makeShell, OWNER_NATURAL } from '../core/world.js';

export const BUILD_INTERVAL = 0.33; // seconds between two decisions (builders)
export const LOCUST_INTERVAL = 0.25;

/* --------------------------------------------------------------- intents */

export function clearIntent(a) {
  a.intent.mx = 0;
  a.intent.mz = 0;
  a.intent.sprint = 0;
  a.intent.turn = 0;
  a.intent.pitch = 0;
}

/**
 * @param {object} a builder agent (entity + inventory + plot)
 * @param {number} action ACT.*
 * @param {object} ctx {world, phase, now, entities, log, grant}
 */
export function executeBuilderAction(a, action, ctx) {
  const world = ctx.world;
  const res = { reward: 0, events: [], info: '' };
  const it = a.intent;
  // human clicks must not wipe the movement intent the keyboard is holding
  if (!ctx.preserveIntent) clearIntent(a);

  switch (action) {
    case ACT.NOOP:
      break;
    case ACT.FORWARD:
      it.mz = -1;
      break;
    case ACT.BACKWARD:
      it.mz = 1;
      break;
    case ACT.STRAFE_LEFT:
      it.mx = -1;
      break;
    case ACT.STRAFE_RIGHT:
      it.mx = 1;
      break;
    case ACT.TURN_LEFT:
      it.turn = -1;
      break;
    case ACT.TURN_RIGHT:
      it.turn = 1;
      break;
    case ACT.LOOK_UP:
      it.pitch = 1;
      break;
    case ACT.LOOK_DOWN:
      it.pitch = -1;
      break;
    case ACT.JUMP:
      if (a.ent.onGround) {
        a.ent.vel.y = a.ent.jump;
        a.ent.onGround = false;
        res.events.push({ t: 'jump', x: a.ent.pos.x, y: a.ent.pos.y, z: a.ent.pos.z });
      } else if (a.ent.inWater) {
        a.ent.vel.y = 4.2;
      }
      break;
    case ACT.SPRINT:
      it.sprint = 1;
      it.mz = it.mz || -1;
      break;
    case ACT.PLACE_FRONT:
    case ACT.PLACE_DOWN:
      res.reward += placeBlock(a, world, ctx, res, action === ACT.PLACE_DOWN);
      break;
    case ACT.BREAK_FRONT:
    case ACT.BREAK_DOWN:
      res.reward += breakBlock(a, world, ctx, res, action === ACT.BREAK_DOWN);
      break;
    default:
      res.reward += RW.ILLEGAL;
      res.info = 'unknown action';
  }
  if (it.sprint) it.mz = it.mz || -1;
  return res;
}

function cellForPlace(a, world, ctx, downward) {
  const e = a.ent;
  if (downward) {
    const x = Math.floor(e.pos.x), y = Math.floor(e.pos.y - 0.5), z = Math.floor(e.pos.z);
    return { x, y, z };
  }
  const t = placeTarget(world, e.pos.x, e.pos.y + e.eye, e.pos.z, e.yaw, e.pitch, ENTITY.REACH);
  return t.place;
}

/**
 * Is `cell` a legal placement for `a`? Air, supported (the Minecraft rule),
 * not inside an entity, and for AI builders inside their own plot.
 */
export function validPlace(a, world, ctx, cell) {
  if (!cell) return false;
  const { x, y, z } = cell;
  if (!world.inside(x, y, z) || world.get(x, y, z) !== B.AIR) return false;
  if (!a.isHuman && (Math.abs(x - a.plot.x) > BASE.RADIUS || Math.abs(z - a.plot.z) > BASE.RADIUS)) return false;
  const support = isSolid(world.getSoft(x - 1, y, z)) || isSolid(world.getSoft(x + 1, y, z))
    || isSolid(world.getSoft(x, y, z - 1)) || isSolid(world.getSoft(x, y, z + 1))
    || isSolid(world.getSoft(x, y - 1, z)) || isSolid(world.getSoft(x, y + 1, z));
  if (!support) return false;
  for (const o of ctx.entities) {
    if (!o.alive) continue;
    const ox = o.ent.pos.x, oz = o.ent.pos.z;
    if (x + 1 > ox - 0.31 && x < ox + 0.31 && z + 1 > oz - 0.31 && z < oz + 0.31
      && y + 1 > o.ent.pos.y && y < o.ent.pos.y + o.ent.height) return false;
  }
  return true;
}

/**
 * Which cell the PLACE action resolves to.
 *
 * The aimed cell is tried first for everybody, so the verb is literally the
 * player's verb. AI builders get a second chance: their aim only moves in
 * 0.55 rad steps per decision, so demanding crosshair precision would put the
 * entire fortify skill outside the exploration budget — so if the aimed cell is
 * illegal we fall back to the nearest *legal cell of the shell they are
 * building* inside reach. That is aim assistance in the executor only; the
 * network still has to decide to place at all, and it is only paid for filling
 * shell cells.
 */
export function placementCandidate(a, world, ctx) {
  const aimed = cellForPlace(a, world, ctx, false);
  if (validPlace(a, world, ctx, aimed)) return { cell: aimed, ok: true, aimed: true };
  if (a.isHuman || !a.shell) return { cell: aimed, ok: false, aimed: true };
  const e = a.ent;
  const goal = a.targetCell;
  let best = null, bestS = Infinity;
  const list = a.shell.all;
  for (let i = 0; i < list.length; i++) {
    const c = list[i];
    const d = Math.max(Math.abs(c.x + 0.5 - e.pos.x), Math.abs(c.z + 0.5 - e.pos.z));
    if (d > ENTITY.REACH + 0.6) continue;
    if (Math.abs(c.y - e.pos.y) > 3.2) continue;
    if (!validPlace(a, world, ctx, c)) continue;
    let s = d + Math.abs(c.y - e.pos.y) * 0.8;
    if (goal && c.x === goal.x && c.y === goal.y && c.z === goal.z) s -= 6;
    if (c.ring === 'wall') s -= 1.2;
    if (s < bestS) { bestS = s; best = c; }
  }
  if (best) return { cell: best, ok: true, aimed: false };
  return { cell: aimed, ok: false, aimed: true };
}

function placeBlock(a, world, ctx, res, downward) {
  let cell;
  if (downward) {
    cell = cellForPlace(a, world, ctx, true);
    if (!validPlace(a, world, ctx, cell)) {
      // pillaring: allow the cell we are standing over even when it is
      // "unsupported" by definition (it rests on the floor below it)
      const e = a.ent;
      cell = { x: Math.floor(e.pos.x), y: Math.floor(e.pos.y - 0.05), z: Math.floor(e.pos.z) };
      if (!validPlace(a, world, ctx, cell)) { res.reward += RW.ILLEGAL; res.info = 'illegal (down)'; return RW.ILLEGAL; }
    }
  } else {
    const cand = placementCandidate(a, world, ctx);
    if (!cand.ok || !cand.cell) { res.reward += RW.ILLEGAL; res.info = 'no legal cell'; return RW.ILLEGAL; }
    cell = cand.cell;
    res.placedFrom = cand.aimed ? 'aim' : 'shell';
  }
  const id = a.isHuman ? a.inv.selected : a.inv.pickForBuild();
  if (!id || !a.inv.has(id)) {
    res.reward += RW.ILLEGAL;
    res.info = 'empty hand';
    return RW.ILLEGAL;
  }
  a.inv.take(id, 1);
  world.place(cell.x, cell.y, cell.z, id, a.owner, ctx.now);
  a.stats.placed++;
  res.events.push({ t: 'place', x: cell.x, y: cell.y, z: cell.z, id, by: a.id, own: a.owner ?? 0 });
  const inPlot = Math.abs(cell.x - a.plot.x) <= BASE.RADIUS && Math.abs(cell.z - a.plot.z) <= BASE.RADIUS;
  if (!inPlot) {
    res.reward += RW.PLACE_WASTE;
    return RW.PLACE_WASTE;
  }
  const ring = a.shellIndex ? a.shellIndex.get(world.index(cell.x, cell.y, cell.z)) : null;
  let r = RW.PLACE;
  if (ring === 'wall') r += RW.PLACE_WALL;
  else if (ring === 'roof') r += RW.PLACE_WALL * 0.85;
  else if (ring === 'outer') r += RW.PLACE_OUTER;
  res.info = ring ? `shell:${ring}` : 'off-shell';
  res.reward += r;
  res.ring = ring;
  return r;
}

function breakBlock(a, world, ctx, res, downward) {
  const e = a.ent;
  let cell;
  if (downward) {
    cell = { x: Math.floor(e.pos.x), y: Math.floor(e.pos.y - 0.55), z: Math.floor(e.pos.z) };
  } else {
    const t = placeTarget(world, e.pos.x, e.pos.y + e.eye, e.pos.z, e.yaw, e.pitch, ENTITY.REACH);
    cell = t.breakCell;
  }
  if (!cell) { res.reward += RW.ILLEGAL; return RW.ILLEGAL; }
  const who = { kind: a.isHuman ? 'player' : 'builder', owner: a.owner };
  if (!world.canBreak(cell.x, cell.y, cell.z, who)) {
    res.reward += RW.ILLEGAL;
    res.info = 'not yours to break';
    return RW.ILLEGAL;
  }
  const id = world.get(cell.x, cell.y, cell.z);
  const own = world.getOwner(cell.x, cell.y, cell.z) === a.owner;
  world.breakBlock(cell.x, cell.y, cell.z, who, ctx.now);
  if (!a.isHuman) a.inv.add(id, 1);
  else a.inv.add(id, Math.max(1, Math.min(4, a.inv.count(id) === 0 ? 4 : 1)));
  a.stats.broken++;
  res.events.push({ t: 'break', x: cell.x, y: cell.y, z: cell.z, id, by: a.id });
  const r = own ? RW.BREAK_OWN : RW.MINE;
  res.reward += r;
  return r;
}

/* ------------------------------------------------------- legal action mask */

export function builderLegalMask(a, world, ctx) {
  const n = 15;
  const m = new Uint8Array(n);
  m.fill(1);
  const e = a.ent;
  if (!a.alive) {
    m.fill(0);
    m[ACT.NOOP] = 1;
    return m;
  }
  if (!e.onGround) m[ACT.JUMP] = 0;
  if (a.coolPlace > 0 || a.inv.total <= 0) {
    m[ACT.PLACE_FRONT] = 0;
    m[ACT.PLACE_DOWN] = 0;
  } else {
    // forbid placing outside the plot for AI builders (the player may build
    // anywhere) – keeps the search focused on the actual objective
    if (!a.isHuman) {
      if (!placementCandidate(a, world, ctx).ok) m[ACT.PLACE_FRONT] = 0;
      const d = { x: Math.floor(e.pos.x), y: Math.floor(e.pos.y - 0.05), z: Math.floor(e.pos.z) };
      if (!validPlace(a, world, ctx, d)) m[ACT.PLACE_DOWN] = 0;
    } else {
      const t = placeTarget(world, e.pos.x, e.pos.y + e.eye, e.pos.z, e.yaw, e.pitch, ENTITY.REACH);
      if (!t.place || world.get(t.place.x, t.place.y, t.place.z) !== B.AIR) m[ACT.PLACE_FRONT] = 0;
    }
  }
  if (a.coolBreak > 0) {
    m[ACT.BREAK_FRONT] = 0;
    m[ACT.BREAK_DOWN] = 0;
  } else {
    const t = placeTarget(world, e.pos.x, e.pos.y + e.eye, e.pos.z, e.yaw, e.pitch, ENTITY.REACH);
    const who = { kind: a.isHuman ? 'player' : 'builder', owner: a.owner };
    const front = t.breakCell ? world.canBreak(t.breakCell.x, t.breakCell.y, t.breakCell.z, who) : false;
    if (!front) m[ACT.BREAK_FRONT] = 0;
    const d = { x: Math.floor(e.pos.x), y: Math.floor(e.pos.y - 0.55), z: Math.floor(e.pos.z) };
    const below = world.canBreak(d.x, d.y, d.z, who);
    if (!below) m[ACT.BREAK_DOWN] = 0;
  }
  if (a.ent.grabbed) {
    // struggling is the only thing that works while you are in its hand
    m.fill(0);
    m[ACT.JUMP] = 1;
    m[ACT.BACKWARD] = 1;
    m[ACT.FORWARD] = 1;
    m[ACT.NOOP] = 1;
  }
  void ctx;
  return m;
}

/* ------------------------------------------------------------ locust verbs */

/**
 * The Locust has two verbs of its own beyond locomotion: it *smashes* blocks
 * (hardness measured in hits) and it *strikes* a prey — which, on success,
 * turns into the grab → stab kill animation.
 */
export function executeLocustAction(a, action, ctx) {
  const world = ctx.world;
  const res = { reward: 0, events: [], info: '' };
  const e = a.ent;
  a.intent.mx = 0;
  a.intent.mz = 0;
  a.intent.turn = 0;
  a.intent.pitch = 0;
  a.intent.leap = 0;
  switch (action) {
    case LACT.NOOP:
      break;
    case LACT.FORWARD:
      a.intent.mz = -1;
      break;
    case LACT.BACKWARD:
      a.intent.mz = 1;
      break;
    case LACT.STRAFE_LEFT:
      a.intent.mx = -1;
      break;
    case LACT.STRAFE_RIGHT:
      a.intent.mx = 1;
      break;
    case LACT.TURN_LEFT:
      a.intent.turn = -1;
      break;
    case LACT.TURN_RIGHT:
      a.intent.turn = 1;
      break;
    case LACT.LOOK_UP:
      a.intent.pitch = 1;
      break;
    case LACT.LOOK_DOWN:
      a.intent.pitch = -1;
      break;
    case LACT.LEAP:
      if (e.onGround) {
        e.vel.y = e.jump;
        e.onGround = false;
        a.intent.mz = -1;
        res.events.push({ t: 'leap', x: e.pos.x, y: e.pos.y, z: e.pos.z });
      }
      break;
    case LACT.SMASH_BLOCK: {
      const t = placeTarget(world, e.pos.x, e.pos.y + 2.4, e.pos.z, e.yaw, e.pitch, ENTITY.REACH + 2.5);
      let cell = t.breakCell;
      if (!cell || !world.canBreak(cell.x, cell.y, cell.z, { kind: 'locust' })) cell = nearestSmashable(a, world);
      if (!cell || !world.canBreak(cell.x, cell.y, cell.z, { kind: 'locust' })) {
        res.reward += RW.ILLEGAL;
        res.info = 'nothing to smash';
        break;
      }
      const id = world.get(cell.x, cell.y, cell.z);
      const d = BLOCK_DEFS[id];
      const idx = world.index(cell.x, cell.y, cell.z);
      a.smashHits.set(idx, (a.smashHits.get(idx) || 0) + 1);
      res.events.push({ t: 'smash', x: cell.x, y: cell.y, z: cell.z, id });
      if (a.smashHits.get(idx) >= (d?.locustHP ?? 2)) {
        a.smashHits.delete(idx);
        // the owner has to be read *before* the break, which clears it
        const own = world.getOwner(cell.x, cell.y, cell.z);
        world.breakBlock(cell.x, cell.y, cell.z, { kind: 'locust' }, ctx.now);
        a.stats.smashed++;
        res.events.push({ t: 'break', x: cell.x, y: cell.y, z: cell.z, id, by: a.id, own });
        res.reward += RW.L_BREAK;
        // ...and the plot owner is always told, not only when a "victim" is
        // flagged: the loss of a wall block is the feedback the builders learn from
        ctx.onBlockBroken?.(cell, own);
      }
      break;
    }
    case LACT.STRIKE: {
      if (a.attackCd > 0) {
        res.reward += RW.ILLEGAL;
        res.info = 'cooldown';
        break;
      }
      const prey = findPrey(a, ctx.entities, ctx.world);
      if (!prey) {
        res.reward += RW.L_IDLE_TARGET;
        res.info = 'no prey in reach';
        break;
      }
      a.attackCd = 0.75;
      res.events.push({ t: 'grab', by: a.id, victim: prey.id });
      prey.grabbed = true;
      prey.grabbedBy = a.id;
      prey.grabT = 0;
      a.grabbed = prey;
      a.intent.mz = 0;
      res.reward += RW.L_GRAB;
      break;
    }
    default:
      res.reward += RW.ILLEGAL;
  }
  return res;
}

/**
 * The Locust's smash is aim-assisted the same way a builder's place is: if the
 * cell under its gaze is not breakable it smashes the nearest breakable block in
 * front of it (within reach). Without this a creature whose pitch only moves in
 * 0.4 rad steps would mostly hit air and the hunt would never threaten a wall.
 */
/** The cell a Locust smash would open, plus its distance (null if none). */
export function smashTargetInfo(a, world, maxR = 3, want) {
  const cell = nearestSmashable(a, world, maxR, want);
  if (!cell) return null;
  const e = a.ent;
  return { cell, dist: Math.hypot(cell.x + 0.5 - e.pos.x, cell.y - e.pos.y, cell.z + 0.5 - e.pos.z) };
}

export function nearestSmashable(a, world, maxR = 3, want) {
  // Only blocks that are actually IN THE WAY count: at or above the natural
  // terrain of their column, and roughly towards the goal. Without the second
  // condition a Locust with nothing better to do happily farms smash-reward by
  // demolishing the floor under its own feet.
  const wantLen = want ? Math.hypot(want.x, want.z) : 0;
  const e = a.ent;
  const fx = -Math.sin(e.yaw), fz = -Math.cos(e.yaw);
  let best = null, bestS = Infinity;
  const MAXR = maxR;
  const cx = Math.floor(e.pos.x), cz = Math.floor(e.pos.z);
  const cy = Math.floor(e.pos.y);
  for (let dy = -1; dy <= 4; dy++) {
    for (let r = 0; r <= 3; r++) {
      for (let side = -r; side <= r; side++) {
        const cands = r === 0 ? [{ x: cx, z: cz }] : [
          { x: cx + (fx >= 0 ? r : -r), z: cz + side },
          { x: cx + (fx >= 0 ? -r : r), z: cz + side },
          { x: cx + side, z: cz + (fz >= 0 ? r : -r) },
          { x: cx + side, z: cz + (fz >= 0 ? -r : r) },
        ];
        for (const c of cands) {
          const y = cy + dy;
          if (y < world.terrainSurface(c.x, c.z)) continue;
          if (!world.canBreak(c.x, y, c.z, { kind: 'locust' })) continue;
          const dx = c.x + 0.5 - e.pos.x, dz = c.z + 0.5 - e.pos.z;
          const len = Math.hypot(dx, dz) || 1;
          const dot = (dx / len) * fx + (dz / len) * fz;
          // a Locust can only hit what it is looking at (~70 deg cone): without
          // this it would happily smash the scenery behind it and never turn
          if (dot < 0.34) continue;
          if (wantLen > 0.5) {
            const cl = Math.hypot(dx, dz) || 1;
            const align = (dx * want.x + dz * want.z) / (cl * wantLen);
            if (align < 0.2) continue;
          }
          let s = r * 1.0 + Math.max(0, dy - 1) * 2.2 + Math.abs(dy + 1) * 0.7 - dot * 2.5;
          // Smash what the base is made of, not the ground under its feet: without
          // this a block *standing on* the terrain and the terrain itself tie (or
          // the terrain wins, being nearer), and a Locust at a wall chews the
          // floor forever. Digging is allowed — never preferred.
          if (world.getOwner(c.x, y, c.z) === OWNER_NATURAL) s += 2.2;
          if (wantLen > 0.5) {
            const cl = Math.hypot(dx, dz) || 1;
            const align = (dx * want.x + dz * want.z) / (cl * wantLen);
            s += (1 - align) * 3.5;
          }
          if (s < bestS) { bestS = s; best = { x: c.x, y, z: c.z }; }
        }
      }
    }
  }
  return best;
}

export function findPrey(a, entities, world) {
  const e = a.ent;
  let best = null, bestD = Infinity;
  for (const o of entities) {
    if (!o.alive || o.id === a.id || o.kind === 'locust' || o.grabbed) continue;
    const dx = o.ent.pos.x - e.pos.x, dy = o.ent.pos.y - e.pos.y, dz = o.ent.pos.z - e.pos.z;
    const d = Math.hypot(dx, dy, dz);
    const fx = -Math.sin(e.yaw), fz = -Math.cos(e.yaw);
    const len = Math.hypot(dx, dz) || 1;
    const dot = (dx / len) * fx + (dz / len) * fz;
    if (dot < 0.1) continue;
    // its arms are absurdly long: it can also fish prey out through a hole in
    // the roof, as long as there is line of sight down the hole
    const reach = d <= ENTITY.GRAB_RANGE ? true
      : (d <= ENTITY.REACH_THROUGH_HOLE && dy < 0 && world
        && hasLineOfSight(world, { x: e.pos.x, y: e.pos.y + e.eye, z: e.pos.z },
          { x: o.ent.pos.x, y: o.ent.pos.y + 1.2, z: o.ent.pos.z }, 9));
    if (!reach) continue;
    if (d < bestD) { bestD = d; best = o; }
  }
  return best;
}

export function locustLegalMask(a, ctx) {
  const n = 12;
  const m = new Uint8Array(n);
  m.fill(1);
  if (!a.active) { m.fill(0); m[LACT.NOOP] = 1; return m; }
  if (a.ent.grabbed === false && a.attackCd > 0) m[LACT.STRIKE] = 0;
  if (!findPrey(a, ctx.entities, ctx.world)) m[LACT.STRIKE] = 0;
  const e = a.ent;
  const t = placeTarget(ctx.world, e.pos.x, e.pos.y + 2.4, e.pos.z, e.yaw, e.pitch, ENTITY.REACH + 2.5);
  const can = (t.breakCell && ctx.world.canBreak(t.breakCell.x, t.breakCell.y, t.breakCell.z, { kind: 'locust' }))
    || !!nearestSmashable(a, ctx.world, 3, ctx.want);
  if (!can) m[LACT.SMASH_BLOCK] = 0;
  if (!e.onGround) m[LACT.LEAP] = 0;
  return m;
}

/** Apply the held intent for one physics tick. */
export function integrateIntent(a, dt) {
  const e = a.ent;
  const it = a.intent;
  if (a.grabbed) return;
  const turning = (it.turn || 0) * (a.kind === 'locust' ? 2.6 : 3.2);
  if (turning) e.yaw += turning * dt;
  if (it.pitch) {
    e.pitch = clamp(e.pitch + it.pitch * (a.kind === 'locust' ? 1.6 : 2.0) * dt, -1.25, 1.1);
  }
  const speedMul = (it.sprint ? 1.5 : 1) * (e.inWater ? 0.55 : 1);
  const cs = Math.cos(e.yaw), sn = Math.sin(e.yaw);
  const mx = it.mx || 0, mz = it.mz || 0;
  // facing convention shared with raycast.dirFromAngles: forward = (-sin y, -cos y),
  // right = (cos y, -sin y); the builder actions encode "forward" as mz = -1
  const fwd = -mz;
  const wx = -sn * fwd + cs * mx;
  const wz = -cs * fwd - sn * mx;
  const len = Math.hypot(mx, mz);
  const sp = e.speed * speedMul * (len > 0 ? 1 : 0);
  const targetX = len > 0 ? (wx / Math.max(len, 1e-6)) * sp : 0;
  const targetZ = len > 0 ? (wz / Math.max(len, 1e-6)) * sp : 0;
  // accel toward the requested velocity (blocky but not teleporty)
  const k = e.onGround ? 0.35 : 0.16;
  e.vel.x += (targetX - e.vel.x) * k;
  e.vel.z += (targetZ - e.vel.z) * k;
}

export { makeShell, HOTBAR, BASE };
