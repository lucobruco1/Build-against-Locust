/**
 * ai/prior.js
 * ---------------------------------------------------------------------------
 * A scripted *expert prior*, not a scripted agent. Its only job is to give the
 * root of the tree a decent starting distribution (AlphaZero-style "move
 * priors from a domain policy"), which is what lets the game be playable on the
 * very first minute while the EfficientZero networks are still untrained.
 *
 * The prior is blended into the search target policy with weight
 * `assist · decay(trainSteps)`, so the networks imitate it early and then
 * overrule it as their own value/policy heads get better. With assist = 0 the
 * agents are 100 % EfficientZero: the prior is never computed and never used.
 * ---------------------------------------------------------------------------
 */

import { ACT, LACT, B, isSolid } from '../shared/rules.js';
import { placeTarget } from '../core/raycast.js';

function soft(argmax, scores, tau = 0.6) {
  const n = scores.length;
  const m = Math.max(...scores);
  let s = 0;
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    p[i] = Math.exp((scores[i] - m) / tau);
    s += p[i];
  }
  if (s <= 0) p[argmax] = 1;
  else for (let i = 0; i < n; i++) p[i] /= s;
  return p;
}

/**
 * Builder prior.
 * @param {object} ctx {ent, world, owner, base, shell, targetShellCell, inventory,
 *                      phase, locust, security, grabbed}
 */
export function builderPrior(ctx) {
  const nA = 15;
  const s = new Float32Array(nA).fill(0.02);
  const e = ctx.ent;
  const w = ctx.world;
  if (ctx.grabbed) {
    s[ACT.JUMP] = 1.4;
    s[ACT.BACKWARD] = 0.6;
    return s;
  }
  const base = ctx.base;
  const dx = base ? base.x + 0.5 - e.pos.x : 0;
  const dz = base ? base.z + 0.5 - e.pos.z : 0;
  const dist = Math.hypot(dx, dz);
  const wantYaw = Math.atan2(-dx, -dz);
  let diff = wantYaw - e.yaw;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;

  const atHome = dist < 3.4;
  if (Math.abs(diff) > 0.5) s[diff > 0 ? ACT.TURN_RIGHT : ACT.TURN_LEFT] += 1.0;
  else if (dist > 2.0) s[ACT.FORWARD] += 0.9;
  if (!e.onGround) s[ACT.NOOP] += 0.4;

  // hunt phase: get inside, cover up
  if (ctx.phase === 'hunt') {
    s[ACT.FORWARD] += atHome ? 0.1 : 0.7;
    if (ctx.security && ctx.security.breaches > 0 && atHome) s[ACT.PLACE_FRONT] += 0.8;
    if (e.pos.y - (ctx.base.groundY ?? 0) > 0.5) s[ACT.JUMP] += 0.15;
  }

  // the block it wants to fill next
  const goal = ctx.targetShellCell;
  if (goal) {
    const gdx = goal.x + 0.5 - e.pos.x, gdz = goal.z + 0.5 - e.pos.z, gdy = goal.y - e.pos.y;
    const gd = Math.hypot(gdx, gdz);
    const gy = Math.atan2(-gdx, -gdz);
    let gdiff = gy - e.yaw;
    while (gdiff > Math.PI) gdiff -= Math.PI * 2;
    while (gdiff < -Math.PI) gdiff += Math.PI * 2;
    if (Math.abs(gdiff) > 0.45) s[gdiff > 0 ? ACT.TURN_RIGHT : ACT.TURN_LEFT] += 1.25;
    else {
      if (gd < 4.2 && Math.abs(gdy) <= 2.2) s[ACT.PLACE_FRONT] += 1.6;
      if (gd > 4.2 || gdy < -0.5) s[ACT.FORWARD] += 0.9;
      if (gdy > 1.4) s[ACT.JUMP] += 0.4;
    }
    if (Math.abs(gdy) > 1.6) s[gdy > 0 ? ACT.LOOK_UP : ACT.LOOK_DOWN] += 0.35;
  }
  // dig out of a corner / reclaim own misplaced block
  const t = placeTarget(w, e.pos.x, e.pos.y + 1.5, e.pos.z, e.yaw, e.pitch || 0, 4.2);
  if (t.breakCell) {
    const own = w.getOwner(t.breakCell.x, t.breakCell.y, t.breakCell.z) === ctx.owner;
    if (own && !goal) s[ACT.BREAK_FRONT] += 0.5;
    if (t.place && !isSolid(w.get(t.place.x, t.place.y, t.place.z)) && ctx.inventory?.total > 0) s[ACT.PLACE_FRONT] += 0.4;
  }
  // pillaring is how you get a roof on
  if (ctx.needRoof && e.onGround) {
    s[ACT.PLACE_DOWN] += 0.7;
    s[ACT.JUMP] += 0.7;
  }
  let am = 0;
  for (let i = 1; i < nA; i++) if (s[i] > s[am]) am = i;
  return soft(am, s, 0.55);
}

/**
 * Locust prior: follow the A* direction, smash what is in the way, strike prey
 * that is within reach.
 */
export function locustPrior(ctx) {
  const nA = 12;
  const s = new Float32Array(nA).fill(0.02);
  const e = ctx.ent;
  const path = ctx.pathDir || { dx: 0, dz: 0, dy: 0, yaw: null };
  let off = Infinity;
  if (path.yaw !== null && path.yaw !== undefined) {
    off = path.yaw - e.yaw;
    while (off > Math.PI) off -= Math.PI * 2;
    while (off < -Math.PI) off += Math.PI * 2;
  }
  const facing = Math.abs(off) <= 0.5;
  if (path.yaw !== null && path.yaw !== undefined) {
    // Turn first: smashing while side-on to the target opens a tunnel next to
    // the breach instead of the breach itself.
    if (!facing) s[off > 0 ? LACT.TURN_RIGHT : LACT.TURN_LEFT] += 2.6;
    else s[LACT.FORWARD] += 1.7;
  }
  if (ctx.canStrike) s[LACT.STRIKE] += 4.5;
  else {
    if (ctx.blocked && facing) {
      s[LACT.SMASH_BLOCK] += 2.0;
      s[LACT.FORWARD] += 0.9;   // step into the hole as soon as it opens
    }
    if (ctx.canSmash && facing) s[LACT.SMASH_BLOCK] += 0.8;
  }
  if (ctx.targetUp && ctx.targetUp > 1.4) s[LACT.LOOK_UP] += 0.4;
  if (ctx.needsLeap) s[LACT.LEAP] += 0.8;
  if (ctx.coolingDown) {
    s[LACT.FORWARD] += 0.7;
    s[LACT.NOOP] += 0.2;
  }
  let am = 0;
  for (let i = 1; i < nA; i++) if (s[i] > s[am]) am = i;
  return soft(am, s, 0.5);
}

export { soft };
