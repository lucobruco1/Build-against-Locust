/**
 * Physics and block interaction: the rules the game is played with. A body must
 * collide with the grid, climb one-block steps, be hurt by falls, swim, and — the
 * regression that matters most here — move in the direction its yaw points.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { VoxelWorld, OWNER_BOT0, OWNER_PLAYER, OWNER_NATURAL } from '../core/world.js';
import { generateFlat } from '../core/worldgen.js';
import { makeEntity, stepEntity, groundSnap, boxCollides, aabb, applyMoveIntent, jump, distance3 } from '../core/physics.js';
import { raycast, dirFromAngles, placeTarget, hasLineOfSight } from '../core/raycast.js';
import { integrateIntent } from '../game/actions.js';
import { B, ENTITY } from '../shared/rules.js';

const flat = () => new VoxelWorld(17, generateFlat);
/* A hand-carved hall with a perfectly level floor at y=GY, so movement tests
   measure the physics and not the terrain generator. */
const GY = 12;
function hall(w = flat(), gy = GY) {
  for (let x = 28; x <= 46; x++) {
    for (let z = 32; z <= 40; z++) {
      for (let y = 0; y <= gy - 1; y++) w.set(x, y, z, B.STONE, OWNER_NATURAL, 0);
      for (let y = gy; y <= gy + 6; y++) w.set(x, y, z, B.AIR, 0, 0);
    }
  }
  return w;
}
/* physics.makeEntity is positional: (kind, x, y, z) */
const ent = (kind, x, y, z) => makeEntity(kind, x, y, z);
/* physics.makeEntity is positional: (kind, x, y, z) */
const STEP = 1 / 30;

test('a body falls, lands on the ground and stays on top of it', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  const e = ent('builder', 36.5, gy + 3, 36.5);
  let landed = false;
  for (let i = 0; i < 60 && !landed; i++) landed = stepEntity(w, e, STEP).landed;
  assert.ok(landed, 'it must land within a second of falling');
  assert.ok(Math.abs(e.pos.y - gy) < 0.05, `ended at y=${e.pos.y}, ground is ${gy}`);
  assert.equal(e.onGround, true);
  for (let i = 0; i < 20; i++) stepEntity(w, e, STEP);
  assert.ok(Math.abs(e.pos.y - gy) < 0.05, 'resting contact is stable');
});

test('it cannot walk through a wall', () => {
  const w = hall(flat());
  const e = ent('builder', 35.5, GY, 36.5);
  for (let dz = -2; dz <= 2; dz++) for (let dy = 0; dy < 4; dy++) w.set(38, GY + dy, 36 + dz, B.STONE, OWNER_BOT0, 0);
  for (let i = 0; i < 40; i++) { e.vel.x = 6; stepEntity(w, e, STEP); }
  assert.ok(e.pos.x < 38 - 0.29, `it stopped at x=${e.pos.x.toFixed(2)}, the wall face is at 38`);
  assert.ok(e.pos.x > 36.2, `it must actually have approached the wall (x=${e.pos.x.toFixed(2)})`);
  assert.equal(w.get(38, GY, 36), B.STONE, 'and the wall is still standing');
  assert.equal(e.onGround, true, 'walking must not lift it off the floor');
});

test('the Locust strides over one-block rubble; builders must jump', () => {
  const w = hall(flat());
  w.set(37, GY, 36, B.STONE, OWNER_BOT0, 0);       // a 1-block rubble step
  const e = ent('builder', 35.5, GY, 36.5);
  for (let i = 0; i < 40; i++) { e.vel.x = 5; stepEntity(w, e, STEP); }
  assert.ok(e.pos.y < GY + 0.5 && e.pos.x < 37 - 0.2, 'a player-height body stops dead at the rubble');

  const l = ent('locust', 35.5, GY, 36.5);
  for (let i = 0; i < 4; i++) stepEntity(w, l, STEP);            // settle
  let top = l.pos.y, crossed = false;
  for (let i = 0; i < 40; i++) {
    l.vel.x = 5;
    stepEntity(w, l, STEP);
    top = Math.max(top, l.pos.y);
    if (l.pos.x > 37.5 && l.pos.y > GY + 0.4) crossed = true;    // on top of the rubble
  }
  assert.ok(top > GY + 0.4, `it lifted to y=${top.toFixed(2)} (floor is ${GY})`);
  assert.ok(crossed, 'and walked over the top of it, not around');

  for (let dy = 0; dy < 3; dy++) w.set(39, GY + 1 + dy, 36, B.STONE, OWNER_BOT0, 0);
  const l2 = ent('locust', 37.5, GY + 1, 36.5);
  for (let i = 0; i < 4; i++) stepEntity(w, l2, STEP);
  for (let i = 0; i < 40; i++) { l2.vel.x = 5; stepEntity(w, l2, STEP); }
  assert.ok(l2.pos.x < 39 - 0.4, 'a 4-block-high wall still stops it: that is what walls are for');
});

test('jumping clears a step, and the Locust leaps much higher', () => {
  const w = hall(flat());
  const e = ent('builder', 36.5, GY, 36.5);
  for (let i = 0; i < 8; i++) stepEntity(w, e, STEP);
  assert.equal(e.onGround, true, 'it must be standing before jumping');
  assert.ok(jump(e), 'a grounded body can jump');
  let peak = e.pos.y;
  for (let i = 0; i < 60; i++) { stepEntity(w, e, STEP); peak = Math.max(peak, e.pos.y); }
  // v²/2g = 8.6²/60 ≈ 1.23 blocks: a hop over rubble, never over a wall
  assert.ok(peak - GY > 0.9 && peak - GY < 1.7, `jump height ${(peak - GY).toFixed(2)}`);
  assert.equal(e.onGround, true, 'and gravity brings it back down');

  const l = ent('locust', 36.5, GY, 36.5);
  for (let i = 0; i < 8; i++) stepEntity(w, l, STEP);
  assert.ok(jump(l), 'the Locust can leap too');
  let lPeak = l.pos.y;
  for (let i = 0; i < 60; i++) { stepEntity(w, l, STEP); lPeak = Math.max(lPeak, l.pos.y); }
  assert.ok(lPeak - GY > 1.9, `its leap must clear a 2-block lip, got ${(lPeak - GY).toFixed(2)}`);
});

test('falling a long way hurts; a short step does not', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  const e = ent('builder', 36.5, gy + 22, 36.5);
  e.health = 20;
  let dmg = 0;
  for (let i = 0; i < 120; i++) dmg += stepEntity(w, e, STEP).fallDamage || 0;
  assert.ok(dmg > 0, 'a 22 block drop must hurt');
  const e2 = ent('builder', 36.5, gy + 1.2, 36.5);
  let dmg2 = 0;
  for (let i = 0; i < 40; i++) dmg2 += stepEntity(w, e2, STEP).fallDamage || 0;
  assert.equal(dmg2, 0, 'a small hop is free');
});

test('water is soft, slows you and lets you swim up', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  for (let y = gy; y < gy + 3; y++) w.set(36, y, 36, B.WATER, 0, 0);
  const e = ent('builder', 36.5, gy, 36.5);
  for (let i = 0; i < 20; i++) stepEntity(w, e, STEP);
  assert.equal(e.inWater, true, 'the body knows it is submerged');
  const before = e.pos.y;
  e.vel.y = 4;
  for (let i = 0; i < 10; i++) stepEntity(w, e, STEP);
  assert.ok(e.pos.y > before - 0.5, 'swimming up works (no drowning sink)');
  assert.ok(Math.abs(e.vel.x) < 6);
});

test('forward means forward: intent + yaw produce motion along the facing vector', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  // This is the regression that used to matter: the yaw→velocity mapping has to
  // agree with dirFromAngles, or every agent walks sideways from its own aim.
  const cases = [0, Math.PI / 2, Math.PI, -Math.PI / 2, 0.7];
  for (const yaw of cases) {
    const e = ent('builder', 36.5, gy, 36.5);
    const a = { ent: e, intent: { mx: 0, mz: -1, turn: 0, pitch: 0, sprint: 0 }, kind: 'builder', grabbed: false };
    e.yaw = yaw;
    const p0 = { x: e.pos.x, z: e.pos.z };
    for (let i = 0; i < 12; i++) { integrateIntent(a, STEP); stepEntity(w, e, STEP); }
    const moved = { x: e.pos.x - p0.x, z: e.pos.z - p0.z };
    const len = Math.hypot(moved.x, moved.z);
    assert.ok(len > 0.35, `yaw ${yaw}: it barely moved (${len.toFixed(3)})`);
    const want = dirFromAngles(yaw, 0);
    const dot = (moved.x * want.x + moved.z * want.z) / len;
    assert.ok(dot > 0.97, `yaw ${yaw}: moved ${(dot).toFixed(2)}·facing — must be along the aim`);
  }
  // strafing is 90° right of the aim
  const e = ent('builder', 36.5, gy, 36.5);
  e.yaw = 0;
  const a = { ent: e, intent: { mx: 1, mz: 0, turn: 0, pitch: 0, sprint: 0 }, kind: 'builder', grabbed: false };
  const p0 = { x: e.pos.x, z: e.pos.z };
  for (let i = 0; i < 12; i++) { integrateIntent(a, STEP); stepEntity(w, e, STEP); }
  assert.ok(e.pos.x - p0.x > 0.2, 'strafe right goes +x when facing -z');
  assert.ok(Math.abs(e.pos.z - p0.z) < 0.25, 'and not forwards');
});

test('applyMoveIntent agrees with integrateIntent', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  const e = ent('builder', 36.5, gy, 36.5);
  e.yaw = 0.6;
  applyMoveIntent(e, 1, 0, -1);
  const d = dirFromAngles(0.6, 0);
  const v = Math.hypot(e.vel.x, e.vel.z) || 1;
  assert.ok((e.vel.x / v) * d.x + (e.vel.z / v) * d.z > 0.98, 'the physics helper uses the same convention');
});

test('turn intent rotates the head, pitch is clamped', () => {
  const e = ent('builder', 0.5, 5, 0.5);
  const a = { ent: e, intent: { mx: 0, mz: 0, turn: 1, pitch: 1, sprint: 0 }, kind: 'builder', grabbed: false };
  const y0 = e.yaw;
  integrateIntent(a, 0.1);
  assert.notEqual(e.yaw, y0, 'a turn actually turns');
  for (let i = 0; i < 200; i++) integrateIntent(a, 0.1);
  assert.ok(e.pitch <= 1.1 + 1e-6 && e.pitch >= -1.3, `pitch escaped its clamp: ${e.pitch}`);
});

test('a grabbed body loses control but is not frozen in the air', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  const e = ent('builder', 36.5, gy + 1, 36.5);
  e.grabbed = true;
  e.vel.x = 9;
  const ev = stepEntity(w, e, STEP);
  assert.equal(ev.landed, false);
  assert.equal(e.vel.x, 0, 'no control while held');
  assert.equal(e.onGround, false);
});

test('groundSnap seats a body on the surface, and aabb queries agree', () => {
  const w = flat();
  const e = ent('builder', 20.5, 35, 20.5);
  groundSnap(w, e);
  assert.ok(Math.abs(e.pos.y - w.surfaceY(20, 20)) < 1.2, `snapped to ${e.pos.y}`);
  const box = aabb(e);
  assert.ok(box.maxY - box.minY >= e.height - 0.01, 'aabb spans the body height');
  assert.ok(Math.abs((box.maxX - box.minX) - e.width) < 1e-6);
  const gy = w.surfaceY(20, 20);
  assert.equal(boxCollides(w, 20.5, gy + 1, 20.5, e.width / 2, e.height), false, 'standing in the open does not collide');
  assert.equal(boxCollides(w, 20.5, gy - 2, 20.5, e.width / 2, e.height), true, 'two blocks down it is inside rock');
  assert.ok(distance3({ x: 0, y: 0, z: 0 }, { x: 1, y: 2, z: 2 }) === 3);
});

/* -------------------------------------------------------------- raycast */

test('DDA raycast finds the first solid cell, its face and the place target', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  for (let dy = 0; dy < 3; dy++) for (let dz = -1; dz <= 1; dz++) w.set(40, gy + dy, 36 + dz, B.STONE, OWNER_BOT0, 0);
  const eye = { x: 36.5, y: gy + 1.6, z: 36.5 };
  const yaw = -Math.PI / 2;                       // -x is west, so this is due east
  const d = dirFromAngles(yaw, 0);
  assert.ok(d.x > 0.99, 'sanity: this yaw faces +x');
  const hit = raycast(w, eye.x, eye.y, eye.z, d.x, d.y, d.z, 8);
  assert.equal(hit.hit, true, 'looking east at a wall must hit it');
  assert.equal(hit.x, 40);
  assert.equal(hit.nx, -1, 'the normal points back at the shooter');
  const t = placeTarget(w, eye.x, eye.y, eye.z, yaw, 0, 5);
  assert.deepEqual(t.breakCell, { x: 40, y: hit.y, z: 36 });
  assert.equal(t.place.x, 39, 'placement goes on the near side of the face');
  assert.ok(t.dist > 3 && t.dist < 5);
  // looking at the sky: no hit, but a free-standing fallback cell is offered
  const sky = placeTarget(w, eye.x, eye.y, eye.z, 0, 0.9, 5);
  assert.equal(sky.breakCell, null);
  assert.ok(sky.place);
});

test('line of sight is blocked by the same grid the renderer draws', () => {
  const w = flat();
  const gy = w.surfaceY(36, 36);
  const a = { x: 34.5, y: gy + 1.5, z: 36.5 };
  const b = { x: 42.5, y: gy + 1.5, z: 36.5 };
  assert.equal(hasLineOfSight(w, a, b, 20), true, 'open field of view');
  for (let dy = 0; dy < 4; dy++) for (let dz = -2; dz <= 2; dz++) w.set(38, gy + dy, 36 + dz, B.STONE, OWNER_BOT0, 0);
  assert.equal(hasLineOfSight(w, a, b, 20), false, 'a wall blocks it');
  assert.equal(hasLineOfSight(w, a, b, 2), false, 'and so does range');
  w.set(38, gy + 1, 36, B.AIR, 0, 0);
  assert.equal(hasLineOfSight(w, a, { x: 39.5, y: gy + 1.5, z: 36.5 }, 8), true, 'a hole lets it see through');
});
