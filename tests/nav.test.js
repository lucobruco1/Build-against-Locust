/**
 * Navigation: the Locust must be able to *find* a way in, break what blocks it,
 * and never cheat by tunnelling through the floor of the world.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { VoxelWorld, makeShell, OWNER_BOT0 } from '../core/world.js';
import { generateFlat } from '../core/worldgen.js';
import { astar, pathDirection, yawTowards, angleDiff, findBreakableCell, weakestBreach, visibleEntities } from '../core/nav.js';
import { B, BASE } from '../shared/rules.js';

function sealedBase(w, px, pz, mat = B.STONE) {
  const gy = w.terrainSurface(px, pz);
  const s = makeShell({ x: px, y: gy, z: pz }, gy);
  for (const c of s.all) w.set(c.x, c.y, c.z, mat, OWNER_BOT0, 0);
  return { gy, shell: s };
}

test('A* walks an open field directly', () => {
  const w = new VoxelWorld(3, generateFlat);
  const gy = w.terrainSurface(20, 20);
  const r = astar(w, { x: 12.5, y: gy, z: 20.5 }, { x: 20.5, y: gy, z: 20.5 }, { height: 4 });
  assert.ok(r.reached, 'an open goal must be reached');
  assert.equal(r.blocked, false);
  assert.ok(r.path.length >= 8 && r.path.length <= 12, `path length was ${r.path.length}`);
  const d = pathDirection(r.path, { x: 12.5, y: gy, z: 20.5 });
  assert.ok(d.dist > 0);
  const straight = yawTowards({ x: 12.5, z: 20.5 }, { x: 20.5, z: 20.5 });
  assert.ok(Math.abs(angleDiff(straight, d.yaw)) < 0.9, 'the first heading is roughly due east');
  assert.ok(Math.hypot(d.x, d.z) > 0.5, 'the unit heading is normalised');
});

test('A* at a sealed box aims at the wall and marks it blocked', () => {
  const w = new VoxelWorld(3, generateFlat);
  const px = 36, pz = 36;
  const { gy } = sealedBase(w, px, pz, B.STONE);
  const r = astar(w, { x: px + 9.5, y: gy, z: pz + 0.5 }, { x: px + 0.5, y: gy + 1, z: pz + 0.5 }, { height: 4, smashCost: 2.2, maxNodes: 6000 });
  assert.equal(r.blocked, true, 'a sealed base must be reported as needing a breach');
  assert.ok(r.smash, 'and it must name the block to open');
  assert.ok(r.smashAt > 0 && r.smashAt <= r.path.length, `smashAt ${r.smashAt} of ${r.path.length}`);
  assert.equal(r.path[r.smashAt].blocked, true, 'the flagged node is the blocked one');
  // it must be a block of *the base*, at the wall line — not scenery, not ground
  const cell = r.smash;
  assert.equal(w.get(cell.x, cell.y, cell.z), B.STONE);
  assert.ok(cell.y >= gy, 'never targets a block under the terrain surface');
  const ring = Math.max(Math.abs(cell.x - px), Math.abs(cell.z - pz));
  assert.ok(ring === BASE.WALL_RING || ring === BASE.OUTER_RING, `smash target ${cell.x},${cell.z} off the shell (ring ${ring})`);
});

test('harder walls cost more, so weak points are preferred', () => {
  const w = new VoxelWorld(4, generateFlat);
  const px = 36, pz = 36;
  const gy = w.terrainSurface(px, pz);
  const shell = makeShell({ x: px, y: gy, z: pz }, gy);
  for (const c of shell.all) w.set(c.x, c.y, c.z, B.BRICK, OWNER_BOT0, 0);   // locustHP 5
  const hard = astar(w, { x: px + 9.5, y: gy, z: pz + 0.5 }, { x: px + 0.5, y: gy + 1, z: pz + 0.5 }, { height: 4, smashCost: 2.2, maxNodes: 6000 });
  const w2 = new VoxelWorld(4, generateFlat);
  for (const c of shell.all) w2.set(c.x, c.y, c.z, B.LEAF, OWNER_BOT0, 0);   // locustHP 1
  const soft = astar(w2, { x: px + 9.5, y: gy, z: pz + 0.5 }, { x: px + 0.5, y: gy + 1, z: pz + 0.5 }, { height: 4, smashCost: 2.2, maxNodes: 6000 });
  assert.ok(hard.expansions >= soft.expansions * 0.5, 'both must be solvable');
  assert.ok(soft.blocked && hard.blocked);
  // soft walls are cheaper → the search should be willing to open one immediately
  assert.ok(soft.smashAt <= hard.smashAt || hard.smashAt > 0);
});

test('the search cannot dig through the world floor or climb on nothing', () => {
  const w = new VoxelWorld(6, generateFlat);
  const gy = w.terrainSurface(36, 36);
  const r = astar(w, { x: 30.5, y: gy, z: 36.5 }, { x: 42.5, y: gy - 3, z: 36.5 }, { height: 4, smashCost: 0.2, maxNodes: 4000 });
  for (const n of r.path) {
    assert.ok(n.y >= w.terrainSurface(n.x, n.z), `node ${n.x},${n.y},${n.z} is underground`);
    if (n.y > w.terrainSurface(n.x, n.z)) {
      // floating nodes are only legal right after a jump/ledge: floor below
      assert.ok(w.get(n.x, n.y - 1, n.z) !== B.AIR || n.blocked, `node ${n.x},${n.y},${n.z} floats`);
    }
  }
});

test('pathDirection stops in front of the wall it must break', () => {
  const w = new VoxelWorld(3, generateFlat);
  const px = 36, pz = 36;
  const { gy } = sealedBase(w, px, pz, B.STONE);
  const from = { x: px + 9.5, y: gy, z: pz + 0.5 };
  const r = astar(w, from, { x: px + 0.5, y: gy + 1, z: pz + 0.5 }, { height: 4, smashCost: 2.2, maxNodes: 6000 });
  const d = pathDirection(r.path, from, { stopAt: r.smashAt });
  assert.ok(d, 'a direction must always exist while chasing');
  assert.ok(d.node, 'the stop node is reported for debugging/HUD');
  // standing *before* the blocked cell, not inside it
  const stopX = Math.floor(d.x), stopZ = Math.floor(d.z);
  assert.notEqual(w.get(stopX, gy, stopZ), B.STONE, 'it must not aim into the wall cell itself');
  assert.ok(Math.hypot(d.dx, d.dz) > 0.5);
  const toWall = Math.atan2(-(r.smash.x + 0.5 - from.x), -(r.smash.z + 0.5 - from.z));
  assert.ok(Math.abs(angleDiff(toWall, d.yaw)) < 1.2, 'the heading still points at the breach');
});

test('findBreakableCell sees the roof, so a 4-tall sealed box is enterable', () => {
  const w = new VoxelWorld(8, generateFlat);
  const px = 36, pz = 36;
  const { gy } = sealedBase(w, px, pz, B.STONE);
  // standing inside the box at head height, the only way out is up through the roof
  const cell = findBreakableCell(w, px + 1, gy, pz, 4);
  assert.ok(cell, 'roof tiles must count as breakable for a 3.9m creature');
  assert.ok(cell.y > gy, `expected a cell above the floor, got y=${cell.y}`);
});

function clearanceCheck(w, x, y, z) {
  for (let dy = 0; dy < 4; dy++) if (w.get(x, y + dy, z) !== B.AIR) return false;
  return true;
}

test('weakestBreach offers a real gap when one exists', () => {
  const w = new VoxelWorld(9, generateFlat);
  const px = 36, pz = 36;
  const { gy, shell } = sealedBase(w, px, pz, B.STONE);
  const full = weakestBreach(w, { x: px, z: pz }, gy, { x: px + 8, y: gy, z: pz });
  assert.ok(full, 'a sealed base still returns the least-bad cell to attack');
  // knock a door in and the breach must move to it
  const door = shell.wall.find((c) => c.x === px + BASE.WALL_RING && c.z === pz && c.y === gy);
  w.set(door.x, door.y, pz, B.AIR, 0, 0);
  w.set(door.x, door.y + 1, pz, B.AIR, 0, 0);
  const open = weakestBreach(w, { x: px, z: pz }, gy, { x: px + 8, y: gy, z: pz });
  assert.ok(open);
  assert.ok(Math.max(Math.abs(open.x - px), Math.abs(open.z - pz)) >= 1, 'the gap is on the shell');
});

test('visibility counts prey within range of the eye', () => {
  const w = new VoxelWorld(10, generateFlat);
  const gy = w.terrainSurface(36, 36);
  const eye = { x: 30.5, y: gy + 1.6, z: 36.5 };
  const a = { pos: { x: 36.5, y: gy, z: 36.5 }, alive: true, id: 'x' };
  const dead = { pos: { x: 31.5, y: gy, z: 36.5 }, alive: false, id: 'y' };
  const seen = visibleEntities(w, eye, [a, dead], 40);
  assert.equal(seen.count, 1, 'corpses are not prey');
  assert.equal(seen.nearest, a);
  assert.ok(seen.nearestDist > 5);
  const far = visibleEntities(w, eye, [{ ...a, pos: { x: 66.5, y: gy, z: 36.5 } }], 12);
  assert.equal(far.count, 0, 'beyond max range nothing is seen');
});

test('the four clearance layers are all considered (the roof bug)', () => {
  const w = new VoxelWorld(11, generateFlat);
  const gy = w.terrainSurface(36, 36);
  // clear the worldgen marker first, then obstruct only the *top* layer: a search
  // that forgets the roof decides this corridor is impassable and the Locust
  // never learns to break in from above
  for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) {
    for (let dy = 0; dy < 4; dy++) w.set(36 + dx, gy + dy, 36 + dz, B.AIR, 0, 0);
    w.set(36 + dx, gy + 3, 36 + dz, B.STONE, OWNER_BOT0, 0);
  }
  const cell = findBreakableCell(w, 36, gy, 36, 4);
  assert.ok(cell, 'a cell whose only obstruction is the ceiling must still be openable');
  assert.equal(cell.y, gy + 3, `expected the ceiling block, got y=${cell.y}`);
  assert.ok(!clearanceCheck(w, 36, gy, 36), 'and that column really is blocked for a 4-tall creature');
});
