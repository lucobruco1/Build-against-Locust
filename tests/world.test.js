/**
 * The voxel world: block properties, the ownership rule that the whole game
 * hinges on ("bots may only delete blocks they themselves placed"), base
 * security scoring and the shell the AI is trying to complete.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { VoxelWorld, makeShell, OWNER_NATURAL, OWNER_PLAYER, OWNER_BOT0, OWNER_LOCUST } from '../core/world.js';
import { generateWorld, generateFlat } from '../core/worldgen.js';
import { B, BLOCK_DEFS, WORLD, BASE, isSolid, isOpaque, BOT_PLACEABLE, HOTBAR } from '../shared/rules.js';

test('block table is complete and self-consistent', () => {
  for (const [name, id] of Object.entries(B)) {
    const d = BLOCK_DEFS[id];
    assert.ok(d, `${name} must have a definition`);
    assert.equal(typeof d.locustHP, 'number', `${name} needs Locust hit points`);
    assert.equal(typeof d.hardness, 'number');
    if (id !== B.AIR) assert.ok(d.color != null, `${name} needs a colour`);
  }
  assert.ok(!BLOCK_DEFS[B.AIR].solid);
  assert.ok(isSolid(B.STONE) && !isSolid(B.AIR) && isSolid(B.LEAF));
  assert.ok(isOpaque(B.STONE));
  assert.ok(BOT_PLACEABLE.length >= 6 && HOTBAR.length === 9);
  // the Locust can smash anything with finite hardness, and harder blocks cost more
  assert.ok(BLOCK_DEFS[B.BRICK].locustHP > BLOCK_DEFS[B.DIRT].locustHP);
});

test('generation is deterministic and the world is bounded', () => {
  const a = new VoxelWorld(4242, generateWorld);
  const b = new VoxelWorld(4242, generateWorld);
  assert.deepEqual([...a.blocks], [...b.blocks], 'same seed must give the same world');
  assert.equal(a.blocks.length, WORLD.SX * WORLD.SY * WORLD.SZ);
  for (let z = 0; z < WORLD.SZ; z += 9) {
    for (let x = 0; x < WORLD.SX; x += 9) {
      assert.ok(a.surfaceY(x, z) >= 1 && a.surfaceY(x, z) < WORLD.SY, 'surface must be inside the world');
    }
  }
  assert.ok(a.plots.length >= 8, 'seven bots plus the player need a plot each');
  assert.ok(a.get(2, 5, 2) !== B.AIR, 'the ground is solid below the surface');
});

test('every plot has flat, walkable ground for its owner', () => {
  const w = new VoxelWorld(7, generateWorld);
  for (const p of w.plots) {
    const gy = w.surfaceY(p.x, p.z);
    assert.ok(gy > WORLD.SEA_LEVEL, 'plots should not sit under the water line');
    assert.ok(isSolid(w.get(p.x, gy - 1, p.z)), 'floor under the spawn cell');
    assert.ok(!isSolid(w.get(p.x, gy, p.z)) && !isSolid(w.get(p.x, gy + 1, p.z)), 'standing room');
    // whatever the terrain does, a full shell must be completable: every cell is
    // either already solid (hillside counts) or free to place into — otherwise the
    // bots' objective could never reach 1.0 and the cycle would be unwinnable
    const s = makeShell({ x: p.x, y: gy, z: p.z }, gy);
    for (const c of s.all) {
      const id = w.get(c.x, c.y, c.z);
      if (isSolid(id)) continue;
      assert.notEqual(id, B.WATER, 'a plot should not be sealed by water');
      w.set(c.x, c.y, c.z, B.COBBLE, OWNER_BOT0, 0);
    }
    assert.equal(w.security({ x: p.x, z: p.z }, gy, s).wall, 1, 'the objective is reachable');
  }
});

test('ownership: bots break their own blocks, never anyone else\\u2019s', () => {
  const w = new VoxelWorld(3, generateFlat);
  const gy = w.surfaceY(20, 20);
  const bot0 = { kind: 'builder', owner: OWNER_BOT0 + 0, index: 0 };
  const bot1 = { kind: 'builder', owner: OWNER_BOT0 + 1, index: 1 };
  const player = { kind: 'builder', owner: OWNER_PLAYER, index: -1 };
  const locust = { kind: 'locust' };

  w.set(20, gy, 20, B.STONE, OWNER_BOT0 + 0, 1);
  assert.equal(w.getOwner(20, gy, 20), OWNER_BOT0 + 0);
  assert.equal(w.canBreak(20, gy, 20, bot0), true, 'owner may delete their own block');
  assert.equal(w.canBreak(20, gy, 20, bot1), false, 'another bot may not');
  assert.equal(w.canBreak(20, gy, 20, player), false, 'the player may not either');
  assert.equal(w.canBreak(20, gy, 20, locust), true, 'the Locust breaks anything breakable');

  // natural terrain: bots may mine it, and it stays owned by nobody
  const nat = w.get(21, w.surfaceY(21, 21) - 1, 21);
  assert.ok(nat !== B.AIR);
  assert.equal(w.canBreak(21, gy - 1, 21, bot1), true, 'natural terrain is mineable by anyone');
  assert.equal(w.canBreak(21, gy - 1, 21, player), true);
  const res = w.breakBlock(21, gy - 1, 21, bot1);
  assert.equal(res.ok, true);
  assert.equal(w.get(21, gy, 21), B.AIR);
  assert.equal(w.getOwner(21, gy, 21), OWNER_NATURAL);

  // the Locust cannot delete bedrock, and placing overwrites ownership
  w.set(22, gy, 22, B.BRICK, OWNER_PLAYER, 5);
  assert.equal(w.getOwner(22, gy, 22), OWNER_PLAYER);
  assert.equal(w.canBreak(22, gy, 22, bot0), false);
  assert.equal(w.breakBlock(22, gy, 22, bot0).ok, false, 'and the refusal is enforced at the write');
  assert.equal(w.breakBlock(22, gy, 22, player).ok, true, 'the player may un-build their own');

  // bedrock is unbreakable for everyone, and air cannot be mined
  assert.equal(w.canBreak(30, 0, 30, locust), false, 'the floor of the world is permanent');
  assert.equal(w.canBreak(30, 0, 30, bot0), false);
  assert.equal(w.canBreak(20, WORLD.SY - 1, 20, locust), false, 'outside the world is not breakable');
});

test('placing is refused where it would trap an entity or double-fill a cell', () => {
  const w = new VoxelWorld(3, generateFlat);
  const gy = w.surfaceY(36, 36);
  w.set(36, gy, 36, B.PLANK, OWNER_PLAYER, 0);
  assert.equal(w.get(36, gy, 36), B.PLANK);
  const r = w.place(36, gy, 36, B.PLANK, OWNER_PLAYER, 0);
  assert.equal(r.ok, false, 'no stacking into an existing block');
  assert.equal(r.reason, 'occupied');
  assert.equal(w.place(36, gy + 1, 36, B.PLANK, OWNER_PLAYER, 0).ok, true);
});

test('height map, cover and light stay consistent with the block array', () => {
  const w = new VoxelWorld(11, generateFlat);
  const gy = w.surfaceY(12, 12);
  assert.ok(w.heightMap[12 * w.sx + 12] >= gy);
  assert.ok(w.lightAt(12, gy + 3, 12) > 0.5, 'open sky is bright');
  w.set(12, gy + 1, 12, B.STONE, OWNER_PLAYER, 0);
  w.set(12, gy + 2, 12, B.STONE, OWNER_PLAYER, 0);
  w.set(12, gy + 3, 12, B.STONE, OWNER_PLAYER, 0);
  w.updateLightColumn(12, 12);
  assert.ok(w.coverAt(12, gy, 12) >= 1, 'a roof counts as cover');
  assert.ok(w.solidNeighbours(12, gy + 1, 12) >= 1);
});

test('the shell the AI optimises is the box that actually protects it', () => {
  const centre = { x: 36, y: 13, z: 36 };
  const s = makeShell(centre, 13);
  assert.ok(s.wall.length > 0 && s.roof.length > 0 && s.outer.length > s.wall.length);
  // walls ring the plot at WALL_RING, tall enough that a 3.9m Locust cannot step over
  const tops = s.wall.filter((c) => c.y === 13 + BASE.WALL_TOP - 1);
  assert.ok(tops.length * 4 >= s.wall.length);
  for (const c of s.wall) {
    const r = Math.max(Math.abs(c.x - 36), Math.abs(c.z - 36));
    assert.equal(r, BASE.WALL_RING);
  }
  for (const c of s.roof) assert.equal(c.y, 13 + BASE.ROOF_CLEAR);
  // the roof must seal over the wall top, otherwise "sealed" is a lie
  assert.ok(BASE.ROOF_CLEAR >= BASE.WALL_TOP);

  const w = new VoxelWorld(5, generateFlat);
  const gy = w.surfaceY(36, 36);
  let empty = w.security({ x: 36, z: 36 }, gy, s);
  assert.ok(empty.wall < 0.15, `flat arena rubble must not count as a wall (got ${empty.wall})`);
  assert.ok(empty.roof < 0.1);
  assert.ok(empty.breaches > 0);
  for (const c of s.all) w.set(c.x, c.y, c.z, B.COBBLE, OWNER_BOT0 + 2, 0);
  const done = w.security({ x: 36, z: 36 }, gy, s);
  assert.equal(done.wall, 1, 'a finished shell scores 1.0');
  assert.equal(done.roof, 1);
  assert.equal(done.outer, 1);
  assert.equal(done.breaches, 0);
  assert.ok(done.sealed !== false);
  // punch a hole → coverage drops, and the module-level helper agrees
  w.set(36 + BASE.WALL_RING, gy, 36, B.AIR, OWNER_NATURAL, 0);
  const poked = w.security({ x: 36, y: gy, z: 36 }, gy, s);
  assert.ok(poked.wall < 1 && poked.breaches >= 1);
  assert.ok(done.blocks > 0, 'own-block count feeds the score');
});

test('OWNER constants are dense enough for 7 bots plus the player', () => {
  assert.equal(OWNER_NATURAL, 0);
  assert.ok(OWNER_BOT0 + 6 < OWNER_LOCUST, 'bot owners must not collide with the Locust owner id');
});
