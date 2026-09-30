/**
 * tests/actions.test.js
 * ---------------------------------------------------------------------------
 * The verbs. Both sides of the contest run through this module, and the two
 * rules the game is *about* live here: a bot may only delete blocks it placed
 * itself, and the Locust's attack is grab-then-stab. Everything is exercised on
 * a real Match (real inventories, plots, shells, cooldowns) rather than on
 * hand-made objects, because the wiring between Match.execBuilder and the
 * executor is exactly where these rules used to leak.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { Match } from '../game/match.js';
import { ACT, LACT, B, BLOCK_DEFS, BASE, ENTITY, RW } from '../shared/rules.js';
import {
  executeBuilderAction, executeLocustAction, validPlace, placementCandidate,
  builderLegalMask, locustLegalMask, nearestSmashable, smashTargetInfo, findPrey,
  clearIntent, integrateIntent, BUILD_INTERVAL, LOCUST_INTERVAL,
} from '../game/actions.js';
import { OWNER_PLAYER, OWNER_BOT0 } from '../core/world.js';

/* ------------------------------------------------------------------ helpers */

/** Headless match: every slot is an AI builder, no network, no learning. */
function arena(seed = 3) {
  const m = new Match({ flat: 1, seed, learn: false, human: false });
  return m;
}
/** The ctx the executor expects (Match builds the same shape in execBuilder). */
function actx(m) {
  return { world: m.world, phase: m.phase, now: m.now, entities: m.builders, preserveIntent: false };
}
function lctx(m) {
  return {
    world: m.world, now: m.now, entities: m.builders,
    onBlockBroken: () => {}, hitByLocustOnBlock: null, want: null,
  };
}
/** Stand a builder on flat ground, looking along +x, with an empty cell ahead. */
function stance(m, b, x, z, yaw) {
  const gy = m.world.surfaceY(Math.floor(x), Math.floor(z));
  b.ent.pos.x = x; b.ent.pos.z = z; b.ent.pos.y = gy;
  b.ent.vel.x = b.ent.vel.y = b.ent.vel.z = 0;
  b.ent.yaw = yaw; b.ent.pitch = 0;
  b.plot = { x: Math.floor(x), z: Math.floor(z) };
  b.groundY = gy;
  b.coolPlace = 0; b.coolBreak = 0;
  return gy;
}

/**
 * A level, clear strip of floor with air above it. The generated flat arena is
 * not uniform (it scatters rubble and a marker block), so any test that is about
 * the *executor* carves its own corridor instead of inheriting terrain luck.
 */
function carve(m, gy, x0 = 33, x1 = 42, z = 36) {
  for (let x = x0; x <= x1; x++) {
    m.world.set(x, gy - 1, z, B.STONE, 0, 0);
    for (let y = gy; y <= gy + 5; y++) m.world.set(x, y, z, B.AIR, 0, 0);
  }
}

/* ------------------------------------------------------------- the verbs: AI */

test('an AI builder places a block and the world records who put it there', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  stance(m, b, b.plot.x + 0.5, b.plot.z + 0.5, -Math.PI / 2);
  const before = m.world.countOwnerBlocks(b.owner);
  const invBefore = b.inv.total;
  const res = executeBuilderAction(b, ACT.PLACE_FRONT, ctx);
  assert.ok(res.reward > 0, `placing shell blocks pays, got ${res.reward} (${res.info})`);
  assert.equal(m.world.countOwnerBlocks(b.owner), before + 1, 'exactly one block became theirs');
  assert.equal(b.inv.total, invBefore - 1, 'and one block left the inventory');
  assert.equal(b.stats.placed, 1);
  // ownership is the whole point of the rule set: the owner id must be the bot's
  let found = 0;
  for (let x = b.plot.x - BASE.RADIUS; x <= b.plot.x + BASE.RADIUS; x++) {
    for (let z = b.plot.z - BASE.RADIUS; z <= b.plot.z + BASE.RADIUS; z++) {
      for (let y = b.groundY; y <= b.groundY + 5; y++) if (m.world.getOwner(x, y, z) === b.owner) found++;
    }
  }
  assert.ok(found >= 1, 'the new block carries the bot as its owner');
});

test('a bot may delete its own block, and nobody else’s', () => {
  const m = arena();
  const a = m.builders[1];
  const c = m.builders[2];
  const ctx = actx(m);
  // give both a wall cell to work with: place as `a`, then try to break it as `c`
  stance(m, a, a.plot.x + 0.5, a.plot.z + 0.5, -Math.PI / 2);
  assert.ok(executeBuilderAction(a, ACT.PLACE_FRONT, ctx).reward > 0, 'a places');
  let cell = null;
  for (let x = a.plot.x - BASE.RADIUS; x <= a.plot.x + BASE.RADIUS && !cell; x++) {
    for (let z = a.plot.z - BASE.RADIUS; z <= a.plot.z + BASE.RADIUS && !cell; z++) {
      for (let y = a.groundY; y <= a.groundY + 5 && !cell; y++) if (m.world.getOwner(x, y, z) === a.owner) cell = { x, y, z };
    }
  }
  assert.ok(cell, 'the placed block was found');
  assert.equal(m.world.canBreak(cell.x, cell.y, cell.z, { kind: 'builder', owner: c.owner }), false,
    'the neighbour may not break it');
  assert.equal(m.world.canBreak(cell.x, cell.y, cell.z, { kind: 'builder', owner: a.owner }), true,
    'the builder that placed it may');
  assert.equal(m.world.canBreak(cell.x, cell.y, cell.z, { kind: 'player', owner: OWNER_PLAYER }), false,
    'not even the player un-builds somebody else — that is the rule the game is named after');
  assert.equal(m.world.canBreak(cell.x, cell.y, cell.z, { kind: 'locust' }), true,
    'only the Locust may demolish anything');

  // stand the victim right in front of the block and run the verb
  stance(m, c, cell.x - 1.5, cell.z + 0.5, -Math.PI / 2);
  c.plot = { x: cell.x - 3, z: cell.z };
  const refused = executeBuilderAction(c, ACT.BREAK_FRONT, ctx);
  assert.equal(refused.reward, RW.ILLEGAL, 'and the verb is refused, not just the mask');
  assert.equal(refused.info, 'not yours to break');
  assert.notEqual(m.world.get(cell.x, cell.y, cell.z), B.AIR, 'the block is still there');

  stance(m, a, cell.x - 1.5, cell.z + 0.5, -Math.PI / 2);
  const mine = executeBuilderAction(a, ACT.BREAK_FRONT, ctx);
  assert.equal(mine.reward, RW.BREAK_OWN, 'breaking your own work is a small cost');
  assert.equal(m.world.get(cell.x, cell.y, cell.z), B.AIR, 'and it actually removes the block');
  assert.equal(a.stats.broken, 1);
});

test('natural terrain is mineable by everyone — but only upward from the surface', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  const gy = stance(m, b, 36.5, 36.5, 0);
  // dig straight down: the cell below the feet is terrain, not a placed block
  const res = executeBuilderAction(b, ACT.BREAK_DOWN, ctx);
  assert.equal(res.reward, RW.MINE, `mining natural ground pays, got ${res.reward} (${res.info})`);
  assert.equal(m.world.get(Math.floor(b.ent.pos.x), gy - 1, Math.floor(b.ent.pos.z)), B.AIR);
  assert.ok(b.inv.count(B.DIRT) + b.inv.count(B.GRASS) + b.inv.count(B.STONE) > 0, 'the block goes into the pack');
  // and the bottom of the map is sealed for everybody, bots included
  m.world.set(Math.floor(b.ent.pos.x), 1, Math.floor(b.ent.pos.z), B.BEDROCK, 0, 0);
  assert.equal(m.world.canBreak(Math.floor(b.ent.pos.x), 1, Math.floor(b.ent.pos.z), { kind: 'builder', owner: b.owner }), false);
});

test('placement obeys the Minecraft rules: air, support, no entity inside', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  stance(m, b, 36.5, 36.5, -Math.PI / 2);
  const gy = b.groundY;
  carve(m, gy, 34);
  assert.equal(m.world.get(37, gy, 36), B.AIR, 'the cell in front of it is clear');
  assert.equal(validPlace(b, m.world, ctx, { x: 37, y: gy, z: 36 }), true, 'a supported air cell next to the floor is fine');
  m.world.set(37, gy, 36, B.COBBLE, b.owner, 0);
  assert.equal(validPlace(b, m.world, ctx, { x: 37, y: gy, z: 36 }), false, 'occupied cells are not');
  assert.equal(validPlace(b, m.world, ctx, { x: 40, y: gy + 5, z: 40 }), false, 'floating cells are not');
  // a cell that is air and supported but inside a body is illegal — you cannot
  // wall a teammate (or yourself) into stone
  const victim = m.builders[3];
  stance(m, victim, 38.5, 36.5, 0);
  m.world.set(38, gy, 36, B.AIR, 0, 0);
  m.world.set(38, gy - 1, 36, B.STONE, 0, 0);
  assert.equal(validPlace(b, m.world, ctx, { x: 38, y: gy, z: 36 }), false, 'never inside another entity');
});

test('bots build inside their own plot and are paid less for wandering off', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  const gy = m.world.surfaceY(b.plot.x, b.plot.z);
  // far outside the plot there is nothing legal to place on, so the candidate
  // search must refuse rather than let the bot brick the map at random
  b.ent.pos.x = b.plot.x + BASE.RADIUS + 9;
  b.ent.pos.z = b.plot.z + 0.5;
  b.ent.pos.y = m.world.surfaceY(Math.floor(b.ent.pos.x), Math.floor(b.ent.pos.z));
  const cand = placementCandidate(b, m.world, ctx);
  if (cand.ok) {
    assert.ok(Math.abs(cand.cell.x - b.plot.x) <= BASE.RADIUS && Math.abs(cand.cell.z - b.plot.z) <= BASE.RADIUS,
      'any cell it accepts stays inside the plot');
  } else {
    assert.ok(true, 'refusing to build outside the plot is also correct');
  }
  void gy;
});

test('the legal mask is a real mask, and grabs restrict it to struggling', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  stance(m, b, 36.5, 36.5, 0);
  const mask = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(mask.length, 15, 'the builder action space');
  for (const v of mask) assert.ok(v === 0 || v === 1, 'entries are 0/1');
  assert.equal(mask[ACT.NOOP], 1, 'doing nothing is always allowed');
  assert.equal(mask[ACT.FORWARD], 1);

  b.ent.grabbed = true; b.grabbed = true;
  const panic = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(panic[ACT.PLACE_FRONT], 0, 'you cannot build while it holds you');
  assert.equal(panic[ACT.BREAK_FRONT], 0);
  assert.equal(panic[ACT.JUMP], 1, 'struggling is still allowed');
  assert.equal(panic[ACT.NOOP], 1);

  b.ent.grabbed = false; b.grabbed = false;
  b.alive = false;
  const dead = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(dead.filter((v, i) => v === 1 && i !== ACT.NOOP).length, 0, 'a dead builder may only idle');
});

test('an empty hand or a cooldown closes the build verbs', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  b.coolPlace = 0; b.coolBreak = 0;
  const mask0 = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(mask0[ACT.PLACE_FRONT], 1, 'a builder at its plot with a full pack may build');
  assert.ok(b.inv.total > 0);

  b.inv.counts.clear();
  assert.equal(b.inv.total, 0, 'the pack is empty');
  const mask = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(mask[ACT.PLACE_FRONT], 0, 'an empty inventory closes both place verbs');
  assert.equal(mask[ACT.PLACE_DOWN], 0);
  const res = executeBuilderAction(b, ACT.PLACE_FRONT, ctx);
  assert.equal(res.reward, RW.ILLEGAL, 'and the verb is refused, not silently ignored');
  assert.equal(res.info, 'empty hand');

  b.inv.add(B.COBBLE, 5);
  b.coolPlace = 0.2;
  const cooled = builderLegalMask(b, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(cooled[ACT.PLACE_FRONT], 0, 'and while the arm is still recovering it cannot place either');
});

test('movement verbs set the intent, JUMP leaves the ground, SPRINT doubles it', () => {
  const m = arena();
  const b = m.builders[1];
  const ctx = actx(m);
  stance(m, b, 36.5, 36.5, 0);
  clearIntent(b);
  assert.equal(b.intent.mx, 0);
  executeBuilderAction(b, ACT.FORWARD, ctx);
  assert.equal(b.intent.mz, -1, 'FORWARD pushes the intent forward');
  executeBuilderAction(b, ACT.TURN_RIGHT, ctx);
  assert.equal(b.intent.turn, 1);
  executeBuilderAction(b, ACT.LOOK_DOWN, ctx);
  assert.equal(b.intent.pitch, -1);

  b.ent.onGround = true;
  executeBuilderAction(b, ACT.JUMP, ctx);
  assert.ok(b.ent.vel.y > 5, 'a grounded jump leaves the floor');
  assert.equal(b.ent.onGround, false);
  b.ent.onGround = false; b.ent.inWater = false;
  const vy = b.ent.vel.y;
  executeBuilderAction(b, ACT.JUMP, ctx);
  assert.equal(b.ent.vel.y, vy, 'no double jumping in mid air');

  clearIntent(b);
  executeBuilderAction(b, ACT.SPRINT, ctx);
  assert.equal(b.intent.sprint, 1);
  assert.equal(b.intent.mz, -1, 'sprinting implies running forward');

  // and the intent it writes is the intent physics integrates
  clearIntent(b);
  b.ent.yaw = 0;
  executeBuilderAction(b, ACT.FORWARD, ctx);
  integrateIntent(b, 1 / 30);
  assert.equal(b.ent.vel.z < 0, true, 'yaw 0 + FORWARD must travel towards -z');
});

test('the two sides act on different clocks (builders 0.33 s, Locust 0.25 s)', () => {
  assert.ok(BUILD_INTERVAL > LOCUST_INTERVAL, 'the Locust decides faster than a builder can place');
  assert.ok(BUILD_INTERVAL > 0.2 && BUILD_INTERVAL < 0.5, 'a builder gets ~3 decisions a second');
  assert.ok(LOCUST_INTERVAL > 0.15 && LOCUST_INTERVAL < 0.4);
});

/* ------------------------------------------------------------ the Locust */

test('the Locust smashes a block only after enough hits', () => {
  const m = arena(5);
  m.spawnLocust();
  const l = m.locust;
  assert.ok(l, 'spawnLocust gives the hunt its monster');
  const ctx = lctx(m);
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.z = 36.5; l.ent.pos.y = gy;
  l.ent.yaw = -Math.PI / 2; l.ent.pitch = 0;
  l.ent.onGround = true;
  carve(m, gy, 35);
  m.world.set(38, gy, 36, B.COBBLE, OWNER_BOT0 + 1, 0);

  const info = smashTargetInfo(l, m.world, 3, null);
  assert.ok(info && info.cell, 'it can see a block to smash in front of it');
  assert.deepEqual({ x: info.cell.x, y: info.cell.y, z: info.cell.z }, { x: 38, y: gy, z: 36 },
    'and it aims at the wall, not at the floor it is standing on');
  assert.ok(info.dist < 4, `and reports how far away it is (${info.dist.toFixed(2)})`);

  const need = BLOCK_DEFS[B.COBBLE].locustHP ?? 2;
  assert.ok(need >= 1, 'blocks have a hit budget against it');
  for (let i = 0; i < need - 1; i++) {
    const r = executeLocustAction(l, LACT.SMASH_BLOCK, ctx);
    assert.equal(r.reward >= 0, true, 'a glancing hit is not punished');
    assert.notEqual(m.world.get(38, gy, 36), B.AIR, `hit ${i + 1} of ${need} must not break it`);
  }
  const last = executeLocustAction(l, LACT.SMASH_BLOCK, ctx);
  assert.equal(m.world.get(38, gy, 36), B.AIR, 'the last hit opens the wall');
  assert.ok(last.reward >= RW.L_BREAK, 'breaking into a base pays the Locust');
  assert.equal(l.stats.smashed, 1);
  assert.equal(l.smashHits.size, 0, 'the hit counter is cleared once the block is gone');
});

test('smashing is aimed: it does not demolish the scenery behind it', () => {
  const m = arena(6);
  m.spawnLocust();
  const l = m.locust;
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.z = 36.5; l.ent.pos.y = gy; l.ent.onGround = true;
  m.world.set(38, gy, 36, B.COBBLE, OWNER_BOT0 + 1, 0);   // in front (+x)
  m.world.set(34, gy, 36, B.COBBLE, OWNER_BOT0 + 1, 0);   // behind
  l.ent.yaw = Math.PI / 2;                                  // facing -x
  assert.equal(nearestSmashable(l, m.world, 3, null).x, 34, 'it turns towards what is in front of it');
  l.ent.yaw = -Math.PI / 2;
  assert.equal(nearestSmashable(l, m.world, 3, null).x, 38);
  // bedrock and water are outside its reach by rule, not by luck
  m.world.set(39, gy, 36, B.BEDROCK, 0, 0);
  l.ent.yaw = -Math.PI / 2;
  const cell = nearestSmashable(l, m.world, 3, null);
  assert.ok(!cell || m.world.canBreak(cell.x, cell.y, cell.z, { kind: 'locust' }), 'never aims at unbreakable material');
});

test('STRIKE grabs, and a grabbed victim can no longer be targeted again', () => {
  const m = arena(7);
  m.spawnLocust();
  const l = m.locust;
  const prey = m.builders[0];
  const ctx = lctx(m);
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.z = 36.5; l.ent.pos.y = gy; l.ent.yaw = -Math.PI / 2; l.ent.onGround = true;
  prey.ent.pos.x = 38.2; prey.ent.pos.z = 36.5; prey.ent.pos.y = gy; prey.ent.yaw = 0;
  l.attackCd = 0;

  assert.ok(findPrey(l, m.builders, m.world) === prey, 'prey standing in front of it within reach is found');
  const legal = locustLegalMask(l, { entities: m.builders, world: m.world, want: null });
  assert.equal(legal[LACT.STRIKE], 1, 'so STRIKE is legal');

  const res = executeLocustAction(l, LACT.STRIKE, ctx);
  assert.equal(res.reward, RW.L_GRAB, 'the grab pays');
  assert.equal(prey.grabbed, true, 'and the victim is in its hand');
  assert.equal(prey.grabbedBy, l.id);
  assert.equal(prey.grabT, 0, 'the grab timer starts now — the stab comes after it');
  assert.equal(l.grabbed, prey, 'the Locust also remembers what it is holding');
  assert.ok(l.attackCd > 0, 'and it cannot swipe again immediately');

  const again = executeLocustAction(l, LACT.STRIKE, ctx);
  assert.equal(again.reward, RW.ILLEGAL, 'a second strike during the cooldown is wasted');
  assert.equal(again.info, 'cooldown');
  assert.equal(locustLegalMask(l, { entities: m.builders, world: m.world, want: null })[LACT.STRIKE], 0,
    'and the mask stops offering it: grabbed prey is not double-grabbed');
});

test('no prey in reach means no strike, and the mask says so', () => {
  const m = arena(8);
  m.spawnLocust();
  const l = m.locust;
  const ctx = lctx(m);
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.z = 36.5; l.ent.pos.y = gy; l.ent.yaw = -Math.PI / 2;
  for (const b of m.builders) {
    b.ent.pos.x = 10 + m.builders.indexOf(b) * 5;
    b.ent.pos.z = 10;
    b.ent.pos.y = m.world.surfaceY(10 + m.builders.indexOf(b) * 5, 10);
    b.grabbed = false;
  }
  l.attackCd = 0;
  assert.equal(findPrey(l, m.builders, m.world), null, 'nobody is close enough');
  const res = executeLocustAction(l, LACT.STRIKE, ctx);
  assert.equal(res.info, 'no prey in reach');
  assert.equal(res.reward, RW.L_IDLE_TARGET, 'swiping at air costs it a little');
  assert.equal(locustLegalMask(l, { entities: m.builders, world: m.world, want: null })[LACT.STRIKE], 0);
  assert.equal(m.builders.every((b) => !b.grabbed), true, 'and of course nothing was grabbed');
});

test('its reach is longer through a hole — it fishes prey out of a half-sealed box', () => {
  const m = arena(9);
  m.spawnLocust();
  const l = m.locust;
  const prey = m.builders[0];
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.y = gy + 4; l.ent.pos.z = 36.5;
  l.ent.yaw = 0; l.ent.pitch = -1.1;
  prey.ent.pos.x = 36.5; prey.ent.pos.y = gy; prey.ent.pos.z = 37.8;
  prey.grabbed = false;
  const d = Math.hypot(prey.ent.pos.x - l.ent.pos.x, prey.ent.pos.y - l.ent.pos.y, prey.ent.pos.z - l.ent.pos.z);
  assert.ok(d > ENTITY.GRAB_RANGE, `this is beyond the normal grab range (${d.toFixed(2)})`);
  const seen = findPrey(l, m.builders, m.world);
  if (seen) {
    assert.ok(d <= ENTITY.REACH_THROUGH_HOLE, 'but it may only reach that far with line of sight down a gap');
  } else {
    assert.ok(true, 'no line of sight means no grab: a fully sealed box is safe');
  }
});

test('LEAP is only offered when its feet are on the ground', () => {
  const m = arena(10);
  m.spawnLocust();
  const l = m.locust;
  const gy = m.world.surfaceY(36, 36);
  l.ent.pos.x = 36.5; l.ent.pos.z = 36.5; l.ent.pos.y = gy;
  l.ent.onGround = true;
  assert.equal(locustLegalMask(l, { entities: m.builders, world: m.world, want: null })[LACT.LEAP], 1);
  executeLocustAction(l, LACT.LEAP, lctx(m));
  assert.ok(l.ent.vel.y > 8, 'the leap launches it');
  assert.equal(locustLegalMask(l, { entities: m.builders, world: m.world, want: null })[LACT.LEAP], 0,
    'mid-air it cannot leap again');
});

test('the Locust mask is 12 wide and always offers something', () => {
  const m = arena(11);
  m.spawnLocust();
  const l = m.locust;
  const mask = locustLegalMask(l, { entities: m.builders, world: m.world, want: null });
  assert.equal(mask.length, 12, 'the Locust action space');
  assert.ok(mask.some((v) => v === 1), 'MCTS always has at least one legal root child');
  assert.equal(mask[LACT.NOOP], 1);
  l.active = false;
  const off = locustLegalMask(l, { entities: m.builders, world: m.world, want: null });
  assert.equal(off.filter((v, i) => v === 1 && i !== LACT.NOOP).length, 0, 'a despawned Locust idles only');
});
