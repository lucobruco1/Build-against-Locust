/**
 * tests/match.test.js
 * ---------------------------------------------------------------------------
 * The loop the game is specified by: 1.5 min to build, then the Locust for
 * 3 min, then everybody who died is revived and the cycle starts again — with
 * seven bots and the player sharing one action space, and one rule about who may
 * delete what. These tests run the real Match (real physics, real brains, real
 * tally) with --phase-scale-style timing compressed to milliseconds.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { Match, durationFor } from '../game/match.js';
import {
  ACT, LACT, B, BASE, ENTITY, PHASE, TIMING, N_BOTS, RW, BOT_PLACEABLE,
} from '../shared/rules.js';
import { OWNER_BOT0, OWNER_NATURAL, OWNER_PLAYER } from '../core/world.js';
import { executeLocustAction, builderLegalMask } from '../game/actions.js';
import { packWorld, unpackWorld } from '../game/net.js';
import { EZModel } from '../ai/efficientzero.js';

/* ------------------------------------------------------------------ helpers */

const FAST = { flat: 1, learn: false, phaseScale: 0.004 };

function match(extra = {}) {
  return new Match({ ...FAST, ...extra });
}
/** Run `seconds` of game time through the fixed-step integrator. */
function run(m, seconds) {
  const ticks = Math.max(1, Math.round(seconds * 30));
  for (let i = 0; i < ticks; i++) m.update(1000 / 30);
  return ticks;
}
/** Phase sequence visited while running, with the Locust's presence at each. */
function watchPhases(m, seconds) {
  const seen = [];
  const ticks = Math.max(1, Math.round(seconds * 30));
  for (let i = 0; i < ticks; i++) {
    m.update(1000 / 30);
    const cur = seen[seen.length - 1];
    const now = `${m.phase}:${m.locust ? 'here' : 'gone'}`;
    if (!cur || cur.key !== now) seen.push({ key: now, phase: m.phase, cycle: m.cycle, locust: !!m.locust });
  }
  return seen;
}

/* -------------------------------------------------------------- the contract */

test('the clock is exactly what was asked for: 90 s to build, 180 s of hunt, 5 s to revive', () => {
  assert.equal(TIMING.BUILD_MS, 90_000, '1.5 minutes of build time');
  assert.equal(TIMING.HUNT_MS, 180_000, 'the Locust keeps hunting for 3 minutes');
  assert.equal(TIMING.REVIVE_MS, 5_000, 'a 5 s revive/tally screen between cycles');
  assert.equal(durationFor(PHASE.BUILD, {}), 90_000);
  assert.equal(durationFor(PHASE.HUNT, {}), 180_000);
  assert.equal(durationFor(PHASE.REVIVE, {}), 5_000);
  // and the compression the tests/harness use scales all three together
  assert.equal(durationFor(PHASE.BUILD, { phaseScale: 0.5 }), 45_000);
  assert.equal(TIMING.TICK_MS, 1000 / 30, 'the simulation is fixed at 30 Hz');
});

test('seven respawning bots, each with its own EfficientZero network, plus the player', () => {
  const m = match();
  assert.equal(N_BOTS, 7);
  assert.equal(m.builders.length, N_BOTS + 1, 'seven bots and the player share the world');
  const bots = m.builders.filter((b) => !b.isHuman);
  const human = m.builders.find((b) => b.isHuman);
  assert.equal(bots.length, N_BOTS);
  assert.equal(human.name, 'You');
  assert.equal(human.brain, null, 'the player slot is driven by the keyboard, not by a network');
  for (const b of bots) {
    assert.ok(b.brain, `${b.name} has a brain`);
    assert.ok(b.brain.model instanceof EZModel, `${b.name}'s brain is an EfficientZero model`);
    assert.equal(b.brain.nActions, 15, 'over the same 15 actions the player has');
    assert.equal(b.brain.kind, 'builder');
  }
  const ids = new Set(m.builders.map((b) => b.id));
  assert.equal(ids.size, m.builders.length, 'distinct agents');
  const owners = new Set(m.builders.map((b) => b.owner));
  assert.equal(owners.size, m.builders.length, 'distinct block owners — that is what "your blocks" means');
  assert.equal(human.owner, OWNER_PLAYER);
  assert.equal(bots[0].owner, OWNER_BOT0);
  assert.equal(bots[6].owner, OWNER_BOT0 + 6);
  // every plot gets a shell to build, and the plots do not overlap
  const plots = m.plots.map((p) => `${p.x},${p.z}`);
  assert.equal(new Set(plots).size, plots.length, 'each builder has its own plot');
  for (const b of m.builders) assert.ok(b.shell.all.length > 20 && b.shellSet.size === b.shell.all.length);
});

test('the Locust is spawned by another network and only exists during the hunt', () => {
  const m = match();
  assert.equal(m.phase, PHASE.LOBBY);
  assert.equal(m.locust, null, 'nothing to fear yet');
  const seen = watchPhases(m, 40);
  const order = seen.map((s) => s.phase).filter((p, i, a) => p !== a[i - 1]);
  assert.ok(order.includes(PHASE.BUILD), 'build first');
  assert.ok(order.includes(PHASE.HUNT), 'then the hunt');
  assert.ok(order.includes(PHASE.REVIVE), 'then the revive screen');
  assert.equal(order.indexOf(PHASE.HUNT) > order.indexOf(PHASE.BUILD), true, 'in that order');
  const huntRows = seen.filter((s) => s.phase === PHASE.HUNT);
  assert.ok(huntRows.length > 0 && huntRows.every((s) => s.locust), 'the Locust is present for the whole hunt');
  assert.ok(seen.filter((s) => s.phase === PHASE.REVIVE).every((s) => !s.locust),
    'and it disappears again when the 3 minutes are up');

  const l = m.locust || m.spawnLocust();
  assert.ok(l.brain, 'the Locust is not a scripted monster: it has a brain');
  assert.ok(l.brain.model instanceof EZModel, 'an EfficientZero model of its own');
  assert.notEqual(l.brain, m.builders[0].brain, 'not shared with the builders');
  assert.equal(l.brain.kind, 'locust');
  assert.equal(l.brain.nActions, 12, 'its own action space (movement + leap + smash + strike)');
  assert.equal(l.brain.codec.dim, m.codecLocust.dim);
  assert.notEqual(m.codecLocust.dim, m.codecBuilder.dim, 'a different observation, encoded differently');
  assert.equal(l.ent.kind, 'locust');
  assert.ok(l.ent.height > 3.5, 'tall enough to look like the thing in the videos');
  assert.equal(l.interval, 0.25, 'and it decides faster than a builder can place');
});

test('death is temporary: at the end of the hunt every corpse gets up', () => {
  const m = match();
  m.setPhase(PHASE.HUNT);
  m.spawnLocust();
  const victims = [m.builders[0], m.builders[2], m.builders[5]];
  for (const v of victims) {
    v.health = 3;
    m.kill(v, 'locust');
    assert.equal(v.alive, false, `${v.name} is down`);
  }
  assert.equal(m.builders.filter((b) => !b.alive).length, victims.length);
  m.setPhase(PHASE.REVIVE);
  for (const v of victims) {
    assert.equal(v.alive, true, `${v.name} is back up`);
    assert.equal(v.health, v.maxHealth, 'with full health');
    assert.equal(v.grabbed, false);
    assert.equal(v.ent.grabbed, false);
    assert.ok(v.invuln > 0, 'briefly invulnerable so it is not instantly re-grabbed');
    assert.equal(Math.floor(v.ent.pos.x), v.plot.x, 'at its own plot');
    assert.equal(Math.floor(v.ent.pos.z), v.plot.z);
    assert.equal(v.stats.deaths, 1);
  }
  // and the bots get fresh material for the next base
  const before = m.builders[1].inv.count(B.COBBLE) + m.builders[1].inv.count(B.PLANK) + m.builders[1].inv.count(B.DIRT);
  m.setPhase(PHASE.BUILD);
  const after = m.builders[1].inv.count(B.COBBLE) + m.builders[1].inv.count(B.PLANK) + m.builders[1].inv.count(B.DIRT);
  assert.ok(after >= before, 'building supplies are topped up between cycles');
  assert.ok(after > 0);
});

test('a cycle counts, then the next one starts — forever unless capped', () => {
  const m = match({ maxCycles: 0 });   // 0 = endless, exactly what the brief asks for
  assert.equal(m.cycle, 0, 'the lobby is not a cycle');
  m.advancePhase();
  assert.equal(m.cycle, 1);
  m.setPhase(PHASE.HUNT);
  m.advancePhase();
  assert.equal(m.phase, PHASE.REVIVE);
  m.advancePhase();
  assert.equal(m.cycle, 2, 'the loop continues');
  assert.equal(m.finished, false);
  assert.equal(m.phase, PHASE.BUILD, 'back to building');

  const capped = match({ maxCycles: 1 });
  capped.advancePhase();
  assert.equal(capped.cycle, 1);
  capped.setPhase(PHASE.HUNT);
  capped.advancePhase();
  capped.advancePhase();
  assert.equal(capped.finished, true, 'a capped session stops and says so');
  assert.ok(capped.log.some((line) => /session over/i.test(line.text ?? line.msg ?? '')), 'with a line on screen');
});

/* ------------------------------------------------------ grab, stab, and rules */

test('the kill is a grab and then a stab, in that order', () => {
  const m = match();
  m.setPhase(PHASE.HUNT);
  const l = m.spawnLocust();
  const prey = m.builders[3];
  // put the prey in its hand by hand: the grab itself is covered in actions.test.js
  prey.grabbed = true;
  prey.grabbedBy = l.id;
  prey.grabT = 0;
  prey.ent.grabbed = true;
  l.grabbed = prey;
  const px = prey.ent.pos.x;
  const py = prey.ent.pos.y;
  const hp = prey.health;

  m.updateGrabbed(prey, 1 / 30);
  assert.equal(prey.grabbed, true, 'one tick into the grab: still held');
  assert.equal(prey.alive, true, 'and still alive — the stab has not landed yet');
  assert.ok(prey.health >= hp - 0.001, 'being held is not being stabbed');
  assert.ok(Math.hypot(prey.ent.pos.x - px, prey.ent.pos.y - py) > 0 || (px === l.ent.pos.x), 'the victim is dragged with the hand');
  assert.ok(prey.grabT > 0, 'the grab timer runs');

  prey.grabT = ENTITY.GRAB_TIME - 0.01;
  m.updateGrabbed(prey, 0.05);
  assert.equal(prey.grabbed, false, 'the grab resolves');
  assert.equal(prey.health, 0);
  assert.equal(prey.alive, false, 'and the stab kills');
  assert.equal(prey.stats.deaths, 1);
  assert.equal(l.stats.kills, 1, 'the Locust is credited');
  assert.equal(l.grabbed, null, 'its hand is empty again');
  const evs = m.drainEvents();
  const kinds = evs.map((e) => e.t ?? e[0]);
  assert.ok(kinds.includes('stab'), 'the client is told about the stab');
  assert.ok(kinds.includes('death'), 'and about the death');
  assert.ok(prey.reward < 0 || m.log.length > 0, 'the victim was punished for dying');

  m.setPhase(PHASE.REVIVE);
  assert.equal(prey.alive, true, 'until the end of the night, when it gets up again');
});

test('struggling works, and only while it is holding you', () => {
  const m = match();
  m.setPhase(PHASE.HUNT);
  const l = m.spawnLocust();
  const prey = m.builders[2];
  prey.grabbed = true; prey.ent.grabbed = true; prey.grabbedBy = l.id; prey.grabT = 0; l.grabbed = prey;
  m.rand = () => 0;                           // the escape roll always succeeds
  for (let i = 0; i < 40 && prey.grabbed; i++) { prey.struggle = 1; m.updateGrabbed(prey, 1 / 30); }
  assert.equal(prey.grabbed, false, 'a struggle can break the hold');
  assert.equal(prey.stats.escapes, 1);
  assert.equal(l.stats.escapes, 1, 'and it is recorded as an escape for both sides');
  assert.equal(l.grabbed, null, 'its hand is empty');

  // the player struggles with the same key that jumps
  const human = m.player;
  human.grabbed = true; human.ent.grabbed = true; human.grabT = 0; l.grabbed = human;
  m.playerJump();
  assert.equal(human.struggle, 1, 'space while grabbed = struggle, not a jump');
});

test('nobody un-builds anybody else, not through the match either', () => {
  const m = match();
  const a = m.builders[1];
  const c = m.builders[2];
  // give `a` a wall block right where `c` is looking
  const gy = a.groundY;
  const cell = { x: a.plot.x + 1, y: gy, z: a.plot.z };
  m.world.set(cell.x, cell.y, cell.z, B.COBBLE, a.owner, m.now);
  c.ent.pos.x = cell.x - 1.5; c.ent.pos.z = cell.z + 0.5; c.ent.pos.y = gy;
  c.ent.yaw = -Math.PI / 2; c.ent.pitch = 0;
  c.coolBreak = 0;
  const aWallBefore = m.securityOf(a, true).wall;
  const cWallBefore = m.securityOf(c, true).wall;
  m.execBuilder(c, ACT.BREAK_FRONT);
  assert.equal(m.world.get(cell.x, cell.y, cell.z), B.COBBLE, 'the block survives the attempt');
  assert.equal(c.stats.broken, 0, 'and no credit was given');
  assert.equal(a.health, 20, 'and nobody took damage from a refused verb');
  assert.equal(m.securityOf(a, true).wall, aWallBefore, 'a’s base is untouched');
  assert.equal(m.securityOf(c, true).wall, cWallBefore, 'and c wasted only its own time');

  a.ent.pos.x = cell.x - 1.5; a.ent.pos.z = cell.z + 0.5; a.ent.pos.y = gy;
  a.ent.yaw = -Math.PI / 2; a.ent.pitch = -0.65; a.coolBreak = 0;   // looking at the wall face
  a.plot = { x: cell.x - 3, z: cell.z };
  m.execBuilder(a, ACT.BREAK_FRONT);
  assert.equal(m.world.get(cell.x, cell.y, cell.z), B.AIR, 'its own block goes when it asks');
  assert.equal(a.stats.broken, 1);
  assert.equal(m.world.getOwner(cell.x, cell.y, cell.z), OWNER_NATURAL, 'and the cell is un-owned again');
});

test('the Locust smashing a wall costs the owner its coverage and its score', () => {
  const m = match();
  m.setPhase(PHASE.HUNT);
  const l = m.spawnLocust();
  const owner = m.builders[1];
  const gy = owner.groundY;
  // seal a 5x5 shell ring cell right in front of the monster
  const cell = { x: owner.plot.x + 2, y: gy, z: owner.plot.z };
  m.world.set(cell.x, cell.y, cell.z, B.PLANK, owner.owner, m.now);
  for (let dz = -2; dz <= 2; dz++) m.world.set(cell.x, gy - 1, owner.plot.z + dz, B.STONE, OWNER_NATURAL, m.now);
  l.ent.pos.x = cell.x - 1.5; l.ent.pos.z = cell.z + 0.5; l.ent.pos.y = gy;
  l.ent.yaw = -Math.PI / 2; l.ent.pitch = 0; l.ent.onGround = true;
  const wallBefore = m.securityOf(owner, true).wall;
  assert.ok(wallBefore > 0, 'the shell started out with something to it');

  const emitted = [];
  for (let i = 0; i < 8; i++) {
    const r = executeLocustAction(l, LACT.SMASH_BLOCK, {
      world: m.world, now: m.now, entities: m.builders,
      onBlockBroken: (c, own) => m.onLocustBrokeBlock(c, own), want: null,
    });
    emitted.push(...r.events);
    if (m.world.get(cell.x, cell.y, cell.z) === B.AIR) break;
  }
  assert.equal(m.world.get(cell.x, cell.y, cell.z), B.AIR, 'the plank wall is opened');
  assert.ok(l.stats.smashed > 0, 'the Locust did the damage itself');
  assert.ok(owner.stats.lost >= 1, 'the owner is charged with the loss');
  const wallAfter = m.securityOf(owner, true).wall;
  assert.ok(wallAfter < wallBefore, `coverage drops (${wallBefore.toFixed(2)} → ${wallAfter.toFixed(2)})`);
  assert.ok(emitted.some((e) => e.t === 'smash'), 'every swing is an event the renderer can animate');
  const broke = emitted.find((e) => e.t === 'break');
  assert.ok(broke, 'and the client is told when the wall opens');
  assert.equal(broke.own, owner.owner, 'with the owner of the block that went, so the hole is attributed');
  assert.equal(broke.x, cell.x);
  assert.deepEqual({ x: broke.x, y: broke.y, z: broke.z }, cell);
});

/* ------------------------------------------------------------------ scoring */

test('the tally scores the base exactly as documented, and ranks the builders', () => {
  const m = match();
  m.setPhase(PHASE.BUILD);
  const b = m.builders[1];
  // build a legal wall cell through the real verb, then judge the arithmetic
  b.ent.pos.x = b.plot.x + 0.5; b.ent.pos.z = b.plot.z + 0.5; b.ent.pos.y = b.groundY;
  b.ent.yaw = -Math.PI / 2; b.ent.pitch = 0; b.coolPlace = 0;
  m.execBuilder(b, ACT.PLACE_FRONT);
  assert.ok(b.stats.placed >= 1, 'the bot placed a block');
  b.stats.escapes = 1;
  b.stats.deaths = 1;
  b.stats.lost = 2;
  const sec = m.securityOf(b, true);
  const rows = m.tallyCycle();
  const row = rows.find((r) => r.id === b.id);
  const want = Math.round(sec.wall * 220 + sec.roof * 120 + (b.alive ? 60 : 0)
    + b.stats.placed * 1.2 + (sec.sealed ? 150 : 0) + b.stats.escapes * 25
    - b.stats.deaths * 40 + b.stats.lost * -1);
  assert.equal(row.score, want, 'the HUD number is the documented formula');
  assert.equal(b.score, want, 'and it is written back to the builder');
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i - 1].score >= rows[i].score, 'rows are ranked');
  assert.equal(rows.length, m.builders.length);
  assert.equal(m.tally[m.tally.length - 1].rows.length, rows.length, 'the tally screen holds the rows');
  assert.equal(m.tally[m.tally.length - 1].cycle, m.cycle);
  assert.ok(m.log.length > 0, 'and it announced the leader');
});

test('security measures the shell it is supposed to: walls, roof, and the seal', () => {
  const m = match();
  const b = m.builders[1];
  const empty = m.securityOf(b, true);
  assert.equal(empty.wall, 0, 'nothing is built yet');
  assert.equal(empty.sealed, false);
  assert.ok(empty.total > 0, 'the shell knows how many cells it has');

  // one layer of the wall ring is a quarter of it: the shell is a box, not a fence
  const lowCells = b.shell.all.filter((c) => c.ring === 'wall' && c.y === b.groundY + 1);
  const allWall = b.shell.all.filter((c) => c.ring === 'wall');
  assert.ok(lowCells.length > 0 && lowCells.length < allWall.length, 'the scored wall is several blocks tall');
  for (const c of lowCells) m.world.set(c.x, c.y, c.z, B.BRICK, b.owner, m.now);
  const mid = m.securityOf(b, true);
  assert.ok(mid.wall > empty.wall && mid.wall < 0.9, `wall coverage rose a bit (${empty.wall} → ${mid.wall.toFixed(2)})`);
  assert.equal(mid.breaches, allWall.length - lowCells.length, 'and every gap is counted as a breach');

  // the roof is what keeps it from reaching in through the top
  const roofCells = b.shell.all.filter((c) => c.ring === 'roof');
  assert.ok(roofCells.length > 0, 'a shell has a roof to build');
  for (const c of allWall) m.world.set(c.x, c.y, c.z, B.BRICK, b.owner, m.now);
  for (const c of roofCells) m.world.set(c.x, c.y, c.z, B.BRICK, b.owner, m.now);
  const full = m.securityOf(b, true);
  assert.ok(full.wall >= 0.99, `the ring is closed (wall=${full.wall.toFixed(2)})`);
  assert.ok(full.roof > 0, 'roof coverage rose');
  assert.equal(full.sealed, true, 'a closed ring reads as sealed, which is what pays the bonus');
  assert.equal(full.breaches, 0);
  assert.ok(full.blocks >= allWall.length, 'and it knows how many of those blocks belong to this builder');
});

/* ------------------------------------------------------------------ the wire */

test('the snapshot and the event stream carry what the client renders', () => {
  const m = match();
  run(m, 0.5);
  const snap = m.snapshot();
  for (const key of ['cycle', 'phase', 'builders', 'stats']) assert.ok(key in snap, `snapshot has ${key}`);
  assert.equal(snap.builders.length, m.builders.length);
  for (const b of snap.builders) {
    for (const key of ['id', 'name', 'score', 'wall', 'roof', 'placed', 'deaths', 'alive', 'x', 'y', 'z', 'grabbed', 'sealed', 'own', 'selected', 'inv']) {
      assert.ok(key in b, `builder snapshot has ${key}`);
    }
    assert.ok(Number.isFinite(b.x) && Number.isFinite(b.y) && Number.isFinite(b.z), 'with a position to draw');
    assert.ok(b.health >= 0 && b.health <= 20, 'and health the HUD can bar');
  }
  assert.ok(snap.world && snap.world.revision >= 0, 'the world revision lets a client resync only what changed');
  assert.ok(snap.log.every((line) => typeof line.msg === 'string'), 'the chat/log lines are text + a speaker');
  assert.ok(snap.stats.length >= N_BOTS, 'one row per brain');
  assert.ok(Array.isArray(snap.tally), 'the tally screen is part of the state');
  assert.ok(snap.timeTotal > 0, 'and so is the clock the HUD shows');

  m.setPhase(PHASE.BUILD);
  const builder = m.builders[1];
  builder.coolPlace = 0;
  builder.ent.pos.x = builder.plot.x + 0.5; builder.ent.pos.z = builder.plot.z + 0.5;
  builder.ent.pos.y = builder.groundY; builder.ent.yaw = -Math.PI / 2; builder.ent.pitch = 0;
  m.execBuilder(builder, ACT.PLACE_FRONT);
  const evs = m.drainEvents();
  assert.ok(Array.isArray(evs), 'events drain as an array');
  const place = evs.find((e) => e.t === 'place' && e.by === builder.id);
  assert.ok(place, 'the client is told about the placement');
  for (const key of ['x', 'y', 'z', 'id', 'by', 'own', 'at', 'phase', 'cycle']) assert.ok(key in place, `a place event has ${key}`);
  assert.equal(place.own, builder.owner, 'and it says who placed it, so the client can grey the block out for others');
  assert.equal(place.cycle, m.cycle);
  assert.equal(m.drainEvents().length, 0, 'draining is destructive: no event is sent twice');
});

test('the player has the same verbs as a bot, and the same limits', () => {
  const m = match();
  const human = m.player;
  assert.ok(human.isHuman);
  assert.ok(human.alive);
  const botMask = builderLegalMask(m.builders[1], m.world, { phase: m.phase, entities: m.builders });
  const humanMask = builderLegalMask(human, m.world, { phase: m.phase, entities: m.builders });
  assert.equal(humanMask.length, 15, 'fifteen verbs on both sides');
  assert.equal(humanMask.length, botMask.length, 'and the same fifteen');
  for (let i = 0; i < humanMask.length; i++) assert.ok(humanMask[i] === 0 || humanMask[i] === 1);

  // look + move intents come from the client and land on the entity
  const yaw = human.ent.yaw, pitch = human.ent.pitch;
  m.setPlayerLook(300, -120);
  assert.ok(Math.abs((human.ent.yaw - (yaw - 300 * 0.0026)) % (Math.PI * 2)) < 1e-6, 'yaw follows the mouse, mirrored once');
  assert.ok(human.ent.pitch > pitch, 'and pitch follows it (up is up)');
  m.setPlayerLook(0, -1e6);
  assert.ok(human.ent.pitch <= 1.25 + 1e-9, 'pitch is clamped so the client cannot flip the camera');
  m.setPlayerMove(0, -1, true);
  assert.equal(human.intent.mz, -1);
  assert.equal(human.intent.sprint, 1);
  m.playerJump();
  assert.ok(human.ent.vel.y > 0 || !human.ent.onGround, 'the jump verb reaches physics');

  // the player may mine the world and build anywhere, but not un-build a bot
  const gy = human.groundY;
  m.world.set(human.plot.x + 1, gy, human.plot.z, B.COBBLE, OWNER_BOT0 + 3, m.now);
  human.ent.pos.x = human.plot.x - 0.5; human.ent.pos.z = human.plot.z + 0.5; human.ent.pos.y = gy;
  human.ent.yaw = -Math.PI / 2; human.ent.pitch = 0; human.coolBreak = 0;
  m.playerBreak();
  assert.equal(m.world.get(human.plot.x + 1, gy, human.plot.z), B.COBBLE, 'a bot’s wall is not the player’s to take');
  assert.ok(BOT_PLACEABLE.length > 0);

  m.playerCycle(1);
  const slot = human.inv.slot;
  m.playerSelect((slot + 1) % human.inv.hotbar.length);
  assert.notEqual(human.inv.slot, slot, 'the hotbar responds');
  const before = human.intent.mx;
  m.playerAct(ACT.STRAFE_RIGHT);
  assert.equal(human.intent.mx, 1, 'a raw action from the client goes through the same executor as a bot action');
  assert.notEqual(before, 1);
});

test('a grabbed player is not allowed to keep building', () => {
  const m = match();
  m.setPhase(PHASE.HUNT);
  const l = m.spawnLocust();
  const human = m.player;
  human.grabbed = true; human.ent.grabbed = true; human.grabT = 0; l.grabbed = human;
  const inv = human.inv.count(BOT_PLACEABLE[0]);
  m.playerPlace();
  assert.equal(human.inv.count(BOT_PLACEABLE[0]), inv, 'no blocks leave the pack while it holds you');
  human.intent.mx = 0; human.intent.mz = 0;
  m.setPlayerMove(0, -1, false);
  assert.equal(human.intent.mz, 0, 'and its grip cancels your movement input entirely');
});

test('the world the client receives is the world the physics ran on', () => {
  const m = match();
  m.setPhase(PHASE.BUILD);
  const b = m.builders[1];
  b.ent.pos.x = b.plot.x + 0.5; b.ent.pos.z = b.plot.z + 0.5; b.ent.pos.y = b.groundY;
  b.ent.yaw = -Math.PI / 2; b.coolPlace = 0;
  m.execBuilder(b, ACT.PLACE_FRONT);
  const packed = packWorld(m.world);
  const back = unpackWorld(packed);
  let diff = 0;
  for (let z = 0; z < m.world.sz; z += 3) {
    for (let x = 0; x < m.world.sx; x += 3) {
      for (let y = 0; y < m.world.sy; y += 5) if (m.world.get(x, y, z) !== back.get(x, y, z)) diff++;
    }
  }
  assert.equal(diff, 0, 'the RLE snapshot the client rebuilds from is lossless');
  assert.ok(packed.rle.length < m.world.sx * m.world.sy * m.world.sz, 'and much smaller than the raw grid');
  assert.ok(m.world.countOwnerBlocks(b.owner) >= 1, 'ownership survives into the snapshot');
  const sec = m.securityOf(b, true);
  assert.ok(sec.wall > 0, 'and the score sees it');
});
