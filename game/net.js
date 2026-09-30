/**
 * Netcode helpers shared by the server (encode) and the browser (decode).
 *
 * The world is small (72×40×72) but mostly empty, so a run-length encoding of
 * the block array is both tiny and trivial to rebuild. After the initial join
 * only *deltas* travel: the `place` / `break` events the match already emits.
 *
 * Entity state is sent as a compact snapshot at 20 Hz; the client smooths it
 * with interpolation because the sim runs at 30 Hz.
 */

import { VoxelWorld } from '../core/world.js';
import { B, BLOCK_DEFS } from '../shared/rules.js';

/* ------------------------------------------------------------ world payload */

/** Run-length encode the whole block array: [id, count, id, count, ...] */
export function packWorld(world) {
  const rle = [];
  const blocks = world.blocks;
  let run = blocks[0] ?? 0;
  let n = 0;
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    if (b === run) { n++; continue; }
    rle.push(run, n);
    run = b; n = 1;
  }
  rle.push(run, n);
  return { sx: world.sx, sy: world.sy, sz: world.sz, revision: world.revision, rle };
}

/** Rebuild a client-side world from `packWorld` output. */
export function unpackWorld(payload) {
  const { sx, sy, sz, rle } = payload;
  const world = new VoxelWorld(0, null);
  const blocks = world.blocks;
  let i = 0;
  for (let k = 0; k < rle.length; k += 2) {
    const id = rle[k], n = rle[k + 1];
    blocks.fill(id, i, i + n);
    i += n;
  }
  world.rebuildCaches();      // height map, column solidity, baked light, terrain
  world.revision = payload.revision ?? 0;
  return world;
}

/* ---------------------------------------------------------- state encoder */

const Q = (v, q = 1000) => Math.round((v || 0) * q) / q;

/**
 * Trim the match snapshot down to what a renderer needs. Kept a level flatter
 * than `match.snapshot()` so the client never has to guess field names.
 */
export function encodeState(match) {
  const snap = match.snapshot();
  const actors = snap.builders.map((b) => ({
    id: b.id, name: b.name, alive: b.alive, human: !!b.isHuman,
    x: Q(b.x), y: Q(b.y), z: Q(b.z), yaw: Q(b.yaw, 1000), pitch: Q(b.pitch, 1000),
    hp: Q(b.health, 10), wall: Q(b.wall, 100), roof: Q(b.roof, 100), sealed: !!b.sealed,
    breaches: b.breaches, grabbed: !!b.grabbed, seen: !!b.seen, score: Math.round(b.score),
    action: b.action, invuln: Q(b.invuln, 10), stuck: Q(b.stuck, 10),
    inv: b.inv, sel: b.selected, placed: b.placed, broken: b.broken, lost: b.lost,
    deaths: b.deaths, escaped: b.escaped,
  }));
  const brains = (snap.stats || []).map((s) => ({
    id: s.id, name: s.name, kind: s.kind, params: s.params,
    updates: s.trainSteps, decisions: s.decisions, priorShare: s.priorShare,
    bufferGames: s.bufferGames, bufferSteps: s.bufferSteps,
    loss: s.loss, policy: s.policy, value: s.valueLoss, prefix: s.prefixLoss,
    consist: s.consist, entropy: s.entropy, gradNorm: s.gradNorm,
    assist: s.assist, temp: s.temp, sims: s.sims, nodes: s.nodes,
    searchMs: s.searchMs, trainMs: s.trainMs, trajLen: s.trajLen,
    totalReward: s.totalReward, badWeights: s.badWeights,
  }));
  const l = snap.locust;
  return {
    phase: snap.phase, cycle: snap.cycle, tick: snap.tick,
    timeLeft: Q(snap.timeLeft, 10), timeTotal: Q(snap.timeTotal, 10),
    finished: !!snap.finished,
    actors,
    locust: l ? {
      x: Q(l.x), y: Q(l.y), z: Q(l.z), yaw: Q(l.yaw, 1000), pitch: Q(l.pitch, 1000),
      kills: l.kills, smashed: l.smashed, grabs: l.grabs, holding: l.holding,
      stuck: Q(l.stuck, 10), action: l.action, active: !!l.active,
    } : null,
    tally: snap.tally,
    log: snap.log,
    brains,
  };
}

/* ------------------------------------------------------------- delta feed */

/**
 * Turn a match event into the minimal patch a client needs. Unknown events are
 * dropped, so gameplay code can add events without touching netcode.
 */
export function packEvent(ev) {
  switch (ev.t) {
    case 'place': return { t: 'place', x: ev.x, y: ev.y, z: ev.z, id: ev.id, by: ev.by, own: ev.own ?? 0 };
    case 'break': return { t: 'break', x: ev.x, y: ev.y, z: ev.z, id: ev.id, by: ev.by };
    case 'smash': return { t: 'smash', x: ev.x, y: ev.y, z: ev.z, id: ev.id, by: ev.by };
    case 'grab': return { t: 'grab', by: ev.by, victim: ev.victim };
    case 'stab': return { t: 'stab', by: ev.by, victim: ev.victim };
    case 'death': return { t: 'death', by: ev.by, cause: ev.cause };
    case 'revive': return { t: 'revive', by: ev.by };
    case 'escape': return { t: 'escape', by: ev.by };
    case 'hurt': return { t: 'hurt', by: ev.by, amount: ev.amount, cause: ev.cause };
    case 'jump': case 'leap': case 'land': return { t: ev.t, x: Q(ev.x), y: Q(ev.y), z: Q(ev.z), by: ev.by };
    case 'phase': return { t: 'phase', phase: ev.phase, cycle: ev.cycle };
    case 'locustSpawn': return { t: 'locustSpawn', x: Q(ev.x), y: Q(ev.y), z: Q(ev.z) };
    case 'locustDespawn': return { t: 'locustDespawn', reason: ev.reason };
    case 'sealed': return { t: 'sealed', by: ev.by };
    case 'msg': return { t: 'msg', text: ev.text, who: ev.who };
    default: return null;
  }
}

export function colorOf(id) {
  const d = BLOCK_DEFS[id] || BLOCK_DEFS[B.AIR];
  return d.color ?? 0x888888;
}
