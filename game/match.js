/**
 * game/match.js
 * ---------------------------------------------------------------------------
 * The match itself, as a deterministic headless simulation: the browser
 * renderer is a pure observer of it, which is why the exact same game can be run
 * in Node for the tests and for `npm run sim`.
 *
 * Phase machine, straight from the brief:
 *   LOBBY ──▶ BUILD (90 s: you + 7 bots fortify) ──▶ HUNT (180 s: the Locust
 *   spawns, navigates the map, smashes blocks, grabs and stabs its prey) ──▶
 *   REVIVE (everyone who died comes back, trajectories are closed and the
 *   EfficientZero networks take their gradient steps) ──▶ BUILD …
 *
 * Every builder gets exactly the verbs the player has. Every reward written
 * here is what that agent's value head and value-prefix head are trained to
 * predict, so the game rules and the learning signal can never drift apart.
 * ---------------------------------------------------------------------------
 */

import {
  WORLD, B, BASE, ENTITY, N_BOTS, BOT_NAMES, PHASE, TIMING, RW, ACT, OBS,
  LOCUST_N_ACTIONS, BUILDER_N_ACTIONS, BOT_PLACEABLE, LOCUST_LINES, rng, clamp,
  plotCenters, isSolid, EZ,
} from '../shared/rules.js';
import { VoxelWorld, makeShell, OWNER_PLAYER, OWNER_BOT0, OWNER_NATURAL, OWNER_LOCUST } from '../core/world.js';
import { generateWorld, generateFlat } from '../core/worldgen.js';
import { makeEntity, stepEntity, groundSnap, distance3 } from '../core/physics.js';
import { hasLineOfSight, raycast } from '../core/raycast.js';
import { astar, pathDirection, weakestBreach } from '../core/nav.js';
import { playerInventory, botInventory } from '../core/inventory.js';
import {
  builderLegalMask, executeBuilderAction, executeLocustAction, findPrey,
  integrateIntent, locustLegalMask, smashTargetInfo, BUILD_INTERVAL, LOCUST_INTERVAL,
} from './actions.js';
import { BrainLeague } from '../ai/brain.js';
import { builderScalars, locustScalars, makeBuilderCodec, makeLocustCodec } from '../ai/obs.js';

export class Match {
  constructor(opts = {}) {
    this.opts = opts;
    this.seed = (opts.seed ?? 20260930) >>> 0;
    this.rand = opts.rand || rng(this.seed);
    this.timeScale = opts.timeScale ?? 1;
    this.learn = opts.learn !== false;
    this.assist = opts.assist ?? 0.9;
    this.playerAI = !!opts.playerAI;
    this.simScale = opts.simScale ?? 1;
    this.trainEvery = opts.trainEvery ?? 100;
    this.decisionJitter = opts.decisionJitter ?? 0.18;
    this.maxCycles = opts.maxCycles ?? Infinity;
    this.acc = 0;

    const gen = opts.flat ? generateFlat : (opts.rawWorld ? null : generateWorld);
    this.world = new VoxelWorld(this.seed, gen);
    this.plots = this.world.plots || plotCenters();

    this.builders = [];
    this.locust = null;
    this.phase = PHASE.LOBBY;
    this.phaseT = durationFor(PHASE.LOBBY, opts);
    this.cycle = 0;
    this.tickCount = 0;
    this.now = 0;
    this.events = [];
    this.log = [];
    this.tally = [];
    this.finished = false;
    this.lastTrain = null;
    this.trainTime = 0;

    this.codecBuilder = makeBuilderCodec();
    this.codecLocust = makeLocustCodec();

    this.league = opts.league || new BrainLeague({
      cfg: { ...EZ, SIMS: Math.max(4, Math.round(EZ.SIMS * this.simScale)) },
      assist: this.assist,
      assistFloor: opts.assistFloor ?? 0.35,
      learning: this.learn,
      poolOpts: { games: opts.poolGames ?? 90, steps: opts.poolSteps ?? 300 },
    });

    this.spawnAll();
    this.say(`${this.builders.length - 1} EfficientZero builders woke up in a Minecraft-ish world. 90 seconds to build.`, 'world');
  }

  /* ------------------------------------------------------------ setup */

  spawnAll() {
    this.builders = [];
    const defs = [{ human: true, name: 'You', owner: OWNER_PLAYER, idx: -1 }];
    for (let i = 0; i < N_BOTS; i++) defs.push({ human: false, name: BOT_NAMES[i], owner: OWNER_BOT0 + i, idx: i });

    for (let i = 0; i < defs.length; i++) {
      const d = defs[i];
      const plot = this.plots[d.human ? 0 : i] || { x: WORLD.SX >> 1, z: WORLD.SZ >> 1 };
      const gy = this.world.surfaceY(plot.x, plot.z);
      const ent = makeEntity('player', plot.x + 0.5, gy, plot.z + 0.5);
      ent.yaw = Math.atan2(-(WORLD.SX / 2 - ent.pos.x), -(WORLD.SZ / 2 - ent.pos.z));
      const groundY = this.world.surfaceY(plot.x, plot.z);
      const shell = makeShell({ x: plot.x, y: groundY, z: plot.z }, groundY);
      const shellSet = new Set();
      const shellIndex = new Map();
      for (const c of shell.all) {
        const i = this.world.index(c.x, c.y, c.z);
        shellSet.add(i);
        shellIndex.set(i, c.ring);
      }
      const a = {
        id: d.human ? 'player' : 'bot' + d.idx,
        kind: 'builder',
        index: i,
        name: d.name,
        isHuman: d.human && !this.playerAI,
        owner: d.owner,
        plot,
        groundY,
        shell,
        shellSet,
        shellIndex,
        targetCell: null,
        ent,
        alive: true,
        health: 20,
        maxHealth: 20,
        inv: d.human ? playerInventory(this.opts.playerStacks ?? 96) : botInventory(this.opts.botStacks ?? 12),
        intent: { mx: 0, mz: 0, turn: 0, pitch: 0, sprint: 0 },
        decisionT: this.rand() * BUILD_INTERVAL,
        interval: BUILD_INTERVAL,
        coolPlace: 0,
        coolBreak: 0,
        reward: 0,
        score: 0,
        cover: 0,
        struggle: 0,
        security: { wall: 0, roof: 0, breaches: 0, blocks: 0, sealed: false, total: 0 },
        securityT: this.rand() * 0.5,
        revAt: -1,
        stuck: 0,
        lastPos: { x: ent.pos.x, z: ent.pos.z },
        hurtRecent: 0,
        invuln: 2.5,
        grabbed: false,
        grabT: 0,
        seenByLocust: false,
        stats: { placed: 0, broken: 0, lost: 0, deaths: 0, survived: 0, cycles: 0, escapes: 0, mined: 0 },
        brain: null,
        pendingAction: ACT.NOOP,
        needRoof: false,
        lastObsCtx: null,
      };
      if (!a.isHuman) a.brain = this.league.makeBuilder(a.id, a.name, this.codecBuilder, BUILDER_N_ACTIONS);
      this.builders.push(a);
    }
  }

  /* --------------------------------------------------------- locust spawn */

  spawnLocust() {
    if (this.locust) return this.locust;
    const k = 1 + Math.floor(this.rand() * Math.max(1, this.plots.length - 1));
    const p = this.plots[k % this.plots.length];
    const ang = this.rand() * Math.PI * 2;
    const x = clamp(p.x + Math.cos(ang) * 15, 4, WORLD.SX - 5);
    const z = clamp(p.z + Math.sin(ang) * 15, 4, WORLD.SZ - 5);
    const y = this.world.surfaceY(Math.floor(x), Math.floor(z)) + 1;
    const ent = makeEntity('locust', x, y, z);
    ent.yaw = Math.atan2(-(p.x + 0.5 - x), -(p.z + 0.5 - z));
    const a = {
      id: 'locust',
      name: 'The Locust',
      kind: 'locust',
      active: true,
      ent,
      intent: { mx: 0, mz: 0, turn: 0, pitch: 0 },
      decisionT: 0,
      interval: LOCUST_INTERVAL,
      attackCd: 0,
      smashHits: new Map(),
      grabbed: null,
      stabT: 0,
      path: null,
      pathT: 0,
      lastGoal: null,
      pathDir: { x: 0, z: 0, dx: 0, dz: 0, dy: 0, yaw: null },
      stuck: 0,
      lastPos: { x, z },
      reward: 0,
      spawnT: 0,
      sayT: 2.2,
      pendingAction: 0,
      stats: { kills: 0, smashed: 0, grabs: 0, escapes: 0, playerKills: 0 },
      brain: this.league.ensureLocustBrain(this.codecLocust, LOCUST_N_ACTIONS),
      targetPlot: null,
    };
    this.locust = a;
    this.pushEvent({ t: 'locustSpawn', x, y, z });
    this.say('the screens go to static. something tall is in the world.', 'world');
    this.say(LOCUST_LINES[0], 'locust');
    return a;
  }

  despawnLocust(reason = 'dawn') {
    const a = this.locust;
    if (!a) return;
    for (const b of this.builders) {
      if (b.grabbed) { b.grabbed = false; b.grabT = 0; b.ent.grabbed = false; }
    }
    a.active = false;
    a.reward += RW.L_SURVIVE;
    a.brain?.remember(a.reward, PHASE.HUNT, true);
    a.brain?.endGame({ cycle: this.cycle, kills: a.stats.kills, smashed: a.stats.smashed, reason });
    this.pushEvent({ t: 'locustDespawn', reason });
    this.locust = null;
    this.say('the static thins out. it is gone… for now.', 'world');
  }

  /* ------------------------------------------------------------ phases */

  setPhase(phase) {
    this.phase = phase;
    this.phaseT = durationFor(phase, this.opts);
    this.pushEvent({ t: 'phase', phase, cycle: this.cycle });
    if (phase === PHASE.BUILD) {
      this.say(`cycle ${this.cycle}: build. ${Math.round(TIMING.BUILD_MS / 1000)}s.`, 'world');
      for (const b of this.builders) {
        b.alive = true;
        b.health = b.maxHealth;
        b.invuln = 3;
        b.reward = 0;
        b.grabbed = false;
        b.grabT = 0;
        b.ent.grabbed = false;
        b.stats.cycles++;
        b.needRoof = false;
        b.seenByLocust = false;
        if (!b.isHuman) for (const id of [B.PLANK, B.COBBLE, B.DIRT]) b.inv.add(id, this.opts.botStacks ?? 12);
        else for (const id of BOT_PLACEABLE.slice(0, 6)) b.inv.add(id, 32);
        groundSnap(this.world, b.ent);
        this.securityOf(b, true);
      }
    } else if (phase === PHASE.HUNT) {
      this.say('it is here. do not let it find you.', 'world');
      this.spawnLocust();
    } else if (phase === PHASE.REVIVE) {
      this.despawnLocust('the night ended');
      this.tallyCycle();
      this.reviveDead();
      this.closeTrajectories();
    }
  }

  advancePhase() {
    switch (this.phase) {
      case PHASE.LOBBY:
        this.cycle = 1;
        this.setPhase(PHASE.BUILD);
        break;
      case PHASE.BUILD:
        this.setPhase(PHASE.HUNT);
        break;
      case PHASE.HUNT:
        this.setPhase(PHASE.REVIVE);
        break;
      case PHASE.REVIVE:
        if (this.maxCycles > 0 && this.cycle >= this.maxCycles) {  // 0 / negative = endless
          this.finished = true;
          this.say('session over — the networks keep what they learned.', 'world');
          return;
        }
        this.cycle++;
        this.setPhase(PHASE.BUILD);
        break;
      default:
        break;
    }
  }

  tallyCycle() {
    const rows = [];
    for (const b of this.builders) {
      const sec = this.securityOf(b, true);
      const score = Math.round(sec.wall * 220 + sec.roof * 120 + (b.alive ? 60 : 0)
        + b.stats.placed * 1.2 + (sec.sealed ? 150 : 0) + b.stats.escapes * 25
        - b.stats.deaths * 40 + b.stats.lost * -1);
      b.score = score;
      rows.push({
        id: b.id, name: b.name, score, alive: b.alive, wall: +sec.wall.toFixed(3),
        roof: +sec.roof.toFixed(3), placed: b.stats.placed, deaths: b.stats.deaths,
      });
      b.reward += b.alive ? RW.SURVIVE_CYCLE : 0;
      if (sec.sealed && b.alive) b.reward += RW.SECURE_BONUS;
      b.stats.survived += b.alive ? 1 : 0;
    }
    rows.sort((x, y) => y.score - x.score);
    this.tally.push({ cycle: this.cycle, rows: rows.slice(), at: this.now });
    if (this.tally.length > 20) this.tally.shift();
    const aliveN = rows.filter((r) => r.alive).length;
    this.say(`cycle ${this.cycle}: ${rows[0].name} leads with ${rows[0].score} pts · ${aliveN}/${rows.length} still breathing`, 'world');
    return rows;
  }

  reviveDead() {
    for (const b of this.builders) {
      if (b.alive) continue;
      b.alive = true;
      b.health = b.maxHealth;
      b.invuln = 3.5;
      b.grabbed = false;
      b.ent.grabbed = false;
      b.ent.vel.x = b.ent.vel.y = b.ent.vel.z = 0;
      b.ent.pos.x = b.plot.x + 0.5;
      b.ent.pos.z = b.plot.z + 0.5;
      groundSnap(this.world, b.ent);
      this.pushEvent({ t: 'revive', by: b.id });
    }
  }

  closeTrajectories() {
    for (const b of this.builders) b.brain?.endGame({ cycle: this.cycle, placed: b.stats.placed, alive: b.alive });
    if (this.opts.onCycleEnd) this.opts.onCycleEnd(this);
  }

  /* ---------------------------------------------------------- main loop */

  /** Real-time entry point. Runs whole 30 Hz sim steps, honouring timeScale. */
  update(dtMsReal) {
    if (this.finished) return;
    this.acc += dtMsReal * this.timeScale;
    const step = TIMING.TICK_MS;
    const maxSteps = TIMING.MAX_CATCHUP_STEPS * Math.max(1, Math.ceil(this.timeScale));
    let n = 0;
    while (this.acc >= step && n < maxSteps) {
      this.acc -= step;
      n++;
      this.step(step / 1000);
      if (this.finished) break;
    }
    if (this.acc > step * 8) this.acc = 0;
  }

  step(dt) {
    this.tickCount++;
    this.now += dt * 1000;
    this.phaseT -= dt * 1000;
    if (this.phaseT <= 0) {
      this.advancePhase();
      if (this.finished) return;
    }

    for (const b of this.builders) {
      b.coolPlace = Math.max(0, b.coolPlace - dt);
      b.coolBreak = Math.max(0, b.coolBreak - dt);
      b.hurtRecent = Math.max(0, b.hurtRecent - dt);
      b.invuln = Math.max(0, b.invuln - dt);
      if (!b.alive) { b.intent.mx = b.intent.mz = b.intent.turn = b.intent.pitch = 0; continue; }
      if (b.grabbed) this.updateGrabbed(b, dt);
      if (!b.isHuman) {
        b.decisionT -= dt;
        if (b.decisionT <= 0) {
          b.decisionT = b.interval * (1 + (this.rand() - 0.5) * this.decisionJitter * 2);
          this.decideBuilder(b);
        }
      }
      this.tickBuilderPhysics(b, dt);
    }

    if (this.locust && this.locust.active) this.tickLocust(dt);

    for (const b of this.builders) {
      b.securityT -= dt;
      if (b.securityT <= 0) {
        b.securityT = 0.5;
        this.securityOf(b, true);
        this.pickTargetCell(b);
      }
    }

    if (this.learn && this.tickCount % this.trainEvery === 0 && this.phase !== PHASE.LOBBY) this.trainStep();
    if (this.events.length > 400) this.events.splice(0, this.events.length - 400);
  }

  tickBuilderPhysics(b, dt) {
    const e = b.ent;
    if (!b.grabbed && !b.isHuman) integrateIntent(b, dt);
    const ev = stepEntity(this.world, e, dt);
    if (ev.fallDamage > 0) {
      this.damage(b, ev.fallDamage, 'fall');
      b.reward += RW.FALL_DAMAGE;
      this.pushEvent({ t: 'fall', by: b.id, dmg: ev.fallDamage });
    }
    const moved = Math.hypot(e.pos.x - b.lastPos.x, e.pos.z - b.lastPos.z);
    const wantMove = Math.abs(b.intent.mx) + Math.abs(b.intent.mz);
    if (wantMove > 0 && moved < 0.02) b.stuck = Math.min(12, b.stuck + dt * 6);
    else b.stuck = Math.max(0, b.stuck - dt * 3);
    b.lastPos.x = e.pos.x;
    b.lastPos.z = e.pos.z;
    b.cover = this.world.coverAt(Math.floor(e.pos.x), Math.floor(e.pos.y), Math.floor(e.pos.z), 2);

    let r = RW.STEP_COST;
    if (this.phase === PHASE.HUNT && this.locust) {
      const d = distance3(e.pos, this.locust.ent.pos);
      if (!b.seenByLocust && b.cover >= 2) r += RW.SHELTER_COVER;
      if (b.seenByLocust) r += RW.LOCUST_VISIBLE;
      if (d < 6) r += RW.NEAR_PREY_PENALTY;
    }
    b.reward += r;
  }

  damage(b, amount, cause) {
    if (b.invuln > 0 && cause !== 'fall') return false;
    b.health -= amount;
    b.hurtRecent = 1.2;
    b.reward += RW.HURT * Math.min(2, amount / 6);
    this.pushEvent({ t: 'hurt', by: b.id, amount, cause });
    if (b.health <= 0) {
      b.health = 0;
      this.kill(b, cause);
      return true;
    }
    return true;
  }

  kill(b, cause) {
    if (!b.alive) return;
    b.alive = false;
    b.stats.deaths++;
    b.grabbed = false;
    b.ent.grabbed = false;
    b.reward += RW.DEATH;
    b.intent.mx = b.intent.mz = 0;
    if (this.locust && cause === 'locust') {
      this.locust.stats.kills++;
      this.locust.reward += b.isHuman ? RW.L_KILL + RW.L_PLAYER_KILL : RW.L_KILL;
      this.say('the body appeared untouched.', 'locust');
    }
    this.pushEvent({ t: 'death', by: b.id, cause });
  }

  updateGrabbed(b, dt) {
    b.grabT += dt;
    const l = this.locust;
    if (!l) { b.grabbed = false; b.ent.grabbed = false; return; }
    const hx = l.ent.pos.x - Math.sin(l.ent.yaw) * 0.9;
    const hz = l.ent.pos.z - Math.cos(l.ent.yaw) * 0.9;
    b.ent.pos.x += (hx - b.ent.pos.x) * Math.min(1, dt * 9);
    b.ent.pos.z += (hz - b.ent.pos.z) * Math.min(1, dt * 9);
    b.ent.pos.y += (l.ent.pos.y + 1.9 - b.ent.pos.y) * Math.min(1, dt * 5);
    b.reward += -0.02;
    if (b.struggle > 0) {
      b.struggleTries = (b.struggleTries || 0) + 1;
      b.struggle = 0;
      if (this.rand() < ENTITY.STRUGGLE_ESCAPE) {
        b.grabbed = false;
        b.ent.grabbed = false;
        b.ent.vel.y = 5.5;
        l.grabbed = null;
        b.reward += RW.ESCAPE_GRAB;
        b.stats.escapes++;
        l.stats.escapes++;
        l.reward -= 0.4;
        this.pushEvent({ t: 'escape', by: b.id });
        this.say(`${b.name} wrenched free of its hand!`, 'world');
        return;
      }
    }
    if (b.grabT >= ENTITY.GRAB_TIME) {
      b.grabbed = false;
      b.ent.grabbed = false;
      l.grabbed = null;
      l.stats.grabs++;
      l.reward += 0.35;
      this.pushEvent({ t: 'stab', by: l.id, victim: b.id });
      b.health = 0;
      this.kill(b, 'locust');
    }
  }

  /* ---------------------------------------------------------- decisions */

  buildCtx(b) {
    const e = b.ent;
    const l = this.locust;
    const others = this.builders
      .filter((o) => o !== b && o.alive)
      .map((o) => ({ x: o.ent.pos.x, y: o.ent.pos.y, z: o.ent.pos.z, alive: 1 }))
      .sort((p, q) => Math.hypot(p.x - e.pos.x, p.z - e.pos.z) - Math.hypot(q.x - e.pos.x, q.z - e.pos.z))
      .slice(0, OBS.MAX_TARGETS);
    const ctx = {
      world: this.world,
      ent: e,
      owner: b.owner,
      phase: this.phase,
      timeLeft: clamp(this.phaseT / durationFor(this.phase, this.opts), 0, 1),
      health: b.health,
      hurtRecent: b.hurtRecent > 0,
      grabbed: b.grabbed,
      cycle: this.cycle,
      stuck: b.stuck,
      score: b.score + b.stats.placed,
      alliesAlive: this.builders.filter((o) => o.alive && o !== b).length,
      base: { x: b.plot.x, z: b.plot.z, groundY: b.groundY },
      security: { ...b.security },
      inventory: { total: b.inv.total, counts: countMap(b.inv) },
      hotbar: b.inv.hotbar,
      targets: others,
      front: frontInfo(this.world, e, b),
      blockLoss: b.stats.lost,
      kills: 0,
      locust: l ? {
        active: l.active,
        pos: l.ent.pos,
        visible: b.seenByLocust,
        huntingMe: l.grabbed === b,
      } : null,
    };
    ctx.scalars = builderScalars(ctx);
    return ctx;
  }

  decideBuilder(b) {
    if (!b.alive) return;
    const ctx = this.buildCtx(b);
    const legal = builderLegalMask(b, this.world, { phase: this.phase, entities: this.builders });
    b.lastObsCtx = ctx;
    let action = ACT.NOOP;
    if (b.brain) {
      const obs = b.brain.encode(ctx);
      const r = b.brain.decide(obs, legal, ctx, this.phase);
      action = r.action;
    }
    this.execBuilder(b, action);
    if (b.brain) {
      b.brain.remember(b.reward, this.phase, false);
      b.reward = 0;
    }
  }

  execBuilder(b, action, opts = {}) {
    const before = b.security;
    const sealedBefore = !!before.sealed;
    const r = executeBuilderAction(b, action, {
      world: this.world,
      phase: this.phase,
      now: this.now,
      entities: this.builders,
      preserveIntent: !!opts.preserveIntent,
      log: (m) => this.say(m, 'world'),
    });
    if (r.events.length) this.pushEvent(...r.events);
    if (b.brain) b.reward += r.reward;
    if (action === ACT.PLACE_FRONT || action === ACT.PLACE_DOWN) b.coolPlace = 0.14;
    if (action === ACT.BREAK_FRONT || action === ACT.BREAK_DOWN) b.coolBreak = 0.22;
    if (action === ACT.PLACE_FRONT || action === ACT.PLACE_DOWN) {
      const s = this.securityOf(b, true);
      if (s.sealed && !sealedBefore) {
        if (b.brain) b.reward += RW.SECURE_BONUS * 0.5;
        this.pushEvent({ t: 'sealed', by: b.id });
        this.say(`${b.name} sealed the base.`, 'world');
      }
    }
    b.pendingAction = action;
  }

  /* -------------------------------------------------------------- locust */

  tickLocust(dt) {
    const l = this.locust;
    const e = l.ent;
    l.spawnT += dt;
    l.attackCd = Math.max(0, l.attackCd - dt);
    l.sayT -= dt;
    if (l.sayT <= 0 && this.builders.some((b) => b.alive)) {
      l.sayT = 9 + this.rand() * 12;
      this.say(LOCUST_LINES[Math.floor(this.rand() * LOCUST_LINES.length)], 'locust');
    }
    if (l.grabbed) {
      l.intent.mx = l.intent.mz = l.intent.turn = 0;
      this.updateGrabbed(l.grabbed, dt);
    } else {
      l.decisionT -= dt;
      if (l.decisionT <= 0) {
        l.decisionT = l.interval * (1 + (this.rand() - 0.5) * 0.2);
        this.decideLocust();
      }
      integrateIntent(l, dt);
    }
    const before = { x: e.pos.x, z: e.pos.z };
    const ev = stepEntity(this.world, e, dt);
    if (ev.landed) this.pushEvent({ t: 'land', by: l.id });
    const moved = Math.hypot(e.pos.x - before.x, e.pos.z - before.z);
    if (moved < 0.02) l.stuck = Math.min(16, l.stuck + dt * 8);
    else l.stuck = Math.max(0, l.stuck - dt * 4);
    if (l.stuck > 9) l.reward += RW.L_STUCK;

    const eye = { x: e.pos.x, y: e.pos.y + e.eye, z: e.pos.z };
    for (const b of this.builders) {
      if (!b.alive) { b.seenByLocust = false; continue; }
      const tp = { x: b.ent.pos.x, y: b.ent.pos.y + 1.2, z: b.ent.pos.z };
      const d = distance3(eye, tp);
      b.seenByLocust = d < 34 && hasLineOfSight(this.world, eye, tp, 34);
    }
    l.reward += RW.L_STEP;
    if (l.grabbed) l.reward += 0.004; // pressure: holding prey is good for it
    // closing distance on the nearest prey is progress
    const near = this.builders.filter((b) => b.alive);
    if (near.length) {
      const d0 = l.lastNearestDist ?? Infinity;
      const d1 = Math.min(...near.map((b) => distance3(e.pos, b.ent.pos)));
      if (d1 < d0 - 0.02) l.reward += RW.L_PROGRESS * 30 * dt;
      l.lastNearestDist = d1;
    }
  }

  locustCtx(l) {
    const e = l.ent;
    const alive = this.builders.filter((b) => b.alive);
    const scored = alive
      .map((b) => ({ b, d: distance3(e.pos, b.ent.pos), vis: b.seenByLocust }))
      .sort((p, q) => (q.vis ? -12 : 0) + q.d - ((p.vis ? -12 : 0) + p.d));
    const targets = scored.slice(0, OBS.MAX_TARGETS).map((s) => ({
      x: s.b.ent.pos.x, y: s.b.ent.pos.y, z: s.b.ent.pos.z,
      visible: s.vis, grabbed: s.b.grabbed,
    }));

    let best = null, bestScore = -Infinity;
    for (const b of this.builders) {
      const d = Math.hypot(e.pos.x - b.plot.x, e.pos.z - b.plot.z);
      const sec = b.security;
      const s = -d * 0.5 + (1 - (sec.wall ?? 0)) * 26 + (sec.breaches > 0 ? 6 : 0) + (b.alive ? 4 : -20);
      if (s > bestScore) { bestScore = s; best = b; }
    }
    l.targetPlot = best || null;

    let pathInfo = { dx: 0, dz: 0, dy: 0, blocked: false, smashDist: 1 };
    if (best) {
      const goal = { x: best.ent.pos.x, y: best.ent.pos.y, z: best.ent.pos.z };
      const breach = weakestBreach(this.world, best.plot, best.groundY, e.pos);
      l.breach = breach;
      const spotted = scored.find((s) => s.vis && s.d < 18);
      if (spotted) {
        // it can see prey → run at it. No search needed; the value head learns
        // quickly that visibility is where the kill reward lives.
        l.path = null;
        l.chasing = spotted.b.id;
        const dx = spotted.b.ent.pos.x - e.pos.x, dz = spotted.b.ent.pos.z - e.pos.z;
        const len = Math.hypot(dx, dz) || 1;
        l.pathDir = {
          x: dx / len, z: dz / len, dx: spotted.b.ent.pos.x - e.pos.x, dz: spotted.b.ent.pos.z - e.pos.z,
          dy: spotted.b.ent.pos.y - e.pos.y, yaw: Math.atan2(-dx, -dz), dist: spotted.d, node: null,
        };
      } else {
        l.chasing = null;
        const stale = !l.path || l.pathT <= 0 || !l.lastGoal
          || Math.hypot(goal.x - l.lastGoal.x, goal.z - l.lastGoal.z) > 3;
        if (stale) {
          l.pathT = 3;
          l.lastGoal = { x: goal.x, y: goal.y, z: goal.z };
          l.path = astar(this.world, e.pos, goal, {
            height: 4,
            smashCost: Math.max(0.5, 2.2 - l.stuck * 0.4),
            maxNodes: 2400,
          });
          l.breachGoal = !!(l.path && l.path.blocked);
        } else {
          l.pathT -= 0.25;
        }
        const stopAt = l.path && l.path.smashAt > 1 ? l.path.smashAt : -1;
        l.pathDir = pathDirection(l.path?.path, e.pos, { stopAt });
      }
      const cs = Math.cos(e.yaw), sn = Math.sin(e.yaw);
      const pd = l.pathDir;
      // A wall is "ahead" if something breakable sits in front of the head, or
      // the planned path has to open one and we are standing at it.
      const wantVec = best ? { x: best.ent.pos.x - e.pos.x, z: best.ent.pos.z - e.pos.z } : null;
      const info = smashTargetInfo(l, this.world, l.stuck > 5 ? 4 : 2.8, wantVec);
      l.smashCell = info ? info.cell : null;
      if (info) {
        // aim the head at the block it intends to open, so turning, smashing and
        // walking through the breach form one continuous line
        const adx = info.cell.x + 0.5 - e.pos.x, adz = info.cell.z + 0.5 - e.pos.z;
        const alen = Math.hypot(adx, adz) || 1;
        l.pathDir = { ...l.pathDir, x: adx / alen, z: adz / alen, yaw: Math.atan2(-adx, -adz), dx: adx, dz: adz };
      }
      const wallAhead = !!info
        || (!!(l.path && l.path.blocked) && pd.dist <= 2.4)
        || (l.stuck > 7 && !!breach);
      l.wallAhead = wallAhead;
      pathInfo = {
        dx: pd.x * cs + pd.z * sn,
        dz: -pd.x * sn + pd.z * cs,
        dy: pd.dy,
        blocked: wallAhead,
        smashDist: clamp((info ? info.dist : (l.path?.smashAt ?? 6)) / 8, 0, 1),
      };
    } else {
      l.pathDir = { x: 0, z: 0, dx: 0, dz: 0, dy: 0, yaw: null, dist: 0, node: null };
      l.wallAhead = false;
    }
    const nb = best ? {
      wall: best.security.wall,
      roof: best.security.roof,
      breach: clamp(best.security.breaches / 60, 0, 1),
      dist: Math.hypot(e.pos.x - best.plot.x, e.pos.z - best.plot.z) / 30,
    } : {};
    const ctx = {
      world: this.world,
      ent: e,
      owner: -1,
      phase: this.phase,
      timeLeft: clamp(this.phaseT / durationFor(this.phase, this.opts), 0, 1),
      cycle: this.cycle,
      stuck: l.stuck,
      attackCd: l.attackCd > 0,
      hits: l.stats.kills + l.stats.grabs,
      smashed: l.stats.smashed,
      rewards: l.reward,
      prey: { aliveCount: alive.length },
      targets,
      pathInfo,
      pathDir: l.pathDir,
      nearestBase: nb,
      targetIsHuman: !!(best && best.isHuman),
      playerAlive: this.builders[0] ? this.builders[0].alive : false,
      anyHiding: alive.some((b) => b.cover >= 2),
      light: this.world.lightAt(Math.floor(e.pos.x), Math.floor(e.pos.y + 1), Math.floor(e.pos.z)),
      canStrike: !!findPrey(l, this.builders, this.world),
      canSmash: !!l.smashCell || !!(l.path && l.path.blocked),
      blocked: pathInfo.blocked,
      needsLeap: pathInfo.dy >= 2,
      targetUp: scored.length ? Math.max(0, ...scored.map((s) => s.b.ent.pos.y - e.pos.y)) : 0,
      coolingDown: l.attackCd > 0,
    };
    ctx.scalars = locustScalars(ctx);
    return ctx;
  }

  decideLocust() {
    const l = this.locust;
    if (!l || !l.active || l.grabbed) return;
    const ctx = this.locustCtx(l);
    const legal = locustLegalMask(l, {
      entities: this.builders,
      world: this.world,
      want: l.targetPlot ? { x: l.targetPlot.ent.pos.x - l.ent.pos.x, z: l.targetPlot.ent.pos.z - l.ent.pos.z } : null,
    });
    let action = 0;
    if (l.brain) {
      const obs = l.brain.encode(ctx);
      const r = l.brain.decide(obs, legal, ctx, this.phase);
      action = r.action;
      l.pendingAction = action;
    }
    const res = executeLocustAction(l, action, {
      world: this.world,
      now: this.now,
      entities: this.builders,
      onBlockBroken: (cell) => this.onLocustBrokeBlock(cell),
    });
    if (res.events.length) this.pushEvent(...r_events(res));
    l.reward += res.reward;
    if (l.brain) {
      l.brain.remember(l.reward, this.phase, false);
      l.reward = 0;
    }
  }

  /**
   * A block the Locust demolished. `own` has to be handed in: breakBlock() has
   * already reset the cell's owner to natural by the time this runs, so reading
   * it back here used to silently return and nobody was ever charged for the
   * hole in their wall.
   */
  onLocustBrokeBlock(cell, own = null) {
    if (own === null) own = this.world.getOwner(cell.x, cell.y, cell.z);
    if (own === OWNER_NATURAL || own === OWNER_LOCUST) return;
    const b = this.builders.find((x) => x.owner === own);
    if (!b) return;
    b.stats.lost++;
    b.reward += RW.BLOCK_LOST;
  }

  /* ------------------------------------------------------------ scoring */

  /** `who` is a builder (cached per world revision) or a bare plot centre. */
  securityOf(who, force = false) {
    const b = who && who.plot ? who : null;
    const plot = b ? b.plot : who;
    if (!force && b && b.revAt === this.world.revision && b.security) return b.security;
    const gy = plot.groundY ?? plot.y ?? this.world.surfaceY(plot.x, plot.z);
    const s = this.world.security(plot, gy, b ? b.shell : null);
    s.sealed = s.wall >= BASE.GOAL_WALL;
    if (b) { b.security = s; b.revAt = this.world.revision; }
    return s;
  }

  /** the shell cell this builder should fill next (also feeds the expert prior) */
  pickTargetCell(b) {
    const e = b.ent;
    let best = null, bestScore = Infinity;
    const list = b.shell.all;
    for (let i = 0; i < list.length; i++) {
      const c = list[i];
      if (isSolid(this.world.get(c.x, c.y, c.z))) continue;
      const d = Math.hypot(c.x + 0.5 - e.pos.x, c.z + 0.5 - e.pos.z) + Math.abs(c.y - e.pos.y) * 1.4;
      const s = d + (c.y - b.groundY) * 0.9;
      if (s < bestScore) { bestScore = s; best = c; }
    }
    b.targetCell = best;
    b.needRoof = !!best && best.y - e.pos.y > 1.2;
    return best;
  }

  /* --------------------------------------------------- event/log plumbing */

  pushEvent(...evts) {
    for (const e of evts) this.events.push({ ...e, at: this.now, phase: this.phase, cycle: this.cycle });
  }
  drainEvents() {
    const e = this.events;
    this.events = [];
    return e;
  }
  say(msg, who = 'world') {
    this.log.push({ msg, who, at: this.now, cycle: this.cycle });
    if (this.log.length > 160) this.log.shift();
  }

  trainStep(mini) {
    if (!this.learn) return null;
    const t0 = Date.now();
    const stats = this.league.trainAll(mini ?? this.league.cfg.MINI_BATCH);
    this.lastTrain = stats;
    this.trainTime = Date.now() - t0;
    return stats;
  }

  /* ------------------------------------------------------------ snapshots */

  snapshot() {
    return {
      phase: this.phase,
      cycle: this.cycle,
      timeLeft: Math.max(0, this.phaseT / 1000),
      timeTotal: durationFor(this.phase, this.opts) / 1000,
      tick: this.tickCount,
      now: this.now,
      finished: this.finished,
      builders: this.builders.map((b) => ({
        id: b.id, name: b.name, alive: b.alive, isHuman: b.isHuman, health: b.health,
        x: b.ent.pos.x, y: b.ent.pos.y, z: b.ent.pos.z, yaw: b.ent.yaw, pitch: b.ent.pitch,
        wall: b.security.wall, roof: b.security.roof, sealed: !!b.security.sealed,
        breaches: b.security.breaches, own: b.security.blocks, placed: b.stats.placed,
        broken: b.stats.broken, lost: b.stats.lost, deaths: b.stats.deaths,
        escaped: b.stats.escapes, grabbed: b.grabbed, seen: b.seenByLocust,
        score: b.score, inv: b.inv.total, selected: b.inv.selected, action: b.pendingAction,
        reward: +b.reward.toFixed(2), cover: b.cover, stuck: +b.stuck.toFixed(1),
        invuln: b.invuln,
      })),
      locust: this.locust ? {
        active: true,
        x: this.locust.ent.pos.x, y: this.locust.ent.pos.y, z: this.locust.ent.pos.z,
        yaw: this.locust.ent.yaw, pitch: this.locust.ent.pitch,
        kills: this.locust.stats.kills, smashed: this.locust.stats.smashed,
        grabs: this.locust.stats.grabs, escapes: this.locust.stats.escapes,
        holding: this.locust.grabbed ? this.locust.grabbed.name : null,
        stuck: +this.locust.stuck.toFixed(1), action: this.locust.pendingAction,
        reward: +this.locust.reward.toFixed(2),
      } : null,
      tally: this.tally,
      log: this.log.slice(-16),
      world: { revision: this.world.revision, edits: this.world.editCount },
      training: this.lastTrain,
      trainMs: this.trainTime,
      stats: this.league.stats().map(slimBrainStats),
    };
  }

  /* --------------------------------------------- human controller hooks */

  get player() {
    return this.builders.find((x) => x.isHuman) || null;
  }
  setPlayerLook(dx, dy) {
    const b = this.player;
    if (!b || !b.alive) return;
    b.ent.yaw -= dx * 0.0026;
    b.ent.pitch = clamp(b.ent.pitch - dy * 0.0026, -1.35, 1.25);
  }
  setPlayerMove(mx, mz, sprint) {
    const b = this.player;
    if (!b) return;
    if (b.grabbed) return;
    b.intent.mx = mx;
    b.intent.mz = mz;
    b.intent.sprint = sprint ? 1 : 0;
  }
  playerJump() {
    const b = this.player;
    if (!b || !b.alive) return;
    if (b.grabbed) { b.struggle = 1; return; }
    this.execBuilder(b, ACT.JUMP, { preserveIntent: true });
  }
  playerBreak() {
    const b = this.player;
    if (!b || !b.alive) return;
    this.execBuilder(b, ACT.BREAK_FRONT, { preserveIntent: true });
  }
  playerPlace() {
    const b = this.player;
    if (!b || !b.alive) return;
    this.execBuilder(b, ACT.PLACE_FRONT, { preserveIntent: true });
  }
  playerCycle(dir) {
    const b = this.player;
    if (!b) return;
    b.inv.cycle(dir);
  }
  playerSelect(slot) {
    const b = this.player;
    if (!b) return;
    b.inv.selectSlot(slot);
  }
  /** Discrete action from the client (same entry point the bots' policy uses). */
  playerAct(action) {
    const b = this.player;
    if (!b || !b.alive) return;
    this.execBuilder(b, action | 0, { preserveIntent: true });
  }
}

function r_events(res) {
  return res.events;
}

function slimBrainStats(s) {
  return {
    id: s.id, name: s.name, kind: s.kind, params: s.params, trainSteps: s.trainSteps,
    decisions: s.decisions, bufferGames: s.bufferGames, bufferSteps: s.bufferSteps,
    episodeReward: s.episodeReward, totalReward: s.totalReward, assist: s.assist,
    value: s.value, top: s.top, sims: s.sims, nodes: s.nodes, searchMs: s.searchMs,
    trainMs: s.trainMs, temp: s.temp, learning: s.learning, trajLen: s.trajLen,
    loss: s.last ? +s.last.loss.toFixed(4) : null,
    policy: s.last ? +s.last.policy.toFixed(4) : null,
    valueLoss: s.last ? +s.last.value.toFixed(4) : null,
    prefixLoss: s.last ? +s.last.reward.toFixed(4) : null,
    consist: s.last ? +s.last.consist.toFixed(4) : null,
    entropy: s.last ? +s.last.entropy.toFixed(3) : null,
    gradNorm: s.last ? +s.last.gradNorm.toFixed(3) : null,
    badWeights: s.health ? s.health.bad : 0,
  };
}

function countMap(inv) {
  const o = {};
  for (const [id, n] of inv.counts) o[id] = n;
  return o;
}

function frontInfo(world, e, b) {
  const cp = Math.cos(e.pitch || 0);
  const dir = { x: -Math.sin(e.yaw) * cp, y: Math.sin(e.pitch || 0), z: -Math.cos(e.yaw) * cp };
  const r = raycast(world, e.pos.x, e.pos.y + e.eye, e.pos.z, dir.x, dir.y, dir.z, ENTITY.REACH);
  if (!r.hit) {
    const fx = Math.floor(e.pos.x + dir.x * 2), fy = Math.floor(e.pos.y), fz = Math.floor(e.pos.z + dir.z * 2);
    return {
      dist: 2, solid: false,
      canPlace: world.get(fx, fy, fz) === B.AIR,
      canBreak: false,
    };
  }
  const who = { kind: b.isHuman ? 'player' : 'builder', owner: b.owner };
  const p = { x: r.x + r.nx, y: r.y + r.ny, z: r.z + r.nz };
  return {
    dist: r.dist,
    solid: isSolid(r.id),
    canPlace: world.get(p.x, p.y, p.z) === B.AIR,
    canBreak: world.canBreak(r.x, r.y, r.z, who),
  };
}

function durationFor(phase, opts = {}) {
  const k = opts.phaseScale ?? 1;
  switch (phase) {
    case PHASE.BUILD: return TIMING.BUILD_MS * k;
    case PHASE.HUNT: return TIMING.HUNT_MS * k;
    case PHASE.REVIVE: return TIMING.REVIVE_MS * k;
    default: return TIMING.LOBBY_MS * k;
  }
}

export { durationFor, BUILD_INTERVAL, LOCUST_INTERVAL };
