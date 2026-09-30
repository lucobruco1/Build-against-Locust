/**
 * tools/simulate.js
 * ---------------------------------------------------------------------------
 * Headless run of the whole game — same modules, no renderer. Use it to watch
 * the seven EfficientZero builders and the Locust actually learn, to benchmark
 * the search, and as the integration test harness.
 *
 *   node tools/simulate.js --cycles 3 --speed 60 --learn 1
 *
 * flags
 *   --cycles N     how many build+hunt cycles to play (default 2)
 *   --speed  F     game-time multiplier per tick (default 1 = the same 30 Hz the
 *                  browser runs at). >1 makes each tick cover more game time,
 *                  which *skips AI decisions* — fine for a smoke test, never for
 *                  judging how well the networks play.
 *   --learn  0|1   enable the EfficientZero updates (default 1)
 *   --assist F     weight of the expert prior blended into the root (0..1,
 *                  default 0.9 — the same value the live game starts with)
 *   --sims   N     scale factor on MCTS simulations
 *   --flat   0|1   flat test arena instead of generated terrain
 *   --seed   N     worldgen / behaviour seed
 *   --report PATH  write a JSON report
 *   --save   PATH  write the trained weights (Float32 packs) to JSON
 *   --load   PATH  resume weights from JSON
 * ---------------------------------------------------------------------------
 */

import fs from 'node:fs';
import path from 'node:path';
import { Match } from '../game/match.js';
import { TIMING } from '../shared/rules.js';
import { packParams, f32ToBase64 } from '../ai/nn.js';

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0 || i + 1 >= process.argv.length) return def;
  const v = process.argv[i + 1];
  return v === '0' ? 0 : v === '1' ? 1 : Number.isNaN(Number(v)) ? v : Number(v);
}

const cycles = arg('cycles', 2);
const speed = Math.max(0.05, arg('speed', 1));
const learn = arg('learn', 1) === 1;
const assist = arg('assist', 0.9);
const simScale = arg('sims', 1);
const flat = arg('flat', 0) === 1;
const seed = arg('seed', 20260930);
const reportArg = arg('report', null);
const reportPath = typeof reportArg === 'string' && reportArg ? reportArg : null;
if (reportArg !== null && reportPath === null) console.warn('--report wants a file path, e.g. --report .data/report.json');
const savePath = arg('save', null);
const loadPath = arg('load', null);
const phaseScale = arg('phase', 1);

const match = new Match({
  seed,
  flat,
  learn,
  assist,
  simScale,
  playerAI: true,          // the "player" slot is an 8th EfficientZero bot here
  maxCycles: cycles,
  trainEvery: 60,
  timeScale: speed,
  phaseScale,
});

if (loadPath && fs.existsSync(loadPath)) {
  const data = JSON.parse(fs.readFileSync(loadPath, 'utf8'));
  let n = 0;
  for (const b of match.league.brains) {
    const d = data[b.id];
    if (!d) continue;
    if (b.loadCheckpoint(d)) n++;
  }
  console.log(`resumed ${n} brains from ${loadPath}`);
}

console.log('┌──────────────────────────────────────────────────────────────');
console.log('│ Build to Survive the Locust — headless run');
console.log(`│ world ${match.world.sx}×${match.world.sy}×${match.world.sz}  builders ${match.builders.length}`);
console.log(`│ obs dim builder=${match.codecBuilder.dim} locust=${match.codecLocust.dim}`);
console.log(`│ params/brain ${match.league.brains[0].model.nParams}  MCTS sims ${match.league.cfg.SIMS}  learn ${learn}`);
console.log('└──────────────────────────────────────────────────────────────');

// a full cycle is BUILD + HUNT + REVIVE (scaled by --phase); take the real
// numbers from the rules instead of hard-coding them, otherwise a normal-length
// hunt gets truncated at the tick cap and kills are under-reported
const cycleSeconds = Math.ceil(((TIMING.BUILD_MS + TIMING.HUNT_MS + TIMING.REVIVE_MS) * phaseScale) / 1000) + 5;
const targetTicks = Math.max(30, Math.ceil((30 * cycleSeconds * cycles) / speed));
let last = Date.now();
let lastReportTick = 0;
const REPORT_EVERY = 30 * 15;  // one line per 15 s of game time
const history = [];
while (!match.finished && match.tickCount < targetTicks) {
  match.update(1000 / 30);
  const tick = match.tickCount;
  if (tick - lastReportTick >= REPORT_EVERY || tick >= targetTicks) {
    lastReportTick = tick;
    const s = match.snapshot();
    const brains = s.stats;
    const placed = s.builders.reduce((a, b) => a + b.placed, 0);
    const wall = (s.builders.reduce((a, b) => a + b.wall, 0) / s.builders.length).toFixed(2);
    const losses = brains.filter((b) => b.loss !== null);
    const avgLoss = losses.length ? (losses.reduce((a, b) => a + b.loss, 0) / losses.length).toFixed(3) : '—';
    const alive = s.builders.filter((b) => b.alive).length;
    const dtSec = ((Date.now() - last) / 1000).toFixed(1);
    last = Date.now();
    const row = {
      tick, cycle: s.cycle, phase: s.phase, placed, wall, alive,
      locustKills: s.locust ? s.locust.kills : 0,
      loss: avgLoss,
      trainSteps: Math.max(...brains.map((b) => b.trainSteps)),
    };
    history.push(row);
    console.log(
      `c${String(s.cycle).padEnd(2)} ${s.phase.padEnd(6)} t=${dtSec}s  placed=${String(placed).padStart(4)}` +
      ` wall=${wall} alive=${alive}/8  kills=${row.locustKills}  loss=${avgLoss}  updates=${row.trainSteps}`,
    );
  }
}

const snap = match.snapshot();
console.log('\n── final ──────────────────────────────────────────────');
for (const b of snap.builders) {
  console.log(`${b.name.padEnd(7)} score ${String(b.score).padStart(4)} wall ${(b.wall * 100).toFixed(0)}% roof ${(b.roof * 100).toFixed(0)}% placed ${String(b.placed).padStart(3)} deaths ${b.deaths} escaped ${b.escaped}`);
}
if (snap.locust) console.log(`Locust: ${snap.locust.kills} kills, ${snap.locust.smashed} blocks smashed, ${snap.locust.escapes} prey escaped`);
console.log('\nper-brain EfficientZero state:');
for (const b of snap.stats) {
  console.log(`${b.name.padEnd(9)} updates ${String(b.trainSteps).padStart(5)} buffer ${b.bufferGames}g/${b.bufferSteps}t  loss ${b.loss ?? '—'} policy ${b.policy ?? '—'} value ${b.valueLoss ?? '—'} prefix ${b.prefixLoss ?? '—'} consist ${b.consist ?? '—'} H ${b.entropy ?? '—'} assist ${b.assist}`);
}

if (reportPath) {
  const dir = path.dirname(reportPath);
  if (dir && dir !== '.') fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify({ seed, cycles, speed, learn, assist, history, snapshot: snap }, null, 1));
  console.log(`\nreport → ${reportPath}`);
}
if (savePath) {
  const out = {};
  for (const b of match.league.brains) {
    const pack = packParams(b.model.params);
    out[b.id] = {
      b64: f32ToBase64(pack),
      obsDim: b.obsDim,
      nActions: b.nActions,
      stepCounter: b.stepCounter,
      gameCount: b.gameCount,
      decisions: b.decisions,
      trainSteps: b.model.trainSteps,
    };
  }
  fs.mkdirSync(path.dirname(savePath) || '.', { recursive: true });
  fs.writeFileSync(savePath, JSON.stringify(out));
  console.log(`weights → ${savePath}`);
}
