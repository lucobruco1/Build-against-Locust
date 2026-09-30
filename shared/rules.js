/**
 * shared/rules.js
 * ---------------------------------------------------------------------------
 * Single source of truth for the whole game: the voxel world size, the block
 * table, the phase machine timings, both discrete action spaces, the reward
 * shaping coefficients and the EfficientZero hyper-parameters.
 *
 * This file is imported by the browser client, the Node server and the headless
 * simulator/tests, so it must stay free of DOM / node-only APIs.
 * ---------------------------------------------------------------------------
 */

/* ------------------------------------------------------------------ world -- */

export const WORLD = {
  SX: 72, // blocks along +X
  SY: 40, // vertical blocks
  SZ: 72, // blocks along +Z
  CHUNK: 12, // chunk edge length (72/12 = 6x6 chunks)
  SEA_LEVEL: 11,
  BEDROCK: 0, // y of unbreakable bedrock
  MAX_Y_PLAY: 39,
};

export const CHUNKS_X = Math.ceil(WORLD.SX / WORLD.CHUNK);
export const CHUNKS_Z = Math.ceil(WORLD.SZ / WORLD.CHUNK);

/** Height of a builder's AABB / eye height (blocks). */
export const ENTITY = {
  PLAYER: { width: 0.6, height: 1.8, eye: 1.62, speed: 4.35, sprint: 1.75, jump: 8.6 },
  BOT: { width: 0.6, height: 1.8, eye: 1.62, speed: 4.35, sprint: 1.75, jump: 8.6 },
  LOCUST: { width: 1.1, height: 3.9, eye: 3.6, speed: 5.3, jump: 11.5 },
  GRAVITY: -30,
  TERMINAL: -78,
  REACH: 4.5, // block interaction distance
  GRAB_RANGE: 3.1, // locust grab distance
  REACH_THROUGH_HOLE: 5.6, // long-arm grab through a roof breach, needs LOS
  GRAB_TIME: 1.25, // seconds a victim is dangled before the stab
  STRUGGLE_ESCAPE: 0.34,
};

/* --------------------------------------------------------------- blocks --- */

export const B = {
  AIR: 0,
  GRASS: 1,
  DIRT: 2,
  STONE: 3,
  COBBLE: 4,
  PLANK: 5,
  LOG: 6,
  LEAF: 7,
  SAND: 8,
  GLASS: 9,
  BRICK: 10,
  BEDROCK: 11,
  WATER: 12,
  MOSS: 13, // mossy cobble – the "old ruin" block
  BONE: 14, // pale decorative block, locust trophy
};

/**
 * Block table.
 *  solid      – participates in collision & hides neighbour faces
 *  opaque     – blocks skylight / view (glass, leaves, water don't)
 *  hardness   – seconds of continuous mining for the player
 *  locustHP   – "hits" the Locust needs to smash through it
 *  tiles      – [top, side, bottom] indices into the texture atlas
 *  placeable  – offered in the creative hotbar
 */
export const BLOCK_DEFS = [];
function defineBlock(id, name, o) {
  BLOCK_DEFS[id] = {
    id,
    name,
    solid: o.solid !== false,
    opaque: o.opaque !== false,
    hardness: o.hardness ?? 0.6,
    locustHP: o.locustHP ?? 2,
    tiles: o.tiles,
    placeable: !!o.placeable,
    buildable: o.buildable !== false, // counted for base-security scoring
    color: o.color,
  };
}
defineBlock(B.AIR, 'air', { solid: false, opaque: false, hardness: 0, tiles: [0, 0, 0] });
defineBlock(B.GRASS, 'grass', { hardness: 0.45, tiles: [1, 2, 3], placeable: true, color: '#6a9c3f' });
defineBlock(B.DIRT, 'dirt', { hardness: 0.4, tiles: [3, 3, 3], placeable: true, color: '#7a5a3a' });
defineBlock(B.STONE, 'stone', { hardness: 1.5, locustHP: 4, tiles: [4, 4, 4], placeable: true, color: '#8e8e93' });
defineBlock(B.COBBLE, 'cobble', { hardness: 1.35, locustHP: 4, tiles: [5, 5, 5], placeable: true, color: '#7c7c81' });
defineBlock(B.PLANK, 'planks', { hardness: 0.9, locustHP: 3, tiles: [6, 6, 6], placeable: true, color: '#a97c45' });
defineBlock(B.LOG, 'log', { hardness: 1.0, locustHP: 3, tiles: [8, 7, 8], placeable: true, color: '#6b4c2a' });
defineBlock(B.LEAF, 'leaves', { hardness: 0.25, opaque: false, locustHP: 1, tiles: [9, 9, 9], placeable: true, color: '#3f6b28' });
defineBlock(B.SAND, 'sand', { hardness: 0.4, tiles: [10, 10, 10], placeable: true, color: '#d9cb92' });
defineBlock(B.GLASS, 'glass', { hardness: 0.35, opaque: false, locustHP: 1, tiles: [11, 11, 11], placeable: true, color: '#a9d8e8' });
defineBlock(B.BRICK, 'brick', { hardness: 1.7, locustHP: 5, tiles: [12, 12, 12], placeable: true, color: '#9d5a4a' });
defineBlock(B.BEDROCK, 'bedrock', { hardness: Infinity, locustHP: Infinity, tiles: [13, 13, 13], buildable: false, color: '#3a3a3f' });
defineBlock(B.WATER, 'water', { hardness: Infinity, opaque: false, locustHP: Infinity, tiles: [14, 14, 14], buildable: false, color: '#3a6ec9' });
defineBlock(B.MOSS, 'mossy stone', { hardness: 1.4, locustHP: 4, tiles: [15, 15, 15], placeable: true, color: '#5f7a4f' });
defineBlock(B.BONE, 'bone block', { hardness: 0.8, locustHP: 2, tiles: [16, 16, 16], placeable: true, color: '#ded8c4' });

export const HOTBAR = [B.PLANK, B.COBBLE, B.BRICK, B.STONE, B.SAND, B.GLASS, B.LOG, B.DIRT, B.LEAF];

/** Blocks the AI builders are allowed to place (natural + harvested). */
export const BOT_PLACEABLE = [B.DIRT, B.GRASS, B.STONE, B.COBBLE, B.PLANK, B.SAND, B.LOG, B.BRICK, B.MOSS, B.BONE, B.LEAF, B.GLASS];

export function isSolid(id) {
  const d = BLOCK_DEFS[id];
  return !!d && d.solid;
}
export function isOpaque(id) {
  const d = BLOCK_DEFS[id];
  return !!d && d.opaque;
}
export function isBreakable(id) {
  const d = BLOCK_DEFS[id];
  return !!d && d.solid && Number.isFinite(d.hardness);
}
export function def(id) {
  return BLOCK_DEFS[id] || BLOCK_DEFS[B.AIR];
}

/* --------------------------------------------------------------- phases --- */

export const PHASE = {
  LOBBY: 'lobby',
  BUILD: 'build',
  HUNT: 'hunt',
  REVIVE: 'revive',
};

export const TIMING = {
  BUILD_MS: 90_000, // "1.5 minutes to build a secure base"
  HUNT_MS: 180_000, // "it continues for 3 minutes"
  REVIVE_MS: 5_000, // revive / tally screen between cycles
  LOBBY_MS: 4_000,
  TICK_MS: 1000 / 30, // fixed simulation rate
  MAX_CATCHUP_STEPS: 6,
};

/** How much of the world the base-security score looks at. */
export const BASE = {
  RADIUS: 7, // build plot radius around the home marker
  WALL_RING: 2, // the scored wall shell (a 5x5 bunker around the marker)
  OUTER_RING: 3, // bonus shell: a second layer of wall, not required
  // A Locust is 3.9 blocks tall, so a base only counts as sealed when the
  // walls are 4 high: the scoring metric and the monster's hitbox agree.
  WALL_TOP: 4,
  ROOF_CLEAR: 4, // height of the roof plane above the ground plane
  ROOF_EXTENT: 2, // roof spans (2*EXTENT+1)^2 cells
  GOAL_WALL: 0.72, // wall coverage that counts as "sealed"
  GOAL_ROOF: 0.45,
};

/* ---------------------------------------------------------- action spaces --- */

/** Shared action space for the player and the 7 builders (one-hot index). */
export const ACT = {
  NOOP: 0,
  FORWARD: 1,
  BACKWARD: 2,
  STRAFE_LEFT: 3,
  STRAFE_RIGHT: 4,
  TURN_LEFT: 5,
  TURN_RIGHT: 6,
  LOOK_UP: 7,
  LOOK_DOWN: 8,
  JUMP: 9,
  PLACE_FRONT: 10,
  BREAK_FRONT: 11,
  PLACE_DOWN: 12,
  BREAK_DOWN: 13,
  SPRINT: 14,
};
export const BUILDER_ACTION_NAMES = [
  'noop', 'walk fwd', 'walk back', 'strafe left', 'strafe right',
  'turn left', 'turn right', 'look up', 'look down', 'jump',
  'place block', 'break block', 'place under', 'break under', 'sprint',
];
export const BUILDER_N_ACTIONS = BUILDER_ACTION_NAMES.length;

/** Action space of the Locust. */
export const LACT = {
  NOOP: 0,
  FORWARD: 1,
  BACKWARD: 2,
  STRAFE_LEFT: 3,
  STRAFE_RIGHT: 4,
  TURN_LEFT: 5,
  TURN_RIGHT: 6,
  LOOK_UP: 7,
  LOOK_DOWN: 8,
  LEAP: 9,
  SMASH_BLOCK: 10,
  STRIKE: 11,
};
export const LOCUST_ACTION_NAMES = [
  'idle', 'stalk fwd', 'step back', 'sidestep L', 'sidestep R',
  'turn left', 'turn right', 'raise head', 'lower head', 'leap',
  'smash block', 'strike prey',
];
export const LOCUST_N_ACTIONS = LOCUST_ACTION_NAMES.length;

/* -------------------------------------------------------------- rewards --- */

/**
 * Reward shaping. Every number here is a *per-tick* reward unless suffixed
 * with _EV (a one-off event reward). These are what the EfficientZero value
 * head and value-prefix head are trained to predict.
 */
export const RW = {
  // --- per *decision* event rewards ----------------------------------------
  PLACE: 0.10, // placed a block inside own plot
  PLACE_WALL: 0.26, // ... and it filled a cell of the scored shell
  PLACE_OUTER: 0.10, // ... or of the bonus outer ring
  PLACE_WASTE: -0.03, // placed outside the plot / pointless
  BREAK_OWN: -0.05, // deleted one of your own blocks (usually a reposition)
  MINE: 0.05, // harvested a block (resources are finite for the bots)
  FALL_DAMAGE: -0.35,
  HURT: -0.9,
  DEATH: -2.4,
  SURVIVE_CYCLE: 1.8,
  SECURE_BONUS: 1.0, // base reached "sealed" status
  BLOCK_LOST: -0.18, // the Locust smashed one of your blocks
  GRABBED: -0.5,
  ESCAPE_GRAB: 0.8,
  ILLEGAL: -0.02,
  // --- per *tick* rewards (kept tiny: 30 ticks/s × 180 s = 5400 ticks) ----
  STEP_COST: -0.0018,
  SHELTER_COVER: 0.0016, // hidden from the Locust during the hunt
  NEAR_PREY_PENALTY: -0.0022,
  LOCUST_VISIBLE: -0.0014, // it can see you
  // --- Locust -------------------------------------------------------------
  L_STEP: -0.0016,
  L_BREAK: 0.17,
  L_GRAB: 0.8,
  L_KILL: 2.6,
  L_PLAYER_KILL: 1.4,
  L_PROGRESS: 0.0016, // closing the distance on the nearest prey
  L_STUCK: -0.012,
  L_IDLE_TARGET: -0.05,
  L_SURVIVE: 0.4,
};

/* -------------------------------------------------- efficientzero config --- */

export const EZ = {
  LATENT: 64, // latent state size
  HIDDEN: 128, // residual block width
  N_RES: 2, // residual blocks in each tower
  ACC_DIM: 4, // value-prefix accumulator vector ("LSTM" slot of the paper)

  VALUE_SUPPORT: 61, // categorical support (value transformation)
  VALUE_MIN: -12,
  VALUE_MAX: 12,
  REW_SUPPORT: 41,
  REW_MIN: -4,
  REW_MAX: 4,

  GAMMA: 0.997, // discount used inside the search / for the prefix
  UNROLL: 5, // l_unroll
  GAMMA_PREFIX: 0.5, // discount inside the value-prefix sum (EZ uses 0.5)
  SIMS: 24, // MCTS simulations per decision (browser budget)
  SIM_REANALYZE: 3, // MCTS simulations used by the reanalyze step
  CPUCT: 1.6, // PUCT exploration constant
  DIRICHLET_ALPHA: 0.35,
  DIRICHLET_EPS: 0.25,
  FPU_PARENT: 0.25, // first play out urgency of the parent
  RESCALE_ROOT: 0.1, // root visit-count rescaling exponent (AlphaZero/EZ)

  BATCH: 16,
  MINI_BATCH: 4, // sequences per gradient step
  REPLAY_SIZE: 24_000, // transitions stored
  GAMES_PER_BUFFER: 400,
  LR: 3e-4,
  WEIGHT_DECAY: 1e-4,
  GRAD_CLIP: 5.0,

  L_REWARD: 1.0, // λ1 value-prefix loss
  L_POLICY: 1.0, // λ2
  L_VALUE: 0.25, // λ3
  L_CONSIST: 2.0, // λ4 self-supervised temporal consistency (SimSiam)
  L_ENTROPY: 0.005,

  TARGET_SYNC: 40, // update target net every N optimizer steps
  UPDATES_PER_TICK: 2,
  MIN_BUFFER_FOR_TRAINING: 240,
  OFFPOLICY_TAU: 0.3, // τ of the off-policy correction (fraction of total steps)
  OFFPOLICY_MAX_DEPTH: 5,
  ALPHA_PRIO: 0.6, // priority exponent
  BETA_PRIO: 0.4, // importance sampling exponent (annealed to 1)
  ANNEAL_BETA_STEPS: 800,
  SVE_FRESH_STEPS: 400, // EZ-V2: use TD targets for fresh samples, SVE for stale
  NOISE_FLOOR: 1e-6,
};

/** Per-phase search budget: the hunt needs sharper tactics than block placing. */
export const SIMS_BY_PHASE = { [PHASE.BUILD]: 22, [PHASE.HUNT]: 30, [PHASE.LOBBY]: 8, [PHASE.REVIVE]: 8 };

/** Observation geometry. */
export const OBS = {
  SCAN_RX: 2, // horizontal half-extent of the local voxel scan (5x5)
  SCAN_RZ: 2,
  SCAN_DY: -1, // lowest scanned layer relative to the agent's feet
  SCAN_LAYERS: 4, // dy = -1..2
  MAX_TARGETS: 3, // closest entities described in the vector part
  LOCUST_SCAN_RX: 3,
  LOCUST_SCAN_RZ: 3,
  LOCUST_SCAN_DY: -1,
  LOCUST_SCAN_LAYERS: 4,
};

export const N_BOTS = 7;

export const BOT_NAMES = [
  'Notch', 'Ellie', 'Voxel', 'Mira', 'Brickz', 'Sable', 'Pixel',
];

/** Locust taunts – imitative of the child-like, badly punctuated T.O.E. voice. */
export const LOCUST_LINES = [
  'hii!! are you in there??',
  'i came to be your freind!!',
  'knock knock. i know you are hiding.',
  'its so dark in here isnt it??',
  'let me in please please please',
  'i can hear you breathing...',
  'you built a wall. walls are fun!!',
  'found you :)',
  'hold very still now.',
  'i am hungry. are you hungry??',
];

/* --------------------------------------------------------------- helpers --- */

export function idx3(x, y, z) {
  return (y * WORLD.SZ + z) * WORLD.SX + x;
}
export function unidx(i) {
  const x = i % WORLD.SX;
  const rest = (i - x) / WORLD.SX;
  const z = rest % WORLD.SZ;
  const y = (rest - z) / WORLD.SZ;
  return { x, y, z };
}
export function insideWorld(x, y, z) {
  return x >= 0 && y >= 0 && z >= 0 && x < WORLD.SX && y < WORLD.SY && z < WORLD.SZ;
}
export function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}
export function lerp(a, b, t) {
  return a + (b - a) * t;
}
export function sign(x) {
  return x < 0 ? -1 : 1;
}
/** Chebyshev (block) distance. */
export function chebyshev(a, b) {
  return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
}
export function dist2d(ax, az, bx, bz) {
  const dx = ax - bx, dz = az - bz;
  return Math.sqrt(dx * dx + dz * dz);
}
/** Deterministic PRNG (mulberry32) – used for worldgen and ε-greedy noise. */
export function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Base plot centres: the player in the middle, 7 bots on a ring. */
export function plotCenters() {
  const cx = WORLD.SX >> 1, cz = WORLD.SZ >> 1;
  const out = [{ x: cx, z: cz }];
  const R = 20;
  for (let i = 0; i < N_BOTS; i++) {
    const a = (i / N_BOTS) * Math.PI * 2 + Math.PI * 0.12;
    out.push({
      x: Math.round(cx + Math.cos(a) * R),
      z: Math.round(cz + Math.sin(a) * R),
    });
  }
  return out;
}

export function phaseDuration(phase) {
  switch (phase) {
    case PHASE.BUILD: return TIMING.BUILD_MS;
    case PHASE.HUNT: return TIMING.HUNT_MS;
    case PHASE.REVIVE: return TIMING.REVIVE_MS;
    default: return TIMING.LOBBY_MS;
  }
}

export default {
  WORLD, BLOCK_DEFS, B, ACT, LACT, PHASE, TIMING, RW, EZ, BASE, ENTITY, N_BOTS,
};
