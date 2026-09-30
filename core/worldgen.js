/**
 * core/worldgen.js
 * ---------------------------------------------------------------------------
 * Procedural Minecraft-like terrain: fBm value noise heightmap, soil profile,
 * a lake at sea level, scattered oak-ish trees, mossy ruins, and – most
 * importantly – 8 flattened build plots (the player's in the middle, the 7
 * bots on a ring) each marked with a bone block so the builders always know
 * where "home" is.
 * ---------------------------------------------------------------------------
 */

import { WORLD, B, N_BOTS, plotCenters, rng, clamp } from '../shared/rules.js';

/* ------------------------------------------------------------- value noise */

function hash2(seed, x, y) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(seed | 0, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function smooth(t) {
  return t * t * (3 - 2 * t);
}
function value2(seed, x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const a = hash2(seed, xi, yi);
  const b = hash2(seed, xi + 1, yi);
  const c = hash2(seed, xi, yi + 1);
  const d = hash2(seed, xi + 1, yi + 1);
  const u = smooth(xf), v = smooth(yf);
  return a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v;
}
function fbm(seed, x, y, octaves = 4, lacunarity = 2, gain = 0.5) {
  let amp = 1, freq = 1, sum = 0, norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * value2(seed + o * 977, x * freq, y * freq);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}

/* ----------------------------------------------------------------- sculpt */

export function generateWorld(world, seed = 1337) {
  const rand = rng(seed);
  const { SX, SY, SZ, SEA_LEVEL, CHUNK } = WORLD;
  const plots = plotCenters();
  const height = new Int16Array(SX * SZ);

  for (let z = 0; z < SZ; z++) {
    for (let x = 0; x < SX; x++) {
      const base = fbm(seed, x * 0.035, z * 0.035, 4);
      const detail = fbm(seed + 31, x * 0.11, z * 0.11, 3) * 0.35;
      const ridge = Math.pow(Math.abs(fbm(seed + 71, x * 0.018, z * 0.018, 2) * 2 - 1), 2) * 5;
      // gentle bowl so the map edge reads as a valley rim
      const ex = (x - SX / 2) / (SX / 2), ez = (z - SZ / 2) / (SZ / 2);
      const rim = (ex * ex + ez * ez) * 3.4;
      let h = Math.round(9 + base * 11 + detail * 4 + ridge + rim * rim * 0.9);
      h = clamp(h, 3, SY - 12);
      height[z * SX + x] = h;
    }
  }

  // flatten the build plots (blend towards a smooth pad)
  for (const p of plots) {
    const R = 8, F = 10.5;
    let target = 0, n = 0;
    for (let dz = -2; dz <= 2; dz++) {
      for (let dx = -2; dx <= 2; dx++) {
        const x = clamp(p.x + dx, 0, SX - 1), z = clamp(p.z + dz, 0, SZ - 1);
        target += height[z * SX + x]; n++;
      }
    }
    target = Math.round(target / n);
    target = Math.max(target, SEA_LEVEL + 1);
    for (let dz = -F; dz <= F; dz++) {
      for (let dx = -F; dx <= F; dx++) {
        const d = Math.sqrt(dx * dx + dz * dz);
        if (d > F) continue;
        const x = p.x + dx, z = p.z + dz;
        if (x < 0 || z < 0 || x >= SX || z >= SZ) continue;
        const t = clamp((F - d) / (R), 0, 1);
        const i = z * SX + x;
        height[i] = Math.round(height[i] * (1 - t) + target * t);
      }
    }
  }

  for (let z = 0; z < SZ; z++) {
    for (let x = 0; x < SX; x++) {
      const h = height[z * SX + x];
      const damp = fbm(seed + 505, x * 0.05, z * 0.05, 2);
      for (let y = 0; y <= h; y++) {
        let id;
        if (y === 0) id = B.BEDROCK;
        else if (y < h - 3) id = B.STONE;
        else if (y < h) id = damp > 0.62 ? B.SAND : B.DIRT;
        else {
          if (h <= SEA_LEVEL) id = B.SAND;
          else if (damp > 0.72) id = B.MOSS;
          else id = B.GRASS;
        }
        if (y > 0 && y < h && damp > 0.8 && y % 7 === 0) id = B.COBBLE; // stone rubble pockets
        world.blocks[world.index(x, y, z)] = id;
      }
      // water fill in the low places
      if (h < SEA_LEVEL) {
        for (let y = h + 1; y <= SEA_LEVEL; y++) world.blocks[world.index(x, y, z)] = B.WATER;
      }
    }
  }

  /* trees ------------------------------------------------------------- */
  const treeCount = Math.floor(SX * SZ * 0.0032);
  for (let t = 0; t < treeCount; t++) {
    const x = 2 + Math.floor(rand() * (SX - 4));
    const z = 2 + Math.floor(rand() * (SZ - 4));
    const h = height[z * SX + x];
    if (h <= SEA_LEVEL + 1) continue;
    if (world.get(x, h, z) !== B.GRASS) continue;
    if (inPlot(plots, x, z, 6)) continue; // keep the arenas clean
    const trunk = 3 + Math.floor(rand() * 3);
    for (let i = 1; i <= trunk; i++) world.blocks[world.index(x, h + i, z)] = B.LOG;
    const cy = h + trunk;
    for (let dy = -1; dy <= 2; dy++) {
      const r = dy <= 0 ? 2 : dy === 1 ? 2 : 1;
      for (let dx = -r; dx <= r; dx++) {
        for (let dz = -r; dz <= r; dz++) {
          if (Math.abs(dx) === r && Math.abs(dz) === r && rand() < 0.6) continue;
          const yy = cy + dy;
          if (yy >= SY) continue;
          const i = world.index(x + dx, yy, z + dz);
          if (x + dx < 0 || z + dz < 0 || x + dx >= SX || z + dz >= SZ) continue;
          if (world.blocks[i] === B.AIR) world.blocks[i] = B.LEAF;
        }
      }
    }
  }

  /* ruins ------------------------------------------------------------- */
  const ruins = 9;
  for (let r = 0; r < ruins; r++) {
    const x = 5 + Math.floor(rand() * (SX - 10));
    const z = 5 + Math.floor(rand() * (SZ - 10));
    if (inPlot(plots, x, z, 7)) continue;
    const h = height[z * SX + x];
    if (h <= SEA_LEVEL) continue;
    const w = 2 + Math.floor(rand() * 3);
    for (let dx = -w; dx <= w; dx++) {
      for (let dz = -w; dz <= w; dz++) {
        const edge = Math.abs(dx) === w || Math.abs(dz) === w;
        for (let dy = 0; dy < 2 + Math.floor(rand() * 3); dy++) {
          if (!edge && rand() < 0.7) continue;
          const i = world.index(x + dx, h + 1 + dy, z + dz);
          if (world.blocks[i] !== B.AIR) continue;
          world.blocks[i] = rand() < 0.45 ? B.BRICK : B.MOSS;
        }
      }
    }
  }

  /* base markers ------------------------------------------------------ */
  for (let i = 0; i < plots.length; i++) {
    const p = plots[i];
    const h = height[p.z * SX + p.x];
    world.blocks[world.index(p.x, h + 1, p.z)] = B.BONE;
    // small starting plinth so the plot is obvious from a distance
    for (let d = 0; d < 4; d++) {
      const dx = d < 2 ? (d === 0 ? -2 : 2) : 0;
      const dz = d >= 2 ? (d === 2 ? -2 : 2) : 0;
      const hh = height[(p.z + dz) * SX + (p.x + dx)];
      if (world.get(p.x + dx, hh + 1, p.z + dz) === B.AIR) {
        world.blocks[world.index(p.x + dx, hh + 1, p.z + dz)] = B.PLANK;
      }
    }
  }

  world.plots = plots;
  world.height = height;
  world.rebuildCaches();
  // columns: surfaceY = top solid + 1, so terrain walk level is h+1
  for (let z = 0; z < SZ; z++) for (let x = 0; x < SX; x++) world.terrainY[z * SX + x] = Math.min(WORLD.SY - 1, height[z * SX + x] + 1);
  return world;
}

function inPlot(plots, x, z, r) {
  for (const p of plots) {
    if (Math.abs(p.x - x) <= r && Math.abs(p.z - z) <= r) return true;
  }
  return false;
}

export { fbm, value2, hash2 };

/**
 * Flat test/simple arena: a featureless grass plain at y = 12 with the 8 base
 * markers. Used by the headless tests and by "flat world" mode.
 */
export function generateFlat(world, seed = 7) {
  const { SX } = WORLD;
  const SZ = world.sz;
  const rand = rng(seed);
  const h = 12;
  for (let z = 0; z < world.sz; z++) {
    for (let x = 0; x < world.sx; x++) {
      for (let y = 0; y <= h; y++) {
        const id = y === 0 ? B.BEDROCK : y < h - 2 ? B.STONE : y < h ? B.DIRT : B.GRASS;
        world.blocks[world.index(x, y, z)] = id;
      }
    }
  }
  const plots = plotCenters();
  for (const p of plots) world.blocks[world.index(p.x, h + 1, p.z)] = B.BONE;
  // a few loose blocks in a ring so the bots have something to mine
  for (let i = 0; i < 220; i++) {
    const x = 4 + Math.floor(rand() * (SX - 8));
    const z = 4 + Math.floor(rand() * (SZ - 8));
    for (let dy = 0; dy < 1 + Math.floor(rand() * 2); dy++) {
      world.blocks[world.index(x, h + 1 + dy, z)] = rand() < 0.5 ? B.COBBLE : B.SAND;
    }
  }
  world.plots = plots;
  const height = new Int16Array(SX * world.sz).fill(h);
  world.height = height;
  world.rebuildCaches();
  for (let z = 0; z < world.sz; z++) for (let x = 0; x < SX; x++) world.terrainY[z * SX + x] = h + 1;
  return world;
}
