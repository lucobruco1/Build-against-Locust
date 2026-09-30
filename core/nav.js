import { WORLD, B, isSolid, isOpaque, clamp, BLOCK_DEFS, BASE } from '../shared/rules.js';
/**
 * core/nav.js
 * ---------------------------------------------------------------------------
 * Voxel A* for the Locust. The Locust is 3.9 blocks tall and does not fit
 * through 1-block gaps, so passability is clearance-based. The search treats
 * walls as *traversable at a price* (`smashCost`) which is exactly how the
 * creature plays: it walks to the weakest point of a base and then smashes
 * through. The returned path is not an order – it only feeds the Locust's
 * observation vector (and an optional action prior), the EfficientZero network
 * still picks every action.
 * ---------------------------------------------------------------------------
 */


class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(node, prio) {
    const a = this.a;
    a.push({ node, prio });
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].prio <= a[i].prio) break;
      const t = a[p]; a[p] = a[i]; a[i] = t;
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1, r = l + 1;
        let s = i;
        if (l < a.length && a[l].prio < a[s].prio) s = l;
        if (r < a.length && a[r].prio < a[s].prio) s = r;
        if (s === i) break;
        const t = a[s]; a[s] = a[i]; a[i] = t;
        i = s;
      }
    }
    return top ? top.node : null;
  }
}

const key = (x, y, z) => (y * WORLD.SZ + z) * WORLD.SX + x;

/**
 * @param {object} world   VoxelWorld
 * @param {{x,y,z}} start  block-cell feet position
 * @param {{x,y,z}} goal
 * @param {object} o       {height, clearance, smashCost, maxNodes}
 */
export function astar(world, start, goal, o = {}) {
  const height = o.height ?? 4;      // vertical air clearance required
  const smashCost = o.smashCost ?? 3.2;
  const maxNodes = o.maxNodes ?? 2600;
  const sx = Math.floor(start.x), sy = Math.floor(start.y), sz = Math.floor(start.z);
  const gx = Math.floor(goal.x), gy = Math.floor(goal.y), gz = Math.floor(goal.z);

  const g = new Map();
  const came = new Map();
  const closed = new Set();
  const open = new Heap();
  const h = (x, y, z) => {
    const d = Math.max(Math.abs(x - gx), Math.abs(z - gz));
    return d + Math.abs(y - gy) * 0.6 + 0.02 * Math.hypot(x - gx, z - gz);
  };
  g.set(key(sx, sy, sz), 0);
  open.push({ x: sx, y: sy, z: sz, blocked: false }, h(sx, sy, sz));

  const dirs = [
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [1, -1], [-1, 1], [-1, -1],
  ];
  let expansions = 0;
  let best = { x: sx, y: sy, z: sz };
  let bestH = h(sx, sy, sz);

  while (open.size && expansions < maxNodes) {
    const cur = open.pop();
    if (!cur) break;
    const ck = key(cur.x, cur.y, cur.z);
    if (closed.has(ck)) continue;
    closed.add(ck);
    expansions++;
    const dcur = Math.max(Math.abs(cur.x - gx), Math.abs(cur.z - gz));
    if (dcur < bestH) { bestH = dcur; best = cur; }
    if (dcur <= 1 && Math.abs(cur.y - gy) <= 2) {
      return rebuild(came, cur, world, height, expansions, true);
    }
    const cg = g.get(ck) ?? 0;
    for (const [dx, dz] of dirs) {
      for (const dy of [0, 1, -1, -2]) {
        const nx = cur.x + dx, ny = cur.y + dy, nz = cur.z + dz;
        if (nx < 1 || nz < 1 || ny < 1 || nx >= WORLD.SX - 1 || nz >= WORLD.SZ - 1 || ny >= WORLD.SY - height) continue;
        // no digging: a node must sit at or above the column's *natural*
        // terrain, else the search prefers tunnelling through the floor instead
        // of opening the wall it is supposed to attack
        if (ny < world.terrainSurface(nx, nz)) continue;
        const nk = key(nx, ny, nz);
        if (closed.has(nk)) continue;
        const clearance = clearanceOk(world, nx, ny, nz, height);
        const support = isSolid(world.get(nx, ny - 1, nz));
        let moveCost = Math.abs(dx) + Math.abs(dz) > 1 ? 1.41 : 1;
        let blocked = false;
        if (!clearance) {
          // must smash something: scan the *whole* clearance column (a roof
          // block four up is just as much of an obstacle as a wall block one up)
          const open = findBreakableCell(world, nx, ny, nz, height);
          if (!open) continue;
          moveCost += smashCost * hardness(world.get(open.x, open.y, open.z));
          blocked = true;
        } else if (!support) {
          if (dy < 0) {
            if (dy < -3) continue;                 // it can fall 3, not into the void
            moveCost += 0.4;                       // falling is fine, it is a big insect
          } else {
            // A leap needs a take-off surface AND a landing surface, and gains
            // at most ~2 blocks. Without this the search happily stairs its way
            // up into the sky on nothing.
            if (dy > 2) continue;
            if (!isSolid(world.get(cur.x, cur.y - 1, cur.z))) continue;
            if (!isSolid(world.get(nx, ny - 1, nz))) continue; // must land on something
            moveCost += 1.9;
          }
        } else if (dy > 0) {
          // stepping up onto a block: needs headroom over the ledge
          if (!clearanceOk(world, nx, ny, nz, height)) {
            const open = findBreakableCell(world, nx, ny, nz, height);
            if (!open) continue;
            moveCost += smashCost * blockHardness(world.get(open.x, open.y, open.z));
            blocked = true;
          }
        }
        const ng = cg + moveCost + (blocked ? 0.15 : 0);
        const prev = g.get(nk);
        if (prev === undefined || ng < prev) {
          g.set(nk, ng);
          came.set(nk, ck);
          open.push({ x: nx, y: ny, z: nz, blocked }, ng + h(nx, ny, nz));
        }
      }
    }
  }
  return rebuild(came, best, world, height, expansions, false);
}

function rebuild(came, node, world, height, expansions, reached) {
  const path = [];
  let cur = node;
  let guard = 0;
  while (cur && guard++ < 4096) {
    path.push({ x: cur.x, y: cur.y, z: cur.z, blocked: !!cur.blocked });
    const k = key(cur.x, cur.y, cur.z);
    const pk = came.get(k);
    if (pk === undefined) break;
    cur = unkey(pk);
  }
  path.reverse();
  // The search is allowed to walk *through* solid cells at a price, so the
  // first such cell is the wall it should open. `smashAt` is its index in the
  // path, which tells the controller how far away the breach is.
  let smash = null, smashAt = -1;
  for (let i = 1; i < path.length; i++) {
    const p = path[i];
    const clear = clearanceOk(world, p.x, p.y, p.z, height);
    p.blocked = !clear;                 // the renderer/HUD highlight these cells
    if (!clear && !smash) {
      const found = findBreakableCell(world, p.x, p.y, p.z, height);
      if (found) { smash = found; smashAt = i; }
    }
  }
  return {
    path,
    smash,
    smashAt,
    blocked: !!smash,
    expansions,
    reached: !!reached,
  };
}

function unkey(k) {
  const x = k % WORLD.SX;
  const rest = (k - x) / WORLD.SX;
  const z = rest % WORLD.SZ;
  const y = (rest - z) / WORLD.SZ;
  return { x, y, z };
}

function clearanceOk(world, x, y, z, height) {
  for (let dy = 0; dy < height; dy++) {
    const id = world.getSoft(x, y + dy, z);
    if (isSolid(id)) return false;
  }
  return true;
}

function blockHardness(id) {
  // bedrock / water are unbreakable for the Locust
  return (id === B.BEDROCK || id === B.WATER) ? Infinity : (BLOCK_DEFS[id]?.locustHP ?? 2);
}
/** The lowest breakable block inside a column that has to be opened. */
export function findBreakableCell(world, x, y, z, height) {
  for (let dy = 0; dy < height + 1; dy++) {
    const id = world.getSoft(x, y + dy, z);
    if (isSolid(id) && Number.isFinite(blockHardness(id))) return { x, y: y + dy, z, id };
  }
  return null;
}

/**
 * The cheapest cell of a base's wall shell to open, seen from `from`:
 * fewest solid neighbours (thin / already-damaged spot) and closest to us.
 * The Locust paths at this cell and smashes it — "navigate the area, break
 * blocks by attacking them" as an actual strategy rather than a random walk.
 */
export function weakestBreach(world, plot, groundY, from, ring) {
  const r = ring ?? BASE.WALL_RING;
  let best = null, bestScore = Infinity;
  for (let dx = -r; dx <= r; dx++) {
    for (let dz = -r; dz <= r; dz++) {
      if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
      for (let dy = 1; dy < Math.min(3, BASE.WALL_TOP); dy++) {
        const x = plot.x + dx, y = (groundY ?? 0) + dy, z = plot.z + dz;
        if (!world.inside(x, y, z)) continue;
        const near = Math.hypot(x + 0.5 - from.x, z + 0.5 - from.z);
        if (near > 26) continue;
        const solid = world.solidNeighbours(x, y, z);
        const blocked = isSolid(world.get(x, y, z)) ? 1 : 0;
        const hp = blocked ? hardness(world.get(x, y, z)) : 0;
        const score = near * 0.7 + solid * 2.4 + hp * 1.6 - (1 - blocked) * 3;
        if (score < bestScore) {
          bestScore = score;
          best = { x, y, z, blocked: !!blocked, score };
        }
      }
    }
  }
  return best;
}
function hardness(id) {
  const d = BLOCK_DEFS[id];
  if (!d) return 1;
  return Number.isFinite(d.locustHP) ? d.locustHP : 99;
}

/**
 * Steering direction toward the next useful path node.
 * `stopAt` (the index of the first blocked node, i.e. the wall to open) makes
 * the walker stop *in front of* the wall instead of pressing into it, which is
 * what a smash needs. Returns the distance to that node so callers can tell
 * "close enough to attack".
 */
export function pathDirection(path, from, o = {}) {
  if (!path || path.length < 2) return { x: 0, z: 0, yaw: null, dx: 0, dz: 0, dy: 0, dist: 0, node: null };
  let i = 1;
  if (o.stopAt > 1) i = Math.min(path.length - 1, o.stopAt - 1);
  const n = path[i];
  const dx = n.x + 0.5 - from.x;
  const dz = n.z + 0.5 - from.z;
  const len = Math.hypot(dx, dz) || 1;
  return {
    x: dx / len, z: dz / len, dx, dz,
    dy: n.y - Math.floor(from.y),
    yaw: Math.atan2(-dx, -dz),
    dist: len,
    node: n,
  };
}

/** Greedy fallback used by the "assist" prior: turn toward a world point. */
export function yawTowards(from, to) {
  const dx = to.x - from.x, dz = to.z - from.z;
  return Math.atan2(-dx, -dz);
}
export function angleDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return clamp(d, -Math.PI, Math.PI);
}

/** Count of prey visible from a cell – used for the Locust's hunt pressure. */
export function visibleEntities(world, eye, entities, maxDist = 26) {
  let n = 0, best = null, bestD = Infinity;
  for (const e of entities) {
    if (!e.alive) continue;
    const t = { x: e.pos.x, y: e.pos.y + 1.2, z: e.pos.z };
    const d = Math.hypot(t.x - eye.x, t.y - eye.y, t.z - eye.z);
    if (d > maxDist) continue;
    n++;
    if (d < bestD) { bestD = d; best = e; }
  }
  return { count: n, nearest: best, nearestDist: best === null ? Infinity : bestD };
}
