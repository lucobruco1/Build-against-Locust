/**
 * core/inventory.js – the tiny creative-lite block bag every builder carries.
 *
 * The player starts each cycle with a stack of each placeable block so the
 * 90-second build window is about fortifying, not mining. Bots are hungrier:
 * they must harvest terrain (that is what their MINE reward pays for) which
 * makes "dig a trench, use the dirt for the wall" a strategy the value head can
 * actually discover.
 */

import { B, HOTBAR, BOT_PLACEABLE, def } from '../shared/rules.js';

export class Inventory {
  constructor(o = {}) {
    this.counts = new Map();
    this.capacity = o.capacity ?? 999;
    this.hotbar = o.hotbar || HOTBAR.slice();
    this.slot = o.slot ?? 0;
    if (o.start) for (const [id, n] of Object.entries(o.start)) this.counts.set(+id, n);
  }
  count(id) {
    return this.counts.get(id) || 0;
  }
  get total() {
    let s = 0;
    for (const v of this.counts.values()) s += v;
    return s;
  }
  add(id, n = 1) {
    if (id === B.AIR) return 0;
    if (n <= 0) return 0;
    const cur = this.count(id);
    const next = Math.min(this.capacity, cur + n);
    this.counts.set(id, next);
    return next - cur;
  }
  take(id, n = 1) {
    const cur = this.count(id);
    if (cur < n) return false;
    if (cur - n <= 0) this.counts.delete(id);
    else this.counts.set(id, cur - n);
    return true;
  }
  has(id) {
    return this.count(id) > 0;
  }
  get selected() {
    return this.hotbar[Math.min(this.slot, this.hotbar.length - 1)];
  }
  selectSlot(i) {
    this.slot = ((i % this.hotbar.length) + this.hotbar.length) % this.hotbar.length;
  }
  cycle(dir = 1) {
    this.selectSlot(this.slot + dir);
  }
  /** first slot (from `slot`) that still holds blocks – used by the bots */
  bestAvailable(prefer = BOT_PLACEABLE) {
    let fallback = -1;
    for (const [id, n] of this.counts) {
      if (n <= 0) continue;
      if (fallback < 0) fallback = id;
      if (prefer.includes(id)) return id;
    }
    return fallback;
  }
  /** the block a bot should place, favouring hard blocks for walls */
  pickForBuild() {
    const rank = [B.BRICK, B.STONE, B.COBBLE, B.PLANK, B.MOSS, B.BONE, B.LOG, B.GRASS, B.DIRT, B.SAND, B.LEAF, B.GLASS];
    for (const id of rank) if (this.count(id) > 0) return id;
    return this.bestAvailable();
  }
  snapshot() {
    const out = {};
    for (const [id, n] of this.counts) out[def(id).name] = n;
    return { counts: out, total: this.total, selected: this.selected, slot: this.slot, hotbar: this.hotbar };
  }
  asScalarCounts() {
    const arr = [];
    for (const id of this.hotbar) arr.push(this.count(id));
    return arr;
  }
}

export function playerInventory(stacks = 96) {
  const start = {};
  for (const id of HOTBAR) start[id] = stacks;
  return new Inventory({ start, hotbar: HOTBAR.slice() });
}

export function botInventory(stacks = 10) {
  const start = {};
  for (const id of [B.PLANK, B.COBBLE, B.DIRT, B.SAND, B.LOG]) start[id] = stacks;
  return new Inventory({
    start,
    hotbar: [B.PLANK, B.COBBLE, B.DIRT, B.SAND, B.LOG, B.GRASS, B.STONE, B.BRICK, B.LEAF],
  });
}
