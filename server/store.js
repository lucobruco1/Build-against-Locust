/**
 * Disk persistence for the game server. No database, no dependencies: atomic
 * JSON files under `.data/`, which is exactly what this project needs — the
 * learned EfficientZero checkpoints, the run summary ("what did the bots learn
 * across sessions") and a tail of the event log.
 */

import { promises as fs } from 'node:fs';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';

export class Store {
  constructor(dir = '.data') {
    this.dir = dir;
    if (!existsSync(this.dir)) mkdirSync(this.dir, { recursive: true });
    this.dirty = new Map();
    this.timer = null;
  }

  file(name) {
    return path.join(this.dir, `${name.replace(/[^a-z0-9._-]/gi, '_')}.json`);
  }

  async writeJson(name, obj) {
    const p = this.file(name);
    const tmp = `${p}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(obj), 'utf8');
    await fs.rename(tmp, p);
    return p;
  }

  async readJson(name, fallback = null) {
    try {
      const raw = await fs.readFile(this.file(name), 'utf8');
      return JSON.parse(raw);
    } catch {
      return fallback;
    }
  }

  /** Queue a write and flush everything at most once every `ms`. */
  mark(name, obj, ms = 5000) {
    this.dirty.set(name, obj);
    if (this.timer) return;
    this.timer = setInterval(() => this.flush(), ms);
    this.timer.unref?.();
  }

  async flush() {
    const items = [...this.dirty];
    this.dirty.clear();
    for (const [name, obj] of items) {
      try { await this.writeJson(name, obj); } catch (err) { console.error('[store] write failed', name, err.message); }
    }
  }

  /* ------------------------------------------------------------- brains */

  async saveBrains(checkpoints, meta = {}) {
    return this.writeJson('brains', { savedAt: Date.now(), meta, checkpoints });
  }

  async loadBrains() {
    const data = await this.readJson('brains');
    if (!data || !data.checkpoints) return null;
    return data;
  }

  /* --------------------------------------------------------- kill/learn */

  /**
   * Long-lived summary across restarts. The numbers here are the whole point of
   * the project: do the bots actually get better at fortifying, and does the
   * Locust get better at finding them?
   */
  async recordCycle(summary) {
    const s = (await this.readJson('summary', emptySummary())) || emptySummary();
    s.cycles++;
    s.last = summary;
    push(s.history, summary, 60);
    s.totals.kills += summary.kills || 0;
    s.totals.deaths += summary.deaths || 0;
    s.totals.escapes += summary.escapes || 0;
    s.totals.blocksPlaced += summary.placed || 0;
    s.totals.blocksBroken += summary.broken || 0;
    s.totals.blocksSmashed += summary.smashed || 0;
    s.totals.secureSeconds += summary.secureSeconds || 0;
    s.peakWallCoverage = Math.max(s.peakWallCoverage || 0, summary.wallMean || 0);
    s.peakScore = Math.max(s.peakScore || 0, summary.bestScore || 0);
    s.optimizerSteps = (s.optimizerSteps || 0) + (summary.updates || 0);
    for (const b of summary.builders || []) {
      const e = s.bots[b.name] || { deaths: 0, escapes: 0, placed: 0, bestWall: 0, bestScore: 0, killed: 0 };
      e.deaths += b.deaths || 0;
      e.escapes += b.escaped || 0;
      e.placed += b.placed || 0;
      e.killed += b.killedByLocust || 0;
      e.bestWall = Math.max(e.bestWall, b.wall || 0);
      e.bestScore = Math.max(e.bestScore, b.score || 0);
      s.bots[b.name] = e;
    }
    await this.writeJson('summary', s);
    return s;
  }

  async summary() {
    return (await this.readJson('summary', emptySummary())) || emptySummary();
  }

  async close() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    await this.flush();
  }
}

function push(arr, v, cap) {
  arr.push(v);
  if (arr.length > cap) arr.shift();
  return arr;
}

function emptySummary() {
  return {
    cycles: 0,
    optimizerSteps: 0,
    peakWallCoverage: 0,
    peakScore: 0,
    totals: { kills: 0, deaths: 0, escapes: 0, blocksPlaced: 0, blocksBroken: 0, blocksSmashed: 0, secureSeconds: 0 },
    bots: {},
    history: [],
    last: null,
  };
}

export default Store;
