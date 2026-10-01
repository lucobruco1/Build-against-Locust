/**
 * DOM HUD. Owns every overlay element and knows nothing about networking: the
 * caller feeds it the same state object the server sent (`encodeState` output),
 * plus a couple of locally-derived extras (`localId`, `fps`, `ping`).
 */

import { BLOCK_DEFS, B, HOTBAR, ENTITY } from '../../shared/rules.js';
import { BOT_COLORS, hex } from './palette.js';

const $ = (id) => document.getElementById(id);

export class Hud {
  constructor(opts = {}) {
    this.opts = opts;
    this.el = {
      phaseName: $('phase-name'), phaseCycle: $('phase-cycle'), timerFill: $('timer-fill'),
      timerText: $('timer-text'), locustState: $('locust-state'),
      baseList: $('base-list'), meanWall: $('mean-wall'), sealedCount: $('sealed-count'),
      brainList: $('brain-list'), aiSummary: $('ai-summary'), assistNote: $('assist-note'),
      loss: $('loss-canvas'), health: $('health'), status: $('status-line'), hotbar: $('hotbar'),
      crosshair: $('crosshair'), log: $('log'), dread: $('dread'), hurt: $('hurt-flash'),
      veil: $('grabbed-veil'), banner: $('banner'), bannerTitle: $('banner-title'), bannerSub: $('banner-sub'),
      menu: $('menu'), help: $('help'), death: $('death'), deathTimer: $('death-timer'),
      tally: $('tally'), tallyBody: $('tally-body'),
    };
    this.loss = this.el.loss ? this.el.loss.getContext('2d') : null;
    this.history = new Map();     // brainId -> [{loss, policy, value}]
    this.lastHotbar = '';
    this.lastLogLen = 0;
    this.dread = 0;
    this.fps = 0;
    this.ping = 0;
    this.aiOpen = true;
    this.el.btnAiToggle = $('btn-ai-toggle');
    this.el.btnAiToggle?.addEventListener('click', () => {
      this.aiOpen = !this.aiOpen;
      this.el.aiPanel = document.querySelector('.ai-panel');
      this.el.aiPanel?.classList.toggle('collapsed', !this.aiOpen);
      this.el.btnAiToggle.textContent = this.aiOpen ? 'hide' : 'show';
    });
  }

  /* ------------------------------------------------------------- updates */

  update(state, extras = {}) {
    this.state = state;
    this.extras = extras;
    const secs = Math.max(0, state.timeLeft || 0);
    const mm = Math.floor(secs / 60), ss = Math.floor(secs % 60);
    if (this.el.phaseName) {
      this.el.phaseName.textContent = state.phase === 'hunt' ? 'The hunt' : state.phase === 'build' ? 'Build' : state.phase === 'revive' ? 'Aftermath' : 'Lobby';
      this.el.phaseName.className = `phase-name ${state.phase}`;
    }
    if (this.el.phaseCycle) this.el.phaseCycle.textContent = `cycle ${state.cycle} · t${state.tick}`;
    if (this.el.timerText) this.el.timerText.textContent = `${mm}:${String(ss).padStart(2, '0')}`;
    if (this.el.timerFill) {
      const frac = state.timeTotal ? 1 - Math.min(1, secs / state.timeTotal) : 0;
      this.el.timerFill.style.width = `${(frac * 100).toFixed(1)}%`;
      this.el.timerFill.className = `timer-fill ${state.phase}`;
    }

    const l = state.locust;
    if (this.el.locustState) {
      this.el.locustState.textContent = l
        ? `locust · ${l.kills} down · ${l.smashed} broken${l.holding ? ` · holding ${l.holding}` : ''}`
        : 'no locust';
      this.el.locustState.classList.toggle('hot', !!l);
    }

    this.bases(state, extras);
    this.vitals(state, extras);
    this.hotbar(state, extras);
    this.brains(state);
    this.logs(state);
    this.grabbedVeil(state, extras);
  }

  bases(state, extras) {
    const ul = this.el.baseList;
    if (!ul) return;
    const me = extras.localId;
    let sum = 0, sealed = 0;
    const rows = state.actors.map((b) => {
      sum += b.wall || 0;
      if (b.sealed) sealed++;
      const dot = !b.alive ? 'down' : b.grabbed ? 'grab' : 'live';
      return `<li>
        <span><i class="dot ${dot}" style="background:${hex(b.name)}"></i>${b.name === 'You' ? 'you' : b.name.slice(0, 7)}</span>
        <span class="bar" title="wall ${(b.wall * 100) | 0}% · roof ${(b.roof * 100) | 0}%">
          <i style="width:${((b.wall || 0) * 100).toFixed(0)}%"></i>
          <i class="roof" style="width:${((b.roof || 0) * 50).toFixed(0)}%"></i>
        </span>
        <span>${b.alive ? `${((b.wall || 0) * 100) | 0}%` : 'down'}</span>
      </li>`;
    });
    ul.innerHTML = rows.join('');
    if (this.el.meanWall) this.el.meanWall.textContent = `${Math.round((sum / (state.actors.length || 1)) * 100)}%`;
    if (this.el.sealedCount) this.el.sealedCount.textContent = `${sealed}/${state.actors.length}`;
    void me;
  }

  vitals(state, extras) {
    const me = state.actors.find((a) => a.id === extras.localId) || state.actors[0];
    if (!me) return;
    const max = ENTITY.PLAYER.height > 0 ? 20 : 20;
    const hp = Math.max(0, Math.min(max, me.hp ?? max));
    const half = Math.round(hp / 2 * 2);
    if (this.el.health) {
      const hearts = [];
      for (let i = 0; i < 10; i++) {
        const fill = Math.max(0, Math.min(1, half - i * 2));   // 0, .5 or 1
        hearts.push(heart(fill));
      }
      const html = hearts.join('');
      if (this.el.health.innerHTML !== html) this.el.health.innerHTML = html;
    }
    let msg = 'stand still. listen.';
    let cls = 'status-line';
    if (!me.alive) { msg = 'you are down. the cycle will bring you back.'; cls += ' bad'; }
    else if (me.grabbed) { msg = 'GRABBED — hold SPACE to struggle!'; cls += ' bad'; }
    else if (state.phase === 'hunt') {
      const d = state.locust ? Math.hypot(state.locust.x - me.x, state.locust.z - me.z) : Infinity;
      if (d < 12) { msg = `it is ${d.toFixed(0)} blocks away. do not breathe.`; cls += ' bad'; }
      else if (d < 26) { msg = 'something is walking between the trees.'; cls += ' warn'; }
      else if (me.seen) { msg = 'it has looked at you before. it remembers.'; cls += ' warn'; }
      else msg = 'hidden. hold.';
    } else if (state.phase === 'build') {
      msg = me.sealed ? `sealed · wall ${((me.wall || 0) * 100) | 0}% · roof ${((me.roof || 0) * 100) | 0}%`
        : `build. walls before decoration — ${(100 - ((me.wall || 0) * 100)) | 0}% of the shell is still open`;
      if (!me.sealed) cls += ' warn';
    }
    if (this.extras.fps) msg += `  ·  ${this.extras.fps} fps · ${this.extras.ping || 0} ms`;
    if (this.el.status) { this.el.status.textContent = msg; this.el.status.className = cls; }
  }

  hotbar(state, extras) {
    const me = state.actors.find((a) => a.id === extras.localId) || state.actors[0];
    if (!this.el.hotbar || !me) return;
    const inv = me.inv || {};
    const sel = me.sel ?? 0;
    const sig = HOTBAR.map((id) => `${id}:${inv[id] ?? 0}`).join(',') + `|${sel}`;
    if (sig === this.lastHotbar) return;
    this.lastHotbar = sig;
    this.el.hotbar.innerHTML = HOTBAR.map((id, i) => {
      const n = inv[id] ?? 0;
      const d = BLOCK_DEFS[id];
      return `<div class="slot ${i === sel ? 'sel' : ''} ${n ? '' : 'empty'}" title="${d?.name ?? id}">
        <span class="k">${i + 1}</span>
        <span class="swatch" style="background:#${(d?.color ?? 0x888888).toString(16).padStart(6, '0')}"></span>
        <span class="n">${n || ''}</span>
      </div>`;
    }).join('');
  }

  brains(state) {
    const list = this.el.brainList;
    if (!list) return;
    const bs = state.brains || [];
    if (this.el.aiSummary) {
      const up = bs.reduce((a, b) => a + (b.updates || 0), 0);
      const params = bs[0]?.params || 0;
      this.el.aiSummary.textContent = `${bs.length} nets · ${(params / 1000).toFixed(0)}k params · ${up} updates`;
    }
    list.innerHTML = bs.map((b) => {
      if (b.loss != null) {
        let h = this.history.get(b.id) || [];
        const lastPt = h[h.length - 1];
        if (!lastPt || lastPt.upd !== b.updates) {
          h.push({ upd: b.updates, loss: b.loss, policy: b.policy, value: b.value });
          if (h.length > 120) h = h.slice(-120);
          this.history.set(b.id, h);
        }
      }
      const who = b.kind === 'locust' ? 'LOCUST' : (b.name || b.id).slice(0, 7);
      const col = b.kind === 'locust' ? '#ff8f9c' : hex(b.name);
      return `<li>
        <span style="color:${col}">${who}</span>
        <span class="bar" title="prior-driven decisions ${Math.round((b.priorShare ?? 0) * 100)}%">
          <i style="width:${Math.min(100, (b.priorShare ?? 0) * 100).toFixed(0)}%;background:${col};opacity:.75"></i>
        </span>
        <span class="loss">${b.loss == null ? '—' : b.loss.toFixed(1)}</span>
        <span class="sub">upd ${b.updates} · buf ${b.bufferGames}g/${b.bufferSteps}t · π ${b.policy ?? '—'} · v ${b.value ?? '—'} · sims ${b.sims} · ${b.searchMs}ms</span>
      </li>`;
    }).join('');
    if (this.el.assistNote) {
      const a = bs[0]?.assist;
      this.el.assistNote.textContent = a == null ? '' : `currently ${(a * 100) | 0}% of decisions come from the expert`;
    }
    this.drawLoss(bs);
  }

  drawLoss(bs) {
    const g = this.loss;
    if (!g) return;
    const w = g.canvas.width, h = g.canvas.height;
    g.clearRect(0, 0, w, h);
    let max = 1;
    for (const b of bs) for (const p of (this.history.get(b.id) || [])) max = Math.max(max, p.loss);
    g.globalAlpha = 0.9;
    for (const [i, b] of bs.entries()) {
      const hist = this.history.get(b.id) || [];
      if (hist.length < 2) continue;
      const col = b.kind === 'locust' ? '#ff8f9c' : '#ffb03a';
      g.strokeStyle = col;
      g.lineWidth = 1.2 + (i === 0 ? 0.6 : 0);
      g.beginPath();
      hist.forEach((p, k) => {
        const x = (k / Math.max(1, hist.length - 1)) * (w - 4) + 2;
        const y = h - 4 - (p.loss / max) * (h - 10);
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
      g.strokeStyle = '#63b3ff'; g.globalAlpha = 0.45; g.beginPath();
      hist.forEach((p, k) => {
        const x = (k / Math.max(1, hist.length - 1)) * (w - 4) + 2;
        const y = h - 4 - ((p.policy ?? 0) / max) * (h - 10);
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
      g.globalAlpha = 0.45; g.strokeStyle = '#ff8fb1'; g.beginPath();
      hist.forEach((p, k) => {
        const x = (k / Math.max(1, hist.length - 1)) * (w - 4) + 2;
        const y = h - 4 - ((p.value ?? 0) / max) * (h - 10);
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
      g.globalAlpha = 0.9;
    }
    g.fillStyle = 'rgba(255,255,255,.28)';
    g.font = '9px ui-monospace, monospace';
    g.fillText(`max ${max.toFixed(1)}`, 4, 10);
  }

  logs(state) {
    if (!this.el.log) return;
    const log = state.log || [];
    if (log.length === this.lastLogLen) return;
    this.lastLogLen = log.length;
    this.el.log.innerHTML = log.slice(-14).map((e, i, arr) => {
      const fade = Math.min(2, Math.floor((arr.length - i) / 5));
      return `<div class="${e.who} f${fade}">${escapeHtml(e.msg)}</div>`;
    }).join('');
  }

  grabbedVeil(state, extras) {
    const me = state.actors.find((a) => a.id === extras.localId);
    this.el.veil?.classList.toggle('on', !!(me && me.grabbed));
  }

  setDread(v) {
    if (!this.el.dread) return;
    const t = Math.max(0, Math.min(1, v));
    if (Math.abs(t - this.dread) < 0.02) return;
    this.dread = t;
    this.el.dread.style.opacity = (t * 0.9).toFixed(2);
  }

  hurt() {
    const el = this.el.hurt;
    if (!el) return;
    el.classList.add('on');
    setTimeout(() => el.classList.remove('on'), 90);
  }

  banner(title, sub = '', kind = '') {
    const b = this.el.banner;
    if (!b) return;
    this.el.bannerTitle.textContent = title;
    this.el.bannerSub.textContent = sub;
    b.className = `banner ${kind}`;
    clearTimeout(this._bannerT);
    this._bannerT = setTimeout(() => b.classList.add('hidden'), 3200);
  }

  showDeath(show, secs = 0) {
    this.el.death?.classList.toggle('hidden', !show);
    if (show && this.el.deathTimer) this.el.deathTimer.textContent = `${Math.ceil(secs)}s`;
  }

  tickDeathTimer(secs) {
    if (this.el.deathTimer) this.el.deathTimer.textContent = `${Math.max(0, Math.ceil(secs))}s`;
  }

  showTally(tally, locust) {
    if (!this.el.tally || !tally) return;
    const last = Array.isArray(tally) ? tally[tally.length - 1] : tally;
    this.el.tallyBody.innerHTML = tallyHtml(last, locust);
    this.el.tally.classList.remove('hidden');
  }

  hideTally() { this.el.tally?.classList.add('hidden'); }
  hideMenu() { this.el.menu?.classList.add('hidden'); }
  showMenu(msg) {
    this.el.menu?.classList.remove('hidden');
    if (msg && $('menu-status')) $('menu-status').textContent = msg;
  }
  toggleHelp(force) { this.el.help?.classList.toggle('hidden', force === false); }
  setCrosshairActive(on) { this.el.crosshair?.classList.toggle('act', !!on); }
}

function heart(fill) {
  const c = fill >= 1 ? '#ff4d5e' : fill > 0 ? 'url(#half)' : 'rgba(255,255,255,.16)';
  const shape = 'M8 14.5S1.6 10.3 1.6 6.2A3.7 3.7 0 0 1 8 4a3.7 3.7 0 0 1 6.4 2.2c0 4.1-6.4 8.3-6.4 8.3z';
  return `<svg class="heart" viewBox="0 0 16 16"><defs><linearGradient id="half"><stop offset="50%" stop-color="#ff4d5e"/><stop offset="50%" stop-color="rgba(255,255,255,.14)"/></linearGradient></defs><path d="${shape}" fill="${fill >= 1 ? '#ff4d5e' : fill > 0 ? 'url(#half)' : 'rgba(255,255,255,.14)'}" stroke="rgba(0,0,0,.5)"/></svg>`;
}

function tallyHtml(t, locust) {
  if (!t) return '';
  const head = `<div class="tally-row head"><span>survivor</span><span>score</span><span>wall</span><span>deaths</span><span>placed</span></div>`;
  const rows = (t.rows || t.builders || []).map((b) =>
    `<div class="tally-row"><span>${b.name}${b.alive ? '' : ' ✝'}</span><b>${b.score}</b><span>${((b.wall ?? 0) * 100) | 0}%</span><span>${b.deaths ?? 0}</span><span>${b.placed ?? 0}</span></div>`).join('');
  const l = locust || {};
  const meta = `<div class="tally-row"><span>the locust</span><b>${l.kills ?? 0} down</b><span>${l.smashed ?? 0} blocks smashed</span><span>${l.grabs ?? 0} grabs</span><span>cycle ${t.cycle ?? ''}</span></div>`;
  return `<div class="tally-row head"><span>cycle ${t.cycle ?? '—'} report</span></div>` + head + rows + meta;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

export { $, BOT_COLORS };
