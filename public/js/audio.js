/**
 * Procedural ambience: the Locust's low-frequency hum + wet clicking and a bed
 * of electronic static (the sounds Doctor Nowhere's creature arrives with). All
 * synthesised — no audio files to ship, and it scales with distance for free.
 */
export class Ambience {
  constructor() {
    this.ctx = null;
    this.level = 0;
    this.enabled = true;
  }

  start() {
    if (this.ctx || !this.enabled) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.0001;
    this.master.connect(ctx.destination);

    // hum: two detuned low oscillators through a gentle lowpass
    this.humGain = ctx.createGain(); this.humGain.gain.value = 0.9;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 260;
    this.humGain.connect(lp); lp.connect(this.master);
    for (const [f, g] of [[41, 0.5], [55.5, 0.28], [82, 0.12]]) {
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f;
      const gg = ctx.createGain(); gg.gain.value = g;
      o.connect(gg); gg.connect(this.humGain); o.start();
    }

    // static: looping filtered noise, used for proximity and the grab moment
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * 0.6;
    this.noiseBuf = buf;
    this.noise = ctx.createBufferSource(); this.noise.buffer = buf; this.noise.loop = true;
    this.noiseFilter = ctx.createBiquadFilter(); this.noiseFilter.type = 'bandpass';
    this.noiseFilter.frequency.value = 900; this.noiseFilter.Q.value = 0.8;
    this.noiseGain = ctx.createGain(); this.noiseGain.gain.value = 0.0;
    this.noise.connect(this.noiseFilter); this.noiseFilter.connect(this.noiseGain);
    this.noiseGain.connect(this.master);
    this.noise.start();
    this.master.gain.setTargetAtTime(0.5, ctx.currentTime, 0.6);
  }

  /** dread 0..1 — how close/locked-on the Locust is */
  setDread(dread, hunting) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const d = Math.max(0, Math.min(1, dread));
    this.level = d;
    this.humGain.gain.setTargetAtTime(0.05 + d * 0.55, t, 0.25);
    this.noiseGain.gain.setTargetAtTime(0.005 + d * 0.05 + (hunting ? 0.02 : 0), t, 0.4);
    this.noiseFilter.frequency.setTargetAtTime(600 + d * 1800, t, 0.5);
    this.master.gain.setTargetAtTime(hunting ? 0.62 : 0.4, t, 0.8);
  }

  click(kind = 'wet') {
    if (!this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'square';
    o.frequency.setValueAtTime(kind === 'smash' ? 150 : 950, t);
    o.frequency.exponentialRampToValueAtTime(kind === 'smash' ? 40 : 220, t + 0.09);
    g.gain.setValueAtTime(kind === 'smash' ? 0.3 : 0.12, t);
    g.gain.exponentialRampToValueAtTime(0.0008, t + (kind === 'smash' ? 0.22 : 0.09));
    o.connect(g); g.connect(this.master);
    o.start(t); o.stop(t + 0.3);
  }

  stab() {
    if (!this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const src = ctx.createBufferSource(); src.buffer = this.noiseBuf;
    const bp = ctx.createBiquadFilter(); bp.type = 'highpass'; bp.frequency.value = 1400;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.001, t);
    g.gain.linearRampToValueAtTime(0.5, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
    src.connect(bp); bp.connect(g); g.connect(this.master);
    src.start(t); src.stop(t + 0.45);
  }

  toggle(on) {
    this.enabled = on;
    if (this.master) this.master.gain.value = on ? 0.45 : 0.0001;
  }
}
