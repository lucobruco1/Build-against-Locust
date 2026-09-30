/**
 * Actor rendering: the blocky survivors and the Locust.
 *
 * Both are built from plain primitives (no model files, no loaders) so the page
 * boots instantly and works offline. The Locust follows Doctor Nowhere's design:
 * ~13 ft of matte black, impossibly thin, stick arms that end in long tendrils,
 * a distorted human face with hollow eyes and a triangular jaw, tube-like
 * appendages around the head and neck, long dark hair, and pale-pink metal
 * plating on the chest/limbs (the Smilehood variant).
 */

import * as THREE from 'three';
import { BOT_COLORS } from '../palette.js';

const SKIN = 0xd9a066;

export function makeLabel(text, sub = '') {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 96;
  const g = c.getContext('2d');
  drawLabel(g, text, sub);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sp = new THREE.Sprite(mat);
  sp.scale.set(2.4, 0.9, 1);
  sp.userData = { canvas: c, tex, text, sub };
  return sp;
}

function drawLabel(g, text, sub) {
  g.clearRect(0, 0, 256, 96);
  g.fillStyle = 'rgba(6,8,10,0.62)';
  roundRect(g, 8, 8, 240, 80, 14);
  g.fill();
  g.font = '600 30px ui-sans-serif, system-ui, sans-serif';
  g.textAlign = 'center';
  g.fillStyle = '#f3f6f8';
  g.fillText(text, 128, 44);
  if (sub) {
    g.font = '500 20px ui-monospace, monospace';
    g.fillStyle = '#9fb4c0';
    g.fillText(sub, 128, 74);
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

export function setLabel(sprite, text, sub) {
  const u = sprite.userData;
  if (u.text === text && u.sub === sub) return;
  u.text = text; u.sub = sub;
  drawLabel(u.canvas.getContext('2d'), text, sub);
  u.tex.needsUpdate = true;
}

class Limbs {
  constructor(mat, parts) {
    this.mat = mat;
    this.parts = parts;
  }
}

/** A survivor: head, torso, two arms, two legs, held-block indicator. */
export class BuilderMesh {
  constructor(info) {
    const color = BOT_COLORS[info.name] ?? 0x9aa7b1;
    const body = new THREE.MeshLambertMaterial({ color });
    const skin = new THREE.MeshLambertMaterial({ color: SKIN });
    const g = new THREE.Group();
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.66, 0.3), body);
    torso.position.y = 1.13;
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.44, 0.44), skin);
    head.position.y = 1.68;
    const hair = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.12, 0.46), new THREE.MeshLambertMaterial({ color: 0x2b2118 }));
    hair.position.y = 1.86;
    const armGeo = new THREE.BoxGeometry(0.16, 0.6, 0.16);
    const legGeo = new THREE.BoxGeometry(0.19, 0.8, 0.2);
    const armL = new THREE.Mesh(armGeo, body); armL.position.set(-0.36, 1.1, 0);
    const armR = new THREE.Mesh(armGeo, body); armR.position.set(0.36, 1.1, 0);
    const legL = new THREE.Mesh(legGeo, new THREE.MeshLambertMaterial({ color: 0x35404d })); legL.position.set(-0.14, 0.4, 0);
    const legR = new THREE.Mesh(legGeo, new THREE.MeshLambertMaterial({ color: 0x35404d })); legR.position.set(0.14, 0.4, 0);
    for (const m of [armL, armR]) { m.geometry.translate(0, -0.3, 0); m.position.y = 1.4; }
    for (const m of [legL, legR]) { m.geometry.translate(0, -0.4, 0); m.position.y = 0.8; }
    const held = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.22, 0.22), new THREE.MeshLambertMaterial({ color: 0xffffff }));
    held.position.set(0.42, 1.0, 0.28);
    held.visible = false;
    g.add(torso, head, hair, armL, armR, legL, legR, held);
    g.traverse((o) => { if (o.isMesh) { o.castShadow = false; o.receiveShadow = false; } });
    const label = makeLabel(info.name, '');
    label.position.y = 2.25;
    g.add(label);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.42, 0.52, 24),
      new THREE.MeshBasicMaterial({ color: 0x7fe0a8, transparent: true, opacity: 0.5, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.03;
    g.add(ring);

    this.group = g;
    this.head = head; this.torso = torso; this.armL = armL; this.armR = armR;
    this.legL = legL; this.legR = legR; this.label = label; this.ring = ring; this.held = held;
    this.phase = Math.random() * 6;
    this.info = info;
  }

  setHeldBlock(color) {
    if (color == null) { this.held.visible = false; return; }
    this.held.visible = true;
    this.held.material.color.setHex(color);
  }

  update(st, dt, t) {
    const g = this.group;
    g.position.set(st.x, st.y, st.z);
    g.rotation.y = st.yaw ?? 0;
    const speed = this.speed ?? 0;
    const swing = Math.sin(t * 9 + this.phase) * Math.min(1, speed * 1.6) * 0.7;
    this.legL.rotation.x = swing;
    this.legR.rotation.x = -swing;
    const act = st.action ?? 0;
    this.armR.rotation.x = -0.35 - Math.abs(Math.sin(t * 12)) * (st.busy ? 0.9 : 0.1);
    this.armL.rotation.x = -swing * 0.6;
    this.head.rotation.x = clamp(st.pitch ?? 0, -0.9, 0.9);
    const alive = st.alive !== false;
    g.visible = alive || st.deathFade > 0;
    this.torso.visible = alive;
    this.head.visible = alive;
    const col = !alive ? 0xff5d5d : st.grabbed ? 0xffb03a : st.sealed ? 0x7fe0a8 : st.wall > 0.6 ? 0x9fd0ff : 0x6f7d88;
    this.ring.material.color.setHex(col);
    this.ring.material.opacity = alive ? 0.45 + 0.25 * Math.sin(t * 2 + this.phase) : 0.15;
    setLabel(this.label, alive ? st.name : `${st.name} (down)`,
      `${Math.round((st.wall || 0) * 100)}% wall · ${Math.round((st.roof || 0) * 100)}% roof${st.grabbed ? ' · GRABBED' : ''}`);
    this.label.visible = !this.hidden;
  }

  dispose() {
    this.group.traverse((o) => {
      if (o.isMesh || o.isSprite) { o.geometry?.dispose?.(); o.material?.dispose?.(); }
    });
  }
}

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

/**
 * The Locust. Tall, black, wrong. Limbs are long thin boxes; the arms end in
 * tendrils that sway; a crown of tubes curves around the skull; the face is a
 * pale-grey mask with two hollow (glowing) eyes and a triangular jaw.
 */
export class LocustMesh {
  constructor() {
    const black = new THREE.MeshLambertMaterial({ color: 0x07080a });
    const blackFlat = new THREE.MeshBasicMaterial({ color: 0x05060a });
    const metal = new THREE.MeshLambertMaterial({ color: 0xe7c7cf, emissive: 0x2a1620 });
    const face = new THREE.MeshLambertMaterial({ color: 0x2a2a2e });
    const eye = new THREE.MeshBasicMaterial({ color: 0xd8f6ff });

    const g = new THREE.Group();
    const H = 3.9;                       // 13 ft, matching its hitbox
    const hips = new THREE.Group();
    hips.position.y = 1.72;
    g.add(hips);

    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.44, 1.05, 0.26), black);
    torso.position.y = 0.5;
    hips.add(torso);
    const chest = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.34, 0.3), metal);
    chest.position.set(0, 0.72, 0.02);
    hips.add(chest);

    // long, thin limbs
    const mkLimb = (w, l, mat) => {
      const grp = new THREE.Group();
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, l, w), mat);
      m.position.y = -l / 2;
      grp.add(m);
      return grp;
    };
    this.armL = mkLimb(0.09, 1.5, black); this.armL.position.set(-0.28, 0.95, 0);
    this.armR = mkLimb(0.09, 1.5, black); this.armR.position.set(0.28, 0.95, 0);
    hips.add(this.armL, this.armR);
    this.foreL = mkLimb(0.07, 1.15, black); this.foreL.position.y = -1.5;
    this.foreR = mkLimb(0.07, 1.15, black); this.foreR.position.y = -1.5;
    this.armL.add(this.foreL); this.armR.add(this.foreR);
    for (const f of [this.foreL, this.foreR]) {
      f.userData.tendrils = [];
      for (let i = 0; i < 4; i++) {
        const ten = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.004, 0.7, 5), black);
        ten.position.set((i - 1.5) * 0.05, -0.85, 0.02);
        ten.rotation.z = (i - 1.5) * 0.16;
        f.add(ten);
        f.userData.tendrils.push(ten);
      }
    }
    this.legL = mkLimb(0.12, 1.6, black); this.legL.position.set(-0.14, -0.02, 0);
    this.legR = mkLimb(0.12, 1.6, black); this.legR.position.set(0.14, -0.02, 0);
    hips.add(this.legL, this.legR);
    for (const [arm, plate] of [[this.armL, 0.9], [this.armR, 0.9]]) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.22, 0.13), metal);
      p.position.y = -0.45 * plate;
      arm.add(p);
    }
    for (const leg of [this.legL, this.legR]) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.2, 0.16), metal);
      p.position.y = -0.9;
      leg.add(p);
    }

    // neck, head, face
    const neck = new THREE.Group();
    neck.position.y = 1.12;
    hips.add(neck);
    this.neck = neck;
    const skull = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.46, 0.4), face);
    neck.add(skull);
    const jaw = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.3, 3), face);
    jaw.position.set(0, -0.28, 0.04);
    jaw.rotation.x = Math.PI;
    neck.add(jaw);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.22, 4), face);
    nose.position.set(0, -0.06, 0.22);
    nose.rotation.x = Math.PI / 2.2;
    neck.add(nose);
    this.eyes = [];
    for (const sx of [-0.1, 0.1]) {
      const e = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 8), eye);
      e.position.set(sx, 0.06, 0.2);
      neck.add(e);
      this.eyes.push(e);
    }
    // hollow sockets so the eyes read as holes, not LEDs
    for (const sx of [-0.1, 0.1]) {
      const sock = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.1, 0.04), blackFlat);
      sock.position.set(sx, 0.06, 0.19);
      neck.add(sock);
    }
    // tube-like bio-mechanical appendages fanning out of the skull + hair
    this.tubes = [];
    for (let i = 0; i < 9; i++) {
      const a = (i / 9) * Math.PI * 2;
      const len = 0.34 + (i % 3) * 0.12;
      const t = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.014, len, 6), black);
      t.position.set(Math.cos(a) * 0.2, 0.2, Math.sin(a) * 0.2 - 0.05);
      t.rotation.set(Math.cos(a) * 0.9, 0, -Math.sin(a) * 0.9);
      neck.add(t);
      this.tubes.push(t);
    }
    const hair = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.62, 0.16), new THREE.MeshLambertMaterial({ color: 0x0b0b0f }));
    hair.position.set(0, -0.12, -0.2);
    hair.rotation.x = 0.25;
    neck.add(hair);

    // low hum made visible: a faint aura so it is findable in the dark
    const aura = new THREE.Mesh(
      new THREE.SphereGeometry(2.4, 18, 12),
      new THREE.MeshBasicMaterial({ color: 0x120a14, transparent: true, opacity: 0.18, side: THREE.BackSide, depthWrite: false }),
    );
    aura.position.y = H * 0.45;
    g.add(aura);
    this.aura = aura;

    const label = makeLabel('The Locust', 'it remembers every base');
    label.position.y = H + 0.6;
    this.label = label;
    g.add(label);

    g.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
    this.group = g;
    this.hips = hips;
    this.skull = skull;
    this.H = H;
    this.t = 0;
    this.facing = 0;
    this.mode = 'idle';
    this.crouch = 0;
  }

  /**
   * st: {x,y,z,yaw,pitch,stuck,holding,action,grabs,kills}
   * Actions drive pose so the hunt reads clearly: reach, grab, stab, smash.
   */
  update(st, dt, t, preyPos) {
    const g = this.group;
    this.t += dt;
    g.position.set(st.x, st.y, st.z);
    // smooth the yaw so a 20 Hz state feed does not snap the body around
    const want = st.yaw ?? 0;
    let d = want - this.facing;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    this.facing += d * Math.min(1, dt * 12);
    g.rotation.y = this.facing;

    const speed = this.speed ?? 0;
    const walking = Math.min(1, speed / 5);
    const cyc = Math.sin(t * 6.2) * walking;
    const crouch = this.mode === 'stalk' ? 0.25 : 0;
    this.crouch += (crouch - this.crouch) * Math.min(1, dt * 3);
    this.hips.position.y = 1.72 - this.crouch - walking * 0.05;
    this.legL.rotation.x = cyc * 0.55;
    this.legR.rotation.x = -cyc * 0.55;

    const grab = st.holding ? 1 : 0;
    const reach = this.mode === 'strike' ? 1 : 0;
    const smash = this.mode === 'smash' ? 1 : 0;
    const swing = this._swing = (this._swing ?? 0) + ((smash ? 1 : 0) - (this._swing ?? 0)) * Math.min(1, dt * 9);
    const targetArm = -0.3 - grab * 1.25 - reach * 1.35 + Math.sin(swing * Math.PI) * 1.5;
    for (const arm of [this.armL, this.armR]) {
      arm.rotation.x += (targetArm - arm.rotation.x) * Math.min(1, dt * 11);
      arm.rotation.z += ((grab ? 0.12 : 0.02) - arm.rotation.z) * Math.min(1, dt * 8);
    }
    const sway = Math.sin(t * 2.1) * 0.12;
    this.neck.rotation.z = sway * (1 - walking);
    this.neck.rotation.x = -0.12 + Math.sin(t * 1.4) * 0.06 + (st.pitch ?? 0) * 0.5;
    // tendrils twitch
    for (const f of [this.foreL, this.foreR]) {
      for (const [i, ten] of f.userData.tendrils.entries()) {
        ten.rotation.x = Math.sin(t * 5 + i * 1.7) * 0.35 + (grab || reach ? 0.4 : 0);
      }
    }
    for (const [i, tb] of this.tubes.entries()) {
      tb.rotation.y = Math.sin(t * 1.6 + i) * 0.3;
    }
    const eyePulse = 0.55 + 0.45 * Math.abs(Math.sin(t * (grab || reach ? 7 : 1.3)));
    for (const e of this.eyes) e.material.color.setRGB(0.55 * eyePulse + 0.25, 0.85 * eyePulse, 0.95 * eyePulse);
    this.aura.material.opacity = 0.1 + 0.12 * (grab || reach ? 1 : 0.3) + 0.04 * Math.sin(t * 3);
    setLabel(this.label, 'The Locust',
      st.kills ? `${st.kills} down · ${st.smashed} blocks smashed` : 'it remembers every base');
  }

  setMode(mode) { this.mode = mode; }

  dispose() {
    this.group.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
  }
}

/** Small debris burst when a block breaks; reused from a pool. */
export class ParticleBurst {
  constructor(scene, count = 220) {
    this.scene = scene;
    this.geo = new THREE.BufferGeometry();
    this.pos = new Float32Array(count * 3);
    this.col = new Float32Array(count * 3);
    this.vel = new Float32Array(count * 3);
    this.life = new Float32Array(count);
    this.head = 0;
    this.count = count;
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.mat = new THREE.PointsMaterial({ size: 0.14, vertexColors: true, transparent: true, opacity: 0.95 });
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }

  emit(x, y, z, color = 0x999999, n = 14, power = 3) {
    const c = new THREE.Color(color);
    for (let i = 0; i < n; i++) {
      const k = this.head = (this.head + 1) % this.count;
      this.pos[k * 3] = x + Math.random() * 0.6;
      this.pos[k * 3 + 1] = y + Math.random() * 0.6;
      this.pos[k * 3 + 2] = z + Math.random() * 0.6;
      this.vel[k * 3] = (Math.random() - 0.5) * power;
      this.vel[k * 3 + 1] = Math.random() * power * 0.9;
      this.vel[k * 3 + 2] = (Math.random() - 0.5) * power;
      this.col[k * 3] = c.r; this.col[k * 3 + 1] = c.g; this.col[k * 3 + 2] = c.b;
      this.life[k] = 0.55 + Math.random() * 0.4;
    }
  }

  step(dt) {
    const { pos, vel, life, count } = this;
    for (let i = 0; i < count; i++) {
      if (life[i] <= 0) continue;
      life[i] -= dt;
      vel[i * 3 + 1] -= 14 * dt;
      pos[i * 3] += vel[i * 3] * dt;
      pos[i * 3 + 1] += vel[i * 3 + 1] * dt;
      pos[i * 3 + 2] += vel[i * 3 + 2] * dt;
      if (life[i] <= 0) { pos[i * 3 + 1] = -50; }
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
  }
}

export { Limbs };
