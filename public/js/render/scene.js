/**
 * Scene assembly: camera, sky, light rig, chunk meshes, actors and the cursor.
 * The camera is the local player's eyes; every other entity is interpolated from
 * the 20 Hz server feed so the whole world animates smoothly at display rate.
 */

import * as THREE from 'three';
import { VoxelRenderer, makeCursor } from './voxels.js';
import { BuilderMesh, LocustMesh, ParticleBurst } from './actors.js';
import { BLOCK_DEFS, B } from '../../../shared/rules.js';

const DAY = {
  sky: new THREE.Color(0x9ec7e8), fog: new THREE.Color(0xbfd8ea), sun: new THREE.Color(0xfff3d6), hemi: 0.72, sunI: 0.95,
};
const NIGHT = {
  sky: new THREE.Color(0x120f16), fog: new THREE.Color(0x0d0b12), sun: new THREE.Color(0xff9d7a), hemi: 0.24, sunI: 0.34,
};

export class GameScene {
  constructor(canvas, world) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(74, 1, 0.1, 340);
    this.sky = new THREE.Color().copy(DAY.sky);
    this.scene.background = this.sky;
    this.scene.fog = new THREE.Fog(this.sky.clone(), 26, 92);

    this.hemi = new THREE.HemisphereLight(0xdfefff, 0x2a2418, DAY.hemi);
    this.sun = new THREE.DirectionalLight(DAY.sun, DAY.sunI);
    this.sun.position.set(38, 62, 22);
    this.scene.add(this.hemi, this.sun);
    // a soft player-following light so caves/interiors stay readable
    this.torch = new THREE.PointLight(0xffe3b3, 0.9, 16, 1.6);
    this.scene.add(this.torch);

    this.world = world;
    this.voxels = new VoxelRenderer(this.scene, world);
    this.cursor = makeCursor();
    this.scene.add(this.cursor);
    this.particles = new ParticleBurst(this.scene);
    this.actors = new Map();
    this.locustMesh = null;
    this.locustLight = null;
    this.timeOfDay = 0;            // 0 = day (build), 1 = night (hunt)
    this.t = 0;
    this.shake = 0;
    this.prev = new Map();
    this.resize = this.resize.bind(this);
    addEventListener('resize', this.resize);
    this.resize();
  }

  setWorld(world) {
    this.world = world;
    this.voxels.scene = this.scene;
    this.voxels.world = world;
    this.voxels.buildAll();
  }

  rebuildWorld() { this.voxels.buildAll(); }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  ensureActor(st) {
    let a = this.actors.get(st.id);
    if (!a) {
      a = new BuilderMesh(st);
      this.scene.add(a.group);
      this.actors.set(st.id, a);
    }
    return a;
  }

  ensureLocust() {
    if (this.locustMesh) return this.locustMesh;
    this.locustMesh = new LocustMesh();
    this.scene.add(this.locustMesh.group);
    this.locustLight = new THREE.PointLight(0x88aaff, 0.0, 12, 2);
    this.scene.add(this.locustLight);
    return this.locustMesh;
  }

  hideLocust() {
    if (!this.locustMesh) return;
    this.scene.remove(this.locustMesh.group);
    this.locustMesh.dispose();
    this.locustMesh = null;
    if (this.locustLight) { this.scene.remove(this.locustLight); this.locustLight = null; }
  }

  /** speed estimation + smoothing between the last two server states */
  syncState(state, alpha) {
    const seen = new Set();
    for (const st of state.actors) {
      seen.add(st.id);
      const a = this.ensureActor(st);
      const p = this.prev.get(st.id);
      const speed = p ? Math.hypot(st.x - p.x, st.y - p.y, st.z - p.z) / Math.max(1e-3, alpha) : 0;
      a.speed = a.speed ? a.speed * 0.7 + speed * 0.3 : speed;
      a.hidden = st.id === this.localId;
      const heldId = st.inv?.[st.sel];
      a.setHeldBlock(heldId ? (BLOCK_DEFS[heldId]?.color ?? null) : null);
      a.update({ ...st, speed: a.speed, busy: a.speed > 0.4 }, this.t + alpha, this.t);
      this.prev.set(st.id, st);
    }
    for (const [id, a] of this.actors) if (!seen.has(id)) { this.scene.remove(a.group); a.dispose(); this.actors.delete(id); }

    if (state.locust?.active) {
      const l = this.ensureLocust();
      l.update({ ...state.locust, speed: 3 }, this.t, this.t);
    } else if (this.locustMesh) this.hideLocust();
  }

  /** per-frame: interpolate, animate, advance chunk rebuilds */
  frame(dt, localSt, look, targets) {
    this.t += dt;
    this.voxels.markDirty();
    this.voxels.step();
    this.particles.step(dt);

    // day/night: BUILD is bright, HUNT turns red-dark and closes the fog in
    const want = this.targetTimeOfDay ?? 0;
    this.timeOfDay += (want - this.timeOfDay) * Math.min(1, dt * 1.1);
    const k = this.timeOfDay;
    this.sky.copy(DAY.sky).lerp(NIGHT.sky, k);
    this.scene.background = this.sky;
    this.scene.fog.color.copy(DAY.fog).lerp(NIGHT.fog, k);
    this.scene.fog.near = 26 - 10 * k;
    this.scene.fog.far = 92 - 44 * k;
    this.hemi.intensity = DAY.hemi + (NIGHT.hemi - DAY.hemi) * k;
    this.sun.intensity = DAY.sunI + (NIGHT.sunI - DAY.sunI) * k;
    this.sun.color.copy(DAY.sun).lerp(NIGHT.sun, k);

    if (localSt) {
      const eye = localSt.y + 1.62;
      const sh = this.shake > 0 ? this.shake : 0;
      this.camera.position.set(
        localSt.x + (Math.random() - 0.5) * sh,
        eye + (Math.random() - 0.5) * sh + bobY(this.t, localSt.speed),
        localSt.z + (Math.random() - 0.5) * sh,
      );
      this.camera.rotation.order = 'YXZ';
      this.camera.rotation.set(look.pitch, look.yaw, 0);
      this.torch.position.set(localSt.x, eye + 0.4, localSt.z);
      this.torch.intensity = 0.55 + 0.9 * k + (localSt.grabbed ? 0.6 : 0);
      if (sh > 0) this.shake = Math.max(0, this.shake - dt * 2.4);
    }

    if (this.locustMesh) {
      const lp = this.locustMesh.group.position;
      this.locustLight.position.set(lp.x, lp.y + 2.2, lp.z);
      const d = localSt ? Math.hypot(lp.x - localSt.x, lp.z - localSt.z) : 99;
      this.locustLight.intensity = Math.max(0, 1.5 - d * 0.05);
      const near = Math.max(0, 1 - d / 22);
      if (targets?.mode) this.locustMesh.setMode(targets.mode);
      this.dread = near;
    } else this.dread = 0;

    // aim cursor
    if (targets) {
      const { breakCell, place } = targets;
      const c = breakCell || place;
      if (c && this.world.get(c.x, c.y, c.z) !== B.AIR) {
        this.cursor.visible = true;
        this.cursor.cursor.visible = true;
        this.cursor.ghost.visible = false;
        this.cursor.position.set(c.x + 0.5, c.y + 0.5, c.z + 0.5);
      } else if (place) {
        this.cursor.visible = true;
        this.cursor.cursor.visible = false;
        this.cursor.ghost.visible = true;
        this.cursor.position.set(place.x + 0.5, place.y + 0.5, place.z + 0.5);
      } else this.cursor.visible = false;
    }
    for (const a of this.actors.values()) a.update && a.tick?.(dt);
  }

  render() { this.renderer.render(this.scene, this.camera); }

  burst(cell, color, n = 14) {
    if (!cell) return;
    this.particles.emit(cell.x + 0.5, cell.y + 0.5, cell.z + 0.5, color, n);
  }

  kick(amount = 0.2) { this.shake = Math.min(0.55, this.shake + amount); }

  dispose() {
    this.voxels.dispose();
    for (const a of this.actors.values()) { this.scene.remove(a.group); a.dispose(); }
    this.hideLocust();
    this.renderer.dispose();
  }
}

function bobY(t, speed = 0) {
  const s = Math.min(1, (speed || 0) / 4.5);
  return Math.abs(Math.sin(t * 9)) * 0.06 * s;
}
