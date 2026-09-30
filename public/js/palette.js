/**
 * Identity colours for the survivors, keyed by the names in shared/rules.js so
 * the same bot always reads the same colour in the world, the HUD and the graph.
 */
export const BOT_COLORS = {
  You: 0xf2f4f6,
  Notch: 0x63b3ff,
  Ellie: 0xff8fb1,
  Voxel: 0x7ee081,
  Mira: 0xffd166,
  Brickz: 0xe2703a,
  Sable: 0xb08cff,
  Pixel: 0x5fe6d0,
};

export const LOCUST_COLOR = 0x0a0b0f;

export function colorFor(name) {
  return BOT_COLORS[name] ?? 0x9aa7b1;
}

export function hex(name) {
  return `#${colorFor(name).toString(16).padStart(6, '0')}`;
}
