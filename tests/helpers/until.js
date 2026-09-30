import { AssertionError } from 'node:assert';

/**
 * Poll a condition on the real event loop. The client tests have to wait for
 * timers (a probe deadline, a setInterval tick, a dynamic import), and a fixed
 * `sleep` would be either flaky or slow — so each test says what it is waiting
 * for, and that phrase is the failure message when it does not happen.
 */
export async function until(fn, ms = 3000, what = 'condition') {
  const t0 = Date.now();
  for (;;) {
    let ok = false;
    try { ok = !!fn(); } catch (err) { throw new AssertionError({ message: `${what}: probe threw ${err?.message || err}` }); }
    if (ok) return true;
    if (Date.now() - t0 > ms) {
      throw new AssertionError({ message: `timed out after ${ms} ms waiting for ${what}`, operator: 'until' });
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}
