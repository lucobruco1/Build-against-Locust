# Build to Survive the Locust

A Minecraft-like voxel game, in a browser tab, with no npm dependencies — except that
every one of the eight builders (seven bots plus you) is driven by a **real
[EfficientZero](https://arxiv.org/abs/2111.00210) network**, and so is the monster that
comes for them.

The loop is exactly the brief:

1. **1.5 minutes of build time.** Seven bots and you scramble to mine, craft from the
   ground and wall in a plot. A bot may only ever delete blocks **it placed itself** —
   nobody un-builds anybody else, not even you.
2. **3 minutes of hunt.** A Locust (after Doctor Nowhere's creature: thirteen feet of
   black limbs, hollow eye sockets, bio-mechanical tubing around a deformed face) is
   spawned by *another* EfficientZero network. It navigates the map, sees, attacks walls
   until they open, and when it reaches a body it **grabs it, holds it, and stabs it**.
3. **5 seconds of tally.** The Locust vanishes, every corpse — bots *and* the player —
   gets back up at its plot, the material pack refills, scores are settled, and the
   build phase starts again. Forever.

The networks keep what they learned across cycles, and can be saved to disk and resumed.

```
npm start          # http://localhost:3000  (game + server, port via PORT or --port)
npm test           # 95 unit/integration tests, node:test only
npm run sim        # headless match in the terminal, with the learning curves
```

Requires Node ≥ 18. Nothing to install: no build step, no packages, `three.js` r160.1 is
vendored under `public/vendor/`.

### …or no server at all

`index.html` lives at the **repo root** and every path in it (and every module specifier in
`public/js`, `shared/`, `core/`, `ai/`, `game/`) is relative, so any static host works:

```bash
python3 -m http.server 8000      # then open http://localhost:8000/index.html
```

With nothing listening on `/ws`, `public/js/net.js` gives up after a 1.5 s probe and boots
`game/local.js`, which is the *same* `Match` — same 30 Hz clock, same seven builders with
one EfficientZero brain each, same Locust — running in the tab. The HUD's menu line tells
you which of the two you are looking at (`in-tab match` vs the socket address). Weights
trained in the tab go to `localStorage`, and the in-tab engine never touches `/api/*`.
A double-clicked `file://` document is the one case that cannot work, because browsers
refuse ES modules from an opaque origin — hence "any static server", not "no server".

---

## Playing it

| input | action |
| --- | --- |
| `W` `A` `S` `D` / arrows | move |
| mouse | look |
| `Shift` | sprint |
| `Space` | jump — and **struggle** while you are being held |
| left click | break block |
| right click | place selected block |
| `1`–`9`, mouse wheel | hotbar |
| `H` | help overlay · `T` | push the HUD's sliders to the server (or to the tab) |
| `Esc` | release the pointer / dismiss the tally screen |

The bots act through the **same fifteen verbs** you do — the discrete action space is
`NOOP, FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT, LOOK_UP,
LOOK_DOWN, JUMP, PLACE_FRONT, BREAK_FRONT, PLACE_DOWN, BREAK_DOWN, SPRINT`, sampled from
an MCTS search, not from a scripted state machine. The Locust has those ten locomotion
verbs plus `LEAP`, `SMASH_BLOCK` and `STRIKE`.

You can also drive anything from the command line while it runs:

```bash
curl -s localhost:3000/api/health | jq
curl -s localhost:3000/api/state  | jq '.actors[] | {name,hp,wall,grabbed}'   # what the client gets
curl -s localhost:3000/api/snapshot | jq '.builders[] | {name,score,reward,inv}'  # the untrimmed one
curl -s localhost:3000/api/brains | jq '.brains[0] | {trainSteps,loss,policy,assist}'
curl -s -X POST localhost:3000/api/config -d '{"assist":0.6,"sims":1.5,"learn":true}'
curl -s -X POST localhost:3000/api/save            # writes .data/brains.json
curl -s -X POST localhost:3000/api/load            # puts the learned weights back
```

`/api/rules` dumps the single source of truth (`shared/rules.js`) the client, server and
tests all share, `/api/snapshot` the untrimmed match state, `/api/summary` the per-cycle
history, `POST /api/reset` and `POST /api/load` round out the set. The WebSocket at
`/ws` carries 20 Hz state deltas plus 30 Hz-collapsed world edits, RLE-packed (a full
72×40×72 grid is ~4 % of its raw size) and each `place` event carries the owner id, so
the client can grey out blocks that are not yours to break.

Those message shapes, not the transport, are the client's whole world: `game/local.js`
answers `join`, `input`, `config`, `say`, `act`, `reset`, `save`, `load` and `summary`
with byte-identical payloads, which is why the same `public/js/main.js` runs against
either. The server's `Game` and the tab's `LocalGame` share `applyMatchInput`,
`applyMatchConfig`, `foldHunt` and `cycleSummary` from that one file, so a settings change
or a hunt statistic cannot mean two different things depending on who is simulating.

---

## How the EfficientZero part is actually built

Not a "neural-net-flavoured" policy: the three-part latent model, the recurrent value
prefix, the self-supervised consistency loss and the search from the paper, implemented in
plain `Float32Array` math (`ai/nn.js`, `ai/tower.js`, `ai/efficientzero.js`, `ai/mcts.js`).

| paper element | here |
| --- | --- |
| representation `s = H(o)` | `specs.rep` — residual tower over the encoded observation |
| dynamics `ŝ', r̂ = G(s, a)` | `specs.dyn` — `next = latent + out`, action one-hot, plus a reward head |
| prediction `P(s), V(s)` | `specs.pred` — policy and value heads on the *imagined* latent |
| **value prefix** (EZ-V2 §4.1, fixes reward aliasing) | an accumulator channel carried through the unroll: `acc_{k+1} = relu(acc_k + accHead(h_k))`, and the head is trained against `Σ γ_p^k u_{t+k}` with `γ_p = 0.5` |
| SimSiam consistency | `specs.proj.W` projects `ŝ_{k+1}`, compared cosine-wise against the *detached* re-encoding `H(o_{t+1})`; weight `L_CONSIST = 2` |
| two-hot categorical supports | `Support` in `ai/nn.js`: value 61 bins on [-12, 12], reward 41 bins on [-4, 4]; the loss is cross-entropy on the interpolated pair, `L_VALUE = 0.25` |
| MCTS | PUCT (`C_PUCT = 1.6`), 12–120 sims (`--sims` scale), Dirichlet root noise (`α = 0.35`, ε = 0.25), first-play urgency for unvisited children (`FPU_PARENT = 0.25`), 0.5-scaled gradient through the unroll |
| targets | action = argmax root visits; target policy = the **visit-count distribution**; policy cross-entropy + an entropy bonus (`L_ENTROPY = 0.005`) |
| replay | prioritised *game sequences* (`ai/buffer.js`) with running-max \|TD error\| priorities, α = 0.6, β annealed 0.4 → 1, per-game tail trimming, and depth decay for old samples (`offPolicyDepth`, τ = 0.3) |
| stale samples | search-based value estimation (targets frozen at the recorded root value) while `stepsAgo < SVE_FRESH_STEPS`, n-step TD for fresh ones; optional reanalysis with extra sims |
| stability | AdamW (weight decay 1e-4), global gradient-norm clip 5, EMA target net for bootstrap values (`TARGET_SYNC`), NaN-guard that skips a poisoned batch instead of destroying the weights |
| league | `BrainLeague` — one brain per builder plus a persistent Locust brain; checkpoints pack/unpack as `Float32Array` blobs via `server/store.js` |

Two implementation details are worth naming because they are where naive
reimplementations quietly die, and both are pinned by tests:

* **the latent lives on a unit-RMS sphere.** `next = latent + out` has nothing pulling it
  back, so the magnitude climbs ~5 % per update, every head downstream inherits it, and
  the loss limit-cycles forever instead of descending. `_norm()` is applied to the
  representation output and to every imagined latent, and gradients w.r.t. them are
  projected onto the tangent plane, so the optimiser may rotate the state but not
  re-inflate it.
* **the entropy bonus is differentiated properly.** `d(-H)/dz_a = π_a (log π_a + H)`. The
  tempting shortcut (a bare `-(log π + 1)`) is unbounded, so a saturated softmax screams
  at the head with a gradient proportional to the number of nats — policy logits reach
  1e4 and imitation never sticks.

Observations are cheap and dense: `ai/obs.js` encodes a windowed block-type map as
**bit-packed planes** (2 channels for builders, 1 for the Locust, `cells/8` bytes each)
plus a ~30-float scalar tail (health, cover, security, phase clock, the Locust's bearing
and distance, inventory, what the crosshair is on…). 264 floats for a builder, 199 for
the Locust — a whole match's worth of *frames* fits in a few megabytes, which is what
makes self-play at 30 Hz feasible in one Node process.

Rewards are in `shared/rules.js` (`RW` for builders, `L*` for the Locust) and are the
only reason the networks care: filling a wall-shell cell pays
(`PLACE 0.10`, `PLACE_WALL 0.26`), building off-plot is punished, being hurt `−0.9`,
dying `−2.4`, surviving a cycle `+1.8`, sealing the ring `+1.0`; the Locust earns
`+0.17` per block opened, `+0.8` for a grab and `+2.6` for a kill (`+1.4` more if it is
you), and is charged for every stuck second.

While the networks are still cold, an **expert prior** (`ai/prior.js`) plays the action
for them with probability `assist · max(floor, 1 − updates/decay)` and its distribution
becomes the imitation target — that is `0.9 → 0.35` over ~900 updates for builders, and
held at `0.85+` for the Locust so the hunt stays dangerous. The sliders in the HUD
(`assist`, `sims`, `timeScale`, `learn`) retune this live via `POST /api/config`.

---

## What the headless run looks like

`node tools/simulate.js --cycles 2 --learn 1 --flat 1 --sims 0.5` on this repo's HEAD
(two full cycles, 550 s of game time, one process, no GPU):

```
c2  build  placed=1371 wall=0.43  loss=10.131  updates=35
c2  hunt   placed=1504 wall=0.45  loss= 8.134  updates=58
c2  hunt   placed=1835 wall=0.46  loss= 7.022  updates=103
c2  hunt   placed=1961 wall=0.46  loss= 6.840  updates=133     ← one bot killed

You     score  390 wall 38% placed 206 deaths 0
Ellie   score  702 wall 69% placed 413 deaths 0     ← best base, best score
Sable   score  436 wall 73% placed 137 deaths 1     ← the Locust got Sable
The Locust updates 140 buffer 2g/520t loss 5.104 policy 3.030 value 1.773
```

Mean loss over nine brains: **26.4 → 6.8** in 140 updates, with the prefix head at
~0.0001 and the SimSiam term at 0.15–0.29 (cos ≈ 0.8 between imagined and re-encoded
latents) — i.e. the model is fitting all four objectives, not just the easy ones.
Wall coverage ends between 17 % and 73 % per bot, and the Locust is training too, which
it only starts to do once its buffer crosses `MIN_BUFFER_FOR_TRAINING`.

The same loop verified on the real server with no client attached
(`node server/server.js --timeScale 40`, 113 s of wall clock = 2 full cycles):
`/api/summary` reported `cycles 2 · kills 5 · deaths 7 · blocksSmashed 182 ·
blocksPlaced 4174 · peakWallCoverage 0.43 · peakScore 1033`, the phase clock went
build → hunt → revive → build, the Locust was `null` outside its 3 minutes, all
8 builders were alive again at the start of cycle 3 — and the "player" slot, left
uncontrolled, had been eaten three times.

`--speed F` is only a *log cadence* knob: the simulation always steps at a fixed 30 Hz,
so results do not depend on it. (It used to inflate `dt`, which silently skipped 60× of
the AI decisions per second — do not judge behaviour with it.)
Coverage plateaus below 1.0 mainly because a bot must mine its own material: at
12 stacks a cycle a wall ring takes more placements than 90 s of decisions allow, so
roofs (`ring === 'roof'`, paid at 85 % of a wall cell) are a bonus layer the strong
policies reach for late.

Two things the hunt does on demand, verified with `tools/simulate.js`-style stepping:
a fully sealed **dirt** box is opened and its occupant killed in ~28 s (6 smashes, 1
grab); a sealed **stone** box takes ~65 s against the prior alone while the neighbours
patch the breach — which is why the hardness table in `shared/rules.js` matters more
than the wall height for surviving the night.

---

## Layout

```
shared/rules.js      single source of truth: world size, block defs, action ids, rewards,
                     EZ hyper-parameters, phase clock, plot layout, tiny math helpers
core/world.js        VoxelWorld: typed-array grid, per-cell owner ids, canBreak/place/
                     breakBlock, light + height caches, base-security scoring
core/worldgen.js     seeded terrain, and a flat arena for tests/sims
core/physics.js      AABB bodies: gravity, step-up, water, fall damage, grabs
core/raycast.js      DDA through the same grid the renderer draws; place/break targeting,
                     line-of-sight for the Locust's vision
core/nav.js          A* with a "smash through" cost, weakest-breach search, yaw helpers
core/inventory.js    hotbar + stacks, what an AI bot is allowed to place
ai/nn.js             Params (AdamW state per tensor), Linear, ReLU, LayerNorm, residual
                     blocks, categorical two-hot Support, softmax/entropy, pack/unpack
ai/tower.js          residual tower + heads, sharing weights between activation "views"
ai/efficientzero.js  the model: root / dynamic / predictFrom / targetValue /
                     trainSequences / checkpoint / health
ai/mcts.js           PUCT tree search over the learned dynamics, root noise, FPU
ai/obs.js            observation codec (bit-packed planes + scalars)
ai/buffer.js         Trajectory + prioritised replay with β annealing and reanalysis
ai/prior.js          the expert prior that bootstraps imitation, both sides
ai/brain.js          Brain / BrainLeague: decide → remember → train, checkpoints
game/actions.js      the verbs: place/break legality, ownership, smash, grab, strike
game/match.js        the phase clock, spawn/revive, scoring, rewards, human hooks
game/net.js          snapshot encoding, RLE world packing, event packing
game/local.js        the match with no server: the same tick/broadcast/hunt/record loop,
                     driven by a callback instead of a socket and by localStorage
                     instead of files — plus the helpers server/server.js uses too
index.html           the entry point, at the repo root, for both ways of serving it
server/server.js     static files + REST + WebSocket hub, 30 Hz tick / 20 Hz broadcast
server/{ws,store,api}.js  dependency-free RFC6455 hub, atomic JSON persistence, routes
public/              css, js/{main,net,hud,palette,audio}.js,
                     js/render/{voxels,actors,scene}.js, vendor/three.module.min.js
tools/simulate.js    headless runs / benchmarks / the integration harness
tests/               world 8 · nav 9 · physics 13 · actions 16 · match 15 · ai 21 ·
                     server 2 · client 11  → 95 tests, all green with `npm test`
```

The client renders voxels greedily per chunk from the same `shared/rules.js` block table,
so a block's colour, transparency and hardness are defined exactly once. The Locust is
built procedurally (long black body, tendril arms, tube crown around a hollow-eyed face)
in `public/js/render/actors.js` — no asset downloads, and it crouches.

## Things to know

* `BLOCK_DEFS[B.WATER].solid === true` on purpose (you cannot place into it), but it is
  not `opaque`, and `isBreakable`/hardness are separate predicates — the tests pin this.
* Bedrock and water have infinite hardness: the world border and the floor of the map are
  closed, and neither a bot nor the Locust can tunnel out through them.
* The Locust's `REACH_THROUGH_HOLE` is longer than its `GRAB_RANGE` on purpose: it can
  fish prey out of a half-sealed box if it has line of sight down the gap, but a fully
  sealed one is safe.
* Weights persist as base64 Float32 blobs in `.data/brains.json` (≈1.6 MB per brain);
  `POST /api/save` / `POST /api/load`, the 60 s autosave (which skips a write when no
  optimizer step happened since the last one), or `--save/--load` on the simulator.
  Legacy `Array.from(weights)` checkpoints still load, and the tests pin both forms.
* In the in-tab match, *you* are the CPU budget: eight brains doing MCTS at 30 Hz plus
  three towers of gradient work, in the same thread as the renderer. `train` is off by
  default there and the `sims` slider goes down to 0.2 (≈5 root simulations a decision) —
  drop it if the frame rate matters more than the bots' judgement. Learning still writes
  to `localStorage` when you press `T` or close the tab.
* `public/` is served twice by `npm start`, once as the repo layout (`/public/…`) so the
  root `index.html` resolves, once through the short `/js` `/css` `/vendor` aliases. The
  client never imports an absolute path, and `tests/client.test.js` re-resolves every
  specifier in the tree with browser rules against the files on disk — that is what keeps
  a static deployment honest rather than merely plausible.
