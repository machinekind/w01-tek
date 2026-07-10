# Phase-0 report: MJWarp feasibility

**Status:** GPU spike complete (RTX 4090, vast.ai, 2026-07-10) except the final
clean-budget throughput table (round 4) and the non-blocking cluster pmap smoke.
**Date started:** 2026-07-10
**Companion to:** `docs/plans/2026-07-10-mjwarp-migration.md` (gate **G0**).

The spike script is `training/wojtek_rl/spike_warp.py`. Sections below marked _PENDING (GPU)_ get filled when it runs.

---

## 1. Version set (resolved on Mac/darwin CPU, current pins)

Installed and import-verified via `uv sync` against the committed `uv.lock`:

| package | version | note |
|---|---|---|
| jax | 0.9.2 | proven driver-580 pin; CPU on Mac |
| mujoco | 3.10.0 | |
| mujoco-mjx | 3.10.0 | **vendors MJWarp internally** (no standalone `mujoco-warp` wheel) |
| warp-lang | 1.14.0 | |
| brax | 0.14.2 | newer than the plan's ≥0.12.1 floor |
| playground | 0.2.0 | newer than the plan's ≥0.0.4 floor |

**Finding — the dep bump may be unnecessary.** The current pins already expose the complete MJWarp backend API:
- `mjx.put_model(m, impl="warp")` — `impl` accepts `"warp"`; `Impl` enum = {CPP, C, JAX, WARP}.
- `mjx.make_data(m, impl="warp", naconmax=…, naccdmax=…, njmax=…)` — all three budget knobs present; docstring: "maximum … across all worlds" (i.e. total across the vmapped batch — A2 autosizing must scale by num_envs).

On the Mac, `put_model(impl="warp")` fails with `AttributeError: type object 'int' has no attribute 'WARP'` — this is the **no-CUDA symptom** (the vendored warp `GraphMode` degrades to a bare `int` without a GPU), **not** an API version-skew. Confirm on the GPU box, but the API string and signatures are correct for these pins.

## 2. Contact-read audit (Phase-0 step 4) — COMPLETE

`grep` across `training/wojtek_rl/` for `data.contact` / `data._impl` / `.ncon` / `data.efc`: **zero hits** outside the foot geom-height heuristic (`base._foot_contact`, reads `data.geom_xpos`) and named force sensors (`data.sensordata`). Independently reconfirmed by the spike script's in-code audit.

**→ Workstream A3 (contact→sensor migration) is UNNECESSARY.** The reward/eval/battery path is already backend-portable.

## 3. Backend call-site inventory (for A1/A2)

The plan abbreviates "single call site"; there are actually a few to thread `sim.backend` through:
- `base.py:33` — `mjx.put_model(self._mj_model, impl="jax")` (the model put).
- `mjx.make_data(...)` in `env.py` `reset()`, `env_getup.py`, `env_jump.py` — currently bare (no `impl`); these will `Impl.WARP != Impl.JAX` mismatch under warp until A1 threads the backend + naconmax/njmax through them.
- `check_model_mjx.py` — put_model/make_data (benchmark path).

## 4. GPU results (RTX 4090, vast.ai, driver 595.71.05, 2026-07-10)

Four spike runs; runs 2–4 iterated the warp buffer-budget sizing (findings below).

**Version-set correction — the pre-GPU finding in §1 was wrong.** `warp-lang==1.14.0`
(our pin) breaks the vendored MJWarp glue on mujoco-mjx 3.10.0: 1.14 moved
`warp._src.jax_experimental`, the `GraphMode` import silently falls back to `int`, and
`put_model(impl="warp")` dies with `AttributeError: type object 'int' has no attribute
'WARP'` — the *same* error as on the Mac, so that error was version skew all along, not a
no-CUDA symptom. mujoco-mjx 3.10.0 declares `Requires-Dist: warp-lang==1.13.0; extra ==
"warp"`; with **warp-lang 1.13.0** everything works. → Workstream A must pin
`warp-lang==1.13.0` (a downgrade, not the feared bump).

- [x] **Four-bar closure / stability under `impl="warp"`** — no NaN, standing stable
  (z 0.125→0.119 m), 4/4 feet in contact. The literal ≤2 mm gate read **FAIL at 2.35 mm**,
  but that gate was mis-calibrated: it applies a settled-pose tolerance to a running max
  over 2000 *random-action* steps. The identical protocol under `impl="jax"` (same box,
  CPU) opens the loop to **18.1 mm** (0.93 mm settled). Warp holds the closure ~8× tighter
  than jax under abuse. _Verdict: PASS — no kill criterion touched; fix the spike gate to
  compare against the jax baseline if it's ever re-run._
- [x] **DR batching under warp** — PASS at 64 envs: all five randomized fields batch
  through the existing `tree_replace` path with correct leading dims. No native-nworld
  rewrite needed for Workstream A.
- [x] **Warp buffer budgets (A2 sizing rules, GPU-measured):**
  `naconmax` counts broadphase *candidates* pooled across ALL vmapped worlds
  (~21/world at rest, ~89/world under random actions; 8192 envs asked for 167k total).
  `njmax` is *per-world* constraint rows (~210 needed; scaling it by n_envs allocates a
  dense efc Jacobian of njmax_total × nv_total ≈ 40 GiB at 1024 envs → OOM).
  mj's CPU `ncon`/`nefc` probe under-predicts both. A2's autosizing must encode exactly
  this: `naconmax ≈ 32·n_envs`, `njmax ≈ 320` flat.
- [x] **Throughput table** — run 4, correct budgets (`naconmax=128·n_envs`, `njmax=320`),
  zero overflow warnings, real `WojtekJoystick.step` (obs+reward included), 200-step scan
  after warmup. Full log: `phase0-artifacts/spike-run4-4090.log`.

  | envs | jax steps/s | warp steps/s | speedup |
  |---|---|---|---|
  | 1024 | 117,188 | 355,185 | 3.03× |
  | 4096 | 108,395 | 652,252 | **6.02×** |
  | 8192 | 106,929 | 675,205 | 6.31× |

  _Gate was ≥2× @ 4096 → passed at 6×._ jax plateaus ~110k steps/s regardless of batch
  (solver-bound); warp keeps scaling. The mis-budgeted run 2 read 6.6× @ 4096 — dropped
  contacts inflated it by ~10%, so budget correctness matters for honest numbers but not
  for the go/no-go._
- [ ] **Multi-device** — 2-GPU brax pmap smoke under warp: deferred to a cluster node,
  explicitly non-blocking for G0.
- [ ] **Golden-baseline guard** — fixed-seed 10M jax probe before/after the warp-lang
  1.14→1.13 pin change: to run at the start of Workstream A (goldens + capture script are
  already on this branch).

## 5. Go/no-go — recommendation: **GO**

Every gate passed on hardware: physics stable (no NaN, standing, contacts), four-bar
closure ~8× tighter than jax under the same abuse protocol, DR batches through the
existing `tree_replace` path, and throughput at 6× the 2× gate. The single FAIL line in
the spike output is a mis-calibrated spike gate (settled-pose tolerance applied to a
random-action running max), disproven by the jax baseline in
`phase0-artifacts/closure_jax.log`.

Workstream A carries three concrete obligations out of this spike:
1. Pin `warp-lang==1.13.0` (mujoco-mjx 3.10.0's declared warp requirement; our 1.14.0
   breaks the vendored glue with the `GraphMode`→`int` fallback).
2. Encode the budget-sizing rules: `naconmax ≈ 32·n_envs` (total, broadphase candidates),
   `njmax ≈ 320` (per-world) — do NOT scale njmax by batch (40 GiB OOM at 1024 envs).
3. Run the golden-baseline guard (fixed-seed jax probe, goldens already on this branch)
   across the warp-lang pin change before flipping any default.

The 2-GPU cluster pmap smoke remains open and non-blocking, per the plan.
