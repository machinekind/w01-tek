# Phase-0 report: MJWarp feasibility

**Status:** the GPU spike is complete (RTX 4090, vast.ai, 2026-07-10). The 2-GPU pmap
smoke remains open. It waits for a cluster node and does not block G0.
**Date started:** 2026-07-10
**Companion to:** the 2026-07-10 MJWarp migration plan (gate **G0**; not retained in this tree).

The spike script lived at `training/wojtek_rl/spike_warp.py`. It was temporary by design and was deleted from this branch once Phase 0 closed. Its run logs were never committed (`*.log` is gitignored repo-wide), so the numbers quoted below are the record. `phase0-artifacts/` keeps only the closure baseline script.

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

The spike ran four times. Runs 2 to 4 corrected the warp buffer budgets. The sizing rules
they produced are recorded below.

**The version finding in §1 was wrong.** Our `warp-lang==1.14.0` pin breaks the vendored
MJWarp in mujoco-mjx 3.10.0. Version 1.14 removed `warp._src.jax_experimental`, so the
`GraphMode` import falls back to `int` and `put_model(impl="warp")` raises
`AttributeError: type object 'int' has no attribute 'WARP'`. The Mac raised the same
error. It was version skew, not a missing GPU. mujoco-mjx 3.10.0 declares
`warp-lang==1.13.0` as its warp requirement, and everything works on 1.13.0. Workstream A
pins `warp-lang==1.13.0`.

- [x] **Closure and stability under warp: PASS.** No NaN. Standing is stable
  (z 0.125 m to 0.119 m). All four feet read in contact. The spike printed FAIL because
  its 2 mm closure gate applies a settled-pose tolerance to a running max over 2000
  random-action steps, and warp's running max was 2.35 mm. The same protocol under jax
  opens the loop to 18.1 mm and settles at 0.93 mm. Warp holds the closure about eight
  times tighter than jax. If the spike runs again, recalibrate its gate against the jax
  baseline.
- [x] **DR batching under warp: PASS at 64 envs.** All five randomized fields batch
  through the existing `tree_replace` path with correct leading dims. Workstream A keeps
  the current randomizer.
- [x] **Buffer budgets for A2.** `naconmax` counts broadphase candidates across all
  vmapped worlds. This scene needs about 21 per world at rest and up to 89 under random
  actions. `njmax` counts constraint rows per world. This scene needs about 210, and
  dropped rows weaken the four-bar equality constraint. Scaling njmax by the batch
  allocates a dense Jacobian of njmax times total dofs, which is 40 GiB at 1024 envs and
  crashes. The CPU ncon/nefc probe under-predicts both budgets. A2 should size
  `naconmax = 32 * n_envs` and `njmax = 320`.
- [x] **Throughput: gate passed.** Run 4 used correct budgets, produced no overflow
  warnings, and benchmarked the real `WojtekJoystick.step` with observations and rewards
  over a 200-step scan after warmup.

  | envs | jax steps/s | warp steps/s | speedup |
  |---|---|---|---|
  | 1024 | 117,188 | 355,185 | 3.03× |
  | 4096 | 108,395 | 652,252 | **6.02×** |
  | 8192 | 106,929 | 675,205 | 6.31× |

  The gate was 2× at 4096 envs. jax stays near 110k steps/s at every batch size. Warp
  keeps scaling. Run 2 read 6.6× at 4096 with overflowing budgets, so dropped contacts
  inflate throughput by about 10%.
- [ ] **Multi-device.** The 2-GPU pmap smoke waits for a cluster node. It does not block G0.
- [ ] **Golden guard.** Run the fixed-seed jax probe across the warp-lang pin change at
  the start of Workstream A. The goldens and their capture script are on this branch.

## 5. Go/no-go: GO

Every hardware gate passed. The one FAIL line came from the spike's own gate calibration,
and the jax baseline disproves it (reproduce with `phase0-artifacts/closure_jax_baseline.py`).
Workstream A
took three obligations from this spike, and all three are done:

1. `warp-lang==1.13.0` is pinned in pyproject and the lock.
2. `data_budget_kwargs` in base.py sizes warp buffers as `naconmax = 32 * n_envs` and
   `njmax = 320`, and never scales njmax by the batch.
3. The golden bitwise tests pass under the new lock, so the pin change left the jax
   path untouched.

## 6. Workstream A validation (RTX 4090, driver 580.126.09, 2026-07-10)

The `sim.backend` flag went in with three allowed values. `jax` and `warp` force a
backend. `auto` resolves to warp on a CUDA host with the vendored MJWarp importable, and
to jax elsewhere. The default is `auto`, so GPU training runs on warp and every CPU
consumer, including this test suite, stays on jax.

Three checks ran on the box through the new flag, all green:

- `./run.sh check --gpu --backend warp`: stand holds, settled closure 0.91 mm, bare
  physics at 3.3M steps/s for 4096 envs, 0.65 of Go1's jax rate against a 0.20 gate.
- A warp smoke train (100k steps, 64 envs): reward −6.22 to +2.39, every metric finite,
  full-length episodes, no overflow warnings.
- A jax smoke train on the same box for comparison: reward −7.02 to +3.22. Both backends
  learn to the same reward band. The warp run evaluated about four times faster at the
  smoke batch size.
