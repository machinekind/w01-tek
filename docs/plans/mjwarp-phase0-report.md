# Phase-0 report: MJWarp feasibility (IN PROGRESS — pre-GPU findings)

**Status:** pre-GPU findings complete; GPU-runtime sections pending the spike run on an NVIDIA box.
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

## 4. PENDING (GPU) — to fill from the spike run

- [ ] **Four-bar closure under `impl="warp"`** — residual ≤ 2 mm (the `check_model_mjx` gate) at iterations=2, no NaN, standing stable. _Kill criterion if unstable._
- [ ] **DR batching under warp** — vmap 64 envs, per-env friction/mass via `tree_replace`; confirm fields batch (or note the alternative API).
- [ ] **Throughput table** — env steps/s at 1024/4096/8192 envs, jax vs warp, our real env step. _Gate: ≥ 2× @ 4096 on the 4090._
- [ ] **Multi-device** — 2-GPU brax pmap smoke under warp (cluster, non-blocking).
- [ ] **Golden-baseline guard** — fixed-seed 10M jax probe before/after any dep bump.

## 5. Go/no-go — PENDING (GPU)

_Recommendation to Marcin (G0) after the spike + throughput numbers are in._
