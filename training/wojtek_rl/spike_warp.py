"""TEMPORARY Phase-0 spike for the MJWarp migration. Delete after Phase 0.

Implements the 6 numbered steps of "Phase 0 — Feasibility spike" in
docs/plans/2026-07-10-mjwarp-migration.md: version report, load+step under
the MJX warp backend (four-bar closure + stability + foot-contact checks),
a domain-randomization batching probe, a static contact-read audit, a
jax-vs-warp throughput table, and a stub for the later 2-GPU pmap smoke.

NEEDS AN NVIDIA GPU BOX. Steps 2/3/5/6 require `impl="warp"` (MuJoCo-Warp),
which only runs on CUDA hardware. On a Mac/CPU-only machine this script
still runs steps 1 (versions) and 4 (contact-read audit) and then reports
the warp path as unavailable.

Run (on the GPU box): ./run.sh spike-warp   (or) python -m wojtek_rl.spike_warp
CPU dev sanity check:  python -m wojtek_rl.spike_warp --help
"""

import argparse
import functools
import importlib.metadata
import re
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from wojtek_rl import paths
from wojtek_rl.build_model import FOOT_RADIUS

# Reused from check_model_mjx.check_static's `if closure > 2e-3` gate (2 mm);
# that's the tolerance the existing jax-backend gate uses today.
FOUR_BAR_CLOSURE_TOL_M = 2e-3
# Foot-contact height heuristic margin, mirrors base.py WojtekEnv._foot_contact.
FOOT_CONTACT_MARGIN_M = 0.005

DEFAULT_ENV_COUNTS = "1024,4096,8192"

# Packages whose version we can look up directly via importlib.metadata under
# their pip distribution name.
_VERSION_PACKAGES = ("mujoco", "mujoco-mjx", "warp-lang", "jax", "brax", "playground")


# --------------------------------------------------------------------------
# Step 1 — version report
# --------------------------------------------------------------------------
def step1_version_report() -> dict:
    """Print installed versions of the packages this migration depends on.

    Import-defensive: the GPU box may need a dep bump before any of this
    works, so we only report what's actually present rather than assume.
    """
    print("\n--- Step 1: version report ---")
    versions = {}
    for pkg in _VERSION_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "NOT INSTALLED"

    # mujoco-warp: try the standalone pip distribution first (may exist on
    # the GPU box as its own wheel); current mujoco-mjx (3.10.0, checked on
    # this Mac) vendors MJWarp inside mujoco.mjx.warp instead, so fall back
    # to reporting that vendored copy's presence.
    try:
        versions["mujoco-warp"] = importlib.metadata.version("mujoco-warp")
    except importlib.metadata.PackageNotFoundError:
        try:
            from mujoco.mjx import warp as _mjxw

            versions["mujoco-warp"] = (
                f"vendored in mujoco.mjx.warp (WARP_INSTALLED={_mjxw.WARP_INSTALLED})"
            )
        except Exception as e:  # pragma: no cover - defensive, platform-dependent
            versions["mujoco-warp"] = f"NOT INSTALLED ({type(e).__name__}: {e})"

    for pkg, v in versions.items():
        print(f"  {pkg:<14} {v}")

    try:
        print(f"  jax devices    {jax.devices()}")
    except Exception as e:  # pragma: no cover - defensive
        print(f"  jax devices    ERROR: {type(e).__name__}: {e}")

    try:
        import warp as wp

        print(f"  warp CUDA available: {wp.is_cuda_available()}")
    except Exception as e:
        print(f"  warp CUDA available: ERROR ({type(e).__name__}: {e})")

    return versions


# --------------------------------------------------------------------------
# Warp availability gate
# --------------------------------------------------------------------------
def _warp_backend_probe(mj_model: mujoco.MjModel | None = None) -> tuple[bool, str]:
    """Best-effort check: can we actually build a warp-impl model right now?

    Catches broad Exception on purpose — failure modes vary by platform and
    dep versions (ImportError: warp-lang/mujoco_warp missing, AttributeError:
    mujoco-mjx/warp-lang version skew, RuntimeError: no CUDA device, ...).
    We report whichever one actually fired rather than special-case one.

    # TODO(orchestrator, verify on GPU): on this Mac (mujoco 3.10.0,
    # warp-lang 1.14.0, no mujoco_warp/no CUDA), `mjx.put_model(m,
    # impl="warp")` raises `AttributeError: type object 'int' has no
    # attribute 'WARP'` — almost certainly a mujoco-mjx <-> warp-lang
    # version-skew symptom, not a CPU-vs-GPU thing per se. Confirm on the
    # GPU box whether the *pinned* version set from step 1 hits the same
    # error before assuming "unavailable" only means "no GPU".
    """
    try:
        import warp as wp  # noqa: F401
    except Exception as e:
        return False, f"warp-lang not importable: {type(e).__name__}: {e}"

    try:
        from mujoco import mjx as _mjx  # noqa: F401
    except Exception as e:
        return False, f"mujoco.mjx not importable: {type(e).__name__}: {e}"

    m = mj_model
    if m is None:
        m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))

    try:
        mjx.put_model(m, impl="warp")
    except Exception as e:
        return False, f"mjx.put_model(impl='warp') failed: {type(e).__name__}: {e}"

    return True, "ok"


# --------------------------------------------------------------------------
# Shared CPU probe: typical ncon / nefc, used to size naconmax / njmax
# --------------------------------------------------------------------------
def _cpu_probe_ncon(m: mujoco.MjModel, steps: int = 200, seed: int = 0) -> tuple[int, int]:
    """Plain-MuJoCo (CPU) probe of typical `ncon`/`nefc`: home pose settle,
    then random actions. Mirrors check_model_mjx.check_static's stepping
    style but tracks contact/constraint counts instead of loop closure.
    """
    d = mujoco.MjData(m)
    key = m.key("home")
    d.qpos[:] = key.qpos
    d.ctrl[:] = key.ctrl
    mujoco.mj_forward(m, d)

    max_ncon, max_nefc = d.ncon, d.nefc
    for _ in range(steps):
        mujoco.mj_step(m, d)
        max_ncon = max(max_ncon, d.ncon)
        max_nefc = max(max_nefc, d.nefc)

    rng = np.random.default_rng(seed)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for _ in range(steps):
        d.ctrl[:] = rng.uniform(lo, hi)
        mujoco.mj_step(m, d)
        max_ncon = max(max_ncon, d.ncon)
        max_nefc = max(max_nefc, d.nefc)

    return int(max_ncon), int(max_nefc)


def _size_warp_budget(
    m: mujoco.MjModel, steps: int = 200, seed: int = 0, margin: float = 3.0
) -> tuple[int, int]:
    """Per-world naconmax/njmax from the CPU probe, with a safety margin.

    GPU-measured (4090, warp-lang 1.13.0, first spike run): mj's CPU probe
    badly underestimates warp's budgets. This scene probes ncon=5/nefc=54,
    but warp asked for naconmax≈21/world at rest (broadphase *candidates*,
    not final contacts; up to ~89 under random actions) and njmax≈210/world
    (dropped rows there open the four-bar equality constraint). Floors are
    sized to those measurements with ~50% headroom. Budgets are TOTAL across
    a vmapped batch — multiply by n_envs (step 5 does; A2's autosizing must
    too).
    """
    max_ncon, max_nefc = _cpu_probe_ncon(m, steps=steps, seed=seed)
    naconmax = max(128, int(max_ncon * margin))
    njmax = max(320, int(max_nefc * margin))
    print(
        f"  CPU probe: max ncon={max_ncon}, max nefc={max_nefc}  "
        f"-> naconmax={naconmax}, njmax={njmax} (margin={margin}x)"
    )
    return naconmax, njmax


# --------------------------------------------------------------------------
# Step 2 — load + step under warp
# --------------------------------------------------------------------------
def step2_warp_physics_check(steps: int = 2000, seed: int = 0, margin: float = 3.0) -> dict | None:
    """Step the wojtek scene under impl="warp" at home pose, then with random
    actions; check for NaN, four-bar closure residual, standing stability,
    and a plausible foot-contact reading.

    Pass criteria (each printed PASS/FAIL):
      - no NaN in qpos/qvel across either phase
      - four-bar closure residual <= FOUR_BAR_CLOSURE_TOL_M (max over both
        phases; check_model_mjx's own jax-backend gate is a single settled
        reading, we track the running max to also catch transient blowups)
      - standing stable: qpos[2] after the home-pose phase >= 0.8 * qpos[2]
        at the start (same 0.8x rule as check_model_mjx.check_static)
      - foot-contact heuristic (base.py's height threshold) reads a
        plausible number of feet down (1-4) at the end of the home phase
    """
    print("\n--- Step 2: load + step under warp ---")
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    naconmax, njmax = _size_warp_budget(m, seed=seed, margin=margin)

    warp_model = mjx.put_model(m, impl="warp")
    data = mjx.make_data(m, impl="warp", naconmax=naconmax, njmax=njmax)

    key = m.key("home")
    home_qpos = jp.array(key.qpos)
    home_ctrl = jp.array(key.ctrl)
    data = data.replace(qpos=home_qpos, qvel=jp.zeros(m.nv), ctrl=home_ctrl)
    data = mjx.forward(warp_model, data)

    foot_geom_ids = np.array([m.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS])
    closure_pairs = [
        (m.body(f"{leg}_foot_link").id, m.body(f"{leg}_chain_close_a_link").id)
        for leg in paths.LEGS
    ]

    step_fn = jax.jit(lambda d: mjx.step(warp_model, d))

    def closure_residual(d) -> float:
        xpos = np.asarray(d.xpos)
        return max(float(np.linalg.norm(xpos[a] - xpos[b])) for a, b in closure_pairs)

    def run(d, ctrls) -> tuple:
        """ctrls: None (hold current ctrl, home-pose phase) or an array of
        per-step ctrl targets (random-action phase)."""
        nan_hit = False
        max_closure = closure_residual(d)
        for i in range(len(ctrls) if ctrls is not None else steps):
            if ctrls is not None:
                d = d.replace(ctrl=jp.asarray(ctrls[i]))
            d = step_fn(d)
            qpos, qvel = np.asarray(d.qpos), np.asarray(d.qvel)
            if not nan_hit and (np.isnan(qpos).any() or np.isnan(qvel).any()):
                nan_hit = True
            max_closure = max(max_closure, closure_residual(d))
        return d, max_closure, nan_hit

    z0 = float(data.qpos[2])
    data, closure_home, nan_home = run(data, None)
    z_after_home = float(data.qpos[2])

    contact = np.asarray(data.geom_xpos)[foot_geom_ids][:, 2] < (
        FOOT_RADIUS + FOOT_CONTACT_MARGIN_M
    )
    n_feet_down = int(contact.sum())

    rng = np.random.default_rng(seed)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    rand_ctrls = rng.uniform(lo, hi, size=(steps, m.nu))
    data, closure_rand, nan_rand = run(data, rand_ctrls)

    nan_any = nan_home or nan_rand
    closure_max = max(closure_home, closure_rand)
    standing_ok = z_after_home >= 0.8 * z0
    contact_ok = 1 <= n_feet_down <= 4

    results = {
        "no_nan": not nan_any,
        "closure": closure_max <= FOUR_BAR_CLOSURE_TOL_M,
        "standing": standing_ok,
        "contact": contact_ok,
    }
    print(f"  {'PASS' if results['no_nan'] else 'FAIL'}: no NaN in qpos/qvel")
    print(
        f"  {'PASS' if results['closure'] else 'FAIL'}: four-bar closure "
        f"residual {closure_max * 1000:.2f} mm <= {FOUR_BAR_CLOSURE_TOL_M * 1000:.1f} mm"
    )
    print(
        f"  {'PASS' if results['standing'] else 'FAIL'}: standing stable "
        f"z {z0:.3f} -> {z_after_home:.3f} m"
    )
    print(
        f"  {'PASS' if results['contact'] else 'FAIL'}: foot-contact heuristic "
        f"{n_feet_down}/4 feet down after home-pose settle"
    )
    return results


# --------------------------------------------------------------------------
# Step 3 — DR batching probe
# --------------------------------------------------------------------------
def step3_dr_batching_probe(n_envs: int = 64, seed: int = 0) -> bool | None:
    """vmap n_envs copies of the DR model fields (friction/mass/gain/kd) via
    randomize.make_domain_randomize's tree_replace mechanism, against a
    warp-impl model. Confirms the fields batch; on failure, prints the exact
    error and points at the likely alternative API.
    """
    print(f"\n--- Step 3: DR batching probe ({n_envs} envs) ---")
    from wojtek_rl.randomize import make_domain_randomize

    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    warp_model = mjx.put_model(m, impl="warp")
    domain_randomize = make_domain_randomize(m)
    rngs = jax.random.split(jax.random.PRNGKey(seed), n_envs)

    dr_fields = (
        "geom_friction",
        "body_mass",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    )
    try:
        batched_model, _in_axes = domain_randomize(warp_model, rngs)
        shapes = {f: tuple(getattr(batched_model, f).shape) for f in dr_fields}
        bad = {f: s for f, s in shapes.items() if s[0] != n_envs}
        if bad:
            print(f"  FAIL: batched fields with wrong leading dim: {bad}")
            return False
        print(f"  PASS: DR fields batch under warp via tree_replace: {shapes}")
        # TODO(orchestrator, verify on GPU): this only proves the *Model*
        # pytree batches (tree_replace + leading dim). It does NOT prove you
        # can then vmap mjx.make_data/mjx.step across those n_envs worlds —
        # mjx.make_data's warp path only documents a plain mujoco.MjModel
        # input (see mujoco/mjx/_src/io.py _make_data_warp), so per-env
        # Data construction under warp may need MJWarp's native nworld
        # batching instead of jax.vmap(mjx.make_data). Step 5's warp
        # throughput numbers below assume vmap-of-single-world Data works;
        # if step 3 passes but step 5's warp legs blow up, this is why.
        return True
    except Exception as e:
        print(f"  FAIL: DR batching under warp raised {type(e).__name__}: {e}")
        print(
            "  alternative API to check: MJWarp's native per-world batching "
            "(nworld-sized arrays passed directly into the vendored "
            "ModelWarp/DataWarp dataclasses instead of jax.vmap + "
            "tree_replace over the whole Model pytree) — see "
            "mujoco.mjx.warp.mujoco_warp / mujoco.mjx.warp.mjwp_types."
        )
        return False


# --------------------------------------------------------------------------
# Step 4 — contact-read audit
# --------------------------------------------------------------------------
_CONTACT_READ_PATTERNS = (
    re.compile(r"\bdata\.contact\b"),
    re.compile(r"\._impl\b"),
    re.compile(r"\.ncon\b"),
)


def step4_contact_read_audit(package_dir: Path | None = None) -> list[tuple[str, int, str]]:
    """Scan wojtek_rl/*.py for `data.contact`, `data._impl`, `.ncon` reads
    outside the height heuristic. Warp stores contacts privately in `_impl`;
    anything hitting these patterns needs sensor migration (Workstream A3).
    The orchestrator has already found zero such reads by hand — this just
    confirms that mechanically and keeps confirming it as the code changes.
    """
    print("\n--- Step 4: contact-read audit ---")
    package_dir = package_dir or Path(__file__).resolve().parent
    hits = []
    for path in sorted(package_dir.rglob("*.py")):
        if path.name == "spike_warp.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if any(p.search(line) for p in _CONTACT_READ_PATTERNS):
                hits.append((str(path.relative_to(package_dir.parent)), lineno, line.strip()))

    if hits:
        print(f"  FAIL: {len(hits)} hit(s) needing sensor migration (Workstream A3):")
        for f, lineno, text in hits:
            print(f"    {f}:{lineno}: {text}")
    else:
        print(
            "  PASS: none found (no `data.contact` / `data._impl` / `.ncon` "
            "reads outside the height heuristic)"
        )
    return hits


# --------------------------------------------------------------------------
# Step 5 — throughput: jax vs warp, real env step
# --------------------------------------------------------------------------
def _reset_like_warp(env, rng: jax.Array, naconmax: int, njmax: int):
    """Rebuilds WojtekJoystick.reset()'s body against an explicit impl="warp"
    make_data call.

    env.reset() (env.py) calls `mjx.make_data(self._mjx_model)` with no
    impl/naconmax/njmax. That's fine under impl="jax" (make_data's own
    default resolves off the JAX device, matching the model), but under
    impl="warp" it raises `ValueError: Model impl Impl.WARP does not match
    make_data implementation Impl.JAX` — make_data's default-impl resolution
    (mujoco/mjx/_src/io.py:_resolve_impl_and_device) picks impl from the
    *device*, not from the model you hand it, when you don't pass impl=
    explicitly. Real backend plumbing (threading sim.backend + naconmax/
    njmax through base.py/env.py) is Workstream A, after this Phase-0 gate;
    until then this spike duplicates reset()'s body verbatim except for the
    make_data call, so env.step() below still runs on real, unmodified code.
    """
    rng, r_cmd, r_pos = jax.random.split(rng, 3)
    command = env._sample_command(r_cmd)
    anchor = env._height_ctrl(command[3])
    qpos = env._home_qpos.at[env._qadr].set(
        anchor + jax.random.uniform(r_pos, (12,), minval=-0.05, maxval=0.05)
    )
    qpos = qpos.at[2].set(command[3])
    data = mjx.make_data(env.mj_model, impl="warp", naconmax=naconmax, njmax=njmax)
    data = data.replace(qpos=qpos, qvel=jp.zeros(env.mj_model.nv), ctrl=anchor)
    data = mjx.forward(env._mjx_model, data)
    info = {
        "rng": rng,
        "command": command,
        "last_act": jp.zeros(12),
        "last_last_act": jp.zeros(12),
        "filtered_act": jp.zeros(12),
        "steps_since_cmd": jp.array(0),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "motor_targets": anchor,
        "step_count": jp.array(0),
        "phase": jp.array(0.0),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in env._config.reward.scales}
    obs = env._get_obs(data, info)
    from mujoco_playground._src import mjx_env

    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)


def _bench_real_env_step(
    backend: str, n_envs: int, n_steps: int, seed: int = 0, naconmax: int = 0, njmax: int = 0
) -> float:
    """steps/s for wojtek_rl.env.WojtekJoystick.step, vmapped over n_envs,
    after a JIT-warmup call. Mirrors check_model_mjx._bench's structure
    (vmap + lax.scan + block_until_ready timing) but drives the actual
    joystick env step (obs + reward included), not bare mjx.step, per the
    plan's step 5 requirement.
    """
    from wojtek_rl.env import WojtekJoystick

    env = WojtekJoystick()
    if backend == "warp":
        # base.py hardcodes impl="jax" (Workstream A adds the sim.backend
        # flag); swap the device model in place for this probe instead of
        # touching base.py/env.py (invariant: no existing file edits).
        env._mjx_model = mjx.put_model(env.mj_model, impl="warp")
        reset_fn = functools.partial(
            _reset_like_warp, env, naconmax=naconmax, njmax=njmax
        )
    elif backend == "jax":
        reset_fn = env.reset  # unmodified, real code — no bypass needed
    else:
        raise ValueError(f"unknown backend {backend!r}")

    rngs = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    state = jax.jit(jax.vmap(reset_fn))(rngs)
    step_v = jax.vmap(env.step)

    @jax.jit
    def run(state, key):
        def body(carry, _):
            s, k = carry
            k, sub = jax.random.split(k)
            action = jax.random.uniform(
                sub, (n_envs, env.action_size), minval=-1.0, maxval=1.0
            )
            return (step_v(s, action), k), None

        (s, _), _ = jax.lax.scan(body, (state, key), None, length=n_steps)
        return s

    key = jax.random.PRNGKey(seed + 1)
    jax.block_until_ready(run(state, key))  # compile + warmup
    t0 = time.perf_counter()
    jax.block_until_ready(run(state, key))
    dt = time.perf_counter() - t0
    return n_envs * n_steps / dt


def step5_throughput(
    env_counts: list, steps: int = 200, seed: int = 0, margin: float = 3.0
) -> list:
    """steps/s at each env count in env_counts, jax vs warp, same box, using
    the real WojtekJoystick.step (not bare physics).
    """
    print("\n--- Step 5: throughput, jax vs warp (real env step) ---")
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    naconmax, njmax = _size_warp_budget(m, seed=seed, margin=margin)

    rows = []
    for n_envs in env_counts:
        jax_rate = _bench_real_env_step("jax", n_envs, steps, seed=seed)
        try:
            # naconmax/njmax are TOTAL across all vmapped worlds (first GPU
            # run overflowed at 64 total for 8192 envs and silently dropped
            # contacts/constraint rows) — scale the per-world budget.
            warp_rate = _bench_real_env_step(
                "warp", n_envs, steps, seed=seed,
                naconmax=naconmax * n_envs, njmax=njmax * n_envs,
            )
        except Exception as e:
            warp_rate = None
            print(f"  warp @ {n_envs} envs FAILED: {type(e).__name__}: {e}")
        rows.append((n_envs, jax_rate, warp_rate))

    print(f"\n  {'envs':>8}  {'jax steps/s':>14}  {'warp steps/s':>14}  {'speedup':>8}")
    for n_envs, jax_rate, warp_rate in rows:
        warp_str = f"{warp_rate:,.0f}" if warp_rate else "FAILED"
        speedup = f"{warp_rate / jax_rate:.2f}x" if warp_rate else "n/a"
        print(f"  {n_envs:>8}  {jax_rate:>14,.0f}  {warp_str:>14}  {speedup:>8}")
    return rows


# --------------------------------------------------------------------------
# Step 6 — multi-device stub (cluster only, later; not blocking)
# --------------------------------------------------------------------------
def step6_multi_device_stub() -> None:
    """Scaffold only. A 2-GPU brax pmap smoke under warp is cluster-only,
    recorded pass/fail, and explicitly non-blocking for the G0 gate (see the
    plan's kill criteria and risk register). Not implemented here — no
    multi-GPU box is available to a Mac-side author to validate against.

    # TODO(orchestrator, verify on GPU): when run on a 2xH100 cluster node,
    # shard n_devices copies of a small env batch across
    # jax.local_devices()[:n_devices] via jax.pmap (mirroring train.py's
    # existing data-parallel jax pmap path, but with sim.backend="warp"),
    # run a short PPO rollout, and confirm no cross-device NaN/hang. Record
    # pass/fail in the phase0 report; do not gate G0 on it.
    """
    print("\n--- Step 6: multi-device pmap smoke (STUB, cluster-only, non-blocking) ---")
    print("  not implemented here — run later on a 2xH100 cluster node; see plan Phase 0 step 6")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-0 MJWarp feasibility spike (TEMPORARY, delete after Phase 0)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="per-phase step count for step 2's home-pose / random-action check (default: 2000)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=DEFAULT_ENV_COUNTS,
        help="comma-separated env counts for the step-5 throughput table (default: %(default)s)",
    )
    parser.add_argument(
        "--dr-envs", type=int, default=64, help="env count for the step-3 DR batching probe"
    )
    parser.add_argument(
        "--throughput-steps",
        type=int,
        default=200,
        help="scan length per throughput measurement in step 5",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="safety multiplier applied to the CPU-probed ncon/nefc when sizing naconmax/njmax",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase-0 MJWarp feasibility spike — TEMPORARY, delete after Phase 0")
    print("=" * 70)

    step1_version_report()
    step4_contact_read_audit()

    ok, msg = _warp_backend_probe()
    if not ok:
        print(
            "\nwarp backend unavailable on this platform "
            f"({msg}); this step needs an NVIDIA GPU box with a working "
            "mujoco-warp install. Steps 2/3/5/6 skipped."
        )
        return

    step2_warp_physics_check(steps=args.steps, seed=args.seed, margin=args.margin)
    step3_dr_batching_probe(n_envs=args.dr_envs, seed=args.seed)

    env_counts = [int(x) for x in args.envs.split(",") if x.strip()]
    step5_throughput(
        env_counts, steps=args.throughput_steps, seed=args.seed, margin=args.margin
    )
    step6_multi_device_stub()


if __name__ == "__main__":
    main()
