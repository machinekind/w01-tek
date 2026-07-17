"""Randomized per-substep control latency (Workstream B).

See docs/plans/2026-07-10-mjwarp-migration.md, Workstream B. The golden
regression is the key gate: it proves the disabled default (action_delay=1,
latency.enable=False) still reproduces pre-refactor trajectories bitwise.
"""

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths

GOLDEN = np.load(paths.PROJECT_DIR / "tests" / "data" / "latency_golden.npz")


def _rollout(env, seed, actions):
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    step = jax.jit(env.step)
    qpos, qvel = [], []
    for a in actions:
        state = step(state, a)
        qpos.append(np.array(state.data.qpos))
        qvel.append(np.array(state.data.qvel))
    return np.array(qpos), np.array(qvel)


def _config(enable, min_substeps, max_substeps):
    cfg = wojtek_env.default_config()
    cfg.latency.enable = enable
    cfg.latency.min_substeps = min_substeps
    cfg.latency.max_substeps = max_substeps
    return cfg


def test_golden_bitwise_regression():
    """Default config (latency disabled) matches the pre-refactor golden
    exactly: same seed, same stored actions, byte-identical qpos/qvel at
    every one of the 40 steps."""
    seed = int(GOLDEN["seed"])
    env = wojtek_env.WojtekJoystick()
    qpos, qvel = _rollout(env, seed, GOLDEN["actions"])
    assert np.array_equal(qpos, GOLDEN["qpos"])
    assert np.array_equal(qvel, GOLDEN["qvel"])


def test_full_delay_matches_disabled_default():
    """enable=True, min=max=n_substeps(=5) is today's action_delay=1: the
    whole control period applies the previous step's targets. The enabled
    path runs the single where-scan (not the stock mjx_env.step of the
    disabled default), so it matches to float tolerance, not bitwise --
    only the disabled default carries the bitwise-golden guarantee
    (test_golden_bitwise_regression). The tolerance is self-calibrated by
    the contrast below: after one control step d=5 tracks the default 183x
    (qpos) / 51x (qvel) tighter than d=0 (which applies the NEW targets
    immediately), so it sits well below any real semantic difference. The
    single scan is deliberate -- a per-lane lax.cond would triple the
    substep physics under the training vmap."""
    seed = int(GOLDEN["seed"])
    a0 = GOLDEN["actions"][0]

    def _one_step(config):
        env = (
            wojtek_env.WojtekJoystick(config) if config
            else wojtek_env.WojtekJoystick()
        )
        s = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(seed)), a0)
        return np.array(s.data.qpos), np.array(s.data.qvel)

    qp_def, qv_def = _one_step(None)  # disabled default -> mjx_env.step
    qp_full, qv_full = _one_step(_config(True, 5, 5))  # where-scan, d=5
    qp_none, qv_none = _one_step(_config(True, 0, 0))  # where-scan, d=0
    # d=5 reproduces the default's previous-step targets (float drift only:
    # 2.3e-4 qpos / 0.058 qvel measured on the 14 kg model, amplified through
    # the contact-rich first step; the d=0 contrast is 0.042 qpos / 2.97 qvel,
    # so measured drift stays 183x / 51x below a real semantic change)
    assert np.max(np.abs(qp_full - qp_def)) < 1e-3
    assert np.max(np.abs(qv_full - qv_def)) < 0.3
    # ...and those guards are meaningful: d=0 (new targets now) diverges more
    assert np.max(np.abs(qp_none - qp_def)) > 1e-2
    assert np.max(np.abs(qv_none - qv_def)) > 1.0


def test_no_delay_differs_from_full_delay():
    """d=0 (new targets applied immediately) and d=n_substeps (today's
    one-step-late targets) must diverge on a moving command."""
    seed = int(GOLDEN["seed"])
    actions = GOLDEN["actions"]
    env_full = wojtek_env.WojtekJoystick(_config(True, 5, 5))
    env_none = wojtek_env.WojtekJoystick(_config(True, 0, 0))
    qpos_full, _ = _rollout(env_full, seed, actions)
    qpos_none, _ = _rollout(env_none, seed, actions)
    assert not np.array_equal(qpos_full, qpos_none)
    assert np.all(np.isfinite(qpos_none))


def test_mid_delay_is_finite_and_distinct():
    """A mid-range fixed delay (2 of 5 substeps) runs cleanly and produces
    a trajectory distinct from both boundary cases."""
    seed = int(GOLDEN["seed"])
    actions = GOLDEN["actions"]
    env_mid = wojtek_env.WojtekJoystick(_config(True, 2, 2))
    env_full = wojtek_env.WojtekJoystick(_config(True, 5, 5))
    env_none = wojtek_env.WojtekJoystick(_config(True, 0, 0))
    qpos_mid, qvel_mid = _rollout(env_mid, seed, actions)
    qpos_full, _ = _rollout(env_full, seed, actions)
    qpos_none, _ = _rollout(env_none, seed, actions)
    assert np.all(np.isfinite(qpos_mid)) and np.all(np.isfinite(qvel_mid))
    assert not np.array_equal(qpos_mid, qpos_full)
    assert not np.array_equal(qpos_mid, qpos_none)


def test_per_env_delay_varies_and_is_in_range():
    """Vmapped reset over several rngs: per-env ctrl_delay varies and stays
    within [min_substeps, max_substeps]."""
    env = wojtek_env.WojtekJoystick(_config(True, 0, 5))
    rngs = jax.random.split(jax.random.PRNGKey(0), 8)
    state = jax.jit(jax.vmap(env.reset))(rngs)
    d = np.array(state.info["ctrl_delay"])
    assert d.shape == (8,)
    assert len(np.unique(d)) > 1
    assert np.all((d >= 0) & (d <= 5))


def test_ctrl_delay_always_present_disabled():
    """info['ctrl_delay'] exists even when latency is disabled (constant
    pytree structure required for brax scan-carry parity)."""
    env = wojtek_env.WojtekJoystick()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert "ctrl_delay" in state.info
    assert int(state.info["ctrl_delay"]) == env.n_substeps  # action_delay=1


def test_vmapped_step_with_latency_enabled():
    """Vmapped reset + step mirrors the batched shape the training loop
    actually runs: per-env ctrl_delay varies, and the where-scan in
    _step_with_latency must select correctly per-lane with `d` batched
    (the reason it's a jp.where scan and not a per-lane lax.cond, per the
    comment on _step_with_latency). No existing test runs step() under
    vmap with latency enabled."""
    env = wojtek_env.WojtekJoystick(_config(True, 0, 5))
    rngs = jax.random.split(jax.random.PRNGKey(0), 8)
    state = jax.jit(jax.vmap(env.reset))(rngs)
    d = np.array(state.info["ctrl_delay"])
    assert len(np.unique(d)) > 1

    step = jax.jit(jax.vmap(env.step))
    for v in (0.0, 0.2, -0.2):
        action = jp.full((8, 12), v)
        state = step(state, action)
        qpos = np.array(state.data.qpos)
        qvel = np.array(state.data.qvel)
        assert qpos.shape[0] == 8
        assert qvel.shape[0] == 8
        assert np.all(np.isfinite(qpos))
        assert np.all(np.isfinite(qvel))


def test_disabled_reset_consumes_no_extra_rng():
    """The disabled path must not shift the rng stream: info['rng'] after
    reset matches a plain 3-way split (rng, r_cmd, r_pos), same as before
    latency existed."""
    rng = jax.random.PRNGKey(0)
    expected_rng, _, _ = jax.random.split(rng, 3)
    env = wojtek_env.WojtekJoystick()
    state = jax.jit(env.reset)(rng)
    assert np.array_equal(np.array(state.info["rng"]), np.array(expected_rng))
