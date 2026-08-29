"""Regenerate the golden fixtures in tests/data/.

These are pre-feature CPU-jax baselines: a deterministic action/state rollout
(latency_golden.npz) and a deterministic domain-randomization batch
(randomize_golden.npz), captured before the latency-injection and DR-expansion
changes land. The MJWarp migration's regression gates diff against these to
confirm the backend swap doesn't change behavior.

Run:
    cd training && JAX_PLATFORMS=cpu uv run python tests/capture_goldens.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths
from wojtek_rl.randomize import make_domain_randomize

DATA_DIR = paths.PROJECT_DIR / "tests" / "data"

LATENCY_SEED = 0
LATENCY_STEPS = 40

RANDOMIZE_SEED = 0
RANDOMIZE_N_ENVS = 8


def capture_latency_golden():
    env = wojtek_env.WojtekJoystick()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    # test_latency.py replays with reset(PRNGKey(GOLDEN["seed"])), so the
    # capture must reset with that exact key; only the actions come from a
    # derived key (they are stored verbatim in the npz).
    _, _, action_key = jax.random.split(jax.random.PRNGKey(LATENCY_SEED), 3)
    actions = jax.random.uniform(
        action_key, (LATENCY_STEPS, env.action_size), minval=-1.0, maxval=1.0
    )

    state = reset(jax.random.PRNGKey(LATENCY_SEED))
    qpos, qvel = [], []
    for i in range(LATENCY_STEPS):
        state = step(state, actions[i])
        qpos.append(np.asarray(state.data.qpos))
        qvel.append(np.asarray(state.data.qvel))

    np.savez(
        DATA_DIR / "latency_golden.npz",
        seed=np.array(LATENCY_SEED, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.float32),
        qpos=np.stack(qpos).astype(np.float32),
        qvel=np.stack(qvel).astype(np.float32),
    )


def capture_randomize_golden():
    mj_model = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    mjx_model = mjx.put_model(mj_model, impl="jax")
    rng = jax.random.split(jax.random.PRNGKey(RANDOMIZE_SEED), RANDOMIZE_N_ENVS)
    randomize = make_domain_randomize(mj_model)
    model_v, _ = randomize(mjx_model, rng)

    np.savez(
        DATA_DIR / "randomize_golden.npz",
        n_envs=np.array(RANDOMIZE_N_ENVS, dtype=np.int64),
        geom_friction=np.asarray(model_v.geom_friction, dtype=np.float32),
        body_mass=np.asarray(model_v.body_mass, dtype=np.float32),
        actuator_gainprm=np.asarray(model_v.actuator_gainprm, dtype=np.float32),
        actuator_biasprm=np.asarray(model_v.actuator_biasprm, dtype=np.float32),
        actuator_forcerange=np.asarray(model_v.actuator_forcerange, dtype=np.float32),
    )


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    capture_latency_golden()
    capture_randomize_golden()
    print(f"wrote {DATA_DIR / 'latency_golden.npz'}")
    print(f"wrote {DATA_DIR / 'randomize_golden.npz'}")
