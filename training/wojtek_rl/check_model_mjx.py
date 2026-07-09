"""Gate checks for the MJX model.

CPU mode (default): loop closure + stand test on plain MuJoCo, then a
few jitted MJX steps to prove the model compiles under MJX at all.
GPU mode (--gpu): adds a batched step-rate benchmark against the Go1
playground model on the same machine. Gate: fbb >= Go1/5.

Run on the box:  ./run.sh check --gpu
"""

import argparse
import sys
import time

import mujoco
import numpy as np

from wojtek_rl import paths


def check_static() -> bool:
    ok = True
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    d = mujoco.MjData(m)
    key = m.key("home")
    d.qpos[:] = key.qpos
    d.ctrl[:] = key.ctrl
    mujoco.mj_forward(m, d)
    z0 = d.qpos[2]
    for _ in range(int(5.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    closure = max(
        np.linalg.norm(
            d.body(f"{leg}_foot_link").xpos - d.body(f"{leg}_chain_close_a_link").xpos
        )
        for leg in paths.LEGS
    )
    print(f"stand: z {z0:.3f} -> {d.qpos[2]:.3f}   closure {closure * 1000:.2f} mm")
    if d.qpos[2] < 0.8 * z0:
        print("FAIL: robot does not stand under PD hold")
        ok = False
    if closure > 2e-3:
        print("FAIL: loop closure error above 2 mm")
        ok = False
    return ok


def _bench(mjx, jax, jp, mjx_model, qpos0, nenv, nsteps) -> float:
    def init(_):
        d = mjx.make_data(mjx_model)
        return d.replace(qpos=jp.array(qpos0))

    data = jax.vmap(init)(jp.arange(nenv))
    step = jax.vmap(lambda d: mjx.step(mjx_model, d))

    @jax.jit
    def run(data):
        def body(d, _):
            return step(d), None

        d, _ = jax.lax.scan(body, data, None, length=nsteps)
        return d

    jax.block_until_ready(run(data))  # compile
    t0 = time.perf_counter()
    jax.block_until_ready(run(data))
    dt = time.perf_counter() - t0
    return nenv * nsteps / dt


def check_gpu(nenv: int, nsteps: int) -> bool:
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco_playground import registry

    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    wojtek_model = mjx.put_model(m, impl="jax")
    wojtek_rate = _bench(mjx, jax, jp, wojtek_model, m.key("home").qpos, nenv, nsteps)
    print(f"wojtek : {wojtek_rate:,.0f} steps/s ({nenv} envs)")

    go1_env = registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "jax"})
    g = go1_env.mj_model
    go1_rate = _bench(
        mjx, jax, jp, go1_env.mjx_model, g.key("home").qpos, nenv, nsteps
    )
    print(f"go1 : {go1_rate:,.0f} steps/s ({nenv} envs)")
    print(f"ratio: {wojtek_rate / go1_rate:.2f} (gate: >= 0.20)")
    return wojtek_rate >= go1_rate / 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--nenv", type=int, default=4096)
    p.add_argument("--nsteps", type=int, default=200)
    args = p.parse_args()

    ok = check_static()
    if ok and not args.gpu:
        # prove the model compiles under MJX at all (tiny, CPU)
        import jax
        import jax.numpy as jp
        from mujoco import mjx

        m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
        _bench(mjx, jax, jp, mjx.put_model(m, impl="jax"), m.key("home").qpos, 2, 5)
        print("mjx: compiles and steps")
    if ok and args.gpu:
        ok = check_gpu(args.nenv, args.nsteps)
    print("GATE PASS" if ok else "GATE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
