"""Check that the dr.foot_friction draw is the friction the feet walk on.

MuJoCo combines the friction of two equal-priority geoms with an element-wise
max, so a foot draw below the floor's value never reaches the contact unless
the feet carry contact priority. randomize.py sets that priority; this probe
measures the result end to end on whichever backend the box has, which is what
tests/integration/test_dr_expansion.py cannot do on a machine without pytest.

Run on the box:  ./run.sh friction-probe --backend warp
"""

import argparse
import sys

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths
from wojtek_rl.randomize import make_domain_randomize

SETTLE_STEPS = 5


def probe(backend: str, num_envs: int, friction_range) -> bool:
    cfg = wojtek_env.default_config()
    cfg.sim.backend = backend
    cfg.sim.num_envs = num_envs
    env = wojtek_env.WojtekJoystick(cfg)
    m = env.mj_model
    print(f"backend: {env._backend}   envs: {num_envs}")

    randomize = make_domain_randomize(
        m,
        {
            "foot_friction": {"enable": True, "range": list(friction_range)},
            "motor_strength": {"enable": False},
        },
    )
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    model_v, in_axes = randomize(env.mjx_model, keys)

    foot_ids = [m.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
    floor_id = m.geom("floor").id
    priority = np.array(model_v.geom_priority)
    if not np.all(priority[foot_ids] == 1):
        print(f"FAIL: foot geom_priority is {priority[foot_ids]}, expected 1")
        return False

    home = m.key("home")

    def run(model):
        data = env._make_data().replace(
            qpos=jp.array(home.qpos), ctrl=jp.array(home.ctrl)
        )
        data = mjx.forward(model, data)

        def body(d, _):
            return mjx.step(model, d), None

        return jax.lax.scan(body, data, None, length=SETTLE_STEPS)[0]

    out = jax.jit(jax.vmap(run, in_axes=(in_axes,)))(model_v)
    contact = getattr(out, "_impl", out).contact
    geom = np.array(contact.geom)
    dist = np.array(contact.dist)
    friction = np.array(contact.friction)

    base = float(env.mjx_model.geom_friction[foot_ids[0], 0])
    sampled = np.array(model_v.geom_friction)[:, foot_ids, 0]
    ok = True
    for e in range(num_envs):
        seen = np.full(len(foot_ids), np.nan)
        for k in range(geom.shape[1]):
            g1, g2 = int(geom[e, k, 0]), int(geom[e, k, 1])
            if dist[e, k] >= 0 or floor_id not in (g1, g2):
                continue
            foot = g2 if g1 == floor_id else g1
            if foot in foot_ids:
                seen[foot_ids.index(foot)] = friction[e, k, 0]
        for i, leg in enumerate(paths.LEGS):
            drawn = sampled[e, i]
            bad = np.isnan(seen[i]) or abs(seen[i] - drawn) > 1e-5
            ok = ok and not bad
            print(
                f"env {e} {leg:12s} x{drawn / base:5.3f} -> drawn {drawn:6.4f} "
                f"contact {seen[i]:6.4f}" + ("   MISMATCH" if bad else "")
            )
    lo, hi = friction_range
    if not np.all((sampled >= base * lo - 1e-6) & (sampled <= base * hi + 1e-6)):
        print(f"FAIL: draws outside {base * lo:.4f}..{base * hi:.4f}")
        ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument(
        "--range", type=float, nargs=2, default=[0.4, 1.35], metavar=("LO", "HI"),
        help="dr.foot_friction multiplier range (default: the terrain presets')",
    )
    args = p.parse_args()
    ok = probe(args.backend, args.num_envs, args.range)
    print("PROBE PASS" if ok else "PROBE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
