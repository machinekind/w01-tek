"""Domain randomization for four_bar_bot, brax randomization_fn convention.

Modeled on mujoco_playground's Go1 randomize.py: a vmapped function builds
per-environment copies of the physics fields, then tree_replace produces a
batched model plus an in_axes template.

The original five fields (floor friction, base and link mass, one gain/kd
scale) draw from `r1..r5 = jax.random.split(rng, 5)`. New fields draw from a
folded-in sub-key so they leave `r1..r5` unchanged. With every new field
disabled, the output matches the original code (tests/test_dr_expansion.py).
"""

import jax
import jax.numpy as jnp

from wojtek_rl import paths

# Defaults for each dr_cfg field. The conf/config.yaml `dr:` block mirrors
# this shape; missing keys fall back to these values.
_DEFAULT_DR = {
    "com_offset": {"enable": False, "xy": 0.02, "z": 0.01},
    "joint_gains": {"enable": False, "gain_pct": 0.2, "kd_pct": 0.2},
    "dof": {
        "enable": False,
        "damping": [0.9, 1.1],
        "armature": [0.9, 1.1],
        "frictionloss": [0.9, 1.1],
    },
    "foot_friction": {"enable": False, "range": [0.8, 1.2]},
}


def _field_cfg(dr_cfg, name):
    return {**_DEFAULT_DR[name], **((dr_cfg or {}).get(name) or {})}


def make_domain_randomize(mj_model, dr_cfg=None):
    """Returns a domain_randomize(model, rng) closure.

    The closure needs geom and body ids, and only MjModel carries names.
    The ids are resolved here once, outside jit.

    `dr_cfg=None` (or all-`enable: False`) reproduces the original 5-field
    DR (floor friction, base/link mass, single gain/kd scale) bitwise.
    """
    com_cfg = _field_cfg(dr_cfg, "com_offset")
    joint_cfg = _field_cfg(dr_cfg, "joint_gains")
    dof_cfg = _field_cfg(dr_cfg, "dof")
    foot_cfg = _field_cfg(dr_cfg, "foot_friction")

    floor_id = mj_model.geom("floor").id
    root_id = mj_model.body("root").id
    foot_ids = jnp.array(
        [mj_model.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
    )

    def domain_randomize(model, rng):
        @jax.vmap
        def rand(rng):
            r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
            friction = jax.random.uniform(r1, minval=0.6, maxval=1.2)
            geom_friction = model.geom_friction.at[floor_id, 0].set(friction)

            base_scale = jax.random.uniform(r2, minval=0.7, maxval=1.3)
            link_scale = jax.random.uniform(
                r3, (model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass * link_scale
            body_mass = body_mass.at[root_id].set(
                model.body_mass[root_id] * base_scale
            )

            if joint_cfg["enable"]:
                rg, rk = jax.random.split(jax.random.fold_in(rng, 1), 2)
                pct = joint_cfg["gain_pct"]
                kpct = joint_cfg["kd_pct"]
                gain_scale = jax.random.uniform(
                    rg, (model.nu,), minval=1 - pct, maxval=1 + pct
                )
                kd_scale = jax.random.uniform(
                    rk, (model.nu,), minval=1 - kpct, maxval=1 + kpct
                )
            else:
                # One scalar from r4/r5 broadcast to all joints, matching the
                # original single-scale code.
                gain_scale = jnp.full(
                    (model.nu,), jax.random.uniform(r4, minval=0.8, maxval=1.2)
                )
                kd_scale = jnp.full(
                    (model.nu,), jax.random.uniform(r5, minval=0.8, maxval=1.2)
                )

            gainprm = model.actuator_gainprm.at[:, 0].multiply(gain_scale)
            biasprm = model.actuator_biasprm.at[:, 1].multiply(gain_scale)
            biasprm = biasprm.at[:, 2].multiply(kd_scale)
            forcerange = model.actuator_forcerange * gain_scale[:, None]

            out = {
                "geom_friction": geom_friction,
                "body_mass": body_mass,
                "actuator_gainprm": gainprm,
                "actuator_biasprm": biasprm,
                "actuator_forcerange": forcerange,
            }

            if com_cfg["enable"]:
                rc = jax.random.fold_in(rng, 2)
                xy, z = com_cfg["xy"], com_cfg["z"]
                offset = jax.random.uniform(
                    rc,
                    (3,),
                    minval=jnp.array([-xy, -xy, -z]),
                    maxval=jnp.array([xy, xy, z]),
                )
                out["body_ipos"] = model.body_ipos.at[root_id].add(offset)

            if dof_cfg["enable"]:
                rd1, rd2, rd3 = jax.random.split(jax.random.fold_in(rng, 3), 3)
                dlo, dhi = dof_cfg["damping"]
                alo, ahi = dof_cfg["armature"]
                flo, fhi = dof_cfg["frictionloss"]
                out["dof_damping"] = model.dof_damping * jax.random.uniform(
                    rd1, (model.nv,), minval=dlo, maxval=dhi
                )
                out["dof_armature"] = model.dof_armature * jax.random.uniform(
                    rd2, (model.nv,), minval=alo, maxval=ahi
                )
                out["dof_frictionloss"] = model.dof_frictionloss * jax.random.uniform(
                    rd3, (model.nv,), minval=flo, maxval=fhi
                )

            if foot_cfg["enable"]:
                rf = jax.random.fold_in(rng, 4)
                flo2, fhi2 = foot_cfg["range"]
                foot_scale = jax.random.uniform(
                    rf, (foot_ids.shape[0],), minval=flo2, maxval=fhi2
                )
                out["geom_friction"] = out["geom_friction"].at[foot_ids, 0].multiply(
                    foot_scale
                )

            return out

        out = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in out})
        model = model.tree_replace(out)
        return model, in_axes

    return domain_randomize
