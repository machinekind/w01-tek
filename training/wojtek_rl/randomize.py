"""Domain randomization for four_bar_bot, brax randomization_fn convention.

Modeled on mujoco_playground's Go1 randomize.py: a vmapped function builds
per-environment copies of the physics fields, then tree_replace produces a
batched model plus an in_axes template.

The original five fields (floor friction, base and link mass, one gain/kd
scale) draw from `r1..r5 = jax.random.split(rng, 5)`. New fields draw from a
folded-in sub-key so they leave `r1..r5` unchanged. With every new field
disabled, the output matches the original code (tests/test_dr_expansion.py).

Fold-in index taxonomy (jax.random.fold_in(rng, i)): 1 joint_gains, 2
com_offset, 3 dof, 4 foot_friction, 5 motor_strength. motor_strength decouples
forcerange from joint_gains' gain_scale: disabled, forcerange keeps riding
gain_scale bitwise (a weak-motor world also reads as soft); enabled, it draws
its own per-actuator sample, so weak and soft are no longer the same draw.
"""

import jax
import jax.numpy as jnp
import mujoco

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
    # Weak-skewed: the failure mode this guards is motors too weak to pull
    # the hips back under the body, not a symmetric strength spread.
    "motor_strength": {"enable": False, "range": [0.5, 1.1]},
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
    motor_cfg = _field_cfg(dr_cfg, "motor_strength")

    # The terrain scene has no geom named "floor", only the heightfield and
    # the boxes. One friction draw covers them all, like the one floor plane
    # on flat.
    try:
        floor_id = mj_model.geom("floor").id
        terrain_ids = None
    except KeyError:
        floor_id = None
        names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            for i in range(mj_model.ngeom)
        ]
        terrain_ids = jnp.array([
            i for i, n in enumerate(names)
            if n == "terrain_hfield" or n.startswith("terrain_box_")
        ])
    root_id = mj_model.body("root").id
    foot_ids = jnp.array(
        [mj_model.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
    )

    def domain_randomize(model, rng):
        @jax.vmap
        def rand(rng):
            r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
            friction = jax.random.uniform(r1, minval=0.6, maxval=1.2)
            if floor_id is not None:
                geom_friction = model.geom_friction.at[floor_id, 0].set(friction)
            else:
                geom_friction = model.geom_friction.at[terrain_ids, 0].set(friction)

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

            if motor_cfg["enable"]:
                rm = jax.random.fold_in(rng, 5)
                mlo, mhi = motor_cfg["range"]
                motor_scale = jax.random.uniform(
                    rm, (model.nu,), minval=mlo, maxval=mhi
                )
            else:
                # Coupled to the gain draw, matching the original code: a
                # weak-motor world also reads as a soft one.
                motor_scale = gain_scale
            forcerange = model.actuator_forcerange * motor_scale[:, None]

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

        if foot_cfg["enable"]:
            # Equal-priority contacts take the element-wise max of the two
            # geoms' friction, so a foot draw below the floor's draw would
            # never reach the contact. Priority 1 on the feet makes the
            # foot's friction win outright. geom_priority is a static numpy
            # field in mjx (resolved at collision-pair setup, not under
            # jit), so it stays unbatched and is set with numpy indexing.
            priority = model.geom_priority.copy()
            priority[foot_ids] = 1
            model = model.tree_replace({"geom_priority": priority})

        return model, in_axes

    return domain_randomize
