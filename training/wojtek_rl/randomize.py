"""Domain randomization for four_bar_bot, brax randomization_fn convention.

Modeled on mujoco_playground's Go1 randomize.py: a vmapped function builds
per-environment copies of the physics fields, then tree_replace produces a
batched model plus an in_axes template.
"""

import jax


def make_domain_randomize(mj_model):
    """Returns a domain_randomize(model, rng) closure.

    The closure needs geom and body ids, and only MjModel carries names.
    The ids are resolved here once, outside jit.
    """
    floor_id = mj_model.geom("floor").id
    root_id = mj_model.body("root").id

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

            gain_scale = jax.random.uniform(r4, minval=0.8, maxval=1.2)
            kd_scale = jax.random.uniform(r5, minval=0.8, maxval=1.2)
            gainprm = model.actuator_gainprm.at[:, 0].multiply(gain_scale)
            biasprm = model.actuator_biasprm.at[:, 1].multiply(gain_scale)
            biasprm = biasprm.at[:, 2].multiply(kd_scale)
            forcerange = model.actuator_forcerange * gain_scale

            return geom_friction, body_mass, gainprm, biasprm, forcerange

        friction, mass, gainprm, biasprm, forcerange = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_mass": 0,
                "actuator_gainprm": 0,
                "actuator_biasprm": 0,
                "actuator_forcerange": 0,
            }
        )
        model = model.tree_replace(
            {
                "geom_friction": friction,
                "body_mass": mass,
                "actuator_gainprm": gainprm,
                "actuator_biasprm": biasprm,
                "actuator_forcerange": forcerange,
            }
        )
        return model, in_axes

    return domain_randomize
