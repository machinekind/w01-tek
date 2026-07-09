import jax
import mujoco
import numpy as np
from mujoco import mjx

from fbb_rl import paths
from fbb_rl.randomize import make_domain_randomize


def test_fields_vary_across_batch():
    mj_model = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    mjx_model = mjx.put_model(mj_model, impl="jax")
    rng = jax.random.split(jax.random.PRNGKey(0), 4)
    randomize = make_domain_randomize(mj_model)
    model_v, in_axes = randomize(mjx_model, rng)
    assert model_v.body_mass.shape[0] == 4
    assert not np.allclose(model_v.body_mass[0], model_v.body_mass[1])
    assert not np.allclose(model_v.actuator_gainprm[0], model_v.actuator_gainprm[1])
    assert in_axes.qpos0 is None
