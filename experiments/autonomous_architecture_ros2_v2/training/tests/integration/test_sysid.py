"""Engine-parameter identification: space mapping, bag reading, recovery.

The rollout test generates its ground truth in-process (rollout under the
default genome), so it needs no bag fixture and validates the full
losses() path: the true genome must score near zero and clearly beat a
perturbed one. The bag test writes a tiny ros2 bag with the same rosbags
package the reader uses and checks the URDF->MJC conversion round-trips.
"""

import numpy as np
import pytest

import mujoco

from wojtek_rl import paths
from wojtek_rl.sysid.space import ALL_PARAMS, ParamSpace


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))


@pytest.fixture(scope="module")
def act_names(mj_model):
    return [mj_model.actuator(i).name for i in range(mj_model.nu)]


def test_default_genome_encodes_model_values(mj_model):
    space = ParamSpace(mj_model, grouping="per_type")
    d = space.describe(space.default_genome())
    # wojtek_mjx.xml: kp=20, kd=1, damping=0.05, armature=0.01,
    # frictionloss=0.01 on every actuated joint.
    for g in ("first", "second", "third"):
        assert d["kp"][g] == pytest.approx(20.0, rel=0.05)
        assert d["kd"][g] == pytest.approx(1.0, rel=0.05)
        assert d["damping"][g] == pytest.approx(0.05, rel=0.05)
    assert d["torque_scale"] == pytest.approx(1.0, rel=0.05)


def test_encode_describe_roundtrip(mj_model):
    space = ParamSpace(mj_model, grouping="per_type")
    u = space.encode({"kp": [30.0, 40.0, 50.0], "latency": 2.0})
    d = space.describe(u)
    assert d["kp"]["first"] == pytest.approx(30.0, rel=1e-3)
    assert d["kp"]["third"] == pytest.approx(50.0, rel=1e-3)
    assert d["latency"] == pytest.approx(2.0, abs=1e-3)


def test_apply_sets_model_fields(mj_model):
    from mujoco import mjx

    space = ParamSpace(mj_model, params=ALL_PARAMS, grouping="shared")
    mjx_model = mjx.put_model(mj_model)
    u = space.encode(
        {"kp": 42.0, "kd": 2.0, "damping": 0.1, "torque_scale": 0.8,
         "latency": 1.5}
    )
    model, latency = space.apply(mjx_model, u)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 42.0, rtol=1e-4)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -42.0, rtol=1e-4)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -2.0, rtol=1e-4)
    np.testing.assert_allclose(model.dof_damping[space.vadr], 0.1, rtol=1e-4)
    np.testing.assert_allclose(
        model.actuator_forcerange, 0.8 * np.array(mj_model.actuator_forcerange),
        rtol=1e-4,
    )
    assert float(latency) == pytest.approx(1.5, abs=1e-3)


def test_batched_model_marks_in_axes(mj_model):
    from mujoco import mjx

    space = ParamSpace(mj_model, params=("kp", "kd"), grouping="per_type")
    mjx_model = mjx.put_model(mj_model)
    U = np.random.default_rng(0).uniform(0.2, 0.8, size=(3, space.dim))
    model_v, in_axes, latency = space.batched_model(mjx_model, U)
    assert model_v.actuator_gainprm.shape[0] == 3
    assert in_axes.actuator_gainprm == 0
    assert in_axes.dof_damping is None
    assert latency.shape == (3,) and float(latency[0]) == 0.0


def _synthetic_dataset(mj_model, T=30):
    """One window starting at the home keyframe with a knee/hip sine; meas
    is filled by rolling out the true (default) parameters afterwards."""
    from wojtek_rl.sysid.dataset import SysidDataset

    key = mj_model.key("home")
    t = np.arange(T) * 0.02
    cmd = np.tile(key.ctrl, (T, 1))
    cmd[:, 1::3] += 0.15 * np.sin(2 * np.pi * 0.8 * t)[:, None]
    cmd[:, 2::3] += 0.20 * np.sin(2 * np.pi * 1.2 * t + 0.5)[:, None]
    return SysidDataset(
        ctrl_dt=0.02,
        n_substeps=int(round(0.02 / mj_model.opt.timestep)),
        warmup_steps=5,
        cmd=cmd[None],
        meas=np.zeros((1, T, mj_model.nu)),
        qpos0=np.array(key.qpos)[None],
        qvel0=np.zeros((1, mj_model.nv)),
        t0=np.zeros(1),
    )


def test_air_model_fixed_base(mj_model):
    from wojtek_rl.sysid.mount import air_model

    quat = np.array([0.0, 1.0, 0.0, 0.0])  # robot on its back
    m = air_model(paths.SCENE_XML, quat=quat, height=0.5)
    assert m.nq == mj_model.nq - 7
    assert m.nv == mj_model.nv - 6
    assert m.nu == mj_model.nu
    assert not any(
        m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(m.njnt)
    )
    # Home keyframe survives the surgery, joint-for-joint.
    kg, ka = mj_model.key("home"), m.key("home")
    for j in range(m.njnt):
        name = m.joint(j).name
        jg = mj_model.joint(name)
        assert ka.qpos[m.jnt_qposadr[j]] == pytest.approx(
            kg.qpos[mj_model.jnt_qposadr[jg.id]]
        )
    np.testing.assert_allclose(ka.ctrl, kg.ctrl)


def _fake_signals(mj_model, act_names, dur=3.0, hz=100.0):
    """Synthetic BagSignals: a small sine around the home pose, no IMU/odom
    (the layout a real-robot bag has)."""
    from wojtek_rl.sysid.bag import BagSignals

    t = np.arange(0.0, dur, 1.0 / hz)
    base = np.asarray(mj_model.key("home").ctrl)
    wave = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    qpos = base[None, :] + wave[:, None]
    qvel = (0.05 * 2 * np.pi * np.cos(2 * np.pi * 1.0 * t))[:, None] * np.ones(
        (1, len(act_names))
    )
    return BagSignals(
        t_cmd=t, cmd=qpos.copy(), t_meas=t, qpos=qpos, qvel=qvel, passive={},
        t_imu=None, quat=None, gyro=None, t_base=None, linvel=None,
    )


def test_build_dataset(mj_model, act_names):
    from wojtek_rl.sysid.dataset import build_dataset
    from wojtek_rl.sysid.mount import air_model

    model = air_model(paths.SCENE_XML)
    sig = _fake_signals(mj_model, act_names)
    ds = build_dataset(
        model, sig, window_sec=1.0, stride_sec=0.5, warmup_sec=0.2,
        max_windows=3,
    )
    assert ds.qpos0.shape == (3, model.nq)
    assert ds.qvel0.shape == (3, model.nv)
    assert np.isfinite(ds.qpos0).all() and np.isfinite(ds.qvel0).all()
    qadr = np.array(
        [model.jnt_qposadr[model.actuator(i).trnid[0]] for i in range(model.nu)]
    )
    np.testing.assert_allclose(
        ds.qpos0[0][qadr], ds.meas[0][0], atol=1e-9
    )


def test_air_losses_discriminate(mj_model):
    from dataclasses import replace

    from wojtek_rl.sysid.mount import air_model
    from wojtek_rl.sysid.rollout import make_evaluator

    m = air_model(paths.SCENE_XML, quat=[0.0, 1.0, 0.0, 0.0])
    space = ParamSpace(m, params=("kp", "kd", "latency"), grouping="per_type")
    ds = _synthetic_dataset(m)
    ev = make_evaluator(m, space, ds, backend="jax", popsize=2)
    u_true = space.default_genome()
    ds2 = replace(ds, meas=np.asarray(ev.rollout(u_true)))
    ev2 = make_evaluator(m, space, ds2, backend="jax", popsize=2)
    u_off = np.clip(u_true + 0.25, 0.0, 1.0)
    f = np.asarray(ev2.losses(np.stack([u_true, u_off])))
    assert f[0] < 1e-4, f"true params should replay exactly, rmse={f[0]}"
    assert f[1] > 10 * max(f[0], 1e-6), "perturbed params should score worse"


def test_bag_roundtrip(tmp_path, mj_model, act_names):
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    from wojtek_rl.sysid.bag import load_joint_map, read_bag

    ts = get_typestore(Stores.ROS2_HUMBLE)
    JointState = ts.types["sensor_msgs/msg/JointState"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    def js(t_ns, names, pos, vel):
        return JointState(
            header=Header(
                stamp=Time(sec=int(t_ns // 1_000_000_000),
                           nanosec=int(t_ns % 1_000_000_000)),
                frame_id="",
            ),
            name=list(names),
            position=np.asarray(pos, dtype=np.float64),
            velocity=np.asarray(vel, dtype=np.float64),
            effort=np.array([], dtype=np.float64),
        )

    rng = np.random.default_rng(1)
    q_urdf = rng.uniform(-0.5, 0.5, size=(5, len(act_names)))
    bag = tmp_path / "bag"
    with Writer(bag, version=8) as w:
        ct = w.add_connection(
            "/wojtek/joint_targets", JointState.__msgtype__, typestore=ts
        )
        cs = w.add_connection("/joint_states", JointState.__msgtype__, typestore=ts)
        for k in range(5):
            t_ns = k * 10_000_000
            msg = js(t_ns, act_names, q_urdf[k], np.zeros(len(act_names)))
            raw = ts.serialize_cdr(msg, JointState.__msgtype__)
            w.write(ct, t_ns, raw)
            w.write(cs, t_ns, raw)

    sig = read_bag(bag, act_names)
    assert sig.cmd.shape == (5, len(act_names))
    assert sig.qpos.shape == (5, len(act_names))
    jmap = load_joint_map()
    j = act_names.index("rear_left_second_joint")
    sign, offset = jmap["rear_left_second_joint"]
    np.testing.assert_allclose(sig.cmd[:, j], sign * (q_urdf[:, j] - offset))
    assert np.all(np.diff(sig.t_cmd) > 0)
