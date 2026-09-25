"""Fixed-base ("air") model for actuator identification.

With the robot held off the ground (belly on a raised support) the legs
are fixed-base serial chains: no contact, no floating base, and window
initialization needs only the joint encoders. That removes the dominant
error source of on-ground fitting — the unobservable contact/base state
each window has to guess — so the actuator-side parameters (kp, kd,
damping, armature, frictionloss, latency) identify cleanly.

The model is built by deleting the free joint from the scene spec and
welding the base at a given orientation (normally the bag's IMU median,
so gravity acts on the legs exactly as it did during the recording) high
enough that the feet cannot reach the floor. The home keyframe is
re-added minus its free-joint block, so dataset window initialization can
keep seeding unmeasured (four-bar) joints from it.
"""

import mujoco
import numpy as np

# Free-joint block sizes at the front of qpos/qvel.
_NQ_FREE, _NV_FREE = 7, 6


def air_model(xml_path, quat=None, height=0.5):
    """Compile the scene with the base welded in place.

    quat is wxyz world<-base (e.g. the bag's median IMU sample); None
    keeps the model's declared orientation. height puts the base well
    clear of the floor plane so nothing can touch it.
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    free = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(
            f"expected exactly one free joint in {xml_path}, "
            f"found {[j.name for j in free]}"
        )
    base = free[0].parent
    # Keyframes are sized for the free-joint layout; save them, drop the
    # free-joint block, and re-add after the joint is gone.
    keys = [(k.name, np.array(k.qpos), np.array(k.ctrl)) for k in spec.keys]
    for k in list(spec.keys):
        spec.delete(k)
    spec.delete(free[0])
    base.pos = np.array([0.0, 0.0, float(height)])
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        base.quat = q / np.linalg.norm(q)
    for name, qpos, ctrl in keys:
        spec.add_key(name=name, qpos=qpos[_NQ_FREE:], ctrl=ctrl)
    return spec.compile()


def median_quat(quat):
    """Component-wise median of near-constant wxyz samples, normalized.

    Good enough for a robot lying still on a rig; not a general quaternion
    average (sign flips or large motion would break it, neither happens in
    an air-mount recording).
    """
    q = np.median(np.asarray(quat, dtype=float), axis=0)
    return q / np.linalg.norm(q)
