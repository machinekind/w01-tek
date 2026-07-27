"""Build the MJX-ready Wojtek model from the original MJCF.

Reads the original wojtek.xml, applies the training edits from the
spec, and writes wojtek_mjx.xml plus scene_mjx.xml next to it. The
original files stay untouched. Edits:
  - every mesh geom stops colliding; feet get spheres, the base gets a
    box in two halves
  - the base gets an explicit inertial so the total mass hits a parameter
  - the 12 torque motors become PD position actuators (gain/bias overwrite,
    the same trick as 3_jaxpot_robotics/jaxpot_robotics/race_scene.py)
  - timestep 0.004, Newton solver with few iterations (MJX convention)
"""

import argparse

import mujoco
import numpy as np

from wojtek_rl import paths

# The real foot is a half-disc rubber pad bolted to the sixth-link tip, its
# arc centered on the fifth/sixth closure pivot, so a pivot-centered sphere
# of the pad's radius is the right contact model. Radius measured from the
# pad disc baked into sixth_link.stl (mesh extends 0.046 m past the pivot).
FOOT_RADIUS = 0.046
DEFAULT_KP = 20.0
# kd raised 0.5 -> 1.0 after fbb_v2: the policy buzzed at ~7 Hz, right at the
# underdamped PD resonance (sqrt(kp/I)/2pi with reflected inertia ~0.01).
DEFAULT_KD = 1.0
# Real robot measured at 14 kg on a scale (2026-07-16); supersedes the ~16 kg
# owner estimate from 2026-07-05 and the original 10.0 placeholder.
DEFAULT_TOTAL_MASS = 14.0
# Set at the 16 kg estimate to keep the torque-to-weight ratio that trained
# well at 10 kg / 6 Nm; kept at 9 for the measured 14 kg, which lands a bit
# above that ratio (0.64 vs 0.6 Nm/kg). MD80 drives allow 15 Nm; the jump
# task overrides this per-env (12 Nm) for the launch.
FORCERANGE = 9.0
# Base collision box half-sizes, eyeballed from the mesh footprint.
BASE_BOX_HALFSIZE = (0.17, 0.08, 0.05)
# The base collides as a chessboard of small boxes over the original box's
# footprint, not as one box. MJWarp caps a heightfield collision at
# mjMAXCONPAIR = 50 contacts per geom pair and drops the rest without an
# error. The count scales with footprint area over the 4 cm terrain cells:
# the full box measured 68-100 contacts lying on terrain and one x-half
# about 34, so even a two-way split sat within a factor of two of the cap.
# One chessboard cell covers at most a 2x2 patch of terrain cells and stays
# an order of magnitude under it. The outer faces of the original box are
# preserved, so the footprint, the camera placement and the inertial
# (computed from the FULL box) do not move. The gaps between cells are
# covered by the analytic base-contact termination (fall.on_base_contact),
# which reads the height lookup, not collision.
BASE_BOX_GRID = (6, 3)  # cells along x, y; (col + row) even cells are filled
# Keeps diagonally adjacent cell corners from both claiming a contact point.
BASE_BOX_CELL_SHRINK = 0.0005
BASE_BOX_NAMES = tuple(
    f"base_box_r{row}c{col}"
    for row in range(BASE_BOX_GRID[1])
    for col in range(BASE_BOX_GRID[0])
    if (col + row) % 2 == 0
)


def base_box_cells():
    """(name, (x, y) of the centre, half-sizes) for the chessboard cells.

    Iteration order matches BASE_BOX_NAMES.
    """
    hx, hy, hz = BASE_BOX_HALFSIZE
    nx, ny = BASE_BOX_GRID
    cell = (hx / nx - BASE_BOX_CELL_SHRINK, hy / ny - BASE_BOX_CELL_SHRINK, hz)
    cells = []
    for row in range(ny):
        for col in range(nx):
            if (col + row) % 2:
                continue
            x = -hx + (2 * col + 1) * hx / nx
            y = -hy + (2 * row + 1) * hy / ny
            cells.append((f"base_box_r{row}c{col}", (x, y), cell))
    return cells

# Onboard forward camera (the future VLM's eyes): just past the base box's
# front face, optical axis along body +x pitched ~15 deg down, wide FOV.
# xyaxes follow the MJCF convention (camera right, camera up).
EGO_CAM = dict(
    name="ego",
    # Ahead of the front legs and above the hips, else the view is mostly
    # the robot's own linkage (tuned by rendering, 2026-07-09).
    pos=[0.30, 0.0, 0.10],
    fovy=100.0,
    xyaxes=[0.0, -1.0, 0.0, 0.1736, 0.0, 0.9848],  # ~10 deg down
)

# Benchmark camera: matches the VLN-CE agent camera (1.25 m above the floor,
# level, 90 deg HFOV) so VLN-trained policies see a view like their training
# distribution instead of the ego cam's floor-level view. Rides an invisible
# mast on the trunk (z is relative to the base, which sits ~0.14 m up at the
# home keyframe). fovy is vertical: 90 deg HFOV at the 4:3 render = 73.74 deg.
BENCH_CAM = dict(
    name="bench",
    pos=[0.05, 0.0, 1.11],
    fovy=73.74,
    xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # level, facing body +x
)


def _xyaxes_to_quat(xyaxes) -> np.ndarray:
    """MJCF xyaxes (right, up) -> wxyz quaternion, for MjSpec cameras."""
    x = np.array(xyaxes[:3], dtype=float)
    y = np.array(xyaxes[3:], dtype=float)
    x /= np.linalg.norm(x)
    y -= x * (y @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat

# Leg collision primitives, sized from the link mesh AABBs. Needed for the
# getup task: without them a fallen robot's legs sink through the floor
# (only the feet and the base box collide). contype=2 / conaffinity=0 pairs
# them with the floor (conaffinity 15) and nothing else; the bar capsules
# stop at x=0.14 so they can never reach the foot sphere at the 0.21 pivot.
LEG_COLLISION = {
    "second_link": dict(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.035, 0, 0], pos=[0.055, 0, 0]
    ),
    "third_link": dict(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.028, 0, 0],
        fromto=[-0.12, 0, 0.03, 0.04, 0, 0.03],
    ),
    "fifth_link": dict(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.012, 0, 0],
        fromto=[0.02, 0, 0, 0.14, 0, 0],
    ),
    "sixth_link": dict(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.012, 0, 0],
        fromto=[0.02, 0, 0, 0.14, 0, 0],
    ),
}


SCENE_XML_TEXT = """<mujoco model="wojtek_mjx_scene">
  <include file="wojtek_mjx.xml"/>
  <statistic center="0 0 0.15" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01" material="groundplane" condim="3" conaffinity="15"/>
    <camera name="track" mode="trackcom" pos="0.9 -1.3 0.5" xyaxes="0.83 0.55 0 -0.15 0.23 0.96"/>
  </worldbody>
</mujoco>
"""


def build_spec(
    total_mass: float = DEFAULT_TOTAL_MASS,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(paths.SOURCE_XML))

    # Mass of everything that is not the base, read from the unmodified model.
    m0 = spec.compile()
    root_id = m0.body("root").id
    non_base_mass = m0.body_mass.sum() - m0.body_mass[root_id]

    # 1. Mesh geoms stop colliding. Visual geoms already have contype 0.
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0

    # 2. One contact sphere per foot, at the foot site on foot_link. Both
    # linkage branches converge at foot_link; it is the ground contact.
    for leg in paths.LEGS:
        site = spec.site(f"{leg}_foot")
        body = spec.body(f"{leg}_foot_link")
        # conaffinity=1 (not 15) so the leg capsules (contype=2) never pair
        # with the feet; feet still pair with the floor and each other.
        body.add_geom(
            name=f"{leg}_foot_sphere",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[FOOT_RADIUS, 0, 0],
            pos=site.pos.copy(),
            contype=1,
            conaffinity=1,
            group=3,
        )
        # The foot link is nearly massless. Contacts and an equality
        # constraint both act through it, so give it a small real mass.
        body.mass = 0.01
        body.inertia = [1e-6, 1e-6, 1e-6]
        body.explicitinertial = True
        # Global foot velocity, read by the env's feet_slip reward.
        spec.add_sensor(
            name=f"{leg}_foot_linvel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{leg}_foot_sphere",
        )

    # 2b. Floor-only collision primitives along each leg, for fallen poses.
    for leg in paths.LEGS:
        for link, kw in LEG_COLLISION.items():
            spec.body(f"{leg}_{link}").add_geom(
                name=f"{leg}_{link}_floor",
                contype=2,
                conaffinity=0,
                group=3,
                **kw,
            )

    # 3. Contact boxes for the base, one per filled chessboard cell.
    root = spec.body("root")
    for name, (x, y), size in base_box_cells():
        root.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=[x, y, 0],
            contype=1,
            conaffinity=15,
            group=3,
        )

    # 3b. Onboard cameras (physics-inert; rendered by room_app / eval).
    for cam in (EGO_CAM, BENCH_CAM):
        root.add_camera(
            name=cam["name"],
            pos=cam["pos"],
            fovy=cam["fovy"],
            quat=_xyaxes_to_quat(cam["xyaxes"]),
        )

    # 4. Explicit base inertial. Box formula over the FULL base box dims: the
    # split above is a collision detail and does not move any mass, so a half
    # box here would silently change the dynamics of every run.
    base_mass = float(total_mass - non_base_mass)
    if base_mass <= 0:
        raise ValueError(f"total_mass {total_mass} below link mass {non_base_mass}")
    lx, ly, lz = (2 * s for s in BASE_BOX_HALFSIZE)
    root.mass = base_mass
    root.ipos = [0, 0, 0]
    root.inertia = [
        base_mass / 12 * (ly**2 + lz**2),
        base_mass / 12 * (lx**2 + lz**2),
        base_mass / 12 * (lx**2 + ly**2),
    ]
    root.explicitinertial = True

    # 5. Motors -> PD position actuators (gain/bias overwrite, see race_scene.py).
    for act in spec.actuators:
        joint = spec.joint(act.target)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0] = 0.0
        act.biasprm[1] = -kp
        act.biasprm[2] = -kd
        act.ctrlrange = joint.range.copy()
        act.forcerange = [-FORCERANGE, FORCERANGE]

    # 6. Integration options, MJX-style.
    spec.option.timestep = 0.004
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # iterations=1 clears the GPU step-rate gate but diverges in training:
    # under a flailing exploration policy the single Newton iteration goes
    # NaN and poisons the whole batch (fbb_v1, dead by 23M steps). Solver
    # robustness beats the gate heuristic; 2 iterations trains stably.
    spec.option.iterations = 2
    spec.option.ls_iterations = 5

    return spec


# Chosen in Task 3 via pose_explorer. (first, second, third) per leg.
# Settled standing height 0.109 m, the design height per the controller's
# gait code (foot z0 = -0.15 below the hip frame).
STAND_POSE = {
    "rear_left": (0.0, -0.2, 3.1),
    "rear_right": (0.0, -0.2, 3.1),
    "front_right": (0.0, -0.2, 3.1),
    "front_left": (0.0, -0.2, 3.1),
}


def stand_targets() -> np.ndarray:
    return np.array([v for leg in paths.LEGS for v in STAND_POSE[leg]])


def settle_home(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Settle under PD hold and return (qpos, ctrl) for the home keyframe."""
    data = mujoco.MjData(model)
    targets = stand_targets()
    qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    data.qpos[2] = 0.25
    for adr, t in zip(qadr, targets):
        data.qpos[adr] = t
    data.ctrl[:] = targets
    mujoco.mj_forward(model, data)
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    qpos = data.qpos.copy()
    qpos[0:2] = 0.0  # recenter
    qpos[3:7] = [1, 0, 0, 0]  # level the base orientation
    return qpos, targets


def write_models(
    total_mass: float = DEFAULT_TOTAL_MASS,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
    home_qpos: np.ndarray | None = None,
    home_ctrl: np.ndarray | None = None,
) -> None:
    spec = build_spec(total_mass=total_mass, kp=kp, kd=kd)
    if home_qpos is not None:
        spec.add_key(name="home", qpos=home_qpos, ctrl=home_ctrl)
    spec.compile()
    paths.ROBOT_XML.write_text(spec.to_xml())
    paths.SCENE_XML.write_text(SCENE_XML_TEXT)
    print(f"wrote {paths.ROBOT_XML}")
    print(f"wrote {paths.SCENE_XML}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-mass", type=float, default=DEFAULT_TOTAL_MASS)
    p.add_argument("--kp", type=float, default=DEFAULT_KP)
    p.add_argument("--kd", type=float, default=DEFAULT_KD)
    args = p.parse_args()
    # First pass without keyframe: settle_home needs a floor, which lives in
    # the scene file, so generate, settle against the scene, then regenerate.
    write_models(total_mass=args.total_mass, kp=args.kp, kd=args.kd)
    model = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    qpos, ctrl = settle_home(model)
    write_models(
        total_mass=args.total_mass,
        kp=args.kp,
        kd=args.kd,
        home_qpos=qpos,
        home_ctrl=ctrl,
    )


if __name__ == "__main__":
    main()
