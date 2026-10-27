"""Build the MJX-ready Wojtek model from the original MJCF.

Reads the original wojtek.xml, applies the training edits from the
spec, and writes wojtek_mjx.xml plus scene_mjx.xml next to it. The
original files stay untouched. `--robot` builds another robot variant
(robots.py) from its own source file into its own directory. Edits:
  - every mesh geom stops colliding; feet get spheres, the base gets a
    box in two halves
  - the base gets an explicit inertial so the total mass hits a parameter
  - the 12 torque motors become PD position actuators (gain/bias overwrite,
    the same trick as the earlier prototype's race scene)
  - timestep 0.004, Newton solver with few iterations (MJX convention)
"""

import argparse

import mujoco
import numpy as np

from wojtek_rl import paths, robots

# The stock robot's numbers, under the names the rest of the code and the
# tests know them by. They live in robots.WOJTEK, next to the other variants.
FOOT_RADIUS = robots.WOJTEK.foot_radius
DEFAULT_KP = robots.WOJTEK.kp
DEFAULT_KD = robots.WOJTEK.kd
DEFAULT_TOTAL_MASS = robots.WOJTEK.total_mass
FORCERANGE = robots.WOJTEK.forcerange
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

# Leg collision primitives (robots.Robot.leg_collision). Needed for the
# getup task: without them a fallen robot's legs sink through the floor
# (only the feet and the base box collide). contype=2 / conaffinity=0 pairs
# them with the floor (conaffinity 15) and nothing else.
_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


# The scene sits next to the robot file it includes, in every variant's
# directory, so the include stays a bare file name.
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


def _rest_at_first_key(spec: mujoco.MjSpec) -> None:
    """Move the model's rest pose to its first keyframe.

    Each hinge gets `ref` = its keyframe angle, and its body is turned by
    that angle about the joint axis, so the pose drawn in the XML is the
    keyframe pose and qpos means what it meant before. The base is lifted
    to the keyframe height.
    """
    model = spec.compile()
    qpos = model.key_qpos[0]
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if np.any(np.asarray(joint.pos) != 0):
            raise ValueError(f"{joint.name}: a hinge off its body origin is not handled")
        angle = float(qpos[model.joint(joint.name).qposadr[0]])
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.asarray(joint.axis, dtype=float), angle)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.asarray(joint.parent.quat, dtype=float), turn)
        joint.parent.quat = quat
        joint.ref = angle
    root = spec.body("root")
    root.pos = [root.pos[0], root.pos[1], float(qpos[2])]


def build_spec(
    total_mass: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
    robot: str = paths.DEFAULT_ROBOT,
) -> mujoco.MjSpec:
    """The MJX-ready spec of one robot variant. Mass and gains left at None
    take the variant's own values."""
    rb = robots.get(robot)
    total_mass = rb.total_mass if total_mass is None else total_mass
    kp = rb.kp if kp is None else kp
    kd = rb.kd if kd is None else kd
    spec = mujoco.MjSpec.from_file(str(paths.robot_files(robot)["source"]))
    if rb.rest_pose_from_source_key:
        _rest_at_first_key(spec)
    # A CAD export brings its own keyframe; the only keyframe of a built
    # model is the settled `home`.
    for key in list(spec.keys):
        spec.delete(key)

    # Mass of everything that is not the base, read from the unmodified model.
    m0 = spec.compile()
    root_id = m0.body("root").id
    non_base_mass = m0.body_mass.sum() - m0.body_mass[root_id]

    # 1. Mesh geoms stop colliding. Visual geoms already have contype 0.
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0

    # 2. One contact sphere per foot, at the foot site. On the stock robot
    # both linkage branches converge at foot_link and that is the ground
    # contact; a variant whose loop closes elsewhere names the body that
    # carries the pad (robots.Robot.foot_body).
    for leg in paths.LEGS:
        site = spec.site(f"{leg}_foot")
        body = spec.body(f"{leg}_{rb.foot_body}")
        if site.parent.name != body.name:
            raise ValueError(
                f"site {site.name} sits on {site.parent.name}, not on the "
                f"foot body {body.name}; its position would be read in the "
                "wrong frame"
            )
        # conaffinity=1 (not 15) so the leg capsules (contype=2) never pair
        # with the feet; feet still pair with the floor and each other.
        body.add_geom(
            name=f"{leg}_foot_sphere",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[rb.foot_radius, 0, 0],
            pos=np.asarray(site.pos) + rb.foot_offset,
            contype=1,
            conaffinity=1,
            group=3,
        )
        # The closure link is nearly massless and an equality constraint
        # acts through it (on the stock robot the contact does too), so give
        # it a small real mass.
        closure = spec.body(f"{leg}_foot_link")
        closure.mass = 0.01
        closure.inertia = [1e-6, 1e-6, 1e-6]
        closure.explicitinertial = True
        # Global foot velocity, read by the env's feet_slip reward.
        spec.add_sensor(
            name=f"{leg}_foot_linvel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{leg}_foot_sphere",
        )

    # 2b. Floor-only collision primitives along each leg, for fallen poses.
    for leg in paths.LEGS:
        # Left and right legs of a variant can be mirror images in the
        # link's y (the shin of legs_v627 is bent sideways towards its pad).
        # The foot site says which way: the primitives are written for a leg
        # whose site has y >= 0 and are mirrored for the others.
        mirror = -1.0 if spec.site(f"{leg}_foot").pos[1] < 0 else 1.0
        for link, kw in rb.leg_collision.items():
            kw = dict(kw, type=_GEOM_TYPES[kw["type"]])
            for field in ("pos", "fromto"):
                if field in kw:
                    v = np.array(kw[field], dtype=float)
                    v[1::3] *= mirror
                    kw[field] = v.tolist()
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
        act.forcerange = [-rb.forcerange, rb.forcerange]

    # 6. Integration options, MJX-style. The iteration counts are per
    # variant; robots.py says why they are what they are.
    spec.option.timestep = rb.timestep
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = rb.iterations
    spec.option.ls_iterations = rb.ls_iterations
    for suffix, frictionloss in rb.joint_frictionloss.items():
        for leg in paths.LEGS:
            spec.joint(f"{leg}_{suffix}").frictionloss = frictionloss
    if rb.closure_solimp is not None:
        for eq in spec.equalities:
            eq.solimp[:2] = rb.closure_solimp

    return spec


# (first, second, third) per leg of the stock robot; see robots.WOJTEK.
STAND_POSE = dict(zip(paths.LEGS, robots.WOJTEK.stand_pose))


def stand_targets(robot: str = paths.DEFAULT_ROBOT) -> np.ndarray:
    return np.array(robots.get(robot).stand_pose, dtype=float).ravel()


def settle_home(
    model: mujoco.MjModel,
    robot: str = paths.DEFAULT_ROBOT,
    start_qpos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle under PD hold and return (qpos, ctrl) for the home keyframe.

    `start_qpos` is a full qpos with the loops already closed (a CAD
    keyframe). Without it the passive joints start at zero and the equality
    constraints pull the loops shut during the settle, which works on the
    stock legs and is not guaranteed to land on the right branch elsewhere.
    """
    data = mujoco.MjData(model)
    targets = stand_targets(robot)
    qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    if start_qpos is not None:
        data.qpos[:] = start_qpos
    data.qpos[2] = robots.get(robot).settle_drop_height
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
    total_mass: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
    home_qpos: np.ndarray | None = None,
    home_ctrl: np.ndarray | None = None,
    robot: str = paths.DEFAULT_ROBOT,
) -> None:
    spec = build_spec(total_mass=total_mass, kp=kp, kd=kd, robot=robot)
    if home_qpos is not None:
        spec.add_key(name="home", qpos=home_qpos, ctrl=home_ctrl)
    spec.compile()
    files = paths.robot_files(robot)
    files["robot"].write_text(spec.to_xml())
    files["scene"].write_text(SCENE_XML_TEXT)
    print(f"wrote {files['robot']}")
    print(f"wrote {files['scene']}")


def write_view(robot: str, home_qpos: np.ndarray, home_ctrl: np.ndarray, **kw) -> None:
    """A copy of the built model for looking at in a viewer.

    A viewer opens a model with every control at zero, and a zero target is
    far outside these legs' joint ranges. In this copy a control is an
    offset from the home target, so the robot holds its stand untouched and
    the viewer's sliders bend the joints around it. Nothing trains on it.
    """
    spec = build_spec(robot=robot, **kw)
    for act, home in zip(spec.actuators, home_ctrl):
        act.biasprm[0] = act.gainprm[0] * home
        act.ctrlrange = np.asarray(act.ctrlrange) - home
    spec.add_key(name="home", qpos=home_qpos, ctrl=np.zeros(len(home_ctrl)))
    spec.compile()
    files = paths.robot_files(robot)
    files["view_robot"].write_text(spec.to_xml())
    files["view_scene"].write_text(
        SCENE_XML_TEXT.replace(files["robot"].name, files["view_robot"].name)
    )
    print(f"wrote {files['view_scene']}")


def source_stand_qpos(robot: str) -> np.ndarray | None:
    """The first keyframe of the variant's source model, if it has one."""
    source = mujoco.MjModel.from_xml_path(str(paths.robot_files(robot)["source"]))
    return source.key_qpos[0].copy() if source.nkey else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default=paths.DEFAULT_ROBOT, choices=robots.NAMES)
    p.add_argument("--total-mass", type=float, default=None,
                   help="default: the variant's own (robots.py)")
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--kd", type=float, default=None)
    args = p.parse_args()
    # First pass without keyframe: settle_home needs a floor, which lives in
    # the scene file, so generate, settle against the scene, then regenerate.
    kw = dict(total_mass=args.total_mass, kp=args.kp, kd=args.kd, robot=args.robot)
    write_models(**kw)
    model = mujoco.MjModel.from_xml_path(str(paths.robot_files(args.robot)["scene"]))
    qpos, ctrl = settle_home(
        model, robot=args.robot, start_qpos=source_stand_qpos(args.robot)
    )
    write_models(**kw, home_qpos=qpos, home_ctrl=ctrl)
    if robots.get(args.robot).rest_pose_from_source_key:
        kw.pop("robot")
        write_view(args.robot, qpos, ctrl, **kw)


if __name__ == "__main__":
    main()
