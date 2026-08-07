"""Inject the AprilTag benchmark rig into the simulation model.

Mirrors the physics-neutral MjSpec injection pattern of
wojtek_pc.depth_camera.inject_camera: the generated scene XML is never
edited; the rig -- three floor tags in an L, one tag on the robot's back,
and a fixed "tripod" camera -- is added to a loaded MjSpec at load time.
All tag geoms are visual-only (contype/conaffinity 0, mass 0), so the
plant that ros2_control steps and the model this renders stay one physics.

Tag textures are rendered from the same wojtek_benchmark.tag36h11 bitmaps
as the committed printable PDFs, so what the sim camera sees is provably
the tag family the real camera will see.  Tag semantics (ids, roles,
physical sizes) come from config/apriltags.yaml; where the rig sits in the
sim world comes from config/sim_rig.yaml.

Deliberately ROS-free (like tag36h11/generate_tags): the headless check
script drives it with nothing but mujoco + numpy installed.
"""

import tempfile
from pathlib import Path

import numpy as np
import yaml

from . import png, tag36h11

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _default_config(name):
    """Config path that works from the source tree AND a colcon install.

    In the source tree config/ sits next to the module's parent; in an
    install it lives in the package share directory instead (parents[1] is
    site-packages there, which is why a naive relative default 404s).
    """
    src = PACKAGE_ROOT / "config" / name
    if src.exists():
        return src
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("wojtek_benchmark")) / "config" / name


DEFAULT_RIG_CONFIG = _default_config("sim_rig.yaml")
DEFAULT_TAGS_CONFIG = _default_config("apriltags.yaml")

CAMERA_NAME = "benchmark_rig_camera"
FLOOR_ROLES = ("world_origin", "world_x", "world_y")

# One texture cell per tag cell would alias at render time; 64 px per cell
# keeps edges crisp at 720p from tripod distance.
PX_PER_CELL = 64

# The tag geom spans the full printed square (black border + the one-cell
# white quiet zone the bitmap already includes): black edge * 10/8.
TOTAL_OVER_BLACK = tag36h11.TOTAL_WIDTH / tag36h11.WIDTH_AT_BORDER

# Thin box, not a plane: planes must live in the worldbody, and the robot
# tag rides a moving body. 1 mm half-thickness keeps it clear of z-fighting.
TAG_HALF_THICKNESS = 0.001


def load_rig_config(path=DEFAULT_RIG_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    for role in FLOOR_ROLES:
        if role not in cfg["floor_tags"]:
            raise ValueError(f"sim_rig.yaml floor_tags missing role {role!r}")
    return cfg


def load_tags_config(path=DEFAULT_TAGS_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["family"] != tag36h11.FAMILY:
        raise ValueError(f"family must be {tag36h11.FAMILY}")
    return {t["role"]: t for t in cfg["tags"]}


def leg_lengths(rig_cfg):
    """(leg_x_m, leg_y_m) center-to-center, the sim's 'tape measure'."""
    o = np.array(rig_cfg["floor_tags"]["world_origin"]["center_xy"], dtype=float)
    x = np.array(rig_cfg["floor_tags"]["world_x"]["center_xy"], dtype=float)
    y = np.array(rig_cfg["floor_tags"]["world_y"]["center_xy"], dtype=float)
    return float(np.linalg.norm(x - o)), float(np.linalg.norm(y - o))


def world_frame_in_sim(rig_cfg):
    """4x4 T_sim_benchworld: origin at the origin tag center (on the floor),
    x toward the world_x tag, y toward the world_y tag, z up.

    The same construction the tracker performs from detected tag centers,
    evaluated on the configured placement -- this is what makes the error
    monitor's comparison exact rather than approximate.
    """
    o = np.array(rig_cfg["floor_tags"]["world_origin"]["center_xy"] + [0.0])
    x = np.array(rig_cfg["floor_tags"]["world_x"]["center_xy"] + [0.0])
    y = np.array(rig_cfg["floor_tags"]["world_y"]["center_xy"] + [0.0])
    ex = x - o
    ex /= np.linalg.norm(ex)
    ey = y - o
    ey -= ex * (ex @ ey)
    ey /= np.linalg.norm(ey)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2] = ex, ey, np.cross(ex, ey)
    t[:3, 3] = o
    return t


def tag_texture_rows(tag_id, px_per_cell=PX_PER_CELL):
    """Nearest-neighbor upscale of the canonical bitmap to RGB rows."""
    grid = tag36h11.render_bitmap(tag_id)
    rows = []
    for cell_row in grid:
        row = []
        for white in cell_row:
            row.extend([(255, 255, 255) if white else (0, 0, 0)] * px_per_cell)
        rows.extend([row] * px_per_cell)
    return rows


def _yaw_quat(yaw_rad):
    return [float(np.cos(yaw_rad / 2)), 0.0, 0.0, float(np.sin(yaw_rad / 2))]


def _rpy_quat(rpy_deg):
    """wxyz quaternion from roll/pitch/yaw degrees, extrinsic x-y-z."""
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float))
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return [
        float(cy * cp * cr + sy * sp * sr),
        float(cy * cp * sr - sy * sp * cr),
        float(cy * sp * cr + sy * cp * sr),
        float(sy * cp * cr - cy * sp * sr),
    ]


def lookat_quat(pos, target, mujoco):
    """wxyz quaternion for an MJCF camera at pos looking at target.

    MJCF cameras look along their -z with x right, y up.  Up is chosen as
    close to world +z as the look direction allows (a tripod, not a drone).
    """
    forward = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    forward /= np.linalg.norm(forward)
    z = -forward
    x = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("camera looking straight down: up direction ambiguous")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def _free_root_body(spec, mujoco):
    for body in spec.bodies:
        for jnt in body.joints:
            if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                return body
    raise ValueError("model has no free-jointed body to mount the robot tag on")


def _add_tag_assets(spec, role, tag, texture_dir, mujoco):
    """Texture + material for one tag; returns the material name."""
    tex_path = Path(texture_dir) / f"benchmark_{role}.png"
    tex_path.write_bytes(png.write_rgb(tag_texture_rows(tag["id"])))
    tex = spec.add_texture()
    tex.name = f"benchmark_{role}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.file = str(tex_path)
    mat = spec.add_material()
    mat.name = f"benchmark_{role}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    # Flat response: no specular highlight may wash out the black cells, and
    # emission lifts the texture toward light-independence. Detection limits
    # under adversarial lighting are a real-world question, not a sim one.
    mat.specular = 0.0
    mat.shininess = 0.0
    mat.emission = 0.35
    return mat.name


def _add_tag_geom(body, name, material, half_edge, pos, quat, mujoco):
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [half_edge, half_edge, TAG_HALF_THICKNESS]
    geom.pos = list(pos)
    geom.quat = list(quat)
    geom.material = material
    geom.contype = 0
    geom.conaffinity = 0
    geom.mass = 0.0
    return geom


def inject_rig(spec, rig_cfg=None, tags_cfg=None, texture_dir=None):
    """Add tags + rig camera to a loaded MjSpec.  Returns the camera name.

    texture_dir must outlive spec.compile() (textures load at compile);
    default is a fresh mkdtemp that lives until process exit.
    """
    import mujoco

    rig_cfg = rig_cfg or load_rig_config()
    tags_cfg = tags_cfg or load_tags_config()
    texture_dir = texture_dir or tempfile.mkdtemp(prefix="wojtek_benchmark_tex_")

    for role in FLOOR_ROLES:
        tag = tags_cfg[role]
        placement = rig_cfg["floor_tags"][role]
        mat = _add_tag_assets(spec, role, tag, texture_dir, mujoco)
        half_edge = tag["size_m"] * TOTAL_OVER_BLACK / 2.0
        cx, cy = placement["center_xy"]
        _add_tag_geom(
            spec.worldbody, f"benchmark_{role}", mat, half_edge,
            # Sitting on the floor plane: center one half-thickness up.
            [cx, cy, TAG_HALF_THICKNESS],
            _yaw_quat(np.radians(placement.get("yaw_deg", 0.0))), mujoco,
        )

    robot = tags_cfg["robot"]
    mount = rig_cfg["robot_tag"]
    mat = _add_tag_assets(spec, "robot", robot, texture_dir, mujoco)
    _add_tag_geom(
        _free_root_body(spec, mujoco), "benchmark_robot", mat,
        robot["size_m"] * TOTAL_OVER_BLACK / 2.0,
        mount["pos"], _rpy_quat(mount.get("rpy_deg", [0.0, 0.0, 0.0])), mujoco,
    )

    cam_cfg = rig_cfg["camera"]
    cam = spec.worldbody.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = list(cam_cfg["pos"])
    cam.quat = lookat_quat(cam_cfg["pos"], cam_cfg["look_at"], mujoco)
    cam.fovy = float(cam_cfg["fovy_deg"])
    return CAMERA_NAME


def load_model_with_rig(xml_path, rig_cfg=None, tags_cfg=None):
    """Compile xml_path with the benchmark rig injected."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))
    inject_rig(spec, rig_cfg=rig_cfg, tags_cfg=tags_cfg)
    return spec.compile()
