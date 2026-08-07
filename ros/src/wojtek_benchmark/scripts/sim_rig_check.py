#!/usr/bin/env python3
"""Headless end-to-end check of the sim benchmark rig.

Builds the sim scene with the rig injected, renders the rig camera, runs
the real detector on the pixels, calibrates the world frame from the three
floor tags, and compares the tracked robot-tag pose against MuJoCo's exact
ground truth.  No ROS: this is the geometry pipeline in isolation, and the
numbers it prints are the rig's error floor under ideal imaging.

    MUJOCO_GL=cgl scripts/sim_rig_check.py            # macOS headless
    MUJOCO_GL=egl scripts/sim_rig_check.py            # Linux headless
    scripts/sim_rig_check.py --yaw 40 --xy 0.4 -0.3   # move the robot
    scripts/sim_rig_check.py --save render.png

Needs: mujoco, numpy, pupil-apriltags, pyyaml.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wojtek_benchmark import png, sim_rig, tracker  # noqa: E402

SCENE_XML = (
    PACKAGE_ROOT.parent / "wojtek_description" / "mujoco" / "scene_mjx.xml"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=Path, default=SCENE_XML)
    ap.add_argument("--xy", type=float, nargs=2, default=[0.0, 0.0],
                    help="robot base x y in the sim world")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="robot base yaw in degrees")
    ap.add_argument("--save", type=Path, default=None,
                    help="write the rig-camera render to this PNG")
    ap.add_argument("--pos-tol-mm", type=float, default=20.0)
    ap.add_argument("--yaw-tol-deg", type=float, default=2.0)
    args = ap.parse_args()

    import mujoco

    rig_cfg = sim_rig.load_rig_config()
    tags_cfg = sim_rig.load_tags_config()
    cam_cfg = rig_cfg["camera"]

    model = sim_rig.load_model_with_rig(args.scene, rig_cfg, tags_cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    data.qpos[0:2] = args.xy
    half = math.radians(args.yaw) / 2.0
    data.qpos[3:7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
    mujoco.mj_forward(model, data)

    w, h = cam_cfg["width"], cam_cfg["height"]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=sim_rig.CAMERA_NAME)
    rgb = renderer.render()
    renderer.close()

    if args.save:
        args.save.write_bytes(png.write_rgb([
            [tuple(px) for px in row] for row in rgb
        ]))
        print(f"render saved to {args.save}")

    gray = (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    fy = 0.5 * h / math.tan(math.radians(cam_cfg["fovy_deg"]) / 2.0)
    intrinsics = (fy, fy, (w - 1) / 2.0, (h - 1) / 2.0)
    sizes_by_id = {t["id"]: t["size_m"] for t in tags_cfg.values()}
    roles_by_id = {t["id"]: role for role, t in tags_cfg.items()}

    dets = tracker.detect(tracker.make_detector(), gray, intrinsics, sizes_by_id)
    for tag_id, d in sorted(dets.items()):
        print(f"detected id {tag_id} ({roles_by_id[tag_id]}): "
              f"margin {d.decision_margin:.1f}, cam-frame t {np.round(d.t, 4)}")
    missing = set(sizes_by_id) - set(dets)
    if missing:
        print(f"FAIL: tags not detected: {sorted(missing)}")
        return 1

    # Calibrate: camera -> bench world from the floor tags.
    centers = {roles_by_id[i]: dets[i].t for i in dets
               if roles_by_id[i] != "robot"}
    T_cam_world = tracker.solve_world_from_floor(
        centers, sim_rig.leg_lengths(rig_cfg)
    )

    # Tracked robot-tag pose in the bench world.
    d = dets[tags_cfg["robot"]["id"]]
    T_world_tag = tracker.inv_se3(T_cam_world) @ tracker.se3(d.R, d.t)

    # Exact ground truth from the same MjData that was rendered: the tag
    # geom's world pose, mapped into the bench world frame.
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "benchmark_robot")
    T_sim_geom = tracker.se3(
        data.geom_xmat[gid].reshape(3, 3), data.geom_xpos[gid]
    )
    T_world_geom_gt = tracker.inv_se3(sim_rig.world_frame_in_sim(rig_cfg)) @ T_sim_geom

    pos_err_mm = 1000.0 * np.linalg.norm(T_world_tag[:3, 3] - T_world_geom_gt[:3, 3])
    # Map the detected tag frame onto the physical surface frame; what
    # remains vs ground truth is pure estimation error.
    R_est_geom = T_world_tag[:3, :3] @ tracker.TAG_TO_SURFACE
    R_err = T_world_geom_gt[:3, :3].T @ R_est_geom
    ang_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    yaw_err = tracker.yaw_deg(tracker.se3(R_err, [0, 0, 0]))
    print(f"\ncalibration: legs & L angle passed")
    print(f"robot tag position error: {pos_err_mm:.2f} mm")
    print(f"robot tag rotation error: {ang_err:.3f} deg (yaw {yaw_err:.3f} deg)")

    fails = []
    if pos_err_mm > args.pos_tol_mm:
        fails.append(f"position error {pos_err_mm:.1f} mm > {args.pos_tol_mm} mm")
    if abs(yaw_err) > args.yaw_tol_deg:
        fails.append(f"yaw error {yaw_err:.2f} deg > {args.yaw_tol_deg} deg")
    for f in fails:
        print(f"FAIL: {f}")
    print("" if fails else "\nPASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
