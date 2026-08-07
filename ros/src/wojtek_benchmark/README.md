# wojtek_benchmark

Ground-truth instrumentation for benchmarking policies on the real robot:
motion-capture-lite built from four AprilTags and one external camera.
Three floor tags in an L fix the world frame and its scale; one tag on the
robot gives its pose in that frame; benchmark runs are recorded and scored
against the same course scenarios as the simulation battery
(`./training/run.sh courses`), so sim and real numbers are directly
comparable.

**Status:** the printable-tag pipeline, the tracker (detection →
calibration → world-frame pose), and the full simulation rig are
implemented.  Still missing: the course scorer and the robot tag's
`apriltag_link` in `wojtek_description` (its mount pose lives in
`sim_rig.yaml` until the physical mount is measured).

## One source of truth

An AprilTag encodes nothing but an ID.  Everything semantic — which ID is
the world origin, how big each tag is, how long the measured L legs are —
lives in [config/apriltags.yaml](config/apriltags.yaml), and every consumer
(this package's generator today; calibration, `apriltag_ros` config, and
scoring later) must read it from there.

| id | role         | black edge | where |
|----|--------------|------------|-------|
| 0  | world_origin | 160 mm     | floor, the L corner; world origin, z = floor |
| 1  | world_x      | 160 mm     | floor, along +x from the origin tag |
| 2  | world_y      | 160 mm     | floor, along +y from the origin tag |
| 10 | robot        | 120 mm     | mounted on the robot (transform → URDF, pending) |

Three floor tags instead of one because a single tag's orientation error
grows linearly with distance across the arena, and because the two
tape-measured legs (`distance_from_origin_m`) give calibration an
independent check with a physical unit: if the vision-estimated leg length
disagrees with the tape measure, the calibration must refuse to proceed.

## Printing

The print-ready sheets are committed in [tags/](tags/) — grab a PDF and
print it, no dev environment needed.

1. Print at **100 % scale** (no "fit to page" — that is precisely the
   failure the sheet is designed to catch).
2. Check the 100 mm bar on the sheet with a ruler.  If it is not 100 mm,
   the print was scaled: reprint, or set `size_m` in the yaml to the black
   edge you actually measure and regenerate.
3. Measure the black square's edge and confirm it matches the size printed
   on the sheet.  `size_m` is what turns detected pixels into meters — an
   unnoticed 96 % printer scale becomes a 4 % error in every ground-truth
   position.
4. Mount flat and rigid (foam board or similar).  Curvature bends the pose
   estimate; the floor tags must also stay put for a whole session, since
   moving one silently shifts the world frame.

## Laying out the course

1. Place the origin tag, then the x and y tags so the three centers form an
   L (right angle at the origin, legs of roughly 1.5–2 m).  The gray tick
   marks outside each tag's square point at its center — use them to align
   the tape measure.
2. Tape-measure both center-to-center legs and write them into
   `distance_from_origin_m` in the yaml.  They start as `null` on purpose;
   calibration must refuse to run on `null`.
3. The camera must see all three floor tags at a reasonable angle — floor
   tags viewed at grazing incidence detect poorly and condition the pose
   badly.  A tripod at 1.5–2 m looking down at 30–45°, or overhead if the
   room allows, both work.

## Regenerating the sheets

```bash
ros/src/wojtek_benchmark/scripts/generate_tags.py
```

`tags/*.pdf` are generated artifacts under the same contract as the
generated MJCF models: committed, regenerated after every yaml edit, never
hand-edited.  Generation is byte-deterministic (the PDF writer embeds no
timestamps or library metadata), so `verify.sh` T0 and this package's tests
diff the committed sheets against the yaml — a stale regeneration fails the
static gate rather than drifting silently.

The rendering itself is pinned to the canonical family: unit tests compare
the bitmaps against fixtures extracted from the official
[apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs) PNGs.
Orientation is part of that contract — a rotated rendering would still
detect, but every ground-truth yaw would be silently wrong.

```bash
python3 -m pytest ros/src/wojtek_benchmark/test/ -q
```

## The whole rig in simulation

The sim is where the rig proves itself: MuJoCo's state is *perfect* ground
truth, so the difference between the tracked pose and `/sim/qpos` is the
end-to-end error of detection + calibration + tracking.  Measured on the
current config: **3–7 mm position, < 0.05° yaw** across robot poses, under
ideal imaging — the rig's error floor.  Real-world scores sit on top of
optics, print quality, and placement.

Nothing edits the generated scene XML: `sim_rig.py` injects the rig into a
loaded `MjSpec` — the same physics-neutral pattern as the sim's virtual
D435 — using textures rendered from the same `tag36h11` bitmaps as the
printable PDFs.  [config/sim_rig.yaml](config/sim_rig.yaml) plays the role
reality assigns to the tape measure and the tripod: floor-tag placements
(leg lengths are *derived* from them, then fed to the tracker's refusal
gate), the robot tag's mount pose, and the camera.

Run it (container, two shells):

```bash
ros2 launch wojtek_pc sim.launch.py            # the robot, hw:=mujoco
ros2 launch wojtek_benchmark sim_rig.launch.py # camera + tracker + monitor
```

Then drive the robot from the web console and watch
`/benchmark/pose_error_mm` and `/benchmark/yaw_error_deg`.  The tracker
node is the deployment tracker: on the real course, point `image_topic` /
`info_topic` at the webcam driver and drop the monitor (there is no ground
truth in reality — that's why the rig exists).

Headless, no ROS (this is also the CI-able geometry check):

```bash
MUJOCO_GL=cgl ros/src/wojtek_benchmark/scripts/sim_rig_check.py   # macOS
MUJOCO_GL=egl ros/src/wojtek_benchmark/scripts/sim_rig_check.py   # Linux
```

It renders the rig camera, detects, calibrates, and compares against
ground truth, failing on >20 mm / >2° — try `--xy 0.4 -0.3 --yaw 40` to
move the robot, `--save render.png` to look through the rig camera.
Needs `pip install mujoco pupil-apriltags pyyaml numpy` (the ROS container
image already has them).

Sim-vs-print contract in one line: the pixels the sim camera sees and the
ink on the printed sheets come from the same verified bitmaps, so a
sign/orientation bug cannot pass in sim and fail on paper.
