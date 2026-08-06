# wojtek_benchmark

Ground-truth instrumentation for benchmarking policies on the real robot:
motion-capture-lite built from four AprilTags and one external camera.
Three floor tags in an L fix the world frame and its scale; one tag on the
robot gives its pose in that frame; benchmark runs are recorded and scored
against the same course scenarios as the simulation battery
(`./training/run.sh courses`), so sim and real numbers are directly
comparable.

**Status:** the printable-tag pipeline is implemented (this page).  The
camera→world calibration, the live tracker, the scorer, and the robot tag's
`apriltag_link` in `wojtek_description` are not written yet — the tags can
be printed and the course laid out before any of that exists, which is why
this slice ships first.

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
| 10 | robot        | 80 mm      | mounted on the robot (transform → URDF, pending) |

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
