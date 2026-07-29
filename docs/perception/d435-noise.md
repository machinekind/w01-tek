# RealSense D435: measured depth noise, and what it means for the stack

Bench measurements on the unit that will go on the robot, serial
`030522071460`, firmware `5.17.0.10`. These are the numbers behind
`ros/src/wojtek_perception_bringup/config/d435.yaml`; change that config and
this document is the thing to re-measure against.

Measured on a PC, camera static, indoor scene at 2-4.5 m (a ceiling with
ductwork). That scene is close to worst case for stereo -- flat painted
surfaces with little texture and fluorescent tubes saturating the IR
images -- so real terrain at 0.5-2 m will do better than the far bins here.

## The unit

| | |
|---|---|
| depth intrinsics @848x480 | fx = fy = 418.0, ppx 427.3, ppy 237.8 |
| depth FOV | 90.8 deg H x 59.7 deg V |
| stereo baseline | 50.13 mm |
| depth unit | 1.000 mm |
| min-Z @848x480 | ~0.28 m (structural blind zone) |

## Noise vs. range

Temporal standard deviation per pixel over 60-80 frames, binned by each
pixel's own median range.

| preset / filter | fill | 1-2 m | 2-3 m | 3-4 m | 4-5 m |
|---|---|---|---|---|---|
| `default` | 95% | 14.5 mm | 39 mm | 97 mm | 195 mm |
| `default` + temporal (delta 20, the librealsense default) | 96% | 12.8 | 41 | **104** | 201 |
| `default` + temporal delta 100 | 97% | 6.4 | 28 | 99 | 202 |
| `high_accuracy` | 60-72% | 12 | 17 | 33-43 | 75-78 |
| **`high_accuracy` + temporal delta 100** | 65-75% | 5.8 | **10** | **21-37** | 74-83 |
| `high_density` + temporal + laser max | 95% | 6.8 | 25 | 79 | 166 |
| `default` + laser 360 mW | 93% | 15.8 | 37 | 80 | 160 |

Two independent runs; the spread is shown where they differed.

**The preset is worth 3x.** Implied subpixel disparity accuracy is 0.207 px
on `default` and ~0.042 px on `high_accuracy` -- the latter is inside
Intel's own 0.05-0.1 px spec, so the sensor is healthy and the default
preset is simply the wrong one for this use.

**The default temporal filter is worse than none at range.**
`filter_smooth_delta` is 20 (mm); frame-to-frame jumps at 4 m are ~200 mm,
so the filter reads real noise as motion and passes it straight through.
At 100 it engages.

**Laser power is not a lever.** 360 mW (max) buys ~15% on `default` and
nothing measurable on `high_accuracy`.

Fitted error model on this unit, `high_accuracy`: `sigma ~= 0.002 * z^2` m
(`default`: `0.0099 * z^2`). Feed the first one to anything that wants a
per-point variance.

## Spatial aggregation does not denoise

Reducing 848x480 to a coarse grid by patch median, `high_accuracy` +
temporal:

| grid | points | px/point | fill | 2-3 m | 3-4 m | 4-5 m |
|---|---|---|---|---|---|---|
| 848x480 | 407040 | 1 | 74% | 11.3 mm | 33.9 mm | 79.0 mm |
| 32x24 | 768 | 520 | 95% | 10.1 (1.1x) | 33.6 (1.0x) | 56.0 (1.4x) |
| 16x12 | 192 | 2120 | 98% | 10.5 (1.1x) | 21.2 (1.6x) | 38.4 (2.1x) |
| 10x10 | 100 | 4032 | 99% | 10.1 (1.1x) | 18.3 (1.9x) | 39.1 (2.0x) |
| 8x8 | 64 | 6360 | 100% | 9.7 (1.2x) | 18.9 (1.8x) | 41.2 (1.9x) |

Collapsing 6360 pixels into one point removes **1.8x** of the noise where an
uncorrelated sensor would give **80x**. RealSense stereo noise is strongly
correlated in space: whole patches wobble together, which is exactly what
"the depth waves at 4 m" looks like by eye. No amount of downsampling
flattens it.

What aggregation *does* buy is coverage: fill goes 74% -> 100%, because a
patch of thousands of pixels always contains something valid. That is why
the accurate-but-holey preset is the right upstream choice once a grid
reduction sits downstream.

The lever that does work on correlated noise is **temporal averaging from a
moving viewpoint**, since the correlated pattern is locked to the viewpoint,
not to the world. An accumulated map gets this for free; a single frame
never does.

## What actually limits the stack

Ranked by how much they move the outcome, not by how interesting they are.

**1. Camera pitch, by an order of magnitude.** Band classification uses
height, `z ~= r * sin(theta)`, so a pitch error `dtheta` costs `r * dtheta`:

| pitch error | at 2 m | at 3 m | at 4 m |
|---|---|---|---|
| 1 deg | 35 mm | 52 mm | 70 mm |
| 2 deg | 70 mm | 105 mm | 140 mm |
| 3 deg | 105 mm | 157 mm | 209 mm |

With `OBSTACLE_MIN_Z = 0.06`, about **1.15 deg of pitch error at 3 m is
enough for the floor to classify as an obstacle** -- a phantom wall across
the path. The same geometry gives depth noise only ~2.7 mm of influence on
that decision, because floor rays are near-grazing. Mitigations, in order:
classify against a ground plane fitted from the same depth frame rather than
against world z; take TF at the frame's timestamp, not "now"; calibrate the
mount.

**2. Angular resolution at grazing incidence.** For a camera at ~0.25 m
looking forward, the ground patch covered by one vertical grid cell is
`r^2 * dtheta / h`:

| grid | 1 m | 2 m | 3 m |
|---|---|---|---|
| 8x8 (7.5 deg/cell) | 0.52 m | 2.1 m | 4.7 m |
| 16x12 (5.0 deg) | 0.35 m | 1.4 m | 3.1 m |
| 32x24 (2.5 deg) | 0.17 m | 0.69 m | 1.6 m |

A stair tread is ~0.28 m. An 8x8 grid swallows a whole flight from 1.5 m
out and reports a ramp. This is geometry; no preset changes it. Stair
geometry has to come from the near field or from an accumulated map, which
the 0.28 m blind zone forces anyway.

**3. Sensor noise**, last. At `high_accuracy` it is 10 mm at 2-3 m against a
150-180 mm stair riser.

## Consequences for `wojtek_rl/scan/localmap.py`

The sim-side map is a 2D log-odds sliding window, `res = 0.05` m,
`extent_m = 8.0`, band `[0.06, 0.35]`, `inflate_r = 0.20`.

- **`MAX_RANGE_M = 6.0` is too generous for hardware.** At 6 m the radial
  noise is 3-7 cells wide; those returns are fresh every frame, so log-odds
  keeps them alive and walls thicken by several cells on top of the 4-cell
  inflation. 3.0 m keeps radial error under one cell at `high_accuracy`.
- **`l_occupied = 0.80` against `l_hit = 0.85` means one hit is enough.**
  That is calibrated on noiseless simulated depth. On hardware a single
  speckle return becomes an obstacle; requiring two hits (`l_occupied`
  ~1.2-1.5) or scaling `l_hit` down with range is the hardware adaptation.

## Host cost

Measured on an i9-14900HX, per frame, single-threaded; the RPi 5 column is
an estimate scaling by ~3-4x for a Cortex-A76 at 2.4 GHz and **has not been
measured on the RPi**.

| stage | i9 | est. RPi 5 | @10 Hz |
|---|---|---|---|
| decimation x2 | 0.97 ms | ~3.5 ms | 3% of a core |
| dec + spatial + temporal | 5.65 ms | ~20 ms | 20% |
| spatial + temporal at full 848x480 | 14.3 ms | ~50 ms | does not fit 30 Hz |
| deprojection (424x240) | 0.91 ms | ~3 ms | 3% |
| full chain + range clip | 10.9 ms | ~38 ms | ~38% |

Stereo matching itself costs the host nothing -- the D435 does it on its own
D4 ASIC. That is the reason this camera is viable on the robot at all.

A full 3D OctoMap over the raw cloud is not: ~70k points/frame with ~40
voxels of ray casting each is ~3M octree updates per frame, against maybe
0.3-1M/s on an A76. Either pre-filter to ~5k points/frame or stay with the
2D log-odds map, which is what the planner consumes anyway.

## Reproducing

The bench scripts are not in the repo; they were one-off measurements
against `pyrealsense2`. What each did:

- noise vs range: capture N frames, per-pixel temporal std, bin by per-pixel
  median range (needs no wall or ground truth, just a scene with depth
  variety)
- preset/filter comparison: the same, swept over `visual_preset`,
  `filter_smooth_delta` and `laser_power`
- aggregation: patch-median downsample to each grid, then the same temporal
  std per output cell
