# MuJoCo model

`wojtek.xml` is robot-only (no `<option>`, no ground plane); always
load `scene.xml` for physically meaningful simulation, or provide your own
scene/`<option>` wrapper.

## Setup

From repo root:

```
uv venv && uv pip install mujoco
```

## Check

```
.venv/bin/python quadruped_ros2_original/wojtek_description/mujoco/check_model.py
```

## View

Use the native MuJoCo simulate app (download the macOS dmg from the
[MuJoCo releases page](https://github.com/google-deepmind/mujoco/releases)
and copy `MuJoCo.app` to `~/Applications`):

```
open -a ~/Applications/MuJoCo.app --args "$PWD/quadruped_ros2_original/wojtek_description/mujoco/scene.xml"
```

Double-click a body to select it, then Ctrl+right-drag to apply a force or
Ctrl+left-drag to apply a torque. The Control pane has a slider per actuator.

The Python route (`.venv/bin/mjpython -m mujoco.viewer --mjcf …/scene.xml`)
does not currently work on macOS 26: `mjpython` creates the GL window off the
main thread, which newer AppKit rejects (`NSWindow should only be instantiated
on the main thread!`). Headless use — `check_model.py`, RL training — is
unaffected.

## Naming conventions

Geoms follow `{leg}_{link}_visual` / `{leg}_{link}_collision` (e.g.
`rear_left_second_link_visual`); the shared `base_link_visual` /
`base_link_collision` have no leg prefix. Closure-marker sites are
`{leg}_chain_close_a`, `{leg}_chain_close_b`, and `{leg}_foot`. Future
MuJoCo work should follow this convention.
