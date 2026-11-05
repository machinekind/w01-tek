# Documentation validation

This record covers the handbook and supporting fixes based on commit `207480e`.
It records checks performed during the change, not certification of a physical
robot or every documented platform.

## Checks performed

| Check | Result |
|---|---|
| Training unit suite | 550 passed, 8 skipped after sysid input changes |
| Training configuration | Stock bounded-run configuration and new-leg configuration resolve |
| ROS image | Clean ARM64 Docker build completed; 17 ROS packages built |
| ROS Python/launch suite | 397 passed, 18 skipped across navigation, teleop, bring-up and PC packages |
| ROS MuJoCo control | Cached stock schema-2 policy; zero, stand and arm accepted |
| Waypoint example | A 0.3 m relative goal reached in both default and mux sessions; client exited successfully and cancelled on exit |
| Command ownership | Sole `/cmd_vel` publisher; manual priority, lease stop, autonomy inhibition and disarm cancellation passed in ROS MuJoCo |
| Client interruption | Terminal process-group SIGINT returned 130 and left autonomy inhibited |
| External cancellation | In an isolated ROS graph, the active client exited with failure, goto became idle and no further goal refresh arrived |
| Observer bridge | State subscription succeeded; publishing, service calls, parameter writes and arbitrary file access were refused |
| Documentation | 10 checker tests passed; local links, headings, runtime defaults and whitespace checked |

The clean build initially lacked `ros2 control`. `ros2controlcli` is now declared
as a bring-up dependency. The test container installed that package before the
remaining diagnostics checks.

The simulation test used a disposable Docker container on a macOS ARM64 host,
ROS domain 77 and localhost-only discovery. No robot devices, host networking or
SSH credentials were passed to the test process. The goal test disabled the
camera and costmap; it validates the client/controller path, not obstacle
avoidance. A simulation control loop is not an RT-kernel performance measurement.

The observer test used `ros-jazzy-foxglove-bridge` version
`3.6.0-1noble.20260929.221626`, negotiating `foxglove.sdk.v1`. Its server-info
capabilities were exactly `connectionGraph` and `assets`. The client received
`/joint_states`, then received refusals for advertising `/cmd_vel`, setting a
parameter, calling a service and fetching `file:///etc/passwd`. This tests the
restricted socket, not authorization on DDS or the other operator gateways.

The container had temporary outbound network access to install the diagnostics
package. Simulation checks ran without external networking or exposed ports.

## Review corrections, 2026-11-05

A fresh documentation environment installed the two pinned parser dependencies
with hash verification. All 21 checker/workflow tests passed, including merge
whitespace, divergent PR bases, reference links, heading anchors and code spans
across Markdown blocks. All 69 Markdown files passed the updated checker.

The 12 capture/scoring tests also passed, including explicit legacy-fixture
compatibility. This follow-up changed documentation, comments and validation
tooling; the ROS and training results above are from the original implementation.

## Repeat the automated checks

From the repository root:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install --require-hashes -r scripts/requirements-docs.txt
.venv-docs/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv-docs/bin/python scripts/check_docs.py
./training/run.sh test
git diff --check
```

In a built ROS container, from `/ros2_ws`:

```bash
source install/setup.bash
python3 -m pytest src/wojtek_nav/test src/wojtek_teleop/test \
  src/wojtek_bringup/test src/wojtek_pc/test -q
```

The goal check used the following isolated simulation configuration, with a
compatible local export copied to `/tmp/policy`:

```bash
ros2 launch wojtek_pc sim.launch.py policy:=/tmp/policy hw:=mujoco \
  rviz:=false camera:=false console:=none deck:=false nav:=true nav_costmap:=false
# Repeat with command_mux:=true for the mux path.
```

After zero, stand completion and arm acceptance, run:

```bash
ros2 run wojtek_nav goal_client --forward 0.3 --timeout 20
```

Add `--autonomy` for the mux path. An arm request made during the
stand ramp was refused; a goal sent while disarmed timed out and cancelled as
expected. These checks do not make the client an arming interface.

## Remaining acceptance work

| Work | Required evidence |
|---|---|
| Fresh-reader walkthrough | An operator/developer follows the guide without maintainer assistance |
| Clean GPU training | Bounded train/checkpoint/export run on a named supported GPU, followed by the fixed evaluation |
| Steam Deck and Xbox | Physical pairing, mappings, disconnect and handover checks on the actual devices |
| New-leg operation | Confirmed boot stops, ranges, collisions and the installed hardware profile |
| Manufacturing and wiring | Current CAD/BOM, electrical ratings and assembly review; see the hardware evidence register |
| Calibrated twin | Versioned physical measurements, held-out replay/comparison and agreed error limits |
| VLM quality | Human-labeled capture set and measured end-to-end tasks using the named model/version |

Do not replace these entries with a passing unit-test count. Keep results with
the code/model/policy/calibration revision and the command used to obtain them.
