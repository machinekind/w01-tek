# Isaac Sim prototype

This is a historical rendering and PhysX-import prototype for the stock robot.
It is not a supported quickstart or a validated digital twin. Use
[ROS MuJoCo](../../../docs/simulation.md) to exercise the deployed stack, or the
[plain-MuJoCo room demo](../README.md) for a scanned scene and exported policy.

## What remains useful

| File | Purpose |
|---|---|
| `isaac_room_server.py` | Headless physics/render loop and browser server, with apartment/warehouse scene presets. |
| `isaac_room.html` | Browser camera views, minimap and command controls. |
| `wojtek_mjcf2usd.py` | One-off MJCF importer experiment. |
| `body_masses.json` | Historical per-body mass corrections after import. |
| `home_qpos.json` | Historical loop-consistent home configuration. |

The recorded experiment used Isaac Sim 5.0 and Python 3.11. This is provenance,
not a claim that the current code installs or runs in a fresh environment.

## Blockers to a maintained launch

- The server imports a copied `policy.py` and `navigation.py` from a fixed
  home-directory asset path. That is not the current packaged policy runtime,
  and the old guide referenced a training module that no longer exists.
- `POLICY_DIR` is not a working configuration option in this script. The robot
  USD and policy paths are hard-coded.
- The converter writes `wojtek.usd`, while the server expects a different USD
  filename and articulation hierarchy. The two scripts are not a complete
  import-to-launch pipeline.
- Policy, mass, home-pose, collision and camera assumptions are stock-specific.
  New-leg support has not been established.
- Scene assets, importer output and renderer behavior need validation against
  a pinned simulator version before the launch can be called reproducible.

A maintained version needs explicit asset/policy paths, the shared schema-2
runtime, a repeatable USD conversion step and a simulation test covering a
standing hold, walking, stop and restart. Keep that work separate from the
current ROS/MuJoCo quickstart.

## Historical observations

The original experiment corrected imported masses, recreated missing collision
geometry, removed a spurious articulation root and used a settled home pose to
avoid closed-chain instability. The apartment scan needed authored colliders;
visual geometry alone did not produce a physical room. Rendering resolution and
camera smoothing were tuned on one RTX laptop GPU.

Those observations are useful investigation leads. They do not establish
current importer behavior or equivalence with the MuJoCo model. Re-measure
mass/inertia, joint axes/ranges, loop closure, contacts and actuator response
before comparing policy performance across simulators. See the
[digital-twin workflow](../../docs/digital-twin.md).
