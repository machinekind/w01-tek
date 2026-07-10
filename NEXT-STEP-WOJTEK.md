# Next step: converge the robot name on **Wojtek**

> **Temporary handoff doc** — delete once the Wojtek rename round lands. Captures
> what's done, what's next, and how to verify it.

## Where we are (this round)

Branch `mwysocki/repo-reorg`, based on **fresh `origin/main`** (`1fdb74f`). Structural
monorepo reorg is complete:

```
ros/        (was piesek_ws/)      — ROS 2 colcon workspace, deploy on hardware
training/   (was 4_four_bar_bot_rl/) — MJX + Brax PPO RL project
  demo/                            — click-to-walk web app (extracted from fbb_rl)
docs/
verify.sh, Makefile
```

Dropped: `3_jaxpot_robotics/` (Unitree), `quadruped_ros2_original/` (reference; model
repointed into `ros/`, `AK80-9.cfg` preserved into `ros/src/md80_hardware_interface/config/`).
Commits: demo extract → `training/` rename → gitignore hygiene → `ros/` rename →
quadruped drop → verify harness.

## The task: total name convergence → `wojtek_*`

One robot, one name. Rename everything off the legacy `piesek` / `four_bar_bot` / `fbb`
sprawl. **Invasive part = ROS packages**: colcon keys off the package *directory name*,
which must equal `package.xml <name>` = `CMakeLists project()` = the `ament_index`
resource, and every `find_package` / launch / `get_package_share_directory` reference.

| Rename | Where it bites |
|---|---|
| `piesek_bringup` → `wojtek_bringup` | dir, package.xml, CMake/setup.py, launch cmds, `ros/dev.sh` + `README` |
| `four_bar_bot_description` → `wojtek_description` | dir, package.xml/CMake, xacro `$(find …)`, meshes, **`training/fbb_rl/paths.py`** model path |
| `four_bar_bot*.xml` → `wojtek*.xml` | MJCF filenames + `<include>`, `build_model.py`, `paths.py`, `mujoco_sim_node.py` |
| `fbb_rl` → `wojtek_rl` | python pkg; imports across `training/` + `demo/` + tests; `run.sh -m fbb_rl.*`; hpc scripts |
| `fbb_policy` → `wojtek_policy` | ROS package; `mujoco_sim_node` `get_package_share_directory("fbb_policy")` |
| `FourBarBotJoystick` → `WojtekJoystick` | env class; `demo/app.py`, `eval.py`, registry |
| repo `wojtek` → `wojtek` | git remote, `Makefile REMOTE=~/M/wojtek`, HPC `$WORKDIR` |

## Recommended order

1. **Mesh slim-down FIRST.** Renaming `four_bar_bot_description` re-uploads the meshes
   again (see below). Decide **Git LFS** for `*.dae`/`*.stl` **or prune the redundant
   variants** (`base_link.dae` + `base_link_merged.dae` + `.stl` = 118 MB uncompressed
   for a small robot) before any rename that moves the mesh package.
2. ROS package renames (`wojtek_bringup`, `wojtek_description`, `wojtek_policy`) — one
   package per commit, `make verify` (T3) after each.
3. MJCF + `paths.py` (`four_bar_bot*.xml` → `wojtek*.xml`).
4. Python package `fbb_rl` → `wojtek_rl` + class `FourBarBotJoystick` → `WojtekJoystick`.
5. Repo/remote `wojtek` → `wojtek` (last; coordinate with the team).

## Verification method

Use the harness (`./verify.sh`, `make verify*`) — it's built for exactly this kind of
move/repath refactor.

- **Before each rename:** `make verify-static` must be green (T0 baseline).
- **After each rename:** `make verify` (or at least the tier that covers what changed).
  ROS package renames → **T3 is the real gate** (the Docker image build *is* the
  `colcon build`; a missed `package.xml`/CMake/launch ref fails it). Run T3 on Linux/CI
  or the robot host (emulated on Apple Silicon ≈ 20 min).
- **Update the harness in lockstep.** `verify.sh` hardcodes model paths and an
  equivalence baseline. When you rename `four_bar_bot_description` → `wojtek_description`
  and the MJCFs, update:
  - the referenced-file list + `paths.py` grep in **T0**
  - the **equivalence block** paths, and set `VERIFY_BASE` to the *pre-Wojtek* commit
    (e.g. `VERIFY_BASE=<reorg-tip> make verify`) so it still proves byte-identity of the
    model across the rename.
- **Ghost-name guards:** extend T0's `absent` checks to also fail on live
  `piesek_bringup` / `four_bar_bot` / `fbb_rl` / `fbb_policy` once each is renamed
  (leave provenance comments in `*.md` excluded).
- **Mesh re-upload watch:** after renaming the description, `git push` re-sends ~36 MB
  of meshes unless the LFS/slim-down in step 1 is done. Push over SSH and/or
  `git config http.postBuffer 524288000` to avoid the HTTPS hang.

## Decisions still open (not blockers for the rename mechanics)

- **Canonical Wojtek model** — deploy sim is **±6 Nm** (fbb_loco_v8's real training
  physics, *not a bug*); the `training/` baseline description is **±9 Nm**. One robot ⇒
  one model: pick the canonical caps/mass and reconcile the copies during this round.
- **Mesh duplication** — resolve as part of step 1.
