# Spark experiment rigs

Session tooling for renting DGX Spark (GB10) boxes and validating the
agentic stack on them: demo rig, ROS rig (one-box and brain/world split),
TTS benchmark, component probe, and the rental kill-timer. Everything here
is experiment scaffolding, deliberately OUTSIDE the deployable tree
(`ros/`, `training/`): the robot never runs any of it.

House rules these scripts follow:

- No credentials, hostnames, account or machine identifiers in the tree.
  Connection strings arrive per-session via environment (`SPARK_SSH`,
  `BRAIN_SSH`), secrets ride stdin to 600-mode files, and the deployment
  organization comes from `HF_ORGANIZATION` in the caller's environment.
- Arm `rental_watchdog.sh` BEFORE deploying anything.
- `deploy` asserts GPU clocks under load: some marketplace GB10s are
  platform power-capped to a third of their clocks, and every latency
  number from such a box is fiction.
