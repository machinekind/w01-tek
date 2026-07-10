---
name: vastai-gpu-training-ops
description: Operate long-running training jobs on Vast.ai GPU instances. Use for instance selection, CUDA/JAX provisioning checks, detached launches, log and process monitoring, artifact sync, rendering and SSH tunnels, spend control, and teardown verification.---

# Operate Vast.ai GPU training

Instances are disposable: establish artifact recovery before launching, monitor both the process
and its log, and verify the off-box copy before teardown.

Define the connection as a function, not a string variable — zsh (the Bash-tool shell) does not
word-split an unquoted `$SSH`, so `SSH="ssh -p PORT root@HOST"; $SSH cmd` fails with
"command not found":

```bash
sshc() { ssh -o ConnectTimeout=10 -p PORT root@HOST "$@"; }
WORK=/workspace/PROJECT/training
```

For this repository: module `wojtek_rl`, launcher `training/run.sh`, safe process pattern
`wojtek_rl[.]train`.

## Provision

- Prefer EU offers over US ones. US hosts have repeatedly returned `"success": false` contracts
  that sit in "loading" forever; EU boxes schedule reliably. A `success: false` create never
  recovers — destroy it and pick another host immediately.
- Require host reliability ≥ 99.9% (`reliability>0.999` in the offer query). The few cents/hour
  saved on a flakier host are not worth a mid-run recycle.
- Prefer a driver family already proven with the pinned JAX/CUDA wheel; verify GPU compute
  capability before touching dependency versions. ~40 GB disk is the current project's starting
  point, not a universal minimum.
- A CUDA runtime image may train fine yet lack EGL graphics libraries; decide up front whether
  rendering must happen remotely.

```bash
sshc 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'
sshc 'df -h / | tail -1; vast-capabilities | jq .instance.workspace_is_volume'  # false: recycling loses /workspace
sshc "cd '$WORK' && uv venv --python 3.11 .venv && uv sync"
sshc "cd '$WORK' && .venv/bin/python -c 'import jax; print(jax.__version__, jax.devices())'"
```

Do not upgrade major JAX, CUDA, MuJoCo, or Brax versions while debugging provisioning.

## Launch

Pass the CPU-smoke and GPU-probe gates (`brax-locomotion-training`) first. Save the resolved
Hydra config before launching — `run.json` is written only after success:

```bash
sshc "cd '$WORK' && .venv/bin/python -m wojtek_rl.train +experiment=locomotion_v8 \
  run_name=NAME --cfg job --resolve > run_NAME.resolved.yaml"
sshc "cd '$WORK' && rm -f run_NAME.log && (PYTHONUNBUFFERED=1 nohup ./run.sh train \
  +experiment=locomotion_v8 run_name=NAME > run_NAME.log 2>&1 < /dev/null &) && echo LAUNCHED"
```

Verify PID, log, and GPU utilization immediately. Run nothing else GPU-heavy beside training; use
`JAX_PLATFORMS=cpu` for side checks. XLA compilation delays the first progress line, autotuner
messages without a traceback are not failures, and a GPU dip can be an eval or checkpoint pause —
check the process before declaring it dead.

## Monitor

```bash
skills/vastai-gpu-training-ops/scripts/watch_run.sh \
  -s "ssh -o ConnectTimeout=10 -p PORT root@HOST" -l "$WORK/run_NAME.log" \
  -p "wojtek_rl[.]train" -m ./run_NAME.mirror.log
```

Exit 0 = completion marker, 1 = crash signature, 2 = process gone without a captured signature.
A persistent SSH failure is inconclusive: restore connectivity and verify PID and log before
changing instance state. On non-finite evals, stop burning GPU time and switch to the
`brax-locomotion-training` diagnostics.

## Sync after every stage

```bash
rsync -az -e "ssh -p PORT" root@HOST:"$WORK/runs/NAME/" ./training/runs/NAME/
scp -P PORT root@HOST:"$WORK/run_NAME.log" root@HOST:"$WORK/run_NAME.resolved.yaml" \
  root@HOST:"$WORK/*.mp4" ./training/runs/NAME/
```

Remote `run.json` may contain absolute checkpoint paths — normalize after syncing if the local
loader does not resolve the run directory. `training/runs/` is gitignored: a local rsync is not
yet a durable backup.

## Render, tunnel, teardown

- No EGL on the box: sync the checkpoint and render locally (`MUJOCO_GL=cgl` on macOS).
- Dashboards: bind to 127.0.0.1 and tunnel `ssh -p PORT -N -L 8080:127.0.0.1:APP_PORT root@HOST`;
  never expose them on a public interface.
- Before stop/destroy, verify off-box copies of checkpoints, `run.json`, resolved config, raw
  logs (failed runs too), eval media, and anything created only on the instance. Stop keeps the
  disk; destroy only when the repo and artifact store can reproduce the rest.

## Red flags

- Launching before planning artifact recovery
- Treating a silent log during compilation, or an SSH error, as proof the run died
- Running eval or tests on the occupied GPU
- Leaving the only checkpoint on ephemeral storage
- Destroying the instance before opening or hashing the synced artifacts
