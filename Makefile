# cluster HPC workflow for the training project (see training/hpc/ and
# skills/cluster-hpc). The account is shared: every person works in their
# own namespace dir under $HOME (HPC_NS) — never rsync or submit into
# someone else's namespace. cluster_USER, HPC_NS, and HPC_REPO are personal:
# they live in .env (gitignored, template in .env.example), which also
# rides to the cluster checkout via `make hpc-push` so SLURM jobs can
# read it (training/hpc/_common.sh).

cluster_USER ?=
HPC_NS    ?=
HPC_REPO  ?= wojtek
-include .env
HPC        = $(cluster_USER)
HPC_LOGIN  = $(word 1,$(subst @, ,$(cluster_USER)))
REMOTE     = /home/$(HPC_LOGIN)/$(HPC_NS)/$(HPC_REPO)

EXCLUDE  = --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
           --exclude='.venv*/' --exclude='venv/' --exclude='logs/' \
           --exclude='runs/' --exclude='videos/' --exclude='.jax_cache/' \
           --exclude='.claude/' \
           --exclude='docs/' --exclude='*.mp4' --exclude='wandb/' \
           --exclude='outputs/'

.PHONY: hpc-vars hpc-push hpc-train hpc-status hpc-logs hpc-pull verify verify-quick verify-static

# Reorg verification harness (see verify.sh). `verify` runs T0-T3 (ROS build
# needs Docker); `verify-quick` skips the slow train/eval/docker steps;
# `verify-static` is the fast dependency-free T0 gate (good for pre-push/CI).
verify:
	./verify.sh
verify-quick:
	./verify.sh --quick
verify-static:
	./verify.sh --tier 0

hpc-vars:
	@test -n "$(cluster_USER)" && test -n "$(HPC_NS)" || { \
	  echo "Set cluster_USER=<user>@ui.cluster.example and HPC_NS=<your namespace dir> in .env (see .env.example)"; \
	  exit 1; }

hpc-push: hpc-vars
	rsync -avz $(EXCLUDE) ./ $(HPC):$(REMOTE)/

# EXPERIMENT/RUN_NAME/NUM_ENVS/BATCH/EXTRA pass through, e.g.
#   make hpc-train EXPERIMENT=locomotion RUN_NAME=fbb_loco_v1
hpc-train: hpc-vars
	ssh $(HPC) "cd $(REMOTE) && mkdir -p logs && sbatch \
	  --export=ALL$(if $(EXPERIMENT),\,EXPERIMENT=$(EXPERIMENT))$(if $(RUN_NAME),\,RUN_NAME=$(RUN_NAME))$(if $(NUM_ENVS),\,NUM_ENVS=$(NUM_ENVS))$(if $(BATCH),\,BATCH=$(BATCH))$(if $(TERRAIN),\,TERRAIN=$(TERRAIN))$(if $(WANDB),\,WANDB=$(WANDB))$(if $(EXTRA),\,EXTRA='$(EXTRA)') \
	  $(if $(TIME),--time=$(TIME)) training/hpc/train.slurm"

hpc-status: hpc-vars
	ssh $(HPC) "squeue -u $(HPC_LOGIN) -o '%.10i %.12j %.4t %.10M %.20R'"

# make hpc-logs JOB=1234567
hpc-logs: hpc-vars
	ssh $(HPC) "tail -40 $(REMOTE)/logs/*$(JOB)*.out; echo ===ERR===; tail -20 $(REMOTE)/logs/*$(JOB)*.err"

hpc-pull: hpc-vars
	rsync -avz $(HPC):$(REMOTE)/training/runs/ training/runs/
