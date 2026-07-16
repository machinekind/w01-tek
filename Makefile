# cluster HPC workflow for fbb-rl (see training/hpc/ and
# .claude/skills/cluster-hpc). The account is shared: every person works in
# their own namespace dir under $HOME (HPC_NS). ~/M/... belongs to
# Michal — never rsync or submit into someone else's namespace.

HPC_USER ?= ACCOUNT
HPC_NS   ?= USER
HPC      = $(HPC_USER)@ui.cluster.example
REMOTE   = /home/$(HPC_USER)/$(HPC_NS)/wojtek
EXCLUDE  = --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
           --exclude='.venv*/' --exclude='venv/' --exclude='logs/' \
           --exclude='runs/' --exclude='videos/' --exclude='.jax_cache/' \
           --exclude='.claude/' \
           --exclude='docs/' --exclude='*.mp4' --exclude='wandb/' \
           --exclude='outputs/'

.PHONY: hpc-push hpc-train hpc-status hpc-logs hpc-pull verify verify-quick verify-static

# Reorg verification harness (see verify.sh). `verify` runs T0-T3 (ROS build
# needs Docker); `verify-quick` skips the slow train/eval/docker steps;
# `verify-static` is the fast dependency-free T0 gate (good for pre-push/CI).
verify:
	./verify.sh
verify-quick:
	./verify.sh --quick
verify-static:
	./verify.sh --tier 0

hpc-push:
	rsync -avz $(EXCLUDE) ./ $(HPC):$(REMOTE)/

# EXPERIMENT/RUN_NAME/NUM_ENVS/BATCH/EXTRA pass through, e.g.
#   make hpc-train EXPERIMENT=locomotion RUN_NAME=fbb_loco_v1
hpc-train:
	ssh $(HPC) "cd $(REMOTE) && mkdir -p logs && sbatch \
	  --export=ALL$(if $(EXPERIMENT),\,EXPERIMENT=$(EXPERIMENT))$(if $(RUN_NAME),\,RUN_NAME=$(RUN_NAME))$(if $(NUM_ENVS),\,NUM_ENVS=$(NUM_ENVS))$(if $(BATCH),\,BATCH=$(BATCH))$(if $(EXTRA),\,EXTRA='$(EXTRA)') \
	  $(if $(TIME),--time=$(TIME)) training/hpc/train.slurm"

hpc-status:
	ssh $(HPC) "squeue -u $(HPC_USER) -o '%.10i %.12j %.4t %.10M %.20R'"

# make hpc-logs JOB=1234567
hpc-logs:
	ssh $(HPC) "tail -40 $(REMOTE)/logs/*$(JOB)*.out; echo ===ERR===; tail -20 $(REMOTE)/logs/*$(JOB)*.err"

hpc-pull:
	rsync -avz $(HPC):$(REMOTE)/training/runs/ training/runs/
