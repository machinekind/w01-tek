.PHONY: verify verify-quick verify-static

# Reorg verification harness (see verify.sh). `verify` runs T0-T3 (ROS build
# needs Docker); `verify-quick` skips the slow train/eval/docker steps;
# `verify-static` is the fast dependency-free T0 gate (good for pre-push/CI).
verify:
	./verify.sh
verify-quick:
	./verify.sh --quick
verify-static:
	./verify.sh --tier 0
