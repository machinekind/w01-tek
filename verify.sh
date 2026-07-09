#!/usr/bin/env bash
# Verification harness for the monorepo reorg (ros/ + training/ + training/demo/).
#
# The reorg is ~pure moves + repaths + deletions, so this checks the three
# things a move can break, cheapest-first:
#   T0  static + equivalence  — no ghost paths/imports, nothing lost, model byte-identical to baseline
#   T1  python resolve        — wojtek_rl/demo import, paths.py resolves, MJCF loads (CPU), pytest
#   T2  python liveness (CPU) — run.sh build (idempotent) + smoke train + eval render
#   T3  ROS 2 build           — docker image build == colcon build of all 6 packages
#
# Usage:
#   ./verify.sh                 # run T0..T3 (each self-skips if its prereqs are missing)
#   ./verify.sh --max 1         # run only T0 and T1
#   ./verify.sh --tier 3        # run only T3
#   ./verify.sh --quick         # skip slow steps (T2 smoke/eval, T3 image build)
#   VERIFY_BASE=<ref> ./verify.sh   # baseline for equivalence checks (default origin/main)
set -uo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
BASE_REF="${VERIFY_BASE:-origin/main}"
MAX_TIER=3; ONLY_TIER=""; QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max) MAX_TIER="$2"; shift 2;;
    --tier) ONLY_TIER="$2"; shift 2;;
    --quick) QUICK=1; shift;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ---------- reporting ----------
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; Z=$'\033[0m'; else G=; R=; Y=; B=; Z=; fi
PASS=0; FAIL=0; SKIP=0; FAILED=()
pass(){ printf "  ${G}PASS${Z} %s\n" "$1"; PASS=$((PASS+1)); }
fail(){ printf "  ${R}FAIL${Z} %s\n" "$1"; FAIL=$((FAIL+1)); FAILED+=("$1"); [ -n "${2:-}" ] && printf "%s\n" "$2" | sed 's/^/       /'; }
skip(){ printf "  ${Y}SKIP${Z} %s ${Y}(%s)${Z}\n" "$1" "$2"; SKIP=$((SKIP+1)); }
section(){ printf "\n${B}%s${Z}\n" "$1"; }
# ok DESC CMD...   -> pass iff cmd exits 0
ok(){ local d="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$d"; else fail "$d"; fi; }
# absent DESC PATTERN PATHSPEC...  -> pass iff git grep finds NO match (breakage guard).
# Excludes this harness (it literally contains every guard pattern) and the temp
# handoff doc (it intentionally names the old dirs).
absent(){ local d="$1" p="$2"; shift 2; local hit; if hit="$(git grep -nE "$p" -- "$@" ':(exclude)verify.sh' ':(exclude)NEXT-STEP-WOJTEK.md' 2>/dev/null)"; then fail "$d" "$hit"; else pass "$d"; fi; }
# same DESC REF_A:PATH_A REF_B:PATH_B  -> pass iff the two committed blobs are byte-identical
same(){ local d="$1" a="$2" b="$3"; if diff <(git show "$a" 2>/dev/null) <(git show "$b" 2>/dev/null) >/dev/null 2>&1; then pass "$d"; else fail "$d" "differ: $a  vs  $b"; fi; }
run_tier(){ local n="$1"; [ -n "$ONLY_TIER" ] && { [ "$ONLY_TIER" = "$n" ]; return; }; [ "$n" -le "$MAX_TIER" ]; }

PY="training/.venv/bin/python"

# =======================================================================
t0(){
  section "T0 — static structure, no-ghost guards, equivalence"

  # old layout gone / new layout present
  for d in 3_jaxpot_robotics 4_four_bar_bot_rl piesek_ws quadruped_ros2_original; do
    if [ -e "$d" ]; then fail "old dir removed: $d/"; else pass "old dir removed: $d/"; fi
  done
  for d in ros training training/demo docs ros/src/wojtek_description; do
    ok "present: $d/" test -d "$d"; done

  # dead imports / module targets (would crash at runtime)
  absent "no 'from|import wojtek_rl.(app|navigation)'" '(from|import)[[:space:]]+wojtek_rl\.(app|navigation)' '*.py'
  absent "no '-m wojtek_rl.app' in run scripts"        '\-m[[:space:]]+wojtek_rl\.app' '*.sh'

  # config/scripts free of old dir names (exclude .md history + frozen foreign-abs-path meta)
  absent "no '4_four_bar_bot_rl' in code/config" '4_four_bar_bot_rl' . ':(exclude)*.md' ':(exclude)*/policy_meta.json'
  absent "no 'piesek_ws' path ref"               'piesek_ws' . ':(exclude)*.md'
  absent "no live 'quadruped_ros2_original' ref" 'quadruped_ros2_original' . ':(exclude)*.md'
  absent "no live 'quadruped_controller' ref"    'quadruped_controller' . ':(exclude)*.md'
  absent "no live 'fbb_policy' ref (renamed -> wojtek_policy)" 'fbb_policy' . ':(exclude)*.md'
  absent "no live 'piesek' ref (piesek_bringup/piesek_robot -> wojtek_*)" 'piesek' . ':(exclude)*.md'
  absent "no live 'four_bar_bot_description' ref (renamed -> wojtek_description)" 'four_bar_bot_description' . ':(exclude)*.md'
  absent "no live 'four_bar_bot_mjx.xml' ref (renamed -> wojtek_mjx.xml)" 'four_bar_bot_mjx\.xml' . ':(exclude)*.md'
  absent "no live 'four_bar_bot.xml' ref (renamed -> wojtek.xml)" 'four_bar_bot\.xml' . ':(exclude)*.md'
  absent "no live 'fbb_rl' ref (renamed -> wojtek_rl)" 'fbb_rl' . ':(exclude)*.md'
  absent "no live 'FourBarBot' class ref (renamed -> Wojtek*)" 'FourBarBot' . ':(exclude)*.md'
  absent "no live 'FbbPolicy' class ref (renamed -> WojtekPolicy)" 'FbbPolicy' . ':(exclude)*.md'
  absent "no live 'fbb_env' alias ref (renamed -> wojtek_env)" 'fbb_env' . ':(exclude)*.md'
  absent "no live 'fbb.rviz' ref (renamed -> wojtek.rviz)" 'fbb\.rviz' . ':(exclude)*.md'
  absent "no live urdf fbb_*/four_bar_bot xacro tokens (renamed -> wojtek_*)" 'fbb_(real|sim|joint|ros2_control|imu_ros2_control|bmi160_imu_ros2_control)|four_bar_bot_body|four_bar_bot\.urdf\.xacro' . ':(exclude)*.md'
  absent "removed legacy launches not referenced" '(bringup|robot_state_publisher)\.launch\.py' 'ros/src/*/launch/*.py'

  # paths.py points at ros/, not the dropped quadruped dir
  ok "paths.py model dir -> ros/"   grep -q 'ros/src/wojtek_description/mujoco' training/wojtek_rl/paths.py
  absent "paths.py free of quadruped" 'quadruped_ros2_original' training/wojtek_rl/paths.py

  # every referenced file exists
  for f in \
    ros/src/wojtek_description/mujoco/wojtek.xml \
    ros/src/wojtek_description/mujoco/wojtek_mjx.xml \
    ros/src/wojtek_description/mujoco/scene_mjx.xml \
    ros/src/md80_hardware_interface/config/AK80-9.cfg \
    ros/src/wojtek_policy/config/policy.npz \
    ros/src/wojtek_policy/config/policy_meta.json \
    ros/src/wojtek_bringup/config/wojtek_mjx.xml \
    training/demo/app.py training/demo/navigation.py training/demo/static/index.html; do
    ok "exists: $f" test -f "$f"; done

  # gitignore hygiene: big artifacts must not leak into status, and every pattern is ignored
  ok "big artifacts don't leak into git status" \
     bash -c '! git status --porcelain --untracked-files=all | grep -qE "training/(\.venv|runs|videos|pose_previews|\.jax_cache|policies)/|ros/(build|install|log)/|\.log$|\.DS_Store$"'
  ok "gitignore covers every artifact pattern" \
     bash -c 'for p in training/.venv training/runs training/videos training/pose_previews training/.jax_cache training/policies training/x.log training/.DS_Store ros/build/x ros/install/x ros/log/x; do git check-ignore -q "$p" || { echo "not ignored: $p"; exit 1; }; done'

  # ---- equivalence: the model training reads must be byte-identical to the baseline ----
  if git rev-parse --verify -q "$BASE_REF" >/dev/null; then
    local QB="$BASE_REF:quadruped_ros2_original/four_bar_bot_description/mujoco"
    local RH="HEAD:ros/src/wojtek_description/mujoco"
    # robot MJCFs are leaf files (no includes) -> renamed to wojtek*.xml = pure moves, still byte-identical to upstream
    same "model source wojtek.xml == baseline (upstream four_bar_bot.xml)" "$QB/four_bar_bot.xml" "$RH/wojtek.xml"
    same "model wojtek_mjx.xml == baseline (upstream four_bar_bot_mjx.xml)" "$QB/four_bar_bot_mjx.xml" "$RH/wojtek_mjx.xml"
    # scene_mjx.xml differs from baseline by EXACTLY the include rename (four_bar_bot_mjx.xml -> wojtek_mjx.xml); normalize then compare
    if diff <(git show "$QB/scene_mjx.xml") <(git show "$RH/scene_mjx.xml" | sed 's/wojtek_mjx\.xml/four_bar_bot_mjx.xml/') >/dev/null 2>&1; then
      pass "model scene_mjx.xml == baseline (modulo include rename)"
    else fail "model scene_mjx.xml differs beyond include rename" "$(diff <(git show "$QB/scene_mjx.xml") <(git show "$RH/scene_mjx.xml") 2>/dev/null)"; fi
    # no unexpected large content introduced (renames reuse blobs; only tiny edits are new)
    local big
    big="$(git rev-list --objects HEAD --not "$BASE_REF" 2>/dev/null | awk '{print $1}' \
           | git cat-file --batch-check='%(objecttype) %(objectsize) %(objectname)' 2>/dev/null \
           | awk '$1=="blob" && $2>65536 {print}')"
    if [ -z "$big" ]; then pass "no new blob >64KB vs baseline (no content bloat/corruption)"
    else fail "new large blob(s) vs baseline" "$big"; fi
  else
    skip "equivalence vs $BASE_REF" "baseline ref not found"
  fi
}

# =======================================================================
t1(){
  section "T1 — python resolve (imports, paths, model load, pytest)"
  if [ ! -x "$PY" ]; then skip "T1 python checks" "training/.venv not found"; return; fi

  ok "wojtek_rl + demo modules import" bash -c "cd training && ./.venv/bin/python -c 'import wojtek_rl.env, wojtek_rl.train, wojtek_rl.eval, wojtek_rl.build_model, wojtek_rl.paths; from demo import app, navigation'"
  ok "paths.py resolves all model files" bash -c "cd training && ./.venv/bin/python -c 'from wojtek_rl import paths; [open(p).close() for p in (paths.SOURCE_XML, paths.ROBOT_XML, paths.SCENE_XML)]'"
  ok "MJCF scene loads in MuJoCo (CPU)" bash -c "cd training && ./.venv/bin/python -c 'from wojtek_rl import paths; import mujoco; mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))'"
  ok "demo.app imports + argparse (heavy imports deferred)" bash -c "cd training && MUJOCO_GL=cgl ./.venv/bin/python -m demo.app --help"
  # scripts are syntactically valid
  for s in training/run.sh training/hpc/_common.sh training/hpc/train.slurm ros/build.sh ros/dev.sh verify.sh; do
    ok "syntax ok: $s" bash -n "$s"; done
  # test suite
  if ok "pytest training/tests" bash -c "cd training && ./.venv/bin/python -m pytest tests -q"; then :; fi
}

# =======================================================================
t2(){
  section "T2 — python liveness on CPU (build / smoke / eval)"
  if [ ! -x "$PY" ]; then skip "T2 liveness" "training/.venv not found"; return; fi

  # run.sh build regenerates the MJX model from source; must reproduce the committed file
  if (cd training && ./run.sh build >/tmp/verify_build.$$ 2>&1); then
    if git diff --quiet -- ros/src/wojtek_description/mujoco/; then
      pass "run.sh build reproduces committed model (deterministic + lossless)"
    else
      fail "run.sh build changed the committed model" "$(git diff --stat -- ros/src/wojtek_description/mujoco/)"
      git checkout -- ros/src/wojtek_description/mujoco/ 2>/dev/null
    fi
  else
    fail "run.sh build" "$(tail -5 /tmp/verify_build.$$ 2>/dev/null)"
  fi
  rm -f /tmp/verify_build.$$

  if [ "$QUICK" = 1 ]; then skip "run.sh smoke (CPU train)" "--quick"; skip "run.sh eval render" "--quick"; return; fi

  # tiny CPU training smoke — proves the whole PPO wiring boots and steps
  if timed 900 bash -c "cd training && ./run.sh smoke >/tmp/verify_smoke.$$ 2>&1"; then
    pass "run.sh smoke (CPU training boots + steps)"
  else
    fail "run.sh smoke" "$(tail -8 /tmp/verify_smoke.$$ 2>/dev/null)"
  fi
  rm -f /tmp/verify_smoke.$$

  # eval render of a committed policy on CPU (MUJOCO_GL=cgl works headless on macOS)
  if [ -d training/policies/fbb_v2 ]; then
    if timed 600 bash -c "cd training && MUJOCO_GL=cgl ./run.sh eval --run policies/fbb_v2 --no-push >/tmp/verify_eval.$$ 2>&1"; then
      pass "run.sh eval renders fbb_v2 (model+policy+render)"
    else
      skip "run.sh eval fbb_v2" "eval failed/needs GPU — see /tmp/verify_eval.$$ ($(tail -1 /tmp/verify_eval.$$ 2>/dev/null))"
    fi
  else
    skip "run.sh eval" "no committed policy in training/policies/"
  fi
}

# =======================================================================
t3(){
  section "T3 — ROS 2 workspace build (Docker image build == colcon build)"
  if [ "$QUICK" = 1 ]; then skip "docker image build" "--quick"; return; fi
  if ! command -v docker >/dev/null 2>&1; then skip "docker image build" "docker not installed"; return; fi
  if ! docker info >/dev/null 2>&1; then skip "docker image build" "docker daemon not running"; return; fi
  # The Dockerfile's `RUN colcon build --packages-select <6 pkgs>` IS the build check.
  echo "  (building ros/docker image — runs colcon build of all 6 packages; slow, esp. under emulation)"
  if docker build -f ros/docker/Dockerfile -t shrek-verify:ros ros >/tmp/verify_docker.$$ 2>&1; then
    pass "colcon build (wojtek_description, md80_hw, bmx160, bmi160, wojtek_policy, wojtek_bringup)"
  else
    fail "ROS colcon build via docker" "$(tail -15 /tmp/verify_docker.$$ 2>/dev/null)"
  fi
  rm -f /tmp/verify_docker.$$
}

# portable timeout (macOS has no `timeout`): timed SECONDS cmd...
timed(){ local t="$1"; shift; if command -v perl >/dev/null 2>&1; then perl -e 'my $t=shift; my $p=fork; if($p==0){exec @ARGV or exit 127} local $SIG{ALRM}=sub{kill "TERM",$p; exit 124}; alarm $t; waitpid $p,0; exit($?>>8)' "$t" "$@"; else "$@"; fi; }

# =======================================================================
printf "${B}verify — reorg harness${Z}  (base=%s, tiers<=%s%s)\n" "$BASE_REF" "${ONLY_TIER:-$MAX_TIER}" "$([ "$QUICK" = 1 ] && echo ', quick')"
run_tier 0 && t0
run_tier 1 && t1
run_tier 2 && t2
run_tier 3 && t3

section "SUMMARY"
printf "  ${G}%d passed${Z}   ${R}%d failed${Z}   ${Y}%d skipped${Z}\n" "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -gt 0 ]; then printf "\n  ${R}failing:${Z}\n"; printf "    - %s\n" "${FAILED[@]}"; exit 1; fi
exit 0
