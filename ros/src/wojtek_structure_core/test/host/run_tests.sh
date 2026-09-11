#!/usr/bin/env bash
# Host test for the conversion layer: no SDK, no ROS, no camera.
#
# colcon runs the same binary through add_test; this is the fast loop while
# editing. Mirrors test/host/run_tests.sh in the ACS5 firmware tree.
set -euo pipefail
cd "$(dirname "$0")"

CC=${CC:-g++}
CFLAGS=${CFLAGS:-"-std=c++17 -O2 -Wall -Wextra -Werror"}
OUT=${OUT:-/tmp}

# shellcheck disable=SC2086  # CFLAGS is deliberately word-split
$CC $CFLAGS -I../../include \
    test_frame_convert.cpp ../../src/frame_convert.cpp \
    -o "$OUT/test_frame_convert"

"$OUT/test_frame_convert"
