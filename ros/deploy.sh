#!/usr/bin/env bash
# Deploy to the RPi from the PC. Two things in one command:
#   * (optional) PROVISION the RPi   -- ROS, RT kernel, RT tuning, network,
#     service -- via deploy/rpi/install.sh (idempotent; safe to repeat).
#   * BUILD the robot workspace natively on the RPi (rsync src + colcon).
#
# The RPi is a fixed anchor at 10.42.0.2 on whichever link is up -- ethernet
# cable when docked, its own wlan0 AP ("wojtek-link") when mobile. See
# deploy/rpi/README.md for the full onboarding workflow.
#
#   ./deploy.sh                 # fast: rsync src, build, restart the service
#   ./deploy.sh --provision     # fresh RPi: full provision (install.sh) THEN build
#   ./deploy.sh --provision --enable-robot   # ... and make it start at boot
#   ./deploy.sh --no-restart    # skip the wojtek-robot.service restart
#
# Boot autostart is DECLARATIVE: --enable-robot is the state you want, and
# provisioning makes the RPi match it. Provisioning WITHOUT the flag disables
# boot autostart, including on a robot where it was enabled before -- that is
# the point, so the flag alone decides and nobody has to remember what the
# previous run left behind.
#   RPI_HOST=rpi@10.42.0.2 ./deploy.sh
#
# Provisioning needs the Ubuntu Pro token for the RT kernel -- put it in .env
# (see .env.example); it is sourced below and never committed.
#
# Only the robot subset is built (colcon --packages-up-to wojtek_bringup), so
# wojtek_pc / rviz / plotjuggler / mujoco never land on the RPi.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Secrets + local overrides (UBUNTU_PRO_TOKEN, optional RPI_HOST, ...).
if [ -f "${HERE}/.env" ]; then
    set -a; . "${HERE}/.env"; set +a
fi

RPI_HOST="${RPI_HOST:-rpi@10.42.0.2}"      # static anchor; same over eth or the RPi's AP
REMOTE_WS="${REMOTE_WS:-wojtek_ws}"        # relative to the RPi user's home
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
# Keep in sync with docker/Dockerfile's CANDLE_COMMIT.
CANDLE_COMMIT="7c201d744baa107ee408e3e01694efd1a764a174"
RESTART=1
PROVISION=0
ENABLE_ROBOT=0

for a in "$@"; do case "$a" in
    --provision)  PROVISION=1 ;;
    --no-restart) RESTART=0 ;;
    # Passed through to install.sh: turn ON boot autostart. Only meaningful
    # together with --provision.
    --enable-robot) ENABLE_ROBOT=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

wait_for_ssh() {
    echo ">> waiting for ${RPI_HOST} to come back ..."
    for _ in $(seq 1 60); do
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "${RPI_HOST}" true 2>/dev/null; then
            echo "   back."; return 0
        fi
        sleep 5
    done
    echo "!! ${RPI_HOST} did not come back in time" >&2; return 1
}

# ------------------------------------------------------------ provisioning
if [ "${PROVISION}" = 1 ]; then
    echo ">> provisioning ${RPI_HOST} (install.sh)"
    ssh "${RPI_HOST}" "mkdir -p '${REMOTE_WS}/deploy'"
    # Push the whole deploy tree (rpi configs + install.sh + the service unit).
    rsync -az --delete "${HERE}/deploy/" "${RPI_HOST}:${REMOTE_WS}/deploy/"

    # Run install.sh with the Pro token passed through the environment (not
    # baked into any file on the RPi). Token only needed the first time (RT
    # kernel); subsequent runs short-circuit that phase.
    install_args=""
    [ "${ENABLE_ROBOT}" = 1 ] && install_args="--enable-robot"
    ssh "${RPI_HOST}" \
        "UBUNTU_PRO_TOKEN='${UBUNTU_PRO_TOKEN:-}' WOJTEK_AP_PSK='${WOJTEK_AP_PSK:-}' ROS_DISTRO='${ROS_DISTRO}' \
         bash '${REMOTE_WS}/deploy/rpi/install.sh' ${install_args}"

    # Kernel/cmdline changes leave /var/run/reboot-required -- reboot + wait so
    # the RT kernel is active before we build.
    if ssh "${RPI_HOST}" 'test -f /var/run/reboot-required' 2>/dev/null; then
        echo ">> reboot required after provisioning -- rebooting ${RPI_HOST}"
        ssh "${RPI_HOST}" 'sudo reboot' || true
        sleep 5
        wait_for_ssh
    fi
fi

# ------------------------------------------------------------ workspace build
echo ">> ensure ${RPI_HOST}:${REMOTE_WS}/src exists"
ssh "${RPI_HOST}" "mkdir -p '${REMOTE_WS}/src'"

echo ">> rsync src -> ${RPI_HOST}:${REMOTE_WS}/src"
# Exclude build products and the candle checkout (the RPi fetches candle
# itself, exactly like the Docker image, so host submodule state is irrelevant).
rsync -az --delete \
    --exclude 'build/' --exclude 'install/' --exclude 'log/' \
    --exclude '__pycache__/' \
    --exclude 'md80_hardware_interface/3rd_party/candle/' \
    --exclude 'wojtek_pc/' \
    "${HERE}/src/" "${RPI_HOST}:${REMOTE_WS}/src/"

# The policy store: downloaded policy snapshots, prefetched on this PC (python3
# -m wojtek_policy.policy_source <ref>). The RPi has no internet, so this rsync
# IS how policies get there. Skipped when there is no store here, so a fresh
# clone doesn't --delete a working robot's policies away.
if [ -d "${HERE}/policies" ]; then
    echo ">> rsync policies -> ${RPI_HOST}:${REMOTE_WS}/policies"
    rsync -az --delete "${HERE}/policies/" "${RPI_HOST}:${REMOTE_WS}/policies/"
else
    echo ">> no ${HERE}/policies here -- leaving the robot's policy store alone"
fi

echo ">> remote: ensure candle @ ${CANDLE_COMMIT}, rosdep, colcon build, restart"
ssh "${RPI_HOST}" \
    REMOTE_WS="${REMOTE_WS}" ROS_DISTRO="${ROS_DISTRO}" \
    CANDLE_COMMIT="${CANDLE_COMMIT}" RESTART="${RESTART}" 'bash -s' <<'REMOTE'
# Note: no `set -u` -- ROS setup.bash references unbound vars (AMENT_*).
set -eo pipefail
cd "${HOME}/${REMOTE_WS}"

CANDLE_DIR="src/md80_hardware_interface/3rd_party/candle"
if [ ! -e "${CANDLE_DIR}/CMakeLists.txt" ]; then
    echo "   cloning candle ..."
    rm -rf "${CANDLE_DIR}"
    git clone https://github.com/mabrobotics/candle "${CANDLE_DIR}"
fi
git -C "${CANDLE_DIR}" fetch --quiet origin || true
git -C "${CANDLE_DIR}" checkout --quiet "${CANDLE_COMMIT}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
rosdep install --from-paths src -y -i --rosdistro "${ROS_DISTRO}" || true
colcon build --symlink-install \
    --packages-up-to wojtek_bringup \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

if [ "${RESTART}" = "1" ]; then
    if systemctl list-unit-files wojtek-robot.service >/dev/null 2>&1 \
        && systemctl is-enabled wojtek-robot.service >/dev/null 2>&1; then
        echo "   restarting wojtek-robot.service ..."
        sudo systemctl restart wojtek-robot.service
    else
        echo "   wojtek-robot.service not installed/enabled -- skipping restart"
        echo "   (install with ./deploy.sh --provision, then enable when hw is ready)"
    fi
fi
echo ">> done"
REMOTE
