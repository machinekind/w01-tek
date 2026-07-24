#!/usr/bin/env bash
# Idempotent RPi provisioning for the wojtek robot. Runs ON the RPi (invoked by
# deploy.sh from the PC, or standalone). Safe to re-run: every phase guards on
# current state, so on an already-provisioned Pi it's a near no-op.
#
# Phases (skip any with the matching flag):
#   1 base packages : ROS 2 Jazzy + AP tools            (--skip-packages)
#   2 RT kernel     : Ubuntu Pro realtime-kernel        (--skip-kernel)
#   3 RT tuning     : cmdline isolcpus, rtprio limits, cpu governor (--skip-tuning)
#   4 network       : hostapd/dnsmasq/netplan + failover switch      (--skip-network)
#   5 robot service : install wojtek-robot.service (left disabled)   (--skip-service)
#   6 shell env     : ROS env in ~/.bashrc (domain 42 + CycloneDDS)  (--skip-shell-env)
#
# Needs the token for phase 2 (RT kernel), passed by deploy.sh from .env:
#   UBUNTU_PRO_TOKEN=<...> ./install.sh
#
#   ./install.sh              # provision everything, print if a reboot is needed
#   ./install.sh --dry-run    # show what would change, touch nothing
#   ./install.sh --reboot     # reboot automatically at the end if required
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
DRY_RUN=0
DO_REBOOT=0
SKIP_PACKAGES=0 SKIP_KERNEL=0 SKIP_TUNING=0 SKIP_NETWORK=0 SKIP_SERVICE=0 SKIP_SHELL_ENV=0
ENABLE_ROBOT=0
REBOOT_NEEDED=0

for a in "$@"; do case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --reboot) DO_REBOOT=1 ;;
    --skip-packages) SKIP_PACKAGES=1 ;;
    --skip-kernel) SKIP_KERNEL=1 ;;
    --skip-tuning) SKIP_TUNING=1 ;;
    --skip-network) SKIP_NETWORK=1 ;;
    --skip-service) SKIP_SERVICE=1 ;;
    --skip-shell-env) SKIP_SHELL_ENV=1 ;;
    # Opt-in: also start the RT stack automatically on boot. Off by default --
    # `ros2 run wojtek_bringup robot` starts it per-session, and leaving the RPi
    # free lets you run other launches on it during development.
    --enable-robot) ENABLE_ROBOT=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

say()  { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
# run: execute, or just print under --dry-run.
run()  { if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else eval "$@"; fi; }

# ---------------------------------------------------------------- phase 1
provision_packages() {
    say "Phase 1: base packages (ROS 2 ${ROS_DISTRO} + AP tools)"
    if [ ! -f /etc/apt/sources.list.d/ros2.list ] && [ ! -f /etc/apt/sources.list.d/ros2.sources ]; then
        info "adding ROS 2 apt repo"
        run "sudo apt-get update -qq"
        run "sudo apt-get install -y curl gnupg lsb-release"
        run "sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"
        run "echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) main\" | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null"
    else
        info "ROS 2 apt repo already present"
    fi
    run "sudo apt-get update -qq"
    local ros_pkgs="ros-${ROS_DISTRO}-ros-base ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-realtime-tools"
    # setserial: candle runs `setserial <port> low_latency` on the CANdle's
    # ttyACM at startup; without it the driver falls back to "low-speed mode"
    # (see md80 bring-up logs), adding latency to the 400 Hz loop. The PC image
    # already ships it (docker/Dockerfile) -- the RPi is where it actually matters.
    local tools="python3-colcon-common-extensions python3-rosdep build-essential git setserial python3-pip"
    local ap="iw hostapd dnsmasq rfkill"
    run "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ${ros_pkgs} ${tools} ${ap}"
    # policy_node resolves policies by HF repo id (wojtek_policy/
    # policy_source.py); same pip3 install the PC image does.
    run "pip3 install --break-system-packages huggingface_hub"
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        run "sudo rosdep init"
    fi
    run "rosdep update || true"
}

# ---------------------------------------------------------------- phase 2
provision_kernel() {
    say "Phase 2: realtime kernel (Ubuntu Pro)"
    if uname -v | grep -q PREEMPT_RT; then
        info "already running a PREEMPT_RT kernel ($(uname -r)) -- skipping"
        return
    fi
    if ! command -v pro >/dev/null 2>&1; then
        run "sudo apt-get install -y ubuntu-advantage-tools"
    fi
    if ! sudo pro status --format json 2>/dev/null | grep -q '"attached": true'; then
        if [ -z "${UBUNTU_PRO_TOKEN:-}" ]; then
            echo "!! UBUNTU_PRO_TOKEN not set -- cannot enable the RT kernel." >&2
            echo "   Put it in .env (see .env.example) or export it, then re-run." >&2
            exit 1
        fi
        info "attaching to Ubuntu Pro"
        run "sudo pro attach '${UBUNTU_PRO_TOKEN}' --no-auto-enable"
    else
        info "already attached to Ubuntu Pro"
    fi
    info "enabling realtime-kernel (raspi variant)"
    run "sudo pro enable realtime-kernel --variant=raspi --assume-yes"
    REBOOT_NEEDED=1
}

# ---------------------------------------------------------------- phase 3
provision_tuning() {
    say "Phase 3: RT tuning (cmdline / limits / governor)"
    local cmdline=/boot/firmware/cmdline.txt
    local want="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
    if [ -f "$cmdline" ] && grep -q "isolcpus=2,3" "$cmdline"; then
        info "cmdline core-isolation already present"
    elif [ -f "$cmdline" ]; then
        info "appending core-isolation to $cmdline (reboot needed)"
        # cmdline.txt must stay a SINGLE line -- append to it in place.
        run "sudo sed -i \"1 s|\$| ${want}|\" '${cmdline}'"
        REBOOT_NEEDED=1
    else
        info "WARN: $cmdline not found -- skipping core-isolation"
    fi

    local limits=/etc/security/limits.d/99-realtime.conf
    if [ -f "$limits" ] && grep -q "rtprio" "$limits"; then
        info "rtprio/memlock limits already present"
    else
        info "writing $limits"
        run "printf 'rpi   -   rtprio    99\nrpi   -   memlock   unlimited\n' | sudo tee '${limits}' >/dev/null"
    fi

    if systemctl is-enabled cpu-performance.service >/dev/null 2>&1; then
        info "cpu-performance.service already enabled"
    else
        info "installing cpu-performance.service"
        run "sudo cp '${HERE}/cpu-performance.service' /etc/systemd/system/cpu-performance.service"
        run "sudo systemctl enable cpu-performance.service"
    fi
}

# ---------------------------------------------------------------- phase 4
provision_network() {
    say "Phase 4: network (AP failover: eth cable <-> wlan0 AP)"
    run "sudo install -m600 '${HERE}/hostapd.conf'        /etc/hostapd/hostapd.conf"
    run "sudo install -m644 '${HERE}/dnsmasq-wojtek.conf' /etc/dnsmasq.d/wojtek.conf"
    run "sudo install -m755 '${HERE}/wojtek-net-switch.sh' /usr/local/sbin/wojtek-net-switch.sh"
    run "sudo install -m644 '${HERE}/wojtek-net.service'  /etc/systemd/system/wojtek-net.service"
    run "echo 'DAEMON_CONF=/etc/hostapd/hostapd.conf' | sudo tee /etc/default/hostapd >/dev/null"
    run "sudo systemctl unmask hostapd 2>/dev/null || true"
    # hostapd/dnsmasq are driven on-demand by the switch -- never auto-start.
    run "sudo systemctl disable hostapd dnsmasq 2>/dev/null || true"

    # netplan: static eth0 (.2), wlan0 handed to hostapd. Only (re)apply when
    # the effective config differs, and refuse the disruptive first switch if
    # we're reachable over the very eth we'd renumber (that must be done by
    # cloud-init at first boot, or with a console attached).
    run "sudo install -m600 '${HERE}/60-wojtek.yaml' /etc/netplan/60-wojtek.yaml"
    run "echo 'network: {config: disabled}' | sudo tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg >/dev/null"
    run "sudo rm -f /etc/netplan/50-cloud-init.yaml"
    if ip -4 addr show eth0 2>/dev/null | grep -q '10.42.0.2/24'; then
        info "eth0 already at 10.42.0.2 -- applying netplan (no renumber)"
        run "sudo netplan apply"
    else
        info "WARN: eth0 is NOT yet 10.42.0.2. 'netplan apply' would renumber the"
        info "      link and could drop your SSH. Skipping apply -- reboot to pick"
        info "      it up, or run 'sudo netplan apply' from a console."
    fi
    run "sudo systemctl daemon-reload"
    run "sudo systemctl enable wojtek-net.service"
    run "sudo systemctl start wojtek-net.service || true"
}

# ---------------------------------------------------------------- phase 5
provision_service() {
    say "Phase 5: robot control service"
    run "sudo install -m644 '${HERE}/../wojtek-robot.service' /etc/systemd/system/wojtek-robot.service"
    run "sudo systemctl daemon-reload"
    # NOT auto-started on boot by default: `ros2 run wojtek_bringup robot` starts
    # the RT stack per-session (via this service), and leaving the RPi free lets
    # you run other launches on it during development. Opt into boot auto-start
    # with --enable-robot if you ever want it always-on.
    if [ "${ENABLE_ROBOT}" = 1 ]; then
        run "sudo systemctl enable wojtek-robot.service"
        info "installed + ENABLED on boot (--enable-robot; DISARMED until /wojtek/arm)"
    else
        info "installed, not enabled -- started per-session by 'ros2 run wojtek_bringup robot'"
    fi
}

# ---------------------------------------------------------------- phase 6
provision_shell_env() {
    say "Phase 6: ROS env for interactive shells (~/.bashrc)"
    local marker="# wojtek-ros-env"
    if grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
        info "~/.bashrc already has the ROS env block"
        return
    fi
    # Manual shells must match wojtek-robot.service (domain 42 + CycloneDDS);
    # without this a manual `ros2 topic list` lands on domain 0 / Fast DDS and
    # sees NOTHING from the running stack (cost us hours once).
    local block
    block="$(cat <<EOF

${marker} (added by install.sh; keep in sync with wojtek-robot.service)
source /opt/ros/${ROS_DISTRO}/setup.bash
[ -f "\$HOME/wojtek_ws/install/setup.bash" ] && source "\$HOME/wojtek_ws/install/setup.bash"
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
EOF
)"
    if [ "$DRY_RUN" = 1 ]; then
        printf '   [dry-run] append to ~/.bashrc:%s\n' "$block"
    else
        info "appending ROS env block to ~/.bashrc"
        printf '%s\n' "$block" >> "$HOME/.bashrc"
    fi
}

# ---------------------------------------------------------------- main
[ "$DRY_RUN" = 1 ] && say "DRY RUN -- no changes will be made"
[ "$SKIP_PACKAGES" = 1 ] || provision_packages
[ "$SKIP_KERNEL"   = 1 ] || provision_kernel
[ "$SKIP_TUNING"   = 1 ] || provision_tuning
[ "$SKIP_NETWORK"  = 1 ] || provision_network
[ "$SKIP_SERVICE"  = 1 ] || provision_service
[ "$SKIP_SHELL_ENV" = 1 ] || provision_shell_env

say "Provisioning done."
if [ "$REBOOT_NEEDED" = 1 ]; then
    if [ "$DO_REBOOT" = 1 ] && [ "$DRY_RUN" != 1 ]; then
        info "reboot required -> rebooting now"
        sudo reboot
    else
        info "REBOOT REQUIRED to activate the RT kernel / core-isolation."
        info "Run: sudo reboot   (or re-run with --reboot)"
    fi
fi
