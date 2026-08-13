#!/usr/bin/env bash
# Idempotent RPi provisioning for the wojtek robot. Runs ON the RPi (invoked by
# deploy.sh from the PC, or standalone). Safe to re-run: every phase guards on
# current state, so on an already-provisioned Pi it's a near no-op.
#
# Phases (skip any with the matching flag):
#   1 base packages : ROS 2 Jazzy + AP tools            (--skip-packages)
#   2 RT kernel     : Ubuntu Pro realtime-kernel        (--skip-kernel)
#   3 RT tuning     : cmdline isolcpus, rtprio limits, cpu governor, (--skip-tuning)
#                     Pi 3: wojtek-affinity core-partition daemon
#   4 network       : hostapd/dnsmasq/netplan + failover switch      (--skip-network)
#   5 robot service : install wojtek-robot.service + the local       (--skip-service)
#                     drop-in (folded boot, bag off, /home/rpi/policy
#                     symlink); boot autostart only with --enable-robot
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
    # Boot autostart is DECLARATIVE: this flag states the state you want, and
    # phase 5 makes the machine match it. --enable-robot means "start at
    # power-on", its absence means "do not" -- so a provision run without the
    # flag actively disables a service that was enabled. One flag, one
    # meaning, no dependence on what the last person left behind.
    --enable-robot) ENABLE_ROBOT=1 ;;
    --no-enable-robot) ENABLE_ROBOT=0 ;;   # accepted, now the default
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
    # ros-joy: the joystick driver behind wojtek_teleop's pad teleop
    # (robot.launch.py gamepad:=true) -- the pad pairs with the RPi itself.
    local ros_pkgs="ros-${ROS_DISTRO}-ros-base ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-realtime-tools ros-${ROS_DISTRO}-joy"
    # setserial: candle runs `setserial <port> low_latency` on the CANdle's
    # ttyACM at startup; without it the driver falls back to "low-speed mode"
    # (see md80 bring-up logs), adding latency to the 400 Hz loop. The PC image
    # already ships it (docker/Dockerfile) -- the RPi is where it actually matters.
    local tools="python3-colcon-common-extensions python3-rosdep build-essential git setserial python3-pip"
    local ap="iw hostapd dnsmasq rfkill"
    # Xbox pad over bluetooth: bluez for pairing (one-time, manual --
    # bluetoothctl: scan on -> pair -> trust -> connect), joystick/evtest to
    # verify the /dev/input device afterwards.
    local pad="bluez joystick evtest"
    run "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ${ros_pkgs} ${tools} ${ap} ${pad}"
    # policy_node resolves policies by HF repo id (wojtek_policy/
    # policy_source.py); same pip3 install the PC image does.
    run "pip3 install --break-system-packages huggingface_hub"

    # joy_node reads the pad's /dev/input/event* (root:input 0660) -- without
    # the input group it opens nothing and /joy just stays silent (seen on
    # hardware: pad connected, nodes up, zero messages). Takes effect on the
    # next login, which every `robot` ssh session is.
    if id -nG rpi | grep -qw input; then
        info "rpi already in the input group"
    else
        info "adding rpi to the input group (joystick event devices)"
        run "sudo usermod -aG input rpi"
    fi

    # The RealSense driver opens the camera's /dev/video* nodes (root:video
    # 0660). Without the video group librealsense enumerates nothing and says
    # "No RealSense devices were found!" after a wall of "Permission denied"
    # -- which reads like an unplugged camera, not a permissions problem
    # (seen on hardware 2026-07-30). Takes effect on the next login/service
    # start, same as the input group above.
    if id -nG rpi | grep -qw video; then
        info "rpi already in the video group"
    else
        info "adding rpi to the video group (RealSense /dev/video*)"
        run "sudo usermod -aG video rpi"
    fi

    # udev for the D435: group ownership of the video nodes made explicit, and
    # libusb write access so hardware_reset/firmware/advanced-mode work at all
    # (see the rules file for why the camera streams fine without it).
    local rsrules=/etc/udev/rules.d/99-wojtek-realsense.rules
    if cmp -s "${HERE}/99-wojtek-realsense.rules" "$rsrules"; then
        info "realsense udev rules already current"
    else
        info "installing $rsrules"
        run "sudo install -m644 '${HERE}/99-wojtek-realsense.rules' '$rsrules'"
        # Reload + trigger so an already-plugged camera picks the rules up
        # without a replug; on a robot the socket is not always reachable.
        run "sudo udevadm control --reload-rules"
        run "sudo udevadm trigger --subsystem-match=usb --subsystem-match=video4linux"
    fi

    # Xbox pads refuse to stay connected while bluetooth ERTM is on -- the
    # standard fix is to turn it off module-wide (takes effect after reboot
    # or btusb reload; pairing before that just keeps disconnecting).
    local ertm=/etc/modprobe.d/xbox-pad-bt.conf
    if [ -f "$ertm" ]; then
        info "bluetooth ERTM already disabled for the pad"
    else
        info "writing $ertm (disable_ertm=1, reboot needed)"
        run "echo 'options bluetooth disable_ertm=1' | sudo tee '${ertm}' >/dev/null"
        REBOOT_NEEDED=1
    fi
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
    # Ubuntu Pro's realtime-kernel raspi variant supports Pi 4 / Pi 5 only.
    # On older boards (Pi 3) `pro enable` fails hard, so skip with a warning
    # instead of aborting the whole provision run.
    local model
    model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    case "${model}" in
        *"Raspberry Pi 4"*|*"Raspberry Pi 5"*) ;;
        *)
            info "WARN: '${model:-unknown board}' is not supported by the Pro"
            info "      realtime-kernel (Pi 4/5 only) -- skipping. The stack"
            info "      runs on the stock kernel without RT guarantees."
            return ;;
    esac
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

    # IMU I2C at Fast Mode 400 kHz (both LSM6DS3TR-C and LIS3MDL support it).
    # The default 100 kHz makes the 23-byte status+data burst in the driver's
    # read() take ~2.3 ms -- right at the 400 Hz controller_manager budget
    # (overruns seen on hardware); 400 kHz brings it down to ~0.6 ms.
    local bootcfg=/boot/firmware/config.txt
    if [ -f "$bootcfg" ] && grep -q "i2c_arm_baudrate=400000" "$bootcfg"; then
        info "i2c 400 kHz already configured"
    elif [ -f "$bootcfg" ]; then
        info "setting i2c_arm baudrate 400 kHz in $bootcfg (reboot needed)"
        run "printf 'dtparam=i2c_arm=on,i2c_arm_baudrate=400000\n' | sudo tee -a '${bootcfg}' >/dev/null"
        REBOOT_NEEDED=1
    else
        info "WARN: $bootcfg not found -- skipping i2c baudrate"
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

    # Pi 3 only: isolcpus disables load balancing on the isolated cores, so
    # the whole taskset-2,3 stack piles onto core 2 while core 3 idles. The
    # A53 cannot hide that (122/200 Hz loop, 20 Hz policy tick); a loop
    # daemon re-partitions the nodes -- see wojtek-affinity.sh. Pi 4/5 cores
    # are fast enough that the pileup never showed, keep them stock.
    local model
    model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    case "${model}" in
        *"Raspberry Pi 3"*)
            info "Pi 3: installing the core-partition daemon (wojtek-affinity)"
            run "sudo install -m755 '${HERE}/wojtek-affinity.sh' /usr/local/sbin/wojtek-affinity.sh"
            run "sudo install -m644 '${HERE}/wojtek-affinity.service' /etc/systemd/system/wojtek-affinity.service"
            run "sudo systemctl daemon-reload"
            run "sudo systemctl enable wojtek-affinity.service"
            ;;
        *) info "not a Pi 3 -- skipping the core-partition daemon" ;;
    esac
}

# ---------------------------------------------------------------- phase 4
provision_network() {
    say "Phase 4: network (AP failover: eth cable <-> wlan0 AP)"
    if [ -z "${WOJTEK_AP_PSK:-}" ]; then
        echo "!! WOJTEK_AP_PSK not set -- the wlan0 AP needs a passphrase." >&2
        echo "   Put it in .env (see .env.example) or export it, then re-run." >&2
        exit 1
    fi
    run "sed 's|__WOJTEK_AP_PSK__|${WOJTEK_AP_PSK}|' '${HERE}/hostapd.conf' | sudo install -m600 /dev/stdin /etc/hostapd/hostapd.conf"
    run "sudo install -m644 '${HERE}/dnsmasq-wojtek.conf' /etc/dnsmasq.d/wojtek.conf"
    run "sudo install -m755 '${HERE}/wojtek-net-switch.sh' /usr/local/sbin/wojtek-net-switch.sh"
    run "sudo install -m644 '${HERE}/wojtek-net.service'  /etc/systemd/system/wojtek-net.service"
    run "echo 'DAEMON_CONF=/etc/hostapd/hostapd.conf' | sudo tee /etc/default/hostapd >/dev/null"
    run "sudo systemctl unmask hostapd 2>/dev/null || true"
    # hostapd/dnsmasq are driven on-demand by the switch -- never auto-start.
    run "sudo systemctl disable hostapd dnsmasq 2>/dev/null || true"

    # CycloneDDS interface pinning (lo + optional eth0/wlan0): survives the
    # eth-AP failover moving 10.42.0.2 between ifaces. Keep in sync with the
    # CYCLONEDDS_URI lines in wojtek-robot.service and the bashrc block.
    run "sudo install -m644 '${HERE}/cyclonedds-rpi.xml' /etc/cyclonedds-rpi.xml"

    # UDP socket buffers for large samples (camera images). The kernel default
    # is 208 KB, while Cyclone asks for ~1 MB and gets silently clamped. One
    # 848x480 rgb8 frame is 1.22 MB = ~840 UDP fragments, and losing a single
    # fragment discards the whole frame -- which looks like "the image stutters
    # in RViz" and gets misdiagnosed as bandwidth. Applies to any big message,
    # not just the camera; the control loop's own topics are tiny and were
    # never affected.
    run "sudo install -m644 '${HERE}/60-wojtek-dds-buffers.conf' /etc/sysctl.d/60-wojtek-dds-buffers.conf"
    run "sudo sysctl --quiet --system"

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
    # Robot-local overrides (folded boot pose, bag off, /home/rpi/policy
    # symlink as the policy reference) -- see wojtek-robot-local.conf.
    # The hand-written 10-policy.conf from the pre-repo era is retired.
    run "sudo mkdir -p /etc/systemd/system/wojtek-robot.service.d"
    run "sudo install -m644 '${HERE}/wojtek-robot-local.conf' /etc/systemd/system/wojtek-robot.service.d/10-local.conf"
    run "sudo rm -f /etc/systemd/system/wojtek-robot.service.d/10-policy.conf"
    run "sudo systemctl daemon-reload"
    # The unit is always installed/refreshed; --enable-robot decides whether it
    # runs at power-on, and this phase enforces that state either way. Running
    # a provision without the flag is therefore how you turn boot autostart
    # OFF -- it is a statement of intent, not an omission.
    if [ "${ENABLE_ROBOT}" = 1 ]; then
        run "sudo systemctl enable wojtek-robot.service"
        info "installed + ENABLED on boot (DISARMED until /wojtek/arm)"
    else
        run "sudo systemctl disable wojtek-robot.service 2>/dev/null || true"
        info "installed + DISABLED on boot (no --enable-robot)"
        info "  (per-session: 'ros2 run wojtek_bringup robot')"
        # Disabling only governs boot. A service running right now keeps
        # running with the code it started with, which is not what anyone
        # means by "disable it" -- so say so rather than leave it implied.
        if systemctl is-active wojtek-robot.service >/dev/null 2>&1; then
            info "  NOTE: the service is RUNNING right now -- it will not come"
            info "        back after a reboot, but stop it explicitly if you"
            info "        want it down now: sudo systemctl stop wojtek-robot.service"
        fi
    fi
}

# ---------------------------------------------------------------- phase 6
provision_shell_env() {
    say "Phase 6: ROS env for interactive shells (~/.bashrc)"
    local marker="# wojtek-ros-env"
    if grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
        # Upgrade path: the block predates the pinned CycloneDDS profile.
        if ! grep -qF "CYCLONEDDS_URI" "$HOME/.bashrc"; then
            info "adding CYCLONEDDS_URI to the existing ~/.bashrc block"
            run "printf 'export CYCLONEDDS_URI=file:///etc/cyclonedds-rpi.xml\n' >> '$HOME/.bashrc'"
        else
            info "~/.bashrc already has the ROS env block"
        fi
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
export CYCLONEDDS_URI=file:///etc/cyclonedds-rpi.xml
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

# Flush everything provisioned to disk NOW: a hard power cut minutes after
# provisioning once left /etc/netplan/60-wojtek.yaml and the robot unit as
# 0-byte files (ext4 delayed allocation) -- the RPi then booted with no
# network config and no runnable service.
run "sync"

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
