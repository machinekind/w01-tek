#!/bin/bash
# wojtek-cm-watchdog: stop wojtek-robot when the drive link dies, and start
# it again when the motor power switch is cycled.
#
# Three ways the control stack loses the drives, and what this loop sees:
#
#   1. ros2_control_node crashes (SPI dropout during bring-up, a throw):
#      the launch keeps the other nodes up, so the service stays "active"
#      while the controller manager (CM) process is gone.
#   2. Motor power is cut under a live stack: since the driver's link
#      watchdog (md80_hardware_interface, link_timeout_cycles) the CM
#      survives, but MD80HardwareInterface leaves the ACTIVE state -- the
#      CM publishes that on /controller_manager/activity.
#   3. The CANdle HAT resets under a live stack (seen 2026-09-21 on a 5 V
#      dip): same as 2 from the outside.
#
# In every case the fix is a fresh bring-up, and a fresh bring-up only
# works once the drives answer. So: on a dead CM or a non-active hardware
# component, stop the service (frees the SPI bus and leaves the drives
# disabled), then probe the drives with mdtool once a second (as the robot
# user: its ini pins bus=SPI) and start the service again when the motor
# power has been switched off and on by hand: the drives were silent for
# at least SILENT_MIN_S and then answer. Why that and not "when they
# answer": the service boots with boot_pose:=folded and zeroes the
# encoders assuming that pose, and a robot that lost its drives under load
# is lying wherever it fell, possibly in someone's hands. The version that
# ran on the robot until 2026-10-27 started the service as soon as the
# drives answered, so a one-second dip in the motor supply (the drives
# drop out, the link is lost, the supply is back before this loop even
# looks) brought the stack up again on a robot nobody had laid down, with
# a wrong zero on every joint. Such a dip never shows SILENT_MIN_S of
# silence here; a hand on the switch does. The operator's procedure after
# any lost link: lay the robot down folded, motor power off, count to
# five, motor power on. The stack comes up by itself, no ssh.
#
# Two signals for "component not active", cheapest first:
#   * the CM's own journal: the driver logs "Drive link lost" and the CM
#     logs "read cycle resulted in an error" the moment it happens;
#   * the CM's latched /controller_manager/activity topic, read with the
#     python ros2 CLI. On a loaded Pi 4 that CLI alone takes ~20 s to come
#     up (measured 2026-09-22 with the deck + camera on cores 0-1), so a
#     silent probe is NOT evidence of a hung CM: it is logged, never acted
#     on. Acting on it once stopped a healthy stack mid-session.
#
# Do not install by hand: install.sh puts this into /usr/local/sbin and
# enables wojtek-cm-watchdog.service.

ROBOT_USER="${ROBOT_USER:-rpi}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WS="/home/${ROBOT_USER}/wojtek_ws"
HW_NAME="MD80HardwareInterface"
SERVICE="wojtek-robot.service"
# The CLI probe gets this long; see the header for why it is generous.
PROBE_TIMEOUT_S=60
# ...and runs at most this often: it costs half a core for its whole
# duration (python + DDS discovery), on the cores the deck and the camera
# live on. The journal check below is free and runs every loop.
PROBE_EVERY_S=60
# A fresh bring-up needs this long before its hardware is expected ACTIVE;
# a younger service is never judged.
GRACE_S=60
# With the service down the drives are probed this often (the bus is
# free, mdtool is cheap), and the motor power must have been off for at
# least this long before answering drives count as a switch cycle.
DOWN_POLL_S=1
SILENT_MIN_S=3

log() { echo "$*" | systemd-cat -t wojtek-cm-watchdog; }

service_age_s() {
    local since
    since="$(systemctl show "${SERVICE}" -p ActiveEnterTimestampMonotonic --value 2>/dev/null)"
    [ -n "${since}" ] && [ "${since}" != 0 ] || { echo 0; return; }
    local now
    now="$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)"
    echo $(( (now - since) / 1000000 ))
}

cm_alive() { pgrep -f '/controller_manager/ros2_control_node' >/dev/null; }

# Has the running service logged a drive-link failure? The driver's own
# line and the controller manager's reaction to a failed read() -- both
# appear within one control cycle of the event, no DDS needed.
link_lost_logged() {
    local since
    since="$(systemctl show "${SERVICE}" -p ActiveEnterTimestamp --value 2>/dev/null)"
    [ -n "${since}" ] || return 1
    journalctl -u "${SERVICE}" --since "${since}" --no-pager -o cat 2>/dev/null \
        | grep -q -E "Drive link lost|read cycle resulted in an error"
}

# Prints the hardware component's lifecycle label ("active", "unconfigured",
# ...) from the CM's latched activity topic, or nothing when the CM does
# not answer in time (a hung CM counts as dead for our purposes).
hw_state() {
    runuser -u "${ROBOT_USER}" -- /usr/bin/taskset -c 0,1 /bin/bash -c "
        source /opt/ros/${ROS_DISTRO}/setup.bash
        source ${WS}/install/setup.bash
        export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
               CYCLONEDDS_URI=file:///etc/cyclonedds-rpi.xml
        timeout ${PROBE_TIMEOUT_S} ros2 topic echo --once \
            --qos-durability transient_local --qos-reliability reliable \
            /controller_manager/activity 2>/dev/null
    " | awk -v hw="${HW_NAME}" '
        /^hardware_components:/ { in_hw = 1; next }
        in_hw && /name:/  { name = $NF }
        in_hw && /label:/ { if (name == hw) print $NF }
    '
}

drives_answer() {
    runuser -u "${ROBOT_USER}" -- /usr/local/bin/mdtool ping all 2>/dev/null \
        | grep -q "Found drives"
}

sleep 30
poll_s=10
silent_since=0
drives_back_logged=0
misses=0
last_probe=0
while true; do
    sleep "${poll_s}"

    if systemctl is-active --quiet "${SERVICE}"; then
        poll_s=10
        if cm_alive; then
            # A healthy CM with a healthy component: nothing to do. Judge the
            # component only once bring-up had its chance.
            [ "$(service_age_s)" -ge "${GRACE_S}" ] || continue
            if link_lost_logged; then
                log "drive link lost (CM journal) -- stopping ${SERVICE}"
            else
                now="$(date +%s)"
                [ $((now - last_probe)) -ge "${PROBE_EVERY_S}" ] || continue
                last_probe="${now}"
                state="$(hw_state)"
                if [ "${state}" = "active" ]; then misses=0; continue; fi
                if [ -z "${state}" ]; then
                    # A silent probe is what a loaded Pi looks like, not a
                    # hung CM (see the header). Never stop on it: a stopped
                    # stack drops a standing robot. Say so after a while.
                    misses=$((misses + 1))
                    if [ "${misses}" -eq 3 ]; then
                        log "activity probe silent 3x in a row (CM overloaded or hung?) -- watching, not acting"
                    fi
                    continue
                fi
                log "${HW_NAME} is '${state}', not active -- drive link lost; stopping ${SERVICE}"
            fi
        else
            log "CM dead under a running ${SERVICE} -- stopping it"
        fi
        systemctl stop "${SERVICE}"
        poll_s="${DOWN_POLL_S}"
        silent_since=0
        drives_back_logged=0
        misses=0
        continue
    fi

    # Service down (we stopped it, or it never came up). mdtool needs the
    # SPI bus, which is free now. Start only after a hand on the motor power
    # switch: silence of at least SILENT_MIN_S, then answering drives.
    poll_s="${DOWN_POLL_S}"
    now="$(date +%s)"
    if drives_answer; then
        silent_s=0
        [ "${silent_since}" = 0 ] || silent_s=$((now - silent_since))
        silent_since=0
        if [ "${silent_s}" -ge "${SILENT_MIN_S}" ]; then
            if systemctl is-enabled --quiet "${SERVICE}"; then
                log "motor power back after ${silent_s}s off -- starting ${SERVICE} (zeroing assumes the robot was laid down folded)"
                drives_back_logged=0
                systemctl start "${SERVICE}"
                poll_s="${GRACE_S}"
            elif [ "${drives_back_logged}" = 0 ]; then
                # A bench robot (provisioned without --enable-robot) does not
                # start its stack on the power switch either.
                log "motor power back, but ${SERVICE} is disabled (no boot autostart) -- not starting it"
                drives_back_logged=1
            fi
        elif [ "${drives_back_logged}" = 0 ]; then
            log "drives answer; ${SERVICE} stays stopped -- lay the robot down folded, switch the motor power off for ${SILENT_MIN_S}s or more, then on: the stack starts by itself"
            drives_back_logged=1
        fi
    elif [ "${silent_since}" = 0 ]; then
        silent_since="${now}"
        drives_back_logged=0
        log "drives SILENT -- motor power off; the stack starts when it is switched on again"
    fi
done
