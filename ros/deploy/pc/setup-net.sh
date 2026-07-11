#!/usr/bin/env bash
# PC-side network for the wojtek link. Idempotent -- safe to re-run.
#
# Creates the 'wojtek-eth' NetworkManager profile: shared mode, so the PC is
# 10.42.0.1 and serves DHCP + NAT to the RPi docked by cable. autoconnect is
# OFF on purpose -- otherwise NM grabs every cable (incl. your home internet)
# as the wojtek net and its 10.42.0.0/24 route hijacks SSH to the robot.
#
#   ./setup-net.sh            # autodetect the ethernet device
#   ./setup-net.sh eno1       # or name it explicitly
#
# After this, a normal internet cable stays on the default DHCP profile
# automatically; dock the robot by cable with:  nmcli con up wojtek-eth
set -euo pipefail

CON=wojtek-eth

command -v nmcli >/dev/null || { echo "nmcli not found -- this PC isn't on NetworkManager." >&2; exit 1; }

IFACE="${1:-}"
if [ -z "${IFACE}" ]; then
    IFACE="$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="ethernet"{print $1; exit}')"
fi
[ -z "${IFACE}" ] && { echo "no ethernet device found -- pass one: ./setup-net.sh <iface>" >&2; exit 1; }
echo ">> ethernet device: ${IFACE}"

if nmcli -t -f NAME connection show | grep -qx "${CON}"; then
    echo ">> '${CON}' exists -- enforcing settings"
    nmcli connection modify "${CON}" \
        connection.interface-name "${IFACE}" \
        ipv4.method shared \
        connection.autoconnect no
else
    echo ">> creating '${CON}' on ${IFACE}"
    nmcli connection add type ethernet ifname "${IFACE}" con-name "${CON}" \
        ipv4.method shared \
        connection.autoconnect no
fi

# Make sure the default wired profile (whatever locale-specific name it has)
# stays the autoconnect default so a plain internet cable "just works". We
# don't rename it; we only ensure wojtek-eth won't outrank it.
echo ">> done."
echo "   Dock the robot by cable:   nmcli con up ${CON}"
echo "   Back to home internet:     nmcli con up <your default wired profile>  (or replug)"
