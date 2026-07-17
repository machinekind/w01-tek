#!/usr/bin/env bash
# wojtek-net-switch.sh -- RPi failover between the ethernet cable and its own
# wifi AP. The RPi is a fixed anchor at 10.42.0.2 on WHICHEVER link is active:
#
#   eth0 carrier present (cable docked) -> talk over eth (static .2 from
#                                          netplan); AP torn down.
#   eth0 carrier absent  (no cable)     -> bring up the wlan0 AP (wojtek-link)
#                                          so the PC/phone can associate.
#
# Idempotent -- safe to run repeatedly. Driven at boot + on link changes by
# wojtek-net.service.
#
# Install: sudo cp wojtek-net-switch.sh /usr/local/sbin/ && sudo chmod +x /usr/local/sbin/wojtek-net-switch.sh
set -u

LINK_IF=eth0
AP_IF=wlan0
AP_ADDR=10.42.0.2/24

carrier() { cat "/sys/class/net/$1/carrier" 2>/dev/null || echo 0; }

ap_up() {
    if systemctl is-active --quiet hostapd; then return 0; fi
    logger -t wojtek-net "cable absent -> bringing AP up on ${AP_IF}"
    rfkill unblock wlan 2>/dev/null || true
    # hostapd puts wlan0 into AP mode and brings it up; add the anchor IP
    # afterwards, then start DHCP.
    systemctl start hostapd || { logger -t wojtek-net "hostapd failed to start"; return 1; }
    ip addr replace "${AP_ADDR}" dev "${AP_IF}"
    systemctl start dnsmasq
}

ap_down() {
    logger -t wojtek-net "cable present -> tearing AP down on ${AP_IF}"
    systemctl stop dnsmasq 2>/dev/null || true
    systemctl stop hostapd 2>/dev/null || true
    ip addr flush dev "${AP_IF}" 2>/dev/null || true
    ip link set "${AP_IF}" down 2>/dev/null || true
}

if [ "$(carrier "${LINK_IF}")" = "1" ]; then
    ap_down
else
    ap_up
fi
