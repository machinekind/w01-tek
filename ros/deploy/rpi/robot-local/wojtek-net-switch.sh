#!/usr/bin/env bash
# wojtek-net-switch.sh -- RPi failover between the ethernet cable and its own
# wifi AP, plus an independent wifi station for internet. The RPi is a fixed
# anchor at 10.42.0.2 on WHICHEVER link to the PC is active:
#
#   eth0 carrier present (cable docked) -> talk over eth (static .2 from
#                                          netplan); AP torn down.
#   eth0 carrier absent  (no cable)     -> bring up the uap0 AP (wojtek-link)
#                                          so the PC/phone can associate.
#
# The onboard radio does double duty: wlan0 is a STATION on an upstream wifi
# (wpa_supplicant@wlan0 + networkd DHCP, see 30-wojtek-wlan0.network) and uap0
# is a virtual AP interface on the same radio. The station is never touched
# here -- it runs regardless of the cable.
#
# brcmfmac allows 1 station + 1 AP but only on ONE channel, so the AP follows
# the station's channel: when the station (re)associates on another channel,
# hostapd is restarted on it (AP clients drop for a few seconds). While the
# station is disconnected the AP keeps whatever channel it has.
#
# Idempotent -- safe to run repeatedly. Driven at boot + on eth0 link and wlan0
# association events by wojtek-net.service.
#
# Install: sudo cp wojtek-net-switch.sh /usr/local/sbin/ && sudo chmod +x /usr/local/sbin/wojtek-net-switch.sh
set -u

LINK_IF=eth0
STA_IF=wlan0
AP_IF=uap0
AP_ADDR=10.42.0.2/24
AP_CONF_BASE=/etc/hostapd/hostapd.conf
# hostapd runs from this rendered copy (base + the channel picked below);
# /etc/default/hostapd points DAEMON_CONF here.
AP_CONF_RUN=/run/hostapd-wojtek.conf

carrier() { cat "/sys/class/net/$1/carrier" 2>/dev/null || echo 0; }

freq2chan() {
    local f=$1
    if [ "$f" -eq 2484 ]; then echo 14
    elif [ "$f" -ge 2412 ] && [ "$f" -le 2472 ]; then echo $(( (f - 2407) / 5 ))
    elif [ "$f" -ge 5000 ]; then echo $(( (f - 5000) / 5 ))
    fi
}
# Channel of the station's current association (empty when not associated).
sta_chan() {
    local f
    f=$(iw dev "$STA_IF" link 2>/dev/null | awk '/freq:/{print int($2)}')
    [ -n "$f" ] && freq2chan "$f"
}
ap_chan()   { iw dev "$AP_IF" info 2>/dev/null | awk '/channel/{print $2}'; }
base_chan() { awk -F= '/^channel=/{print $2}' "$AP_CONF_BASE"; }

ap_up() {
    local want cur
    want=$(sta_chan)
    if systemctl is-active --quiet hostapd; then
        cur=$(ap_chan)
        # Station down: keep the AP where it is. Same channel: nothing to do.
        if [ -z "$want" ] || [ "$want" = "$cur" ]; then return 0; fi
        logger -t wojtek-net "station moved to channel ${want} -> moving AP from ${cur:-?}"
    else
        logger -t wojtek-net "cable absent -> bringing AP up on ${AP_IF}"
    fi
    want=${want:-$(base_chan)}
    local mode=g; [ "$want" -gt 14 ] && mode=a

    rfkill unblock wlan 2>/dev/null || true
    if [ ! -e "/sys/class/net/${AP_IF}" ]; then
        iw dev "$STA_IF" interface add "$AP_IF" type __ap \
            || { logger -t wojtek-net "creating ${AP_IF} failed"; return 1; }
    fi
    sed -e "s/^channel=.*/channel=${want}/" -e "s/^hw_mode=.*/hw_mode=${mode}/" \
        "$AP_CONF_BASE" | install -m600 /dev/stdin "$AP_CONF_RUN"

    # hostapd puts uap0 into AP mode and brings it up; add the anchor IP
    # afterwards, then (re)start DHCP/DNS so it binds the fresh interface.
    systemctl stop dnsmasq 2>/dev/null || true
    systemctl restart hostapd || { logger -t wojtek-net "hostapd failed to start (channel ${want})"; return 1; }
    ip addr replace "${AP_ADDR}" dev "${AP_IF}"
    systemctl start dnsmasq
    logger -t wojtek-net "AP up on ${AP_IF} channel ${want}"

    # brcmfmac: starting the AP beside an already-associated station leaves
    # the station link dead (still COMPLETED, but no traffic; the kernel logs
    # "GET STA INFO failed, -52"). Power save off plus one reassociation on the
    # same channel brings it back without dropping AP clients.
    if [ -n "$(sta_chan)" ]; then
        iw dev "$STA_IF" set power_save off 2>/dev/null || true
        wpa_cli -i "$STA_IF" reassociate >/dev/null 2>&1 || true
        logger -t wojtek-net "reassociated ${STA_IF} after AP start"
    fi
}

ap_down() {
    if ! systemctl is-active --quiet hostapd && [ -z "$(ip -4 -o addr show dev "$AP_IF" 2>/dev/null)" ]; then
        return 0
    fi
    logger -t wojtek-net "cable present -> tearing AP down on ${AP_IF}"
    systemctl stop dnsmasq 2>/dev/null || true
    systemctl stop hostapd 2>/dev/null || true
    ip addr flush dev "${AP_IF}" 2>/dev/null || true
    ip link set "${AP_IF}" down 2>/dev/null || true
}

iw dev "$STA_IF" set power_save off 2>/dev/null || true

if [ "$(carrier "${LINK_IF}")" = "1" ]; then
    ap_down
else
    ap_up
fi
