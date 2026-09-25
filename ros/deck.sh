#!/usr/bin/env bash
# Work with the Steam Deck that runs the wojtek_deck panel, from the PC.
#
#   ./deck.sh                    log in
#   ./deck.sh install            put the Deck-side helpers and icons on it
#   ./deck.sh where              find the Deck on this network and remember it
#   ./deck.sh panel [url]        open the panel (default: this PC's simulation)
#   ./deck.sh stop|reload        close / reload the panel
#   ./deck.sh steam off|on       close Steam so the browser can read the pad
#   ./deck.sh shot [file]        screenshot the Deck and bring it here
#   ./deck.sh run <command>      run anything over there
#
# The Deck is NOT a fixed anchor the way the RPi is: it takes whatever address
# its DHCP hands out, and its sshd is a user-level one started by hand that
# dies on every reboot. So the address is never written down here. Put it in
# .env if you want to skip the search:
#
#   DECK_HOST=user@address       # see .env.example
#
# Without it, this looks for the one host on any network this PC is on
# answering on the ssh port -- wifi and cable both, since the Deck may sit
# on either -- and remembers it in ~/.config/wojtek/deck-host until it stops
# answering. Nothing about the machine is committed: this repository is
# public (see CLAUDE.md).
#
# If nothing answers at all, the Deck's sshd is down and no command from here
# can start it -- that is what the "Deck SSH on" icon ./deck.sh install puts
# on its desktop is for. Tap it there, then come back.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Local overrides (DECK_HOST, DECK_SSH_PORT, ...); never committed.
if [ -f "${HERE}/.env" ]; then
    set -a; . "${HERE}/.env"; set +a
fi

DECK_USER="${DECK_USER:-deck}"
DECK_SSH_PORT="${DECK_SSH_PORT:-2222}"
DECK_HOST="${DECK_HOST:-}"
CACHE="${XDG_CONFIG_HOME:-$HOME/.config}/wojtek/deck-host"
SSH_OPTS=(-p "$DECK_SSH_PORT" -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new)
# scp spells the port with a capital P, and its -p means something else
# entirely (keep timestamps), so the two lists cannot be shared.
SCP_OPTS=(-P "$DECK_SSH_PORT" -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new)
DECK_DIR="${HERE}/deploy/deck"

# Two seconds, not one: a phone hotspot has shown half a second of latency.
port_open() { nc -z -G 2 "$1" "$DECK_SSH_PORT" 2>/dev/null; }

# Every IPv4 address this PC has, one per line. This machine is often on two
# networks at once (home wifi and the robot's access point, or a cable to
# the Deck's router), and the Deck may be on either of them.
my_ips() {
    if command -v ipconfig >/dev/null 2>&1; then          # macOS
        local i a
        for i in $(ifconfig -lu); do
            a="$(ipconfig getifaddr "$i" 2>/dev/null)" && [ -n "$a" ] && echo "$a"
        done
    else                                                   # Linux
        ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1
    fi
}

# This PC's address as the Deck sees it: the one on the Deck's own /24. The
# panel on the Deck has to reach this address to find the simulation, and an
# address on the other network would be a dead link. Read now, not stored,
# because it moves with the network.
my_ip_toward() {
    local deck="$1" a
    while read -r a; do
        [ "${a%.*}" = "${deck%.*}" ] && { echo "$a"; return 0; }
    done < <(my_ips)
    my_ips | head -1
}

# Every /24 this PC is on, all at once: one address at a time takes minutes,
# and the Deck is the only thing here listening on that port.
search() {
    local nets found
    nets="$(my_ips | sed 's/\.[0-9]*$//' | sort -u)"
    [ -z "$nets" ] && { echo "this machine has no network address" >&2; return 1; }
    found="$(
        for net in $nets; do
            for i in $(seq 1 254); do
                (port_open "$net.$i" && echo "$net.$i") &
            done
        done
        wait
    )"
    found="$(echo "$found" | head -1)"
    [ -z "$found" ] && {
        echo "nothing on $(echo $nets | sed 's/ /.0\/24, /g').0/24 answers on port $DECK_SSH_PORT." >&2
        echo "The Deck's sshd is started by hand and dies on reboot: tap" >&2
        echo "'Deck SSH on' on its desktop, then try again." >&2
        return 1
    }
    mkdir -p "$(dirname "$CACHE")"; printf '%s\n' "$found" > "$CACHE"
    printf '%s\n' "$found"
}

# .env wins; then the last address that worked, but only while it still
# answers; then a search.
host() {
    # A pinned address is a shortcut, not a promise: this machine moves, so
    # when it stops answering there the search still runs rather than leaving
    # ssh to time out on a stale entry.
    if [ -n "$DECK_HOST" ]; then
        local pinned="${DECK_HOST#*@}"
        if port_open "$pinned"; then printf '%s\n' "$pinned"; return 0; fi
        echo "DECK_HOST ($pinned) does not answer on port $DECK_SSH_PORT; looking" >&2
    fi
    if [ -f "$CACHE" ]; then
        local cached; cached="$(cat "$CACHE")"
        if port_open "$cached"; then printf '%s\n' "$cached"; return 0; fi
    fi
    search
}

on_deck() {
    local h; h="$(host)" || return 1
    ssh "${SSH_OPTS[@]}" "${DECK_USER}@${h}" "$@"
}

cmd="${1:-login}"; [ $# -gt 0 ] && shift

case "$cmd" in
login)
    h="$(host)" || exit 1
    echo ">> ${DECK_USER}@${h}:${DECK_SSH_PORT}"
    exec ssh "${SSH_OPTS[@]}" "${DECK_USER}@${h}"
    ;;

where)
    search
    ;;

install)
    # Everything the Deck needs, from this tree, so nothing on it is
    # hand-made and a wiped Deck is one command away from working again.
    h="$(host)" || exit 1
    echo ">> installing to ${DECK_USER}@${h}"
    scp "${SCP_OPTS[@]}" -q \
        "${DECK_DIR}/deck_link.sh" "${DECK_DIR}/panel_ctl.sh" "${DECK_DIR}/deck_panel.sh" \
        "${DECK_USER}@${h}:" || exit 1
    # The desktop icon wears the panel's own favicon, so it is recognisable
    # among the Deck's other icons.
    scp "${SCP_OPTS[@]}" -q "${HERE}/src/wojtek_deck/web/favicon.svg" \
        "${DECK_USER}@${h}:/tmp/wojtek-panel.svg" || exit 1
    scp "${SCP_OPTS[@]}" -q "${DECK_DIR}"/*.desktop "${DECK_USER}@${h}:/tmp/" || exit 1
    on_deck '
        set -e
        chmod +x ~/deck_link.sh ~/panel_ctl.sh ~/deck_panel.sh
        mkdir -p ~/.local/share/icons ~/Desktop ~/.config/wojtek
        mv /tmp/wojtek-panel.svg ~/.local/share/icons/wojtek-panel.svg
        for f in /tmp/*.desktop; do
            mv "$f" ~/Desktop/ && chmod +x ~/Desktop/"$(basename "$f")"
            gio set ~/Desktop/"$(basename "$f")" metadata::trusted true 2>/dev/null || true
        done
        echo "installed: $(ls ~/Desktop/*.desktop | wc -l) icons"
    ' || exit 1
    # Where the panel should look for the simulation. The icon cannot know
    # this, and it must not be baked into a committed file, so it is written
    # here and read there. It is a hint, not an order: the icon checks that
    # it answers, and looks for the robot first.
    "$0" url "http://$(my_ip_toward "$h"):8090/"
    ;;

url)
    h="$(host)" || exit 1
    url="${1:-http://$(my_ip_toward "$h"):8090/}"
    on_deck "mkdir -p ~/.config/wojtek && printf '%s\n' '$url' > ~/.config/wojtek/panel-url" || exit 1
    echo ">> the Deck's panel icon now opens $url"
    ;;

panel)
    # With a url, the Deck opens that. Without one it opens this PC's
    # simulation, at the address the Deck can reach.
    h="$(host)" || exit 1
    url="${1:-http://$(my_ip_toward "$h"):8090/}"
    echo ">> panel -> $url"
    on_deck "~/panel_ctl.sh start '$url'"
    ;;

stop)   on_deck '~/panel_ctl.sh stop' ;;
reload) on_deck '~/panel_ctl.sh reload' ;;

steam)
    case "${1:-}" in
        off) on_deck '~/deck_link.sh steam-off' ;;
        on)  on_deck '~/deck_link.sh steam-on' ;;
        *)   echo "usage: ./deck.sh steam off|on" >&2; exit 2 ;;
    esac
    ;;

shot)
    out="${1:-/tmp/deck-$(date +%H%M%S).png}"
    h="$(host)" || exit 1
    on_deck '~/deck_link.sh shot' >/dev/null || exit 1
    scp "${SCP_OPTS[@]}" -q "${DECK_USER}@${h}:/tmp/deck_shot.png" "$out" || exit 1
    echo "$out"
    command -v open >/dev/null 2>&1 && open "$out"
    ;;

run) on_deck "$@" ;;

*)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
