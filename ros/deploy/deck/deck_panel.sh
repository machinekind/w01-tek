#!/bin/bash
# Open the wojtek_deck panel on this Steam Deck.
#
#   deck_panel.sh [url]
#   deck_panel.sh --probe <host> <port>   does that host serve the panel?
#
# The machine serving the panel changes with the network the Deck is on:
# the robot on its own access point, the PC's simulation at home. So the
# icon does not trust a stored address; it looks, in this order, for the
# first one that serves the panel:
#
#   1. the url given on the command line (../deck.sh panel <url>);
#   2. the robot, at its fixed address on its own access point;
#   3. the address ../deck.sh last wrote to ~/.config/wojtek/panel-url;
#   4. any host on this Deck's networks that serves the panel's page on
#      its port.
#
# A stale entry costs a few seconds, not a blank page. If nothing answers,
# the stored address opens anyway, so RELOAD picks the page up once its
# gateway starts, and a notification says why the window is empty. No
# address of a private machine is in this file: the repository is public.
#
# The window has a frame, so it can be minimised and closed with a finger --
# the Deck has no keyboard, and a kiosk window cannot be left without one.
# The panel's own "full" button fills the screen and gives it back, which is
# what F11 would do.
#
# Chrome is the flatpak, running on its own profile so the Deck's ordinary
# browsing is untouched. --test-type only silences the yellow bar about the
# insecure-origin flag below it; that flag is what lets a page served over
# plain http use the APIs the panel needs. WebGPU wants all three of its
# flags: with only --enable-unsafe-webgpu it lands in SwiftShader and burns
# six cores, and with none it falls back to the CPU at about 117 ms a frame
# instead of 36 on the GPU.
export DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XAUTHORITY="$(ps -o args= -C Xorg | sed -n 's/.*-auth \([^ ]*\).*/\1/p' | head -1)"

PORT=8090
# The robot is a fixed anchor at this address on both of its links, the
# cable and the wojtek-link access point (see ../rpi/README.md).
ROBOT=10.42.0.2
STORED="$HOME/.config/wojtek/panel-url"

# A tap on an icon shows nothing by itself, so the answer goes on the screen.
say() {
  echo "$1"
  notify-send -a "Wojtek deck" "Wojtek deck" "$1" 2>/dev/null ||
    kdialog --passivepopup "$1" 6 2>/dev/null || true
}

# The panel's page carries this title (wojtek_deck/web/index.html). It is
# how a host is told apart from anything else that happens to listen on the
# port: a home router, a phone, some other service on the PC.
MARKER='<title>Wojtek deck</title>'

# Does host:port serve the panel? Plain bash, no nc or curl needed: open the
# socket, ask for the page, look for the title in what comes back. Three
# seconds covers a slow hotspot (half a second of latency has been seen)
# and one round trip for the page. The host, port and title go in as
# arguments ($0, $1, $2 inside), never pasted into the command text. grep
# replaces the inner bash (exec), so when the time runs out the one process
# timeout stops is the one still reading, and nothing is left hanging.
serves_panel() {
  timeout 3 bash -c '
    exec 3<>"/dev/tcp/$0/$1" || exit 1
    printf "GET / HTTP/1.0\r\nHost: %s:%s\r\n\r\n" "$0" "$1" >&3
    exec grep -qF -m1 -- "$2" <&3
  ' "$1" "$2" "$MARKER" 2>/dev/null
}

# host and port out of a url, for the probe. "http://10.42.0.2:8090/?x=1"
# gives "10.42.0.2 8090"; a url without a port means 80.
host_port() {
  local hp="${1#*://}"; hp="${hp%%/*}"; hp="${hp%%\?*}"
  local host="${hp%%:*}" port="${hp##*:}"
  [ "$port" = "$hp" ] && port=80
  echo "$host $port"
}

# Every /24 this Deck has an address on, scanned at once: one address at a
# time takes minutes. Prints the first host that serves the panel.
scan() {
  local nets net i
  nets="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' |
          cut -d/ -f1 | sed 's/\.[0-9]*$//' | sort -u)"
  [ -z "$nets" ] && return 1
  local found
  found="$(
    for net in $nets; do
      for i in $(seq 1 254); do
        (serves_panel "$net.$i" "$PORT" && echo "$net.$i") &
      done
    done
    wait
  )"
  found="$(echo "$found" | head -1)"
  [ -n "$found" ] && echo "$found"
}

find_panel() {
  local url="$1" stored=""
  [ -f "$STORED" ] && stored="$(cat "$STORED")"

  if [ -n "$url" ]; then
    echo "$url"; return 0
  fi
  if serves_panel "$ROBOT" "$PORT"; then
    echo "http://$ROBOT:$PORT/"; return 0
  fi
  if [ -n "$stored" ] && serves_panel $(host_port "$stored"); then
    echo "$stored"; return 0
  fi
  local host; host="$(scan)"
  if [ -n "$host" ]; then
    echo "http://$host:$PORT/"; return 0
  fi
  # Nothing is serving the panel right now. Open where it usually is, so
  # RELOAD finds it later, and say so.
  echo "${stored:-http://$ROBOT:$PORT/}"
  return 1
}

# deck_panel.sh --probe <host> <port>: say whether that host serves the
# panel, and nothing else. For a test, or a quick check over ssh.
if [ "${1:-}" = "--probe" ]; then
  serves_panel "$2" "$3" && { echo panel; exit 0; }
  echo no; exit 1
fi

if URL="$(find_panel "${1:-}")"; then
  say "panel: $URL"
else
  say "nothing serves the panel on port $PORT; opening $URL anyway -- start the gateway, then RELOAD"
fi

ORIGIN="${URL%%/}"; ORIGIN="${ORIGIN%%\?*}"

pkill -x chrome 2>/dev/null; sleep 1
exec flatpak run com.google.Chrome --user-data-dir="$HOME/.deck-panel-profile" \
  --no-first-run --no-default-browser-check --test-type --app="$URL" \
  --enable-features=Vulkan,WebGPU --ignore-gpu-blocklist --enable-unsafe-webgpu \
  --window-position=0,0 --window-size=1280,800 \
  "--unsafely-treat-insecure-origin-as-secure=$ORIGIN"
