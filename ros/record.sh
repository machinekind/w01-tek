#!/usr/bin/env bash
# Record Wojtek's camera to an mp4 on this PC.
#
#   ./record.sh [url|host[:port]] [--minutes N]   record until Ctrl-C, or N minutes
#   ./record.sh where                             find the panel, print its address
#
# The picture is the deck panel's own camera stream: the gateway
# (src/wojtek_deck) serves the camera's JPEGs as MJPEG on /stream.mjpg,
# port 8090. ffmpeg reads that stream and encodes the H.264 here, on the PC.
# The robot encodes nothing for this: while it records, the robot works as
# if one more panel were open. The gateway switches the camera on for a
# viewer and passes its JPEGs on, the same as for the Deck.
#
# The machine serving the panel changes with the network: the robot on its
# own access point or cable, the robot on some other wifi with whatever
# address DHCP gave it, this PC's simulation. So the address is looked for,
# the way the Deck's panel icon does it (deploy/deck/deck_panel.sh), and the
# first that serves the panel wins:
#
#   1. the url or host given on the command line;
#   2. WOJTEK_PANEL_URL from the environment or ros/.env;
#   3. the robot at its fixed address on its own links (deploy/rpi/README.md);
#   4. the last address that worked, kept in ~/.config/wojtek/panel-url;
#   5. this PC's own simulation, localhost:8090;
#   6. any host on this PC's networks that serves the panel on port 8090.
#
# A host counts only if its page carries the panel's title, so a router or
# some other service on the port is never taken for the robot. No address of
# a private machine is in this file: the repository is public.
#
# Recordings go to ~/wojtek_recordings/<UTC start>.mp4; WOJTEK_RECORDINGS
# picks another directory. ffmpeg is $FFMPEG, else the system one, else the
# binary that imageio-ffmpeg bundles (training/.venv has it: the training
# videos use it), so a bare Mac with the training venv needs nothing installed.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Local overrides (WOJTEK_PANEL_URL, WOJTEK_RECORDINGS, FFMPEG); never committed.
if [ -f "${HERE}/.env" ]; then
    set -a; . "${HERE}/.env"; set +a
fi

PORT=8090
# The robot's fixed anchor on both of its links, the cable and the
# wojtek-link access point (deploy/rpi/README.md).
ROBOT=10.42.0.2
CACHE="${XDG_CONFIG_HOME:-$HOME/.config}/wojtek/panel-url"
OUT_DIR="${WOJTEK_RECORDINGS:-$HOME/wojtek_recordings}"
# The panel page's title (src/wojtek_deck/web/index.html), the same marker
# deck_panel.sh looks for. A test keeps the two from drifting apart.
MARKER='<title>Wojtek deck</title>'

die() { echo "record.sh: $*" >&2; exit 1; }

# Does host:port serve the panel? Ask for the page and look for its title.
# The third argument is the whole time allowed, in seconds; the scan below
# uses less, since it asks hundreds of hosts at once.
serves_panel() {
    curl -s --connect-timeout "${3:-2}" -m "${3:-3}" "http://$1:$2/" 2>/dev/null \
        | grep -qF -- "$MARKER"
}

# "http://h:p/x", "h:p" or "h" -> "http://h:p" (port 8090 when not given).
base_url() {
    local hp="${1#*://}"
    hp="${hp%%/*}"; hp="${hp%%\?*}"
    [ "${hp%:*}" = "$hp" ] && hp="$hp:$PORT"
    echo "http://$hp"
}

# "http://h:p" -> "h p", for serves_panel.
host_port() {
    local hp="${1#http://}"
    echo "${hp%:*} ${hp##*:}"
}

# Every IPv4 address this PC has, one per line (as ./deck.sh reads them).
# This machine is often on two networks at once, and the robot may be on
# either.
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

# Every /24 this PC is on, all hosts at once: one at a time takes minutes.
scan() {
    local nets found net i
    nets="$(my_ips | grep -v '^127\.' | sed 's/\.[0-9]*$//' | sort -u)"
    [ -n "$nets" ] || return 1
    found="$(
        for net in $nets; do
            for i in $(seq 1 254); do
                (serves_panel "$net.$i" "$PORT" 2 && echo "$net.$i") &
            done
        done
        wait
    )"
    found="$(echo "$found" | head -1)"
    [ -n "$found" ] && echo "http://$found:$PORT"
}

find_panel() {
    local cand url
    if [ -n "${1:-}" ]; then
        # Given by hand: taken as it is. ffmpeg says so if nothing is there.
        base_url "$1"; return 0
    fi
    for cand in "${WOJTEK_PANEL_URL:-}" "$ROBOT" "$(cat "$CACHE" 2>/dev/null)" "localhost"; do
        [ -n "$cand" ] || continue
        url="$(base_url "$cand")"
        # shellcheck disable=SC2046  # host and port are two words on purpose
        if serves_panel $(host_port "$url"); then
            echo "$url"; return 0
        fi
        [ "$cand" = "${WOJTEK_PANEL_URL:-}" ] &&
            echo "WOJTEK_PANEL_URL ($url) does not serve the panel; looking" >&2
    done
    echo "looking for the panel on this machine's networks..." >&2
    scan
}

remember() {
    mkdir -p "$(dirname "$CACHE")" && printf '%s\n' "$1" > "$CACHE"
}

find_ffmpeg() {
    if [ -n "${FFMPEG:-}" ]; then echo "$FFMPEG"; return 0; fi
    command -v ffmpeg && return 0
    local py
    for py in "$HERE/../training/.venv/bin/python" python3; do
        command -v "$py" >/dev/null 2>&1 &&
            "$py" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null &&
            return 0
    done
    return 1
}

# -- arguments ---------------------------------------------------------------
target="" minutes="" where=0
while [ $# -gt 0 ]; do
    case "$1" in
        where) where=1 ;;
        --minutes)
            [ $# -ge 2 ] || die "--minutes needs a number"
            minutes="$2"; shift ;;
        -h|--help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option $1 (see ./record.sh --help)" ;;
        *) target="$1" ;;
    esac
    shift
done
if [ -n "$minutes" ]; then
    [[ "$minutes" =~ ^[0-9]+$ ]] && [ "$minutes" -gt 0 ] || die "--minutes takes a whole number above 0"
fi

url="$(find_panel "$target")" ||
    die "nothing on port $PORT serves the panel (robot, last address, localhost, this machine's networks).
Is the robot's service up with deck:=true deck_camera:=true, or ./ros/sim.sh running? Or pass the address: ./ros/record.sh <host>"
[ -z "$target" ] && remember "$url"

if (( where )); then
    echo "$url"; exit 0
fi

FF="$(find_ffmpeg)" || die "no ffmpeg: install one (brew install ffmpeg), set FFMPEG, or create training/.venv (it bundles one)"

# Hardware H.264 on a Mac; x264 anywhere else. Keyframe every 30 frames: one
# a second at the panel's 30 fps, so the file seeks and cuts well.
if "$FF" -hide_banner -encoders 2>/dev/null | grep -q ' h264_videotoolbox '; then
    encoder=(-c:v h264_videotoolbox -b:v 3M -g 30)
else
    encoder=(-c:v libx264 -preset veryfast -crf 23 -g 30)
fi

mkdir -p "$OUT_DIR" || die "cannot create $OUT_DIR"
name="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
part="$OUT_DIR/$name.part.mp4"
out="$OUT_DIR/$name.mp4"
limit=()
[ -n "$minutes" ] && limit=(-t "$((minutes * 60))")

echo ">> recording $url/stream.mjpg"
echo "   -> $out"
echo "   Ctrl-C stops${minutes:+ (or after $minutes min)}"

# The stream carries no timestamps, and the gateway sends frames as they
# come: up to its stream rate, fewer when the camera or the wifi lag. ffmpeg
# would assume 25 fps and play the video too fast or too slow. So the
# filter stamps every frame with the time it arrived (RTCTIME, in ms), and
# the output is a steady 30 fps: a frame is repeated until the next one
# comes, so the video plays at the speed it was filmed and a gap shows as a
# still picture. The small probe makes ffmpeg start on the first frame,
# instead of reading a burst of them that would all get the same stamp.
# JPEG pixels are full range; they are turned into the ordinary video range
# every player expects. ffmpeg warns "deprecated pixel format used" while
# doing it: that line is harmless.
#
# The http options ride out a wifi drop: a read that stalls for 10 s, or a
# dropped connection, makes ffmpeg reconnect, with growing pauses (0, 1, 3,
# 7, 15 s), for about half a minute before it gives up and the recording
# ends with what it has. The file is written in
# fragments, so a recording cut short (the PC sleeps, the process is
# killed) still opens.
#
# Ctrl-C goes to ffmpeg too, which closes the file properly; the trap only
# keeps this script alive for the last step.
trap ':' INT
"$FF" -hide_banner -loglevel warning -stats \
    -probesize 32 -analyzeduration 0 \
    -rw_timeout 10000000 -reconnect 1 -reconnect_streamed 1 \
    -reconnect_on_network_error 1 -reconnect_delay_max 30 \
    -f mpjpeg -i "$url/stream.mjpg" \
    -an -vf 'settb=1/1000,setpts=(RTCTIME-RTCSTART)/(TB*1000000),scale=out_range=tv,format=yuv420p' \
    -fps_mode cfr -r 30 "${encoder[@]}" ${limit[@]+"${limit[@]}"} \
    -movflags +frag_keyframe+empty_moov+default_base_moof \
    "$part"
status=$?
trap - INT

if [ ! -s "$part" ]; then
    rm -f "$part"
    die "no picture came from $url/stream.mjpg (ffmpeg exit $status). Open $url in a browser: does the panel show the camera?"
fi

# Put the fragments back together into a plain mp4 (copy only, a second or
# two): some players read a fragmented file's length wrong. If that fails
# the fragmented file stays, and it plays as it is.
if "$FF" -hide_banner -loglevel error -i "$part" -c copy -movflags +faststart "$out" </dev/null; then
    rm -f "$part"
else
    rm -f "$out"
    mv "$part" "$out"
fi
echo ">> saved $out ($(du -h "$out" | awk '{print $1}'))"
