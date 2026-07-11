#!/usr/bin/env bash
# Flash a fresh RPi card end-to-end, from the PC: download + verify the pinned
# Ubuntu image, write it to the card, and drop our cloud-init on the boot
# partition with your SSH key injected. After this the Pi boots reachable at
# 10.42.0.2 -- then `./deploy.sh --provision` finishes the job.
#
#   ./deploy/rpi/flash-card.sh /dev/sdX
#   ./deploy/rpi/flash-card.sh /dev/sdX --key ~/.ssh/id_ed25519.pub
#   ./deploy/rpi/flash-card.sh /dev/sdX --image ./ubuntu-...img.xz   # use a local image
#   ./deploy/rpi/flash-card.sh /dev/sdX --dry-run                    # show, write nothing
#   ./deploy/rpi/flash-card.sh /dev/sdX --yes                        # skip the confirm prompt
#
# WRITES TO A RAW BLOCK DEVICE -- picking the wrong one wipes that disk. The
# script refuses your system disk and makes you confirm, but check `lsblk` first.
set -euo pipefail

# --- pinned image (keep in sync with IMAGE.md) -------------------------------
IMG_VER="24.04.4"
BASE="https://cdimage.ubuntu.com/releases/24.04/release"
IMG_NAME="ubuntu-${IMG_VER}-preinstalled-server-arm64+raspi.img.xz"
IMG_URL="${BASE}/${IMG_NAME}"
SHA_URL="${BASE}/SHA256SUMS"

HERE="$(cd "$(dirname "$0")" && pwd)"
CACHE="${HERE}/cache"
CLOUD_INIT="${HERE}/cloud-init"

DEVICE=""; IMAGE=""; KEY=""; DRY_RUN=0; ASSUME_YES=0
while [ $# -gt 0 ]; do case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --key)   KEY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) DEVICE="$1"; shift ;;
esac; done

[ -z "${DEVICE}" ] && { echo "usage: $0 /dev/sdX [--key <pub>] [--image <img.xz>] [--dry-run] [--yes]" >&2; exit 2; }

say() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }
run() { if [ "${DRY_RUN}" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else eval "$@"; fi; }

# --- resolve SSH public key --------------------------------------------------
if [ -z "${KEY}" ]; then
    for k in "${HOME}/.ssh/id_ed25519.pub" "${HOME}/.ssh/id_rsa.pub"; do
        [ -f "$k" ] && { KEY="$k"; break; }
    done
fi
[ -z "${KEY}" ] || [ ! -f "${KEY}" ] && {
    echo "no SSH public key found. Generate one (ssh-keygen -t ed25519) or pass --key." >&2; exit 1; }
PUBKEY="$(cat "${KEY}")"
say "SSH key: ${KEY}"

# --- device safety -----------------------------------------------------------
[ -b "${DEVICE}" ] || { echo "'${DEVICE}' is not a block device." >&2; exit 1; }
# Refuse the system disk: reject if any partition is mounted at a system path.
mounts="$(lsblk -nro MOUNTPOINT "${DEVICE}" 2>/dev/null | grep -E '^(/|/boot.*|/home|/var.*)$' || true)"
if [ -n "${mounts}" ]; then
    echo "REFUSING: ${DEVICE} has system mountpoints (${mounts//$'\n'/, }). Wrong device?" >&2; exit 1
fi
say "Target device:"
lsblk -o NAME,SIZE,MODEL,TRAN,RM,MOUNTPOINTS "${DEVICE}"
if [ "$(lsblk -ndo RM "${DEVICE}" 2>/dev/null)" != "1" ]; then
    echo "   NOTE: ${DEVICE} is not flagged removable -- double-check it's the card, not an internal disk."
fi
if [ "${DRY_RUN}" != 1 ] && [ "${ASSUME_YES}" != 1 ]; then
    read -r -p "Type the device path to confirm ERASE (${DEVICE}): " ans
    [ "${ans}" = "${DEVICE}" ] || { echo "mismatch -- aborting."; exit 1; }
fi

# --- obtain + verify image ---------------------------------------------------
if [ -z "${IMAGE}" ]; then
    mkdir -p "${CACHE}"
    IMAGE="${CACHE}/${IMG_NAME}"
    if [ ! -f "${IMAGE}" ]; then
        say "downloading ${IMG_NAME}"
        run "curl -fL '${IMG_URL}' -o '${IMAGE}'"
    else
        say "using cached image ${IMAGE}"
    fi
    say "verifying sha256 against ${SHA_URL}"
    if [ "${DRY_RUN}" != 1 ]; then
        sums="$(curl -fsSL "${SHA_URL}")"
        want="$(printf '%s\n' "${sums}" | awk -v f="*${IMG_NAME}" '$2==f{print $1}')"
        [ -z "${want}" ] && want="$(printf '%s\n' "${sums}" | grep -F "${IMG_NAME}" | awk '{print $1}')"
        [ -z "${want}" ] && { echo "could not find ${IMG_NAME} in SHA256SUMS." >&2; exit 1; }
        got="$(sha256sum "${IMAGE}" | awk '{print $1}')"
        [ "${want}" = "${got}" ] || { echo "SHA256 MISMATCH -- corrupt/tampered download. want ${want} got ${got}" >&2; exit 1; }
        echo "   sha256 OK"
    fi
else
    say "using local image ${IMAGE} (no checksum verification)"
    [ -f "${IMAGE}" ] || { echo "no such image: ${IMAGE}" >&2; exit 1; }
fi

# --- flash -------------------------------------------------------------------
say "flashing ${IMAGE} -> ${DEVICE}"
case "${IMAGE}" in
    *.xz) run "xzcat '${IMAGE}' | sudo dd of='${DEVICE}' bs=4M conv=fsync status=progress" ;;
    *)    run "sudo dd if='${IMAGE}' of='${DEVICE}' bs=4M conv=fsync status=progress" ;;
esac
run "sync"
run "sudo partprobe '${DEVICE}' 2>/dev/null || true"
run "sleep 3"

# --- drop cloud-init onto the system-boot (FAT) partition --------------------
# Partition 1 is the FAT boot partition. nvme/mmc use a 'p' separator.
case "${DEVICE}" in
    *[0-9]) BOOTPART="${DEVICE}p1" ;;
    *)      BOOTPART="${DEVICE}1" ;;
esac
say "writing cloud-init to ${BOOTPART}"
if [ "${DRY_RUN}" = 1 ]; then
    printf '   [dry-run] mount %s, copy meta-data/network-config + user-data (key injected), umount\n' "${BOOTPART}"
else
    MNT="$(mktemp -d)"
    sudo mount "${BOOTPART}" "${MNT}"
    trap 'sudo umount "${MNT}" 2>/dev/null || true; rmdir "${MNT}" 2>/dev/null || true' EXIT
    sudo cp "${CLOUD_INIT}/meta-data"      "${MNT}/meta-data"
    sudo cp "${CLOUD_INIT}/network-config" "${MNT}/network-config"
    # user-data with the placeholder key line replaced by the real key.
    awk -v key="      - ${PUBKEY}" '
        /REPLACE_ME/ { print key; next } { print }
    ' "${CLOUD_INIT}/user-data" | sudo tee "${MNT}/user-data" >/dev/null
    sync
    sudo umount "${MNT}"; rmdir "${MNT}"; trap - EXIT
fi

say "Done."
cat <<EOF

Card ready. Next:
  1. Put the card in the Pi, dock it by cable (PC: nmcli con up wojtek-eth), power on.
  2. It should come up at 10.42.0.2 -- check:  ssh rpi@10.42.0.2
  3. From the PC:  ./deploy.sh --provision
EOF
