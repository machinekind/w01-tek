#!/usr/bin/env bash
# Fetch what the deck panel's detector needs into the asset store.
#
#   ros/src/wojtek_deck/fetch_assets.sh          # into ros/deck_assets/
#   WOJTEK_DECK_ASSETS=/some/dir ...             # or somewhere else
#
# The detector runs in the browser on the handheld, so the two things it
# needs are files the page downloads: the YOLOX-nano network and the
# onnxruntime-web runtime that executes it. Neither is committed. They are
# big binaries from other projects, and this repository is public and small
# on purpose -- the same reason policies live in a gitignored store.
#
# The store sits next to the workspace's src/, like the policy store:
# ros/deck_assets in a checkout, ~/wojtek_ws/deck_assets on the robot.
# deploy.sh runs this script and rsyncs the result, because the robot has no
# internet and the page is served from the robot.
#
# Safe to run again: a file that is already there with the right hash is
# left alone, and the script says one line and stops.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# ros/src/wojtek_deck -> ros
STORE="${WOJTEK_DECK_ASSETS:-$(cd "${HERE}/../.." && pwd)/deck_assets}"

# YOLOX-nano, the ONNX the YOLOX authors publish with the 0.1.1rc0 release.
# 416x416 input, 80 COCO classes. Apache-2.0, Megvii.
MODEL_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
MODEL_SHA="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"

# onnxruntime-web, straight from the npm tarball. MIT, Microsoft.
ORT_VERSION="1.29.0"
ORT_URL="https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${ORT_VERSION}.tgz"
ORT_SHA="7a934b7811c3b050ecfb7619722e2b4de771ce6da20520e17a2018a440316ef3"

# Only the three files the page actually loads. The tarball carries every
# build variant (all execution providers, node, webgl, several wasm
# flavours) and we want none of that on the robot's disk or its wifi.
#
# ort.webgpu.min.mjs is the entry: it carries the WebGPU provider and the
# plain-WASM one, which is exactly the pair the worker asks for. It is an ES
# module, which is why det_worker.js is a module worker. The runtime then
# loads its WASM half by name, and in 1.29 the WebGPU build wants the
# "asyncify" flavour -- reading results back off the GPU is asynchronous, so
# the WASM side has to be able to suspend. Those two names are read out of
# the bundle, not guessed; check them again when bumping the version.
ORT_FILES=(
    "ort.webgpu.min.mjs"
    "ort-wasm-simd-threaded.asyncify.mjs"
    "ort-wasm-simd-threaded.asyncify.wasm"
)
# The stamp records which tarball the files in the store came from, since
# the hashes we pin are of the tarball rather than of each extracted file.
ORT_STAMP="${STORE}/.onnxruntime-web.stamp"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1   # macOS has no sha256sum
    fi
}

verify() {  # verify <file> <expected sha>
    [ -f "$1" ] && [ "$(sha256 "$1")" = "$2" ]
}

# The wasm is 26 MB and the page pulls it over the robot's wifi. aiohttp's
# static handler serves a <file>.gz sibling to any browser that asked for
# gzip, which turns that into about 6 MB, so keep one next to every asset.
gzip_beside() {
    [ -f "$1.gz" ] && [ "$1.gz" -nt "$1" ] && return 0
    gzip -9 -c "$1" > "$1.gz"
}

mkdir -p "${STORE}"

have_ort=0
if [ -f "${ORT_STAMP}" ] && [ "$(cat "${ORT_STAMP}")" = "${ORT_SHA}" ]; then
    have_ort=1
    for f in "${ORT_FILES[@]}"; do
        [ -f "${STORE}/${f}" ] || have_ort=0
    done
fi
if verify "${STORE}/yolox_nano.onnx" "${MODEL_SHA}" && [ "${have_ort}" = 1 ]; then
    echo ">> deck assets up to date (${STORE})"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

if ! verify "${STORE}/yolox_nano.onnx" "${MODEL_SHA}"; then
    echo ">> fetch yolox_nano.onnx"
    curl -fsSL -o "${TMP}/yolox_nano.onnx" "${MODEL_URL}"
    got="$(sha256 "${TMP}/yolox_nano.onnx")"
    if [ "${got}" != "${MODEL_SHA}" ]; then
        echo "!! yolox_nano.onnx hash mismatch" >&2
        echo "   want ${MODEL_SHA}" >&2
        echo "   got  ${got}" >&2
        exit 1
    fi
    mv "${TMP}/yolox_nano.onnx" "${STORE}/yolox_nano.onnx"
fi

if [ "${have_ort}" != 1 ]; then
    echo ">> fetch onnxruntime-web ${ORT_VERSION}"
    curl -fsSL -o "${TMP}/ort.tgz" "${ORT_URL}"
    got="$(sha256 "${TMP}/ort.tgz")"
    if [ "${got}" != "${ORT_SHA}" ]; then
        echo "!! onnxruntime-web-${ORT_VERSION}.tgz hash mismatch" >&2
        echo "   want ${ORT_SHA}" >&2
        echo "   got  ${got}" >&2
        exit 1
    fi
    # --strip-components 2 drops the npm "package/dist/" wrapper so the
    # files land flat in the store, which is how the page addresses them.
    for f in "${ORT_FILES[@]}"; do
        tar xzf "${TMP}/ort.tgz" -C "${STORE}" --strip-components 2 \
            "package/dist/${f}"
    done
    printf '%s\n' "${ORT_SHA}" > "${ORT_STAMP}"
fi

gzip_beside "${STORE}/yolox_nano.onnx"
for f in "${ORT_FILES[@]}"; do
    gzip_beside "${STORE}/${f}"
done

cat > "${STORE}/LICENSES.txt" <<EOF
The deck panel's detector assets. Downloaded by
ros/src/wojtek_deck/fetch_assets.sh, not committed, not ours.

yolox_nano.onnx
    YOLOX, Megvii-BaseDetection. Apache License 2.0.
    Release 0.1.1rc0: ${MODEL_URL}

ort.webgpu.min.mjs
ort-wasm-simd-threaded.asyncify.mjs
ort-wasm-simd-threaded.asyncify.wasm
    onnxruntime-web ${ORT_VERSION}, Microsoft. MIT License.
    npm tarball: ${ORT_URL}

The .gz files next to them are the same bytes, pre-compressed so the
gateway can hand them to a browser without compressing on every request.
EOF

echo ">> deck assets in ${STORE}"
