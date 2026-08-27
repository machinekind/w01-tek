#!/usr/bin/env bash
# Question wavs for the scenario scripts: 24 kHz mono PCM16, the browser
# worklet format scenario.py feeds down the websocket.
#
# Voice: piper pl_PL-gosia-medium (a natural female Polish neural voice) --
# macOS `say -v Zosia` mangled diacritics badly enough to distract from the
# takes (user review, 2026-08-27). Drop REAL mic recordings (same format,
# same names) into wavs/ whenever available -- the demo is judged on
# recognising a person, not a synthesiser.
#
# Needs: pip install piper-tts huggingface_hub; ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
# The piper-tts wheel bakes a CI-machine espeak data path; point it at a
# real installation (brew on macOS, apt's /usr/lib/... on Linux).
for d in /opt/homebrew/share/espeak-ng-data /usr/lib/x86_64-linux-gnu/espeak-ng-data /usr/share/espeak-ng-data; do
  [[ -f "$d/phontab" ]] && export ESPEAK_DATA_PATH="$(dirname "$d")" && break
done
mkdir -p wavs
MODEL=$("$PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("rhasspy/piper-voices", "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json")
print(hf_hub_download("rhasspy/piper-voices", "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx"))
PY
)
while IFS=$'\t' read -r id text; do
  [[ -z "$id" || "$id" == \#* ]] && continue
  if [[ ! -f "wavs/$id.wav" ]]; then
    # </dev/null: downstream tools must not slurp the manifest this loop reads.
    # piper reads newline-terminated lines from stdin; without the \n it
    # synthesizes a 0.2 s stub of silence (bitten live).
    printf '%s\n' "$text" | "$PYTHON" -m piper -m "$MODEL" -f "wavs/$id.raw.wav"
    ffmpeg -y -loglevel error -i "wavs/$id.raw.wav" -ac 1 -ar 24000 \
      -sample_fmt s16 "wavs/$id.wav" < /dev/null
    rm -f "wavs/$id.raw.wav"
    echo "made wavs/$id.wav  ($text)"
  fi
done < manifest.tsv
