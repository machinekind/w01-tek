#!/usr/bin/env bash
# Question wavs for the scenario scripts: 24 kHz mono PCM16, the browser
# worklet format scenario.py feeds down the websocket.
#
# macOS `say` with the Polish voice by default; drop REAL mic recordings
# (same format, same names) into wavs/ instead whenever available -- the
# demo is judged on recognising a person, not a synthesiser.
set -euo pipefail
cd "$(dirname "$0")"
VOICE="${VOICE:-Zosia}"
mkdir -p wavs
while IFS=$'\t' read -r id text; do
  [[ -z "$id" || "$id" == \#* ]] && continue
  if [[ ! -f "wavs/$id.wav" ]]; then
    # </dev/null: ffmpeg reads stdin and would otherwise swallow the rest
    # of the manifest this loop is reading (cost the first batch of wavs).
    say -v "$VOICE" -o "wavs/$id.aiff" "$text" < /dev/null
    ffmpeg -y -loglevel error -i "wavs/$id.aiff" -ac 1 -ar 24000 \
      -sample_fmt s16 "wavs/$id.wav" < /dev/null
    rm -f "wavs/$id.aiff"
    echo "made wavs/$id.wav  ($text)"
  fi
done < manifest.tsv
