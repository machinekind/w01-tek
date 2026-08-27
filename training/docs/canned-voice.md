# Canned voice lines: how to make more (or re-record in a new voice)

The dog's instant responses (acks, progress, outcomes, the intro) are
"prerecorded" without any audio files in the repo: each line is plain text
in the phrase bank, synthesized ONCE by the running TTS server at stack
start, and served from the server's in-memory cache afterwards. A canned
line costs a cache hit (~0.1 s) instead of a synthesis (2-8 s), and
changing the voice re-records everything automatically.

## Where the lines live

`ros/src/wojtek_brain/wojtek_brain/prompts/bielik/phrases/<kind>.txt`
— one variant per line, UTF-8. Kinds are declared in
`wojtek_brain/phrases.py` (`KINDS`); add a file + add the kind there.
Sampling is a per-kind shuffle bag: every variant plays once before any
repeats, so more variants directly means less deja vu. 5-8 per kind is a
good floor.

Writing rules (each learned on camera):

- **Full short sentences.** Single words decode/synthesize badly end to end.
- **No onomatopoeia** (hau, woof): Chatterbox renders them as noises that
  sound like a fault.
- **No `{noun}` interpolation**: Polish case endings made
  "Znalazłem: telewizora". Templates must be grammatical with nothing
  glued in.
- Keep each line under ~180 chars; longer lines synthesize fine but the
  whole line plays as one piece.

## Where the actual audio lives

With `--cache-dir` (the rig passes `/root/tts_lines`) every synthesized
line is ALSO a plain 24 kHz mono WAV on disk, named
`<voice-hash>_<text-hash>.wav`. Those files ARE the prerecorded tracks:

- a restart or robot boot serves them with zero GPU work (the prewarm loop
  finds every line already on disk and finishes in a second);
- changing the voice reference or a line's text changes the hash, so stale
  audio can never play -- only the changed lines re-record;
- they are generated artifacts, deliberately NOT in the repo (a cloned
  voice must not be committed publicly). Back the directory up alongside
  the voice reference if you want to keep a take-perfect set.

## How the "recording" happens

1. `tts_server.py` runs with `--cache 256` (an LRU keyed by EXACT text).
2. At stack start the rig pushes every bank line through `POST /synthesize`
   (`spark_ros_rig.sh` `stack`, the prewarm block — it shells out to
   `python3 -m wojtek_brain.phrases` for the list).
3. At runtime the ROS tts node calls `/synthesize_stream`; the server
   checks the whole-line cache BEFORE splitting (do not remove that check:
   the split pieces have different cache keys, and without it every canned
   line misses).

Quality knobs already tuned on the server: `--temperature 0.6` (fewer
hallucinated tails), tail-noise trimming, voice conditionals encoded once.
Playback side: the tts node paces frames at wall-clock speed (a cache hit
would otherwise burst the whole utterance at once and garble the mix).

## Changing the voice

The server owns the voice: start `tts_server.py --ref /path/to/voice.wav`
(a denoised 3-8 s reference clip; see `f5_prep.py` for cutting one), then
re-run the stack so the prewarm re-records every line in the new voice.
Nothing else changes — same texts, same cache mechanics.

## Exporting real WAV files (if ever needed outside the server)

```bash
curl -s -X POST localhost:8120/synthesize \
  -H 'content-type: application/json' \
  -d '{"text": "Jasne, patrz na to!"}' > line.wav
```

One request per line; the response is a finished 24 kHz WAV with the same
trimming the live stack uses.

## The scripted QUESTION voice (scenario takes)

Different pipeline on purpose: `training/scenarios/make_wavs.sh` uses piper
`pl_PL-gosia-medium` (neural, natural diacritics — macOS `say` mangled
"ą/ę"). Regenerate by deleting a wav and re-running the script; swap voices
by changing the model path in that script.
