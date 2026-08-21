# Demo scenarios: scripted, gate-checked, repeatable takes

The August 2026 takes proved single behaviours (walk somewhere, find a
thing) and their scripts died with a session scratchpad. These live in the
repository, and each one composes several behaviours in one continuous take
— goal switching, questions answered mid-walk, barge-in, spatial memory —
because that composition is what makes the robot read as an agent rather
than a voice-controlled RC car.

## Running a take

```bash
# 1. Generate the question wavs (macOS; any 24 kHz mono PCM16 wavs work --
#    real mic recordings are better than synthesis when available):
./scenarios/make_wavs.sh

# 2. Demo stack up (room_app + models), then drive+film+validate on ONE
#    websocket -- never film with a second client (single-viewer trap):
python -m wojtek_rl.agent.scenario --script scenarios/flat_guest.json \
    --video take.mp4 --require-move 0.8 --require-speech 4
```

Every delivered clip additionally gets frames extracted and looked at:
metrics alone have lied twice (frozen-clip bug, narrated-navigation bug).

## Stagecraft rules (paid for in failed takes)

- A search target VISIBLE from the start gets found in seconds — a 0 m
  take. Pick unseen targets or script a `turn_left 180` first.
- Castle searches need a ~140 s window (the scan carousel alone is ~60 s).
- FutureNav walks well in the flat, poorly in the huge castle hall.
- Castle spawn: `WOJTEK_SPAWN="2.5,-3.0"` (open floor).
- Timings assume the GB10-class stack (TTS RTF 0.44, first sound ~1 s). On
  Ada-class boxes stretch every post-question `wait` by ~1.5×.

## The takes

| script | scene | what it proves |
|---|---|---|
| `flat_guest.json` | flat | look → navigate → **status question answered while walking** → goal switch spoken aloud → found object |
| `castle_tour.json` | castle | multi-step route kept VERBATIM → redirect to search with honest "nie widzę" → **stop guard** mid-walk |
| `flat_interrupt.json` | flat | **barge-in**: long answer cut off by a new command; the stale turn is discarded, the new goal runs |
| `castle_memory.json` | castle | spatial memory: `route` ("co robiłeś?") and `map` ("co już znasz?") after real movement |
| `flat_rapidfire.json` | flat | five quick Q/A, no movement: the voice-latency showcase; gate on speech only |
