# wojtek_voice

Voice pipeline nodes for the agentic stack ([design](../../../docs/architecture.md)):

| node | in | out |
|---|---|---|
| `audio_bridge` | browser websocket (PCM16 24 kHz binary frames, JSON control) | `/wojtek/audio/mic`; plays `/wojtek/tts/audio`; flushes playback on `/wojtek/audio/speech_started` |
| `vad` | `/wojtek/audio/mic` | `/wojtek/audio/speech` (live chunks + full-utterance recap), `/wojtek/audio/speech_started` |
| `asr` | `/wojtek/audio/speech` (end-of-utterance chunk) | `/wojtek/asr/final` |

The websocket wire protocol is the #131 room_app mic path, so the existing
browser worklet connects unchanged.  `wojtek_voice/transport.py` is lifted
from `training/wojtek_agent/voice.py` — keep segmentation fixes in sync.

VAD backends (`backend` parameter): `silero` (default — MIT, ~2 MB, <1 ms
per frame), `pyannote` (optional, for noisy rooms / model-based turn-taking;
needs `pyannote.audio` and a HF token on first download), `energy`
(zero-dependency fallback).  ASR is faster-whisper with the hallucination
guards from #131; `model` parameter picks the checkpoint — benchmark
`large-v2` vs `large-v3` on Polish before trusting the default.

Model deps are not rosdep-resolvable — `pip install -r requirements.txt`
into the deployment venv (the provisioning payload does this).

Tests are model-free and rclpy-free:

```bash
python -m pytest ros/src/wojtek_voice/test -q   # or ./run.sh test from the experiment root
```
