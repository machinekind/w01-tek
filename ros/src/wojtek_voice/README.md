# wojtek_voice

Voice pipeline nodes for the agentic stack ([design](../../../docs/plans/agentic-ros2.md)):

| node | in | out |
|---|---|---|
| `audio_bridge` | browser websocket (PCM16 24 kHz binary frames, JSON control) | `/wojtek/audio/mic`; plays `/wojtek/tts/audio`; flushes playback on `/wojtek/audio/speech_started` |
| `vad` | `/wojtek/audio/mic` | `/wojtek/audio/speech` (live chunks + full-utterance recap), `/wojtek/audio/speech_started` |
| `asr` | `/wojtek/audio/speech` (end-of-utterance chunk) | `/wojtek/asr/final` |

The websocket wire protocol is the #131 room_app mic path, so the existing
browser worklet connects unchanged.  `wojtek_voice/transport.py` is lifted
from `training/wojtek_rl/agent/voice.py` — keep segmentation fixes in sync.

VAD backends (`backend` parameter): `energy` (default, no deps), `silero`
(torch), `pyannote` (team choice for noisy rooms; needs `pyannote.audio` and
a HF token on first download).  ASR is faster-whisper `large-v3` with the
hallucination guards from #131.

Model deps are not rosdep-resolvable — `pip install -r requirements.txt`
into the deployment venv (the provisioning payload does this).

Tests are model-free and rclpy-free:

```bash
python -m pytest ros/src/wojtek_voice/test -q
```
