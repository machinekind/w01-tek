# wojtek_agent_perf — where the turn's seconds went

A passive latency probe for the agentic stack.  It subscribes to the five
topics one spoken turn passes through and subtracts their header stamps; no
node is modified, no callback does extra work, and the probe can be started
or killed mid-session.

```
/wojtek/audio/speech (end_of_utterance)  human stopped talking
/wojtek/asr/final                        text exists          -> asr.transcribe
/wojtek/intent                           routed               -> brain.route
/wojtek/say (first sentence)             brain has an answer  -> llm.first_sentence
/wojtek/tts/audio (first frame)          human hears it       -> tts.first_audio
                                         whole wait           -> voice.reply
```

Run it alongside the stack and report afterwards:

```bash
ros2 run wojtek_agent_perf probe --ros-args -p out:=/tmp/wojtek_perf.jsonl
```

```bash
./training/run.sh perf /tmp/wojtek_perf.jsonl
```

`voice_stack.launch.py` starts it with `perf:=true perf_out:=<file>`.

Notes:

- Output is the same `perf.span` JSONL the demo stack writes
  (`wojtek_rl.perf`), with the same stage names, so one report tool ranks
  both stacks and the numbers are comparable.
- `vad_silence_s` must match the VAD node's `silence_end_s`.  The endpoint
  wait is not observable from outside the VAD, so it is reported from the
  configured value and tagged `assumed` — it is real, felt time on every
  turn and leaving it out would flatter the total.
- A stage appears only when both of its endpoints were seen; a missing
  stage means the turn broke there, not that it was instant.
