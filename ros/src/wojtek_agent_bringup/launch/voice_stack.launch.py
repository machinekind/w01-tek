"""The talk-only stack: browser mic → VAD → ASR → router → Bielik → TTS.

W2 milestone bring-up — no navigation yet.  Everything runs on one box
(localhost DDS); the browser connects to the audio bridge websocket.

    ros2 launch wojtek_agent_bringup voice_stack.launch.py \
        bielik_url:=http://127.0.0.1:8091 tts_engine:=chatterbox \
        tts_ref_wav:=/path/to/voice_ref.wav

Voice references come from outside the repository (cloned voices are not
committed).  `tts_engine:=silent` and `asr_model:=tiny` make a smoke run
possible on a CPU box.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("ws_port", "8765", "audio bridge websocket port"),
    ("vad_backend", "silero", "energy | silero | pyannote"),
    ("asr_model", "large-v3", "faster-whisper checkpoint (benchmark v2 vs v3)"),
    ("asr_language", "pl", "ASR language"),
    ("router_model_path", "", "fine-tuned encoder dir; empty = rule router"),
    ("bielik_url", "http://127.0.0.1:8091", "OpenAI-compatible Bielik server"),
    ("bielik_model", "speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic",
     "served model name (FP8 wants Ada+; use the bf16 repo on Ampere)"),
    ("tts_engine", "chatterbox", "chatterbox | f5 | silent"),
    ("tts_ref_wav", "", "voice-clone reference wav (denoised)"),
    ("tts_ref_text", "", "exact transcript of the reference (F5 only)"),
    ("device", "cuda", "device for ASR/TTS models"),
]


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
    cfg = {n: LaunchConfiguration(n) for n, _d, _h in ARGS}

    return LaunchDescription(args + [
        Node(package="wojtek_voice", executable="audio_bridge",
             parameters=[{"port": cfg["ws_port"]}]),
        Node(package="wojtek_voice", executable="vad",
             parameters=[{"backend": cfg["vad_backend"]}]),
        Node(package="wojtek_voice", executable="asr",
             parameters=[{
                 "model": cfg["asr_model"],
                 "language": cfg["asr_language"],
                 "device": cfg["device"],
             }]),
        Node(package="wojtek_brain", executable="router",
             parameters=[{"model_path": cfg["router_model_path"]}]),
        Node(package="wojtek_brain", executable="bielik",
             parameters=[{
                 "url": cfg["bielik_url"],
                 "model": cfg["bielik_model"],
             }]),
        Node(package="wojtek_voice", executable="tts",
             parameters=[{
                 "engine": cfg["tts_engine"],
                 "ref_wav": cfg["tts_ref_wav"],
                 "ref_text": cfg["tts_ref_text"],
                 "device": cfg["device"],
             }]),
    ])
