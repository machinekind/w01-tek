"""The W3 stack: the full agent brain on one box, world elsewhere.

Voice pipeline (VAD -> ASR -> router -> Bielik -> TTS) plus the VLM agent
node that walks and looks. Audio and camera arrive over DDS from whatever
world is running -- the wojtek_sim_bridge node in sim, hardware drivers on
the robot -- so the audio_bridge websocket is OFF by default here; enable
it only for a talk-only session with a local browser.

    ros2 launch wojtek_agent_bringup agent_stack.launch.py \
        agent_url:=http://127.0.0.1:8090/v1 \
        vlm_url:=http://127.0.0.1:8120 \
        bielik_url:=http://127.0.0.1:8091 tts_engine:=chatterbox \
        tts_ref_wav:=/path/to/voice_ref.wav

The demo's rule of thumb survives the split: model servers (vLLM x2,
FutureNav) are separate processes; every node here is a thin client.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("audio_bridge", "false", "serve the browser mic websocket locally "
     "(off: the world's bridge owns audio I/O)"),
    ("ws_port", "8765", "audio bridge websocket port (audio_bridge:=true)"),
    ("vad_backend", "silero", "energy | silero | pyannote"),
    ("asr_model", "large-v3", "faster-whisper checkpoint"),
    ("asr_backend", "auto", "auto | faster-whisper | transformers"),
    ("asr_language", "pl", "ASR language"),
    ("router_model_path", "", "fine-tuned encoder dir; empty = rule router"),
    ("bielik_url", "http://127.0.0.1:8091", "OpenAI-compatible Bielik server"),
    ("bielik_model", "speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic",
     "served model name"),
    ("agent_url", "http://127.0.0.1:8090/v1", "OpenAI-compatible Qwen server"),
    ("agent_model", "", "served agent model name; empty = client default"),
    ("vlm_backend", "futurenav", "futurenav | openai (nav decisions)"),
    ("vlm_url", "", "FutureNav server URL; empty = backend default"),
    ("trace_path", "", "agent trace JSONL; empty = wojtek_rl default"),
    ("forward_scale", "0.0", "multiply FutureNav forward steps (rig used 2); "
     "0 = leave WOJTEK_NAV_FORWARD_SCALE alone"),
    ("tts_engine", "chatterbox", "chatterbox | f5 | remote | silent"),
    ("tts_url", "", "remote engine: the wojtek_rl tts_server URL"),
    ("tts_ref_wav", "", "voice-clone reference wav (denoised)"),
    ("tts_ref_text", "", "exact transcript of the reference (F5 only)"),
    ("device", "cuda", "device for ASR/TTS models"),
    ("vad_silence_s", "0.7", "trailing silence that closes an utterance"),
    ("perf", "true", "run the passive latency probe alongside the stack"),
    ("perf_out", "", "probe JSONL path; read with ./training/run.sh perf"),
]


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
    cfg = {n: LaunchConfiguration(n) for n, _d, _h in ARGS}

    return LaunchDescription(args + [
        Node(package="wojtek_voice", executable="audio_bridge",
             condition=IfCondition(cfg["audio_bridge"]),
             parameters=[{"port": cfg["ws_port"]}]),
        Node(package="wojtek_voice", executable="vad",
             parameters=[{
                 "backend": cfg["vad_backend"],
                 "silence_end_s": cfg["vad_silence_s"],
             }]),
        Node(package="wojtek_voice", executable="asr",
             parameters=[{
                 "model": cfg["asr_model"],
                 "language": cfg["asr_language"],
                 "device": cfg["device"],
                 "backend": cfg["asr_backend"],
             }]),
        Node(package="wojtek_brain", executable="router",
             parameters=[{"model_path": cfg["router_model_path"]}]),
        Node(package="wojtek_brain", executable="bielik",
             parameters=[{
                 "url": cfg["bielik_url"],
                 "model": cfg["bielik_model"],
             }]),
        Node(package="wojtek_brain", executable="vlm_agent",
             parameters=[{
                 "agent_url": cfg["agent_url"],
                 "agent_model": cfg["agent_model"],
                 "vlm_backend": cfg["vlm_backend"],
                 "vlm_url": cfg["vlm_url"],
                 "trace_path": cfg["trace_path"],
                 "forward_scale": cfg["forward_scale"],
             }]),
        Node(package="wojtek_voice", executable="tts",
             parameters=[{
                 "engine": cfg["tts_engine"],
                 "ref_wav": cfg["tts_ref_wav"],
                 "ref_text": cfg["tts_ref_text"],
                 "device": cfg["device"],
                 "url": cfg["tts_url"],
             }]),
        Node(package="wojtek_agent_perf", executable="probe",
             condition=IfCondition(cfg["perf"]),
             parameters=[{
                 "out": cfg["perf_out"],
                 "vad_silence_s": cfg["vad_silence_s"],
             }]),
    ])
