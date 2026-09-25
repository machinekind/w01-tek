"""How fast is the cloned voice on THIS machine?

One command, run where the GPU is:

    ./training/run.sh tts-bench                       # RTF over the fixed lines
    ./training/run.sh tts-bench --ref ~/refs/voice.wav
    ./training/run.sh tts-bench --clock-sweep 1000,1500,2000,2500
    ./training/run.sh tts-bench --variants base,compile --json out.json

The number that decides the architecture is the **real-time factor**: wall
seconds per second of audio produced.

    RTF < 1  the voice generates faster than it plays, so clause streaming
             keeps the speaker fed and first-audio is the first piece only
    RTF > 1  it can never catch up; with a prebuffer B, playback underruns
             after B / (RTF - 1) seconds of speech, and streaming converts
             one long silence into a quick start plus a stutter

Measured 2026-08-17: RTF 1.33-1.61 on an RTX A6000 (Ada), 0.86-0.90 on an
RTX 5080 (Blackwell). The deployment target is a DGX Spark (GB10) with about
half the 5080's compute and a third of its bandwidth, so its number is NOT
known by extrapolation -- run this there.

`--clock-sweep` locks the GPU to each clock ceiling in turn and reports RTF
against it. A flat curve means the loop is overhead-bound (a weaker host CPU
is then the risk); a linear one means compute-bound, and the target's core
count predicts the result. Needs permission to set clocks; it restores them
afterwards and degrades to a warning when it cannot.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

# Short, medium and long, because cost tracks GENERATED AUDIO rather than
# input characters -- 19 characters have produced 4.32 s of speech. Keep this
# list stable: every number in docs/tts-optimization.md was taken on it.
LINES = (
    "Widzę kanapę. Hau hau!",
    "Już się zatrzymuję!",
    "Szukam kanapy, ale nie widzę jej teraz – idę w stronę, gdzie są półki i meble.",
    "Jestem teraz przy stole w kuchni, patrzę w stronę okna. Zaraz sprawdzę co jest dalej.",
)
# What clause-level streaming actually exposes: only the first piece is
# waited for, the rest is generated while it plays.
FIRST_PIECES = ("Szukam kanapy,", "Widzę kanapę.", "Już się zatrzymuję!")


def load_model(device: str, ref: str = ""):
    """Chatterbox multilingual, with the watermarker fallback the fresh-box
    trap needs (resemble-perth resolves to None when its optional deps are
    broken, and chatterbox then crashes constructing it)."""
    import perth

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        from perth.dummy_watermarker import DummyWatermarker

        perth.PerthImplicitWatermarker = DummyWatermarker
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    if ref:
        # Encode the voice ONCE; passing audio_prompt_path per call re-runs
        # the whole reference pipeline every time (see tts_server.py).
        model.prepare_conditionals(ref)
    return model


def time_line(model, text: str, language: str) -> tuple[float, float]:
    """(wall seconds, audio seconds) for one synthesis."""
    import torch

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.monotonic()
    wav = model.generate(text, language_id=language)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.monotonic() - t0, wav.shape[-1] / model.sr


def measure(model, lines, language: str, reps: int) -> dict:
    took, audio = [], []
    for text in lines:
        for _ in range(reps):
            dt, secs = time_line(model, text, language)
            took.append(dt)
            audio.append(secs)
    rtfs = [t / a for t, a in zip(took, audio) if a]
    return {
        "n": len(took),
        "median_s": statistics.median(took),
        "median_audio_s": statistics.median(audio),
        "rtf": statistics.median(rtfs),
        "rtf_min": min(rtfs),
        "rtf_max": max(rtfs),
    }


def apply_variant(model, name: str) -> None:
    """Variants kept because they were MEASURED, and the measurement is the
    point: on an RTX 5080 none of them beat `base` (see the docs). Re-run
    them on new hardware rather than assuming either result."""
    if name == "base":
        return
    if name == "nohidden":
        pm = model.t3.patched_model
        if not getattr(pm, "_nohidden", False):
            orig = pm.forward

            def forward(*a, **kw):
                kw["output_hidden_states"] = False
                return orig(*a, **kw)

            pm.forward, pm._nohidden = forward, True
        return
    if name == "sdpa":
        apply_variant(model, "nohidden")
        model.t3.patched_model.alignment_stream_analyzer = None
        cfg = model.t3.tfmr.config
        cfg._attn_implementation = "sdpa"
        cfg.output_attentions = False
        return
    if name == "compile":
        import torch

        model.t3.patched_model = torch.compile(
            model.t3.patched_model, mode="reduce-overhead"
        )
        return
    raise ValueError(f"unknown variant {name!r}")


def set_clock(mhz: int | None) -> bool:
    """Lock (or release) the graphics clock. False when not permitted."""
    cmd = ["nvidia-smi", "-lgc", str(mhz)] if mhz else ["nvidia-smi", "-rgc"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def verdict(rtf: float) -> str:
    if rtf < 0.8:
        return "streaming is comfortable here"
    if rtf < 1.0:
        return "streaming works, but with little margin"
    return (
        f"SLOWER THAN REAL TIME: with a 1 s prebuffer, playback underruns "
        f"after {1.0 / (rtf - 1.0):.1f} s of speech"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ref", default="", help="voice-clone reference wav")
    p.add_argument("--language", default="pl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--variants", default="base",
                   help="base,nohidden,sdpa,compile (all measured as no-gain "
                        "on an RTX 5080; re-check on new hardware)")
    p.add_argument("--clock-sweep", default="",
                   help="comma-separated MHz ceilings: is this loop clock-bound?")
    p.add_argument("--json", default="", help="also write the results here")
    args = p.parse_args(argv)

    name = gpu_name()
    print(f"gpu: {name}", flush=True)
    t0 = time.monotonic()
    model = load_model(args.device, args.ref)
    print(f"loaded in {time.monotonic() - t0:.1f}s; warming", flush=True)
    time_line(model, "Hau hau.", args.language)   # build patched_model, warm CUDA

    results: dict = {"gpu": name, "ref": bool(args.ref), "variants": {},
                     "first_piece": {}, "clock_sweep": {}}

    for variant in args.variants.split(","):
        apply_variant(model, variant)
        if variant == "compile":
            time_line(model, "Hau hau.", args.language)  # pay compilation once
        r = measure(model, LINES, args.language, args.reps)
        results["variants"][variant] = r
        print(f"{variant:<10} median {r['median_s']:>6.2f}s  "
              f"audio {r['median_audio_s']:>5.2f}s  RTF {r['rtf']:>5.2f}  "
              f"({r['rtf_min']:.2f}-{r['rtf_max']:.2f})", flush=True)

    for piece in FIRST_PIECES:
        dt, secs = time_line(model, piece, args.language)
        results["first_piece"][piece] = {"s": dt, "audio_s": secs}
        print(f"first piece {piece!r:<24} {dt:>5.2f}s ({secs:.2f}s audio)", flush=True)

    if args.clock_sweep:
        for mhz in [int(m) for m in args.clock_sweep.split(",")]:
            if not set_clock(mhz):
                print(f"cannot lock clocks to {mhz} MHz (needs privileges); "
                      "skipping the sweep", flush=True)
                break
            r = measure(model, LINES[:2], args.language, 1)
            results["clock_sweep"][mhz] = r
            print(f"  {mhz:>5} MHz -> RTF {r['rtf']:.2f}", flush=True)
        set_clock(None)

    base = results["variants"].get("base") or next(iter(results["variants"].values()))
    print(f"\nRTF {base['rtf']:.2f} on {name}: {verdict(base['rtf'])}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
