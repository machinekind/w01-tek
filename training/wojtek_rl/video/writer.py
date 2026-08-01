"""The mp4 writer every video-producing tool uses."""

from pathlib import Path


def write_video(out, frames, fps):
    """Write `frames` to `out`, creating its directory."""
    import shutil

    import mediapy

    if shutil.which("ffmpeg") is None:
        # No system ffmpeg (common on a bare Mac); fall back to the binary
        # bundled with imageio-ffmpeg, which is already in the venv.
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=fps)
