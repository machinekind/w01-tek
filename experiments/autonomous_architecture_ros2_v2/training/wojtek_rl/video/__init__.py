"""Shared video rendering: the mp4 writer, the scene renderer, the overlays.

Every tool that writes an mp4 of a rollout goes through here: `SceneView`
renders the chase camera at a chosen size and draws the optional torque strip
and onboard-depth inset, `write_video` writes the file.
"""

from wojtek_rl.video.overlays import compose, depth_rgb
from wojtek_rl.video.render import SceneView, frame_size, scene_model
from wojtek_rl.video.writer import write_video

__all__ = [
    "SceneView",
    "compose",
    "depth_rgb",
    "frame_size",
    "scene_model",
    "write_video",
]
