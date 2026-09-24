"""Render-only copy of the Wojtek model with decimated meshes.

Colab has no NVIDIA OpenGL, so MuJoCo renders in software (llvmpipe) and the
robot's 600k-triangle visual meshes cost ~0.8 s per frame. Decimating them to
a few percent brings a frame under 0.1 s. Physics keeps running on the
original model; only qpos is copied into this one for drawing.
"""

from pathlib import Path

import mujoco
import numpy as np

FRACTION = 0.02   # faces kept per mesh; the simplifier stops early on small parts
MIN_FACES = 300


def decimate_meshes(meshdir: Path, out: Path) -> None:
    import fast_simplification as fs
    import trimesh

    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(meshdir.glob("*.stl")):
        if (out / f.name).exists():
            continue
        m = trimesh.load(f, force="mesh", process=True)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        target = max(MIN_FACES, int(len(m.faces) * FRACTION))
        pts, faces = m.vertices.astype(np.float64), m.faces.astype(np.int64)
        for _ in range(6):
            if len(faces) <= target * 1.1:
                break
            pts, faces = fs.simplify(pts, faces, target_count=target, agg=10)
            d = trimesh.Trimesh(pts, faces, process=True)
            d.merge_vertices()
            pts, faces = d.vertices.astype(np.float64), d.faces.astype(np.int64)
        trimesh.Trimesh(pts, faces, process=True).export(out / f.name)


def render_model(scene_xml: Path, cache: Path) -> mujoco.MjModel:
    """The scene compiled against decimated meshes (built once into `cache`)."""
    cache = Path(cache).resolve()          # meshdir jest względem pliku modelu, więc ścieżka absolutna
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
    decimate_meshes(meshdir, cache)
    spec.meshdir = str(cache)
    return spec.compile()
