"""Smoke test: load FutureNav-4B and run a few steps on synthetic frames.

Verifies weights load on the GPU, the VGGT side input works, and the model
emits a parseable discrete action. Run inside the deployed venv:

  ./venv/bin/python smoke_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from server import FutureNavEngine, WEIGHTS


def synthetic_frame(step: int) -> Image.Image:
    """A 480x360 gradient with a moving block, so frames differ per step."""
    rng = np.random.default_rng(step)
    arr = np.zeros((360, 480, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 480, dtype=np.uint8)[None, :]
    arr[..., 2] = np.linspace(255, 0, 360, dtype=np.uint8)[:, None]
    x = 40 + step * 60
    arr[140:220, x : x + 80] = rng.integers(0, 255, 3, dtype=np.uint8)
    return Image.fromarray(arr)


def main():
    t0 = time.time()
    engine = FutureNavEngine(WEIGHTS)
    print(f"load: {time.time() - t0:.1f}s, device={engine.device}")
    print(f"vram after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    engine.reset("walk forward past the table and stop at the door")
    for step in range(3):
        t = time.time()
        out = engine.act(synthetic_frame(step))
        vram = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"step {out['step']}: action={out['action']!r} raw={out['raw']!r} "
            f"({time.time() - t:.1f}s, peak vram {vram:.2f} GiB)"
        )
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
