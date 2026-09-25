"""A/B validation: prefix-cached decode must match the uncached path exactly.

Runs the same 12-frame synthetic episode through the engine twice (uncached
generate() vs prefix-cached decode) and compares raw outputs token-for-token,
plus reports per-step latency. Run inside the deployed venv:

  ./venv/bin/python validate_prefix_cache.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image


def synthetic_frame(step: int) -> Image.Image:
    rng = np.random.default_rng(step)
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    arr[..., 2] = np.linspace(255, 0, 480, dtype=np.uint8)[:, None]
    x = 30 + step * 40
    arr[140:300, x : x + 90] = rng.integers(0, 255, 3, dtype=np.uint8)
    return Image.fromarray(arr)


def run_episode(engine, n_steps: int):
    engine.reset("walk past the table and stop at the door")
    outs, times = [], []
    for step in range(n_steps):
        t = time.time()
        out = engine.act(synthetic_frame(step))
        times.append(time.time() - t)
        outs.append(out["raw"])
    return outs, times


def main():
    import server

    engine = server.FutureNavEngine(server.WEIGHTS)
    n = 12

    server.PREFIX_CACHE = False
    base, t_base = run_episode(engine, n)
    server.PREFIX_CACHE = True
    cached, t_cached = run_episode(engine, n)

    ok = True
    for i, (a, b) in enumerate(zip(base, cached)):
        mark = "==" if a == b else "!!"
        if a != b:
            ok = False
        print(f"step {i+1:2d}: {mark} uncached {a!r:16} cached {b!r:16} "
              f"{t_base[i]:4.2f}s -> {t_cached[i]:4.2f}s")
    print(f"mean latency: uncached {sum(t_base)/n:.2f}s cached {sum(t_cached)/n:.2f}s")
    print("VALIDATE_OK" if ok else "VALIDATE_MISMATCH")


if __name__ == "__main__":
    main()
