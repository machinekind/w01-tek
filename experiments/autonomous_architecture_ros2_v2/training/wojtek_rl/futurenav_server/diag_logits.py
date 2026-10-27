"""Per-step A/B: cached-suffix decode vs full prefill, identical state.

Both calls per step see the same VGGT cache and the same frames, so any
output difference is attributable to the LLM prefix-cache path alone.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
from validate_prefix_cache import synthetic_frame


def main():
    engine = server.FutureNavEngine(server.WEIGHTS)
    engine.reset("walk past the table and stop at the door")

    mismatches = 0
    for step in range(12):
        engine.rgb_list.append(synthetic_frame(step).convert("RGB"))
        images = engine._prepare_images()

        vggt_pre = engine.model.past_key_values_vggt

        server.PREFIX_CACHE = True
        raw_cached = engine._call_model(images, engine.instruction)
        llm_post = (engine._llm_cache, engine._llm_cache_ids, engine._llm_cache_images)

        # Same step again, full prefill, same VGGT starting state.
        engine.model.past_key_values_vggt = vggt_pre
        server.PREFIX_CACHE = False
        raw_full = engine._call_model(images, engine.instruction)

        # Continue the episode along the cached path.
        engine._llm_cache, engine._llm_cache_ids, engine._llm_cache_images = llm_post

        match = "==" if raw_cached == raw_full else "!!"
        if raw_cached != raw_full:
            mismatches += 1
        print(f"step {step+1:2d}: {match} cached {raw_cached!r:16} full {raw_full!r:16}", flush=True)

    print(f"mismatches: {mismatches}/12")
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
