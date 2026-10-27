"""FutureNav-4B action server.

Wraps the FutureNav inference path from eval/agent.py (minus habitat) behind a
small HTTP API so a remote simulator can drive it frame by frame.

Episode state (frame history + VGGT KV cache) lives server-side because the
model accumulates past_key_values_vggt across steps; the client only sends the
current ego frame each step.

  POST /reset {"instruction": str}          -> {"ok": true}
  POST /act   {"frame_b64": str(jpeg/png)}  -> {"action": str, "raw": str, "step": int}
  GET  /health                              -> {"status": "ok", ...}

Run:
  FUTURENAV_WEIGHTS=~/futurenav/weights/FutureNav-4B-Base \
  ~/futurenav/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8100
"""

import base64
import os
import sys
import threading
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

FUTURENAV_SRC = os.path.expanduser(
    os.environ.get("FUTURENAV_SRC", "~/futurenav/FutureNav/src")
)
sys.path.insert(0, FUTURENAV_SRC)

from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # noqa: E402
from qwen_vl_utils import extract_vision_info  # noqa: E402
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from qwen_vl.model.modeling_qwen3_vl_obshead import Qwen3VLForFutureNav  # noqa: E402

MIN_PIXELS = 28 * 28
MAX_PIXELS = 1605632
ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
DEFAULT_ACTION = "MOVE_FORWARD"

WEIGHTS = os.path.expanduser(
    os.environ.get("FUTURENAV_WEIGHTS", "~/futurenav/weights/FutureNav-4B-Base")
)
MAX_HISTORY_FRAMES = int(os.environ.get("FUTURENAV_MAX_HISTORY", "8"))
# Upstream eval uses eager attention, but on a 16 GB card shared with other
# processes the eager softmax buffers OOM; sdpa is numerically equivalent
# (same math, fused kernels) and far leaner.
ATTN_IMPL = os.environ.get("FUTURENAV_ATTN", "sdpa")
# bf16 weights need ~11.3 GiB and the card is shared (~4.2 GiB used by other
# services), leaving no activation headroom. 8-bit quantizes the language
# model blocks only; vision tower, VGGT and heads stay bf16.
QUANT = os.environ.get("FUTURENAV_QUANT", "8bit")  # "8bit" | "none"
# VGGT geometry cache policy. Upstream ships a StreamingLLM-style eviction
# (StartRecentKVCache(start=8, recent=48, k_seq_dim=2)) but dim 2 of the
# aggregator cache is the FRAME axis -- measured (1, 16, n_frames, 1615, 64),
# ~151 MiB per frame across layers. 8+48 = 56 frames never triggers within an
# R2R-length episode, so on their 96 GB H20 the cache silently grows all
# episode and on a 16 GB card it OOMs around frame 15 (latency also grows,
# 0.5 s -> 1.5 s/step by frame 15: the aggregator attends over every cached
# frame).
#
#   "0"     drop the cache every step (no cross-step geometry, min memory)
#   "full"  upstream behavior verbatim (needs a big dedicated GPU)
#   "<N>"   bounded window: first frame kept as the geometry anchor
#           (config.reference_frame == "first") + the N-1 most recent frames,
#           evicted with upstream's own StartRecentKVCache mechanism.
VGGT_CACHE = os.environ.get("FUTURENAV_VGGT_CACHE", "6")
# Actions are at most ~5 tokens ("MOVE_FORWARD" + eos); upstream's 24 just
# spends decode steps on the rare ramble, which the parser ignores anyway.
MAX_NEW_TOKENS = int(os.environ.get("FUTURENAV_MAX_NEW_TOKENS", "8"))
# Reuse the LLM KV cache across steps. The prompt is [system][img1..imgN]
# [instruction+action menu]: history is append-only while it fits (<= 9
# frames), so each step shares everything up to the previous step's last
# image verbatim. We keep the prompt KV cache per episode, crop it to the
# longest common token prefix, and prefill only the new image + trailing
# text (~470 tokens instead of ~3600). Token-exact with the uncached path:
# mrope position ids are computed on the FULL sequence and sliced, because
# the upstream cached-continuation branch assumes a text-only suffix.
PREFIX_CACHE = os.environ.get("FUTURENAV_PREFIX_CACHE", "1") == "1"
# History frame selection once the episode outgrows the window:
#   "uniform"  upstream-exact: linspace over the whole episode. Reshuffles
#              which frames appear mid-prompt, so from ~step 11 the prefix
#              cache degrades and model time doubles (measured 0.45 -> 1.0 s).
#   "window"   first frame (anchor) + a recent tail with CHUNKED eviction:
#              the tail grows append-only (full cache reuse) and sheds its
#              4 oldest frames in one go when full, so the expensive
#              re-prefill happens 1 step in 4 instead of every step. (A
#              stride-1 sliding window would shift every position each step
#              and be exactly as cache-hostile as uniform.)
HISTORY_MODE = os.environ.get("FUTURENAV_HISTORY", "window")  # window | uniform
HISTORY_EVICT_CHUNK = 4

SYSTEM_PROMPT = (
    "You are a visual language navigation model, and your should go to the "
    "locations to complete the given task. Compare the observation and "
    "instruction to infer your current progress, and then select the correct "
    "direction from the candidates to go to the target location and finish "
    "the task."
)


class NoEpisodeError(RuntimeError):
    """POST /act arrived before any /reset."""


class FutureNavEngine:
    """Single-episode FutureNav inference engine (mirrors eval/agent.py)."""

    def __init__(self, checkpoint_path: str, max_history_frames: int = 8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_history_frames = max_history_frames
        self.lock = threading.Lock()

        config = AutoConfig.from_pretrained(checkpoint_path)
        if getattr(config, "model_type", "") != "qwen3_vl":
            raise ValueError("FutureNav server expects a Qwen3-VL checkpoint.")
        quant_kwargs = {}
        if QUANT == "8bit":
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    "visual",
                    "vggt",
                    "lm_head",
                    "forward_head",
                    "inverse_head",
                    "inverse_next_proj",
                    "gen_head",
                ],
            )
        # ignore_mismatched_sizes: the released FutureNav-4B-Base checkpoint
        # carries auxiliary observation heads from an older head layout
        # (Qwen3VLForJanusVLN_ObsHeadV4); evaluation never runs those heads,
        # so reinitializing the mismatched aux params is harmless.
        self.model = Qwen3VLForFutureNav.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            attn_implementation=ATTN_IMPL,
            mode="evaluation",
            ignore_mismatched_sizes=True,
            **quant_kwargs,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, padding_side="left"
        )
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            max_pixels=MAX_PIXELS,
            min_pixels=MIN_PIXELS,
            padding_side="left",
        )

        self.instruction = ""
        self.rgb_list: list[Image.Image] = []
        self.step = 0

        self._vggt_window = None
        if VGGT_CACHE not in ("0", "full"):
            from qwen_vl.model.modeling_qwen3_vl import StartRecentKVCache

            window = max(2, int(VGGT_CACHE))
            self._vggt_window = StartRecentKVCache(
                start_size=1, recent_size=window - 1, k_seq_dim=2, v_seq_dim=2
            )

        self._llm_cache = None        # DynamicCache over the prompt prefix
        self._llm_cache_ids = None    # 1-D tensor of token ids the cache covers
        self._llm_cache_images = None  # PIL objects those tokens showed
        self._hist_tail = None        # window-mode recent frames (chunked eviction)

    def reset(self, instruction: str):
        with self.lock:
            self.instruction = instruction
            self.rgb_list = []
            self.step = 0
            self.model.past_key_values_vggt = None
            self._llm_cache = None
            self._llm_cache_ids = None
            self._llm_cache_images = None
            self._hist_tail = None

    def _prepare_images(self):
        history_len = len(self.rgb_list) - 1
        if history_len <= self.max_history_frames:
            return self.rgb_list[:history_len] + [self.rgb_list[-1]]
        if HISTORY_MODE == "window":
            # Anchor + append-only tail; evict a chunk of oldest tail frames
            # only when full, keeping the prompt prefix stable between
            # evictions.
            if self._hist_tail is None:
                self._hist_tail = list(self.rgb_list[1:-1])
            self._hist_tail.append(self.rgb_list[-1])
            if len(self._hist_tail) > self.max_history_frames:
                del self._hist_tail[:HISTORY_EVICT_CHUNK]
            return [self.rgb_list[0]] + list(self._hist_tail)
        indices = np.linspace(0, history_len, self.max_history_frames + 1, dtype=int)
        return [self.rgb_list[i] for i in indices]

    def _call_model(self, images, instruction: str) -> str:
        t0 = time.perf_counter()
        message = [{"role": "system", "content": SYSTEM_PROMPT}]
        context = (
            "These images are your historical observations and your current "
            f"observation.\n Your task is to {instruction} \n You should take "
            "one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n "
            "TURN_RIGHT\n STOP."
        )
        image_content = [{"type": "image", "image": v} for v in images]
        message.append(
            {"role": "user", "content": image_content + [{"type": "text", "text": context}]}
        )
        messages = [message]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        images_vggt = []
        image_inputs = []
        for msg in messages:
            vision_info = extract_vision_info(msg)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" not in ele:
                    continue
                image = ele["image"]
                if not isinstance(image, Image.Image):
                    raise NotImplementedError("Unsupported image type")
                image = load_and_preprocess_images([image])[0]
                if i == len(vision_info) - 1:
                    cur_images_vggt.append(image)

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image_inputs.append(image[:, :height, :width])
            images_vggt.append(torch.stack(cur_images_vggt))

        t_pil = time.perf_counter()
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )
        device = self.model.device
        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
        t_proc = time.perf_counter()

        if PREFIX_CACHE:
            out = self._decode_with_prefix_cache(inputs, images)
            self.timings = {
                "pil_preproc_s": round(t_pil - t0, 3),
                "hf_processor_s": round(t_proc - t_pil, 3),
                "model_s": round(time.perf_counter() - t_proc, 3),
                "total_s": round(time.perf_counter() - t0, 3),
            }
            return out

        with torch.no_grad():
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self.timings = {
            "pil_preproc_s": round(t_pil - t0, 3),
            "hf_processor_s": round(t_proc - t_pil, 3),
            "model_s": round(time.perf_counter() - t_proc, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        }
        return answers[0]

    # -- prefix-cached greedy decode ----------------------------------------

    def _image_blocks(self, ids: torch.Tensor) -> list[tuple[int, int]]:
        """[start, end) index ranges of contiguous image-pad token runs."""
        mask = ids == self.model.config.image_token_id
        blocks = []
        start = None
        for i, m in enumerate(mask.tolist()):
            if m and start is None:
                start = i
            elif not m and start is not None:
                blocks.append((start, i))
                start = None
        if start is not None:
            blocks.append((start, len(mask)))
        return blocks

    def _decode_with_prefix_cache(self, inputs, images: list) -> str:
        """Greedy decode reusing the episode's prompt KV cache.

        Token-exact re-implementation of generate(do_sample=False): position
        ids come from get_rope_index over the full sequence (sliced to the
        suffix), so cached and uncached paths see identical mrope positions.
        """
        from transformers import DynamicCache

        ids = inputs.input_ids  # [1, L]
        full_len = ids.shape[1]
        grid = inputs.image_grid_thw  # [n_images, 3]
        blocks = self._image_blocks(ids[0])

        # Longest common prefix with the cached prompt. Token equality alone
        # is NOT enough: image tokens are identical placeholder ids whatever
        # the pixels, and once history subsampling starts, two consecutive
        # prompts can be token-identical while showing different frames. So
        # the prefix is additionally capped at the first image whose PIL
        # object differs from the previous step's (rgb_list entries are
        # stable objects, so identity comparison is exact).
        lcp = 0
        if self._llm_cache_ids is not None:
            prev = self._llm_cache_ids
            n = min(len(prev), full_len)
            neq = (prev[:n] != ids[0, :n]).nonzero()
            lcp = int(neq[0, 0]) if len(neq) else n

            prev_imgs = self._llm_cache_images or []
            n_common_imgs = 0
            for a, b in zip(images, prev_imgs):
                if a is not b:
                    break
                n_common_imgs += 1
            if n_common_imgs < len(blocks):
                lcp = min(lcp, blocks[n_common_imgs][0])
            for start, end in blocks:
                if start < lcp < end:
                    lcp = start
                    break

        suffix_block_idx = [i for i, (start, _) in enumerate(blocks) if start >= lcp]
        assert suffix_block_idx, "suffix must contain the current frame's image"

        # Rows of pixel_values belonging to the suffix images (t*h*w patches
        # per image, in prompt order).
        patches_per_img = [int(t * h * w) for t, h, w in grid.tolist()]
        row_ofs = np.cumsum([0] + patches_per_img)
        first = suffix_block_idx[0]
        pixel_values = inputs.pixel_values[row_ofs[first]:]
        suffix_grid = grid[first:]

        # mrope positions for the whole sequence, sliced to the suffix.
        attention_mask = torch.ones_like(ids)
        position_ids, rope_deltas = self.model.get_rope_index(
            ids, grid, None, attention_mask
        )
        self.model.rope_deltas = rope_deltas

        if self._llm_cache is None or lcp == 0:
            cache = DynamicCache()
            lcp = 0
        else:
            cache = self._llm_cache
            cache.crop(lcp)

        eos = self.tokenizer.eos_token_id
        generated: list[int] = []
        with torch.no_grad():
            out = self.model(
                input_ids=ids[:, lcp:],
                position_ids=position_ids[:, :, lcp:],
                past_key_values=cache,
                cache_position=torch.arange(lcp, full_len, device=ids.device),
                pixel_values=pixel_values,
                image_grid_thw=suffix_grid,
                images_vggt=inputs["images_vggt"],
                use_cache=True,
            )
            cache = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax())
            for i in range(MAX_NEW_TOKENS):
                generated.append(next_tok)
                if next_tok == eos:
                    break
                out = self.model(
                    input_ids=torch.tensor([[next_tok]], device=ids.device),
                    past_key_values=cache,
                    cache_position=torch.tensor([full_len + i], device=ids.device),
                    use_cache=True,
                )
                cache = out.past_key_values
                next_tok = int(out.logits[0, -1].argmax())

        # Keep only the prompt in the stored cache; generated tokens are not
        # part of the next step's prefix.
        cache.crop(full_len)
        self._llm_cache = cache
        self._llm_cache_ids = ids[0].clone()
        self._llm_cache_images = list(images)

        return self.processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    @staticmethod
    def parse_action(raw: str) -> str:
        raw = raw.strip()
        if raw in ACTIONS:
            return raw
        for name in ACTIONS:
            if name in raw:
                return name
        return DEFAULT_ACTION

    def act(self, frame: Image.Image) -> dict:
        with self.lock:
            if not self.instruction:
                raise NoEpisodeError("No active episode; call /reset first.")
            self.rgb_list.append(frame.convert("RGB"))
            images = self._prepare_images()
            try:
                raw = self._call_model(images, self.instruction)
            except torch.OutOfMemoryError:
                self.model.past_key_values_vggt = None
                self._llm_cache = None
                self._llm_cache_ids = None
                torch.cuda.empty_cache()
                print(f"OOM at step {self.step + 1}: dropped caches, retrying")
                raw = self._call_model(images, self.instruction)
            finally:
                if VGGT_CACHE == "0":
                    self.model.past_key_values_vggt = None
                elif self._vggt_window is not None:
                    self.model.past_key_values_vggt = self._vggt_window(
                        self.model.past_key_values_vggt
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.step += 1
            return {"action": self.parse_action(raw), "raw": raw, "step": self.step,
                    "timings": getattr(self, "timings", None)}


class ResetRequest(BaseModel):
    instruction: str


class ActRequest(BaseModel):
    frame_b64: str


app = FastAPI(title="FutureNav action server")
engine: FutureNavEngine | None = None


@app.on_event("startup")
def _load_model():
    global engine
    engine = FutureNavEngine(WEIGHTS, MAX_HISTORY_FRAMES)
    print(f"FutureNav engine ready on {engine.device} (weights: {WEIGHTS})")


@app.get("/health")
def health():
    vram = None
    if torch.cuda.is_available():
        vram = round(torch.cuda.memory_allocated() / 2**30, 2)
    return {
        "status": "ok" if engine is not None else "loading",
        "device": engine.device if engine else None,
        "step": engine.step if engine else None,
        "vram_gib": vram,
    }


@app.post("/reset")
def reset(req: ResetRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    if not req.instruction.strip():
        raise HTTPException(status_code=422, detail="Instruction must be non-empty")
    engine.reset(req.instruction.strip())
    return {"ok": True}


@app.post("/act")
def act(req: ActRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        data = base64.b64decode(req.frame_b64, validate=True)
        frame = Image.open(BytesIO(data))
        frame.load()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Bad frame_b64: {exc}") from exc
    try:
        return engine.act(frame)
    except NoEpisodeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except torch.OutOfMemoryError as exc:
        raise HTTPException(status_code=503, detail=f"GPU OOM: {exc}") from exc
