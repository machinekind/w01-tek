#!/usr/bin/env python3
"""Offline pointing benchmark against the Ollama VLM endpoint.

For every reviewed frame (index.json from capture_frames.py) and every
object visible in it, ask the model to point at the object; score the pixel
against the projected truth and, through the frame's own depth image, the
metric error of the resolved point (the same maths pixel_goal_node runs).
Also asks for an object that is NOT visible, to measure hallucination.

  python3 point_bench.py --frames ~/e2e/frames --url http://127.0.0.1:11434/v1 \
      --model qwen3-vl:30b-a3b-instruct [--fmt qwen|absolute] [--json-schema]

Output: one line per query + a summary; a JSON report next to the frames.
"""

import argparse
import base64
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

NAMES = {  # what a user would call the thing, EN and PL
    "crate": ("the orange wooden crate", "pomarańczowa drewniana skrzynia"),
    "pillar": ("the purple pillar", "fioletowy słup"),
    "low_box": ("the low orange box on the floor", "niska pomarańczowa skrzynka na podłodze"),
}
ABSENT = {  # a plausible object that is never in this scene
    "chair": ("the chair", "krzesło"),
    "person": ("the person", "człowiek"),
}

SYSTEM = (
    "You are the navigation brain of a small quadruped robot. You see one photo "
    "from its forward camera, mounted 20 cm above the floor. Answer with exactly "
    "one JSON object and nothing else."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["goal", "turn", "done", "not_visible"]},
        "point_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                     "minItems": 2, "maxItems": 2},
        "label": {"type": "string"},
        "deg": {"type": "integer", "minimum": -180, "maximum": 180},
    },
    "required": ["type"],
    "additionalProperties": False,
}


def prompt(name, fmt, lang, width=848, height=480):
    obj = NAMES.get(name, ABSENT.get(name))[0 if lang == "en" else 1]
    if lang == "en":
        task = f"Task: walk to {obj}."
    else:
        task = f"Zadanie: podejdź do: {obj}."
    if fmt == "absolute":
        coords = f"point_2d = [x, y] in pixels of this image ({width} wide, {height} high)."
    else:
        coords = "point_2d = [x, y], integers 0-1000 normalised to the image width and height."
    return (
        f"{task}\n"
        "If the target is visible, reply {\"type\":\"goal\",\"point_2d\":[x,y],\"label\":\"<what>\"} "
        "with the point ON the target object. If it is not visible in this photo, reply "
        "{\"type\":\"not_visible\"}. Never guess a point for something you cannot see.\n"
        f"{coords}"
    )


def ask(url, model, jpeg_b64, text, use_schema, max_tokens=60, timeout=120):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
                {"type": "text", "text": text},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if use_schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "nav", "schema": SCHEMA, "strict": True}}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    msg = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    return msg, dt, usage


def parse(msg, fmt, width=848, height=480):
    s = msg.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    try:
        d = json.loads(s[s.find("{"):s.rfind("}") + 1])
    except Exception:
        return None
    if d.get("type") == "goal" and isinstance(d.get("point_2d"), list) and len(d["point_2d"]) == 2:
        a, b = float(d["point_2d"][0]), float(d["point_2d"][1])
        if fmt == "absolute":
            d["u_norm"], d["v_norm"] = a / width, b / height
        else:
            d["u_norm"], d["v_norm"] = a / 1000.0, b / 1000.0
    return d


def resolve(depth, u_norm, v_norm, kc, kd, radius=4, width=848, height=480):
    """pixel_goal.py's chain: colour pixel -> depth pixel -> patch median -> point."""
    u_c, v_c = u_norm * width, v_norm * height
    tx, ty = (u_c - kc[2]) / kc[0], (v_c - kc[5]) / kc[4]
    u_d, v_d = kd[2] + kd[0] * tx, kd[5] + kd[4] * ty
    h, w = depth.shape
    ui, vi = int(round(u_d)), int(round(v_d))
    patch = depth[max(0, vi - radius):min(h, vi + radius + 1), max(0, ui - radius):min(w, ui + radius + 1)]
    if patch.size == 0:
        return None
    z = patch.astype(np.float32) * 1e-3
    ok = z[(z > 0.2) & (z < 3.0)]
    if ok.size < 0.3 * patch.size:
        return None
    zm = float(np.median(ok))
    return ((u_d - kd[2]) / kd[0] * zm, (v_d - kd[5]) / kd[4] * zm, zm)


def validate_annotation(entry):
    """Refuse unreviewed captures; an absent target must be labelled explicitly."""
    # Legacy labelled fixtures have no annotation_status field; keep accepting
    # that format. New captures explicitly start as unreviewed, even once labelled.
    if entry.get("annotation_status", "reviewed") != "reviewed":
        raise ValueError(f"frame {entry['i']}: annotate truth and set annotation_status to reviewed")
    truth = entry.get("truth")
    if not isinstance(truth, dict) or not truth:
        raise ValueError(f"frame {entry['i']}: at least one reviewed label is required")
    for name, target in truth.items():
        if name not in NAMES and name not in ABSENT:
            raise ValueError(f"frame {entry['i']}: unknown label {name!r}; add its prompt name first")
        if not isinstance(target, dict) or type(target.get("visible")) is not bool:
            raise ValueError(f"frame {entry['i']}: {name} needs a boolean visible value")
        if target["visible"]:
            coords = [target.get("u_norm"), target.get("v_norm")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in coords):
                raise ValueError(f"frame {entry['i']}: {name} needs finite normalized coordinates in [0, 1]")
    width, height = entry.get("colour_width", 848), entry.get("colour_height", 480)
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f"frame {entry['i']}: invalid colour dimensions")
    return width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    ap.add_argument("--fmt", default="qwen", choices=["qwen", "absolute"])
    ap.add_argument("--json-schema", action="store_true")
    ap.add_argument("--lang", default="en", choices=["en", "pl"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    frames = Path(a.frames)
    index = json.load(open(frames / "index.json"))
    # Validate the whole fixture before making any model requests.
    for e in index:
        if "error" not in e:
            validate_annotation(e)
    rows = []
    for e in index:
        if "error" in e:
            continue
        i = e["i"]
        width, height = validate_annotation(e)
        jpeg = base64.b64encode((frames / f"frame_{i:02d}.jpg").read_bytes()).decode()
        depth = np.load(frames / f"frame_{i:02d}_depth.npy")
        kc, kd = e["k_colour"], e["k_depth"]
        # the truth in the camera frame, for the metric error: re-derive from the
        # truth pixel and the frame's depth (what a perfect pointer would get)
        queries = [(n, t) for n, t in e["truth"].items() if t["visible"]]
        absent = [n for n in e["truth"] if not e["truth"][n]["visible"]]
        for name, t in queries + [(n, None) for n in absent]:
            try:
                msg, dt, usage = ask(a.url, a.model, jpeg, prompt(name, a.fmt, a.lang, width, height), a.json_schema)
            except Exception as ex:  # noqa: BLE001
                rows.append(dict(frame=i, obj=name, error=f"{type(ex).__name__}: {ex}"))
                print(f"frame {i} {name}: ERROR {ex}", flush=True)
                continue
            d = parse(msg, a.fmt, width, height)
            row = dict(frame=i, obj=name, visible=t is not None, seconds=round(dt, 2),
                       out_tokens=usage.get("completion_tokens"), raw=msg[:160], parsed=bool(d))
            if t is None:
                row["correct"] = bool(d and d.get("type") in ("not_visible", "turn"))
                row["hallucinated"] = bool(d and d.get("type") == "goal")
            elif d and d.get("type") == "goal":
                du = (d["u_norm"] - t["u_norm"]) * width
                dv = (d["v_norm"] - t["v_norm"]) * height
                row["px_err"] = round(math.hypot(du, dv), 1)
                p_model = resolve(depth, d["u_norm"], d["v_norm"], kc, kd, width=width, height=height)
                p_truth = resolve(depth, t["u_norm"], t["v_norm"], kc, kd, width=width, height=height)
                if p_model and p_truth:
                    row["metric_err_m"] = round(math.dist(p_model, p_truth), 3)
                    row["range_model_m"] = round(p_model[2], 2)
                row["depth_ok"] = p_model is not None and p_truth is not None
                # Pointing is judged in metres when the frame's depth reaches the
                # object (the resolver's own answer); beyond the D435's window
                # (>3 m) by the pixel alone, 60 px ~ 7 % of the width.
                if row["depth_ok"]:
                    row["correct"] = bool(row["metric_err_m"] < 0.25)
                else:
                    row["correct"] = bool(row["px_err"] < 60.0 * width / 848.0)
                    row["judged_by"] = "pixel (no depth at this range)"
            else:
                row["correct"] = False
                row["missed"] = True
            rows.append(row)
            print(f"frame {i} {name:8s} vis={t is not None!s:5s} {dt:5.2f}s tok={usage.get('completion_tokens')} "
                  f"-> {msg.strip()[:90]}  {'OK' if row.get('correct') else 'X'}"
                  + (f" px_err={row['px_err']}" if 'px_err' in row else "")
                  + (f" m_err={row['metric_err_m']}" if 'metric_err_m' in row else ""), flush=True)
    vis = [r for r in rows if r.get("visible") and "error" not in r]
    abs_ = [r for r in rows if r.get("visible") is False and "error" not in r]
    lat = [r["seconds"] for r in rows if "seconds" in r]
    summ = dict(model=a.model, fmt=a.fmt, lang=a.lang, json_schema=a.json_schema, tag=a.tag,
                n_visible=len(vis), hit_rate=round(sum(r.get("correct", False) for r in vis) / max(1, len(vis)), 3),
                n_depth_resolvable=sum(1 for r in vis if r.get("depth_ok")),
                hit_rate_resolvable=round(sum(r.get("correct", False) for r in vis if r.get("depth_ok")) / max(1, sum(1 for r in vis if r.get("depth_ok"))), 3),
                missed=sum(1 for r in vis if r.get("missed")),
                median_px_err=statistics.median([r["px_err"] for r in vis if "px_err" in r]) if any("px_err" in r for r in vis) else None,
                median_metric_err_m=statistics.median([r["metric_err_m"] for r in vis if "metric_err_m" in r]) if any("metric_err_m" in r for r in vis) else None,
                n_absent=len(abs_), absent_correct=round(sum(r.get("correct", False) for r in abs_) / max(1, len(abs_)), 3),
                hallucination_rate=round(sum(r.get("hallucinated", False) for r in abs_) / max(1, len(abs_)), 3),
                latency_p50=round(statistics.median(lat), 2) if lat else None,
                latency_max=round(max(lat), 2) if lat else None,
                errors=sum(1 for r in rows if "error" in r))
    print(json.dumps(summ), flush=True)
    tag = (a.tag or a.model.split("/")[-1]) + f"_{a.fmt}_{a.lang}" + ("_schema" if a.json_schema else "")
    with open(frames / f"bench_{tag}.json", "w") as f:
        json.dump(dict(summary=summ, rows=rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
