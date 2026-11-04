"""Camera fixture integrity and human-label validation, without ROS or a model."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_script("capture_frames")
bench = load_script("point_bench")


def frame(sec, nanosec=0):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def test_capture_uses_nearest_depth_and_refuses_stale_pair():
    colour = frame(2)
    stale, near = frame(1), frame(2, 50_000_000)
    assert capture.nearest_depth(colour, [stale, near], 0.1) is near
    assert capture.nearest_depth(colour, [stale], 0.1) is None
    assert capture.nearest_depth(colour, [], 0.1) is None


def test_depth_decoding_preserves_millimetres_across_endianness_and_padding():
    msg = SimpleNamespace(encoding="16UC1", width=2, height=2, step=6,
                          is_bigendian=True, data=b"\x00\x64\x03\xe8xx\x07\xd0\x0b\xb8yy")
    assert capture.depth_array(msg).tolist() == [[100, 1000], [2000, 3000]]


def test_depth_rejects_unsupported_units_and_truncated_buffer():
    msg = SimpleNamespace(encoding="32FC1", width=1, height=1, step=4,
                          is_bigendian=False, data=bytes(4))
    with pytest.raises(ValueError, match="16UC1"):
        capture.depth_array(msg)
    msg.encoding, msg.step, msg.data = "16UC1", 2, b"\x00"
    with pytest.raises(ValueError, match="buffer length"):
        capture.depth_array(msg)


def test_benchmark_refuses_unreviewed_or_empty_annotations():
    with pytest.raises(ValueError, match="annotation_status"):
        bench.validate_annotation({"i": 0, "annotation_status": "unreviewed", "truth": {}})
    with pytest.raises(ValueError, match="label"):
        bench.validate_annotation({"i": 0, "annotation_status": "reviewed", "truth": {}})


def test_legacy_labels_remain_compatible_but_new_captures_need_explicit_review():
    entry = {"i": 0, "truth": {
        "pillar": {"visible": True, "u_norm": 0.5, "v_norm": 0.5},
    }}
    assert bench.validate_annotation(entry) == (848, 480)
    # Supplying valid labels does not clear a new capture's review requirement.
    entry["annotation_status"] = "unreviewed"
    with pytest.raises(ValueError, match="annotation_status"):
        bench.validate_annotation(entry)
    entry["annotation_status"] = "reviewed"
    assert bench.validate_annotation(entry) == (848, 480)


@pytest.mark.parametrize("coordinate", [float("nan"), -0.1, 1.1, None])
def test_benchmark_refuses_invalid_human_point(coordinate):
    entry = {"i": 1, "truth": {"pillar": {"visible": True, "u_norm": coordinate, "v_norm": 0.5}}}
    with pytest.raises(ValueError, match="normalized coordinates"):
        bench.validate_annotation(entry)


def test_explicit_absent_label_needs_no_invented_pixel():
    assert bench.validate_annotation({"i": 0, "truth": {"chair": {"visible": False}}}) == (848, 480)


def test_nonlegacy_frame_size_is_used_for_pixel_and_metric_scoring():
    parsed = bench.parse('{"type":"goal","point_2d":[320,240]}', "absolute", 640, 480)
    assert (parsed["u_norm"], parsed["v_norm"]) == (0.5, 0.5)
    kc = [300, 0, 320, 0, 300, 240, 0, 0, 1]
    kd = [100, 0, 2, 0, 100, 2, 0, 0, 1]
    point = bench.resolve(np.full((5, 5), 1000, dtype=np.uint16), 0.5, 0.5,
                          kc, kd, width=640, height=480)
    assert point == pytest.approx((0, 0, 1))
    assert "640 wide, 480 high" in bench.prompt("pillar", "absolute", "en", 640, 480)


def test_benchmark_never_invents_an_absent_object(tmp_path, monkeypatch):
    entry = {"i": 0, "colour_width": 640, "colour_height": 480,
             "annotation_status": "reviewed",
             "truth": {"pillar": {"visible": True, "u_norm": 0.5, "v_norm": 0.5}},
             "k_colour": [300, 0, 320, 0, 300, 240, 0, 0, 1],
             "k_depth": [100, 0, 2, 0, 100, 2, 0, 0, 1]}
    (tmp_path / "index.json").write_text(json.dumps([entry]))
    (tmp_path / "frame_00.jpg").write_bytes(b"not decoded by the stubbed model")
    np.save(tmp_path / "frame_00_depth.npy", np.full((5, 5), 1000, dtype=np.uint16))
    calls = []

    def ask(*args):
        calls.append(args)
        return '{"type":"goal","point_2d":[500,500]}', 0.1, {}

    monkeypatch.setattr(bench, "ask", ask)
    monkeypatch.setattr(bench.sys, "argv", ["point_bench", "--frames", str(tmp_path), "--tag", "test"])
    assert bench.main() == 0
    assert len(calls) == 1
    summary = json.loads((tmp_path / "bench_test_qwen_en.json").read_text())["summary"]
    assert summary["n_visible"] == 1
    assert summary["n_absent"] == 0
