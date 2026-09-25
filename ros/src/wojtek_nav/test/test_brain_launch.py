"""brain.launch.py: the brain node with the endpoint and the camera wired
from the arguments (and VLM_URL from the environment), the 8B on vLLM as
the default model, the compressed picture as the default input."""

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

PKG_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launch_mod():
    path = PKG_DIR / "launch" / "brain.launch.py"
    spec = importlib.util.spec_from_file_location("brain_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node_params(launch_mod, monkeypatch, env_url=None, env_model=None, **overrides):
    for var, val in (("VLM_URL", env_url), ("VLM_MODEL", env_model)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    ld = launch_mod.generate_launch_description()
    ctx = LaunchContext()
    args = {}
    for entity in ld.entities:
        if hasattr(entity, "default_value") and entity.name not in overrides:
            args[entity.name] = "".join(
                s.perform(ctx) for s in entity.default_value
            )
    args.update(overrides)
    ctx.launch_configurations.update(args)
    nodes = [e for e in ld.entities if isinstance(e, Node)]
    assert len(nodes) == 1
    params = evaluate_parameters(ctx, nodes[0]._Node__parameters)
    return str(nodes[0]._Node__node_executable), params[0]


def test_defaults_are_the_8b_on_vllm_over_the_compressed_picture(launch_mod, monkeypatch):
    executable, params = _node_params(launch_mod, monkeypatch)
    assert executable == "vlm_brain_node"
    assert params["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert params["url"] == "http://127.0.0.1:8000/v1"
    assert params["compressed"] is True
    assert params["image_topic"] == "/camera/camera/color/image_raw"
    assert params["instruction"] == ""


def test_vlm_url_and_model_from_the_environment_are_the_defaults(launch_mod, monkeypatch):
    _, params = _node_params(launch_mod, monkeypatch,
                             env_url="http://dgx.example:8000", env_model="qwen3-vl-8b")
    assert params["url"] == "http://dgx.example:8000"
    assert params["model"] == "qwen3-vl-8b"


def test_arguments_reach_the_node(launch_mod, monkeypatch):
    _, params = _node_params(
        launch_mod, monkeypatch, env_url="http://ignored:1",
        url="http://box:11434/v1", model="qwen3-vl:8b", compressed="false",
        instruction="podejdź do fioletowego słupa",
    )
    assert params["url"] == "http://box:11434/v1"
    assert params["model"] == "qwen3-vl:8b"
    assert params["compressed"] is False
    assert params["instruction"] == "podejdź do fioletowego słupa"
