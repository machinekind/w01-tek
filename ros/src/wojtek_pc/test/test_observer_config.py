from pathlib import Path
import re

import yaml


def test_observer_cannot_expose_control_or_arbitrary_files():
    config = Path(__file__).resolve().parents[1] / "config/observer.yaml"
    params = yaml.safe_load(config.read_text())["/**"]["ros__parameters"]
    assert params["address"] == "127.0.0.1"
    assert set(params["capabilities"]) == {"connectionGraph", "assets"}
    for name in ("client_topic_whitelist", "service_whitelist", "param_whitelist"):
        assert not any(re.search(pattern, "/wojtek/arm") for pattern in params[name])
    assets = params["asset_uri_allowlist"]
    assert any(re.fullmatch(p, "package://wojtek_description/meshes/body.stl") for p in assets)
    for uri in ("file:///etc/passwd", "https://example.com/private", "package://other/private.stl"):
        assert not any(re.fullmatch(p, uri) for p in assets)
