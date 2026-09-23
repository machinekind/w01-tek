"""Robot profiles: one place per robot for the facts that follow the legs.

The profiles decide which URDF legs, which joint map, which default policy
and whether the knee clamp is on. The stock profile has to stay exactly what
the stack did before profiles existed, and a policy for other legs must be
refused before it reaches the drives.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy import policy_source, robots  # noqa: E402
from wojtek_policy.policy_source import default_policy, load_policy  # noqa: E402

CONFIG = PKG / "config"
SRC = PKG.parent
BODY_XACRO = SRC / "wojtek_description" / "urdf" / "body.urdf.xacro"

# The pin as it stood in policy_source.py before it moved into the profile.
STOCK_PIN = (
    "wojtek-quiet-locomotion@553795b13001cc1f519a4abc0235f275095129f8"
)


def test_profiles_are_well_formed():
    assert robots.DEFAULT_ROBOT in robots.PROFILES
    body = BODY_XACRO.read_text()
    for name, profile in robots.PROFILES.items():
        assert profile.name == name
        # A legs value the body macro accepts.
        assert f"legs == '{profile.legs}'" in body, profile.legs
        assert profile.training_robot in ("wojtek", "legs_v627")
        assert (CONFIG / profile.joint_map).is_file(), profile.joint_map
        assert isinstance(profile.clamp_knee, bool)
        if profile.default_policy is not None:
            # Without the organization: that comes from the environment.
            assert "/" not in profile.default_policy
            assert re.fullmatch(r"[\w.-]+@[0-9a-f]{40}", profile.default_policy)
        package, path = profile.sim_scene
        assert (SRC / package / path).is_file(), profile.sim_scene


def test_stock_profile_is_what_the_stack_did_before():
    wojtek = robots.get("wojtek")
    assert robots.DEFAULT_ROBOT == "wojtek"
    assert wojtek.legs == "stock"
    assert wojtek.training_robot == "wojtek"
    assert wojtek.joint_map == "joint_map.yaml"
    assert wojtek.default_policy == STOCK_PIN
    assert wojtek.clamp_knee is True
    assert wojtek.sim_scene == ("wojtek_pc", "config/scene_sim.xml")


def test_v2_profile_has_no_pin_and_no_knee_clamp():
    v2 = robots.get("wojtek_v2")
    assert v2.legs == "legs_v627"
    assert v2.training_robot == "legs_v627"
    assert v2.default_policy is None
    assert v2.clamp_knee is False


def test_unknown_robot_is_refused():
    with pytest.raises(ValueError, match="no_such_robot"):
        robots.get("no_such_robot")


def test_default_policy_follows_the_profile(monkeypatch):
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    assert default_policy("wojtek") == f"org/{STOCK_PIN}"
    assert default_policy() == f"org/{STOCK_PIN}"
    assert default_policy("wojtek_v2") == ""
    monkeypatch.delenv("HF_ORGANIZATION")
    assert default_policy("wojtek") == ""


def test_active_policy_for_a_robot_without_a_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path / "policies"))
    assert policy_source.active_policy("wojtek_v2") == ""
    # The override file still wins. The launch then checks it against the
    # profile when it loads the meta.
    (tmp_path / "policy_override").write_text("org/other@main\n")
    assert policy_source.active_policy("wojtek_v2") == "org/other@main"


def test_default_cli_for_a_robot_without_a_pin(monkeypatch, capsys):
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    assert policy_source.main(["--default", "--robot", "wojtek_v2"]) == 2
    assert "wojtek_v2" in capsys.readouterr().out
    assert policy_source.main(["--default", "--robot", "no_such_robot"]) == 2


def test_robots_cli_prints_the_pin(capsys):
    # What deploy.sh reads to decide whether there is a default to resolve.
    assert robots.main(["wojtek"]) == 0
    assert capsys.readouterr().out.strip() == STOCK_PIN
    assert robots.main(["wojtek_v2"]) == 0
    assert capsys.readouterr().out.strip() == ""
    assert robots.main(["no_such_robot"]) == 2


def test_identity_map_maps_every_joint_to_itself():
    yaml = pytest.importorskip("yaml")
    from wojtek_policy.joint_map import JointMap

    stock = yaml.safe_load((CONFIG / "joint_map.yaml").read_text())["joint_map"]
    jm = JointMap(CONFIG / robots.get("wojtek_v2").joint_map)
    # All 20 joints the stock map covers, passive ones included, because
    # the MuJoCo plant converts every joint it publishes.
    assert jm.names() == list(stock)
    names = jm.names()
    q = np.random.default_rng(0).uniform(-3, 3, len(names))
    assert np.array_equal(jm.to_urdf(names, q), q)
    assert np.array_equal(jm.to_mjc(names, q), q)
    assert np.array_equal(jm.vel_to_urdf(names, q), q)


# -- a policy for the wrong legs is refused ------------------------------------

def test_meta_for_other_legs_is_refused():
    v627 = {"robot": "legs_v627"}
    stock = {"robot": "wojtek"}
    with pytest.raises(ValueError, match="legs_v627.*wojtek|wojtek.*legs_v627"):
        robots.check_policy(v627, "wojtek")
    with pytest.raises(ValueError, match="wojtek_v2"):
        robots.check_policy(stock, "wojtek_v2")
    robots.check_policy(stock, "wojtek")
    robots.check_policy(v627, "wojtek_v2")


def test_meta_without_a_robot_field_is_the_stock_robot():
    # Every policy exported before the field existed, the current pin
    # included, was trained on the stock legs.
    robots.check_policy({}, "wojtek")
    with pytest.raises(ValueError):
        robots.check_policy({}, "wojtek_v2")


def _policy_dir(tmp_path, robot):
    """A policy directory whose meta names `robot`. Only the meta is read."""
    (tmp_path / "policy.npz").write_bytes(b"")
    meta = {
        "schema_version": 2,
        "run_name": "synthetic",
        "robot": robot,
        "knee_singularity": None,
        "pd": {"kp": 60.0, "kd": 2.0, "max_torque": 22.0},
    }
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))
    return str(tmp_path)


def test_load_policy_refuses_other_legs_when_given_a_robot(tmp_path):
    ref = _policy_dir(tmp_path, "legs_v627")
    with pytest.raises(ValueError, match="legs_v627"):
        load_policy(ref, robot="wojtek")
    assert load_policy(ref, robot="wojtek_v2").meta["robot"] == "legs_v627"
    # No robot, no check: the behaviour callers had before profiles.
    assert load_policy(ref).meta["robot"] == "legs_v627"


def test_runtime_refuses_other_legs_when_given_a_robot(tmp_path):
    from wojtek_policy.policy import WojtekPolicy

    _policy_dir(tmp_path, "legs_v627")
    with pytest.raises(ValueError, match="legs_v627"):
        WojtekPolicy(tmp_path / "policy.npz", robot="wojtek")
