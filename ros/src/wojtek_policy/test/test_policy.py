"""Unit tests for the numpy policy runtime, resolver, and joint map.

Run: ros/build.sh test (or pytest with numpy+yaml on the host).

No shipped policy artifacts exist in the repo anymore (policies load by
reference, usually a Hugging Face repo id -- see policy_source.py), so
every policy here is synthetic: a single-layer network with a zero kernel,
whose output tanh(bias) is known in closed form for any observation.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy.joint_map import JointMap  # noqa: E402
from wojtek_policy.policy import (  # noqa: E402
    WojtekPolicy,
    gravity_from_quat,
    height_anchor,
)
from wojtek_policy import policy_source  # noqa: E402
from wojtek_policy.policy_source import (  # noqa: E402
    active_policy,
    default_policy,
    load_meta,
    load_policy,
    pd_settings,
    policy_store,
    resolve_policy,
)

CONFIG = PKG / "config"

HOME = [0.0, -0.2, 3.1] * 4
META = {
    "schema_version": 2,
    "run_name": "synthetic",
    "checkpoint": "",
    "task": "joystick",
    "obs_size": 40,
    "action_size": 12,
    "obs_layout": ["joint_pos:12", "joint_vel:12", "last_act:12", "command:4"],
    "actuator_names": [
        f"{leg}_{joint}_joint"
        for leg in ("rear_left", "rear_right", "front_right", "front_left")
        for joint in ("first", "second", "third")
    ],
    "home_ctrl": HOME,
    "anchor_ctrl": [0.0, -0.2 + 1 / 30, 3.1 + 2 / 30] * 4,
    "action_scale": [0.25, 0.5, 0.5] * 4,
    "target_low": [-0.44, -3.14, 0.425] * 4,
    "target_high": [0.44, 3.14, 3.15] * 4,
    "command_low": [-0.8, -0.5, -1.0, 0.125],
    "command_high": [1.2, 0.5, 1.0, 0.125],
    "command_fill": [0.125],
    "action_filter": 0.0,
    "ctrl_dt": 0.02,
    "knee_singularity": 3.2,
    "pd": {"kp": 20.0, "kd": 1.0, "max_torque": 6.0},
}


def make_policy(tmp_path, bias12=None, meta_updates=None, clamp_knee=False):
    """Synthetic zero-kernel policy: action = tanh(bias) for any obs."""
    meta = dict(META)
    meta.update(meta_updates or {})
    obs_size = meta["obs_size"]
    bias = np.zeros(24, np.float32)
    if bias12 is not None:
        bias[:12] = bias12
    np.savez(
        tmp_path / "policy.npz",
        norm_mean=np.zeros(obs_size, np.float32),
        norm_std=np.ones(obs_size, np.float32),
        hidden_0_kernel=np.zeros((obs_size, 24), np.float32),
        hidden_0_bias=bias,
    )
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))
    return WojtekPolicy(tmp_path / "policy.npz", clamp_knee=clamp_knee)


@pytest.fixture
def policy(tmp_path):
    return make_policy(tmp_path, bias12=np.linspace(-0.4, 0.4, 12))


# -- contract interpretation --------------------------------------------------

def test_obs_assembly_matches_layout(policy):
    assert policy.uses_imu is False
    q = policy.home_ctrl + np.linspace(0.01, 0.12, 12)
    dq = np.linspace(-1.0, 1.0, 12)
    policy.step(np.zeros(3), [0, 0, -1.0], q, dq, [0.3, -0.1, 0.2])
    obs = policy.last_obs
    assert obs.shape == (META["obs_size"],)
    assert np.allclose(obs[0:12], q - policy.home_ctrl, atol=1e-6)
    assert np.allclose(obs[12:24], dq, atol=1e-6)
    assert np.allclose(obs[24:36], 0.0)  # last_act starts at zero
    # 3-D /cmd_vel command is completed from command_fill
    assert np.allclose(obs[36:40], [0.3, -0.1, 0.2, 0.125], atol=1e-6)


def test_full_width_command_passes_through(policy):
    policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12),
                [0.3, -0.1, 0.2, 0.125])
    assert np.allclose(policy.last_obs[36:40], [0.3, -0.1, 0.2, 0.125])


def test_ignores_imu_when_layout_omits_it(policy):
    q, dq, cmd = policy.home_ctrl, np.zeros(12), [0.4, 0.0, 0.1]
    t1 = policy.step(np.zeros(3), [0, 0, -1.0], q, dq, cmd)
    policy.reset()
    t2 = policy.step([9.0, -9.0, 9.0], [1.0, 0.0, 0.0], q, dq, cmd)
    assert np.allclose(t1, t2)


def test_imu_layout_observes_gyro_and_gravity(tmp_path):
    pol = make_policy(
        tmp_path,
        meta_updates={
            "obs_size": 46,
            "obs_layout": ["gyro:3", "gravity:3", "joint_pos:12",
                           "joint_vel:12", "last_act:12", "command:4"],
        },
    )
    assert pol.uses_imu is True
    gyro, grav = [0.1, -0.2, 0.3], [0.0, 0.1, -0.99]
    pol.step(gyro, grav, pol.home_ctrl, np.zeros(12), [0.3, 0.1, -0.2])
    assert np.allclose(pol.last_obs[0:3], gyro, atol=1e-6)
    assert np.allclose(pol.last_obs[3:6], grav, atol=1e-6)


LIVE_HEIGHT_META = {
    "command_low": [-0.8, -0.5, -1.0, 0.10],
    "command_high": [1.2, 0.5, 1.0, 0.16],
    "command_fill": [0.125],
    "ctrl_low": [-0.6, -3.4, 0.3] * 4,
    "ctrl_high": [0.6, 3.4, 3.6] * 4,
}


def test_height_command_moves_anchor_and_obs(tmp_path):
    # A live-height contract re-anchors the stance to command[3] (as the
    # training env does every step); the command shows up verbatim in obs.
    pol = make_policy(tmp_path, meta_updates=LIVE_HEIGHT_META)
    pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
             [0.1, 0.0, 0.0, 0.16])
    assert np.allclose(pol.last_obs[36:40], [0.1, 0.0, 0.0, 0.16], atol=1e-6)
    expect = height_anchor(pol.home_ctrl, 0.16, pol.ctrl_low, pol.ctrl_high)
    assert np.allclose(pol.anchor_ctrl, expect, atol=1e-6)
    # dropping back to a 3-D command falls back to the contract's fill
    # height in both the obs padding and the anchor
    pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
             [0.1, 0.0, 0.0])
    assert np.allclose(pol.last_obs[36:40], [0.1, 0.0, 0.0, 0.125], atol=1e-6)
    default = height_anchor(pol.home_ctrl, 0.125, pol.ctrl_low, pol.ctrl_high)
    assert np.allclose(pol.anchor_ctrl, default, atol=1e-6)


def test_pinned_height_keeps_resolved_anchor(policy):
    # The default fixture pins height (low == high): the resolved anchor is
    # the contract's word, never recomputed at runtime.
    before = policy.anchor_ctrl.copy()
    policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12),
                [0.1, 0.0, 0.0, 0.125])
    assert np.allclose(policy.anchor_ctrl, before)


def test_live_height_contract_requires_ctrlrange(tmp_path):
    # A live height range without ctrl_low/ctrl_high must refuse to load --
    # silently keeping a fixed anchor would mis-anchor the stance.
    updates = {k: v for k, v in LIVE_HEIGHT_META.items()
               if k not in ("ctrl_low", "ctrl_high")}
    with pytest.raises(ValueError, match="ctrl_low/ctrl_high"):
        make_policy(tmp_path, meta_updates=updates)


def test_closed_form_targets(tmp_path):
    b = np.linspace(-0.4, 0.4, 12).astype(np.float32)
    pol = make_policy(tmp_path, bias12=b)
    t = pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
                 [0.3, 0.1, -0.2])
    expect = np.clip(
        np.array(META["anchor_ctrl"]) + np.tanh(b) * np.array(META["action_scale"]),
        META["target_low"], META["target_high"],
    )
    assert np.allclose(t, expect, atol=1e-6)


def test_targets_respect_contract_bounds(tmp_path):
    # bias saturating every joint positive: knees (anchor 3.167 + 0.5) pin
    # to the 3.15 knee cap, the rest stay inside the box
    pol = make_policy(tmp_path, bias12=np.full(12, 9.0, np.float32))
    t = pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
                 [0.5, 0, 0])
    expect = np.clip(
        np.array(META["anchor_ctrl"]) + np.tanh(9.0) * np.array(META["action_scale"]),
        META["target_low"], META["target_high"],
    )
    assert np.allclose(t, expect, atol=1e-6)
    assert np.allclose(t[2::3], 3.15)  # knee target cap binds
    assert np.all(np.abs(t[0::3]) <= 0.44 + 1e-6)  # abduction clamp


def test_action_filter_is_ema(tmp_path):
    b = np.full(12, 0.5, np.float32)
    pol = make_policy(tmp_path, bias12=b, meta_updates={"action_filter": 0.8})
    a = np.tanh(b)
    args = (np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12), [0.2, 0, 0])
    t1 = pol.step(*args)
    t2 = pol.step(*args)
    scale = np.array(META["action_scale"])
    anchor = np.array(META["anchor_ctrl"])
    lo, hi = META["target_low"], META["target_high"]
    assert np.allclose(t1, np.clip(anchor + 0.2 * a * scale, lo, hi), atol=1e-6)
    assert np.allclose(
        t2, np.clip(anchor + (0.8 * 0.2 + 0.2) * a * scale, lo, hi), atol=1e-6
    )
    pol.reset()
    assert np.allclose(pol.filtered_action, 0.0)


def test_clamp_knee_safety_clip(tmp_path):
    pol = make_policy(
        tmp_path, bias12=np.full(12, 9.0, np.float32), clamp_knee=True,
        meta_updates={"target_high": [0.44, 3.14, 5.8] * 4},
    )
    t = pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
                 [0.5, 0, 0])
    assert np.all(t[2::3] <= pol.knee_singularity + 1e-9)


def test_determinism(policy):
    seq_a = []
    policy.reset()
    for _ in range(10):
        seq_a.append(
            policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                        np.zeros(12), [0.2, 0, 0])
        )
    policy.reset()
    for i in range(10):
        t = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                        np.zeros(12), [0.2, 0, 0])
        assert np.allclose(t, seq_a[i])


def test_last_action_feeds_back(tmp_path):
    # identity-ish kernel so the action depends on last_act: use a nonzero
    # kernel row from last_act slice into the outputs
    obs_size = META["obs_size"]
    kernel = np.zeros((obs_size, 24), np.float32)
    kernel[24:36, :12] = np.eye(12, dtype=np.float32) * 0.5
    np.savez(
        tmp_path / "policy.npz",
        norm_mean=np.zeros(obs_size, np.float32),
        norm_std=np.ones(obs_size, np.float32),
        hidden_0_kernel=kernel,
        hidden_0_bias=np.full(24, 0.3, np.float32),
    )
    (tmp_path / "policy_meta.json").write_text(json.dumps(META))
    pol = WojtekPolicy(tmp_path / "policy.npz")
    args = (np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12), [0.3, 0, 0])
    t1 = pol.step(*args)
    assert not np.allclose(pol.last_action, 0.0)
    t2 = pol.step(*args)
    assert not np.allclose(t1, t2)  # last_act changed -> different action
    pol.reset()
    assert np.allclose(pol.last_action, 0.0)


# -- contract enforcement -----------------------------------------------------

def test_rejects_wrong_schema_version(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        make_policy(tmp_path, meta_updates={"schema_version": None})
    with pytest.raises(ValueError, match="schema_version"):
        make_policy(tmp_path, meta_updates={"schema_version": 1})


def test_rejects_unknown_obs_component(tmp_path):
    with pytest.raises(ValueError, match="phase"):
        make_policy(
            tmp_path,
            meta_updates={
                "obs_size": 48,
                "obs_layout": ["joint_pos:12", "joint_vel:12", "last_act:12",
                               "command:4", "phase:8"],
            },
        )


def test_rejects_inconsistent_command_fill(tmp_path):
    with pytest.raises(ValueError, match="command_fill"):
        make_policy(tmp_path, meta_updates={"command_fill": []})


# -- resolver -----------------------------------------------------------------

def test_resolve_local_dir(tmp_path):
    make_policy(tmp_path)
    r = resolve_policy(str(tmp_path))
    assert r.npz == tmp_path / "policy.npz"
    assert r.meta == tmp_path / "policy_meta.json"
    assert r.source == f"local:{tmp_path}"
    assert WojtekPolicy(r.npz, meta_path=r.meta).joint_names[0] == (
        "rear_left_first_joint"
    )


def test_resolve_local_dir_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="policy.npz"):
        resolve_policy(str(tmp_path))


def test_load_meta_and_pd_settings(tmp_path):
    make_policy(tmp_path, meta_updates={"pd": {"kp": 80.0, "kd": 2.26,
                                               "max_torque": 9.0}})
    meta, source = load_meta(str(tmp_path))
    assert source == f"local:{tmp_path}"
    # servo settings come from the contract verbatim
    assert pd_settings(meta) == {"kp": 80.0, "kd": 2.26, "max_torque": 9.0}


def test_load_policy_resolves_once_and_applies_overrides(tmp_path):
    make_policy(tmp_path, meta_updates={"pd": {"kp": 80.0, "kd": 2.26,
                                               "max_torque": 9.0}})
    loaded = load_policy(str(tmp_path))
    assert loaded.source == f"local:{tmp_path}"
    assert loaded.run_name == "synthetic"
    # the resolved dir (what a node gets as `policy`) holds both files
    assert loaded.directory == tmp_path
    assert loaded.npz == tmp_path / "policy.npz"
    assert loaded.meta_path == tmp_path / "policy_meta.json"
    assert loaded.meta["pd"]["kp"] == 80.0
    # contract verbatim when no overrides
    assert loaded.pd == {"kp": 80.0, "kd": 2.26, "max_torque": 9.0}
    # non-empty overrides replace individual entries; empty string / None
    # keep the contract value
    over = load_policy(str(tmp_path),
                       overrides={"max_torque": "2", "kp": "", "kd": None})
    assert over.pd == {"kp": 80.0, "kd": 2.26, "max_torque": 2.0}


def test_resolve_rejects_non_reference():
    with pytest.raises(ValueError):
        resolve_policy("")
    with pytest.raises(ValueError):
        resolve_policy("no-slash-and-no-such-dir")
    with pytest.raises(ValueError):
        resolve_policy("/nonexistent/dir/policy.npz")


# -- policy store -------------------------------------------------------------
#
# The robot resolves Hugging Face references offline, from the store that
# deploy.sh rsyncs to it. These tests stand in for that machine, with a
# store on disk and no way to download.

SHA = "a" * 40


def store_snapshot(store, repo_id, commit):
    """A materialized snapshot, the way a prefetch on the PC leaves it."""
    snapshot = store / repo_id / commit
    snapshot.mkdir(parents=True)
    make_policy(snapshot)
    return snapshot


def offline(*args, **kwargs):
    raise policy_source._HFUnavailable("offline")


def test_pinned_commit_resolves_from_store_without_fetching(tmp_path, monkeypatch):
    # A commit never moves, so a stored snapshot answers with no network.
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path))
    snapshot = store_snapshot(tmp_path, "org/name", SHA)

    def no_network(*args, **kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(policy_source, "_fetch_into_store", no_network)
    r = resolve_policy(f"org/name@{SHA}")
    assert r.npz == snapshot / "policy.npz"
    assert r.meta == snapshot / "policy_meta.json"
    assert r.source == f"hf:org/name@{SHA}"
    assert load_policy(f"org/name@{SHA}").directory == snapshot


def test_branch_offline_falls_back_to_recorded_ref(tmp_path, monkeypatch):
    # A branch is fetched when possible. Offline, the refs file gives the
    # commit it pointed at last time, and that snapshot answers.
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path))
    store_snapshot(tmp_path, "org/name", SHA)
    refs = tmp_path / "org" / "name" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(SHA + "\n")
    (refs / "somebranch").write_text(SHA + "\n")
    monkeypatch.setattr(policy_source, "_fetch_into_store", offline)
    assert resolve_policy("org/name").source == f"hf:org/name@{SHA}"
    assert resolve_policy("org/name@somebranch").source == f"hf:org/name@{SHA}"


def test_missing_from_store_offline_says_how_to_prefetch(tmp_path, monkeypatch):
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path))
    store_snapshot(tmp_path, "org/name", SHA)
    monkeypatch.setattr(policy_source, "_fetch_into_store", offline)
    with pytest.raises(RuntimeError, match="prefetch"):
        resolve_policy("org/name@" + "b" * 40)   # a commit nobody fetched
    with pytest.raises(RuntimeError, match="prefetch"):
        resolve_policy("org/name@unrecorded")    # a branch with no refs file


class _NeverRaised(Exception):
    """Stands in for LocalEntryNotFoundError when the stub always succeeds."""


def test_fetch_materializes_real_files_and_records_the_branch(
        tmp_path, monkeypatch):
    # This is a fetch on the PC. The download cache hands back symlinks into
    # its blob directory. The store must end up with real files, because
    # they get rsynced to the robot, and with a refs entry recording where
    # the branch pointed.
    commit = "c" * 40
    cache = tmp_path / "hf-cache"
    (cache / "blobs").mkdir(parents=True)
    (cache / "snapshots" / commit).mkdir(parents=True)

    def fake_download(repo_id, fname, revision=None):
        blob = cache / "blobs" / f"blob-{fname}"
        blob.write_text(f"payload of {fname}")
        link = cache / "snapshots" / commit / fname
        if not link.is_symlink():
            link.symlink_to(blob)
        return str(link)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub.errors",
        types.SimpleNamespace(LocalEntryNotFoundError=_NeverRaised))

    store = tmp_path / "store"
    r = policy_source._fetch_into_store(store, "org/name", "somebranch")
    snapshot = store / "org" / "name" / commit
    assert r.npz == snapshot / "policy.npz"
    assert r.source == f"hf:org/name@{commit}"
    for fname in ("policy.npz", "policy_meta.json"):
        assert (snapshot / fname).is_file()
        assert not (snapshot / fname).is_symlink()
        assert (snapshot / fname).read_text() == f"payload of {fname}"
    refs = store / "org" / "name" / "refs"
    assert (refs / "somebranch").read_text().strip() == commit

    # A pinned commit is its own name, so nothing is recorded.
    policy_source._fetch_into_store(store, "org/name", commit)
    assert not (refs / commit).exists()


def test_policy_store_default_is_beside_the_workspace_src(monkeypatch):
    # No env var: the store sits next to the src/ this file was found in --
    # ros/policies in a checkout, wojtek_ws/policies on the robot.
    monkeypatch.delenv("WOJTEK_POLICY_STORE", raising=False)
    assert policy_store() == PKG.parents[1] / "policies"


# -- the default policy -------------------------------------------------------

def test_default_policy_needs_an_organization(monkeypatch):
    monkeypatch.delenv("HF_ORGANIZATION", raising=False)
    assert default_policy() == ""
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    assert default_policy() == "org/" + policy_source._DEFAULT_REPO


def test_default_policy_is_pinned_to_a_commit():
    # The robot is offline and can only answer a commit that is already in
    # its store, so the shipped pin must be a commit.
    _, _, revision = policy_source._DEFAULT_REPO.partition("@")
    assert policy_source._is_commit(revision)


def test_default_cli_resolves_from_the_store(tmp_path, monkeypatch):
    # What deploy.sh runs, on a machine that cannot download. The pinned
    # default is already in the store, so the resolve succeeds.
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path))
    repo, _, commit = default_policy().partition("@")
    store_snapshot(tmp_path, repo, commit)

    def no_network(*args, **kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(policy_source, "_fetch_into_store", no_network)
    assert policy_source.main(["--default"]) == 0


def test_default_cli_without_an_organization_fails(monkeypatch):
    monkeypatch.delenv("HF_ORGANIZATION", raising=False)
    assert policy_source.main(["--default"]) == 2


def test_active_policy_prefers_the_override_file(tmp_path, monkeypatch):
    # This is a robot that deploy.sh --policy has visited. The override file
    # sits beside the store and it wins over the pin.
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path / "policies"))
    assert policy_source.policy_override_file() == tmp_path / "policy_override"
    (tmp_path / "policy_override").write_text(f"org/other@{SHA}\n")
    assert active_policy() == f"org/other@{SHA}"


def test_active_policy_without_an_override_is_the_pin(tmp_path, monkeypatch):
    # Every machine that never got --policy, the operator PC included.
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path / "policies"))
    assert active_policy() == default_policy()


def test_active_policy_ignores_an_empty_override(tmp_path, monkeypatch):
    # A blank file names no policy, so the pin still runs.
    monkeypatch.setenv("HF_ORGANIZATION", "org")
    monkeypatch.setenv("WOJTEK_POLICY_STORE", str(tmp_path / "policies"))
    for text in ("", "\n", "   \n"):
        (tmp_path / "policy_override").write_text(text)
        assert active_policy() == default_policy()


# -- shared helpers -----------------------------------------------------------

def test_gravity_from_quat():
    # identity: upright -> gravity along -z in body frame
    assert np.allclose(gravity_from_quat(1, 0, 0, 0), [0, 0, -1])
    # 90 deg pitch about +y: gravity ends up along a body x axis
    g = gravity_from_quat(np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0)
    assert np.allclose(g, [1, 0, 0], atol=1e-9) or np.allclose(g, [-1, 0, 0], atol=1e-9)
    assert np.isclose(np.linalg.norm(g), 1.0)


def test_joint_map_roundtrip():
    jm = JointMap(CONFIG / "joint_map.yaml")
    names = jm.names()
    q = np.random.default_rng(0).uniform(-2, 2, len(names))
    back = jm.to_mjc(names, jm.to_urdf(names, q))
    assert np.allclose(back, q)
    dq = np.random.default_rng(1).uniform(-5, 5, len(names))
    assert np.allclose(jm.vel_to_mjc(names, jm.vel_to_urdf(names, dq)), dq)


def test_joint_map_home_within_urdf_limits_for_second_joint():
    """The trained home pose maps to finite URDF angles (sanity of offsets)."""
    jm = JointMap(CONFIG / "joint_map.yaml")
    urdf_home = jm.to_urdf(META["actuator_names"], np.array(HOME))
    assert np.all(np.isfinite(urdf_home))
