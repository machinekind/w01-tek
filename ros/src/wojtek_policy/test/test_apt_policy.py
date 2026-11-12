"""Unit tests for the APT-RL ("kind": "apt") runtime adapter and its numpy kernels.

Model-free: the real APT export is a ~2 MB npz kept out of the repo, so the
adapter is tested against a stub controller with known outputs, and the
deployment speed-ups in apt_np are checked against their reference forms.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy import apt_np, apt_policy  # noqa: E402
from wojtek_policy.poses import ACTUATOR_NAMES  # noqa: E402

S = 2.09


class StubCtl:
    """Stands in for AptNumpy: returns a fixed (qc, tau_ff_true) per step."""

    def __init__(self, path):
        self.m = {"ctrl_dt": 0.02}
        self.S = S
        self.t = 0
        self.calls = []
        self.qc = np.full(12, 0.1)
        self.ff = np.linspace(-20.0, 20.0, 12)  # true N*m, beyond any clip

    def reset(self):
        self.t = 0

    def step(self, gyro, grav, acc, q, dq, tau_rep, cmd):
        self.calls.append(dict(acc=acc, tau_rep=np.array(tau_rep), cmd=np.array(cmd)))
        self.t += 1
        return self.qc.copy(), self.ff.copy()


def make(tmp_path, monkeypatch, **tff):
    monkeypatch.setattr(apt_policy, "AptNumpy", StubCtl)
    meta = {
        "schema_version": 3, "kind": "apt", "run_name": "stub", "robot": "legs_v627",
        "actuator_names": list(ACTUATOR_NAMES),
        "command_low": [-0.6, -0.4, -0.5], "command_high": [1.2, 0.4, 0.5],
        "pd": {"kp": 45.0, "kp_per_actuator": [60.0, 45.0, 45.0] * 4, "kd": 1.5, "max_torque": 9.83},
        "tau_ff": {"enable": True, "scale": 5.0, **tff},
    }
    (tmp_path / "policy.npz").write_bytes(b"")
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))
    return tmp_path


def step(p, **kw):
    return p.step(np.zeros(3), [0, 0, -1.0], np.zeros(12), np.zeros(12), [0.3, 0, 0],
                  acc=np.array([0, 0, 9.81]), tau_rep=np.ones(12), **kw)


def test_is_apt_meta(tmp_path, monkeypatch):
    d = make(tmp_path, monkeypatch)
    assert apt_policy.is_apt_meta(d / "policy_meta.json")
    (tmp_path / "other.json").write_text(json.dumps({"schema_version": 2}))
    assert not apt_policy.is_apt_meta(tmp_path / "other.json")
    assert not apt_policy.is_apt_meta(tmp_path / "missing.json")


def test_tau_ff_is_reported_units_and_clipped(tmp_path, monkeypatch):
    p = apt_policy.AptPolicy(make(tmp_path, monkeypatch) / "policy.npz")
    step(p)
    assert p.tau_ff_limit == 5.0
    expect = np.clip(p.ctl.ff / S, -5.0, 5.0)
    assert np.allclose(p.last_tau_ff, expect, atol=1e-6)


def test_tau_ff_limit_override_beats_contract(tmp_path, monkeypatch):
    d = make(tmp_path, monkeypatch, limit=4.0)
    assert apt_policy.AptPolicy(d / "policy.npz").tau_ff_limit == 4.0
    p = apt_policy.AptPolicy(d / "policy.npz", tau_ff_limit=7.5)
    step(p)
    assert p.tau_ff_limit == 7.5 and np.max(np.abs(p.last_tau_ff)) == pytest.approx(7.5)


def test_first_step_sees_zero_tau_rep_then_the_measured_one(tmp_path, monkeypatch):
    p = apt_policy.AptPolicy(make(tmp_path, monkeypatch) / "policy.npz")
    step(p)
    step(p)
    assert np.all(p.ctl.calls[0]["tau_rep"] == 0) and np.all(p.ctl.calls[1]["tau_rep"] == 1)


def test_missing_acc_is_refused(tmp_path, monkeypatch):
    p = apt_policy.AptPolicy(make(tmp_path, monkeypatch) / "policy.npz")
    with pytest.raises(ValueError):
        p.step(np.zeros(3), [0, 0, -1.0], np.zeros(12), np.zeros(12), [0, 0, 0])


def test_output_lpf_is_first_order_on_targets_and_torque(tmp_path, monkeypatch):
    p = apt_policy.AptPolicy(make(tmp_path, monkeypatch, limit=100.0) / "policy.npz")
    p.output_lpf_alpha = 0.5
    first = step(p)                      # first step initialises the filter
    assert np.allclose(first, 0.1)
    p.ctl.qc = np.full(12, 1.1)
    p.ctl.ff = p.ctl.ff * 0.0
    second = step(p)
    assert np.allclose(second, 0.6, atol=1e-6)
    assert np.allclose(p.last_tau_ff_true, 0.5 * np.linspace(-20.0, 20.0, 12))
    p.reset()                            # reset restarts the filter
    assert np.allclose(step(p), 1.1, atol=1e-6)


def test_vectorised_conv_matches_loop():
    rng = np.random.default_rng(0)
    h, W, b = rng.normal(size=(50, 32)), rng.normal(size=(4, 32, 32)), rng.normal(size=32)
    To = (50 - 4) // 2 + 1
    ref = np.stack([np.einsum("kc,kcd->d", h[2 * t:2 * t + 4], W) + b for t in range(To)])
    assert np.allclose(apt_np.conv1d_valid_stride2(h, W, b), ref, atol=1e-12)


def test_layernorm_and_fused_qkv_match_reference():
    rng = np.random.default_rng(1)
    D, NH = apt_np.D, apt_np.NH
    x = rng.normal(size=(5, D))
    g, b = rng.normal(size=D), rng.normal(size=D)
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    assert np.allclose(apt_np.ln(g, b, x), (x - m) / np.sqrt(v + 1e-5) * g + b, atol=1e-12)
    p = {k: rng.normal(size=(D, D)) * 0.1 for k in ("q.W", "k.W", "v.W", "o.W")}
    p.update({k: rng.normal(size=D) * 0.1 for k in ("q.b", "k.b", "v.b", "o.b", "n1.b", "n2.b", "f2.b")})
    p.update({"n1.g": np.ones(D), "n2.g": np.ones(D), "f1.W": rng.normal(size=(D, 4 * D)) * 0.1,
              "f1.b": np.zeros(4 * D), "f2.W": rng.normal(size=(4 * D, D)) * 0.1})
    p["qkv.W"] = np.concatenate([p["q.W"], p["k.W"], p["v.W"]], 1)
    p["qkv.b"] = np.concatenate([p["q.b"], p["k.b"], p["v.b"]], 0)
    L = x.shape[0]
    sp = lambda y: y.reshape(L, NH, D // NH).transpose(1, 0, 2)
    q, k, vv = sp(x @ p["q.W"] + p["q.b"]), sp(x @ p["k.W"] + p["k.b"]), sp(x @ p["v.W"] + p["v.b"])
    a = apt_np._softmax(q @ k.transpose(0, 2, 1) / math.sqrt(D // NH))
    h = (a @ vv).transpose(1, 0, 2).reshape(L, D)
    y = apt_np.ln(p["n1.g"], p["n1.b"], x + h @ p["o.W"] + p["o.b"])
    ref = apt_np.ln(p["n2.g"], p["n2.b"], y + apt_np._gelu(y @ p["f1.W"] + p["f1.b"]) @ p["f2.W"] + p["f2.b"])
    assert np.allclose(apt_np.tlayer(p, x), ref, atol=1e-10)


def test_apt_contract_builds_a_loadable_meta(tmp_path, monkeypatch):
    from wojtek_policy import apt_contract
    pack = {"name": "A16", "kind": "apt", "ctrl_dt": 0.02, "kp_rep": [60.0, 45.0, 45.0],
            "kd_rep": 1.5, "cap_rep": 9.83, "S": S}
    meta = apt_contract.build_meta(pack, wz_max=0.5)
    assert meta["command_high"][2] == 0.5 and meta["command_low"][2] == -0.5
    assert meta["pd"]["kp_per_actuator"][:3] == [60.0, 45.0, 45.0] and meta["pd"]["kp"] == 45.0
    monkeypatch.setattr(apt_policy, "AptNumpy", StubCtl)
    (tmp_path / "policy.npz").write_bytes(b"")
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))
    p = apt_policy.AptPolicy(tmp_path / "policy.npz")
    assert p.joint_names == list(ACTUATOR_NAMES) and p.tau_ff_limit == 5.0
