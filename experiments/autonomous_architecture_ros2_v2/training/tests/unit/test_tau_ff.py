"""Feed-forward torque head: the model-free pieces.

The env-side plumbing (24-wide actions, qfrc_applied under both delay
paths) is exercised in tests/integration/test_tau_ff_env.py; here we pin
the swing-cost math, the config defaults, and the preset that turns the
head on (so a yaml typo cannot silently become a zero-weight reward).
"""

import jax.numpy as jp
import numpy as np
import yaml

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths

PRESET_YAML = (
    paths.PROJECT_DIR
    / "wojtek_rl"
    / "conf"
    / "experiment"
    / "flat_tff_rnd_v1.yaml"
)
CONFIG_YAML = paths.PROJECT_DIR / "wojtek_rl" / "conf" / "config.yaml"


def test_swing_cost_zero_when_all_feet_down():
    tau = jp.ones(12) * 2.0
    contact = jp.ones(4, dtype=bool)
    assert float(wojtek_env.tau_ff_swing_cost(tau, contact)) == 0.0


def test_swing_cost_full_when_airborne():
    tau = jp.arange(12.0) - 6.0  # mixed signs: the cost is |tau|
    contact = jp.zeros(4, dtype=bool)
    expected = float(jp.sum(jp.abs(tau)))
    np.testing.assert_allclose(
        float(wojtek_env.tau_ff_swing_cost(tau, contact)), expected
    )


def test_swing_cost_gates_per_leg():
    # 1 Nm on every joint; legs 0 and 2 airborne -> 2 legs * 3 joints.
    tau = jp.ones(12)
    contact = jp.array([False, True, False, True])
    np.testing.assert_allclose(
        float(wojtek_env.tau_ff_swing_cost(tau, contact)), 6.0
    )
    # The cost must charge the AIRBORNE legs' torque, not the stance legs'.
    tau_stance_only = jp.zeros(12).at[3:6].set(5.0).at[9:12].set(5.0)
    np.testing.assert_allclose(
        float(wojtek_env.tau_ff_swing_cost(tau_stance_only, contact)), 0.0
    )


def test_target_sag_gates_on_stance():
    # 0.1 rad of sag on every joint; legs 1 and 3 in contact -> only their
    # 2*3 joints are charged.
    q = jp.zeros(12)
    targets = jp.full(12, 0.1)
    contact = jp.array([False, True, False, True])
    np.testing.assert_allclose(
        float(wojtek_env.target_sag_cost(q, targets, contact)),
        6 * 0.1**2,
        rtol=1e-6,
    )
    # Swing legs are free: sag there costs nothing.
    q_swing_only = jp.zeros(12).at[0:3].set(-0.3).at[6:9].set(0.3)
    np.testing.assert_allclose(
        float(
            wojtek_env.target_sag_cost(q_swing_only, jp.zeros(12), contact)
        ),
        0.0,
    )


def test_target_sag_zero_when_tracking():
    q = jp.arange(12.0) * 0.1
    contact = jp.ones(4, dtype=bool)
    assert float(wojtek_env.target_sag_cost(q, q, contact)) == 0.0


def test_sag_v2_preset_fades_yaw():
    preset = yaml.safe_load(
        (PRESET_YAML.parent / "flat_tff_sag_v2.yaml").read_text()
    )
    r = preset["task"]["env"]["reward"]
    assert r["target_sag_wz_fade"] > 0
    assert r["scales"]["target_sag"] < 0
    # The fade knob must exist in the code defaults (typo guard), off.
    assert wojtek_env.default_config().reward.target_sag_wz_fade == 0.0


def test_sag_v3_preset_fixes_foot_friction():
    """v3 = the proven v2b operating point + real friction DR. Without
    foot_friction the feet's fixed mu=0.9 max-combines over the floor draw
    and effective friction never goes below 0.9; the enabled switch gives
    the feet contact priority and the range must reach well under 0.9."""
    preset = yaml.safe_load(
        (PRESET_YAML.parent / "flat_tff_sag_v3.yaml").read_text()
    )
    ff = preset["dr"]["foot_friction"]
    assert ff["enable"] is True
    assert ff["range"][0] < 0.9 - 0.2  # genuinely slippery worlds exist
    r = preset["task"]["env"]["reward"]
    # The v2b-proven near-binary yaw gate rides along.
    assert 0 < r["target_sag_wz_fade"] <= 0.2
    assert r["scales"]["target_sag"] < 0


def test_sag_preset_layers_on_winner_config():
    preset = yaml.safe_load(
        (PRESET_YAML.parent / "flat_tff_sag_v1.yaml").read_text()
    )
    env_cfg = preset["task"]["env"]
    assert env_cfg["reward"]["scales"]["target_sag"] < 0
    assert env_cfg["command"]["pure_slow_prob"] > 0
    assert preset["defaults"] == ["flat_tff_rnd_v1"]
    # The scale key must exist in the code defaults (typo guard).
    assert "target_sag" in wojtek_env.default_config().reward.scales


def test_tau_ff_disabled_by_default():
    cfg = wojtek_env.default_config()
    assert not cfg.tau_ff.enable
    assert cfg.reward.scales.tau_ff_swing == 0.0
    assert cfg.reward.scales.tau_ff == 0.0


def test_preset_scale_keys_exist_in_code_defaults():
    """Every reward.scales key the preset sets must exist in the code's
    default scales — a typo there would otherwise train with the intended
    term silently absent (the sum iterates the code's keys, not yaml's)."""
    preset = yaml.safe_load(PRESET_YAML.read_text())
    default_scales = set(wojtek_env.default_config().reward.scales.keys())
    preset_scales = set(
        preset["task"]["env"]["reward"]["scales"].keys()
    )
    unknown = preset_scales - default_scales
    assert not unknown, f"preset sets unknown reward scales: {sorted(unknown)}"


def test_preset_turns_the_head_and_its_pricing_on():
    preset = yaml.safe_load(PRESET_YAML.read_text())
    env_cfg = preset["task"]["env"]
    assert env_cfg["tau_ff"]["enable"] is True
    assert env_cfg["tau_ff"]["scale"] > 0
    scales = env_cfg["reward"]["scales"]
    assert scales["tau_ff_swing"] < 0
    assert scales["tau_ff"] < 0
    # The critic list must not carry the phase clock.
    assert "phase" not in env_cfg["obs"]["privileged"]
    # RND on, with a coefficient that cannot dominate the extrinsic reward.
    assert preset["rnd"]["enable"] is True
    assert 0 < preset["rnd"]["coef"] <= 0.1


def test_rnd_yaml_block_carries_the_trainer_keys():
    """ppo_rnd.train reads these keys from cfg.rnd; a rename in one place
    must fail here, not at step one of a GPU run."""
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    required = {"enable", "coef", "learning_rate", "out_dim", "hidden", "obs_key"}
    assert required <= set(cfg["rnd"].keys())
    assert cfg["rnd"]["enable"] is False  # opt-in per preset
