from wojtek_rl.train import build_ppo_params


def test_defaults_come_from_go1_tuning():
    p = build_ppo_params([], smoke=False)
    assert p.num_timesteps >= 100_000_000
    assert p.network_factory.policy_obs_key == "state"
    assert p.network_factory.value_obs_key == "privileged_state"


def test_overrides_apply_with_type_coercion():
    p = build_ppo_params(["learning_rate=1e-4", "num_envs=512"], smoke=False)
    assert p.learning_rate == 1e-4
    assert p.num_envs == 512


def test_smoke_is_tiny():
    p = build_ppo_params([], smoke=True)
    assert p.num_timesteps <= 200_000
    assert p.num_envs <= 64
