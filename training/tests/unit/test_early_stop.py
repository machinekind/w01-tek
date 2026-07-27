"""plateau_stop: the early-stopping rule, model-free."""

from wojtek_rl.train import plateau_stop


def test_never_stops_before_min_evals():
    flat = [10.0] * 9
    assert not plateau_stop(flat, min_evals=10, patience=3, min_delta=0.5)
    assert plateau_stop(flat + [10.0], min_evals=10, patience=3, min_delta=0.5)


def test_growing_reward_never_stops():
    growing = [float(i) for i in range(30)]  # +1 per eval > min_delta
    assert not plateau_stop(growing, min_evals=10, patience=6, min_delta=0.5)


def test_plateau_after_growth_stops():
    rewards = [float(i) for i in range(10)] + [9.1] * 6
    assert plateau_stop(rewards, min_evals=10, patience=6, min_delta=0.5)


def test_noise_below_min_delta_does_not_reset_patience():
    # oscillating within +-0.4 of the best never counts as improvement
    rewards = [50.0] + [50.0 + 0.4 * (-1) ** i for i in range(12)]
    assert plateau_stop(rewards, min_evals=5, patience=6, min_delta=0.5)


def test_late_breakthrough_resets_patience():
    rewards = [50.0] * 12 + [55.0]  # new best on the last eval
    assert not plateau_stop(rewards, min_evals=5, patience=6, min_delta=0.5)
