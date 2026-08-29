"""lay_track: both voices on one wall-clock audio track, model-free."""

import numpy as np

from wojtek_rl.agent.scenario import lay_track
from wojtek_rl.agent.voice import SAMPLE_RATE


def tone(n, value=1000):
    return np.full(n, value, np.int16)


def test_question_pastes_at_send_time():
    q = tone(SAMPLE_RATE)  # 1 s
    audio = lay_track([("q", 2.0, q)], total_s=4.0)
    assert audio[int(1.9 * SAMPLE_RATE)] == 0
    assert audio[int(2.5 * SAMPLE_RATE)] == 1000
    assert audio[int(3.5 * SAMPLE_RATE)] == 0


def test_reply_frames_pack_back_to_back():
    # Two frames arriving nearly at once must not overwrite each other.
    f = tone(SAMPLE_RATE // 2, 500)  # 0.5 s each
    audio = lay_track([("a", 1.0, f), ("a", 1.01, f)], total_s=3.0)
    assert audio[int(1.25 * SAMPLE_RATE)] == 500
    assert audio[int(1.75 * SAMPLE_RATE)] == 500  # second frame after first
    assert audio[int(2.2 * SAMPLE_RATE)] == 0


def test_question_and_reply_coexist():
    q = tone(SAMPLE_RATE // 2, 300)
    a = tone(SAMPLE_RATE // 2, 700)
    audio = lay_track([("q", 0.5, q), ("a", 2.0, a)], total_s=3.0)
    assert audio[int(0.7 * SAMPLE_RATE)] == 300
    assert audio[int(2.2 * SAMPLE_RATE)] == 700


def test_overrun_is_clipped_not_fatal():
    a = tone(2 * SAMPLE_RATE, 900)
    audio = lay_track([("a", 1.5, a)], total_s=2.0)  # runs past total
    assert audio[-1] in (0, 900)  # no exception; tail within buffer
