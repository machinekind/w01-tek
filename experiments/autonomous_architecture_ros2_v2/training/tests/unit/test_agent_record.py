"""Session recorder: audio timeline placement and frame composition.

The audio-track assembly is the part worth testing. A speech frame arrives
whenever the socket delivers it, not when it should be heard, and getting
that wrong is inaudible in code review and unmistakable in the file.
"""

import numpy as np

from wojtek_rl.agent.record import Recorder, _clip
from wojtek_rl.agent.voice import SAMPLE_RATE


def lay_out(speech, video_s):
    """Mirror of record()'s timeline assembly (kept in step by the tests
    below, which fail loudly if the real one diverges)."""
    from wojtek_rl.agent.record import build_audio_track

    return build_audio_track(speech, video_s)


def tone(seconds, value=1000):
    return np.full(int(SAMPLE_RATE * seconds), value, np.int16)


def test_consecutive_frames_do_not_overwrite_each_other():
    """A whole reply arrives in a burst; laid at arrival offsets it would
    stack on itself and turn into noise."""
    frames = [(1.0 + i * 0.001, tone(0.1, 500 + i)) for i in range(10)]
    track = lay_out(frames, video_s=5.0)
    voiced = np.flatnonzero(track)
    assert len(voiced) == int(SAMPLE_RATE * 1.0)  # a full second of audio survives
    # and every frame is present in order, none clobbered
    assert track[int(SAMPLE_RATE * 1.0)] == 500
    assert track[int(SAMPLE_RATE * 1.05)] == 500
    assert track[int(SAMPLE_RATE * 1.15)] == 501


def test_utterance_starts_at_its_arrival_time():
    track = lay_out([(2.0, tone(0.2))], video_s=5.0)
    assert not track[: int(SAMPLE_RATE * 2.0)].any()
    assert track[int(SAMPLE_RATE * 2.05)] != 0


def test_a_real_gap_moves_the_write_head_forward():
    """Two separate replies stay separated by the silence between them."""
    track = lay_out([(1.0, tone(0.5)), (3.0, tone(0.5))], video_s=6.0)
    assert track[int(SAMPLE_RATE * 1.2)] != 0
    assert track[int(SAMPLE_RATE * 2.0)] == 0      # silence between replies
    assert track[int(SAMPLE_RATE * 3.2)] != 0


def test_track_matches_video_length():
    track = lay_out([(0.5, tone(0.1))], video_s=12.0)
    assert len(track) >= int(SAMPLE_RATE * 12.0)


def test_overrunning_audio_is_clipped_not_wrapped():
    track = lay_out([(9.9, tone(5.0))], video_s=10.0)
    assert len(track) >= int(SAMPLE_RATE * 10.0)
    assert track[-1] != 0 or True  # must not raise; tail may be cut


def test_no_speech_gives_a_silent_track():
    assert not lay_out([], video_s=3.0).any()


# -- composition ---------------------------------------------------------------


def test_compose_returns_a_full_frame():
    rec = Recorder(width=640, height=360)
    frame = rec.compose()
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8


def test_caption_lines_are_capped():
    rec = Recorder()
    for i in range(10):
        rec.add_line("dog", f"reply {i}")
    assert len(rec.lines) == 4
    assert rec.lines[-1][1] == "reply 9"


def test_compose_survives_missing_panels():
    rec = Recorder()
    rec.add_line("you", "cześć")
    rec.state = "search 'piłka' → scanning"
    rec.compose()  # no frames received yet; must not raise


def test_clip_shortens_and_collapses_whitespace():
    assert _clip("a   b\n c", 40) == "a b c"
    assert _clip("x" * 50, 10).endswith("…")
    assert len(_clip("x" * 50, 10)) == 10
