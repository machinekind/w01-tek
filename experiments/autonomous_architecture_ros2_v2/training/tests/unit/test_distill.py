"""Distillation helpers, model-free (wojtek_rl.distill only)."""

import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import terrain
from wojtek_rl.distill import anneal, select_rows, teacher_table

TEACHERS = ["keeper", "flat", "soft"]
ROUTING = {
    "flat_row": "flat",
    "types": {
        "rough_uniform": "keeper",
        "pyramid_stairs": "keeper",
        "inverted_pyramid_stairs": "keeper",
        "random_grid": "keeper",
        "pyramid_slope": "soft",
        "inverted_pyramid_slope": "soft",
        "discrete_obstacles": "soft",
        "wave": "soft",
    },
}


def test_teacher_table_full_coverage():
    flat_idx, type_idx = teacher_table(ROUTING, TEACHERS, terrain.TYPES)
    assert flat_idx == 1
    assert len(type_idx) == len(terrain.TYPES)
    by_type = dict(zip(terrain.TYPES, type_idx))
    assert by_type["pyramid_stairs"] == 0
    assert by_type["wave"] == 2


def test_teacher_table_rejects_holes():
    incomplete = {"flat_row": "flat", "types": {"wave": "soft"}}
    with pytest.raises(ValueError, match="missing"):
        teacher_table(incomplete, TEACHERS, terrain.TYPES)
    unknown = {
        "flat_row": "flat",
        "types": {**ROUTING["types"], "lava": "keeper"},
    }
    with pytest.raises(ValueError, match="unknown"):
        teacher_table(unknown, TEACHERS, terrain.TYPES)
    bad_name = {"flat_row": "nobody", "types": ROUTING["types"]}
    with pytest.raises(ValueError, match="flat_row"):
        teacher_table(bad_name, TEACHERS, terrain.TYPES)


def test_select_rows_picks_per_env():
    stacked = jp.stack(
        [jp.full((5, 3), float(k)) for k in range(3)]
    )  # (K=3, N=5, A=3)
    idx = jp.array([0, 2, 1, 1, 0])
    out = np.asarray(select_rows(stacked, idx))
    np.testing.assert_allclose(out[:, 0], [0.0, 2.0, 1.0, 1.0, 0.0])
    assert out.shape == (5, 3)


def test_anneal_endpoints_and_clip():
    assert anneal(0.5, 0.0, 0, 100) == 0.5
    assert anneal(0.5, 0.0, 99, 100) == 0.0
    assert anneal(0.5, 0.0, 200, 100) == 0.0
    assert anneal(0.5, 0.0, 0, 1) == 0.0
    mid = anneal(0.5, 0.0, 50, 101)
    assert abs(mid - 0.25) < 1e-9


def test_command_router():
    from wojtek_rl.distill import command_router

    win, mov = command_router(
        {"mode": "command", "command_window": "flat", "moving": "keeper"},
        TEACHERS,
    )
    assert (win, mov) == (1, 0)
    with pytest.raises(ValueError, match="needs routing"):
        command_router({"mode": "command", "moving": "keeper"}, TEACHERS)
    with pytest.raises(ValueError, match="unknown teachers"):
        command_router(
            {"command_window": "nobody", "moving": "keeper"}, TEACHERS
        )
