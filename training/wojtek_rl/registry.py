"""Task registry: name -> (env class, default config), plus override merge.

Env configs are ml_collections ConfigDicts built in code; Hydra yaml carries
plain dicts/lists. _apply_overrides merges the latter onto the former with
tuple/float coercion so yaml `vx: [-0.8, 1.8]` lands on a tuple field.
"""

from ml_collections import config_dict

from wojtek_rl import env as env_joystick
from wojtek_rl import env_getup, env_jump

TASKS = {
    "joystick": (env_joystick.WojtekJoystick, env_joystick.default_config),
    "getup": (env_getup.WojtekGetup, env_getup.default_config),
    "jump": (env_jump.WojtekJump, env_jump.default_config),
}


def _apply_overrides(cfg: config_dict.ConfigDict, overrides: dict) -> None:
    for key, value in (overrides or {}).items():
        current = getattr(cfg, key)
        if isinstance(current, config_dict.ConfigDict):
            _apply_overrides(current, value)
            continue
        if isinstance(current, tuple) and isinstance(value, (list, tuple)):
            value = tuple(value)
        if isinstance(current, float) and isinstance(value, int):
            value = float(value)
        # scalar defaults may be overridden with per-element vectors
        # (e.g. action_scale: [0.2, 0.5, 0.5]); bypass the type lock
        if isinstance(current, float) and isinstance(value, (list, tuple)):
            with cfg.ignore_type():
                setattr(cfg, key, tuple(float(v) for v in value))
            continue
        setattr(cfg, key, value)


def make_env(task: str, env_overrides: dict | None = None):
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}', have {sorted(TASKS)}")
    cls, default_config = TASKS[task]
    cfg = default_config()
    _apply_overrides(cfg, env_overrides or {})
    return cls(cfg)
