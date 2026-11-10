"""Turn an APT-RL deploy pack into a policy dir the robot stack loads.

    python3 -m wojtek_policy.apt_contract PACK.npz PACK.json OUT_DIR [--wz-max 0.5]

PACK.npz/PACK.json are what apt_np.py export writes (e.g. A16_deploy.*).
OUT_DIR gets policy.npz (the pack, unchanged) and a "kind": "apt"
policy_meta.json: the servo block from the pack's reported gains, the
feed-forward clip (reported N*m), and the command box the pad and
policy_node clip to. --wz-max narrows the yaw-rate box (the weights still
saw +-1.0 in training; a narrower box only limits what is commanded).
"""

import argparse
import json
import shutil
from pathlib import Path

from wojtek_policy.poses import ACTUATOR_NAMES

# The APT training command box (vx, vy, wz).
APT_COMMAND_LOW = [-0.6, -0.4, -1.0]
APT_COMMAND_HIGH = [1.2, 0.4, 1.0]


def build_meta(pack: dict, wz_max=None, tau_ff_limit=5.0) -> dict:
    lo, hi = list(APT_COMMAND_LOW), list(APT_COMMAND_HIGH)
    if wz_max is not None:
        lo[2], hi[2] = -abs(wz_max), abs(wz_max)
    kp = list(pack["kp_rep"])
    return {
        "schema_version": 3,
        "kind": "apt",
        "run_name": f"{pack['name']}_apt",
        "task": "joystick",
        "robot": "legs_v627",
        "actuator_names": list(ACTUATOR_NAMES),
        "ctrl_dt": pack["ctrl_dt"],
        "command_low": lo,
        "command_high": hi,
        "command_fill": [],
        "pd": {"kp": kp[1], "kp_per_actuator": kp * 4, "kd": pack["kd_rep"],
               "max_torque": pack["cap_rep"]},
        # Reported N*m (true / S): the decoder's own clip on the MD80 side.
        "tau_ff": {"enable": True, "scale": tau_ff_limit, "limit": tau_ff_limit,
                   "units": "reported N*m (true/S)", "S": pack["S"]},
        "knee_singularity": None,
        "apt": pack,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack_npz"); ap.add_argument("pack_json"); ap.add_argument("out_dir")
    ap.add_argument("--wz-max", type=float, default=None)
    ap.add_argument("--tau-ff-limit", type=float, default=5.0)
    a = ap.parse_args(argv)
    pack = json.loads(Path(a.pack_json).read_text())
    if pack.get("kind") != "apt":
        raise SystemExit(f"{a.pack_json} is not an apt deploy pack")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.pack_npz, out / "policy.npz")
    meta = build_meta(pack, a.wz_max, a.tau_ff_limit)
    (out / "policy_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {meta['run_name']}, pd {meta['pd']}, "
          f"wz box +-{meta['command_high'][2]:g} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
