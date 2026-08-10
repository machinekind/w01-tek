#!/usr/bin/env python3

# Copyright 2026 Jakub Delicat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flash one motor config onto MD80 drives, then read every drive back.

The motor config is what the drive uses to convert commanded torque to
phase current (torque constant, gear ratio) and to bound it (current
limits). Flashing a config for a different motor than the one physically
attached makes every impedance gain, torque command, and torque limit
execute scaled by Kt_real/Kt_configured -- silently, per drive. The
config file is therefore a required argument here, there is no default,
and a flash always ends with a readback of every drive so the operator
sees one consolidated view of what the fleet actually runs.

Usage:
  ./flash_motors.py --config path/to/motor.cfg          # flash all 12 + verify
  ./flash_motors.py --config motor.cfg --ids 22         # one drive + verify
  ./flash_motors.py --verify-only                       # readback only

The readback (`mdtool setup info`) prints each drive's configured motor
name, torque constant and gear ratio; eyeball them for uniformity before
enabling the drives. This script only shells out to mdtool -- it needs a
configured CANdle bus and the mdtool config pointing at it.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

# All twelve joints, paths.LEGS order: FL 1x, FR 2x, RR 3x, RL 4x.
ALL_DRIVE_IDS = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
# mdtool resolves `setup motor` config paths relative to this directory.
MDTOOL_MOTOR_DIR = pathlib.Path.home() / ".config/mdtool/mdtool_motors"
ROBOT_SUBDIR = "quadruped_robot"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"command failed with exit code {result.returncode}: {cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="motor .cfg to flash; required unless --verify-only",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=ALL_DRIVE_IDS,
        help=f"drive IDs to flash (default: all {len(ALL_DRIVE_IDS)})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip flashing, only read back every drive's config",
    )
    args = parser.parse_args()

    if not args.verify_only:
        if args.config is None:
            parser.error("--config is required (there is no default motor)")
        if not args.config.is_file():
            sys.exit(f"config file not found: {args.config}")
        dest_dir = MDTOOL_MOTOR_DIR / ROBOT_SUBDIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, dest_dir / args.config.name)
        for drive_id in args.ids:
            run(
                [
                    "mdtool",
                    "setup",
                    "motor",
                    str(drive_id),
                    f"{ROBOT_SUBDIR}/{args.config.name}",
                ]
            )

    # Always read back the WHOLE fleet, not just the flashed subset: a
    # partial flash leaving two motor models on the bus is exactly the
    # failure mode this readback exists to surface.
    for drive_id in ALL_DRIVE_IDS:
        run(["mdtool", "setup", "info", str(drive_id)])


if __name__ == "__main__":
    main()
