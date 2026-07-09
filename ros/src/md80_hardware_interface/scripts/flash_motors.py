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

import os

# List of motor IDs to set up
motor_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
config_path = "quadruped_robot/AK80-9.cfg"


os.system("mkdir -p /home/${USER}/.config/mdtool/mdtool_motors/quadruped_robot")
os.system("cp ../config/AK80-9.cfg /home/${USER}/.config/mdtool/mdtool_motors/quadruped_robot")

# Iterate over motor IDs and run the command for each one
for motor_id in motor_ids:
    command = f"mdtool setup motor {motor_id} {config_path}"
    os.system(command)  # Execute the command in the shell
    print(f"Executed: {command}")
