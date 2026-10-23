# SO-101 arms with LeLab, the LeRobot web GUI

> A learning guide, not part of the Wojtek robot. Nothing here is deployed
> by `ros/deploy.sh`, and nothing in `ros/` or `training/` depends on this
> directory. It is a setup guide for a different piece of hardware.

[LeLab](https://github.com/huggingface/leLab) is Hugging Face's browser UI on
top of [LeRobot](https://huggingface.co/lerobot). It runs a local FastAPI
server that talks to the arms and serves a React app at
`http://localhost:8000`. The whole loop for a pair of
[SO-101](https://github.com/TheRobotStudio/SO-ARM100) arms happens in the
browser: calibrate, teleoperate, record, browse the dataset, train, run the
policy, upload to the Hub. No terminal prompts and no keyboard shortcuts.

One step is not in the GUI: assigning ids to brand-new motors. That is a
one-time CLI command and is covered in step 2.

## 1. Install and start LeLab

Prerequisites: [uv](https://docs.astral.sh/uv/) and a Python 3.12 or newer
that uv can find or install. LeLab pins its own LeRobot (v0.6.0 from git)
and PyTorch, so keep it out of `training/.venv` where JAX/MJX live. The
`uv tool install` below gives it an isolated environment automatically.

```bash
# uv, if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# the one-liner from the LeLab Space: installs LeLab + LeRobot, then starts it
uv tool install git+https://github.com/huggingface/leLab.git && lelab
```

The first install downloads PyTorch and takes a few minutes. `lelab` then
starts the server on port 8000 and opens `http://localhost:8000` in your
browser. Later runs are just `lelab`.

Useful flags:

| command | what it does |
|---|---|
| `lelab` | start the server and open the browser |
| `lelab --no-browser` | start without opening a browser tab |
| `lelab --stop` | free ports 8000 and 8080 if a previous run is stuck |
| `lelab --dev` | hot-reload mode for hacking on LeLab (needs Node.js 22) |
| `uv tool upgrade lelab` | update to the latest LeLab |

Linux only: make the serial ports readable once, then log out and in.

```bash
sudo usermod -aG dialout "$USER"
```

Or per boot, the LeRobot docs way:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

## 2. One-time: assign the motor ids (CLI)

Feetech STS3215 motors ship with id `1`. Each joint needs a unique id and all
motors one baudrate, written to the motor's EEPROM once. LeLab has no page
for this, so use LeRobot's script from the environment `uv tool install`
created.

Plug in the adapter's USB and power. Find its port with:

```bash
"$(uv tool dir)/lelab/bin/lerobot-find-port"
```

Unplug the USB when asked and it prints the port: `/dev/ttyACM0` style on
Linux, `/dev/tty.usbmodemXXXX` on macOS. Then, for the follower:

```bash
"$(uv tool dir)/lelab/bin/lerobot-setup-motors" \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0        # the port you just found
```

and for the leader:

```bash
"$(uv tool dir)/lelab/bin/lerobot-setup-motors" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
```

The script asks you to connect **only the named motor** to the adapter, one
at a time, gripper first, and press Enter. Ids end up as:

| joint | id |
|---|---|
| shoulder_pan | 1 |
| shoulder_lift | 2 |
| elbow_flex | 3 |
| wrist_flex | 4 |
| wrist_roll | 5 |
| gripper | 6 |

Check power, USB and the 3-pin cable before every Enter. On a Waveshare
adapter both jumpers go on channel `B`. When it finishes, daisy-chain the
motors and plug motor 1 into the adapter. Skip this section entirely if the
arms came pre-assembled with ids already set.

## 3. The loop in the browser

Open `http://localhost:8000`. The `?` button in the corner starts a guided
tour; the pages follow this order.

1. **Pick or name your arm.** Type a name to create a robot profile. Every
   later page hangs off the profile you select, and it is what LeRobot
   uses as the arm `id` for the calibration file, so keep using the same
   one for the same physical arm.
2. **Ports.** The port detector lists the serial devices and remembers the
   leader and follower ports for you. Plug one arm at a time if the two
   look alike.
3. **Calibrate.** Run it once for the leader, then the follower. Press
   Start, move every joint to the middle of its range, confirm, then sweep
   each joint through its full range. The page shows the range each motor
   has covered and turns it green once it clears the target. Calibration
   files land in `~/.cache/huggingface/lerobot/calibration/`.
4. **Cameras.** Add one or more cameras if the task needs vision. LeLab
   lists what the OS sees; pick each one and give it a name such as
   `front` or `wrist`. Cameras are saved with the robot profile and reused
   when recording. A good check: could *you* do the task watching only the
   camera images?
5. **Teleoperate.** Move the leader and the follower mirrors it, with live
   joint values and a 3D model of the arm. Unlocked once the arm is
   calibrated. Use it to confirm the calibration feels right before you
   record anything.
6. **Record.** Set the task description, the number of episodes, the episode
   time and the reset time between episodes. The page has buttons for
   "next episode", "re-record this one" and "stop", replacing LeRobot's
   arrow-key controls. Aim for 10 to 50 clean episodes to start: fixed
   cameras, a consistent grasp, the object always visible.
7. **Browse.** Watch the episodes of the dataset on disk, every camera,
   frame by frame, with a joint-motion trace. No upload needed. Delete the
   bad takes here before training.
8. **Train.** Pick the dataset and a policy. ACT is the fast first choice;
   SmolVLA is a larger vision-language model. LeLab installs the training
   extra on demand the first time. Train on your own GPU, or log in to
   Hugging Face and rent one through Hugging Face Jobs. The job shows up
   with live logs and metrics.
9. **Run.** Once a job has a usable checkpoint, a green play button appears
   on the model. Clear the workspace, keep a hand near the power switch,
   and press play. The follower attempts the task on its own.
10. **Upload.** Log in to Hugging Face from the top bar (it gives you the
    exact login command to paste) and push datasets and models to the Hub
    in one click.

## Where things live

Everything LeLab writes is under `~/.cache/huggingface/lerobot/`:

| path | what |
|---|---|
| `calibration/robots/so101_follower/<name>.json` | follower calibration |
| `calibration/teleoperators/so101_leader/<name>.json` | leader calibration |
| `ports/{leader,follower}_port.txt` | last-used serial ports |
| `<hf_user>/<dataset>/` | recorded datasets, LeRobot v3 layout |

Hugging Face tokens are entered through the login flow, never written into
this repository.

## Troubleshooting

| symptom | usual cause |
|---|---|
| `lelab` says port 8000 is in use | a previous run is still alive; `lelab --stop` |
| no ports detected | adapter has no power, charge-only USB cable, or Linux permissions (step 1) |
| `setup-motors` fails on the first motor | more than one motor on the bus, loose 3-pin cable, Waveshare jumpers not on `B` |
| follower jerks or is offset from the leader | calibration missing or done under a different profile name; recalibrate both arms |
| a calibration motor never turns green | the joint was not swept to both ends of its range; run it again |
| camera missing from the list | on macOS grant camera access to the terminal app; on Linux check `/dev/video*` permissions |
| phone camera does not work | browsers only allow camera access over HTTPS off `localhost`; see `HTTPS_SETUP.md` in the LeLab repo |
| policy does nothing useful | look at the dataset first in Browse: too few episodes, cameras moved, inconsistent grasps |

## Sources

- [LeLab on GitHub](https://github.com/huggingface/leLab) and the
  [LeLab Space](https://huggingface.co/spaces/lerobot/LeLab), which carries
  the install one-liner.
- LeRobot docs: [installation](https://huggingface.co/docs/lerobot/installation),
  [SO-101](https://huggingface.co/docs/lerobot/so101),
  [cameras](https://huggingface.co/docs/lerobot/cameras),
  [imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots).

Checked against those pages in September 2026. LeLab moves fast; when the UI
differs from this guide, the UI wins.
