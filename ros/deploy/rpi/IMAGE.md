# Raspberry Pi image and provisioning

The repository provisions a stock Ubuntu Server image instead of distributing a
preconfigured SD-card image. Commands below run from the repository root on a
Linux host. The flashing script uses Linux block devices, `lsblk`, `partprobe`,
mount tools, `dd`, `xz`, `curl` and `sha256sum`.

## Image identity

| Field | Repository setting |
|---|---|
| Image pinned by `flash-card.sh` | Ubuntu Server 24.04.4 LTS |
| Architecture | arm64 |
| Flavor | `preinstalled-server-arm64+raspi` |
| Previously documented hardware validation | Raspberry Pi 5, 8 GB |
| ROS distribution installed by default | Jazzy |

Current control configuration also contains Pi 3 tuning. That does not extend
the image-validation record to every Pi model. Record the actual board, image,
kernel and test outcome when commissioning a new unit.

The script obtains the pinned image and `SHA256SUMS` from Ubuntu's release
server, then checks the checksum before writing. If that image is no longer at
the configured URL, resolve the download/pin explicitly. Do not assume an
unverified newer point release is interchangeable.

## Prepare the host and card

1. Set up the Ethernet link using [Robot networking](README.md).
2. Copy `.env.example` to `.env` if needed, and configure the robot profile and
   policy organization/access. Put `UBUNTU_PRO_TOKEN` and `WOJTEK_AP_PSK` in
   `ros/.env`, using `ros/.env.example` as the template.
3. Identify the removable card with `lsblk`. Flashing erases the selected device.
   Use the whole card path, not a partition.

Inspect the proposed operation first; replace `/dev/sdX` with the actual card:

```bash
./ros/deploy/rpi/flash-card.sh /dev/sdX --dry-run
./ros/deploy/rpi/flash-card.sh /dev/sdX
```

The script asks for the device path again before erasing. It injects an existing
SSH public key into cloud-init; `--key /path/to/key.pub` chooses another public
key. The first boot uses the repository's static robot address.

`--image /path/to/image.img.xz` uses a local image **without checksum verification
by the script**. Verify that image against its official checksum before selecting
it. Retain its name and checksum in the unit record.

## Provision the robot

Boot the Pi with Ethernet attached to the configured PC. Keep the motors
unpowered during initial provisioning and inspect the intended service/profile
before enabling normal operation.

```bash
./ros/deploy.sh --provision
```

Provisioning installs ROS, the RT kernel, tuning, network and services, then
builds the robot workspace. The Ubuntu Pro token is used for RT-kernel setup.
The script may reboot the Pi and wait for SSH to return.

Without `--enable-robot`, provisioning disables robot-service boot autostart;
this does not stop a service that is already running. `--enable-robot` explicitly
requests autostart. Provisioning also installs/enables the controller-manager
watchdog, so stop both services before later bench work. See
[Commissioning](../../../docs/commissioning.md) before first motor operation.

## Verify the resulting installation

On the Pi, record:

```bash
tr -d '\0' < /proc/device-tree/model
cat /etc/os-release
uname -r
cat /sys/kernel/realtime
systemctl cat wojtek-robot.service
systemctl is-enabled wojtek-robot.service wojtek-cm-watchdog.service
```

An RT kernel exposes `1` in `/sys/kernel/realtime`. Check the complete service,
including its drop-in, for the intended robot profile, folded boot pose and
recording settings. Compare the actual installation with the
[hardware record](../../../docs/hardware-evidence.md); an image download or
successful build alone is not a physical acceptance test.
