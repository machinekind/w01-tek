# RPi base OS image

The robot RPi runs a **stock Ubuntu Server image** — we do NOT keep a binary
image blob in git (multi-GB, opaque, drifts). Instead this file pins its
identity so anyone can flash the exact same base, and `deploy/rpi/install.sh`
turns a fresh flash into a provisioned robot.

## Exact image

| | |
|---|---|
| Distro   | Ubuntu Server 24.04.x LTS (Noble Numbat) |
| Verified on | 24.04.4 LTS |
| Arch     | arm64 |
| Flavor   | `preinstalled-server` for Raspberry Pi (`+raspi`) |
| Hardware | Raspberry Pi 5 (8 GB) |

Download page: https://ubuntu.com/download/raspberry-pi
Direct (point release, adjust `24.04.N` to the latest):
```
https://cdimage.ubuntu.com/releases/24.04/release/ubuntu-24.04.4-preinstalled-server-arm64+raspi.img.xz
```

## Verify before flashing

```bash
# Fetch the official checksums and check the download against them:
curl -O https://cdimage.ubuntu.com/releases/24.04/release/SHA256SUMS
sha256sum -c SHA256SUMS 2>/dev/null | grep 'preinstalled-server-arm64+raspi'
# -> ...+raspi.img.xz: OK
```

## Flash + first-boot config

The scripted way (does download, verify, flash, and cloud-init with your SSH
key injected — all of the below in one command):

```bash
./deploy/rpi/flash-card.sh /dev/sdX      # check `lsblk` for the right device!
```

Then boot the Pi **docked by cable** (PC on `wojtek-eth`, so it has internet via
NAT), and from the PC run `./deploy.sh --provision`.

Manual fallback, if you'd rather not use the script:
1. Flash with Raspberry Pi Imager or `xzcat ...img.xz | sudo dd of=/dev/sdX bs=4M status=progress`.
2. Copy `deploy/rpi/cloud-init/{user-data,network-config,meta-data}` onto the FAT
   `system-boot` partition — paste your SSH public key into `user-data` first.
   This gives the Pi its hostname, your key, and static `10.42.0.2` on eth0.

## Note on point releases

Ubuntu refreshes the `24.04.x` image over time; the download page always serves
the latest. Provisioning is version-tolerant (it's just apt on top), so a newer
`24.04.x` is fine. Bump the "Verified on" row when you move to a new one.
