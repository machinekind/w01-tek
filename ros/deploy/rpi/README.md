# RPi networking: ethernet cable ⇄ self-hosted wifi AP

The RPi is a **fixed anchor at `10.42.0.2`** and picks its link automatically:

| Situation            | Link                | RPi IP      | PC IP           | Internet on RPi     |
|----------------------|---------------------|-------------|-----------------|---------------------|
| Cable docked         | eth0 (static)       | 10.42.0.2   | 10.42.0.1       | yes, via PC NAT     |
| No cable (mobile)    | wlan0 AP `wojtek-link` | 10.42.0.2 | DHCP .10–.50    | no (robot anchor)   |

`wojtek-net.service` watches `eth0`'s carrier: cable present → tear the AP down
and use eth; cable absent → bring the `wlan0` AP up. Same `.2` on both
interfaces is safe because they are mutually exclusive.

The PC stays `10.42.0.1` over the cable (`wojtek-eth`, NM shared). On the AP
path the PC associates to `wojtek-link` and gets a DHCP lease from the RPi.

## Files

| File                     | Goes to                                   |
|--------------------------|-------------------------------------------|
| `60-wojtek.yaml`         | `/etc/netplan/60-wojtek.yaml` (chmod 600) |
| `hostapd.conf`           | `/etc/hostapd/hostapd.conf`               |
| `dnsmasq-wojtek.conf`    | `/etc/dnsmasq.d/wojtek.conf`              |
| `wojtek-net-switch.sh`   | `/usr/local/sbin/` (chmod +x)             |
| `wojtek-net.service`     | `/etc/systemd/system/`                    |

## Install

You don't wire this by hand — it's phase 4 of `deploy/rpi/install.sh`, run for
you by `./deploy.sh --provision` from the PC (see the top-level README and
`IMAGE.md`). `install.sh` is idempotent: it drops the files above into place,
points hostapd at its conf, leaves hostapd/dnsmasq disabled (the switch runs
them on demand), swaps netplan to the static-eth `60-wojtek.yaml`, disables the
cloud-init network default, and enables `wojtek-net.service`.

The one thing the script deliberately WON'T do is the very first eth renumber
(DHCP → static `.2`) while you're connected over that same eth — that would drop
SSH. On a fresh Pi the static `.2` comes from the `cloud-init/` network-config
at first boot instead (see `IMAGE.md`), so by the time `install.sh` runs, eth0
is already `.2` and it just re-applies cleanly.

## PC side (NetworkManager)

The PC has ONE ethernet port (`enp67s0`) that does double duty, so its role is
chosen by which profile is active — NM can't tell by the cable alone:

| Profile                  | autoconnect | ipv4   | Role                                  |
|--------------------------|-------------|--------|---------------------------------------|
| `Połączenie przewodowe 1`| **yes**     | auto   | DEFAULT: any cable = DHCP client (home internet) |
| `wojtek-eth`             | **no**      | shared | RPi dock: PC = 10.42.0.1, DHCP+NAT to the RPi |
| `wojtek-link 1`          | yes         | auto   | wifi client of the RPi's AP (mobile)  |

So plugging the home internet cable "just works" (DHCP). To **dock the RPi by
cable** instead, activate the shared profile explicitly:

```bash
nmcli con up wojtek-eth       # dock: PC serves .1 + NAT, talk to RPi over eth
nmcli con up "Połączenie przewodowe 1"   # undock: back to home-internet DHCP
```

(`wojtek-eth` is `autoconnect no` on purpose — otherwise NM grabs every cable
as the wojtek net and its 10.42.0.0/24 route hijacks SSH to the RPi away from
the wifi path.) The old PC-side `wojtek-link` AP profile was deleted; the PC is
a client now.

## Test the failover

- **Cable in:** `ip -4 addr show eth0` → `10.42.0.2`; `hostapd` inactive.
- **Cable out:** within a few seconds `hostapd` active, `wlan0` has `10.42.0.2`;
  connect the PC to `wojtek-link` (psk: the `WOJTEK_AP_PSK` value from `.env`) → PC gets a `.10–.50`
  lease; `ros2 topic list` sees the robot.
- `journalctl -t wojtek-net -f` shows each switch decision.
