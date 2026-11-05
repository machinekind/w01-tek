# Robot networking

The provisioned Pi uses `10.42.0.2` on the active robot link. These addresses are
part of the repository's network design. Check for a conflicting `10.42.0.0/24`
network on the development PC before using it.

| Connection | Robot | PC or handheld | Robot internet access |
|---|---|---|---|
| Ethernet to a configured Linux PC | `eth0`, `10.42.0.2` | PC `10.42.0.1` | Through the PC's shared connection |
| Robot's `wojtek-link` Wi-Fi AP | `wlan0`, `10.42.0.2` | DHCP `.10`–`.50` | No uplink supplied by the AP |

`wojtek-net.service` watches Ethernet carrier. With a cable present it disables
the AP; without one it enables the AP. Moving between links can interrupt SSH,
DDS and operator connections. Test the transition while the robot is stationary.

## Provisioning

From the repository root, use the [image procedure](IMAGE.md), then
`./ros/deploy.sh --provision`. The network phase of `install.sh` installs:

| File | Destination |
|---|---|
| `60-wojtek.yaml` | `/etc/netplan/60-wojtek.yaml` |
| `hostapd.conf` | `/etc/hostapd/hostapd.conf` |
| `dnsmasq-wojtek.conf` | `/etc/dnsmasq.d/wojtek.conf` |
| `wojtek-net-switch.sh` | `/usr/local/sbin/wojtek-net-switch.sh` |
| `wojtek-net.service` | `/etc/systemd/system/wojtek-net.service` |

Set `WOJTEK_AP_PSK` in the gitignored `ros/.env` before provisioning. The installer
fills the AP passphrase at runtime. Do not put it in a committed configuration.
The first boot's cloud-init supplies static Ethernet addressing; the installer
refuses the disruptive first DHCP-to-static change over an existing SSH link.

## Linux PC with NetworkManager

Run on the host, from the repository root:

```bash
./ros/deploy/pc/setup-net.sh             # auto-detect an Ethernet interface
# Or: ./ros/deploy/pc/setup-net.sh <interface>
nmcli con up wojtek-eth
nmcli -f GENERAL.DEVICE,IP4.ADDRESS device show
```

Confirm the selected Ethernet interface has `10.42.0.1/24` before provisioning
the Pi. `wojtek-eth` uses shared mode and intentionally has autoconnect disabled,
so an ordinary Internet cable can still use your normal wired profile. To switch
back, use `nmcli con show` to find that profile, then activate it with
`nmcli con up '<profile name>'`.

This script requires NetworkManager. It does not configure macOS Internet
Sharing or another Linux network manager. Configure an equivalent static link
and routing there, or use the robot AP for operation after provisioning on a
supported host.

## Mobile operation

Join `wojtek-link` using the passphrase configured during provisioning. The AP
provides an on-link address and deliberately advertises neither a default route
nor DNS. A client with another network connection can retain its Internet route.
The robot itself still has no Internet uplink through this AP.

Policies and browser assets must already be cached on the robot. The
[operator guide](../../../docs/operations.md) covers the panel and pad paths;
the [ROS guide](../../README.md) covers policy deployment.

## Verify the link

On the Pi:

```bash
ip -4 addr show eth0
ip -4 addr show wlan0
systemctl status wojtek-net.service hostapd dnsmasq
journalctl -t wojtek-net -b --no-pager
```

With cable connected, expect `eth0` at `.2` and `hostapd` inactive. With the
cable removed, expect `wlan0` at `.2`, `hostapd` active and AP clients receiving
`.10`–`.50`. Discovery also requires matching `ROS_DOMAIN_ID=42` and CycloneDDS.
Use a sourced ROS shell for `ros2 node list` and `ros2 topic list`; a working SSH
link alone does not establish DDS connectivity.

If the PC reaches the wrong interface, inspect its routes and active network
profiles before changing robot addresses. Keep network testing separate from a
moving-robot session.
