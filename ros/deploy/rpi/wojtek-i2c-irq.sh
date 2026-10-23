#!/bin/bash
# Pin the IMU I2C controller's interrupt onto the RT core (3).
#
# isolcpus keeps IRQs off the isolated cores by default, so the i2c
# completion interrupt lands on CPU0 together with the camera's USB traffic
# and the rest of the system. The RT loop's read() blocks on exactly that
# interrupt: whenever CPU0 is loaded, the completion is late and the whole
# 200 Hz loop stalls in the kernel (measured 2026-08-24: a stall every
# 1-2 min with perception on; zero after pinning). Handling the IRQ on the
# loop's own core is nearly free -- most of the time the loop is asleep
# waiting for precisely this interrupt, and the handler itself is a few
# microseconds of FIFO shuffling against a 5000 us cycle budget.
#
# The IRQ number is dynamic on device-tree platforms, so resolve it by the
# controller's name instead of hardcoding it.
set -eu
IRQ=$(awk -F: '/fe804000\.i2c/ {gsub(/ /,"",$1); print $1; exit}' /proc/interrupts)
if [ -z "${IRQ}" ]; then
    echo "wojtek-i2c-irq: fe804000.i2c not found in /proc/interrupts" >&2
    exit 1
fi
echo 3 > "/proc/irq/${IRQ}/smp_affinity_list"
echo "wojtek-i2c-irq: i2c IRQ ${IRQ} pinned to cpu3"
