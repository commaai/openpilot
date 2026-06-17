#!/usr/bin/env python3
import os

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

RATE = 10  # Hz
DEVICE = "/sys/bus/usb/devices/4-1"  # aux USB
PORT = "/sys/bus/usb/devices/usb4/4-0:1.0/usb4-port1"


def read(attr: str) -> str | None:
  try:
    with open(os.path.join(DEVICE, attr)) as f:
      return f.read().strip()
  except OSError:
    return None


def over_current_count() -> int:
  try:
    with open(os.path.join(PORT, "over_current_count")) as f:
      return int(f.read())
  except (OSError, ValueError):
    return 0


def main():
  pm = messaging.PubMaster(['usbState'])
  rk = Ratekeeper(RATE)
  disconnect_count = 0
  was_connected = False

  while True:
    speed = read("speed")
    connected = speed is not None
    if was_connected and not connected:
      disconnect_count += 1
      cloudlog.event("usb_disconnected", count=disconnect_count)
    elif connected and not was_connected:
      cloudlog.event("usb_connected", idVendor=read("idVendor"), idProduct=read("idProduct"), speed=speed)
    was_connected = connected

    msg = messaging.new_message('usbState', valid=True)
    state = msg.usbState
    state.connected = connected
    state.speedMbps = int(speed) if (speed and speed.isdigit()) else 0
    state.pmActive = read("power/runtime_status") == "active"
    state.disconnectCount = disconnect_count
    state.overCurrentCount = over_current_count()

    pm.send('usbState', msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
