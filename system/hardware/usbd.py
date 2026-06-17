#!/usr/bin/env python3
from pathlib import Path

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.helpers import USBGPU_VID, USBGPU_PID

RATE = 10  # Hz


def find_device() -> Path | None:
  # discover the eGPU by VID/PID
  for d in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
          int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        return d
    except Exception:
      pass
  return None


def read(device: Path, attr: str) -> str | None:
  try:
    return (device / attr).read_text().strip()
  except OSError:
    return None


def over_current_count(device: Path) -> int:
  # upstream root-hub port, e.g. device "4-1" -> usb4/4-0:1.0/usb4-port1
  bus, _, port = device.name.partition("-")
  try:
    return int((Path(f"/sys/bus/usb/devices/usb{bus}/{bus}-0:1.0/usb{bus}-port{port}") / "over_current_count").read_text())
  except (OSError, ValueError):
    return 0


def main():
  pm = messaging.PubMaster(['usbState'])
  rk = Ratekeeper(RATE)
  disconnect_count = 0
  was_connected = False

  while True:
    device = find_device()
    connected = device is not None
    if was_connected and not connected:
      disconnect_count += 1
      cloudlog.event("usb_disconnected", count=disconnect_count)
    elif connected and not was_connected:
      cloudlog.event("usb_connected", speed=read(device, "speed"))
    was_connected = connected

    msg = messaging.new_message('usbState', valid=True)
    state = msg.usbState
    state.connected = connected
    if device is not None:
      speed = read(device, "speed")
      state.speedMbps = int(speed) if (speed and speed.isdigit()) else 0
      state.pmActive = read(device, "power/runtime_status") == "active"
      state.overCurrentCount = over_current_count(device)
    state.disconnectCount = disconnect_count

    pm.send('usbState', msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
