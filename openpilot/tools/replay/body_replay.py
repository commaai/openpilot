#!/usr/bin/env python3
import argparse
import os
import threading
import time

import usb1

from opendbc.can import CANPacker
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Ratekeeper
from panda import PandaJungle


# Set both values in a pair to cycle power or ignition.
PWR_ON = int(os.getenv("PWR_ON", "0"))
PWR_OFF = int(os.getenv("PWR_OFF", "0"))
IGN_ON = int(os.getenv("ON", "0"))
IGN_OFF = int(os.getenv("OFF", "0"))
ENABLE_IGN = IGN_ON > 0 and IGN_OFF > 0
ENABLE_PWR = PWR_ON > 0 and PWR_OFF > 0

PACKER = CANPacker("comma_body")


def create_body_messages(frame: int, bus: int) -> list[tuple[int, bytes, int]]:
  """Create one 100 Hz tick of the messages broadcast by a comma body."""
  messages = [
    PACKER.make_can_msg("MOTORS_DATA", bus, {"COUNTER": frame % 16}),
    PACKER.make_can_msg("MOTORS_CURRENT", bus, {}),
    PACKER.make_can_msg("MOTORS_ANGLE", bus, {}),
    (0x206, b"\x00" * 6, bus),  # BOARD_IMU_RAW1 (not present in comma_body.dbc)
    (0x207, b"\x00" * 6, bus),  # BOARD_IMU_RAW2 (not present in comma_body.dbc)
  ]

  if frame % 10 == 0:
    messages.append(PACKER.make_can_msg("VAR_VALUES", bus, {
      "IGNITION": 1,
      "ENABLE_MOTORS": 1,
    }))

  if frame % 100 == 0:
    messages.append(PACKER.make_can_msg("BODY_DATA", bus, {
      "MCU_TEMP": 25.0,
      "BATT_VOLTAGE": 29.4,
      "BATT_PERCENTAGE": 100,
    }))

  return messages


def send_thread(j: PandaJungle, flashing_lock: threading.Lock, bus: int, can_speed: int) -> None:
  if "FLASH" in os.environ:
    with flashing_lock:
      j.flash()

  j.reset()
  for i in [0, 1, 2, 3, 0xFFFF]:
    j.can_clear(i)
    j.set_can_speed_kbps(i, can_speed)
  j.set_ignition(True)
  j.set_panda_power(True)
  j.set_can_loopback(False)

  rk = Ratekeeper(1 / DT_CTRL, print_delay_threshold=None)
  while True:
    if ENABLE_PWR:
      power_on = (rk.frame * DT_CTRL) % (PWR_ON + PWR_OFF) < PWR_ON
      j.set_panda_power(power_on)
    if ENABLE_IGN:
      ignition_on = (rk.frame * DT_CTRL) % (IGN_ON + IGN_OFF) < IGN_ON
      j.set_ignition(ignition_on)

    try:
      j.can_send_many(create_body_messages(rk.frame, bus))
    except usb1.USBErrorTimeout:
      # A full CAN TX buffer is harmless; the next tick will contain fresh data.
      pass

    j.can_recv()  # Drain the panda message buffer.
    rk.keep_time()


def connect(bus: int, can_speed: int) -> None:
  config_realtime_process(3, 55)

  serials: dict[str, threading.Thread] = {}
  flashing_lock = threading.Lock()
  while True:
    for serial in PandaJungle.list():
      if serial not in serials:
        print("starting BODY send thread for", serial)
        thread = threading.Thread(target=send_thread, args=(PandaJungle(serial), flashing_lock, bus, can_speed))
        serials[serial] = thread
        thread.start()

    for serial, thread in serials.copy().items():
      thread.join(0.01)
      if not thread.is_alive():
        del serials[serial]

    time.sleep(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Send default comma body CAN messages to all connected pandas and jungles.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--bus", type=int, choices=range(4), default=0, help="CAN bus to transmit on")
  parser.add_argument("--can-speed", type=int, default=500, help="CAN bus speed in kbps")
  args = parser.parse_args()

  if ENABLE_PWR:
    print(f"Cycling power: on for {PWR_ON}s, off for {PWR_OFF}s")
  if ENABLE_IGN:
    print(f"Cycling ignition: on for {IGN_ON}s, off for {IGN_OFF}s")

  connect(args.bus, args.can_speed)
