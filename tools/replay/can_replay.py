#!/usr/bin/env python3
import argparse
import os
import time
import usb1
import threading

os.environ['FILEREADER_CACHE'] = '1'

from openpilot.common.realtime import config_realtime_process, Ratekeeper, DT_CTRL
from openpilot.selfdrive.pandad import can_capnp_to_can_list
from openpilot.tools.lib.logreader import LogReader
from panda import PandaJungle

# set both to cycle power or ignition
PWR_ON = int(os.getenv("PWR_ON", "0"))
PWR_OFF = int(os.getenv("PWR_OFF", "0"))
IGN_ON = int(os.getenv("ON", "0"))
IGN_OFF = int(os.getenv("OFF", "0"))
ENABLE_IGN = IGN_ON > 0 and IGN_OFF > 0
ENABLE_PWR = PWR_ON > 0 and PWR_OFF > 0


def send_thread(j: PandaJungle, flock):
  if "FLASH" in os.environ:
    with flock:
      j.flash()

  j.reset()
  for i in [0, 1, 2, 3, 0xFFFF]:
    j.can_clear(i)
    j.set_can_speed_kbps(i, 500)
  j.set_ignition(True)
  j.set_panda_power(True)
  j.set_can_loopback(False)

  rk = Ratekeeper(1 / DT_CTRL, print_delay_threshold=None)
  while True:
    # handle cycling
    if ENABLE_PWR:
      i = (rk.frame*DT_CTRL) % (PWR_ON + PWR_OFF) < PWR_ON
      j.set_panda_power(i)
    if ENABLE_IGN:
      i = (rk.frame*DT_CTRL) % (IGN_ON + IGN_OFF) < IGN_ON
      j.set_ignition(i)

    snd = CAN_MSGS[rk.frame % len(CAN_MSGS)]
    snd = list(filter(lambda x: x[-1] <= 2, snd))
    try:
      j.can_send_many(snd)
    except usb1.USBErrorTimeout:
      # timeout is fine, just means the CAN TX buffer is full
      pass

    # Drain panda message buffer
    j.can_recv()
    rk.keep_time()


def connect():
  config_realtime_process(3, 55)

  serials = {}
  flashing_lock = threading.Lock()
  while True:
    # look for new devices
    for s in PandaJungle.list():
      if s not in serials:
        print("starting send thread for", s)
        serials[s] = threading.Thread(target=send_thread, args=(PandaJungle(s), flashing_lock))
        serials[s].start()

    # try to join all send threads
    cur_serials = serials.copy()
    for s, t in cur_serials.items():
      if t is  not None:
        t.join(0.01)
        if not t.is_alive():
          del serials[s]

    time.sleep(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Replay CAN messages from a route to all connected pandas and jungles in a loop.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to replay. If not specified, a default public route will be used.")
  args = parser.parse_args()

  def process(lr):
    return [can_capnp_to_can_list(m.can) for m in lr if m.which() == 'can']

  print("Loading log...")
  if args.route_or_segment_name is None:
    args.route_or_segment_name = "77611a1fac303767/2020-03-24--09-50-38/1:3"

  sr = LogReader(args.route_or_segment_name)

  CP = sr.first("carParams")

  print(f"carFingerprint (for hardcoding fingerprint): '{CP.carFingerprint}'")

  CAN_MSGS = sr.run_across_segments(24, process)

  print("Finished loading...")

  if ENABLE_PWR:
    print(f"Cycling power: on for {PWR_ON}s, off for {PWR_OFF}s")
  if ENABLE_IGN:
    print(f"Cycling ignition: on for {IGN_ON}s, off for {IGN_OFF}s")

  connect()
