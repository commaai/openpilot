#!/usr/bin/env python3

import os
import sys
from panda.tests.safety import libpandasafety_py
from panda.tests.safety_replay.helpers import package_can_msg, init_segment

# replay a drive to check for safety violations
def replay_drive(lr, safety_mode, param):
  safety = libpandasafety_py.libpandasafety

  err = safety.set_safety_hooks(safety_mode, param)
  assert err == 0, "invalid safety mode: %d" % safety_mode

  if "SEGMENT" in os.environ:
    init_segment(safety, lr, mode)

  rx_tot, rx_invalid, tx_tot, tx_blocked, tx_controls, tx_controls_blocked = 0, 0, 0, 0, 0, 0
  blocked_addrs = set()
  invalid_addrs = set()
  start_t = None

  for msg in lr:
    if start_t is None:
      start_t = msg.logMonoTime
    safety.set_timer(((msg.logMonoTime // 1000)) % 0xFFFFFFFF)

    if msg.which() == 'sendcan':
     for canmsg in msg.sendcan:
        to_send = package_can_msg(canmsg)
        sent = safety.safety_tx_hook(to_send)
        if not sent:
          tx_blocked += 1
          tx_controls_blocked += safety.get_controls_allowed()
          blocked_addrs.add(canmsg.address)

          if "DEBUG" in os.environ:
            print("blocked bus %d msg %d at %f" % (canmsg.src, canmsg.address, (msg.logMonoTime - start_t) / (1e9)))
        tx_controls += safety.get_controls_allowed()
        tx_tot += 1
    elif msg.which() == 'can':
      for canmsg in msg.can:
        # ignore msgs we sent
        if canmsg.src >= 128:
          continue
        to_push = package_can_msg(canmsg)
        recv = safety.safety_rx_hook(to_push)
        if not recv:
          rx_invalid += 1
          invalid_addrs.add(canmsg.address)
        rx_tot += 1

  print("\nRX")
  print("total rx msgs:", rx_tot)
  print("invalid rx msgs:", rx_invalid)
  print("invalid addrs:", invalid_addrs)
  print("\nTX")
  print("total openpilot msgs:", tx_tot)
  print("total msgs with controls allowed:", tx_controls)
  print("blocked msgs:", tx_blocked)
  print("blocked with controls allowed:", tx_controls_blocked)
  print("blocked addrs:", blocked_addrs)

  return tx_controls_blocked == 0 and rx_invalid == 0

if __name__ == "__main__":
  from tools.lib.route import Route
  from tools.lib.logreader import MultiLogIterator  # pylint: disable=import-error

  mode = int(sys.argv[2])
  param = 0 if len(sys.argv) < 4 else int(sys.argv[3])
  r = Route(sys.argv[1])
  lr = MultiLogIterator(r.log_paths(), wraparound=False)

  print("replaying drive %s with safety mode %d and param %d" % (sys.argv[1], mode, param))

  replay_drive(lr, mode, param)
