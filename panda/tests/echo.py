#!/usr/bin/env python3
import os
import sys
import time
import _thread

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda  # noqa: E402

# This script is intended to be used in conjunction with the echo_loopback_test.py test script from panda jungle.
# It sends a reversed response back for every message received containing b"test".

def heartbeat_thread(p):
  while True:
    try:
      p.send_heartbeat()
      time.sleep(1)
    except Exception:
      break

# Resend every CAN message that has been received on the same bus, but with the data reversed
if __name__ == "__main__":
  p = Panda()
  _thread.start_new_thread(heartbeat_thread, (p,))
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_power_save(False)

  while True:
    incoming = p.can_recv()
    for message in incoming:
      address, notused, data, bus = message
      if b'test' in data:
        p.can_send(address, data[::-1], bus)
