#!/usr/bin/env python3

import os
import sys
import time
import random
import argparse

from hexdump import hexdump
from itertools import permutations

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

def get_test_string():
  return b"test"+os.urandom(10)

def run_test(sleep_duration):
  pandas = Panda.list()
  print(pandas)

  if len(pandas) == 0:
    print("NO PANDAS")
    assert False

  if len(pandas) == 1:
    # if we only have one on USB, assume the other is on wifi
    pandas.append("WIFI")
  run_test_w_pandas(pandas, sleep_duration)

def run_test_w_pandas(pandas, sleep_duration):
  h = list([Panda(x) for x in pandas])
  print("H", h)

  for hh in h:
    hh.set_controls_allowed(True)

  # test both directions
  for ho in permutations(list(range(len(h))), r=2):
    print("***************** TESTING", ho)

    panda0, panda1 = h[ho[0]], h[ho[1]]

    if(panda0._serial == "WIFI"):
      print("  *** Can not send can data over wifi panda. Skipping! ***")
      continue

    # **** test health packet ****
    print("health", ho[0], h[ho[0]].health())

    # **** test K/L line loopback ****
    for bus in [2,3]:
      # flush the output
      h[ho[1]].kline_drain(bus=bus)

      # send the characters
      st = get_test_string()
      st = b"\xaa"+chr(len(st)+3).encode()+st
      h[ho[0]].kline_send(st, bus=bus, checksum=False)

      # check for receive
      ret = h[ho[1]].kline_drain(bus=bus)

      print("ST Data:")
      hexdump(st)
      print("RET Data:")
      hexdump(ret)
      assert st == ret
      print("K/L pass", bus, ho, "\n")
      time.sleep(sleep_duration)

    # **** test can line loopback ****
#    for bus, gmlan in [(0, None), (1, False), (2, False), (1, True), (2, True)]:
for bus, gmlan in [(0, None), (1, None)]:
      print("\ntest can", bus)
      # flush
      cans_echo = panda0.can_recv()
      cans_loop = panda1.can_recv()

      if gmlan is not None:
        panda0.set_gmlan(gmlan, bus)
        panda1.set_gmlan(gmlan, bus)

      # send the characters
      # pick addresses high enough to not conflict with honda code
      at = random.randint(1024, 2000)
      st = get_test_string()[0:8]
      panda0.can_send(at, st, bus)
      time.sleep(0.1)

      # check for receive
      cans_echo = panda0.can_recv()
      cans_loop = panda1.can_recv()

      print("Bus", bus, "echo", cans_echo, "loop", cans_loop)

      assert len(cans_echo) == 1
      assert len(cans_loop) == 1

      assert cans_echo[0][0] == at
      assert cans_loop[0][0] == at

      assert cans_echo[0][2] == st
      assert cans_loop[0][2] == st

      assert cans_echo[0][3] == 0x80 | bus
      if cans_loop[0][3] != bus:
        print("EXPECTED %d GOT %d" % (bus, cans_loop[0][3]))
      assert cans_loop[0][3] == bus

      print("CAN pass", bus, ho)
      time.sleep(sleep_duration)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", type=int, help="Number of test iterations to run")
  parser.add_argument("-sleep", type=int, help="Sleep time between tests", default=0)
  args = parser.parse_args()

  if args.n is None:
    while True:
      run_test(sleep_duration=args.sleep)
  else:
    for i in range(args.n):
      run_test(sleep_duration=args.sleep)
