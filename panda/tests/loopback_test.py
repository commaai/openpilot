#!/usr/bin/env python3

import os
import time
import random
import argparse
from itertools import permutations

from panda import Panda

def get_test_string():
  return b"test" + os.urandom(10)

def run_test(sleep_duration):
  pandas = Panda.list()
  print(pandas)

  if len(pandas) < 2:
    raise Exception("Minimum two pandas are needed for test")

  run_test_w_pandas(pandas, sleep_duration)

def run_test_w_pandas(pandas, sleep_duration):
  h = [Panda(x) for x in pandas]
  print("H", h)

  for hh in h:
    hh.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # test both directions
  for ho in permutations(list(range(len(h))), r=2):
    print("***************** TESTING", ho)

    panda0, panda1 = h[ho[0]], h[ho[1]]

    # **** test health packet ****
    print("health", ho[0], h[ho[0]].health())

    # **** test can line loopback ****
    for bus, obd in [(0, False), (1, False), (2, False), (1, True), (2, True)]:
      print("\ntest can", bus)
      # flush
      cans_echo = panda0.can_recv()
      cans_loop = panda1.can_recv()

      panda0.set_obd(None)
      panda1.set_obd(None)

      if obd is True:
        panda0.set_obd(bus)
        panda1.set_obd(bus)
        bus = 3

      # send the characters
      at = random.randint(1, 2000)
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

      assert cans_echo[0][1] == st
      assert cans_loop[0][1] == st

      assert cans_echo[0][2] == 0x80 | bus
      if cans_loop[0][2] != bus:
        print("EXPECTED %d GOT %d" % (bus, cans_loop[0][2]))
      assert cans_loop[0][2] == bus

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
    for _ in range(args.n):
      run_test(sleep_duration=args.sleep)
