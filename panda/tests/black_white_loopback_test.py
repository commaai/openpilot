#!/usr/bin/env python3

# Loopback test between black panda (+ harness and power) and white/grey panda
# Tests all buses, including OBD CAN, which is on the same bus as CAN0 in this test.
# To be sure, the test should be run with both harness orientations


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

counter = 0
nonzero_bus_errors = 0
zero_bus_errors = 0
content_errors = 0

def run_test(sleep_duration):
  global counter, nonzero_bus_errors, zero_bus_errors, content_errors

  pandas = Panda.list()
  print(pandas)

  # make sure two pandas are connected
  if len(pandas) != 2:
    print("Connect white/grey and black panda to run this test!")
    assert False

  # connect
  pandas[0] = Panda(pandas[0])
  pandas[1] = Panda(pandas[1])

  black_panda = None
  other_panda = None

  # find out which one is black
  if pandas[0].is_black() and not pandas[1].is_black():
    black_panda = pandas[0]
    other_panda = pandas[1]
  elif not pandas[0].is_black() and pandas[1].is_black():
    black_panda = pandas[1]
    other_panda = pandas[0]
  else:
    print("Connect white/grey and black panda to run this test!")
    assert False

  # disable safety modes
  black_panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  other_panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # test health packet
  print("black panda health", black_panda.health())
  print("other panda health", other_panda.health())

  # test black -> other
  while True:
    test_buses(black_panda, other_panda, True, [(0, False, [0]), (1, False, [1]), (2, False, [2]), (1, True, [0])], sleep_duration)
    test_buses(black_panda, other_panda, False, [(0, False, [0]), (1, False, [1]), (2, False, [2]), (0, True, [0, 1])], sleep_duration)
    counter += 1
    print("Number of cycles:", counter, "Non-zero bus errors:", nonzero_bus_errors, "Zero bus errors:", zero_bus_errors, "Content errors:", content_errors)

    # Toggle relay
    black_panda.set_safety_mode(Panda.SAFETY_NOOUTPUT)
    time.sleep(1)
    black_panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
    time.sleep(1)


def test_buses(black_panda, other_panda, direction, test_array, sleep_duration):
  global nonzero_bus_errors, zero_bus_errors, content_errors

  if direction:
    print("***************** TESTING (BLACK --> OTHER) *****************")
  else:
    print("***************** TESTING (OTHER --> BLACK) *****************")

  for send_bus, obd, recv_buses in test_array:
    black_panda.send_heartbeat()
    other_panda.send_heartbeat()
    print("\ntest can: ", send_bus, " OBD: ", obd)

    # set OBD on black panda
    black_panda.set_gmlan(True if obd else None)

    # clear and flush
    if direction:
      black_panda.can_clear(send_bus)
    else:
      other_panda.can_clear(send_bus)

    for recv_bus in recv_buses:
      if direction:
        other_panda.can_clear(recv_bus)
      else:
        black_panda.can_clear(recv_bus)

    black_panda.can_recv()
    other_panda.can_recv()

    # send the characters
    at = random.randint(1, 2000)
    st = get_test_string()[0:8]
    if direction:
      black_panda.can_send(at, st, send_bus)
    else:
      other_panda.can_send(at, st, send_bus)
    time.sleep(0.1)

    # check for receive
    if direction:
      cans_echo = black_panda.can_recv()
      cans_loop = other_panda.can_recv()
    else:
      cans_echo = other_panda.can_recv()
      cans_loop = black_panda.can_recv()

    loop_buses = []
    for loop in cans_loop:
      if (loop[0] != at) or (loop[2] != st):
	      content_errors += 1

      print("  Loop on bus", str(loop[3]))
      loop_buses.append(loop[3])
    if len(cans_loop) == 0:
      print("  No loop")
      if not os.getenv("NOASSERT"):
        assert False

    # test loop buses
    recv_buses.sort()
    loop_buses.sort()
    if(recv_buses != loop_buses):
      if len(loop_buses) == 0:
	      zero_bus_errors += 1
      else:
	      nonzero_bus_errors += 1
      if not os.getenv("NOASSERT"):
        assert False
    else:
      print("  TEST PASSED")

    time.sleep(sleep_duration)
  print("\n")

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
