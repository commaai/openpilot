#!/usr/bin/env python3

# Loopback test between two black pandas (+ harness and power)
# Tests all buses, including OBD CAN, which is on the same bus as CAN0 in this test.
# To be sure, the test should be run with both harness orientations


import os
import sys
import time
import random
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda  # noqa: E402

def get_test_string():
  return b"test" + os.urandom(10)

def run_test(sleep_duration):
  pandas = Panda.list()
  print(pandas)

  # make sure two pandas are connected
  if len(pandas) != 2:
    print("Connect white/grey and black panda to run this test!")
    assert False

  # connect
  pandas[0] = Panda(pandas[0])
  pandas[1] = Panda(pandas[1])

  # find out the hardware types
  if not pandas[0].is_black() or not pandas[1].is_black():
    print("Connect two black pandas to run this test!")
    assert False

  for panda in pandas:
    # disable safety modes
    panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

    # test health packet
    print("panda health", panda.health())

  # setup test array (send bus, sender obd, reciever obd, expected busses)
  test_array = [
    (0, False, False, [0]),
    (1, False, False, [1]),
    (2, False, False, [2]),
    (0, False, True, [0, 1]),
    (1, False, True, []),
    (2, False, True, [2]),
    (0, True, False, [0]),
    (1, True, False, [0]),
    (2, True, False, [2]),
    (0, True, True, [0, 1]),
    (1, True, True, [0, 1]),
    (2, True, True, [2])
  ]

  # test both orientations
  print("***************** TESTING (0 --> 1) *****************")
  test_buses(pandas[0], pandas[1], test_array, sleep_duration)
  print("***************** TESTING (1 --> 0) *****************")
  test_buses(pandas[1], pandas[0], test_array, sleep_duration)


def test_buses(send_panda, recv_panda, test_array, sleep_duration):
  for send_bus, send_obd, recv_obd, recv_buses in test_array:
    send_panda.send_heartbeat()
    recv_panda.send_heartbeat()
    print("\nSend bus:", send_bus, " Send OBD:", send_obd, " Recv OBD:", recv_obd)

    # set OBD on pandas
    send_panda.set_gmlan(True if send_obd else None)
    recv_panda.set_gmlan(True if recv_obd else None)

    # clear and flush
    send_panda.can_clear(send_bus)
    for recv_bus in recv_buses:
      recv_panda.can_clear(recv_bus)
    send_panda.can_recv()
    recv_panda.can_recv()

    # send the characters
    at = random.randint(1, 2000)
    st = get_test_string()[0:8]
    send_panda.can_send(at, st, send_bus)
    time.sleep(0.1)

    # check for receive
    _ = send_panda.can_recv()  # cans echo
    cans_loop = recv_panda.can_recv()

    loop_buses = []
    for loop in cans_loop:
      print("  Loop on bus", str(loop[3]))
      loop_buses.append(loop[3])
    if len(cans_loop) == 0:
      print("  No loop")

    # test loop buses
    recv_buses.sort()
    loop_buses.sort()
    assert recv_buses == loop_buses
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
