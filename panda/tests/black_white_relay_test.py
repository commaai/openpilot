#!/usr/bin/env python3

# Relay test with loopback between black panda (+ harness and power) and white/grey panda
# Tests the relay switching multiple times / second by looking at the buses on which loop occurs.


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
open_errors = 0
closed_errors = 0
content_errors = 0

def run_test(sleep_duration):
  global counter, open_errors, closed_errors, content_errors

  pandas = Panda.list()
  #pandas = ["540046000c51363338383037", "07801b800f51363038363036"]
  print(pandas)

  # make sure two pandas are connected
  if len(pandas) != 2:
    print("Connect white/grey and black panda to run this test!")
    assert False

  # connect
  pandas[0] = Panda(pandas[0])
  pandas[1] = Panda(pandas[1])

  # find out which one is black
  type0 = pandas[0].get_type()
  type1 = pandas[1].get_type()

  black_panda = None
  other_panda = None
  
  if type0 == "\x03" and type1 != "\x03":
    black_panda = pandas[0]
    other_panda = pandas[1]
  elif type0 != "\x03" and type1 == "\x03":
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
    # Switch on relay
    black_panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
    time.sleep(0.05)

    if not test_buses(black_panda, other_panda, (0, False, [0])):
      open_errors += 1
      print("Open error")
      assert False

    # Switch off relay
    black_panda.set_safety_mode(Panda.SAFETY_NOOUTPUT)
    time.sleep(0.05)

    if not test_buses(black_panda, other_panda, (0, False, [0, 2])):
      closed_errors += 1
      print("Close error")
      assert False

    counter += 1
    print("Number of cycles:", counter, "Open errors:", open_errors, "Closed errors:", closed_errors, "Content errors:", content_errors)	

def test_buses(black_panda, other_panda, test_obj):
  global content_errors
  send_bus, obd, recv_buses = test_obj
    
  black_panda.send_heartbeat()
  other_panda.send_heartbeat()
    
  # Set OBD on send panda
  other_panda.set_gmlan(True if obd else None)

  # clear and flush
  other_panda.can_clear(send_bus)

  for recv_bus in recv_buses:
    black_panda.can_clear(recv_bus)
    
  black_panda.can_recv()
  other_panda.can_recv()

  # send the characters
  at = random.randint(1, 2000)
  st = get_test_string()[0:8]
  other_panda.can_send(at, st, send_bus)
  time.sleep(0.05)

  # check for receive
  cans_echo = other_panda.can_recv()
  cans_loop = black_panda.can_recv()

  loop_buses = []
  for loop in cans_loop:
    if (loop[0] != at) or (loop[2] != st):
      content_errors += 1
    loop_buses.append(loop[3])
    
  # test loop buses
  recv_buses.sort()
  loop_buses.sort()
  if(recv_buses != loop_buses):
    return False
  else:
    return True

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
