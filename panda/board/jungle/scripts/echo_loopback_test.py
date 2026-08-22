#!/usr/bin/env python3
import os
import time
import contextlib
import random
from termcolor import cprint

from panda import PandaJungle

# This script is intended to be used in conjunction with the echo.py test script from panda.
# It sends messages on bus 0, 1, 2 and checks for a reversed response to be sent back.

#################################################################
############################# UTILS #############################
#################################################################
def print_colored(text, color):
  cprint(text + " "*40, color, end="\r")

def get_test_string():
  return b"test"+os.urandom(4)

#################################################################
############################# TEST ##############################
#################################################################

def test_loopback():
  for bus in range(3):
    # Clear can
    jungle.can_clear(bus)
    # Send a random message
    address = random.randint(1, 2000)
    data = get_test_string()
    jungle.can_send(address, data, bus)
    time.sleep(0.1)

    # Make sure it comes back reversed
    incoming = jungle.can_recv()
    found = False
    for message in incoming:
      incomingAddress, incomingData, incomingBus = message
      if incomingAddress == address and incomingData == data[::-1] and incomingBus == bus:
        found = True
        break
    if not found:
      cprint("\nFAILED", "red")
      raise AssertionError

#################################################################
############################# MAIN ##############################
#################################################################
jungle = None
counter = 0

if __name__ == "__main__":
  # Connect to jungle silently
  print_colored("Connecting to jungle", "blue")
  with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
      jungle = PandaJungle()
  jungle.set_panda_power(True)
  jungle.set_ignition(False)

  # Run test
  while True:
    jungle.can_clear(0xFFFF)
    test_loopback()
    counter += 1
    print_colored(str(counter) + " loopback cycles complete", "blue")
