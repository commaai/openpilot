#!/usr/bin/env python3
import os
import time
import contextlib
import random
from termcolor import cprint

from panda import Panda, PandaJungle

NUM_PANDAS_PER_TEST = 1
FOR_RELEASE_BUILDS = os.getenv("RELEASE") is not None       # Release builds do not have ALLOUTPUT mode

BUS_SPEEDS = [125, 500, 1000]

#################################################################
############################# UTILS #############################
#################################################################
# To suppress the connection text
def silent_panda_connect(serial):
  with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
      panda = Panda(serial)
  return panda

def print_colored(text, color):
  cprint(text + " "*40, color, end="\r")

def connect_to_pandas():
  print_colored("Connecting to pandas", "blue")
  # Connect to pandas
  pandas = []
  for serial in panda_serials:
    pandas.append(silent_panda_connect(serial))
  print_colored("Connected", "blue")

def start_with_orientation(orientation):
  print_colored("Restarting pandas with orientation " + str(orientation), "blue")
  jungle.set_panda_power(False)
  jungle.set_harness_orientation(orientation)
  time.sleep(4)
  jungle.set_panda_power(True)
  time.sleep(2)
  connect_to_pandas()

def can_loopback(sender):
  receivers = list(filter((lambda p: (p != sender)), [jungle] + pandas))

  for bus in range(4):
    obd = False
    if bus == 3:
      obd = True
      bus = 1

    # Clear buses
    for receiver in receivers:
      receiver.set_obd(obd)
      receiver.can_clear(bus)     # TX
      receiver.can_clear(0xFFFF)  # RX

    # Send a random string
    addr = 0x18DB33F1 if FOR_RELEASE_BUILDS else random.randint(1, 2000)
    string = b"test"+os.urandom(4)
    sender.set_obd(obd)
    time.sleep(0.2)
    sender.can_send(addr, string, bus)
    time.sleep(0.2)

    # Check if all receivers have indeed received them in their receiving buffers
    for receiver in receivers:
      content = receiver.can_recv()

      # Check amount of messages
      if len(content) != 1:
        raise Exception("Amount of received CAN messages (" + str(len(content)) + ") does not equal 1. Bus: " + str(bus) +" OBD: " + str(obd))

      # Check content
      if content[0][0] != addr or content[0][1] != string:
        raise Exception("Received CAN message content or address does not match")

      # Check bus
      if content[0][2] != bus:
        raise Exception("Received CAN message bus does not match")

#################################################################
############################# TEST ##############################
#################################################################

def test_loopback():
  # disable safety modes
  for panda in pandas:
    panda.set_safety_mode(Panda.SAFETY_ELM327 if FOR_RELEASE_BUILDS else Panda.SAFETY_ALLOUTPUT)

  # perform loopback with jungle as a sender
  can_loopback(jungle)

  # perform loopback with each possible panda as a sender
  for panda in pandas:
    can_loopback(panda)

  # enable safety modes
  for panda in pandas:
    panda.set_safety_mode(Panda.SAFETY_SILENT)

#################################################################
############################# MAIN ##############################
#################################################################
jungle = None
pandas = []  # type: ignore
panda_serials = []
counter = 0

if __name__ == "__main__":
  # Connect to jungle silently
  print_colored("Connecting to jungle", "blue")
  with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
      jungle = PandaJungle()
  jungle.set_panda_power(True)
  jungle.set_ignition(False)

  # Connect to <NUM_PANDAS_PER_TEST> new pandas before starting tests
  print_colored("Waiting for " + str(NUM_PANDAS_PER_TEST) + " pandas to be connected", "yellow")
  while True:
    connected_serials = Panda.list()
    if len(connected_serials) == NUM_PANDAS_PER_TEST:
      panda_serials = connected_serials
      break

  start_with_orientation(PandaJungle.HARNESS_ORIENTATION_1)

  # Set bus speeds
  for device in pandas + [jungle]:
    for bus in range(len(BUS_SPEEDS)):
      device.set_can_speed_kbps(bus, BUS_SPEEDS[bus])

  # Run test
  while True:
    test_loopback()
    counter += 1
    print_colored(str(counter) + " loopback cycles complete", "blue")
