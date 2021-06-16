#!/usr/bin/env python3
import binascii
import sys
from collections import defaultdict

import cereal.messaging as messaging
from common.realtime import sec_since_boot


def can_printer(bus=0):
  """ Collects messages and prints when a bit transitions that hasn't changed before
  This is very usefull to find signals based on user triggered actions, such as blinkers and seatbelt.
  Leave the script running until no new transitions are seen, then perform the action."""
  logcan = messaging.sub_sock('can')

  msgs = defaultdict(int)

  while 1:
    can_recv = messaging.drain_sock(logcan, wait_for_one=True)
    for x in can_recv:
      for y in x.can:
        if y.src == bus:
          i = int.from_bytes(y.dat, byteorder='big')
          j = msgs[y.address]

          if (i | j) != j:  # Did any new bits flip?
            msgs[y.address] = i | j
            print(f"{sec_since_boot():.2f}\t{hex(y.address)} ({y.address})\t{binascii.hexlify(y.dat)}")



if __name__ == "__main__":
  if len(sys.argv) > 1:
    can_printer(int(sys.argv[1]))
  else:
    can_printer()
