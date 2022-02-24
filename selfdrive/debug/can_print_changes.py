#!/usr/bin/env python3
import binascii
import sys
from collections import defaultdict

import cereal.messaging as messaging
from common.realtime import sec_since_boot


def can_printer(bus=0):
  """Collects messages and prints when a new bit transition is observed.
  This is very useful to find signals based on user triggered actions, such as blinkers and seatbelt.
  Leave the script running until no new transitions are seen, then perform the action."""
  logcan = messaging.sub_sock('can')

  low_to_high = defaultdict(int)
  high_to_low = defaultdict(int)

  while 1:
    can_recv = messaging.drain_sock(logcan, wait_for_one=True)
    for x in can_recv:
      for y in x.can:
        if y.src == bus:
          i = int.from_bytes(y.dat, byteorder='big')

          l_h = low_to_high[y.address]
          h_l = high_to_low[y.address]

          change = None
          if (i | l_h) != l_h:
            low_to_high[y.address] = i | l_h
            change = "+"

          if (~i | h_l) != h_l:
            high_to_low[y.address] = ~i | h_l
            change = "-"

          if change:
            print(f"{sec_since_boot():.2f}\t{hex(y.address)} ({y.address})\t{change}{binascii.hexlify(y.dat)}")


if __name__ == "__main__":
  if len(sys.argv) > 1:
    can_printer(int(sys.argv[1]))
  else:
    can_printer()
