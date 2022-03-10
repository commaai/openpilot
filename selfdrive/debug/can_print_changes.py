#!/usr/bin/env python3
import argparse
import binascii
import time
from collections import defaultdict

import cereal.messaging as messaging
from tools.lib.logreader import logreader_from_route_or_segment


def update(msgs, bus, low_to_high, high_to_low, quiet=False):
  for x in msgs:
    if x.which() != 'can':
      continue

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

        if change and not quiet:
          print(f"{time.monotonic():.2f}\t{hex(y.address)} ({y.address})\t{change}{binascii.hexlify(y.dat)}")


def can_printer(bus=0, init_msgs=None, new_msgs=None):
  logcan = messaging.sub_sock('can')

  low_to_high = defaultdict(int)
  high_to_low = defaultdict(int)

  if init_msgs is not None:
    update(init_msgs, bus, low_to_high, high_to_low, quiet=True)

  if new_msgs is not None:
    update(new_msgs, bus, low_to_high, high_to_low)
  else:
    # Live mode
    while 1:
      can_recv = messaging.drain_sock(logcan, wait_for_one=True)
      update(can_recv, bus, low_to_high, high_to_low)


if __name__ == "__main__":
  desc = """Collects messages and prints when a new bit transition is observed.
  This is very useful to find signals based on user triggered actions, such as blinkers and seatbelt.
  Leave the script running until no new transitions are seen, then perform the action."""

  parser = argparse.ArgumentParser(description=desc,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--bus", type=int, help="CAN bus to print out", default=0)

  parser.add_argument("--init", type=str, help="Route or segment to initialize with")
  parser.add_argument("--comp", type=str, help="Route or segment to compare against init")

  args = parser.parse_args()

  init_lr, new_lr = None, None
  if args.init:
    init_lr = logreader_from_route_or_segment(args.init)
  if args.comp:
    new_lr = logreader_from_route_or_segment(args.comp)

  can_printer(args.bus, init_msgs=init_lr, new_msgs=new_lr)
