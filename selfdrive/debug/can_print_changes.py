#!/usr/bin/env python3
import argparse
import binascii
import time
from collections import defaultdict

import cereal.messaging as messaging
from selfdrive.debug.can_table import can_table
from tools.lib.logreader import logreader_from_route_or_segment

RED = '\033[91m'
CLEAR = '\033[0m'

def update(msgs, bus, dat, low_to_high, high_to_low, quiet=False):
  for x in msgs:
    if x.which() != 'can':
      continue

    for y in x.can:
      if y.src == bus:
        dat[y.address] = y.dat

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


def can_printer(bus=0, init_msgs=None, new_msgs=None, table=False):
  logcan = messaging.sub_sock('can', timeout=10)

  dat = defaultdict(int)
  low_to_high = defaultdict(int)
  high_to_low = defaultdict(int)

  if init_msgs is not None:
    update(init_msgs, bus, dat, low_to_high, high_to_low, quiet=True)

  low_to_high_init = low_to_high.copy()
  high_to_low_init = high_to_low.copy()

  if new_msgs is not None:
    update(new_msgs, bus, dat, low_to_high, high_to_low)
  else:
    # Live mode
    try:
      while 1:
        can_recv = messaging.drain_sock(logcan)
        update(can_recv, bus, dat, low_to_high, high_to_low)
        time.sleep(0.02)
    except KeyboardInterrupt:
      pass

  print("\n\n")
  tables = ""
  for addr in sorted(dat.keys()):
    init = low_to_high_init[addr] & high_to_low_init[addr]
    now = low_to_high[addr] & high_to_low[addr]
    d = now & ~init
    if d == 0:
      continue
    b = d.to_bytes(len(dat[addr]), byteorder='big')

    byts = ''.join([(c if c == '0' else f'{RED}{c}{CLEAR}') for c in str(binascii.hexlify(b))[2:-1]])
    header = f"{hex(addr).ljust(6)}({str(addr).ljust(4)})"
    print(header, byts)
    tables += f"{header}\n"
    tables += can_table(b) + "\n\n"

  if table:
    print(tables)

if __name__ == "__main__":
  desc = """Collects messages and prints when a new bit transition is observed.
  This is very useful to find signals based on user triggered actions, such as blinkers and seatbelt.
  Leave the script running until no new transitions are seen, then perform the action."""
  parser = argparse.ArgumentParser(description=desc,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--bus", type=int, help="CAN bus to print out", default=0)
  parser.add_argument("--table", action="store_true", help="Print a cabana-like table")
  parser.add_argument("init", type=str, nargs='?', help="Route or segment to initialize with")
  parser.add_argument("comp", type=str, nargs='?', help="Route or segment to compare against init")

  args = parser.parse_args()

  init_lr, new_lr = None, None
  if args.init:
    init_lr = logreader_from_route_or_segment(args.init)
  if args.comp:
    new_lr = logreader_from_route_or_segment(args.comp)

  can_printer(args.bus, init_msgs=init_lr, new_msgs=new_lr, table=args.table)
