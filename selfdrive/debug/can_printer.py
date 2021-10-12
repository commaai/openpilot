#!/usr/bin/env python3
import argparse
import binascii
from collections import defaultdict

import cereal.messaging as messaging
from common.realtime import sec_since_boot


def can_printer(bus, max_msg, addr):
  logcan = messaging.sub_sock('can', addr=addr)

  start = sec_since_boot()
  lp = sec_since_boot()
  msgs = defaultdict(list)
  while 1:
    can_recv = messaging.drain_sock(logcan, wait_for_one=True)
    for x in can_recv:
      for y in x.can:
        if y.src == bus:
          msgs[y.address].append(y.dat)

    if sec_since_boot() - lp > 0.1:
      dd = chr(27) + "[2J"
      dd += "%5.2f\n" % (sec_since_boot() - start)
      for k, v in sorted(zip(msgs.keys(), map(lambda x: binascii.hexlify(x[-1]), msgs.values()))):
        if max_msg is None or k < max_msg:
          dd += "%s(%6d) %s\n" % ("%04X(%4d)" % (k, k), len(msgs[k]), v.decode('ascii'))
      print(dd)
      lp = sec_since_boot()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PlotJuggler plugin for reading openpilot logs",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--bus", help="CAN bus to print out", default=0)
  parser.add_argument("--max_msg", help="max addr ")
  parser.add_argument("--addr", default="127.0.0.1")

  args = parser.parse_args()
  can_printer(args.bus, args.max_msg, args.addr)
