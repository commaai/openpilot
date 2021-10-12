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
      for addr in sorted(msgs.keys()):
        a = msgs[addr][-1].decode('ascii', 'backslashreplace')
        x = binascii.hexlify(msgs[addr][-1]).decode('ascii')
        if max_msg is None or addr < max_msg:
          dd += "%04X(%4d)(%6d) %s \"%s\"\n" % (addr, addr, len(msgs[addr]), x.ljust(20), a)
      print(dd)
      lp = sec_since_boot()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="can data viewer",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--bus", type=int, help="CAN bus to print out", default=0)
  parser.add_argument("--max_msg", type=int, help="max addr ")
  parser.add_argument("--addr", default="127.0.0.1")

  args = parser.parse_args()
  can_printer(args.bus, args.max_msg, args.addr)
