#!/usr/bin/env python3
import binascii
import os
import sys
from collections import defaultdict

import cereal.messaging as messaging
from common.realtime import sec_since_boot


def can_printer(bus=0, max_msg=None, addr="127.0.0.1"):
  logcan = messaging.sub_sock('can', addr=addr)

  start = sec_since_boot()
  lp = sec_since_boot()
  msgs = defaultdict(list)
  canbus = int(os.getenv("CAN", bus))
  while 1:
    can_recv = messaging.drain_sock(logcan, wait_for_one=True)
    for x in can_recv:
      for y in x.can:
        if y.src == canbus:
          msgs[y.address].append(y.dat)

    if sec_since_boot() - lp > 0.1:
      dd = chr(27) + "[2J"
      dd += f"{sec_since_boot() - start:5.2f}\n"
      for k, v in sorted(zip(msgs.keys(), map(lambda x: binascii.hexlify(x[-1]), msgs.values()))):
        if max_msg is None or k < max_msg:
          dd += "%s(%6d) %s\n" % ("%04X(%4d)" % (k, k), len(msgs[k]), v.decode('ascii'))
      print(dd)
      lp = sec_since_boot()

if __name__ == "__main__":
  if len(sys.argv) > 3:
    can_printer(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
  elif len(sys.argv) > 2:
    can_printer(int(sys.argv[1]), int(sys.argv[2]))
  elif len(sys.argv) > 1:
    can_printer(int(sys.argv[1]))
  else:
    can_printer()
