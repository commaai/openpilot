#!/usr/bin/env python
import os
import struct
from collections import defaultdict
from common.realtime import sec_since_boot
import zmq
import selfdrive.messaging as messaging
from selfdrive.services import service_list


def can_printer():
  context = zmq.Context()
  logcan = messaging.sub_sock(context, service_list['can'].port)

  start = sec_since_boot()
  lp = sec_since_boot()
  msgs = defaultdict(list)
  canbus = int(os.getenv("CAN", 0))
  while 1:
    can_recv = messaging.drain_sock(logcan, wait_for_one=True)
    for x in can_recv:
      for y in x.can:
        if y.src == canbus:
          msgs[y.address].append(y.dat)

    if sec_since_boot() - lp > 0.1:
      dd = chr(27) + "[2J"
      dd += "%5.2f\n" % (sec_since_boot() - start)
      for k,v in sorted(zip(msgs.keys(), map(lambda x: x[-1].encode("hex"), msgs.values()))):
        dd += "%s(%6d) %s\n" % ("%04X(%4d)" % (k,k),len(msgs[k]), v)
      print dd
      lp = sec_since_boot()

if __name__ == "__main__":
  can_printer()

