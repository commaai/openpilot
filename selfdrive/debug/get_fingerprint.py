#!/usr/bin/env python

# simple script to get a vehicle fingerprint.

# Instructions:
# - connect to a Panda
# - run selfdrive/boardd/boardd
# - launching this script
# - since some messages are published at low frequency, keep this script running for few
#   seconds, until all messages are received at least once

import zmq
import selfdrive.messaging as messaging
from selfdrive.services import service_list

context = zmq.Context()
logcan = messaging.sub_sock(context, service_list['can'].port)
msgs = {}
while True:
  lc = messaging.recv_sock(logcan, True)
  for c in lc.can:
    # read also msgs sent by EON on CAN bus 0x80 and filter out the
    # addr with more than 11 bits
    if c.src%0x80 == 0 and c.address < 0x800:
      msgs[c.address] = len(c.dat)

  fingerprint = ', '.join("%d: %d" % v for v in sorted(msgs.items()))

  print("number of messages {0}:".format(len(msgs)))
  print("fingerprint {0}".format(fingerprint))
