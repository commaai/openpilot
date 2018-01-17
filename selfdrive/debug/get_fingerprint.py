#!/usr/bin/env python

# simple script to get a vehicle fingerprint.
# keep this script running for few seconds: some messages are published every few seconds

# Instructions:
# - connect to a Panda
# - run selfdrive/boardd/boardd
# - launching this script
# - since some messages are published at low frequency, keep this script running for few seconds, 
#   until all messages are received at least once

import zmq
import selfdrive.messaging as messaging
from selfdrive.services import service_list

context = zmq.Context()
logcan = messaging.sub_sock(context, service_list['can'].port)
msgs = {}
while True:
  lc = messaging.recv_sock(logcan, True)
  for c in lc.can:
    if c.src == 0:
      msgs[c.address] = len(c.dat)

  fingerprint = ', '.join("%d: %d" % v for v in sorted(msgs.items()))

  print "number of messages:", len(msgs)
  print "fingerprint", fingerprint
