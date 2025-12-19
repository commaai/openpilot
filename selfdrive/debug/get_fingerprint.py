#!/usr/bin/env python3

# simple script to get a vehicle fingerprint.

# Instructions:
# - connect to a Panda
# - run selfdrive/pandad/pandad
# - launching this script
#   Note: it's very important that the car is in stock mode, in order to collect a complete fingerprint
# - since some messages are published at low frequency, keep this script running for at least 30s,
#   until all messages are received at least once

import cereal.messaging as messaging

logcan = messaging.sub_sock('can')
msgs = {}
while True:
  lc = messaging.recv_sock(logcan, True)
  if lc is None:
    continue

  for c in lc.can:
    # read also msgs sent by EON on CAN bus 0x80 and filter out the
    # addr with more than 11 bits
    if c.src % 0x80 == 0 and c.address < 0x800 and c.address not in (0x7df, 0x7e0, 0x7e8):
      msgs[c.address] = len(c.dat)

  fingerprint = ', '.join(f"{v[0]}: {v[1]}" for v in sorted(msgs.items()))

  print(f"number of messages {len(msgs)}:")
  print(f"fingerprint {fingerprint}")
