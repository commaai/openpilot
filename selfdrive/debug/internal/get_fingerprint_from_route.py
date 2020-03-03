#!/usr/bin/env python3

import sys
from tools.lib.logreader import MultiLogIterator
from xx.chffr.lib.route import Route


def get_fingerprint(lr):
  can_msgs = [m for m in lr if m.which() == 'can']

  msgs = {}

  for msg in can_msgs:
    for c in msg.can:
      # read also msgs sent by EON on CAN bus 0x80 and filter out the
      # addr with more than 11 bits
      if c.src % 0x80 == 0 and c.address < 0x800:
        msgs[c.address] = len(c.dat)

    fingerprint = ', '.join("%d: %d" % v for v in sorted(msgs.items()))
  print("number of messages {0}:".format(len(msgs)))
  print("fingerprint {0}".format(fingerprint))


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: ./get_fingerprint_internal.py <route>")
    sys.exit(1)

  route = sys.argv[1]
  route = Route(route)
  lr = MultiLogIterator(route.log_paths()[:5], wraparound=False)
  get_fingerprint(lr)
