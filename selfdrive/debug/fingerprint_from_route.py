#!/usr/bin/env python3

import sys
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator


def get_fingerprint(lr):
  # TODO: make this a nice tool for car ports. should also work with qlogs for FW

  fw = None
  msgs = {}
  for msg in lr:
    if msg.which() == 'carParams':
      fw = msg.carParams.carFw
    elif msg.which() == 'can':
      for c in msg.can:
        # read also msgs sent by EON on CAN bus 0x80 and filter out the
        # addr with more than 11 bits
        if c.src % 0x80 == 0 and c.address < 0x800:
          msgs[c.address] = len(c.dat)

  # show CAN fingerprint
  fingerprint = ', '.join("%d: %d" % v for v in sorted(msgs.items()))
  print(f"\nfound {len(msgs)} messages. CAN fingerprint:\n")
  print(fingerprint)

  # TODO: also print the fw fingerprint merged with the existing ones
  # show FW fingerprint
  print("\nFW fingerprint:\n")
  for f in fw:
    print(f"    (Ecu.{f.ecu}, {hex(f.address)}, {None if f.subAddress == 0 else f.subAddress}): [")
    print(f"      {f.fwVersion},")
    print("    ],")
  print()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: ./fingerprint_from_route.py <route>")
    sys.exit(1)

  route = Route(sys.argv[1])
  lr = MultiLogIterator(route.log_paths()[:5])
  get_fingerprint(lr)
