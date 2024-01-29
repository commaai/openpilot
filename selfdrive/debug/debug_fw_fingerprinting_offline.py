#!/usr/bin/env python3
import argparse

from tools.lib.logreader import LogReader
from panda.python import uds


def main(route: str, addrs: list[int]):
  if not route.endswith('/r'):
    route = route + '/r'
  lr = LogReader(route)

  start_mono_time = None
  prev_mono_time = 0

  # include rx addresses
  addrs = addrs + [uds.get_rx_addr_for_tx_addr(addr) for addr in addrs]

  for msg in lr:
    if msg.which() == 'can':
      if start_mono_time is None:
        start_mono_time = msg.logMonoTime

    if msg.which() in ("can", 'sendcan'):
      for can in getattr(msg, msg.which()):
        if can.address in addrs:
          if msg.logMonoTime != prev_mono_time:
            print()
            prev_mono_time = msg.logMonoTime
          print(f"{msg.logMonoTime} rxaddr={can.address}, bus={can.src}, {round((msg.logMonoTime - start_mono_time) * 1e-6, 2)} ms, 0x{can.dat.hex()}, {can.dat}, {len(can.dat)=}")


if __name__ == "__main__":
  # argparse:
  parser = argparse.ArgumentParser(description='View back and forth ISO-TP communication between various ECUs given an address')
  parser.add_argument('route', help='Route name')
  parser.add_argument('addrs', nargs='*', help='List of tx address to view (0x7e0 for engine)')
  args = parser.parse_args()

  addrs = [int(addr, base=16) if addr.startswith('0x') else int(addr) for addr in args.addrs]
  main(args.route, addrs)
