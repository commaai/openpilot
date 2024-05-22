#!/usr/bin/env python3
import argparse

from openpilot.tools.lib.logreader import LogReader, ReadMode
from panda.python import uds


def main(route: str, addrs: list[int]):
  """
  TODO:
  - highlight TX vs RX clearly
  - disambiguate sendcan and can (useful to know if something sent on sendcan made it to the bus on can->128)
  - print as fixed width table, easier to read
  """

  lr = LogReader(route, default_mode=ReadMode.RLOG, sort_by_time=True)

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
          print(f"{msg.which():>7}: rxaddr={can.address}, bus={can.src}, {round((msg.logMonoTime - start_mono_time) * 1e-6, 2)} ms, " +
                f"0x{can.dat.hex()}, {can.dat}, {len(can.dat)=}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='View back and forth ISO-TP communication between various ECUs given an address')
  parser.add_argument('route', help='Route name')
  parser.add_argument('addrs', nargs='*', help='List of tx address to view (0x7e0 for engine)')
  args = parser.parse_args()

  addrs = [int(addr, base=16) if addr.startswith('0x') else int(addr) for addr in args.addrs]
  main(args.route, addrs)
