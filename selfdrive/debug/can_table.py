#!/usr/bin/env python3
import argparse
import pandas as pd

import cereal.messaging as messaging


def can_table(dat):
  rows = []
  for b in dat:
    r = list(bin(b).lstrip('0b').zfill(8))
    r += [hex(b)]
    rows.append(r)

  df = pd.DataFrame(data=rows)
  df.columns = [str(n) for n in range(7, -1, -1)] + [' ']
  table = df.to_markdown(tablefmt='grid')
  return table


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Cabana-like table of bits for your terminal",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("addr", type=str, nargs=1)
  parser.add_argument("bus", type=int, default=0, nargs='?')

  args = parser.parse_args()

  addr = int(args.addr[0], 0)
  can = messaging.sub_sock('can', conflate=False, timeout=None)

  print(f"waiting for {hex(addr)} ({addr}) on bus {args.bus}...")

  latest = None
  while True:
    for msg in messaging.drain_sock(can, wait_for_one=True):
      for m in msg.can:
        if m.address == addr and m.src == args.bus:
          latest = m

    if latest is None:
      continue

    table = can_table(latest.dat)
    print(f"\n\n{hex(addr)} ({addr}) on bus {args.bus}\n{table}")
