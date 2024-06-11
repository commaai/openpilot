#!/usr/bin/env python3
import argparse
import bz2
from collections import defaultdict

import matplotlib.pyplot as plt

from cereal.services import SERVICE_LIST
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route

MIN_SIZE = 0.5  # Percent size of total to show as separate entry


def make_pie(msgs, typ):
  compressed_length_by_type = {k: len(bz2.compress(b"".join(v))) for k, v in msgs.items()}

  total = sum(compressed_length_by_type.values())

  sizes = sorted(compressed_length_by_type.items(), key=lambda kv: kv[1])

  print(f"{typ} - Total {total / 1024:.2f} kB")
  for (name, sz) in sizes:
    print(f"{name} - {sz / 1024:.2f} kB")
  print()

  sizes_large = [(k, sz) for (k, sz) in sizes if sz >= total * MIN_SIZE / 100]
  sizes_large += [('other', sum(sz for (_, sz) in sizes if sz < total * MIN_SIZE / 100))]

  labels, sizes = zip(*sizes_large, strict=True)

  plt.figure()
  plt.title(f"{typ}")
  plt.pie(sizes, labels=labels, autopct='%1.1f%%')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Check qlog size based on a rlog')
  parser.add_argument('route', help='route to use')
  parser.add_argument('segment', type=int, help='segment number to use')
  args = parser.parse_args()

  r = Route(args.route)
  rlog = r.log_paths()[args.segment]
  msgs = list(LogReader(rlog))

  msgs_by_type = defaultdict(list)
  for m in msgs:
    msgs_by_type[m.which()].append(m.as_builder().to_bytes())

  qlog_by_type = defaultdict(list)
  for name, service in SERVICE_LIST.items():
    if service.decimation is None:
      continue

    for i, msg in enumerate(msgs_by_type[name]):
      if i % service.decimation == 0:
        qlog_by_type[name].append(msg)

  make_pie(msgs_by_type, 'rlog')
  make_pie(qlog_by_type, 'qlog')
  plt.show()
