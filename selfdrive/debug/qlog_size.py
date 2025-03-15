#!/usr/bin/env python3
import argparse
import zstandard as zstd
from collections import defaultdict

import matplotlib.pyplot as plt

from cereal.services import SERVICE_LIST
from openpilot.common.file_helpers import LOG_COMPRESSION_LEVEL
from openpilot.tools.lib.logreader import LogReader
from tqdm import tqdm

MIN_SIZE = 0.5  # Percent size of total to show as separate entry


def make_pie(msgs, typ):
  msgs_by_type = defaultdict(list)
  for m in msgs:
    msgs_by_type[m.which()].append(m.as_builder().to_bytes())

  total = len(zstd.compress(b"".join([m.as_builder().to_bytes() for m in msgs]), LOG_COMPRESSION_LEVEL))
  uncompressed_total = len(b"".join([m.as_builder().to_bytes() for m in msgs]))

  length_by_type = {k: len(b"".join(v)) for k, v in msgs_by_type.items()}
  # calculate compressed size by calculating diff when removed from the segment
  compressed_length_by_type = {}
  for k in tqdm(msgs_by_type.keys(), desc="Compressing"):
    compressed_length_by_type[k] = total - len(zstd.compress(b"".join([m.as_builder().to_bytes() for m in msgs if m.which() != k]), LOG_COMPRESSION_LEVEL))

  sizes = sorted(compressed_length_by_type.items(), key=lambda kv: kv[1])

  print("name - comp. size (uncomp. size)")
  for (name, sz) in sizes:
    print(f"{name:<22} - {sz / 1024:.2f} kB ({length_by_type[name] / 1024:.2f} kB)")
  print()
  print(f"{typ} - Real total {total / 1024:.2f} kB")
  print(f"{typ} - Breakdown total {sum(compressed_length_by_type.values()) / 1024:.2f} kB")
  print(f"{typ} - Uncompressed total {uncompressed_total / 1024 / 1024:.2f} MB")

  sizes_large = [(k, sz) for (k, sz) in sizes if sz >= total * MIN_SIZE / 100]
  sizes_large += [('other', sum(sz for (_, sz) in sizes if sz < total * MIN_SIZE / 100))]

  labels, sizes = zip(*sizes_large, strict=True)

  plt.figure()
  plt.title(f"{typ}")
  plt.pie(sizes, labels=labels, autopct='%1.1f%%')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='View log size breakdown by message type')
  parser.add_argument('route', help='route to use')
  parser.add_argument('--as-qlog', action='store_true', help='decimate rlog using latest decimation factors')
  args = parser.parse_args()

  msgs = list(LogReader(args.route))

  if args.as_qlog:
    new_msgs = []
    msg_cnts: dict[str, int] = defaultdict(int)
    for msg in msgs:
      msg_which = msg.which()
      if msg.which() in ("initData", "sentinel"):
        new_msgs.append(msg)
        continue

      if msg_which not in SERVICE_LIST:
        continue

      decimation = SERVICE_LIST[msg_which].decimation
      if decimation is not None and msg_cnts[msg_which] % decimation == 0:
        new_msgs.append(msg)
      msg_cnts[msg_which] += 1

    msgs = new_msgs

  make_pie(msgs, 'qlog')
  plt.show()
